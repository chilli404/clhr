"""A1: Soft-to-hard MoE discretization experiment (closed-loop discretization
failure generality test).

Registered predictions this experiment must make measurable are frozen in
predictions/principle_generality.json (instance A1_soft_mixture_moe). This
file is a clean, self-contained MoE transformer with an added soft-mixture
forward path (which does not exist anywhere else in the repo) plus the five
evaluation modes and two training conditions required by that registration.

Does NOT modify src/revision_moe_coadaptation.py. Code here is copied /
adapted from that file (Expert, Router, CausalSelfAttention skeleton) but the
soft-mixture path, the eval modes, and the CLHR training loss are new.

Evaluation modes
-----------------
native_soft        : single pass, full softmax mixture over all experts at
                     every layer.
open_loop_top1     : TWO passes. Pass 1 is a soft-mixture forward; each
                     layer's router logits are recorded and argmax'd to get
                     per-layer hard indices. Pass 2 is a *fresh* forward that
                     dispatches with those stored indices (forced_routes).
                     The indices come from soft-pass hidden states, not from
                     the states the hard pass itself produces.
closed_loop_top1   : ONE pass. At each layer the router logits are computed
                     from the *current* (already hard-dispatched) hidden
                     state, argmax'd, and dispatched immediately. This is the
                     deployment metric: each layer's decision is conditioned
                     on upstream discretization, not upstream soft mixing.
shuffled_top1      : Take the actual closed_loop_top1 indices (one real
                     closed-loop pass), then, independently per layer,
                     permute the per-token index assignment across the full
                     flattened (batch*seq) token axis for that layer, and
                     replay with those permuted indices forced. This is a
                     permutation of the *exact* multiset of decisions the
                     closed-loop pass made, so the per-expert token count for
                     each layer (and hence the aggregate histogram) is
                     IDENTICAL to closed_loop_top1's, by construction, not
                     merely in expectation. What is destroyed is which token
                     received which expert, i.e. input-conditioning.
                     (Alternative schemes exist -- e.g. permuting only within
                     each sequence, or drawing a fresh permutation per
                     layer-if-recomputed-in-a-second closed-loop pass -- but
                     both of those either couple the permutation to sequence
                     boundaries for no principled reason, or would only
                     approximately preserve the histogram since a second
                     independently-run closed-loop pass could pick different
                     indices upstream. Permuting the recorded closed-loop
                     indices directly gives an exact-histogram control.)
random_top1        : uniformly random expert per token at every layer,
                     independent of the router entirely. Repeated
                     `n_draws` times; mean and std reported. Utility floor.

Training conditions
--------------------
standard : soft-mixture forward only (`L_soft`).
clhr     : `(L_soft + lambda_rca * L_hard) / (1 + lambda_rca)` where L_hard is
           the closed_loop hard-dispatch forward on the SAME batch. The
           argmax used to pick the dispatch index is always under
           torch.no_grad(); expert and attention weights receive gradient
           through the dispatched path, the router does not (except via the
           aux loss, exactly as in the closed-loop-only case).
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ═══════════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════════

class Expert(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.gelu(self.w1(x)))


class Router(nn.Module):
    """Just the gate. Callers decide what to do with the logits (soft mix,
    argmax for hard dispatch, etc.) -- unlike the hard-only Router in
    revision_moe_coadaptation.py, this Router does not itself commit to
    argmax."""

    def __init__(self, d_model: int, n_experts: int):
        super().__init__()
        self.gate = nn.Linear(d_model, n_experts, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gate(x)


def _aux_load_balance_loss(logits: torch.Tensor, indices: torch.Tensor, n_experts: int) -> torch.Tensor:
    """Switch-Transformer-style load balancing aux loss. `logits` must be the
    differentiable router logits; `indices` is whatever hard assignment is
    being used for utilization stats (may be under no_grad).

    `indices` may be either:
      - 1-D, shape (N,): the original top-1 convention. `freq[e_idx]` is the
        fraction of tokens whose single assignment equals `e_idx`.
      - 2-D, shape (N, k): the top-k-dispatch generalization (GShard/Switch
        style). A token "uses" expert `e_idx` if it appears in ANY of its k
        selected slots: `freq[e_idx] = (indices == e_idx).any(dim=-1).mean()`.
        At k=1 (indices shape (N, 1)) this is identical to the 1-D case --
        see test_aux_loss_any_dim_reduces_correctly_at_k_1."""
    probs = F.softmax(logits, dim=-1)
    avg_probs = probs.mean(dim=0)
    freq = torch.zeros(n_experts, device=logits.device, dtype=probs.dtype)
    if indices.dim() == 1:
        for e_idx in range(n_experts):
            freq[e_idx] = (indices == e_idx).float().mean()
    else:
        for e_idx in range(n_experts):
            freq[e_idx] = (indices == e_idx).any(dim=-1).float().mean()
    return (avg_probs * freq).sum() * n_experts


class SoftTopK(torch.autograd.Function):
    """LapSum soft top-k (Zasada et al., "SoftMoE: Soft Differentiable
    Routing for Mixture-of-Experts in LLMs", ICML 2026). Ported verbatim
    from the official implementation (dlcuda/SoftMoE,
    patches/megatron-softmoe.patch -> megatron/core/transformer/moe/
    soft_topk.py). Defines p_i = LaplaceCDF(r_i/alpha - b) with b solved
    per-row in closed form so that sum_i p_i == k_row (a continuous
    relaxation of "select k of n"). See
    test_soft_top_k_matches_independent_bisection_reference for an
    independent re-derivation (bisection on the same monotonic equation,
    not trusting this closed-form solver) confirming this ported code
    computes what the docstring claims."""

    @staticmethod
    def _solve(s, t, a, b, e):
        z = torch.abs(e) + torch.sqrt(e**2 + a * b * torch.exp(s - t))
        ab = torch.where(e > 0, a, b)

        return torch.where(
            e > 0, t + torch.log(z) - torch.log(ab), s - torch.log(z) + torch.log(ab)
        )

    @staticmethod
    def forward(ctx, r, k, alpha, descending=False):
        assert r.shape[0] == k.shape[0], "k must have same batch size as r"

        batch_size, num_dim = r.shape
        x = torch.empty_like(r, requires_grad=False)

        def finding_b():
            scaled = torch.sort(r, dim=1)[0]
            scaled.div_(alpha)

            eB = torch.logcumsumexp(scaled, dim=1)
            eB.sub_(scaled).exp_()

            torch.neg(scaled, out=x)
            eA = torch.flip(x, dims=(1,))
            torch.logcumsumexp(eA, dim=1, out=x)
            idx = torch.arange(start=num_dim - 1, end=-1, step=-1, device=x.device)
            torch.index_select(x, 1, idx, out=eA)
            eA.add_(scaled).exp_()

            row = torch.arange(1, 2 * num_dim + 1, 2, device=r.device)

            torch.add(torch.add(eA, eB, alpha=-1, out=x), row.view(1, -1), out=x)

            w = (k if descending else num_dim - k).unsqueeze(1)
            i = torch.searchsorted(x, 2 * w)
            m = torch.clamp(i - 1, 0, num_dim - 1)
            n = torch.clamp(i, 0, num_dim - 1)

            b = SoftTopK._solve(
                scaled.gather(1, m),
                scaled.gather(1, n),
                torch.where(i < num_dim, eA.gather(1, n), 0),
                torch.where(i > 0, eB.gather(1, m), 0),
                w - i,
            )
            return b

        b = finding_b()

        sign = -1 if descending else 1
        torch.div(r, alpha * sign, out=x)
        x.sub_(sign * b)

        sign_x = x > 0
        p = torch.abs(x)
        p.neg_().exp_().mul_(0.5)

        inv_alpha = -sign / alpha
        S = torch.sum(p, dim=1, keepdim=True).mul_(inv_alpha)

        torch.where(sign_x, 1 - p, p, out=p)

        ctx.save_for_backward(r, x, S)
        ctx.alpha = alpha
        return p

    @staticmethod
    def backward(ctx, grad_output):
        # .clone() before the in-place ops below: the saved tensors are
        # mutated in-place in the reference implementation, which is safe
        # for a single backward() call (ordinary training) but violates
        # autograd's tensor-version tracking if backward is invoked more
        # than once against the same saved context (e.g.
        # torch.autograd.gradcheck's row-by-row Jacobian construction, or
        # retain_graph=True call sites). Cloning makes this robust to that
        # case without changing the result for the single-call case.
        r, x, S = ctx.saved_tensors
        r = r.clone()
        x = x.clone()
        S = S.clone()
        alpha = ctx.alpha

        x.abs_().neg_()
        q = torch.softmax(x, dim=1)

        torch.mul(q, grad_output, out=x)
        grad_k = x.sum(dim=1, keepdim=True)

        grad_r = grad_k - grad_output
        grad_r.mul_(q).mul_(S)

        q.mul_(r)
        x.mul_(S / alpha)  # grad_alpha = (S / alpha) * x
        r.sub_(q.sum(dim=1, keepdim=True))
        x.mul_(r)  # grad_alpha.mul_(r)
        grad_alpha = x.sum()  # grad_alpha = grad_alpha.sum()
        return grad_r, grad_k.squeeze(1), grad_alpha, None


def soft_top_k(r: torch.Tensor, k: torch.Tensor, alpha: torch.Tensor, descending: bool = False):
    return SoftTopK.apply(r, k, alpha, descending)


class MoEBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, n_experts: int):
        super().__init__()
        self.router = Router(d_model, n_experts)
        self.experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(n_experts)])
        self.n_experts = n_experts

    def soft_forward(self, x: torch.Tensor):
        """Full softmax mixture over ALL experts. Real gradient flows to the
        router through the mixture weights."""
        shape = x.shape
        flat = x.reshape(-1, shape[-1])

        logits = self.router(flat)
        probs = F.softmax(logits, dim=-1)

        out = torch.zeros_like(flat)
        for e_idx in range(self.n_experts):
            out = out + probs[:, e_idx : e_idx + 1] * self.experts[e_idx](flat)

        with torch.no_grad():
            indices = logits.argmax(dim=-1)
        aux_loss = _aux_load_balance_loss(logits, indices, self.n_experts)

        return out.reshape(shape), logits, aux_loss

    def hard_forward(
        self,
        x: torch.Tensor,
        forced_indices: torch.Tensor | None = None,
        shuffle: bool = False,
        random_mode: bool = False,
        capacity_factor: float | None = None,
        top_k_dispatch: int = 1,
    ):
        """Hard top-k dispatch (top_k_dispatch=1, the default, is top-1 --
        bit-identical to the original algorithm, see
        test_top_k_dispatch_1_is_bit_identical).

        - forced_indices given -> use them verbatim (no router call at all).
          Reshaped to (N, k). No router logits exist in this branch, so no
          softmax weight is available; each of the k slots gets an equal
          weight 1/k (an extension beyond the four prescribed
          generalizations below, since none of them specify forced-indices
          weighting for k>1 -- documented here rather than left implicit).
        - random_mode -> uniformly random expert per token per slot, no
          router call; same equal-weight-1/k convention as forced_indices,
          for the same reason (no logits to softmax over).
        - shuffle -> compute router top-k logits/indices/weights from the
          CURRENT state (same as the plain closed-loop case) but then
          permute the ROWS of the resulting (N, k) index/weight matrices
          across the flattened token axis before dispatch -- i.e. token i's
          entire top-k assignment SET (and its matching per-slot weights)
          moves as a unit to a different token's position. Per-expert
          counts are preserved by construction (same reasoning as the
          original top-1 shuffle, generalized to rows instead of scalars --
          see test_shuffle_generalizes_correctly_at_k_gt_1).
        - otherwise -> closed-loop: router top-k logits/indices from the
          CURRENT (already-dispatched-by-upstream-layers) hidden state.

        Combining multiple experts' outputs (Mixtral-style, decision 1): for
        each token, take the top-k router logits, softmax over ONLY those k
        values (not over all n_experts) to get per-token, per-slot weights
        that sum to 1, then ACCUMULATE:
        `out[token] = sum over the k selected slots of (weight * expert(token))`.
        The top-k SELECTION (`logits.topk`) is taken under torch.no_grad()
        (non-differentiable, exactly like the original argmax); the softmax
        WEIGHTS are computed from the original (grad-attached) logits
        gathered at the selected indices, so gradient can flow to the router
        through the combine weights -- this doesn't change bit-identical
        behavior at k=1 (see below).

        At top_k_dispatch=1, softmax over a single logit value is
        mathematically and numerically exactly 1.0 (exp(v-v)/exp(v-v) = 1.0
        exactly in floating point, since numerator and denominator are the
        identical computed value), and accumulation into a zero-initialized
        tensor with exactly one contribution per token
        (`out[mask] = out[mask] + 1.0 * expert(...)`, i.e. `0 + x == x`
        exactly) is identical to the original direct assignment
        (`out[mask] = expert(flat[mask])`). This is proven by
        test_top_k_dispatch_1_is_bit_identical against an INDEPENDENT
        reference reimplementation of the pre-existing top-1 algorithm
        (`_reference_hard_forward_top1` in the test file), not merely by
        comparing this code's default against its own explicit k=1 call.

        The argmax/topk selection is always taken under torch.no_grad(); the
        returned index tensor never requires grad.

        capacity_factor (default None = unlimited, bit-identical to the
        original algorithm with no capacity limiting at all -- see
        test_capacity_none_is_bit_identical in tests/test_moe_soft_to_hard.py)
        adds Switch-Transformer-style expert-capacity limiting and token
        dropping (Fedus, Zoph & Shazeer 2021, "Switch Transformers: Scaling
        to Trillion Parameter Models with Simple and Efficient Sparsity").
        Real deployed top-1/top-k MoEs (Switch Transformer, GShard) give each
        expert a fixed per-batch buffer of size
        `capacity = ceil(capacity_factor * n_tokens / n_experts)`
        (n_tokens = flat.shape[0], the flattened batch*seq TOKEN count for
        THIS call -- unaffected by top_k_dispatch, exactly the physical-
        buffer-size interpretation: an expert's buffer is a fixed resource,
        not something that grows just because dispatch became wider);
        (token, slot) pairs routed to that expert beyond its buffer overflow
        and are DROPPED.

        Capacity + top-k-dispatch interaction (decision 4): capacity is
        per-EXPERT and counts ALL (token, slot) pairs routed to that expert
        regardless of which top-k SLOT (1st choice, 2nd choice, ...)
        produced the assignment. All (token, slot) pairs are flattened
        (row-major: for a fixed token, slot 0..k-1 in order, then the next
        token -- i.e. `indices.reshape(-1)` on the (N, k) matrix) into one
        arrival-order sequence per expert, and the SAME
        capacity/drop-in-arrival-order rule below is applied to that larger
        flattened set instead of the (N,) set used at k=1. A token's 2nd-
        choice expert can independently hit capacity regardless of whether
        its 1st-choice expert did -- see
        test_capacity_and_top_k_dispatch_interact_correctly for a hand-
        verified construction.

        Two design choices, made explicitly (both are ORDER/RULE-dependent
        choices with more than one defensible convention -- picked once here
        and applied consistently):
          1. WHICH (token, slot) pairs are dropped when an expert is over
             capacity: arrival order within the flattened (token, slot) axis
             -- the first `capacity` pairs (by position) assigned to that
             expert are kept, any additional ones are dropped. This is the
             Switch Transformer convention (not a router-confidence /
             top-k-by-logit selection rule, which is a different, also-used
             convention in some later work but not the one implemented
             here).
          2. WHAT a dropped (token, slot) pair's contribution is: that
             slot's WEIGHTED contribution becomes the token's own pre-expert
             representation, weighted by that slot's own softmax weight --
             i.e. `out[token] += weight[token, slot] * flat[token]` for a
             dropped slot, treating a dropped slot as if
             `expert(token) == token` for that slot's contribution
             (residual/identity passthrough, NOT zero), consistent with
             "this expert was unavailable, pass through instead." At
             top_k_dispatch=1 this is exactly the original
             `out[dropped] = flat[dropped]` (weight is always 1.0, and
             `0 + 1.0 * flat[dropped] == flat[dropped]` exactly).
        Capacity limiting is applied AFTER `indices`/`weights` have been
        determined by whichever branch above produced them -- it models a
        physical resource constraint on an expert's buffer, not a property
        of the routing decision itself, so it is applied identically
        regardless of whether `indices` came from forced_indices,
        random_mode, shuffle, or the plain closed-loop router call.

        Returns (out, indices, aux_loss). `indices` has shape shape[:-1] at
        top_k_dispatch=1 (unchanged from the original return contract) or
        shape[:-1] + (top_k_dispatch,) at k>1.
        """
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        n_tokens = flat.shape[0]
        k = top_k_dispatch

        logits = None
        if forced_indices is not None:
            indices = forced_indices.reshape(n_tokens, k)
            weights = torch.full((n_tokens, k), 1.0 / k, device=x.device, dtype=flat.dtype)
        elif random_mode:
            indices = torch.randint(0, self.n_experts, (n_tokens, k), device=x.device)
            weights = torch.full((n_tokens, k), 1.0 / k, device=x.device, dtype=flat.dtype)
        else:
            logits = self.router(flat)
            with torch.no_grad():
                _, indices = logits.topk(k, dim=-1)
            # Gather the grad-attached logit values at the (non-differentiable)
            # selected indices, then softmax over ONLY those k values -- lets
            # gradient flow to the router through the combine weights while
            # the selection itself stays non-differentiable.
            topk_logits = torch.gather(logits, dim=-1, index=indices)
            weights = F.softmax(topk_logits, dim=-1)
            if shuffle:
                perm = torch.randperm(n_tokens, device=x.device)
                indices = indices[perm]
                weights = weights[perm]

        dropped_mask = None
        if capacity_factor is not None:
            capacity = math.ceil(capacity_factor * n_tokens / self.n_experts)
            flat_indices_for_capacity = indices.reshape(-1)  # row-major (token, slot)
            dropped_flat = torch.zeros(n_tokens * k, dtype=torch.bool, device=x.device)
            for e_idx in range(self.n_experts):
                expert_positions = (flat_indices_for_capacity == e_idx).nonzero(as_tuple=True)[0]
                if expert_positions.numel() > capacity:
                    dropped_flat[expert_positions[capacity:]] = True
            dropped_mask = dropped_flat.reshape(n_tokens, k)

        out = torch.zeros_like(flat)
        for slot in range(k):
            slot_indices = indices[:, slot]
            slot_weights = weights[:, slot]
            slot_dropped = dropped_mask[:, slot] if dropped_mask is not None else None
            for e_idx in range(self.n_experts):
                mask = slot_indices == e_idx
                if slot_dropped is not None:
                    mask = mask & ~slot_dropped
                if mask.any():
                    out[mask] = out[mask] + slot_weights[mask].unsqueeze(-1) * self.experts[e_idx](flat[mask])
            if slot_dropped is not None and slot_dropped.any():
                out[slot_dropped] = out[slot_dropped] + slot_weights[slot_dropped].unsqueeze(-1) * flat[slot_dropped]

        aux_loss = torch.tensor(0.0, device=x.device)
        if logits is not None:
            aux_loss = _aux_load_balance_loss(logits, indices, self.n_experts)

        if k == 1:
            return out.reshape(shape), indices.reshape(shape[:-1]), aux_loss
        return out.reshape(shape), indices.reshape(shape[:-1] + (k,)), aux_loss

    def relu_forward(
        self,
        x: torch.Tensor,
        l1_reg_coeff: float | None = None,
        target_topk: int = 1,
        capacity_factor: float | None = None,
    ):
        """ReMoE (Wang et al., ICLR 2025, thu-ml/ReMoE) -- ReLU routing.

        Ported faithfully from the OFFICIAL SOURCE
        (megatron/core/transformer/moe/router.py::ReLURouter,
        megatron/core/transformer/moe/moe_utils.py::switch_load_balancing_loss_func),
        read directly, not reconstructed from memory or the paper's prose.

        Replaces softmax+top-k entirely: a token routes to expert `e` iff
        `relu(logit_e) > 0`, giving a VARIABLE number of active experts per
        token (not a fixed k -- some tokens may activate zero experts, others
        many). Selected experts' outputs are weighted by the RAW relu value,
        not renormalized to sum to 1 (unlike this file's own top-k-dispatch
        softmax-over-selected-k). Sparsity toward a target average k is
        nudged via an L1 regularization term sharing the Switch Transformer
        load-balancing loss's exact formula (verified byte-for-byte against
        source):
            aux_loss = sum(probs_per_expert * tokens_per_expert)
                       * (n_experts * l1_reg_coeff) / (num_tokens^2 * target_topk)
        A zero-expert token gets a genuine zero MoE contribution -- this is
        architecturally fine (not a bug) because TransformerBlock.forward_*
        always wraps the MoE output as a residual (`x = x + moe_out`), so a
        zero contribution is simply "skip the FFN this layer for this token."

        Args:
            l1_reg_coeff: None (default) disables the L1 term entirely
                (returns a zero loss) -- matches every other forward
                variant's None-disables convention in this file.
            target_topk: the `topk` ReMoE's own loss formula scales by --
                purely a loss-scale parameter (this routing mechanism has no
                literal top-k step), matching the source's own convention of
                referencing `self.topk` when computing the L1 loss even
                though the router itself never calls `.topk()`.
            capacity_factor: None (default) disables capacity limiting
                entirely -- bit-identical to the pre-existing algorithm (see
                test_relu_capacity_none_is_bit_identical), matching every
                other None-disables convention in this file. When set,
                applies the SAME Switch-Transformer-style per-expert
                capacity + arrival-order dropping + residual/identity-
                passthrough convention as `hard_forward`'s `capacity_factor`
                (see that docstring for the full formula and citation),
                adapted to this method's VARIABLE cardinality: a token may
                be routed to zero, one, or several experts simultaneously
                (`routing_map[token, e]` is True independently per expert --
                there is no fixed "slot" index shared across experts the way
                `hard_forward`'s top-k slots are). Each (token, expert) pair
                with `routing_map[token, e]` True is treated as one of
                `hard_forward`'s (token, slot) pairs for capacity counting.
                Ordering convention (documented explicitly, matching
                `hard_forward`'s own explicit-choice docstring): capacity is
                tracked independently PER EXPERT (experts do not compete for
                each other's buffers, since a token's assignment to expert A
                is fully independent of its assignment to expert B under
                ReLU routing -- unlike top-k dispatch, there is no shared
                per-token "slot" resource, only n_experts separate per-
                expert resources). Within a single expert's queue, arrival
                order is simply ascending flattened-token index (the same
                batch*seq row-major order as `flat = x.reshape(-1,
                d_model)`) -- i.e. for expert e, the first `capacity` tokens
                (by token index) with `routing_map[token, e]` True are kept,
                any additional ones for that expert are dropped. This
                collapses to `hard_forward`'s own row-major convention when
                there is exactly one active expert per token (the
                top_k_dispatch=1 case), since there token order IS the
                entire flattening order (no slot dimension to interleave).
                A dropped (token, expert) pair contributes
                `probs[token, expert] * flat[token]` (residual/identity
                passthrough weighted by that pair's OWN relu probability,
                not a shared/renormalized weight) instead of
                `probs[token, expert] * expert_fn(flat[token])`, exactly
                mirroring `hard_forward`'s "expert unavailable, pass through
                instead" convention. Capacity dropping does not affect the
                L1 aux loss (computed from the pre-drop `probs`/
                `routing_map`, exactly as `hard_forward`'s aux_loss uses the
                pre-drop `indices` -- capacity is a physical resource
                constraint on the FORWARD computation, not a property of the
                routing decision the aux loss measures).
        """
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        n = flat.shape[0]

        logits = self.router(flat)
        probs = F.relu(logits)
        routing_map = probs > 0

        dropped_mask = None
        if capacity_factor is not None:
            capacity = math.ceil(capacity_factor * n / self.n_experts)
            dropped_mask = torch.zeros_like(routing_map)
            for e_idx in range(self.n_experts):
                expert_positions = routing_map[:, e_idx].nonzero(as_tuple=True)[0]
                if expert_positions.numel() > capacity:
                    dropped_mask[expert_positions[capacity:], e_idx] = True

        out = torch.zeros_like(flat)
        for e_idx in range(self.n_experts):
            mask = routing_map[:, e_idx]
            drop_e = dropped_mask[:, e_idx] if dropped_mask is not None else None
            active_mask = mask & ~drop_e if drop_e is not None else mask
            if active_mask.any():
                out[active_mask] = out[active_mask] + probs[active_mask, e_idx : e_idx + 1] * self.experts[e_idx](flat[active_mask])
            if drop_e is not None and drop_e.any():
                out[drop_e] = out[drop_e] + probs[drop_e, e_idx : e_idx + 1] * flat[drop_e]

        l1_loss = torch.zeros((), device=x.device, dtype=probs.dtype)
        if l1_reg_coeff is not None and l1_reg_coeff > 0:
            tokens_per_expert = routing_map.sum(dim=0).to(probs.dtype)
            probs_per_expert = probs.sum(dim=0)
            l1_loss = torch.sum(probs_per_expert * tokens_per_expert) * (
                self.n_experts * l1_reg_coeff / (n * n * target_topk)
            )

        return out.reshape(shape), routing_map.reshape(shape[:-1] + (self.n_experts,)), l1_loss

    def softtopk_forward(
        self,
        x: torch.Tensor,
        topk: float = 1.0,
        alpha: float = 1.0,
        threshold: float = 1.8,
        hard_threshold_coeff: float = 2.0,
        aux_loss_coeff: float | None = None,
        capacity_factor: float | None = None,
    ):
        """SoftMoE (Zasada et al., "SoftMoE: Soft Differentiable Routing for
        Mixture-of-Experts in LLMs", ICML 2026) -- LapSum soft top-k
        routing, truncated to sparse execution.

        Ported faithfully from the OFFICIAL SOURCE (dlcuda/SoftMoE,
        patches/megatron-softmoe.patch -> megatron/core/transformer/moe/
        moe_utils.py::compute_soft_topk), read directly via `gh api`, not
        reconstructed from memory or the paper's prose. FIXED-BUDGET variant
        only: `topk` is a constant hyperparameter (matching this project's
        own top_k_dispatch, for a like-for-like comparison). The paper's
        LEARNED global per-layer expert budget (a separate learnable scalar
        mapped to a budget under a cross-layer constraint -- see the
        source's moe_router_soft_topk_learn_k path) is intentionally NOT
        ported; that studies a different question (adaptive compute
        allocation across layers) orthogonal to this project's fixed-k
        dispatch-width comparison.

        Mechanism: soft_top_k(logits, topk, alpha) (see SoftTopK above)
        gives continuous probabilities p_i = LaplaceCDF(logit_i/alpha - b)
        with sum_i p_i == topk -- a soft relaxation of "select topk of
        n_experts". These are then hard-truncated to exact sparsity: only
        probabilities above a per-token threshold survive, where
        `thresh = clamp(threshold * topk/n_experts, between the value at
        rank ceil(topk*hard_threshold_coeff) and the rank-1 value)`,
        giving the VARIABLE (not fixed) active-expert count SoftMoE
        actually executes at inference -- same flavor as ReMoE's
        relu_forward (see above) but driven by the LapSum operator's
        continuous relaxation instead of a ReLU threshold.

        Default alpha=1.0, hard_threshold_coeff=2.0, threshold=1.8 are the
        paper's own published fixed-budget reference config
        (train_configs/soft_topk.sh in the official repo, which used
        topk=1.5 as its own comparison point).

        aux_loss_coeff (default None = disabled) applies the same Switch-
        style load-balancing formula used elsewhere in this file
        (aux_loss = sum(probs_per_expert * tokens_per_expert) * n_experts *
        coeff / (n_tokens^2 * realized_avg_active_experts)), matching the
        official source's own convention of scaling by the REALIZED average
        active-expert count rather than the nominal `topk` (source:
        router.py's aux_loss_load_balancing passes `topk=active_experts`,
        not the configured topk, to the load-balancing loss).

        capacity_factor: None (default) disables capacity limiting entirely
            -- bit-identical to the pre-existing algorithm (see
            test_softtopk_capacity_none_is_bit_identical). When set, applies
            the IDENTICAL convention documented in `relu_forward`'s
            `capacity_factor` docstring, adapted to this method's own
            variable-cardinality `routing_map`/`probs` (post-threshold-
            truncation): capacity is tracked independently per expert
            (`capacity = ceil(capacity_factor * n_tokens / n_experts)`),
            arrival order within an expert's queue is ascending flattened-
            token index (the same row-major order as `flat`), and a dropped
            (token, expert) pair contributes `probs[token, expert] *
            flat[token]` (residual/identity passthrough weighted by that
            pair's own post-truncation probability) instead of
            `probs[token, expert] * expert_fn(flat[token])`. Capacity
            dropping does not affect `aux_loss` (computed from the pre-drop
            `probs`/`routing_map`, same reasoning as `relu_forward`).
        """
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        n = flat.shape[0]

        logits = self.router(flat)
        k_tensor = torch.full((n,), float(topk), device=x.device, dtype=torch.float32)
        alpha_tensor = torch.tensor(float(alpha), device=x.device, dtype=torch.float32)
        probs_fp32 = soft_top_k(logits.to(torch.float32), k_tensor, alpha_tensor)

        experts_threshold = min(math.ceil(topk * hard_threshold_coeff), self.n_experts)
        topk_vals, _ = torch.topk(probs_fp32.detach(), k=experts_threshold, dim=1)
        lower_bound = topk_vals[:, -1].unsqueeze(1)
        higher_bound = topk_vals[:, 0].unsqueeze(1)
        thresh = torch.clamp(
            torch.full_like(lower_bound, threshold * topk / self.n_experts),
            min=lower_bound, max=higher_bound,
        )
        probs_fp32 = torch.where(probs_fp32 >= thresh, probs_fp32, torch.zeros_like(probs_fp32))
        probs = probs_fp32.to(logits.dtype)
        routing_map = probs > 0

        dropped_mask = None
        if capacity_factor is not None:
            capacity = math.ceil(capacity_factor * n / self.n_experts)
            dropped_mask = torch.zeros_like(routing_map)
            for e_idx in range(self.n_experts):
                expert_positions = routing_map[:, e_idx].nonzero(as_tuple=True)[0]
                if expert_positions.numel() > capacity:
                    dropped_mask[expert_positions[capacity:], e_idx] = True

        out = torch.zeros_like(flat)
        for e_idx in range(self.n_experts):
            mask = routing_map[:, e_idx]
            drop_e = dropped_mask[:, e_idx] if dropped_mask is not None else None
            active_mask = mask & ~drop_e if drop_e is not None else mask
            if active_mask.any():
                out[active_mask] = out[active_mask] + probs[active_mask, e_idx : e_idx + 1] * self.experts[e_idx](flat[active_mask])
            if drop_e is not None and drop_e.any():
                out[drop_e] = out[drop_e] + probs[drop_e, e_idx : e_idx + 1] * flat[drop_e]

        aux_loss = torch.zeros((), device=x.device, dtype=probs.dtype)
        if aux_loss_coeff is not None and aux_loss_coeff > 0:
            active_experts = torch.clamp(routing_map.sum().float() / n, min=1e-6)
            softmax_probs = F.softmax(logits, dim=-1)
            probs_per_expert = softmax_probs.sum(dim=0)
            tokens_per_expert = routing_map.sum(dim=0).to(probs.dtype)
            aux_loss = torch.sum(probs_per_expert * tokens_per_expert) * (
                self.n_experts * aux_loss_coeff / (n * n * active_experts)
            )

        return out.reshape(shape), routing_map.reshape(shape[:-1] + (self.n_experts,)), aux_loss


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_len: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_len, max_len)).unsqueeze(0).unsqueeze(0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        out = (att @ v).transpose(1, 2).reshape(B, T, D)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, n_experts: int, max_len: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_len)
        self.ln2 = nn.LayerNorm(d_model)
        self.moe = MoEBlock(d_model, d_ff, n_experts)

    def forward_soft(self, x: torch.Tensor):
        x = x + self.attn(self.ln1(x))
        moe_out, logits, aux_loss = self.moe.soft_forward(self.ln2(x))
        x = x + moe_out
        return x, logits, aux_loss

    def forward_hard(
        self,
        x: torch.Tensor,
        forced_indices: torch.Tensor | None = None,
        shuffle: bool = False,
        random_mode: bool = False,
        capacity_factor: float | None = None,
        top_k_dispatch: int = 1,
    ):
        x = x + self.attn(self.ln1(x))
        moe_out, indices, aux_loss = self.moe.hard_forward(
            self.ln2(x), forced_indices=forced_indices, shuffle=shuffle, random_mode=random_mode,
            capacity_factor=capacity_factor, top_k_dispatch=top_k_dispatch,
        )
        x = x + moe_out
        return x, indices, aux_loss

    def forward_relu(
        self,
        x: torch.Tensor,
        l1_reg_coeff: float | None = None,
        target_topk: int = 1,
        capacity_factor: float | None = None,
    ):
        x = x + self.attn(self.ln1(x))
        moe_out, routing_map, l1_loss = self.moe.relu_forward(
            self.ln2(x), l1_reg_coeff=l1_reg_coeff, target_topk=target_topk,
            capacity_factor=capacity_factor,
        )
        x = x + moe_out
        return x, routing_map, l1_loss

    def forward_softtopk(
        self,
        x: torch.Tensor,
        topk: float = 1.0,
        alpha: float = 1.0,
        threshold: float = 1.8,
        hard_threshold_coeff: float = 2.0,
        aux_loss_coeff: float | None = None,
        capacity_factor: float | None = None,
    ):
        x = x + self.attn(self.ln1(x))
        moe_out, routing_map, aux_loss = self.moe.softtopk_forward(
            self.ln2(x), topk=topk, alpha=alpha, threshold=threshold,
            hard_threshold_coeff=hard_threshold_coeff, aux_loss_coeff=aux_loss_coeff,
            capacity_factor=capacity_factor,
        )
        x = x + moe_out
        return x, routing_map, aux_loss


class MoETransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 1024,
        n_experts: int = 4,
        max_len: int = 512,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_experts = n_experts
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, n_experts, max_len)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        mode: str = "soft",
        forced_routes: list | None = None,
        shuffle: bool = False,
        random_mode: bool = False,
        capacity_factor: float | None = None,
        top_k_dispatch: int = 1,
        l1_reg_coeff: float | None = None,
        target_topk: int = 1,
        topk: float = 1.0,
        alpha: float = 1.0,
        threshold: float = 1.8,
        hard_threshold_coeff: float = 2.0,
        aux_loss_coeff: float | None = None,
    ):
        """
        mode == "softtopk": SoftMoE (Zasada et al., ICML 2026) LapSum soft
            top-k routing, truncated to sparse execution -- a VARIABLE (not
            fixed-k) number of active experts per token, softmax/topk/
            l1_reg_coeff/target_topk are ignored. `topk` (fixed-budget
            target, default 1.0), `alpha` (LapSum temperature), `threshold`/
            `hard_threshold_coeff` (truncation-to-sparsity parameters), and
            `aux_loss_coeff` (Switch-style load-balancing term, scaled by
            the REALIZED average active-expert count) -- see
            MoEBlock.softtopk_forward's docstring for the exact mechanism
            and formula citations. `capacity_factor` (default None =
            unlimited) IS applied here -- see MoEBlock.softtopk_forward's
            docstring for the exact capacity/drop convention (adapted for
            this mode's variable-cardinality routing_map). `forced_routes`,
            `shuffle`, `random_mode`, `top_k_dispatch` are ignored.
        mode == "relu": ReMoE (Wang et al., ICLR 2025) ReLU routing at every
            layer -- probs = relu(router_logits), routing_map = probs > 0,
            a variable (not fixed-k) number of active experts per token,
            weighted by the raw (non-renormalized) relu value. `l1_reg_coeff`
            (default None = no regularization) and `target_topk` control the
            switch-style L1 sparsity term -- see MoEBlock.relu_forward's
            docstring for the exact formula. `capacity_factor` (default None
            = unlimited) IS applied here -- see MoEBlock.relu_forward's
            docstring for the exact capacity/drop convention (adapted for
            this mode's variable-cardinality routing_map). `forced_routes`,
            `shuffle`, `random_mode`, `top_k_dispatch` are ignored.
        mode == "soft": full softmax mixture at every layer. `forced_routes`,
            `shuffle`, `random_mode`, `capacity_factor`, `top_k_dispatch` are
            ignored (the soft mixture has no notion of expert capacity or
            top-k dispatch width -- every expert contributes to every token
            via its mixture weight).
        mode == "hard": hard top-k dispatch at every layer (top_k_dispatch=1,
            the default, is top-1, bit-identical to the original algorithm).
            Per layer i:
              - if forced_routes is not None and forced_routes[i] is not
                None: dispatch using that forced index tensor (no router
                call at all for that layer).
              - elif random_mode: uniformly random expert, no router call.
              - elif shuffle: closed-loop router decision, then permuted
                before dispatch.
              - else: plain closed-loop (decision from current dispatched
                hidden state).
              `capacity_factor` (default None = unlimited) is applied
              identically in every one of the above branches -- see
              MoEBlock.hard_forward's docstring for the exact capacity
              formula and drop convention.

        Returns (lm_logits, final_hidden, all_router_logits, all_indices,
        total_aux_loss). `final_hidden` is the tensor entering ln_f (i.e.
        before the final norm and lm_head). `all_router_logits` is populated
        only in "soft" mode (list of (B*T, n_experts) tensors, one per
        layer); `all_indices` is populated in "hard" mode (list of (B, T)
        tensors, one per layer) and in "relu" mode (list of (B, T, n_experts)
        boolean routing_map tensors, one per layer).
        """
        assert mode in ("soft", "hard", "relu", "softtopk")
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)

        all_router_logits = []
        all_indices = []
        total_aux = torch.tensor(0.0, device=input_ids.device)

        for i, block in enumerate(self.blocks):
            if mode == "soft":
                x, logits, aux = block.forward_soft(x)
                all_router_logits.append(logits)
            elif mode == "relu":
                x, routing_map, aux = block.forward_relu(
                    x, l1_reg_coeff=l1_reg_coeff, target_topk=target_topk,
                    capacity_factor=capacity_factor,
                )
                all_indices.append(routing_map)
            elif mode == "softtopk":
                x, routing_map, aux = block.forward_softtopk(
                    x, topk=topk, alpha=alpha, threshold=threshold,
                    hard_threshold_coeff=hard_threshold_coeff, aux_loss_coeff=aux_loss_coeff,
                    capacity_factor=capacity_factor,
                )
                all_indices.append(routing_map)
            else:
                fr = None
                if forced_routes is not None:
                    fr = forced_routes[i]
                x, indices, aux = block.forward_hard(
                    x, forced_indices=fr, shuffle=shuffle, random_mode=random_mode,
                    capacity_factor=capacity_factor, top_k_dispatch=top_k_dispatch,
                )
                all_indices.append(indices)
            total_aux = total_aux + aux

        final_hidden = x
        x_normed = self.ln_f(x)
        lm_logits = self.lm_head(x_normed)
        return lm_logits, final_hidden, all_router_logits, all_indices, total_aux

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


# ═══════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════

class TokenizedDataset(Dataset):
    def __init__(self, tokens: torch.Tensor, seq_len: int):
        self.tokens = tokens
        self.seq_len = seq_len
        self.n_seqs = (len(tokens) - 1) // seq_len

    def __len__(self):
        return self.n_seqs

    def __getitem__(self, idx):
        start = idx * self.seq_len
        x = self.tokens[start : start + self.seq_len]
        y = self.tokens[start + 1 : start + self.seq_len + 1]
        return x, y


def load_wikitext_cached(data_dir: str, seq_len: int = 512, vocab_size: int = 50257):
    """Only the cached (.npy) path. The cold path in
    revision_moe_coadaptation.py is broken (references undefined
    cache_train/cache_val); we deliberately do not replicate it. If the
    cache is missing, fail loudly rather than silently falling back."""
    train_npy = Path(data_dir) / "wt103_train_tokens.npy"
    val_npy = Path(data_dir) / "wt103_val_tokens.npy"
    if not (train_npy.exists() and val_npy.exists()):
        raise FileNotFoundError(
            f"Expected cached tokens at {train_npy} and {val_npy}. "
            "This script only supports the cached path."
        )
    train_tok = torch.from_numpy(np.load(str(train_npy))).long()
    val_tok = torch.from_numpy(np.load(str(val_npy))).long()
    train_ds = TokenizedDataset(train_tok, seq_len)
    val_ds = TokenizedDataset(val_tok, seq_len)
    return train_ds, val_ds, vocab_size


# ═══════════════════════════════════════════════════════════════════════
# Corpus dispatch (corpus-generality extension)
#
# WikiText-103 remains the default and goes through load_wikitext_cached()
# UNCHANGED (bit-identical -- see test_wikitext_path_unchanged_by_default in
# tests/test_moe_soft_to_hard.py). FineWeb-Edu is added via the shared
# src/data_loading.py module (load_corpus()), following the exact
# integration pattern already used by src/sparse_attention_fineweb.py and
# src/sparse_attention_hierarchical.py: data_loading yields items shaped
# {"input_ids": tensor of length seq_len+1}; callers split that into
# (x, y) = (ids[:-1], ids[1:]). data_loading.py itself is not modified.
# ═══════════════════════════════════════════════════════════════════════

class _XYDatasetWrapper(Dataset):
    """Wraps a data_loading Dataset yielding {"input_ids": (seq_len+1,)}
    (or a bare tensor) into the (x, y) tuple shape this script's train/eval
    loops expect, matching TokenizedDataset's own __getitem__ contract."""

    def __init__(self, base: Dataset):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        ids = item["input_ids"] if isinstance(item, dict) else item
        return ids[:-1], ids[1:]


class _XYIterableDatasetWrapper(torch.utils.data.IterableDataset):
    """Same unpacking as _XYDatasetWrapper, for the IterableDataset case
    (data_loading.ShardedTokenDataset, used for FineWeb-Edu train shards)."""

    def __init__(self, base):
        self.base = base

    def __iter__(self):
        for item in self.base:
            ids = item["input_ids"] if isinstance(item, dict) else item
            yield ids[:-1], ids[1:]


def load_corpus_for_moe(
    corpus: str,
    data_dir: str,
    seq_len: int = 512,
    seed: int = 0,
    rank: int = 0,
    world_size: int = 1,
):
    """Dispatch to the requested corpus, explicitly -- no silent fallback.

    "wikitext-103" (the default) is routed to load_wikitext_cached() and is
    therefore bit-identical to this script's pre-extension behavior.
    "fineweb-edu" is routed through src/data_loading.py's load_corpus(),
    the same shared module already used by src/sparse_attention_fineweb.py
    and src/sparse_attention_hierarchical.py. Any other value raises
    ValueError immediately -- this project has previously been bitten by a
    confounded run caused by an ambiguous/wrong corpus (see
    results/RUN_LEDGER.md Section 2g), so an unrecognized corpus name must
    fail loudly here rather than defaulting to WikiText.

    Returns (train_ds, val_ds, vocab_size), where both datasets yield
    (x, y) tuples of shape (seq_len,) each, matching TokenizedDataset.
    """
    if corpus == "wikitext-103":
        return load_wikitext_cached(data_dir, seq_len)

    elif corpus == "fineweb-edu":
        # Local import: avoids adding data_loading's heavier deps
        # (transformers/datasets) to the default WikiText-only startup path.
        from data_loading import load_corpus as _load_corpus

        train_base = _load_corpus(
            "fineweb-edu", data_dir, seq_len=seq_len, split="train",
            tokenizer_name="gpt2", cycling=True,
            rank=rank, world_size=world_size, seed=seed,
        )
        val_base = _load_corpus(
            "fineweb-edu", data_dir, seq_len=seq_len, split="validation",
            tokenizer_name="gpt2", cycling=False,
            rank=rank, world_size=world_size, seed=seed,
        )

        if isinstance(train_base, torch.utils.data.IterableDataset):
            train_ds = _XYIterableDatasetWrapper(train_base)
        else:
            train_ds = _XYDatasetWrapper(train_base)
        val_ds = _XYDatasetWrapper(val_base)

        vocab_size = 50257  # gpt2 tokenizer vocab, per data_loading's "gpt2" backend
        return train_ds, val_ds, vocab_size

    else:
        raise ValueError(
            f"Unknown corpus: {corpus!r}. Choose from: wikitext-103, fineweb-edu"
        )


# ═══════════════════════════════════════════════════════════════════════
# Evaluation: five modes + derived metrics
# ═══════════════════════════════════════════════════════════════════════

def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate_relu(
    model: MoETransformer,
    loader,
    device: torch.device,
    max_batches: int | None = None,
    capacity_factor: float | None = None,
) -> dict:
    """ReMoE has no soft/hard mismatch -- routing (mode="relu") is identical
    at train and eval time, so unlike run_full_evaluation there is only one
    mode to measure. Returns token-count-weighted mean NLL under mode="relu"
    plus the mean number of active experts per token (averaged over all
    tokens and layers), a direct measure of the L1 regularization's actual
    realized sparsity.

    `capacity_factor` (default None = unlimited, bit-identical to the
    pre-existing behavior) is passed straight through to mode="relu" --
    see MoEBlock.relu_forward's docstring for the capacity/drop convention.
    `avg_active_experts` counts routing_map True entries exactly as before
    (i.e. still counts a dropped (token, expert) pair as "active", since
    the pair WAS routed there by the router -- capacity dropping is a
    downstream resource constraint on the forward computation, not a
    change to the routing decision this statistic measures)."""
    model.eval()

    nll_sum = 0.0
    token_count = 0
    active_sum = 0.0
    active_count = 0

    n_batches = 0
    for x, y in loader:
        if max_batches is not None and n_batches >= max_batches:
            break
        n_batches += 1
        x, y = x.to(device), y.to(device)
        n_tok = y.numel()

        logits, _, _, all_indices, _ = model(x, mode="relu", capacity_factor=capacity_factor)
        ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        nll_sum += ce.item() * n_tok
        token_count += n_tok

        for routing_map in all_indices:
            active_sum += routing_map.sum().item()
            active_count += routing_map[..., 0].numel()

    return {
        "relu_nll": nll_sum / token_count,
        "avg_active_experts": active_sum / active_count,
    }


@torch.no_grad()
def evaluate_softtopk(
    model: MoETransformer,
    loader,
    device: torch.device,
    max_batches: int | None = None,
    topk: float = 1.0,
    alpha: float = 1.0,
    threshold: float = 1.8,
    hard_threshold_coeff: float = 2.0,
    capacity_factor: float | None = None,
) -> dict:
    """SoftMoE's truncated soft top-k is its OWN inference-time execution
    (no separate hard-dispatch deployment step to compare against) -- like
    evaluate_relu, there is only one mode to measure, not run_full_
    evaluation's five CLHR-specific modes. Returns token-count-weighted
    mean NLL under mode="softtopk" plus the mean number of active experts
    per token (averaged over all tokens and layers), the realized sparsity
    the truncation actually achieves relative to the nominal `topk`
    budget.

    `capacity_factor` (default None = unlimited, bit-identical to the
    pre-existing behavior) is passed straight through to mode="softtopk" --
    see MoEBlock.softtopk_forward's docstring for the capacity/drop
    convention. `avg_active_experts` is unaffected by capacity dropping for
    the same reason documented in evaluate_relu."""
    model.eval()

    nll_sum = 0.0
    token_count = 0
    active_sum = 0.0
    active_count = 0

    n_batches = 0
    for x, y in loader:
        if max_batches is not None and n_batches >= max_batches:
            break
        n_batches += 1
        x, y = x.to(device), y.to(device)
        n_tok = y.numel()

        logits, _, _, all_indices, _ = model(
            x, mode="softtopk", topk=topk, alpha=alpha, threshold=threshold,
            hard_threshold_coeff=hard_threshold_coeff, capacity_factor=capacity_factor,
        )
        ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        nll_sum += ce.item() * n_tok
        token_count += n_tok

        for routing_map in all_indices:
            active_sum += routing_map.sum().item()
            active_count += routing_map[..., 0].numel()

    return {
        "softtopk_nll": nll_sum / token_count,
        "avg_active_experts": active_sum / active_count,
    }


def benchmark_inference(
    model: MoETransformer,
    mode: str,
    device: torch.device,
    batch_size: int = 8,
    seq_len: int = 256,
    vocab_size: int = 1000,
    input_ids: torch.Tensor | None = None,
    n_warmup: int = 5,
    n_iters: int = 20,
    capacity_factor: float | None = None,
    top_k_dispatch: int = 1,
    l1_reg_coeff: float | None = None,
    target_topk: int = 1,
    topk: float = 1.0,
    alpha: float = 1.0,
    threshold: float = 1.8,
    hard_threshold_coeff: float = 2.0,
    seed: int | None = None,
) -> dict:
    """Wall-clock inference throughput/latency benchmark comparing mode=
    "hard" (fixed-k dispatch, a static/predictable per-token compute
    pattern) against mode="relu"/"softtopk" (variable-cardinality
    dispatch, a ragged/data-dependent compute pattern), at matched
    architecture/batch/seq_len/device/dtype, in eval mode under
    torch.no_grad().

    IMPORTANT SCOPE NOTE: this measures the STRUCTURAL compute cost of the
    routing mechanism given the model's architecture and its CURRENT
    realized dispatch pattern -- it is not a claim about a specific
    trained checkpoint's exact behavior. mode="relu"/"softtopk"'s realized
    average-active-experts count depends on the router logits actually
    produced (e.g. untrained random-init logits give a different realized
    sparsity than a trained model would, which changes the ragged
    dispatch's actual FLOPs/sec). This is a known, documented limitation
    of a benchmark run against a freshly-initialized model, not a bug --
    callers who want a specific trained checkpoint's throughput should
    load its weights into `model` before calling this function.

    If `input_ids` is not given, a fixed random batch of shape
    (batch_size, seq_len) is generated once (via torch.randint, optionally
    seeded via `seed`) and reused for every warmup/timed iteration --
    matched-batch, matched-everything-else, as required. mode-specific
    kwargs (top_k_dispatch/capacity_factor for "hard"; l1_reg_coeff/
    target_topk for "relu"; topk/alpha/threshold/hard_threshold_coeff for
    "softtopk") are forwarded to MoETransformer.forward exactly as the
    corresponding evaluate_relu/evaluate_softtopk/run_full_evaluation
    helpers already do for their respective modes -- capacity_factor is
    honored for every mode (None = unlimited, unchanged default).

    Timing: runs `n_warmup` UNTIMED iterations first (JIT/allocator/cache
    warmup), then `n_iters` TIMED iterations. Synchronization is device-
    type-branched and applied BEFORE AND AFTER EACH timed iteration (not
    just once around the whole timed block): CUDA kernel launches are
    asynchronous, so without a sync immediately before starting a given
    iteration's clock and immediately after stopping it, that iteration's
    recorded wall-clock time could include work queued by (or bleed into)
    a neighboring iteration, corrupting the per-batch latency distribution
    (min/max/mean) even though the TOTAL elapsed time across all iterations
    would still come out roughly correct. Per-iteration sync costs a small
    amount of serialization overhead but is required for correct
    per-iteration latency; total throughput is computed from the summed,
    correctly-attributed per-iteration times. For device.type == "mps",
    torch.mps.synchronize() is used if available; for "cpu", no
    synchronization is needed (torch ops on CPU are already synchronous
    with the calling Python thread).

    Returns a dict with keys: mode, batch_size, seq_len, n_iters,
    total_elapsed_s, tokens_per_sec, mean_latency_s, min_latency_s,
    max_latency_s.
    """
    if mode not in ("hard", "relu", "softtopk"):
        raise ValueError(f"benchmark_inference: unsupported mode {mode!r} (expected 'hard', 'relu', or 'softtopk')")

    model.eval()
    model.to(device)

    if input_ids is None:
        gen = torch.Generator(device="cpu")
        if seed is not None:
            gen.manual_seed(seed)
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), generator=gen)
    input_ids = input_ids.to(device)
    actual_batch_size, actual_seq_len = input_ids.shape

    forward_kwargs = {"mode": mode, "capacity_factor": capacity_factor}
    if mode == "hard":
        forward_kwargs["top_k_dispatch"] = top_k_dispatch
    elif mode == "relu":
        forward_kwargs["l1_reg_coeff"] = l1_reg_coeff
        forward_kwargs["target_topk"] = target_topk
    else:  # softtopk
        forward_kwargs["topk"] = topk
        forward_kwargs["alpha"] = alpha
        forward_kwargs["threshold"] = threshold
        forward_kwargs["hard_threshold_coeff"] = hard_threshold_coeff

    def _sync():
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps" and hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()

    with torch.no_grad():
        for _ in range(n_warmup):
            model(input_ids, **forward_kwargs)
        _sync()

        latencies = []
        for _ in range(n_iters):
            _sync()
            t0 = time.perf_counter()
            model(input_ids, **forward_kwargs)
            _sync()
            t1 = time.perf_counter()
            latencies.append(t1 - t0)

    total_elapsed = sum(latencies)
    tokens_per_sec = (actual_batch_size * actual_seq_len * n_iters) / total_elapsed

    return {
        "mode": mode,
        "batch_size": actual_batch_size,
        "seq_len": actual_seq_len,
        "n_iters": n_iters,
        "total_elapsed_s": total_elapsed,
        "tokens_per_sec": tokens_per_sec,
        "mean_latency_s": total_elapsed / n_iters,
        "min_latency_s": min(latencies),
        "max_latency_s": max(latencies),
    }


@torch.no_grad()
def run_full_evaluation(
    model: MoETransformer,
    loader,
    device: torch.device,
    max_batches: int | None = None,
    n_random_draws: int = 5,
    capacity_factor: float | None = None,
    top_k_dispatch: int = 1,
):
    """Single sweep over (up to) max_batches of `loader` computing all five
    evaluation modes plus the cosine-similarity and histogram statistics
    needed for the registered metrics. Returns a dict with exactly the keys
    the caller (main script / tests) needs to assemble the final metrics
    JSON, EXCLUDING run metadata (condition, seed, etc.) which the caller
    knows and this function does not.

    `capacity_factor` (default None = unlimited, bit-identical to the
    pre-existing behavior) is passed through to every one of the four hard-
    dispatch modes evaluated here (open_loop_top1, closed_loop_top1,
    shuffled_top1, random_top1) -- applied identically in each, per
    MoEBlock.hard_forward's consistency rule (a capacity limit is a
    resource constraint, not a property of how the routing decision was
    produced). native_soft is unaffected (no notion of capacity in the
    soft-mixture path).

    `top_k_dispatch` (default 1, bit-identical to the pre-existing
    behavior) is likewise passed through to all four hard-dispatch modes.
    open_loop_top1's forced routes are built from the soft pass's top-k
    logits (`logits.topk(top_k_dispatch, dim=-1).indices`, generalizing the
    original `logits.argmax(dim=-1)`); shuffled_top1's external permutation
    of the closed-loop indices operates on rows of the (N, top_k_dispatch)
    per-layer index matrix instead of on scalars, matching
    MoEBlock.hard_forward's own internal shuffle generalization (decision
    2)."""
    model.eval()

    soft_loss_sum = 0.0
    open_loss_sum = 0.0
    closed_loss_sum = 0.0
    shuffled_loss_sum = 0.0
    token_count = 0

    cos_sum = 0.0
    cos_count = 0

    histogram = torch.zeros(model.n_experts, dtype=torch.long)

    random_draw_loss_sums = [0.0 for _ in range(n_random_draws)]
    random_token_count = 0

    n_batches = 0
    for x, y in loader:
        if max_batches is not None and n_batches >= max_batches:
            break
        n_batches += 1
        first_batch = n_batches == 1
        x, y = x.to(device), y.to(device)
        n_tok = y.numel()

        # --- native_soft (also gives us the soft final hidden state + the
        # router logits needed to build the open-loop forced routes) ---
        if first_batch:
            print("[eval] mode native_soft: starting", flush=True)
        soft_logits, soft_hidden, soft_router_logits, _, _ = model(x, mode="soft")
        soft_loss_sum += F.cross_entropy(
            soft_logits.reshape(-1, soft_logits.size(-1)), y.reshape(-1), reduction="sum"
        ).item()

        # --- open_loop_top1: indices from the soft pass, dispatched fresh ---
        if first_batch:
            print("[eval] mode open_loop_top1: starting", flush=True)
        if top_k_dispatch == 1:
            open_loop_routes = [
                logits.argmax(dim=-1).reshape(x.shape[0], x.shape[1]) for logits in soft_router_logits
            ]
        else:
            open_loop_routes = [
                logits.topk(top_k_dispatch, dim=-1).indices.reshape(x.shape[0], x.shape[1], top_k_dispatch)
                for logits in soft_router_logits
            ]
        open_logits, _, _, _, _ = model(
            x, mode="hard", forced_routes=open_loop_routes, capacity_factor=capacity_factor,
            top_k_dispatch=top_k_dispatch,
        )
        open_loss_sum += F.cross_entropy(
            open_logits.reshape(-1, open_logits.size(-1)), y.reshape(-1), reduction="sum"
        ).item()

        # --- closed_loop_top1: the deployment metric, one pass ---
        if first_batch:
            print("[eval] mode closed_loop_top1: starting", flush=True)
        closed_logits, closed_hidden, _, closed_indices, _ = model(
            x, mode="hard", capacity_factor=capacity_factor, top_k_dispatch=top_k_dispatch,
        )
        closed_loss_sum += F.cross_entropy(
            closed_logits.reshape(-1, closed_logits.size(-1)), y.reshape(-1), reduction="sum"
        ).item()
        for layer_indices in closed_indices:
            counts = torch.bincount(layer_indices.reshape(-1).cpu(), minlength=model.n_experts)
            histogram += counts

        # --- cosine similarity between soft and closed-loop final hidden states ---
        a = soft_hidden.reshape(soft_hidden.shape[0], -1)
        b = closed_hidden.reshape(closed_hidden.shape[0], -1)
        cos = F.cosine_similarity(a, b, dim=-1)
        cos_sum += cos.sum().item()
        cos_count += cos.shape[0]

        # --- shuffled_top1: permute the *actual* closed-loop indices per
        # layer across the flattened token axis, then replay forced ---
        if first_batch:
            print("[eval] mode shuffled_top1: starting", flush=True)
        shuffled_routes = []
        for layer_indices in closed_indices:
            # top_k_dispatch=1: layer_indices is (B, T) -- permute scalars, as
            # before. top_k_dispatch>1: layer_indices is (B, T, k) -- permute
            # ROWS of the (N, k) matrix (decision 2, generalized identically
            # to MoEBlock.hard_forward's own internal shuffle branch).
            flat = layer_indices.reshape(-1, top_k_dispatch) if top_k_dispatch > 1 else layer_indices.reshape(-1)
            perm = torch.randperm(flat.shape[0], device=flat.device)
            shuffled_routes.append(flat[perm].reshape(layer_indices.shape))
        shuffled_logits, _, _, _, _ = model(
            x, mode="hard", forced_routes=shuffled_routes, capacity_factor=capacity_factor,
            top_k_dispatch=top_k_dispatch,
        )
        shuffled_loss_sum += F.cross_entropy(
            shuffled_logits.reshape(-1, shuffled_logits.size(-1)), y.reshape(-1), reduction="sum"
        ).item()

        # --- random_top1: n_random_draws independent draws ---
        if first_batch:
            print(f"[eval] mode random_top1: starting ({n_random_draws} draws)", flush=True)
        for d in range(n_random_draws):
            rand_logits, _, _, _, _ = model(
                x, mode="hard", random_mode=True, capacity_factor=capacity_factor, top_k_dispatch=top_k_dispatch,
            )
            random_draw_loss_sums[d] += F.cross_entropy(
                rand_logits.reshape(-1, rand_logits.size(-1)), y.reshape(-1), reduction="sum"
            ).item()
        random_token_count += n_tok

        token_count += n_tok

    native_soft_nll = soft_loss_sum / token_count
    open_loop_top1_nll = open_loss_sum / token_count
    closed_loop_top1_nll = closed_loss_sum / token_count
    shuffled_top1_nll = shuffled_loss_sum / token_count
    random_draw_nlls = [s / random_token_count for s in random_draw_loss_sums]
    random_top1_nll = float(np.mean(random_draw_nlls))
    random_top1_std = float(np.std(random_draw_nlls))

    one_minus_cos_L = 1.0 - (cos_sum / cos_count)

    G_CL = closed_loop_top1_nll - native_soft_nll
    G_OL = open_loop_top1_nll - native_soft_nll
    compounding_ratio = G_CL / max(G_OL, 1e-6)
    gate_utility = random_top1_nll - closed_loop_top1_nll
    G_CL_shuffled = shuffled_top1_nll - native_soft_nll

    return {
        "native_soft_nll": native_soft_nll,
        "open_loop_top1_nll": open_loop_top1_nll,
        "closed_loop_top1_nll": closed_loop_top1_nll,
        "shuffled_top1_nll": shuffled_top1_nll,
        "random_top1_nll": random_top1_nll,
        "random_top1_std": random_top1_std,
        "G_CL": G_CL,
        "G_OL": G_OL,
        "compounding_ratio": compounding_ratio,
        "gate_utility": gate_utility,
        "G_CL_shuffled": G_CL_shuffled,
        "one_minus_cos_L": one_minus_cos_L,
        "expert_load_histogram": histogram.tolist(),
    }


# ═══════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════

def clhr_loss(
    model: MoETransformer, x: torch.Tensor, y: torch.Tensor, lambda_rca: float, aux_coeff: float,
    capacity_factor: float | None = None, top_k_dispatch: int = 1,
):
    """standard: aux_coeff * aux_soft is folded into L_soft as in the base
    repo's convention (loss = lm_loss + aux_coeff * aux_loss).
    clhr: (L_soft + lambda_rca * L_hard) / (1 + lambda_rca), where each of
    L_soft / L_hard already includes its own aux term.

    `capacity_factor` (default None = unlimited) is passed straight through
    to the closed-loop hard-dispatch forward that computes L_hard, so CLHR
    training is capacity-aware when the flag is set: the hard branch the
    model is trained against is the SAME capacity-limited deployment
    behavior it will be evaluated under (run_full_evaluation's
    closed_loop_top1 also receives the same capacity_factor). Training on
    an uncapacitated hard pass while evaluating under a capacitated one
    would reintroduce exactly the kind of train/deploy mismatch this flag
    exists to study."""
    soft_logits, _, _, _, soft_aux = model(x, mode="soft")
    L_soft = F.cross_entropy(soft_logits.reshape(-1, soft_logits.size(-1)), y.reshape(-1))
    L_soft = L_soft + aux_coeff * soft_aux

    if lambda_rca == 0.0:
        return L_soft, L_soft, torch.tensor(0.0, device=x.device)

    hard_logits, _, _, _, hard_aux = model(
        x, mode="hard", capacity_factor=capacity_factor, top_k_dispatch=top_k_dispatch,
    )
    L_hard = F.cross_entropy(hard_logits.reshape(-1, hard_logits.size(-1)), y.reshape(-1))
    L_hard = L_hard + aux_coeff * hard_aux

    combined = (L_soft + lambda_rca * L_hard) / (1.0 + lambda_rca)
    return combined, L_soft, L_hard


def train_model(
    model: MoETransformer,
    train_loader,
    device: torch.device,
    condition: str,
    steps: int,
    lr: float,
    lambda_rca: float,
    aux_coeff: float,
    log_every: int = 200,
    capacity_factor: float | None = None,
    top_k_dispatch: int = 1,
    grad_accum_steps: int = 1,
    l1_reg_coeff: float | None = None,
    relu_target_topk: int = 1,
    softtopk_topk: float = 1.0,
    softtopk_alpha: float = 1.0,
    softtopk_threshold: float = 1.8,
    softtopk_hard_threshold_coeff: float = 2.0,
    softtopk_aux_loss_coeff: float | None = None,
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    train_iter = iter(train_loader)
    model.train()

    t0 = time.time()
    interval_t0 = t0
    steps_completed = 0

    # Running-mean accumulators for the current logging interval. Kept as
    # on-device tensors and only converted to python floats (.item()) at
    # print time, so logging never adds a host/device sync more than once
    # per `log_every` steps.
    soft_loss_sum = torch.zeros((), device=device)
    hard_loss_sum = torch.zeros((), device=device)
    aux_loss_sum = torch.zeros((), device=device)
    hard_loss_count = 0
    interval_steps = 0

    for step in range(steps):
        optimizer.zero_grad()

        # Micro-batch accumulation: grad_accum_steps forward/backward passes
        # share ONE optimizer.step(), so the accumulated gradient equals the
        # gradient of a single pass over a batch grad_accum_steps times
        # larger (each micro-batch's loss is divided by grad_accum_steps
        # before .backward(); autograd sums gradients across repeated
        # .backward() calls between zero_grad()/step() by default -- exactly
        # the arithmetic needed for the accumulated gradient to equal the
        # mean-loss gradient over the concatenated larger batch, PROVIDED
        # every micro-batch is the same size, which the caller must ensure).
        # grad_accum_steps=1 (the default) reduces to exactly the prior
        # single-batch behavior -- see
        # test_grad_accum_steps_1_is_bit_identical.
        micro_soft_ce_sum = torch.zeros((), device=device)
        micro_hard_sum = torch.zeros((), device=device)
        micro_hard_count = 0
        micro_aux_sum = torch.zeros((), device=device)
        for _ in range(grad_accum_steps):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)
            x, y = x.to(device), y.to(device)

            hard_loss_val = None
            if condition == "standard":
                soft_logits, _, _, _, soft_aux = model(x, mode="soft")
                soft_ce = F.cross_entropy(soft_logits.reshape(-1, soft_logits.size(-1)), y.reshape(-1))
                micro_loss = soft_ce + aux_coeff * soft_aux
                aux_val = soft_aux
            elif condition == "clhr":
                micro_loss, l_soft, l_hard = clhr_loss(
                    model, x, y, lambda_rca, aux_coeff, capacity_factor=capacity_factor, top_k_dispatch=top_k_dispatch,
                )
                soft_ce = l_soft
                aux_val = None  # folded into l_soft/l_hard by clhr_loss; not separable here
                if lambda_rca != 0.0:
                    hard_loss_val = l_hard
            elif condition == "hard":
                # Genuine hard-dispatch-from-scratch training (Switch/Mixtral/
                # GShard/DeepSeekMoE's own recipe): only the selected expert(s)
                # are ever computed, no full soft mixture at any point. Reuses
                # model(x, mode="hard", ...), already differentiable through
                # the selected expert(s)' own softmax-derived gate weight --
                # no soft branch, no lambda_rca mixing (there is nothing to mix).
                if lambda_rca != 0.0:
                    raise ValueError(
                        "condition='hard' has no soft branch, so lambda_rca "
                        "(CLHR's soft/hard mixing weight) does not apply -- "
                        "pass lambda_rca=0.0"
                    )
                hard_logits, _, _, _, hard_aux = model(
                    x, mode="hard", capacity_factor=capacity_factor, top_k_dispatch=top_k_dispatch,
                )
                hard_ce = F.cross_entropy(hard_logits.reshape(-1, hard_logits.size(-1)), y.reshape(-1))
                micro_loss = hard_ce + aux_coeff * hard_aux
                soft_ce = hard_ce  # logged in both the "soft" and "hard" columns for this condition
                aux_val = None
                hard_loss_val = hard_ce
            elif condition == "relu":
                # ReMoE (Wang et al., ICLR 2025) ReLU routing from scratch:
                # no soft branch, so lambda_rca has nothing to mix, exactly
                # as with condition="hard".
                if lambda_rca != 0.0:
                    raise ValueError(
                        "condition='relu' has no soft branch, so lambda_rca "
                        "(CLHR's soft/hard mixing weight) does not apply -- "
                        "pass lambda_rca=0.0"
                    )
                relu_logits, _, _, _, l1_loss = model(
                    x, mode="relu", l1_reg_coeff=l1_reg_coeff, target_topk=relu_target_topk,
                    capacity_factor=capacity_factor,
                )
                relu_ce = F.cross_entropy(relu_logits.reshape(-1, relu_logits.size(-1)), y.reshape(-1))
                micro_loss = relu_ce + l1_loss
                soft_ce = relu_ce  # logged in the "soft" column for this condition
                aux_val = None
                hard_loss_val = None
            elif condition == "softmoe":
                # SoftMoE (Zasada et al., ICML 2026) LapSum soft top-k
                # routing from scratch: no separate hard-dispatch branch --
                # the truncated soft top-k IS the deployment-time execution
                # -- so lambda_rca has nothing to mix, exactly as with
                # condition="hard"/"relu".
                if lambda_rca != 0.0:
                    raise ValueError(
                        "condition='softmoe' has no soft branch, so lambda_rca "
                        "(CLHR's soft/hard mixing weight) does not apply -- "
                        "pass lambda_rca=0.0"
                    )
                softtopk_logits, _, _, _, softtopk_aux = model(
                    x, mode="softtopk", topk=softtopk_topk, alpha=softtopk_alpha,
                    threshold=softtopk_threshold, hard_threshold_coeff=softtopk_hard_threshold_coeff,
                    aux_loss_coeff=softtopk_aux_loss_coeff, capacity_factor=capacity_factor,
                )
                softtopk_ce = F.cross_entropy(
                    softtopk_logits.reshape(-1, softtopk_logits.size(-1)), y.reshape(-1),
                )
                micro_loss = softtopk_ce + softtopk_aux
                soft_ce = softtopk_ce  # logged in the "soft" column for this condition
                aux_val = None
                hard_loss_val = None
            else:
                raise ValueError(f"unknown condition {condition}")

            (micro_loss / grad_accum_steps).backward()

            micro_soft_ce_sum = micro_soft_ce_sum + soft_ce.detach()
            if hard_loss_val is not None:
                micro_hard_sum = micro_hard_sum + hard_loss_val.detach()
                micro_hard_count += 1
            if aux_val is not None:
                micro_aux_sum = micro_aux_sum + aux_val.detach()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        steps_completed += 1

        # Accumulate scalars cheaply (no .item() here -- stays on device).
        # Logged as the MEAN across this step's micro-batches, matching
        # what a single grad_accum_steps-times-larger batch would report.
        soft_loss_sum = soft_loss_sum + micro_soft_ce_sum / grad_accum_steps
        if micro_hard_count > 0:
            hard_loss_sum = hard_loss_sum + micro_hard_sum / micro_hard_count
            hard_loss_count += 1
        if condition == "standard":
            aux_loss_sum = aux_loss_sum + micro_aux_sum / grad_accum_steps
        interval_steps += 1

        is_last_step = step == steps - 1
        if log_every > 0 and ((step + 1) % log_every == 0 or is_last_step):
            now = time.time()
            interval_elapsed = max(now - interval_t0, 1e-9)
            mean_soft = (soft_loss_sum / max(interval_steps, 1)).item()
            mean_hard = (
                (hard_loss_sum / hard_loss_count).item() if hard_loss_count > 0 else None
            )
            mean_aux = (
                (aux_loss_sum / max(interval_steps, 1)).item() if condition == "standard" else None
            )
            steps_per_sec = interval_steps / interval_elapsed
            elapsed_total = now - t0
            pct = 100.0 * (step + 1) / steps
            remaining_steps = steps - (step + 1)
            eta = remaining_steps / max(steps_per_sec, 1e-9)

            hard_str = f"{mean_hard:.4f}" if mean_hard is not None else "n/a"
            aux_str = f"{mean_aux:.4f}" if mean_aux is not None else "n/a"
            print(
                f"[train] step {step + 1}/{steps} ({pct:.1f}%) "
                f"loss_soft={mean_soft:.4f} loss_hard={hard_str} aux_loss={aux_str} "
                f"steps/s={steps_per_sec:.2f} elapsed={elapsed_total:.1f}s eta={eta:.1f}s",
                flush=True,
            )

            soft_loss_sum = torch.zeros((), device=device)
            hard_loss_sum = torch.zeros((), device=device)
            aux_loss_sum = torch.zeros((), device=device)
            hard_loss_count = 0
            interval_steps = 0
            interval_t0 = now

    elapsed = time.time() - t0
    return steps_completed, elapsed


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="A1: soft-to-hard MoE discretization experiment")
    parser.add_argument("--condition", type=str, default="standard", choices=["standard", "clhr", "hard", "relu", "softmoe"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--corpus", type=str, default="wikitext-103",
        choices=["wikitext-103", "fineweb-edu"],
        help="Training/eval corpus. Default 'wikitext-103' preserves this "
             "script's original (pre-corpus-generality-extension) behavior "
             "exactly. 'fineweb-edu' reads sharded tokens via "
             "src/data_loading.py from --data-dir (expects "
             "<data-dir>/fineweb_edu_shards/ and <data-dir>/fineweb_edu_eval/).",
    )
    parser.add_argument("--data-dir", type=str, default="./wikitext103_cache")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--lambda-rca", type=float, default=1.0)
    parser.add_argument("--aux-coeff", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--d-ff", type=int, default=1024)
    parser.add_argument("--n-experts", type=int, default=4)
    parser.add_argument("--max-eval-batches", type=int, default=100)
    parser.add_argument("--random-draws", type=int, default=5)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--log-every", type=int, default=200,
                         help="print a training progress line every N steps (and on the final step)")
    parser.add_argument(
        "--capacity-factor", type=float, default=None,
        help="Switch-Transformer-style per-expert capacity factor, shared "
             "across --condition hard/clhr/relu/softmoe. Default None = "
             "unlimited (bit-identical to the original algorithm, no token "
             "dropping). When set (e.g. 1.25), each expert's per-call "
             "capacity is ceil(capacity_factor * n_tokens / n_experts); "
             "(token, expert) pairs routed to an expert beyond capacity are "
             "dropped (arrival-order selection, residual/identity "
             "passthrough for dropped pairs -- see MoEBlock.hard_forward's "
             "docstring for fixed-k top-1/top-k dispatch, and "
             "MoEBlock.relu_forward's / MoEBlock.softtopk_forward's "
             "docstrings for the variable-cardinality generalization used "
             "by --condition relu/softmoe). Applied identically in training "
             "and in every eval path for the relevant condition (hard/clhr: "
             "the CLHR hard branch and every hard-dispatch eval mode -- "
             "open_loop_top1, closed_loop_top1, shuffled_top1, random_top1; "
             "relu: evaluate_relu; softmoe: evaluate_softtopk).",
    )
    parser.add_argument(
        "--top-k-dispatch", type=int, default=1,
        help="Number of experts each token is dispatched to in hard mode "
             "(Mixtral-style top-k combine). Default 1 = top-1, bit-"
             "identical to the original algorithm (see "
             "test_top_k_dispatch_1_is_bit_identical). When set (e.g. 2, "
             "Mixtral's standard choice), the top-k router logits are "
             "softmaxed over ONLY those k values to get per-token per-slot "
             "combine weights that sum to 1, and the expert outputs are "
             "accumulated (weight * expert(token)) rather than assigned. "
             "Applied identically in training (the CLHR hard branch) and "
             "in every hard-dispatch eval mode (open_loop_top1, "
             "closed_loop_top1, shuffled_top1, random_top1); interacts with "
             "--capacity-factor by flattening all (token, slot) pairs "
             "routed to an expert before applying the same capacity/drop "
             "rule -- see MoEBlock.hard_forward docstring.",
    )
    parser.add_argument(
        "--l1-reg-coeff", type=float, default=None,
        help="ReMoE (condition='relu') only: coefficient of the switch-style "
             "L1 load-balancing regularizer applied to the raw relu router "
             "outputs. Default None = no regularization (unbounded number "
             "of active experts per token). Ignored by every other "
             "condition.",
    )
    parser.add_argument(
        "--relu-target-topk", type=int, default=1,
        help="ReMoE (condition='relu') only: the 'topk' divisor in the L1 "
             "regularization formula (matches thu-ml/ReMoE's convention of "
             "expressing the target average active-expert count relative to "
             "a nominal top-k). Ignored by every other condition.",
    )
    parser.add_argument(
        "--softtopk-topk", type=float, default=1.0,
        help="SoftMoE (condition='softmoe') only: fixed nominal expert "
             "budget for the LapSum soft top-k operator (this project's "
             "fixed-budget comparison variant, not the paper's learned "
             "global per-layer budget). Ignored by every other condition.",
    )
    parser.add_argument(
        "--softtopk-alpha", type=float, default=1.0,
        help="SoftMoE (condition='softmoe') only: LapSum temperature. "
             "Default 1.0 matches the paper's own reference fixed-budget "
             "config (train_configs/soft_topk.sh). Ignored by every other "
             "condition.",
    )
    parser.add_argument(
        "--softtopk-threshold", type=float, default=1.8,
        help="SoftMoE (condition='softmoe') only: truncation-to-sparsity "
             "threshold parameter. Default 1.8 matches the paper's own "
             "reference fixed-budget config. Ignored by every other "
             "condition.",
    )
    parser.add_argument(
        "--softtopk-hard-threshold-coeff", type=float, default=2.0,
        help="SoftMoE (condition='softmoe') only: multiplier on --softtopk-"
             "topk defining how many top-ranked experts bound the "
             "truncation threshold. Default 2.0 matches the paper's own "
             "reference fixed-budget config. Ignored by every other "
             "condition.",
    )
    parser.add_argument(
        "--softtopk-aux-loss-coeff", type=float, default=None,
        help="SoftMoE (condition='softmoe') only: coefficient of the "
             "switch-style load-balancing term (scaled by the realized "
             "average active-expert count, matching the official source's "
             "own convention). Default None = no regularization. Ignored "
             "by every other condition.",
    )
    parser.add_argument(
        "--benchmark-inference", action="store_true",
        help="Skip training and the usual NLL/perplexity evaluation entirely. "
             "Instead, run benchmark_inference (wall-clock inference "
             "throughput/latency) for --condition's corresponding mode "
             "(hard/clhr/standard -> mode='hard'; relu -> mode='relu'; "
             "softmoe -> mode='softtopk') on a freshly-initialized model "
             "(no training, no --data-dir corpus loading -- input_ids are "
             "generated internally via torch.randint of shape "
             "(--batch-size, --seq-len)), and write the benchmark dict to "
             "--output instead of the usual training+eval metrics. This "
             "measures the STRUCTURAL compute cost of the routing "
             "mechanism given its architecture and current (random-init) "
             "realized dispatch pattern -- see benchmark_inference's "
             "docstring for why that's a documented limitation, not a bug, "
             "for mode='relu'/'softtopk' (whose realized sparsity depends "
             "on the router logits actually produced).",
    )
    parser.add_argument(
        "--benchmark-warmup-iters", type=int, default=5,
        help="--benchmark-inference only: number of UNTIMED warmup "
             "forward passes run before the timed loop.",
    )
    parser.add_argument(
        "--benchmark-iters", type=int, default=20,
        help="--benchmark-inference only: number of TIMED forward passes "
             "the throughput/latency statistics are computed over.",
    )
    parser.add_argument(
        "--benchmark-vocab-size", type=int, default=1000,
        help="--benchmark-inference only: vocabulary size used to build "
             "the freshly-initialized model and to generate random "
             "input_ids (no real corpus/tokenizer is loaded for a "
             "benchmark-only run).",
    )
    return parser


def main(argv=None):
    args = build_argparser().parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    run_t0 = time.time()

    device = select_device()

    if args.benchmark_inference:
        # Pure structural inference-throughput benchmark: no training, no
        # --data-dir corpus loading (see --benchmark-inference's help text
        # and benchmark_inference's docstring for the documented scope
        # limitation of measuring a freshly-initialized model).
        print("=" * 72, flush=True)
        print("A1 soft-to-hard MoE -- inference throughput benchmark (no training)", flush=True)
        print(f"  condition           : {args.condition}", flush=True)
        print(f"  device              : {device}", flush=True)
        print(f"  d_model/n_heads/n_layers/d_ff/n_experts : "
              f"{args.d_model}/{args.n_heads}/{args.n_layers}/{args.d_ff}/{args.n_experts}", flush=True)
        print(f"  batch_size/seq_len  : {args.batch_size}/{args.seq_len}", flush=True)
        print(f"  benchmark_warmup_iters/benchmark_iters : "
              f"{args.benchmark_warmup_iters}/{args.benchmark_iters}", flush=True)
        print("=" * 72, flush=True)

        model = MoETransformer(
            args.benchmark_vocab_size,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            d_ff=args.d_ff,
            n_experts=args.n_experts,
            max_len=args.seq_len,
        ).to(device)

        condition_to_mode = {
            "standard": "hard", "clhr": "hard", "hard": "hard",
            "relu": "relu", "softmoe": "softtopk",
        }
        bench_mode = condition_to_mode[args.condition]

        result = benchmark_inference(
            model, bench_mode, device,
            batch_size=args.batch_size, seq_len=args.seq_len,
            vocab_size=args.benchmark_vocab_size,
            n_warmup=args.benchmark_warmup_iters, n_iters=args.benchmark_iters,
            capacity_factor=args.capacity_factor, top_k_dispatch=args.top_k_dispatch,
            l1_reg_coeff=args.l1_reg_coeff, target_topk=args.relu_target_topk,
            topk=args.softtopk_topk, alpha=args.softtopk_alpha,
            threshold=args.softtopk_threshold,
            hard_threshold_coeff=args.softtopk_hard_threshold_coeff,
            seed=args.seed,
        )
        result.update({
            "condition": args.condition,
            "seed": args.seed,
            "n_experts": args.n_experts,
            "n_layers": args.n_layers,
            "d_model": args.d_model,
            "d_ff": args.d_ff,
            "capacity_factor": args.capacity_factor,
            "top_k_dispatch": args.top_k_dispatch,
            "hardware": f"{platform.system()}-{platform.machine()}-{device.type}",
            "model_params": model.count_parameters(),
        })

        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"Saved benchmark to {args.output}", flush=True)
        print(json.dumps(result, indent=2, default=str), flush=True)
        return

    train_ds, val_ds, vocab_size = load_corpus_for_moe(
        args.corpus, args.data_dir, seq_len=args.seq_len, seed=args.seed
    )
    # IterableDataset (the FineWeb-Edu sharded path) forbids `shuffle=` entirely
    # -- torch.utils.data.DataLoader raises ValueError if it's passed at all,
    # even shuffle=False, regardless of value. shuffling for that corpus is
    # handled upstream by ShardedTokenDataset's own shard-order/seed logic
    # (src/data_loading.py), not by DataLoader. WikiText's TokenizedDataset is
    # a regular (non-iterable) Dataset and is unaffected -- shuffle=True stays
    # exactly as before for that path.
    train_loader_kwargs = dict(batch_size=args.batch_size, num_workers=0, drop_last=True)
    if not isinstance(train_ds, torch.utils.data.IterableDataset):
        train_loader_kwargs["shuffle"] = True
    train_loader = DataLoader(train_ds, **train_loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=True)

    model = MoETransformer(
        vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        n_experts=args.n_experts,
        max_len=args.seq_len,
    ).to(device)
    model_params = model.count_parameters()

    print("=" * 72, flush=True)
    print("A1 soft-to-hard MoE discretization -- startup", flush=True)
    print(f"  condition          : {args.condition}", flush=True)
    print(f"  corpus              : {args.corpus}", flush=True)
    print(f"  seed                : {args.seed}", flush=True)
    print(f"  device              : {device}", flush=True)
    print(f"  model parameters    : {model_params / 1e6:.2f}M ({model_params})", flush=True)
    print(f"  total planned steps : {args.steps}", flush=True)
    print(f"  batch_size          : {args.batch_size}", flush=True)
    print(f"  seq_len             : {args.seq_len}", flush=True)
    print(f"  lambda_rca          : {args.lambda_rca}", flush=True)
    print(f"  aux_coeff           : {args.aux_coeff}", flush=True)
    print(f"  lr                  : {args.lr}", flush=True)
    print(f"  d_model/n_heads/n_layers/d_ff/n_experts : "
          f"{args.d_model}/{args.n_heads}/{args.n_layers}/{args.d_ff}/{args.n_experts}", flush=True)
    print(f"  log_every           : {args.log_every}", flush=True)
    print(f"  capacity_factor     : {args.capacity_factor}", flush=True)
    print(f"  top_k_dispatch      : {args.top_k_dispatch}", flush=True)
    print(f"  l1_reg_coeff        : {args.l1_reg_coeff}", flush=True)
    print(f"  relu_target_topk    : {args.relu_target_topk}", flush=True)
    print(f"  softtopk_topk       : {args.softtopk_topk}", flush=True)
    print(f"  softtopk_alpha      : {args.softtopk_alpha}", flush=True)
    print(f"  softtopk_threshold  : {args.softtopk_threshold}", flush=True)
    print(f"  softtopk_hard_threshold_coeff : {args.softtopk_hard_threshold_coeff}", flush=True)
    print(f"  softtopk_aux_loss_coeff       : {args.softtopk_aux_loss_coeff}", flush=True)
    print("  >>> training starting", flush=True)
    print("=" * 72, flush=True)

    steps_completed, elapsed = train_model(
        model, train_loader, device, args.condition, args.steps, args.lr, args.lambda_rca, args.aux_coeff,
        log_every=args.log_every, capacity_factor=args.capacity_factor, top_k_dispatch=args.top_k_dispatch,
        grad_accum_steps=args.grad_accum_steps, l1_reg_coeff=args.l1_reg_coeff,
        relu_target_topk=args.relu_target_topk, softtopk_topk=args.softtopk_topk,
        softtopk_alpha=args.softtopk_alpha, softtopk_threshold=args.softtopk_threshold,
        softtopk_hard_threshold_coeff=args.softtopk_hard_threshold_coeff,
        softtopk_aux_loss_coeff=args.softtopk_aux_loss_coeff,
    )
    print(f"[train] training complete: {steps_completed} steps in {elapsed:.1f}s", flush=True)

    if args.checkpoint_dir:
        ckpt_dir = Path(args.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "steps": steps_completed}, ckpt_dir / f"{args.condition}_s{args.seed}_final.pt")

    if args.condition == "relu":
        # ReMoE has no soft/hard mismatch to measure -- run_full_evaluation's
        # five modes (native_soft, open_loop, closed_loop, shuffled, random)
        # are all specific to the soft-mixture/hard-dispatch CLHR framing and
        # do not apply.
        print("[eval] starting relu evaluation", flush=True)
        metrics = evaluate_relu(
            model, val_loader, device, max_batches=args.max_eval_batches,
            capacity_factor=args.capacity_factor,
        )
        print("[eval] evaluation complete", flush=True)
    elif args.condition == "softmoe":
        # SoftMoE's truncated soft top-k IS its own deployment-time
        # execution -- no separate hard-dispatch mismatch to measure.
        print("[eval] starting softmoe evaluation", flush=True)
        metrics = evaluate_softtopk(
            model, val_loader, device, max_batches=args.max_eval_batches,
            topk=args.softtopk_topk, alpha=args.softtopk_alpha,
            threshold=args.softtopk_threshold,
            hard_threshold_coeff=args.softtopk_hard_threshold_coeff,
            capacity_factor=args.capacity_factor,
        )
        print("[eval] evaluation complete", flush=True)
    else:
        print("[eval] starting full evaluation (5 modes)", flush=True)
        metrics = run_full_evaluation(
            model, val_loader, device, max_batches=args.max_eval_batches, n_random_draws=args.random_draws,
            capacity_factor=args.capacity_factor, top_k_dispatch=args.top_k_dispatch,
        )
        print("[eval] evaluation complete", flush=True)
    metrics.update({
        "condition": args.condition,
        "seed": args.seed,
        "n_experts": args.n_experts,
        "n_layers": args.n_layers,
        "d_model": args.d_model,
        "d_ff": args.d_ff,
        "lambda_rca": args.lambda_rca,
        "aux_coeff": args.aux_coeff,
        "capacity_factor": args.capacity_factor,
        "top_k_dispatch": args.top_k_dispatch,
        "l1_reg_coeff": args.l1_reg_coeff,
        "relu_target_topk": args.relu_target_topk,
        "softtopk_topk": args.softtopk_topk,
        "softtopk_alpha": args.softtopk_alpha,
        "softtopk_threshold": args.softtopk_threshold,
        "softtopk_hard_threshold_coeff": args.softtopk_hard_threshold_coeff,
        "softtopk_aux_loss_coeff": args.softtopk_aux_loss_coeff,
        "steps_completed": steps_completed,
        "training_elapsed_seconds": elapsed,
        "hardware": f"{platform.system()}-{platform.machine()}-{device.type}",
        "model_params": model_params,
        "corpus": args.corpus,
        "data_dir": args.data_dir,
    })

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"Saved to {args.output}", flush=True)

    print("=" * 72, flush=True)
    print("Final metrics", flush=True)
    for key in (
        "native_soft_nll", "open_loop_top1_nll", "closed_loop_top1_nll",
        "shuffled_top1_nll", "random_top1_nll", "random_top1_std",
        "G_CL", "G_OL", "compounding_ratio", "gate_utility",
        "G_CL_shuffled", "one_minus_cos_L",
        "relu_nll", "softtopk_nll", "avg_active_experts",
    ):
        val = metrics.get(key)
        val_str = f"{val:.6f}" if isinstance(val, (int, float)) else str(val)
        print(f"  {key:<24} {val_str}", flush=True)
    print("=" * 72, flush=True)
    print(json.dumps(metrics, indent=2, default=str), flush=True)

    total_elapsed = time.time() - run_t0
    print(f"[done] total wall-clock elapsed: {total_elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
