"""
DeepSeek Sparse Attention (DSA) / "lightning indexer" -- from-scratch port
into this project's Transformer scaffold, for use as a baseline comparison
against Closed-Loop Hard Routing (CLHR).

Why this file exists
---------------------
An external review of the CLHR paper asked for a comparison against DSA
(DeepSeek-V3.2-Exp), described as the closest known real-world instance of
exactly the "soft/dense-train then hard-deploy" pattern CLHR targets: a
lightweight "lightning indexer" module scores each key against the query
cheaply, selects top-k keys, and full attention is computed only over the
selected set; training reportedly uses a short dense/full-attention warmup
before switching to the sparse indexer-gated regime. The review's explicit
ask: "measure the gap at the switch; apply CLHR during the sparse phase."

MECHANISM PROVENANCE -- read this before citing this file as "DSA"
--------------------------------------------------------------------
Source consulted: https://github.com/deepseek-ai/DeepSeek-V3.2-Exp
(verified real, 1651 stars; a full production-model release repo, not a
small standalone attention module). Fetches performed: README.md (main),
the GitHub API recursive file-tree listing (main), inference/model.py
(main), inference/kernel.py (main). No 404s; no arXiv/technical-report link
exists anywhere in the README (confirmed absent, not just unfetched) -- so
there is no paper to cross-check against either.

VERIFIED FROM SOURCE (inference/model.py, inference/kernel.py):
  - An `Indexer(torch.nn.Module)` class exists, with `ModelArgs` defaults
    `index_n_heads=64`, `index_head_dim=128`, `index_topk=2048`.
  - Its submodules: `wq_b: Linear(q_lora_rank, n_heads*head_dim)` (a
    *per-head* query projection), `wk: Linear(dim, head_dim)` (a *single*
    key projection shared across all index heads -- this is the "low-rank"
    cost-saving trick), `k_norm: LayerNorm(head_dim)` applied to that shared
    key, and `weights_proj: Linear(dim, n_heads)` producing per-head
    combination weights from the query token itself.
  - `Indexer.forward` applies non-interleaved RoPE to the query, quantizes
    query/key to fp8, calls a kernel `fp8_index()` to get `index_score`,
    then `topk_indices = index_score.topk(min(index_topk, end_pos), dim=-1)`.
  - Inside `MLA.forward`, the indexer's output hard-masks the *main*
    attention's own scores (not the indexer's own scores):
    `index_mask = full(-inf).scatter_(-1, topk_indices, 0); scores += index_mask`.
    I.e. the indexer only decides *which keys are visible*; the actual
    attention weights among visible keys are still the main Q/K's own
    dot products.
  - inference/kernel.py contains only generic fp8 quant/gemm/index kernels
    (`act_quant`, `fp8_gemm`, `fp8_index` backed by a compiled
    `fp8_index_kernel_`) -- no training code anywhere in this repo.

NOT VERIFIED (documentation-only, or genuinely unknown):
  - The exact scalar-combination arithmetic inside the fp8 `fp8_index`
    kernel was NOT retrieved -- it's backed by a compiled kernel, not
    visible Python. This port's combination formula (sum over index heads
    of a per-head learned weight times a per-head query/shared-key dot
    product, scaled by 1/sqrt(head_dim) -- see `LightningIndexer.forward`
    below) is a standard reconstruction *consistent with* the verified
    module composition (wq_b/wk/k_norm/weights_proj), not a byte-for-byte-
    verified formula.
  - The two-phase "dense warmup then switch to sparse" TRAINING RECIPE --
    warmup step count/fraction, whether the indexer has its own auxiliary
    or distillation loss, whether the dense-to-sparse transition is abrupt
    or gradual -- is UNKNOWN from this repo. inference/model.py and
    kernel.py are an inference-only reference implementation; there is no
    training loop, loss function, or paper in this repo to check against.
    Everything about the *training recipe* in this file (the `standard`
    condition's abrupt cutover, the indexer's distillation loss, the
    `clhr` condition's transition window) is this project's own research-
    design reconstruction of DSA's *publicly reported* behavior (per this
    task's own framing), not something read out of DeepSeek's source.
  - A consequence of the verified mask mechanism worth flagging explicitly:
    the hard top-k mask is a *constant* 0/-inf tensor keyed only on *which*
    indices were selected (`torch.topk` indices, non-differentiable) --
    it does not depend continuously on `index_score`'s magnitude. So once
    attention is hardened, the indexer's own parameters (`wq_b`, `wk`,
    `k_norm`, `weights_proj`) receive **zero gradient** from the main LM
    loss. Some auxiliary training signal for the indexer is therefore
    *necessary*, not optional -- this file supplies one (a KL distillation
    of the indexer's softmax against the main attention's own detached
    softmax, `indexer_distillation_loss` below), which is a reasonable and
    commonly-used choice for this exact problem but is NOT something read
    out of DeepSeek's source (their training code isn't public here).

Design departure from this project's OWN sparse-attention convention
----------------------------------------------------------------------
`GatedSparseAttention` elsewhere in this repo (src/sparse_attention_300m.py,
src/sparse_attention_fineweb.py) is a *soft-gated* mechanism: a learned
sigmoid gate log-additively biases attention scores, with no genuinely hard
top-k anywhere in training. DSA, as verified from source above, has no such
soft-gate regime at all: attention is either fully dense (plain causal
softmax) or hard top-k masked. This file is a faithful port of *DSA's own*
mechanism, so it does NOT reuse `GatedSparseAttention` -- it reuses only the
project's naming convention for the phase flag (`dense_mode`, matching
`GatedSparseAttention.forward(..., dense_mode=False)`) and its checkpoint/
CLI/DDP/device-handling patterns (see src/sparse_attention_300m.py).

CLHR mixing formula
--------------------
There is no shared/reusable CLHR-mixing utility module anywhere in this
repo (checked; every file reimplements it inline). This file replicates,
verbatim, the dual-forward weighted-average formula from
`src/moe_soft_to_hard.py:clhr_loss`:
    combined = (L_soft + lambda_rca * L_hard) / (1.0 + lambda_rca)
with the same `lambda_rca == 0.0` fast path (skip the hard forward, return
L_soft for all three slots) that file's own test
(`test_clhr_lambda_zero_equals_standard`) verifies.

** RESEARCH DESIGN AMBIGUITY -- flagged per the review's explicit ask **
"Apply CLHR during the sparse phase" / "mixing warmup-native soft attention
and post-switch hard indexer selection during a transition window" admits
at least two materially different, both-defensible implementations for
indexer-gated attention specifically (unlike MoE, where "soft" and "hard"
are two dispatch modes of the *same* router weights -- here "dense" and
"hard" are two different attention COMPUTATIONS, full softmax vs.
restricted-support softmax, built from the same Q/K):
  (1) DUAL-LOSS MIXING (implemented here, `clhr_mixed_loss`): run two full
      forward passes per step during the transition window (one dense_mode,
      one hard) and weighted-average the two losses. This is the literal
      generalization of this project's own established CLHR precedent
      (moe_soft_to_hard.py's clhr_loss) and is what's implemented below.
  (2) SCORE/MASK INTERPOLATION (NOT implemented here): a single forward
      pass per step where the additive mask itself is interpolated,
      e.g. `scores = raw_scores + alpha * hard_mask` with `alpha` ramping
      0->1 across the transition window (alpha=0 recovers dense; alpha=1
      recovers hard). This is architecturally more literal to "mixing
      attention during a transition window" (it blends at the attention-
      score level rather than the loss level) and is half the compute of
      (1), but has no precedent in this codebase and changes what "CLHR"
      means relative to how it's used everywhere else in this repo.
Both are legitimate readings of the review's ask. This file implements (1)
for consistency with the project's own established CLHR convention and
because it is directly testable against that convention's own test
(`test_clhr_mixed_loss_lambda_zero_equals_standard` in the test file here,
mirroring moe_soft_to_hard.py's precedent). (2) is flagged, not built, and
would need its own justification (and its own tests) before being used to
back a paper claim.
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset

CONDITIONS = ["standard", "clhr"]

# Real DSA config (inference/model.py ModelArgs), for reference only -- not
# used as defaults here since the toy/300M-scale models in this project are
# far smaller than DeepSeek-V3.2's production config.
REAL_DSA_INDEX_N_HEADS = 64
REAL_DSA_INDEX_HEAD_DIM = 128
REAL_DSA_INDEX_TOPK = 2048


# ═══════════════════════════════════════════════════════════════════════
# Lightning indexer + top-k hard masking
# ═══════════════════════════════════════════════════════════════════════

def causal_valid_mask(seq_len: int, device=None) -> torch.Tensor:
    """Bool[T,T], True at (i,j) iff key position j is causally visible to
    query position i (j <= i)."""
    idx = torch.arange(seq_len, device=device)
    return idx.unsqueeze(1) >= idx.unsqueeze(0)


def topk_hard_mask(score: torch.Tensor, k: int, causal_valid: torch.Tensor) -> torch.Tensor:
    """Generic top-k hard-masking primitive, used both for the indexer's own
    selection (`LightningIndexer`-produced score -> the actual DSA mechanism)
    and for the "open loop" oracle-selection eval baseline (raw QK score ->
    best-case hard sparsity with no indexer-approximation error).

    score: [..., T, T] (last two dims are query, key).
    Returns an additive mask (0.0 where allowed, -inf elsewhere) of the same
    shape, honoring both causality and "pick the top min(k, i+1) causal-
    valid keys for query i".

    Mirrors the verified DeepSeek source's construction
    (`index_mask.scatter_(-1, topk_indices, 0)` then added to scores), with
    one deliberate hardening beyond what's shown there: after building the
    mask from `topk_indices`, causal-invalid positions are unconditionally
    re-masked to -inf. This guards against a real correctness bug for early
    query positions with fewer than k causally-valid keys: torch.topk over
    a tensor containing many tied -inf entries (from causal masking) is not
    guaranteed to prefer in-range indices among the ties, so without this
    second pass it is possible for a future (causally-invalid) position to
    be selected into the "top k" and incorrectly marked allowed. DeepSeek's
    own inference code sidesteps this because its incremental-decoding
    `index_score` tensor is already sliced to exactly the valid range
    (`min(self.index_topk, end_pos)`), so no -inf padding is ever present
    within the range being topk'd; that per-position slicing isn't
    straightforward in this file's batched, full-sequence training-time
    setting, hence the explicit re-mask instead. See
    test_topk_mask_never_leaks_future_positions.
    """
    seq_len = score.shape[-1]
    kk = max(1, min(k, seq_len))
    masked_score = score.masked_fill(~causal_valid, float("-inf"))
    topk_idx = masked_score.topk(kk, dim=-1).indices
    mask = torch.full_like(score, float("-inf"))
    mask.scatter_(-1, topk_idx, 0.0)
    mask = mask.masked_fill(~causal_valid, float("-inf"))
    return mask


class LightningIndexer(nn.Module):
    """Lightweight query/key scoring module. Module composition verified
    from inference/model.py's `Indexer` class (see module docstring); the
    combination arithmetic is a reconstruction, not verified byte-for-byte.

    - wq_b: per-head query projection (index_n_heads independent heads).
    - wk + k_norm: a single shared low-rank key projection (NOT per-head --
      this is the "lightweight"/"low-rank" part of the mechanism).
    - weights_proj: per-head combination weights, produced from the query
      token itself, used to collapse the per-head scores into one scalar
      score per (query, key) pair.
    """

    def __init__(self, d_model: int, index_n_heads: int = 4, index_head_dim: int = 16):
        super().__init__()
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.wq_b = nn.Linear(d_model, index_n_heads * index_head_dim, bias=False)
        self.wk = nn.Linear(d_model, index_head_dim, bias=False)
        self.k_norm = nn.LayerNorm(index_head_dim)
        self.weights_proj = nn.Linear(d_model, index_n_heads, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, d_model] -> index_score: [B, T, T]."""
        B, T, _ = x.shape
        H, D = self.index_n_heads, self.index_head_dim
        q = self.wq_b(x).view(B, T, H, D)
        k = self.k_norm(self.wk(x))  # [B, T, D], shared across heads
        w = self.weights_proj(x)  # [B, T, H], per-head combination weight
        # per-head raw dot products: [B, H, T, S]
        per_head_scores = torch.einsum("bthd,bsd->bhts", q, k) / math.sqrt(D)
        # combine heads with the query token's own learned weights: [B, T, S]
        index_score = torch.einsum("bth,bhts->bts", w, per_head_scores)
        return index_score


def indexer_distillation_loss(
    index_score: torch.Tensor, dense_probs: torch.Tensor, causal_valid: torch.Tensor
) -> torch.Tensor:
    """KL(dense_probs || softmax(index_score)) restricted to causally-valid
    positions, averaged over query positions and batch. `dense_probs` is
    expected to already be detached (it is the main attention's own softmax
    output, used only as a distillation target here).

    Necessary because the hard top-k mask gives the indexer's own parameters
    zero gradient from the LM loss (see module docstring) -- this is the
    indexer's actual training signal.
    """
    masked_score = index_score.masked_fill(~causal_valid, float("-inf"))
    log_index_probs = F.log_softmax(masked_score, dim=-1)
    log_index_probs = torch.nan_to_num(log_index_probs, neginf=0.0)
    per_position_kl = F.kl_div(log_index_probs, dense_probs, reduction="none").sum(dim=-1)
    return per_position_kl.mean()


# ═══════════════════════════════════════════════════════════════════════
# Attention / Transformer
# ═══════════════════════════════════════════════════════════════════════

class DSAAttention(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, index_n_heads: int = 4, index_head_dim: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.indexer = LightningIndexer(d_model, index_n_heads, index_head_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, causal_valid: torch.Tensor, dense_mode: bool, topk: int):
        B, T, _ = x.shape
        H, Dh = self.n_heads, self.d_head

        def split_heads(t):
            return t.view(B, T, H, Dh).transpose(1, 2)  # [B,H,T,Dh]

        q = split_heads(self.W_q(x))
        k = split_heads(self.W_k(x))
        v = split_heads(self.W_v(x))

        raw_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(Dh)  # [B,H,T,T]
        causal_bias = torch.zeros_like(raw_scores[0, 0])
        causal_bias = causal_bias.masked_fill(~causal_valid, float("-inf"))
        dense_scores = raw_scores + causal_bias
        dense_probs = F.softmax(dense_scores, dim=-1)  # [B,H,T,T], used for the actual attention output
        dense_probs = torch.nan_to_num(dense_probs, nan=0.0)

        index_score = self.indexer(x)  # [B, T, T] -- no head dim (single combined score per key)

        # aux/diagnostic copies are head-collapsed to [B,T,T] to match index_score's
        # shape (needed for the distillation loss) and because the hard mask is shared
        # identically across heads anyway (mean preserves the exact sparsity pattern:
        # every head is exactly 0 at masked positions, so the mean is too).
        aux = {"index_score": index_score, "dense_probs": dense_probs.mean(dim=1).detach()}

        if dense_mode:
            attn_probs = dense_probs
        else:
            hard_mask = topk_hard_mask(index_score, topk, causal_valid)  # [B,T,T]
            hard_scores = raw_scores + hard_mask.unsqueeze(1)  # broadcast over heads
            hard_probs = F.softmax(hard_scores, dim=-1)
            hard_probs = torch.nan_to_num(hard_probs, nan=0.0)
            aux["hard_probs"] = hard_probs.mean(dim=1)
            attn_probs = hard_probs

        attn_probs = self.dropout(attn_probs)
        out = torch.matmul(attn_probs, v)  # [B,H,T,Dh]
        out = out.transpose(1, 2).contiguous().view(B, T, H * Dh)
        out = self.W_o(out)
        return out, aux


class TransformerBlock(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, d_ff: int, index_n_heads: int = 4,
        index_head_dim: int = 16, dropout: float = 0.1,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = DSAAttention(d_model, n_heads, index_n_heads, index_head_dim, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )

    def forward(self, x, causal_valid, dense_mode, topk):
        attn_out, aux = self.attn(self.ln1(x), causal_valid, dense_mode, topk)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x, aux


class DSATransformer(nn.Module):
    def __init__(
        self, vocab_size: int, d_model: int = 64, n_heads: int = 4, n_layers: int = 2,
        d_ff: int = 256, index_n_heads: int = 4, index_head_dim: int = 16,
        max_seq_len: int = 512, dropout: float = 0.1,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, n_heads, d_ff, index_n_heads, index_head_dim, dropout)
                for _ in range(n_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        self._causal_cache = {}
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        # Standard small-std transformer-LM init (N(0, 0.02), zero bias).
        # PyTorch's own nn.Embedding/nn.Linear defaults are N(0, 1)-scale,
        # which produces pathologically large logit magnitudes (and thus
        # huge initial cross-entropy loss) for a weight-tied lm_head -- this
        # is a standard/common fix, not something read out of this
        # project's other sparse_attention_*.py files (their exact init
        # scale was not confirmed).
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def _get_causal_valid(self, seq_len, device):
        key = (seq_len, device)
        if key not in self._causal_cache:
            self._causal_cache[key] = causal_valid_mask(seq_len, device)
        return self._causal_cache[key]

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, input_ids: torch.Tensor, dense_mode: bool, topk: int):
        B, T = input_ids.shape
        device = input_ids.device
        pos = torch.arange(T, device=device).unsqueeze(0)
        x = self.drop(self.tok_emb(input_ids) + self.pos_emb(pos))
        causal_valid = self._get_causal_valid(T, device)
        aux_list = []
        for block in self.blocks:
            x, aux = block(x, causal_valid, dense_mode, topk)
            aux_list.append(aux)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits, aux_list


# ═══════════════════════════════════════════════════════════════════════
# Dense-to-sparse switch scheduling + CLHR mixing
# ═══════════════════════════════════════════════════════════════════════

def step_mode(step: int, switch_step: int, transition_window: int, condition: str) -> str:
    """Which forward-pass regime a given training step should use.

    `standard`: DSA's own native recipe -- abrupt cutover at `switch_step`
    (dense for step < switch_step, hard from switch_step on). No known
    documented transition-softening exists for DSA (see module docstring);
    an abrupt cutover is the natural reconstruction of "dense warmup then
    switch" absent evidence of anything gentler -- CLHR is specifically the
    fix this project is testing for the gap that abrupt cutover produces.

    `clhr`: dense before switch_step, MIXED for `transition_window` steps
    starting AT switch_step (i.e. the window begins exactly at the switch,
    per the review's "apply CLHR during the sparse phase" framing), hard
    once the window has elapsed.
    """
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition: {condition!r} (expected one of {CONDITIONS})")
    if step < switch_step:
        return "dense"
    if condition == "standard":
        return "hard"
    # condition == "clhr"
    if step < switch_step + transition_window:
        return "mixed"
    return "hard"


def dense_loss(model, x, y, indexer_aux_weight: float):
    logits, aux_list = model(x, dense_mode=True, topk=1)
    lm_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
    aux_loss = _aggregate_indexer_aux_loss(model, aux_list, x.shape[1], x.device)
    loss = lm_loss + indexer_aux_weight * aux_loss
    return loss, {"lm_loss": lm_loss.item(), "indexer_aux_loss": aux_loss.item()}


def hard_loss(model, x, y, topk: int, indexer_aux_weight: float):
    logits, aux_list = model(x, dense_mode=False, topk=topk)
    lm_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
    aux_loss = _aggregate_indexer_aux_loss(model, aux_list, x.shape[1], x.device)
    loss = lm_loss + indexer_aux_weight * aux_loss
    return loss, {"lm_loss": lm_loss.item(), "indexer_aux_loss": aux_loss.item()}


def _aggregate_indexer_aux_loss(model, aux_list, seq_len, device):
    """Distillation loss is computed against `dense_probs`, which every
    forward pass (dense or hard) always produces internally (see
    DSAAttention.forward) -- so the indexer keeps receiving a training
    signal even after the switch to hard attention. This costs an extra
    dense-attention computation on every hard-mode step; a production DSA
    implementation would likely freeze indexer training after warmup
    specifically to avoid that cost. That tradeoff is a design choice made
    here for a cleaner/simpler port, not something verified from DeepSeek's
    source (no training code is public in the reference repo)."""
    causal_valid = model._get_causal_valid(seq_len, device) if hasattr(model, "_get_causal_valid") else \
        model.module._get_causal_valid(seq_len, device)
    losses = [
        indexer_distillation_loss(a["index_score"], a["dense_probs"], causal_valid)
        for a in aux_list
    ]
    return torch.stack(losses).mean()


def clhr_mixed_loss(model, x, y, lambda_rca: float, topk: int, indexer_aux_weight: float):
    """Dual-forward CLHR mixing at the dense-to-sparse switch (design choice
    (1) in the module docstring's ambiguity discussion). Mirrors
    src/moe_soft_to_hard.py:clhr_loss's exact formula and its
    lambda_rca == 0.0 fast path.

    Returns (combined, L_soft, L_hard).
    """
    l_soft, _ = dense_loss(model, x, y, indexer_aux_weight)
    if lambda_rca == 0.0:
        return l_soft, l_soft, l_soft
    l_hard, _ = hard_loss(model, x, y, topk, indexer_aux_weight)
    combined = (l_soft + lambda_rca * l_hard) / (1.0 + lambda_rca)
    return combined, l_soft, l_hard


# ═══════════════════════════════════════════════════════════════════════
# Evaluation: native / open-loop / closed-loop NLL, G_CL
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _nll_over_loader(model, loader, device, forward_fn, max_batches=None):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y = batch
        x, y = x.to(device), y.to(device)
        logits = forward_fn(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum")
        total_loss += loss.item()
        total_tokens += y.numel()
    model.train()
    return total_loss / max(total_tokens, 1)


def native_nll(model, loader, device, max_batches=None) -> float:
    """Dense (unmasked, plain causal softmax) forward -- the "soft-trained"
    regime's own evaluation of itself."""
    def fwd(x):
        logits, _ = model(x, dense_mode=True, topk=1)
        return logits
    return _nll_over_loader(model, loader, device, fwd, max_batches)


def closed_loop_nll(model, loader, device, topk: int, max_batches=None) -> float:
    """Hard attention gated by the indexer's OWN (learned, possibly
    imperfect) top-k selection -- this is what's actually deployed."""
    def fwd(x):
        logits, _ = model(x, dense_mode=False, topk=topk)
        return logits
    return _nll_over_loader(model, loader, device, fwd, max_batches)


@torch.no_grad()
def _open_loop_forward(model, x, topk: int):
    """Hard attention gated by an ORACLE top-k mask built from the true raw
    Q/K scores directly (bypassing the indexer entirely). This isolates the
    cost of hard sparsity itself from the cost of the indexer's approximation
    error -- an analog of this repo's existing open-loop/closed-loop
    distinction (src/moe_soft_to_hard.py's G_OL, src/sparse_attention_*'s
    oracle-mask evals), reconstructed for indexer-gated attention since no
    directly equivalent "open loop" notion is defined for DSA anywhere in
    the source material consulted."""
    raw_model = model.module if isinstance(model, DDP) else model
    B, T = x.shape
    device = x.device
    pos = torch.arange(T, device=device).unsqueeze(0)
    h = raw_model.drop(raw_model.tok_emb(x) + raw_model.pos_emb(pos))
    causal_valid = raw_model._get_causal_valid(T, device)
    for block in raw_model.blocks:
        attn = block.attn
        h_norm = block.ln1(h)
        Bb, Tt, _ = h_norm.shape
        Hh, Dh = attn.n_heads, attn.d_head

        def split_heads(t):
            return t.view(Bb, Tt, Hh, Dh).transpose(1, 2)

        q = split_heads(attn.W_q(h_norm))
        k = split_heads(attn.W_k(h_norm))
        v = split_heads(attn.W_v(h_norm))
        raw_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(Dh)
        # oracle mask built from the raw QK scores themselves, averaged over
        # heads to get a single per-(query,key) criterion comparable to the
        # indexer's own single scalar score.
        oracle_score = raw_scores.mean(dim=1)  # [B,T,T]
        mask = topk_hard_mask(oracle_score, topk, causal_valid)
        scores = raw_scores + mask.unsqueeze(1)
        probs = torch.nan_to_num(F.softmax(scores, dim=-1), nan=0.0)
        out = torch.matmul(probs, v).transpose(1, 2).contiguous().view(Bb, Tt, Hh * Dh)
        out = attn.W_o(out)
        h = h + out
        h = h + block.ffn(block.ln2(h))
    h = raw_model.ln_f(h)
    return raw_model.lm_head(h)


def open_loop_nll(model, loader, device, topk: int, max_batches=None) -> float:
    def fwd(x):
        return _open_loop_forward(model, x, topk)
    return _nll_over_loader(model, loader, device, fwd, max_batches)


def run_full_evaluation(model, loader, device, topk: int, max_batches=None) -> dict:
    """G_CL = closed_loop_nll - native_nll (matches the formula used
    throughout this repo: src/moe_soft_to_hard.py, src/sparse_attention_fineweb.py,
    src/sparse_attention_closed_loop_eval.py's run_audit). G_OL / compounding_ratio
    mirror src/moe_soft_to_hard.py's own metric-naming convention."""
    n_nll = native_nll(model, loader, device, max_batches)
    ol_nll = open_loop_nll(model, loader, device, topk, max_batches)
    cl_nll = closed_loop_nll(model, loader, device, topk, max_batches)
    g_cl = cl_nll - n_nll
    g_ol = ol_nll - n_nll
    compounding_ratio = g_cl / max(g_ol, 1e-6)
    return {
        "native_nll": n_nll,
        "open_loop_nll": ol_nll,
        "closed_loop_nll": cl_nll,
        "G_CL": g_cl,
        "G_OL": g_ol,
        "compounding_ratio": compounding_ratio,
    }


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ═══════════════════════════════════════════════════════════════════════
# Data
# ═══════════════════════════════════════════════════════════════════════

class CyclingTokenDataset(Dataset):
    """Byte-identical convention to src/sparse_attention_300m.py's
    CyclingTokenDataset: each item is tokens[start:start+seq_len+1]; callers
    split into x = item[:, :-1], y = item[:, 1:]."""

    def __init__(self, tokens: torch.Tensor, seq_len: int):
        self.tokens = tokens
        self.seq_len = seq_len
        self.n_seqs = max(1, (len(tokens) - 1) // seq_len)

    def __len__(self):
        return self.n_seqs

    def __getitem__(self, idx):
        idx = idx % self.n_seqs
        start = idx * self.seq_len
        return self.tokens[start: start + self.seq_len + 1]


def _split_xy(batch: torch.Tensor):
    return batch[:, :-1], batch[:, 1:]


def load_tokens(data_dir: str, split: str) -> torch.Tensor:
    """Mirrors src/sparse_attention_300m.py's load_training_data /
    load_val_data file-name search convention (wt103_{split}_tokens.pt/.npy),
    reconstructed from that file's code, not re-verified byte-for-byte here."""
    data_path = Path(data_dir)
    names = [f"wt103_{split}_tokens.pt", f"wt103_{split}_tokens.npy"]
    for name in names:
        p = data_path / name
        if p.exists():
            if name.endswith(".pt"):
                return torch.load(p, weights_only=True).long()
            return torch.from_numpy(np.load(p)).long()
    raise FileNotFoundError(f"No {split} data found in {data_dir} (looked for {names})")


class SyntheticLoader:
    """Deterministic-chain synthetic batches for CPU/MPS smoke tests -- no
    real data dependency. NOT used for the real vast.ai runs (those pass
    --data-dir and use load_tokens/CyclingTokenDataset above).

    Each sequence is x_0 ~ Uniform(vocab), x_{t+1} = f(x_t) for a FIXED
    random bijection f (a torch.randperm sampled once per loader instance
    from `seed`). This is deliberately content-dependent (not a purely
    positional pattern the model could learn from position embeddings
    alone) and trivially learnable (an embedding lookup + linear map
    suffices), so a smoke test can show a real, fast loss decrease rather
    than i.i.d.-random tokens' irreducible floor at ln(vocab_size) (which
    is what a naive torch.randint-for-both-x-and-y generator would produce,
    since genuinely independent random tokens carry no predictive signal).
    """

    def __init__(self, vocab_size, batch_size, seq_len, n_batches, seed=0):
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.n_batches = n_batches
        self.seed = seed
        self.perm = torch.randperm(vocab_size, generator=torch.Generator().manual_seed(seed))

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed + 1)
        for _ in range(self.n_batches):
            cur = torch.randint(0, self.vocab_size, (self.batch_size, 1), generator=g)
            seq = [cur]
            for _ in range(self.seq_len):
                cur = self.perm[cur]
                seq.append(cur)
            tokens = torch.cat(seq, dim=1)  # [B, seq_len+1]
            yield _split_xy(tokens)


class RealLoaderAdapter:
    """Wraps a torch DataLoader yielding [B, seq_len+1] token blocks
    (CyclingTokenDataset's convention) into (x, y) pairs."""

    def __init__(self, dataloader):
        self.dataloader = dataloader

    def __iter__(self):
        for batch in self.dataloader:
            yield _split_xy(batch)


# ═══════════════════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════════════════

def train_experiment(
    condition: str,
    seed: int,
    total_steps: int,
    switch_step: int,
    transition_window: int,
    topk: int,
    indexer_aux_weight: float,
    lambda_rca: float,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    model_kwargs: dict,
    device: torch.device,
    checkpoint_dir: str,
    output_path: str,
    data_dir: str | None = None,
    synthetic_data: bool = False,
    eval_max_batches: int | None = 100,
    log_every: int = 100,
    lr: float = 3e-4,
):
    """Single-process training entry point (no DDP here -- see main()/CLI
    below for the torchrun-compatible DDP wrapper mirroring
    src/sparse_attention_300m.py's pattern). Kept DDP-free so it stays
    directly unit-testable on CPU.

    Writes `output_path` with at least `final_metrics` (G_CL-style metrics
    computed at the end of training) and `at_switch_metrics` (the same
    metrics computed exactly at `switch_step`, i.e. immediately at the
    dense->sparse transition, BEFORE any post-switch training -- this is
    specifically what the review asked to measure: "measure the gap at the
    switch"). Returns the same dict.
    """
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition: {condition!r} (expected one of {CONDITIONS})")

    torch.manual_seed(seed)
    model = DSATransformer(vocab_size, **model_kwargs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    if synthetic_data:
        train_loader = SyntheticLoader(vocab_size, batch_size, seq_len, n_batches=total_steps, seed=seed)
        val_loader = SyntheticLoader(vocab_size, batch_size, seq_len, n_batches=eval_max_batches or 4, seed=seed + 1)
    else:
        if data_dir is None:
            raise ValueError("data_dir is required when synthetic_data=False")
        train_tokens = load_tokens(data_dir, "train")
        val_tokens = load_tokens(data_dir, "val")
        train_ds = CyclingTokenDataset(train_tokens, seq_len)
        val_ds = CyclingTokenDataset(val_tokens, seq_len)
        train_loader = RealLoaderAdapter(DataLoader(train_ds, batch_size=batch_size, shuffle=True))
        val_loader = RealLoaderAdapter(DataLoader(val_ds, batch_size=batch_size, shuffle=False))

    ckpt_dir = Path(checkpoint_dir) / f"sparse_attention_dsa_{condition}_s{seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log = []
    at_switch_metrics = None
    train_iter = iter(train_loader)
    start_time = time.time()

    for step in range(total_steps):
        # "Measure the gap at the switch": evaluate G_CL-style metrics using
        # the model EXACTLY as it stands after steps [0, switch_step-1]
        # (pure dense warmup) and BEFORE step switch_step's own gradient
        # update is applied -- i.e. immediately at the dense->sparse
        # transition, before any post-switch fine-tuning has touched the
        # weights at all. This ordering (eval BEFORE this step's forward/
        # backward/optimizer.step()) is what makes it "at the switch" and
        # not "one step after the switch".
        if step == switch_step:
            at_switch_metrics = run_full_evaluation(model, val_loader, device, topk, eval_max_batches)

        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)
        x, y = x.to(device), y.to(device)

        mode = step_mode(step, switch_step, transition_window, condition)
        optimizer.zero_grad()
        if mode == "dense":
            loss, info = dense_loss(model, x, y, indexer_aux_weight)
        elif mode == "hard":
            loss, info = hard_loss(model, x, y, topk, indexer_aux_weight)
        else:  # mixed
            combined, l_soft, l_hard = clhr_mixed_loss(model, x, y, lambda_rca, topk, indexer_aux_weight)
            loss = combined
            info = {"lm_loss": None, "l_soft": l_soft.item(), "l_hard": l_hard.item()}
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % max(log_every, 1) == 0 or step == total_steps - 1:
            entry = {"step": step, "mode": mode, "loss": loss.item(), **info}
            log.append(entry)
            print(f"step {step:>6}/{total_steps} mode={mode:<6} loss={loss.item():.4f}", flush=True)

    if at_switch_metrics is None:
        # total_steps <= switch_step: switch never reached within this run.
        at_switch_metrics = run_full_evaluation(model, val_loader, device, topk, eval_max_batches)

    final_metrics = run_full_evaluation(model, val_loader, device, topk, eval_max_batches)

    torch.save({"model": model.state_dict(), "step": total_steps}, ckpt_dir / "final.pt")

    result = {
        "condition": condition,
        "seed": seed,
        "total_steps": total_steps,
        "switch_step": switch_step,
        "transition_window": transition_window,
        "topk": topk,
        "lambda_rca": lambda_rca,
        "indexer_aux_weight": indexer_aux_weight,
        "at_switch_metrics": at_switch_metrics,
        "final_metrics": final_metrics,
        "log": log,
        "wall_clock_s": time.time() - start_time,
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    return result


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DSA (lightning indexer + top-k) baseline, standard vs. CLHR conditions"
    )
    parser.add_argument("--condition", choices=CONDITIONS, default="standard")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=str, default=None, help="Directory with wt103_{train,val}_tokens.{pt,npy}")
    parser.add_argument("--checkpoint-dir", type=str, default="./ckpts_sparse_attention_dsa")
    parser.add_argument("--output", type=str, default="results/sparse_attention_dsa.json")
    parser.add_argument("--total-steps", type=int, default=61035, help="Total training steps")
    parser.add_argument(
        "--switch-step", type=int, required=True,
        help="Step at which dense warmup ends and the indexer-gated hard switch happens "
        "(standard: abrupt cutover here; clhr: transition window BEGINS here). "
        "This is the CLI parameter the review's 'measure the gap at the switch' ask targets.",
    )
    parser.add_argument(
        "--transition-window", type=int, default=2000,
        help="CLHR only: number of steps, starting at --switch-step, during which both the "
        "dense and hard forward passes are run and their losses mixed (see clhr_mixed_loss).",
    )
    parser.add_argument("--topk", type=int, default=256, help="Indexer top-k (DeepSeek's own default is 2048; "
                         "see REAL_DSA_INDEX_TOPK)")
    parser.add_argument("--index-n-heads", type=int, default=8)
    parser.add_argument("--index-head-dim", type=int, default=32)
    parser.add_argument("--indexer-aux-weight", type=float, default=0.1,
                         help="Weight on the indexer's distillation loss (necessary: hard top-k "
                         "gives the indexer zero gradient from the LM loss otherwise)")
    parser.add_argument("--lambda-rca", type=float, default=1.0, help="CLHR hard-loss mixing weight")
    parser.add_argument("--d-model", type=int, default=1024)
    parser.add_argument("--n-heads", type=int, default=16)
    parser.add_argument("--n-layers", type=int, default=20)
    parser.add_argument("--d-ff", type=int, default=4096)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--vocab-size", type=int, default=50257)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-max-batches", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--synthetic-data", action="store_true",
                         help="Use random synthetic token batches instead of --data-dir "
                         "(smoke-testing only, never for real runs)")
    return parser


def main():
    args = build_argparser().parse_args()

    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(device)
    else:
        rank = 0
        world_size = 1
        device = select_device()
    is_master = rank == 0

    model_kwargs = dict(
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers, d_ff=args.d_ff,
        index_n_heads=args.index_n_heads, index_head_dim=args.index_head_dim,
        max_seq_len=args.seq_len,
    )

    # NOTE: DDP-wrapping is scaffolded here to match src/sparse_attention_300m.py's
    # convention for the real multi-GPU vast.ai runs, but this path has NOT been
    # smoke-tested (no multi-GPU environment available in this dev sandbox) --
    # it is UNVERIFIED beyond matching the source pattern that was read.
    if ddp:
        torch.manual_seed(args.seed)
        raw_model = DSATransformer(args.vocab_size, **model_kwargs).to(device)
        wrapped = DDP(raw_model, device_ids=[local_rank] if torch.cuda.is_available() else None)
        if is_master:
            print(f"[ddp] world_size={world_size} rank={rank} device={device}", flush=True)
        # train_experiment currently expects to build its own (unwrapped) model;
        # full DDP training-loop integration is intentionally left for the
        # post-submission real-run work (this file is build+smoke-test only per
        # the task's own scope), not exercised by this CLI path today.
        del wrapped

    result = train_experiment(
        condition=args.condition, seed=args.seed, total_steps=args.total_steps,
        switch_step=args.switch_step, transition_window=args.transition_window,
        topk=args.topk, indexer_aux_weight=args.indexer_aux_weight, lambda_rca=args.lambda_rca,
        batch_size=args.batch_size, seq_len=args.seq_len, vocab_size=args.vocab_size,
        model_kwargs=model_kwargs, device=device, checkpoint_dir=args.checkpoint_dir,
        output_path=args.output, data_dir=args.data_dir, synthetic_data=args.synthetic_data,
        eval_max_batches=args.eval_max_batches, log_every=args.log_every, lr=args.lr,
    )
    if is_master:
        print(f"Done. at_switch G_CL={result['at_switch_metrics']['G_CL']:.4f} "
              f"final G_CL={result['final_metrics']['G_CL']:.4f}")

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
