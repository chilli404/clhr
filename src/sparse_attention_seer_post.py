"""SeerAttention-style POST-HOC block-sparse gate distillation, +/- CLHR.

## Why this file exists

External review of the CLHR paper asked for a comparison against SeerAttention /
InfLLM-V2 — methods that learn block-sparsity GATES on top of an already-pretrained
(frozen or lightly fine-tuned) model, i.e. "the regime people actually deploy",
rather than training sparsity from scratch. The review also notes this project
already ran a frozen-Qwen experiment showing NO benefit from CLHR in a related
regime, and asks: does a long-context + high-sparsity block-sparse *post-hoc gate*
setting show a gap CLHR can close, or should the from-scratch-training scope be
documented as a hard boundary?

## What already exists in this repo (do not re-read these to re-derive this, this
## summary was produced by literally reading all of them in full):

1. `src/qwen3_sparse_rca.py`, `src/qwen3_multilayer_sparse_rca.py`,
   `src/qwen3_gate_utility.py` — the actual "frozen-Qwen" experiment. Loads a
   local Qwen3-1.7B-base checkpoint, freezes everything, unfreezes only
   self_attn (1 layer or all 28), uses **token-level top-k=64** sparsity (not
   block-sparse) over **seq_len=512** (short), and tests whether replaying a
   *historical* gate snapshot ("coherent") beats the model's own current/random
   gate. Finding: no benefit (single-layer: coherent gives the *highest*, i.e.
   worst, oracle-excess NLL of all conditions; multi-layer: coherent ties
   contemporary/learned as worst, ~40-400x smaller effect size than the 31M
   from-scratch calibration). Root cause (from
   predictions/qwen3_multilayer_sparse_rca.json): frozen FFN/embeddings at 1.7B
   scale supply representations strong enough that gate history/coherence stops
   mattering. This experiment never trained an add-on gate on a frozen
   backbone with self-distillation, was never block-sparse, and never tested
   context lengths beyond 512 or sparsity beyond ~87.5% (k=64/512).

2. `src/sparse_attention_closed_loop_eval.py` has `posthoc_kl_distillation` /
   `posthoc_bce_distillation` — closer in spirit (frozen backbone + freshly
   trained external gate params), but: (a) TOKEN-level, per-(query,key) pair,
   not block-level; (b) the loss is an ELEMENTWISE Bernoulli KL/BCE between a
   sigmoid-per-edge gate probability and a *hard* 0/1 top-k oracle mask over raw
   QK scores — not a categorical softmax-KL against the dense *distribution*,
   which is what SeerAttention actually does; (c) training is strictly
   open-loop (the oracle and gate-input hidden states both come from a single
   no-grad soft rollout, computed once) — closed-loop hard deployment is used
   ONLY for post-hoc evaluation (`forward_closed_loop_with_fresh_gate`), never
   fed back into the training objective itself.

None of the above is block-sparse, none uses a categorical distillation target,
and none has a closed-loop *training-time* term. This file builds all three,
genuinely extending (rather than duplicating) the existing infra.

## Mechanism (SeerAttention, verified against microsoft/SeerAttention source:
## README.md, seer_attn/modules/attention_distill.py,
## seer_attn/prefill_sparse/llama/modeling_llama_seerattn.py):
  - A lightweight "AttnGate" pools Q (mean) and K (max+min concat) within each
    block ("Qavg_Kmaxmin", their documented default gate type), projects
    through small learnable linear layers, and produces block-level importance
    logits via a scaled dot product. Block size is fixed at 64 in their
    released kernels; we expose it as a free parameter (`--block-size`) since
    the review specifically wants a (context_length, sparsity) sweep.
  - Ground truth = a block-max-pooled version of the frozen model's own dense
    attention distribution (they compute this via a custom Triton kernel;
    we do it in plain PyTorch — see `pool_dense_attention_to_blocks`).
  - Loss: `KLDivLoss(log_softmax(gate_logits), ground_truth)`. Their published
    code instantiates `torch.nn.KLDivLoss()` with the PyTorch default
    `reduction="mean"`, which averages over every element of a
    (batch, heads, n_qblocks, n_kblocks) tensor rather than normalizing per
    query-block distribution. We deliberately deviate: `SelfDistillationLoss`
    sums each row's KL divergence properly, then averages over rows, which is
    the mathematically correct per-distribution KL. This is a documented,
    deliberate deviation for correctness, not a replication of their exact
    code.
  - Base model is frozen during gate training (confirmed in their code via
    `requires_grad=False` except for gate params).

## Backbone reuse decision

Rather than rebuilding a frozen-model harness from scratch, or requiring the
real Qwen3-1.7B weights (forbidden pre-submission: no large downloads, no real
GPU training), this file reuses this project's OWN `GatedTransformer` /
`ModelConfig` / `GatedSparseAttention` (../routing-absorption/src/models/) —
the exact same infra `posthoc_kl_distillation` already depends on —
instantiated with `sparsity_mode="dense"` so the "backbone" is a plain causal
Transformer with NO gate of its own, standing in for a frozen pretrained model.
`build_frozen_dense_backbone` freezes every parameter, matching the identical
`for p in model.parameters(): p.requires_grad = False` idiom used by
`qwen3_gate_utility.py`/`qwen3_sparse_rca.py` and by `posthoc_kl_distillation`.

For the REAL post-submission experiment: swap `build_frozen_dense_backbone`'s
output for an actual loaded+frozen Qwen3 checkpoint (identical freezing idiom,
see `src/qwen3_gate_utility.py` lines ~73-100), and adapt `_layer_qk` to read
`layer.self_attn.q_proj`/`k_proj`/`v_proj` (HF naming) instead of
`layer.attention.W_q`/`W_k`/`W_v` (this project's naming). Everything else
(`BlockImportanceGate`, `SelfDistillationLoss`, the STE hard-path mixing, the
CLI) is architecture-agnostic and untouched by that swap.

## `standard` vs `clhr`

- `standard`: gate trained with the self-distillation loss only, against a
  static teacher trajectory from one no-grad dense rollout. This is the
  open-loop regime — structurally the same "train against a frozen oracle,
  never see your own hard deployment during training" pattern as
  `posthoc_kl_distillation` and, at a coarser grain, the historical-vs-learned
  gate framing of the existing frozen-Qwen experiment. This is the "existing
  regime that showed no benefit", now generalized to block-sparse.
- `clhr`: ADDITIONALLY cascades a hard-deployed rollout through the frozen
  backbone, where at each layer the gate is recomputed from the CURRENT
  hard-path hidden state (not the teacher's), block selection is hardened via
  a straight-through estimator (`block_gate_ste_indicator` — forward value
  identical to true deployment, backward gradient flows through
  `sigmoid(gate_logits)`), and the resulting end-to-end LM loss is added to the
  distillation loss (`distill_loss + lambda_rca * closed_loop_lm_loss`,
  matching `src/sparse_attention_300m.py`'s additive dual-loss convention).
  This is the CLHR-style closed-loop mixing applied to the gate-distillation
  phase itself — testing whether closing the loop helps THIS regime, per the
  review's request.

Known limitation (documented, not hidden): because the backbone is frozen, the
gate receives gradient only through the STE surrogate on ITS OWN selection
decision, not through backbone-side attention-weight recalibration (which
would require unfreezing QKV, defeating the point of "post-hoc"/"frozen"). This
is weaker gradient signal than the from-scratch CLHR setting, where gate and
value/query/key projections adapt jointly. That asymmetry is itself part of
what this experiment is testing.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
RA_ROOT = Path(__file__).parent.parent.parent / "routing-absorption"
sys.path.insert(0, str(RA_ROOT))

from src.models.config import ModelConfig  # noqa: E402
from src.models.transformer import GatedTransformer  # noqa: E402

from sparse_attention_closed_loop_eval import compute_gate_oracle_agreement  # noqa: E402


# --------------------------------------------------------------------------- #
# Block-level primitives
# --------------------------------------------------------------------------- #


def n_blocks_for(seq_len: int, block_size: int) -> int:
    return math.ceil(seq_len / block_size)


def _pad_seq_to_block_multiple(x: torch.Tensor, block_size: int) -> torch.Tensor:
    """Zero-pad the sequence dim (second-to-last axis) up to a multiple of block_size.

    Uses F.pad's from-the-last-dim convention explicitly: (0, 0) for the last
    axis (no-op) then (0, extra) for the second-to-last axis, so this never
    accidentally pads d_head instead of the sequence length.
    """
    T = x.shape[-2]
    n_b = n_blocks_for(T, block_size)
    extra = n_b * block_size - T
    if extra == 0:
        return x
    return F.pad(x, (0, 0, 0, extra))


class BlockImportanceGate(nn.Module):
    """SeerAttention-style block-importance gate ("Qavg_Kmaxmin"): mean-pool Q
    and max+min-pool K within each block, project through small learnable
    linear heads, and score block compatibility via a scaled dot product.
    """

    def __init__(self, d_head: int, block_size: int, gate_dim: int | None = None):
        super().__init__()
        self.d_head = d_head
        self.block_size = block_size
        self.gate_dim = gate_dim or d_head
        self.q_proj = nn.Linear(d_head, self.gate_dim, bias=False)
        self.k_proj = nn.Linear(2 * d_head, self.gate_dim, bias=False)
        nn.init.normal_(self.q_proj.weight, std=0.02)
        nn.init.normal_(self.k_proj.weight, std=0.02)

    def _pool_q_mean(self, q: torch.Tensor) -> torch.Tensor:
        q = _pad_seq_to_block_multiple(q, self.block_size)
        *lead, T, d = q.shape
        n_b = T // self.block_size
        q = q.reshape(*lead, n_b, self.block_size, d)
        return q.mean(dim=-2)

    def _pool_k_maxmin(self, k: torch.Tensor) -> torch.Tensor:
        k = _pad_seq_to_block_multiple(k, self.block_size)
        *lead, T, d = k.shape
        n_b = T // self.block_size
        k = k.reshape(*lead, n_b, self.block_size, d)
        return torch.cat([k.amax(dim=-2), k.amin(dim=-2)], dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """q, k: (B, H, T, d_head) -> block logits (B, H, n_qblocks, n_kblocks)."""
        qb = self.q_proj(self._pool_q_mean(q))
        kb = self.k_proj(self._pool_k_maxmin(k))
        return torch.matmul(qb, kb.transpose(-2, -1)) / (self.gate_dim**0.5)


def pool_dense_attention_to_blocks(attn_probs: torch.Tensor, block_size: int) -> torch.Tensor:
    """Ground-truth block importance distribution from a frozen model's dense
    softmax attention: max-pool within each key-block, mean-pool within each
    query-block, renormalize each query-block row to sum to 1.

    attn_probs: (B, H, T, T), softmax probabilities (rows sum to 1 over valid
    causal positions). Returns (B, H, n_qblocks, n_kblocks).
    """
    B, H, T, _ = attn_probs.shape
    n_b = n_blocks_for(T, block_size)
    pad = n_b * block_size - T
    if pad:
        attn_probs = F.pad(attn_probs, (0, pad, 0, pad))
    x = attn_probs.reshape(B, H, n_b, block_size, n_b, block_size)
    pooled = x.amax(dim=-1)  # max over key-within-block -> (B,H,n_b,block_size,n_b)
    pooled = pooled.mean(dim=3)  # mean over query-within-block -> (B,H,n_b,n_b)
    row_sum = pooled.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return pooled / row_sum


def block_causal_mask(seq_len: int, block_size: int, device: torch.device) -> torch.Tensor:
    n_b = n_blocks_for(seq_len, block_size)
    idx = torch.arange(n_b, device=device)
    return (idx.unsqueeze(1) >= idx.unsqueeze(0)).float()


def harden_block_mask(
    gate_logits: torch.Tensor, topk_blocks: int, causal_block_mask: torch.Tensor
) -> torch.Tensor:
    """Discrete top-k key-block selection per query-block, respecting causality."""
    illegal = ~causal_block_mask.bool()
    masked = gate_logits.masked_fill(illegal, float("-inf"))
    n_kb = masked.shape[-1]
    k = min(topk_blocks, n_kb)
    _, idx = torch.topk(masked, k, dim=-1)
    hard = torch.zeros_like(masked)
    hard.scatter_(-1, idx, 1.0)
    return hard * causal_block_mask


def block_gate_ste_indicator(
    gate_logits: torch.Tensor, causal_block_mask: torch.Tensor, topk_blocks: int
) -> torch.Tensor:
    """Straight-through estimator for hard block selection: forward value is
    IDENTICAL to the true discrete top-k mask used at deployment; backward
    gradient flows through sigmoid(gate_logits) as a differentiable surrogate.
    """
    illegal = ~causal_block_mask.bool()
    soft = torch.sigmoid(gate_logits.masked_fill(illegal, float("-inf")))
    soft = torch.nan_to_num(soft, nan=0.0)
    with torch.no_grad():
        hard = harden_block_mask(gate_logits, topk_blocks, causal_block_mask)
    return hard + soft - soft.detach()


def expand_block_mask_to_tokens(block_mask: torch.Tensor, block_size: int, seq_len: int) -> torch.Tensor:
    m = block_mask.repeat_interleave(block_size, dim=-2).repeat_interleave(block_size, dim=-1)
    return m[..., :seq_len, :seq_len]


def topk_blocks_from_sparsity(n_kb_total: int, target_sparsity: float) -> int:
    keep_frac = max(0.0, 1.0 - target_sparsity)
    return max(1, round(n_kb_total * keep_frac))


class SelfDistillationLoss(nn.Module):
    """Categorical KL divergence between the gate's predicted block
    distribution and the ground-truth block-pooled dense attention
    distribution, averaged per query-block row (see module docstring for why
    this differs from SeerAttention's literal `reduction="mean"` code).
    """

    def forward(
        self,
        gate_logits: torch.Tensor,
        ground_truth_block_probs: torch.Tensor,
        causal_block_mask: torch.Tensor,
    ) -> torch.Tensor:
        illegal = ~causal_block_mask.bool()
        masked_logits = gate_logits.masked_fill(illegal, float("-inf"))
        log_pred = F.log_softmax(masked_logits, dim=-1)
        log_pred = torch.nan_to_num(log_pred, nan=0.0, neginf=0.0)
        kl_per_row = F.kl_div(log_pred, ground_truth_block_probs, reduction="none").sum(dim=-1)
        return kl_per_row.mean()


# --------------------------------------------------------------------------- #
# Frozen backbone (this project's own tiny Transformer, standing in for a
# frozen pretrained model for smoke-test purposes — see module docstring).
# --------------------------------------------------------------------------- #


def make_induction_batch(
    batch_size: int, seq_len: int, vocab_size: int, device: torch.device, period: int = 4
) -> torch.Tensor:
    """Synthetic periodic-copy task: a random length-`period` pattern tiled to
    fill the sequence. Predicting token t requires attending back exactly
    `period` positions -- a small, fixed-offset dependency.

    A randomly-initialized backbone produces essentially UNIFORM dense
    attention (verified empirically: max/min attention weight in a fresh tiny
    model differed by <1% at a mid-sequence position) -- a real frozen
    pretrained model would have sharply peaked attention instead. Since the
    whole point of this smoke test is to check that a gate can distill a
    NON-TRIVIAL dense attention pattern, we briefly train the backbone on
    this task first (this is what "pretrained" stands in for at smoke-test
    scale).

    An earlier version of this task used a half-sequence repeat ("induction
    head"-style: [random_half, random_half]), which requires attending back
    seq_len/2 positions. Empirically (verified: 800 steps, lr=1e-2, vocab=32,
    d_model=32) that task's loss stayed exactly pinned at the uniform-random
    baseline ln(vocab_size) -- no learning at all at this tiny scale. This
    small-period task with a much shorter fixed offset converges reliably
    (verified: ln(32)=3.47 -> 0.17 over 500 steps under the same tiny config),
    so it is used instead.
    """
    n_rep = seq_len // period + 1
    base = torch.randint(0, vocab_size, (batch_size, period), device=device)
    seq = base.repeat(1, n_rep)
    return seq[:, :seq_len]


def pretrain_then_freeze_backbone(
    backbone: GatedTransformer,
    steps: int,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
    lr: float = 1e-2,
    period: int = 4,
) -> list[float]:
    """Briefly train `backbone` (all params) on the induction task, then
    freeze it. Returns the LM loss curve so callers can confirm the backbone
    actually learned a non-trivial attention pattern before treating it as
    "frozen pretrained".
    """
    backbone.train()
    for p in backbone.parameters():
        p.requires_grad = True
    optimizer = torch.optim.AdamW(backbone.parameters(), lr=lr)
    losses = []
    for _ in range(steps):
        batch = make_induction_batch(batch_size, seq_len, vocab_size, device, period=period)
        logits = backbone(batch)
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)), batch[:, 1:].reshape(-1)
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    for p in backbone.parameters():
        p.requires_grad = False
    backbone.eval()
    return losses


def build_frozen_dense_backbone(
    vocab_size: int,
    d_model: int,
    n_heads: int,
    n_layers: int,
    d_ff: int,
    max_seq_len: int,
    dropout: float = 0.0,
) -> GatedTransformer:
    config = ModelConfig(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        n_layers=n_layers,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        dropout=dropout,
        sparsity_mode="dense",
    )
    backbone = GatedTransformer(config)
    for p in backbone.parameters():
        p.requires_grad = False
    backbone.eval()
    return backbone


class SeerPostHocGateModel(nn.Module):
    """Frozen backbone + one trainable BlockImportanceGate per layer."""

    def __init__(self, backbone: GatedTransformer, block_size: int, gate_dim: int | None = None):
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.block_size = block_size
        d_head = backbone.config.d_model // backbone.config.n_heads
        self.gates = nn.ModuleList(
            [BlockImportanceGate(d_head, block_size, gate_dim) for _ in backbone.layers]
        )

    @staticmethod
    def _layer_qkv(attn_module, normed_x, batch_size, seq_len):
        n_h = attn_module.config.n_heads
        d_head = attn_module.d_head
        q = attn_module.W_q(normed_x).view(batch_size, seq_len, n_h, d_head).transpose(1, 2)
        k = attn_module.W_k(normed_x).view(batch_size, seq_len, n_h, d_head).transpose(1, 2)
        v = attn_module.W_v(normed_x).view(batch_size, seq_len, n_h, d_head).transpose(1, 2)
        return q, k, v

    def teacher_forward(self, input_ids: torch.Tensor):
        """One no-grad rollout through the frozen dense backbone. Returns a
        per-layer record (q, k, dense attn_probs) used as the static
        self-distillation oracle, plus the backbone's own logits.
        """
        with torch.no_grad():
            b, seq_len = input_ids.shape
            device = input_ids.device
            positions = torch.arange(seq_len, device=device).unsqueeze(0)
            x = self.backbone.embedding(input_ids) + self.backbone.pos_embedding(positions)
            causal = torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0).unsqueeze(0)
            records = []
            for layer in self.backbone.layers:
                normed = layer.attn_norm(x)
                attn = layer.attention
                q, k, _ = self._layer_qkv(attn, normed, b, seq_len)
                out, attn_probs = attn(normed, causal, return_attention=True)
                records.append({"q": q, "k": k, "attn_probs": attn_probs})
                x = x + out
                x = x + layer.ff(layer.ff_norm(x))
            logits = self.backbone.lm_head(self.backbone.final_norm(x))
        return records, logits

    def distillation_loss(self, records) -> torch.Tensor:
        losses = []
        for rec, gate in zip(records, self.gates):
            seq_len = rec["q"].shape[-2]
            causal_block = block_causal_mask(seq_len, self.block_size, rec["q"].device)
            gate_logits = gate(rec["q"], rec["k"])
            ground_truth = pool_dense_attention_to_blocks(rec["attn_probs"], self.block_size)
            losses.append(SelfDistillationLoss()(gate_logits, ground_truth, causal_block))
        return torch.stack(losses).mean()

    def closed_loop_forward(self, input_ids: torch.Tensor, topk_blocks: int):
        """Hard-deployed rollout: at each layer, the gate is recomputed from
        the CURRENT hard-path hidden state (closing the loop between training
        and deployment), hardened via STE, and cascaded through the frozen
        backbone end-to-end.
        """
        b, seq_len = input_ids.shape
        device = input_ids.device
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        x = self.backbone.embedding(input_ids) + self.backbone.pos_embedding(positions)
        causal_tok = torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0).unsqueeze(0)
        causal_block = block_causal_mask(seq_len, self.block_size, device)
        gate_logits_by_layer = []
        for layer, gate in zip(self.backbone.layers, self.gates):
            normed = layer.attn_norm(x)
            attn = layer.attention
            q, k, v = self._layer_qkv(attn, normed, b, seq_len)
            gate_logits = gate(q, k)
            gate_logits_by_layer.append(gate_logits)
            ste = block_gate_ste_indicator(gate_logits, causal_block, topk_blocks)

            raw_scores = torch.matmul(q, k.transpose(-2, -1)) / (attn.d_head**0.5)
            scores = raw_scores.masked_fill(causal_tok == 0, float("-inf"))
            probs = torch.nan_to_num(F.softmax(scores, dim=-1), nan=0.0)
            gate_tokens = expand_block_mask_to_tokens(ste, self.block_size, seq_len)
            probs = probs * gate_tokens
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)

            out = torch.matmul(probs, v)
            out = out.transpose(1, 2).contiguous().view(b, seq_len, -1)
            out = attn.W_o(out)

            x = x + out
            x = x + layer.ff(layer.ff_norm(x))
        logits = self.backbone.lm_head(self.backbone.final_norm(x))
        return logits, gate_logits_by_layer

    def deploy_hard(self, input_ids: torch.Tensor, topk_blocks: int):
        """True hard deployment: gate top-k selection, no STE, no bias — the
        actual inference-time behavior of a trained gate. Used for evaluation
        only (matches this project's closed-loop *evaluation* convention).
        """
        with torch.no_grad():
            logits, gate_logits_by_layer = self.closed_loop_forward(input_ids, topk_blocks)
        return logits, gate_logits_by_layer

    def compute_loss(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor,
        mode: str,
        topk_blocks: int,
        lambda_rca: float = 1.0,
    ):
        if mode not in ("standard", "clhr"):
            raise ValueError(f"Unknown mode: {mode!r} (expected 'standard' or 'clhr')")

        records, _ = self.teacher_forward(input_ids)
        distill_loss = self.distillation_loss(records)

        if mode == "standard":
            return distill_loss, {"distill_loss": distill_loss.item(), "closed_loop_lm_loss": None}

        logits, _ = self.closed_loop_forward(input_ids, topk_blocks)
        closed_loop_lm_loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)), targets[:, 1:].reshape(-1)
        )
        total = distill_loss + lambda_rca * closed_loop_lm_loss
        return total, {
            "distill_loss": distill_loss.item(),
            "closed_loop_lm_loss": closed_loop_lm_loss.item(),
        }

    @torch.no_grad()
    def block_oracle_agreement(self, input_ids: torch.Tensor, topk_blocks: int) -> float:
        """Sanity metric: F1 overlap between the trained gate's hard top-k
        blocks and the true dense attention's own top-k blocks (oracle),
        reusing this project's existing `compute_gate_oracle_agreement`.
        """
        records, _ = self.teacher_forward(input_ids)
        f1s = []
        for rec, gate in zip(records, self.gates):
            seq_len = rec["q"].shape[-2]
            causal_block = block_causal_mask(seq_len, self.block_size, rec["q"].device)
            gate_logits = gate(rec["q"], rec["k"])
            gate_hard = harden_block_mask(gate_logits, topk_blocks, causal_block)
            ground_truth = pool_dense_attention_to_blocks(rec["attn_probs"], self.block_size)
            oracle_hard = harden_block_mask(ground_truth, topk_blocks, causal_block)
            f1s.append(compute_gate_oracle_agreement(gate_hard, oracle_hard))
        return sum(f1s) / len(f1s)


# --------------------------------------------------------------------------- #
# REAL HF Qwen3 adapter -- the actual post-submission target.
#
# `SeerPostHocGateModel` above stands in for "a frozen pretrained model" with
# this project's own tiny vanilla-attention Transformer, entirely to respect
# the pre-submission "no large downloads, no real GPU training" constraint.
# This class is the real thing: it wraps an actual `transformers` Qwen3
# architecture (RoPE, per-head QK-RMSNorm, grouped-query attention) and
# freezes it exactly as `src/qwen3_gate_utility.py` does. It has been
# smoke-tested ONLY against a TINY, randomly-initialized `Qwen3Config` (no
# weight download -- see tests/test_sparse_attention_seer_post.py::
# TestQwen3SeerPostHocGateModel), which verifies the RoPE/QK-norm/GQA-repeat
# plumbing is correct against the real library code, but says nothing about
# behavior with real pretrained weights or at real long-context/high-sparsity
# scale. To run the real experiment: load the actual frozen checkpoint (same
# idiom as qwen3_gate_utility.py: `AutoModelForCausalLM.from_pretrained(
# "/s3-data/models/qwen3-1.7b-base", torch_dtype=torch.bfloat16,
# attn_implementation="eager", local_files_only=True)`), pass it to this
# class, and use the CLI's `--model-source qwen3` path (see main()).
#
# GQA note: Q has num_attention_heads heads, K/V have num_key_value_heads
# (fewer) heads, repeated via `repeat_kv` to align with Q before attention.
# We reuse the library's own `apply_rotary_pos_emb`/`repeat_kv` so RoPE and
# GQA are exactly as faithful as the real model's own forward pass, rather
# than reimplementing rotary math by hand. `BlockImportanceGate` and all
# block-level utilities need no changes at all: they only ever see
# (B, n_heads, T, head_dim) tensors, and GQA-repeated K looks identical in
# shape to a native multi-head K.
# --------------------------------------------------------------------------- #


class Qwen3SeerPostHocGateModel(nn.Module):
    """Frozen HF Qwen3 backbone + one trainable BlockImportanceGate per layer."""

    def __init__(self, backbone, block_size: int, gate_dim: int | None = None):
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()
        self.block_size = block_size
        config = backbone.config
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.gates = nn.ModuleList(
            [BlockImportanceGate(head_dim, block_size, gate_dim) for _ in backbone.model.layers]
        )

    @staticmethod
    def _layer_qkv_rope(self_attn, hidden_states, position_embeddings):
        """Q/K/V exactly as Qwen3Attention.forward computes them (q_proj/k_proj/
        v_proj -> per-head QK-RMSNorm -> RoPE -> GQA repeat_kv), stopping short
        of the attention/softmax step so callers can substitute their own.
        """
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self_attn.head_dim)
        q = self_attn.q_norm(self_attn.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        k = self_attn.k_norm(self_attn.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        v = self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        k = repeat_kv(k, self_attn.num_key_value_groups)
        v = repeat_kv(v, self_attn.num_key_value_groups)
        return q, k, v

    def _position_embeddings(self, hidden_states, seq_len, device):
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        return self.backbone.model.rotary_emb(hidden_states, position_ids)

    def teacher_forward(self, input_ids: torch.Tensor):
        """No-grad rollout through the real frozen Qwen3 model, using its own
        (real) self_attn.forward for the dense causal attention (RoPE,
        QK-norm, GQA, and softmax all handled by the library itself -- this
        is the actual dense pattern a real frozen checkpoint would produce),
        while separately recomputing q, k via `_layer_qkv_rope` for the gate.
        """
        with torch.no_grad():
            b, seq_len = input_ids.shape
            device = input_ids.device
            x = self.backbone.model.embed_tokens(input_ids)
            position_embeddings = self._position_embeddings(x, seq_len, device)
            causal_additive = torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1
            )
            causal_additive = causal_additive.unsqueeze(0).unsqueeze(0)
            records = []
            for layer in self.backbone.model.layers:
                normed = layer.input_layernorm(x)
                q, k, _ = self._layer_qkv_rope(layer.self_attn, normed, position_embeddings)
                attn_out, attn_probs = layer.self_attn(
                    hidden_states=normed,
                    attention_mask=causal_additive,
                    position_embeddings=position_embeddings,
                )
                records.append({"q": q, "k": k, "attn_probs": attn_probs})
                x = x + attn_out
                x = x + layer.mlp(layer.post_attention_layernorm(x))
            x = self.backbone.model.norm(x)
            logits = self.backbone.lm_head(x)
        return records, logits

    def distillation_loss(self, records) -> torch.Tensor:
        losses = []
        for rec, gate in zip(records, self.gates):
            seq_len = rec["q"].shape[-2]
            causal_block = block_causal_mask(seq_len, self.block_size, rec["q"].device)
            gate_logits = gate(rec["q"], rec["k"])
            ground_truth = pool_dense_attention_to_blocks(rec["attn_probs"], self.block_size)
            losses.append(SelfDistillationLoss()(gate_logits, ground_truth, causal_block))
        return torch.stack(losses).mean()

    def closed_loop_forward(self, input_ids: torch.Tensor, topk_blocks: int):
        """Hard-deployed rollout: at each layer, the gate is recomputed from
        the CURRENT hard-path hidden state, hardened via STE, and the
        block-sparse attention output is computed manually from q, k, v
        (bypassing self_attn's own softmax) -- everything else (input/post
        layernorms, mlp, o_proj) reuses the real frozen submodules directly.
        """
        b, seq_len = input_ids.shape
        device = input_ids.device
        x = self.backbone.model.embed_tokens(input_ids)
        position_embeddings = self._position_embeddings(x, seq_len, device)
        causal_tok = torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0).unsqueeze(0)
        causal_block = block_causal_mask(seq_len, self.block_size, device)
        gate_logits_by_layer = []
        for layer, gate in zip(self.backbone.model.layers, self.gates):
            self_attn = layer.self_attn
            normed = layer.input_layernorm(x)
            q, k, v = self._layer_qkv_rope(self_attn, normed, position_embeddings)
            gate_logits = gate(q, k)
            gate_logits_by_layer.append(gate_logits)
            ste = block_gate_ste_indicator(gate_logits, causal_block, topk_blocks)

            raw_scores = torch.matmul(q, k.transpose(-2, -1)) * self_attn.scaling
            scores = raw_scores.masked_fill(causal_tok == 0, float("-inf"))
            probs = torch.nan_to_num(F.softmax(scores, dim=-1), nan=0.0)
            gate_tokens = expand_block_mask_to_tokens(ste, self.block_size, seq_len)
            probs = probs * gate_tokens
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)

            out = torch.matmul(probs, v)
            out = out.transpose(1, 2).contiguous().reshape(b, seq_len, -1)
            out = self_attn.o_proj(out)

            x = x + out
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        x = self.backbone.model.norm(x)
        logits = self.backbone.lm_head(x)
        return logits, gate_logits_by_layer

    def deploy_hard(self, input_ids: torch.Tensor, topk_blocks: int):
        with torch.no_grad():
            logits, gate_logits_by_layer = self.closed_loop_forward(input_ids, topk_blocks)
        return logits, gate_logits_by_layer

    def compute_loss(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor,
        mode: str,
        topk_blocks: int,
        lambda_rca: float = 1.0,
    ):
        if mode not in ("standard", "clhr"):
            raise ValueError(f"Unknown mode: {mode!r} (expected 'standard' or 'clhr')")

        records, _ = self.teacher_forward(input_ids)
        distill_loss = self.distillation_loss(records)

        if mode == "standard":
            return distill_loss, {"distill_loss": distill_loss.item(), "closed_loop_lm_loss": None}

        logits, _ = self.closed_loop_forward(input_ids, topk_blocks)
        closed_loop_lm_loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)), targets[:, 1:].reshape(-1)
        )
        total = distill_loss + lambda_rca * closed_loop_lm_loss
        return total, {
            "distill_loss": distill_loss.item(),
            "closed_loop_lm_loss": closed_loop_lm_loss.item(),
        }

    @torch.no_grad()
    def block_oracle_agreement(self, input_ids: torch.Tensor, topk_blocks: int) -> float:
        records, _ = self.teacher_forward(input_ids)
        f1s = []
        for rec, gate in zip(records, self.gates):
            seq_len = rec["q"].shape[-2]
            causal_block = block_causal_mask(seq_len, self.block_size, rec["q"].device)
            gate_logits = gate(rec["q"], rec["k"])
            gate_hard = harden_block_mask(gate_logits, topk_blocks, causal_block)
            ground_truth = pool_dense_attention_to_blocks(rec["attn_probs"], self.block_size)
            oracle_hard = harden_block_mask(ground_truth, topk_blocks, causal_block)
            f1s.append(compute_gate_oracle_agreement(gate_hard, oracle_hard))
        return sum(f1s) / len(f1s)


# --------------------------------------------------------------------------- #
# CLI: smoke test / training entry point
# --------------------------------------------------------------------------- #


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser(
        description="SeerAttention-style post-hoc block-sparse gate distillation (+/- CLHR)"
    )
    parser.add_argument("--mode", type=str, choices=["standard", "clhr"], default="standard")
    parser.add_argument("--seq-len", type=int, default=256, help="context length")
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--target-sparsity", type=float, default=0.5)
    parser.add_argument("--lambda-rca", type=float, default=1.0)
    parser.add_argument("--gate-lr", type=float, default=1e-3)
    parser.add_argument("--gate-dim", type=int, default=None)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--d-ff", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "mps", "cuda"])
    parser.add_argument(
        "--induction-period",
        type=int,
        default=4,
        help="Period of the synthetic periodic-copy task used both to pretrain "
        "the frozen backbone and to drive the gate-training steps.",
    )
    parser.add_argument(
        "--backbone-pretrain-steps",
        type=int,
        default=500,
        help=(
            "Steps to train the backbone on a synthetic induction task before "
            "freezing it, so the frozen model has a non-trivial (non-uniform) "
            "dense attention pattern to distill from -- a fresh random-init "
            "backbone produces near-uniform attention (empirically verified), "
            "which is not a meaningful distillation target. Ignored when "
            "--model-source qwen3 (a real checkpoint is already pretrained)."
        ),
    )
    parser.add_argument(
        "--model-source",
        type=str,
        choices=["synthetic", "qwen3"],
        default="synthetic",
        help=(
            "'synthetic' (default): this project's own tiny Transformer, "
            "briefly trained on a synthetic task then frozen, entirely "
            "local/no-download -- what every smoke test in this repo uses. "
            "'qwen3': load and freeze a REAL frozen Qwen3 causal LM (e.g. "
            "the checkpoint used by src/qwen3_gate_utility.py) via "
            "Qwen3SeerPostHocGateModel. WARNING: the --model-source qwen3 "
            "CLI path itself has NOT been executed end-to-end (no real "
            "checkpoint/GPU available pre-submission) -- only the underlying "
            "Qwen3SeerPostHocGateModel class has been verified, against a "
            "tiny randomly-initialized Qwen3Config (see "
            "tests/test_sparse_attention_seer_post.py::TestQwen3SeerPostHocGateModel). "
            "Run a short --steps 5 dry run first after wiring up a real "
            "checkpoint, before committing to a full budget."
        ),
    )
    parser.add_argument(
        "--qwen-model-path",
        type=str,
        default="/s3-data/models/qwen3-1.7b-base",
        help="Local HF checkpoint path, loaded exactly as in "
        "src/qwen3_gate_utility.py (local_files_only=True, eager attention).",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="If set (only meaningful with --model-source qwen3), load real "
        "text via data_loading.load_corpus(--corpus-name, --data-dir, "
        "seq_len=--seq-len) instead of the synthetic periodic-copy task.",
    )
    parser.add_argument("--corpus-name", type=str, default="wikitext-103")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device) if args.device else _pick_device()

    pretrain_losses = []
    if args.model_source == "qwen3":
        from transformers import AutoModelForCausalLM

        qwen = AutoModelForCausalLM.from_pretrained(
            args.qwen_model_path,
            torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
            attn_implementation="eager",
            local_files_only=True,
        ).to(device)
        model = Qwen3SeerPostHocGateModel(qwen, block_size=args.block_size, gate_dim=args.gate_dim).to(
            device
        )
        args.vocab_size = qwen.config.vocab_size
    else:
        backbone = build_frozen_dense_backbone(
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            d_ff=args.d_ff,
            max_seq_len=args.seq_len,
        ).to(device)
        pretrain_losses = pretrain_then_freeze_backbone(
            backbone,
            steps=args.backbone_pretrain_steps,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            vocab_size=args.vocab_size,
            device=device,
            period=args.induction_period,
        )
        if pretrain_losses:
            print(
                f"backbone pretrain: loss {pretrain_losses[0]:.4f} -> {pretrain_losses[-1]:.4f} "
                f"over {len(pretrain_losses)} steps"
            )
        model = SeerPostHocGateModel(backbone, block_size=args.block_size, gate_dim=args.gate_dim).to(device)

    n_kb_total = n_blocks_for(args.seq_len, args.block_size)
    topk_blocks = topk_blocks_from_sparsity(n_kb_total, args.target_sparsity)

    real_data_iter = None
    if args.data_dir:
        from torch.utils.data import DataLoader

        from data_loading import load_corpus

        dataset = load_corpus(args.corpus_name, args.data_dir, seq_len=args.seq_len)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

        def _cycle(loader):
            while True:
                for batch in loader:
                    yield batch["input_ids"][:, : args.seq_len].to(device)

        real_data_iter = _cycle(loader)

    def next_batch():
        if real_data_iter is not None:
            seq = next(real_data_iter)
            return seq, seq
        seq = make_induction_batch(
            args.batch_size, args.seq_len, args.vocab_size, device, period=args.induction_period
        )
        return seq, seq

    optimizer = torch.optim.AdamW(model.gates.parameters(), lr=args.gate_lr)

    history = []
    for step in range(args.steps):
        # Gate-training data is drawn from the SAME task distribution the
        # backbone was "pretrained" on (matches SeerAttention's real
        # methodology of using in-distribution calibration data). targets is
        # the same sequence as input_ids -- compute_loss shifts internally
        # (logits[:, :-1] vs targets[:, 1:]) for standard next-token LM loss.
        input_ids, targets = next_batch()

        optimizer.zero_grad()
        loss, info = model.compute_loss(
            input_ids, targets, mode=args.mode, topk_blocks=topk_blocks, lambda_rca=args.lambda_rca
        )
        loss.backward()
        optimizer.step()

        info["step"] = step
        info["total_loss"] = loss.item()
        history.append(info)
        if step % max(1, args.steps // 10) == 0 or step == args.steps - 1:
            print(
                f"step {step:4d} | total_loss {info['total_loss']:.4f} "
                f"| distill_loss {info['distill_loss']:.4f} "
                f"| closed_loop_lm_loss {info['closed_loop_lm_loss']}"
            )

    eval_input, eval_targets = next_batch()
    agreement_f1 = model.block_oracle_agreement(eval_input, topk_blocks)
    with torch.no_grad():
        _, teacher_logits = model.teacher_forward(eval_input)
        teacher_nll = F.cross_entropy(
            teacher_logits[:, :-1].reshape(-1, teacher_logits.size(-1)), eval_targets[:, 1:].reshape(-1)
        ).item()
        hard_logits, _ = model.deploy_hard(eval_input, topk_blocks)
        hard_deploy_nll = F.cross_entropy(
            hard_logits[:, :-1].reshape(-1, hard_logits.size(-1)), eval_targets[:, 1:].reshape(-1)
        ).item()
    g_cl = hard_deploy_nll - teacher_nll
    print(
        f"final gate/oracle block-agreement F1: {agreement_f1:.4f} | "
        f"teacher_nll {teacher_nll:.4f} | hard_deploy_nll {hard_deploy_nll:.4f} | G_CL {g_cl:.4f}"
    )

    results = {
        "mode": args.mode,
        "model_source": args.model_source,
        "seq_len": args.seq_len,
        "block_size": args.block_size,
        "target_sparsity": args.target_sparsity,
        "topk_blocks": topk_blocks,
        "n_kb_total": n_kb_total,
        "lambda_rca": args.lambda_rca,
        "seed": args.seed,
        "steps": args.steps,
        "history": history,
        "final_block_oracle_agreement_f1": agreement_f1,
        "final_teacher_nll": teacher_nll,
        "final_hard_deploy_nll": hard_deploy_nll,
        "final_G_CL": g_cl,
        "backbone_pretrain_loss_first_last": (
            [pretrain_losses[0], pretrain_losses[-1]] if pretrain_losses else None
        ),
    }
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
