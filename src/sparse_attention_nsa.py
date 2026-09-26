"""src/sparse_attention_nsa.py

Faithful (but deliberately simplified for a fast, correctness-first CPU port)
reimplementation of DeepSeek's Native Sparse Attention mechanism (NSA; Yuan
et al. 2025, "Native Sparse Attention") ported into this project's
self-contained Transformer/CLHR scaffold (matching
src/sparse_attention_fineweb.py's conventions: GatedSparseAttention's
sigmoid-soft-mask-vs-hard-topk duality, G_CL = closed_loop_nll - native_nll,
argparse/checkpoint/DDP conventions), for the "does the soft-train/hard-
deploy generalization gap (G_CL) appear inside a REAL, widely-used
sparse-attention training scheme" experiment.

Reference implementation studied: github.com/lucidrains/
native-sparse-attention-pytorch, native_sparse_attention_pytorch/
native_sparse_attention.py + README.md, main branch, read directly via
WebFetch of the raw GitHub source on 2026-09-24 (not from memory).

MECHANISM CORRESPONDENCE
-------------------------
Verified against the real repo's source (see report accompanying this port
for full detail):
  - Three parallel branches: compressed/coarse, selected/fine, local/sliding.
  - Compressed branch: block content is aggregated by a LEARNED per-position
    intra-block embedding + a learned 2-layer MLP (Linear->ReLU->Linear), not
    mean-pooling. A small number of learnable, always-visible "memory KV"
    tokens are prepended (`num_compressed_mem_kv`).
  - The fine/selected branch's block-importance score is derived directly
    from the compressed branch's own (pre-softmax) attention logits -- it is
    NOT a separately-learned scoring head.
  - The fine branch's top-k block selection operates on REAL (uncompressed)
    key/value tokens gathered from the selected blocks, not compressed reps.
  - The three branches are combined via a learned per-token, per-head
    sigmoid gate (Linear(d_model, 3*n_heads) -> Sigmoid) applied to the raw
    hidden state (not the branch outputs), producing 3 INDEPENDENT weights
    (not a softmax -- they need not sum to 1).
  - Key finding that directly informs this file's CLHR design (see
    "GENUINE DESIGN AMBIGUITY" below): the real repo's block-MEMBERSHIP
    selection (top-k) is HARD, unconditionally, from initialization, in both
    training and inference. There is no native soft/dense relaxation of
    block selection anywhere in the reference implementation -- its optional
    `use_diff_topk` straight-through path only rescales the VALUE of
    already-hard-selected blocks' keys; it does not make membership
    differentiable. This confirms the paper's premise that this project's
    own soft-to-hard closed-loop framing has no native counterpart inside
    NSA itself.

DELIBERATE SIMPLIFICATIONS (documented, not silent -- correctness and
testability prioritized over kernel-for-kernel fidelity, appropriate for
CPU-scale smoke-test prep, not the eventual 300M-scale run):
  1. No GQA-style query-head grouping (`query_heads_share_selected_kv`):
     every head has independent Q/K/V and its own block selection.
  2. No block overlap: `block_size` is shared by the compression AND
     selection partitions (non-overlapping), instead of independent
     `compress_block_size`/`compress_block_sliding_stride`/
     `selection_block_size` params. Both this and (1) are efficiency/
     parameter-count refinements orthogonal to the soft-train/hard-deploy
     gap under study.
  3. No straight-through `use_diff_topk` K-rescaling gate (it doesn't affect
     block membership, which is what CLHR cares about).
  4. The fine/selected branch's hard top-k restriction is implemented as an
     EQUIVALENT full (T,T)-attention-matrix + boolean block-selection mask,
     rather than a real tensor gather over only the selected tokens. This is
     mathematically EXACT (softmax over an explicitly masked subset of terms
     is identical to softmax computed over only those terms gathered out) --
     it forfeits only NSA's compute/memory efficiency (irrelevant at
     CPU-smoke-test scale; a real 300M-scale run should replace this with a
     gather-based implementation or this project's own
     src/flex_block_mask.py machinery).

GENUINE DESIGN AMBIGUITY -- how CLHR applies to NSA's selection branch
------------------------------------------------------------------------
Because NSA's own selection mechanism is a hard top-k gather UNCONDITIONALLY
(there is no "soft training phase" native to NSA itself), it is genuinely
ambiguous what a "standard" / soft-trained NSA baseline even means. This
file's answer (a real research design choice, not read off the reference
implementation):
  - mode="soft": the fine/selected branch is a FULL causal attention over
    ALL tokens, with each key's logit additively biased by
    log(sigmoid(block_importance_logit).clamp(min=1e-6)) -- an independent
    per-block sigmoid probability, not a softmax -- for the query's own
    already-settled past blocks, and a NEUTRAL (zero) bias for the query's
    own current, not-yet-importance-scored block. This directly mirrors
    src/sparse_attention_fineweb.py's GatedSparseAttention: sigmoid gate ->
    clamp(min=1e-6) -> log -> additive bias -> fully differentiable.
  - mode="hard": the fine/selected branch restricts attention to the
    current block plus the true top-k most-important STRICTLY-PAST blocks
    (ranking by the raw importance logits directly -- equivalent to ranking
    by softmax(logits) since softmax is monotonic, so omitting simplification
    3's softmax-before-topk step does not change which blocks are selected).
    This is exactly the real NSA mechanism.
  - condition="standard": trained with mode="soft" only.
  - condition="native_hard": trained with mode="hard" only -- this is
    literally the real NSA repo's own default behavior from initialization,
    directly testing "does training with genuine hard selection from the
    start avoid the gap?".
  - condition="clhr": trained primarily with mode="soft" (L_soft), with a
    mode="hard" auxiliary loss (L_hard) computed on the SAME batch/weights
    and mixed in as (L_soft + lambda_rca*L_hard)/(1+lambda_rca) -- identical
    mixing formula to src/moe_soft_to_hard.py's clhr_loss and
    src/sparse_attention_fineweb.py's CLHR loss. Only the selection branch's
    differentiability boundary differs between the two forwards; the
    compressed and local branches (and the gate) are computed identically
    in both, so CLHR is applied to NSA's selection branch specifically, as
    requested.
  An alternative design (NOT chosen here, flagged for the writeup) would
  have used softmax-normalized block weights (competing, sum-to-1) rather
  than independent per-block sigmoids for the soft relaxation. Independent
  sigmoids were chosen for exact structural correspondence with this
  project's own existing GatedSparseAttention baseline (so the comparison
  is apples-to-apples with the project's other conditions' notion of "soft"),
  at the cost of NOT matching the real NSA repo's own softmax-based
  importance normalization used internally for its (unimplemented, see
  simplification 3) straight-through gate. Which choice is more scientifically
  appropriate for the paper's argument is a genuine open question, not
  resolved by this port.

SCOPE NOTES
-----------
  - Checkpoint/eval machinery here is intentionally MINIMAL (final-only
    checkpoint, no periodic milestones/resume, no gate-snapshot ring buffer
    for coherent/contemporary CLHR variants) -- appropriate for pre-deadline
    CPU smoke-testing; a genuine 300M-scale run should adopt
    sparse_attention_fineweb.py's fuller checkpoint/resume cadence first.
  - No gate_utility / random-baseline floor metric (unlike
    src/moe_soft_to_hard.py's run_full_evaluation) -- left for a follow-up
    extension round, matching this project's convention of adding such
    controls incrementally rather than in an initial port.
  - No gradient-accumulation / auto-tokens-per-step derivation (unlike
    src/sparse_attention_300m.py's/fineweb.py's `grad_accum = max(1, 65536 //
    (micro_batch*seq_len*world_size))`): effective batch size here is simply
    micro_batch * world_size tokens/step * seq_len, whatever that comes out
    to for the chosen --micro-batch/GPU count -- NOT forced to match the
    existing GatedSparseAttention baselines' 65,536 tokens/step convention,
    since NSA's 3-branch attention has a materially different (larger)
    memory footprint per token and forcing an identical tokens/step target
    without first profiling real GPU memory would risk OOM. See the vast.ai
    run scripts for the chosen (conservative, unverified-on-real-GPU)
    micro-batch sizing.
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
from torch.utils.data import DataLoader, DistributedSampler, IterableDataset
from torch.nn.parallel import DistributedDataParallel as DDP

from data_loading import load_corpus

VOCAB_SIZE = 50257

CONDITIONS = ["standard", "native_hard", "clhr"]

# NSA-specific block/window hyperparameters are this port's own choice
# (not read off any reference config), scaled proportionally to each
# preset's max_seq_len. d_model/n_heads/n_layers/d_ff/max_seq_len/dropout
# for "100m"/"300m" are copied verbatim from src/sparse_attention_fineweb.py's
# MODEL_CONFIGS for direct comparability with this project's existing
# sparse-attention baselines.
NSA_MODEL_CONFIGS = {
    "dev": dict(
        d_model=64, n_heads=4, n_layers=2, d_ff=256,
        block_size=8, sliding_window_size=16, num_selected_blocks=2,
        num_compressed_mem_kv=1, max_seq_len=64, dropout=0.1,
    ),
    "31m": dict(
        d_model=256, n_heads=4, n_layers=6, d_ff=1024,
        block_size=32, sliding_window_size=64, num_selected_blocks=4,
        num_compressed_mem_kv=1, max_seq_len=512, dropout=0.1,
    ),
    "100m": dict(
        d_model=768, n_heads=12, n_layers=12, d_ff=3072,
        block_size=32, sliding_window_size=64, num_selected_blocks=4,
        num_compressed_mem_kv=1, max_seq_len=512, dropout=0.1,
    ),
    "300m": dict(
        d_model=1024, n_heads=16, n_layers=20, d_ff=4096,
        block_size=32, sliding_window_size=64, num_selected_blocks=4,
        num_compressed_mem_kv=1, max_seq_len=512, dropout=0.1,
    ),
}


# ─────────────────────────────────────────────────────────────────────────
# Core mechanism pieces (tested independently of the full module/model)
# ─────────────────────────────────────────────────────────────────────────

def block_edges(n_blocks, block_size, seq_len, device=None):
    """Block j's true rightmost REAL (unpadded) covered token index, i.e.
    the position at/after which block j's compressed representation is
    causally safe to attend to. Clamped to seq_len-1 so that trailing
    zero-padded blocks (added to make seq_len an exact multiple of
    block_size) never claim to cover a real position beyond the sequence."""
    idx = torch.arange(n_blocks, device=device)
    return torch.clamp((idx + 1) * block_size - 1, max=seq_len - 1)


def select_blocks_hard(importance_logits, edges, current_block_idx, num_selected):
    """Hard top-k block-membership selection -- the differentiability
    boundary at the heart of NSA's (and this port's native_hard/clhr)
    selection branch.

    importance_logits: (B, H, T, n_blocks) raw (pre-softmax) scores.
    edges: (n_blocks,) from block_edges().
    current_block_idx: (T,) each query position's own block index.
    num_selected: max number of STRICTLY-PAST blocks to select in addition
      to the always-included current block.

    Returns a (B, H, T, n_blocks) bool tensor with requires_grad=False
    (selection is a hard, non-differentiable choice -- gradient must flow
    only through the attention VALUES of whichever blocks end up selected,
    never through the selection decision itself, matching this project's
    "no gradient through argmax" convention -- see
    tests/test_moe_soft_to_hard.py::test_no_gradient_through_argmax)."""
    B, H, T, n_blocks = importance_logits.shape
    device = importance_logits.device
    positions = torch.arange(T, device=device)

    causal_valid = edges.view(1, -1) < positions.view(-1, 1)  # (T, n_blocks): strictly past
    block_ids = torch.arange(n_blocks, device=device).view(1, -1)
    is_current = block_ids == current_block_idx.view(-1, 1)  # (T, n_blocks)
    candidate = causal_valid & ~is_current

    with torch.no_grad():
        masked = importance_logits.detach().masked_fill(
            ~candidate.view(1, 1, T, n_blocks), float("-inf")
        )
        k = min(num_selected, n_blocks)
        topk_vals, topk_idx = masked.topk(k, dim=-1)
        valid = topk_vals > float("-inf")
        selected = torch.zeros(B, H, T, n_blocks, dtype=torch.bool, device=device)
        selected.scatter_(-1, topk_idx, valid)
        selected = selected | is_current.view(1, 1, T, n_blocks)
    return selected


def gate_combine(weights, branch_outputs):
    """Combine branch outputs with independent per-branch weights (NSA's
    real gate is 3 independent sigmoids, NOT a softmax -- weights need not
    sum to 1; see module docstring).

    weights: (B, H, T, n_branches) in [0, 1] (already post-sigmoid).
    branch_outputs: list of n_branches tensors, each (B, H, T, d_head).
    Returns (B, H, T, d_head).
    """
    stacked = torch.stack(branch_outputs, dim=-1)  # (B,H,T,Dh,n_branches)
    w = weights.unsqueeze(-2)  # (B,H,T,1,n_branches)
    return (stacked * w).sum(dim=-1)


# ─────────────────────────────────────────────────────────────────────────
# NSAAttention
# ─────────────────────────────────────────────────────────────────────────

class NSAAttention(nn.Module):
    def __init__(self, d_model, n_heads, block_size, sliding_window_size,
                 num_selected_blocks, num_compressed_mem_kv=1, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.block_size = block_size
        self.sliding_window_size = sliding_window_size
        self.num_selected_blocks = num_selected_blocks
        self.num_compressed_mem_kv = num_compressed_mem_kv
        self.scale = self.d_head ** -0.5

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.compress_pos_emb = nn.Parameter(torch.randn(n_heads, block_size, self.d_head) * 0.02)
        # Compress immediately to d_head, then a small same-width refinement
        # layer -- a genuine learned 2-layer MLP over the whole block's
        # content, but avoiding an unnecessarily large square
        # (block_size*d_head, block_size*d_head) first layer, which would
        # otherwise dominate the model's parameter count (e.g. at the 300m
        # preset, block_size*d_head=2048, so a naive square hidden layer
        # would add ~173M params across layers -- nearly 60% over the
        # intended ~300M scale; this design keeps compression overhead small
        # relative to the rest of the model, matching the preset names).
        flat_dim = block_size * self.d_head
        self.k_compress = nn.Sequential(
            nn.Linear(flat_dim, self.d_head), nn.ReLU(), nn.Linear(self.d_head, self.d_head)
        )
        self.v_compress = nn.Sequential(
            nn.Linear(flat_dim, self.d_head), nn.ReLU(), nn.Linear(self.d_head, self.d_head)
        )

        self.mem_k = nn.Parameter(torch.randn(n_heads, num_compressed_mem_kv, self.d_head) * 0.02)
        self.mem_v = nn.Parameter(torch.randn(n_heads, num_compressed_mem_kv, self.d_head) * 0.02)

        # 3 branches: compressed, selected/fine, local/sliding. Bias init
        # [-2,-2,2] (sigmoid ~0.12, 0.12, 0.88) mirrors the real repo's own
        # gate init, favoring the cheap local branch early in training.
        self.gate_proj = nn.Linear(d_model, 3 * n_heads)
        with torch.no_grad():
            self.gate_proj.weight.zero_()
            self.gate_proj.bias.copy_(torch.tensor([-2.0, -2.0, 2.0]).repeat(n_heads))

        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, t, B, T):
        return t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B,H,T,Dh)

    def _merge_heads(self, t, B, T):
        return t.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.d_head)

    def _make_blocks(self, k_in, v_in, T):
        B, H, _, Dh = k_in.shape
        n_blocks = math.ceil(T / self.block_size)
        pad_len = n_blocks * self.block_size - T
        if pad_len > 0:
            k_in = F.pad(k_in, (0, 0, 0, pad_len))
            v_in = F.pad(v_in, (0, 0, 0, pad_len))
        k_blocks = k_in.view(B, H, n_blocks, self.block_size, Dh)
        v_blocks = v_in.view(B, H, n_blocks, self.block_size, Dh)
        return k_blocks, v_blocks, n_blocks

    def _compress(self, blocks, mlp):
        B, H, n_blocks, bs, Dh = blocks.shape
        pos = self.compress_pos_emb.view(1, H, 1, bs, Dh)
        blocks = blocks + pos
        flat = blocks.reshape(B, H, n_blocks, bs * Dh)
        return mlp(flat)  # (B,H,n_blocks,Dh)

    def forward(self, x, mode="soft"):
        if mode not in ("soft", "hard"):
            raise ValueError(f"unknown mode {mode!r}")
        B, T, _ = x.shape
        device = x.device
        q = self._split_heads(self.W_q(x), B, T)
        k = self._split_heads(self.W_k(x), B, T)
        v = self._split_heads(self.W_v(x), B, T)

        positions = torch.arange(T, device=device)
        base_causal_bias = torch.zeros(T, T, device=device)
        base_causal_bias.masked_fill_(positions.view(-1, 1) < positions.view(1, -1), float("-inf"))

        # ---- compressed / coarse branch ----
        k_blocks, v_blocks, n_blocks = self._make_blocks(k, v, T)
        comp_k = self._compress(k_blocks, self.k_compress)  # (B,H,n_blocks,Dh)
        comp_v = self._compress(v_blocks, self.v_compress)

        mem_k = self.mem_k.unsqueeze(0).expand(B, -1, -1, -1)
        mem_v = self.mem_v.unsqueeze(0).expand(B, -1, -1, -1)
        full_k = torch.cat([mem_k, comp_k], dim=2)  # (B,H,mem+n_blocks,Dh)
        full_v = torch.cat([mem_v, comp_v], dim=2)

        edges = block_edges(n_blocks, self.block_size, T, device=device)
        block_causal = edges.view(1, -1) <= positions.view(-1, 1)  # (T, n_blocks)
        mem_causal = torch.ones(T, self.num_compressed_mem_kv, dtype=torch.bool, device=device)
        full_causal = torch.cat([mem_causal, block_causal], dim=1)  # (T, mem+n_blocks)

        comp_logits = torch.einsum("bhtd,bhsd->bhts", q, full_k) * self.scale
        comp_bias = torch.zeros_like(comp_logits)
        comp_bias.masked_fill_(~full_causal.view(1, 1, T, -1), float("-inf"))
        comp_attn = self.dropout(torch.softmax(comp_logits + comp_bias, dim=-1))
        compressed_out = torch.einsum("bhts,bhsd->bhtd", comp_attn, full_v)

        importance_logits = comp_logits[..., self.num_compressed_mem_kv:]  # (B,H,T,n_blocks)
        importance_logits = importance_logits.masked_fill(
            ~block_causal.view(1, 1, T, -1), float("-inf")
        )

        # ---- selected / fine branch ----
        current_block_idx = torch.clamp(positions // self.block_size, max=n_blocks - 1)
        fine_logits = torch.einsum("bhtd,bhsd->bhts", q, k) * self.scale

        if mode == "hard":
            selected_block_mask = select_blocks_hard(
                importance_logits, edges, current_block_idx, self.num_selected_blocks
            )  # (B,H,T,n_blocks) bool, no grad
            token_mask = selected_block_mask.repeat_interleave(self.block_size, dim=-1)[..., :T]
            fine_bias = torch.zeros(B, self.n_heads, T, T, device=device)
            fine_bias = fine_bias.masked_fill(~token_mask, float("-inf"))
            fine_bias = fine_bias + base_causal_bias.view(1, 1, T, T)
        else:
            is_current_block = (positions.view(T, 1) // self.block_size) == torch.arange(
                n_blocks, device=device
            ).view(1, n_blocks)  # (T, n_blocks)
            soft_weight_per_block = torch.sigmoid(importance_logits)  # -inf logit -> 0
            soft_weight_per_block = torch.where(
                is_current_block.view(1, 1, T, n_blocks),
                torch.ones_like(soft_weight_per_block),  # neutral (bias 0) for own unsettled block
                soft_weight_per_block,
            )
            token_weight = soft_weight_per_block.repeat_interleave(self.block_size, dim=-1)[..., :T]
            fine_bias = torch.log(token_weight.clamp(min=1e-6)) + base_causal_bias.view(1, 1, T, T)

        fine_attn = self.dropout(torch.softmax(fine_logits + fine_bias, dim=-1))
        fine_out = torch.einsum("bhts,bhsd->bhtd", fine_attn, v)

        # ---- local / sliding-window branch ----
        window_bias = base_causal_bias.clone()
        too_far = (positions.view(-1, 1) - positions.view(1, -1)) >= self.sliding_window_size
        window_bias = window_bias.masked_fill(too_far, float("-inf"))
        local_logits = torch.einsum("bhtd,bhsd->bhts", q, k) * self.scale
        local_attn = self.dropout(torch.softmax(local_logits + window_bias.view(1, 1, T, T), dim=-1))
        local_out = torch.einsum("bhts,bhsd->bhtd", local_attn, v)

        # ---- gate combine ----
        gate_logits = self.gate_proj(x).view(B, T, self.n_heads, 3).permute(0, 2, 1, 3)
        weights = torch.sigmoid(gate_logits)  # (B,H,T,3), independent sigmoids
        combined = gate_combine(weights, [compressed_out, fine_out, local_out])  # (B,H,T,Dh)

        out = self._merge_heads(combined, B, T)
        out = self.W_o(out)
        return out, {"importance_logits": importance_logits, "gate_weights": weights}


class NSATransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, block_size, sliding_window_size,
                 num_selected_blocks, num_compressed_mem_kv=1, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = NSAAttention(
            d_model, n_heads, block_size, sliding_window_size,
            num_selected_blocks, num_compressed_mem_kv, dropout,
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )

    def forward(self, x, mode="soft"):
        attn_out, aux = self.attn(self.ln1(x), mode=mode)
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x, aux


class NSATransformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, block_size,
                 sliding_window_size, num_selected_blocks, num_compressed_mem_kv=1,
                 max_seq_len=512, dropout=0.1):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            NSATransformerBlock(
                d_model, n_heads, d_ff, block_size, sliding_window_size,
                num_selected_blocks, num_compressed_mem_kv, dropout,
            )
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        # self.apply() above re-initializes every nn.Linear (including each
        # block's gate_proj) to normal_(std=0.02) -- restore the deliberate
        # gate bias init that NSAAttention.__init__ set, which self.apply
        # would otherwise clobber.
        for block in self.blocks:
            with torch.no_grad():
                block.attn.gate_proj.weight.zero_()
                block.attn.gate_proj.bias.copy_(
                    torch.tensor([-2.0, -2.0, 2.0]).repeat(block.attn.n_heads)
                )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, input_ids, mode="soft"):
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device)
        x = self.tok_emb(input_ids) + self.pos_emb(pos).unsqueeze(0)
        x = self.drop(x)
        for block in self.blocks:
            x, _ = block(x, mode=mode)
        x = self.ln_f(x)
        return self.lm_head(x)


# ─────────────────────────────────────────────────────────────────────────
# Training / eval
# ─────────────────────────────────────────────────────────────────────────

def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _compute_nll(model, x, y, mode):
    logits = model(x, mode=mode)
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))


def clhr_loss(model, x, y, lambda_rca):
    """(combined, l_soft, l_hard). At lambda_rca==0, combined IS l_soft
    (same tensor, not just numerically close) and l_hard is a zero
    placeholder -- mirrors src/moe_soft_to_hard.py's clhr_loss."""
    l_soft = _compute_nll(model, x, y, mode="soft")
    if lambda_rca == 0.0:
        return l_soft, l_soft, torch.tensor(0.0, device=l_soft.device)
    l_hard = _compute_nll(model, x, y, mode="hard")
    combined = (l_soft + lambda_rca * l_hard) / (1.0 + lambda_rca)
    return combined, l_soft, l_hard


def train_model(model, loader, device, condition, steps, lr, lambda_rca=0.0,
                log_every=100, grad_clip=1.0):
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}, expected one of {CONDITIONS}")
    if condition == "native_hard" and lambda_rca != 0.0:
        raise ValueError(
            "lambda_rca is meaningless for condition='native_hard' "
            "(no soft path is trained, so there is nothing for it to mix with)"
        )

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1, betas=(0.9, 0.95))
    train_iter = iter(loader)
    start = time.time()
    steps_completed = 0

    for step in range(1, steps + 1):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(loader)
            x, y = next(train_iter)
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        if condition == "standard":
            loss = _compute_nll(model, x, y, mode="soft")
        elif condition == "native_hard":
            loss = _compute_nll(model, x, y, mode="hard")
        else:  # clhr
            loss, _, _ = clhr_loss(model, x, y, lambda_rca)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        steps_completed = step

        if log_every and (step % log_every == 0 or step == steps):
            print(
                f"[train] step {step}/{steps} condition={condition} loss={loss.item():.4f}",
                flush=True,
            )

    elapsed = time.time() - start
    return steps_completed, elapsed


def evaluate_mode(model, loader, device, mode, max_batches):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if i >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            logits = model(x, mode=mode)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum"
            )
            total_loss += loss.item()
            total_tokens += y.numel()
    model.train()
    return total_loss / max(total_tokens, 1)


def run_full_evaluation(model, loader, device, max_batches=10):
    """G_CL = closed_loop_nll - native_nll, matching this project's
    established definition (see results/RUN_LEDGER.md and
    src/sparse_attention_fineweb.py:387)."""
    native_nll = evaluate_mode(model, loader, device, "soft", max_batches)
    closed_loop_nll = evaluate_mode(model, loader, device, "hard", max_batches)
    return {
        "native_nll": native_nll,
        "closed_loop_nll": closed_loop_nll,
        "G_CL": closed_loop_nll - native_nll,
    }


# ─────────────────────────────────────────────────────────────────────────
# Data loading (minimal self-contained cached-token convention, matching
# src/moe_soft_to_hard.py's wt103_{train,val}_tokens.npy cache format)
# ─────────────────────────────────────────────────────────────────────────

class TokenSeqDataset(torch.utils.data.Dataset):
    def __init__(self, tokens, seq_len):
        self.tokens = tokens
        self.seq_len = seq_len

    def __len__(self):
        return max(0, (len(self.tokens) - 1) // self.seq_len)

    def __getitem__(self, idx):
        start = idx * self.seq_len
        chunk = self.tokens[start: start + self.seq_len + 1]
        x = torch.as_tensor(chunk[:-1], dtype=torch.long)
        y = torch.as_tensor(chunk[1:], dtype=torch.long)
        return x, y


def load_wikitext_cached(data_dir, seq_len):
    data_dir = Path(data_dir)
    train_tokens = np.load(data_dir / "wt103_train_tokens.npy")
    val_tokens = np.load(data_dir / "wt103_val_tokens.npy")
    vocab_size = int(max(train_tokens.max(), val_tokens.max())) + 1
    return (
        TokenSeqDataset(train_tokens, seq_len),
        TokenSeqDataset(val_tokens, seq_len),
        vocab_size,
    )


class _XYAdapterDataset(torch.utils.data.Dataset):
    """Wraps an INDEXED dataset yielding {"input_ids": tensor(seq_len+1)}
    dicts (src/data_loading.py's load_corpus convention) into (x, y)
    tuples, matching TokenSeqDataset's contract that NSA's train loop
    expects. Only valid when `inner` supports __len__/__getitem__ -- see
    _XYAdapterIterable for the cycling=True (IterableDataset) case."""

    def __init__(self, inner):
        self.inner = inner

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, idx):
        chunk = self.inner[idx]["input_ids"]
        return chunk[:-1].clone(), chunk[1:].clone()


class _XYAdapterIterable(IterableDataset):
    """Same (x, y)-tuple adaptation as _XYAdapterDataset, but for an inner
    IterableDataset (load_corpus's real fineweb-edu cycling=True path,
    ShardedTokenDataset -- no __len__/__getitem__, only __iter__). Must
    itself be an IterableDataset so build_train_loader's isinstance check
    (and DataLoader's own shuffle=True rejection of IterableDataset)
    sees through the adapter rather than being fooled by it."""

    def __init__(self, inner):
        self.inner = inner

    def __iter__(self):
        for item in self.inner:
            chunk = item["input_ids"]
            yield chunk[:-1].clone(), chunk[1:].clone()


def _xy_adapter(inner):
    if isinstance(inner, IterableDataset):
        return _XYAdapterIterable(inner)
    return _XYAdapterDataset(inner)


def build_train_loader(train_ds, batch_size, rank=0, world_size=1):
    """DataLoader(shuffle=True) raises on an IterableDataset (the real
    fineweb-edu path, ShardedTokenDataset with cycling=True) -- mirror
    sparse_attention_fineweb.py's is_iterable branch: no shuffle/sampler
    for the iterable case (it shards itself internally via rank/world_size
    already passed to load_corpus), DistributedSampler under DDP for a
    regular indexed Dataset (the wikitext-103 path) -- without this every
    rank would iterate the same full dataset, making gradients redundant
    across GPUs and DDP a pure waste of GPU-hours."""
    if isinstance(train_ds, IterableDataset):
        return DataLoader(train_ds, batch_size=batch_size, drop_last=True)
    sampler = (
        DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        if world_size > 1 else None
    )
    return DataLoader(
        train_ds, batch_size=batch_size, shuffle=(sampler is None),
        sampler=sampler, drop_last=True,
    )


def load_corpus_cached(corpus, data_dir, seq_len, rank=0, world_size=1):
    """Dispatch to the existing WikiText-103 .npy-cache path (unchanged,
    backward-compatible) or the shared src/data_loading.py load_corpus
    utility for fineweb-edu -- needed so NSA at 300M scale can train on a
    real, non-repeating corpus instead of ~34 cycles over WikiText-103's
    117.9M tokens (see sparse_attention_fineweb.py's own corpus-generality
    rationale). rank/world_size thread through to load_corpus so each DDP
    rank shards a DIFFERENT slice of the data -- without this every GPU
    trains on identical shards, making DDP's gradient averaging a no-op
    that burns GPU-hours for zero speedup (see sparse_attention_fineweb.py's
    own rank=rank, world_size=world_size call for the reference pattern)."""
    if corpus == "wikitext-103":
        return load_wikitext_cached(data_dir, seq_len)
    elif corpus == "fineweb-edu":
        train_ds = load_corpus(
            "fineweb-edu", data_dir, seq_len=seq_len, split="train",
            tokenizer_name="gpt2", cycling=True,
            rank=rank, world_size=world_size,
        )
        val_ds = load_corpus(
            "fineweb-edu", data_dir, seq_len=seq_len, split="validation",
            tokenizer_name="gpt2", cycling=False,
        )
        return _xy_adapter(train_ds), _xy_adapter(val_ds), VOCAB_SIZE
    else:
        raise ValueError(f"unknown corpus: {corpus!r}, expected 'wikitext-103' or 'fineweb-edu'")


# ─────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────

def build_argparser():
    p = argparse.ArgumentParser(description="NSA (Native Sparse Attention) CLHR port")
    p.add_argument("--condition", choices=CONDITIONS, default="standard")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--corpus", choices=["wikitext-103", "fineweb-edu"], default="wikitext-103")
    p.add_argument("--checkpoint-dir", type=str, default=None)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--model-size", choices=list(NSA_MODEL_CONFIGS.keys()), default="dev")
    p.add_argument("--total-tokens", type=int, default=2_000_000_000)
    p.add_argument("--lambda-rca", type=float, default=1.0)
    p.add_argument("--seq-len", type=int, default=None,
                    help="override the model-size preset's max_seq_len")
    p.add_argument("--micro-batch", type=int, default=32)
    p.add_argument("--max-steps", type=int, default=None,
                    help="dev/test override: run exactly this many steps, ignoring --total-tokens")
    p.add_argument("--max-eval-batches", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--vocab-size", type=int, default=None,
                    help="override the vocab size inferred from --data-dir's cached tokens")
    return p


def main(argv=None):
    args = build_argparser().parse_args(argv)
    torch.manual_seed(args.seed)

    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        is_master = rank == 0
    else:
        device = select_device()
        world_size = 1
        rank = 0
        is_master = True

    if is_master:
        print(
            f"[nsa] condition={args.condition} seed={args.seed} "
            f"model_size={args.model_size} device={device} world_size={world_size}",
            flush=True,
        )

    if args.data_dir is None:
        raise ValueError("--data-dir is required")

    cfg = dict(NSA_MODEL_CONFIGS[args.model_size])
    if args.seq_len is not None:
        cfg["max_seq_len"] = args.seq_len
    seq_len = cfg["max_seq_len"]

    train_ds, val_ds, data_vocab_size = load_corpus_cached(
        args.corpus, args.data_dir, seq_len, rank=rank, world_size=world_size,
    )
    vocab_size = args.vocab_size if args.vocab_size is not None else data_vocab_size

    model = NSATransformer(vocab_size=vocab_size, **cfg)
    n_params = sum(p.numel() for p in model.parameters())
    if is_master:
        print(f"[nsa] parameters={n_params}", flush=True)

    model.to(device)
    if ddp:
        model = DDP(model, device_ids=[local_rank], gradient_as_bucket_view=True)

    train_loader = build_train_loader(
        train_ds, batch_size=args.micro_batch, rank=rank, world_size=world_size,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.micro_batch, shuffle=False, drop_last=True,
    )

    tokens_per_step = args.micro_batch * seq_len * world_size
    steps = args.max_steps if args.max_steps is not None else max(
        1, args.total_tokens // tokens_per_step
    )

    if is_master:
        print(">>> training starting", flush=True)
    steps_completed, elapsed = train_model(
        model, train_loader, device, args.condition, steps, args.lr,
        lambda_rca=args.lambda_rca, log_every=args.log_every,
    )

    eval_model = model.module if ddp else model
    if is_master:
        print("[eval] native_soft", flush=True)
    result = run_full_evaluation(eval_model, val_loader, device, max_batches=args.max_eval_batches)
    if is_master:
        print("[eval] closed_loop_hard", flush=True)

    result.update({
        "condition": args.condition,
        "seed": args.seed,
        "model_size": args.model_size,
        "steps_completed": steps_completed,
        "elapsed_s": elapsed,
        "parameters": n_params,
        "lambda_rca": args.lambda_rca,
    })

    if is_master:
        if args.checkpoint_dir:
            ckpt_dir = Path(args.checkpoint_dir)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"model": eval_model.state_dict(), "step": steps_completed, "tokens_seen": steps_completed * tokens_per_step},
                ckpt_dir / "final.pt",
            )
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[nsa] wrote {args.output}", flush=True)

    if ddp:
        dist.destroy_process_group()
    return result


if __name__ == "__main__":
    main()
