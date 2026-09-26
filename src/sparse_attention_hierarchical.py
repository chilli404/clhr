"""Hierarchical sparse-attention transformer with CLHR for long-context LM.

Three attention pathways per layer:
  1. Local window -- always attend to nearest w keys (no routing needed)
  2. Learned block selector -- attention-pooled block representations (NOT
     mean-pooled; a prior experiment showed negative gate utility with mean
     pooling at 2048 context).  Learned attention pooling lets the model
     select which tokens in a block are informative for routing.
  3. Their union forms the effective attention mask.

Novel routing component: BlockGateModule
  - W_bq, W_bk: block query/key projections
  - pool_query_q, pool_query_k: learnable queries for attention-pooling
    tokens within each block into a single block representation
  - Block-level RoPE applied to block representations
  - Block scores: pooled_bq @ pooled_bk^T / sqrt(d_gate)
  - Soft: sigmoid(block_scores);  Hard: top-k blocks

Conditions:
  standard                      -- soft block gating + local window
  contemporary_closedloop_hard  -- CLHR with current-model gate (detached)
  dense                         -- no routing, full causal baseline

Usage:
    python src/sparse_attention_hierarchical.py \
        --condition standard --seed 42 \
        --data-dir ./wikitext103_cache \
        --checkpoint-dir ./ckpts_hierarchical \
        --output results/hierarchical_standard_s42.json \
        --model-size small --seq-len 4096 --block-size 64 \
        --top-k-blocks 8 --local-window 256

    python src/sparse_attention_hierarchical.py \
        --condition contemporary_closedloop_hard --seed 42 \
        --data-dir ./wikitext103_cache \
        --checkpoint-dir ./ckpts_hierarchical \
        --output results/hierarchical_clhr_s42.json \
        --model-size small --seq-len 4096 --block-size 64 \
        --top-k-blocks 8 --local-window 256 --lambda-rca 1.0
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
import time
import warnings
from pathlib import Path

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.checkpoint import checkpoint as grad_checkpoint

# ---------------------------------------------------------------------------
# Optional flexible corpus loader; falls back to built-in WikiText-103.
# ---------------------------------------------------------------------------
try:
    from data_loading import load_corpus, prepare_fineweb_shards  # noqa: F401
    _HAS_DATA_LOADING = True
except ImportError:
    _HAS_DATA_LOADING = False

from flex_block_mask import build_block_mask_direct, build_causal_block_mask_direct

ATTENTION_IMPLS = ("bias", "flex")


logger = logging.getLogger(__name__)


# ===================================================================
# FlexAttention module-wide init (compile ONCE, never inside forward())
# ===================================================================
#
# Root-cause note (see benchmark/debug notes): the original code called
# `torch.compile(flex_attention)` fresh *inside* HierarchicalSparseAttention
# .forward() on every call, for every layer.  Two problems with that:
#
#   1. It re-wraps the raw function every call.  torch.compile() wrapping
#      itself is cheap, but doing it inside a function that is *itself*
#      already being torch.compile()'d (train_experiment() wraps the whole
#      model with `torch.compile(model)`) means dynamo is tracing through a
#      *nested* `torch.compile()` call.  Nested/dynamic torch.compile()
#      construction inside an already-compiled frame is unsupported and
#      forces a graph break for that op, which silently falls back to
#      *eager*, uncompiled flex_attention for that call.  Uncompiled
#      flex_attention explicitly materializes the full T x T score matrix
#      (`flex_attention called without torch.compile()` UserWarning) --
#      i.e. it does exactly the dense, non-fused computation this whole
#      effort exists to avoid, while *looking* like the "flex" branch ran.
#      No exception is raised anywhere, so nothing is logged: this is the
#      most likely mechanism behind the reported ~7%-of-peak utilization.
#   2. The only place a real failure *could* raise was the import statement
#      immediately preceding it, and only `ImportError` was caught -- any
#      other failure (e.g. a shape/BLOCK_SIZE assertion, a torch.compile
#      backend error) would have propagated and crashed training instead of
#      degrading gracefully.
#
# Fix: compile flex_attention exactly once, at module import time, and
# reuse the compiled callable everywhere.  Any failure (import or first
# compile) is recorded and surfaced via a single loud warning instead of
# being silently absorbed.

_FLEX_ATTN_RAW = None
_FLEX_CREATE_BLOCK_MASK = None
_FLEX_ATTN_COMPILED = None
_FLEX_IMPORT_ERROR: Exception | None = None
_FLEX_KERNEL_BLOCK_SIZE = 128  # FlexAttention's own sparse-block granularity;
# unrelated to our routing `block_size` (32/64). Never conflate the two.

# Above this token count, the dense-mask SDPA fallback (triggered when a
# flex_attention runtime call fails) is refused rather than attempted: it
# materializes a full (B, H, T, T) additive bias, which at T=32768, H=8,
# bf16 is 8 GiB for the bias tensor ALONE (measured; matches the observed
# "Tried to allocate 8.00 GiB" OOM at that shape), before SDPA's own
# internal buffers are even counted. Falling back at that scale doesn't
# save the run -- it just moves the crash a few lines later into a more
# confusing OOM traceback that looks unrelated to flex_attention. Below
# this threshold, dense SDPA fits comfortably and the fallback is safe
# exactly as before.
_FLEX_DENSE_FALLBACK_MAX_T = 8192


def _init_flex_attention() -> bool:
    """Import + compile flex_attention exactly once (module-wide).

    Returns True if flex_attention is importable and compilable, False
    otherwise.  Safe to call repeatedly (cheap no-op after the first call).
    """
    global _FLEX_ATTN_RAW, _FLEX_CREATE_BLOCK_MASK, _FLEX_ATTN_COMPILED
    global _FLEX_IMPORT_ERROR
    if _FLEX_ATTN_COMPILED is not None:
        return True
    if _FLEX_IMPORT_ERROR is not None:
        return False
    try:
        from torch.nn.attention.flex_attention import (
            flex_attention as _raw,
            create_block_mask as _cbm,
        )
        _FLEX_ATTN_RAW = _raw
        _FLEX_CREATE_BLOCK_MASK = _cbm
        _FLEX_ATTN_COMPILED = torch.compile(_raw)
    except Exception as e:  # intentionally broad -- see module docstring
        _FLEX_IMPORT_ERROR = e
        return False
    return True


_FLEX_FALLBACK_WARNED = False
_FLEX_RUNTIME_WARNED: set[str] = set()


def _warn_flex_unavailable() -> None:
    """Loud, one-time warning when flex_attention could not be initialized."""
    global _FLEX_FALLBACK_WARNED
    if _FLEX_FALLBACK_WARNED:
        return
    _FLEX_FALLBACK_WARNED = True
    exc = _FLEX_IMPORT_ERROR
    msg = (
        "[sparse_attention_hierarchical] FlexAttention is unavailable "
        f"({type(exc).__name__ if exc else 'unknown'}: {exc}). Falling "
        "back to dense-mask F.scaled_dot_product_attention for ALL layers. "
        "This materializes a full (B,H,T,T) mask and gets ZERO FLOP "
        "savings from block sparsity -- expect ~5x slower training and "
        "much higher memory traffic than the flex path."
    )
    logger.warning(msg)
    warnings.warn(msg, RuntimeWarning, stacklevel=2)


def _warn_flex_runtime_failure(context: str, exc: Exception) -> None:
    """Loud, one-time-per-context warning when flex_attention fails at
    runtime (after successful import/compile) -- e.g. a shape mismatch, a
    dynamo guard failure, or an OOM inside the compiled kernel. Falls back
    to the dense path for that call only.
    """
    if context in _FLEX_RUNTIME_WARNED:
        return
    _FLEX_RUNTIME_WARNED.add(context)
    msg = (
        f"[sparse_attention_hierarchical] flex_attention raised at runtime "
        f"in context={context!r}; falling back to dense-mask SDPA for this "
        f"call (and will keep doing so for this context). "
        f"{type(exc).__name__}: {exc}"
    )
    logger.warning(msg, exc_info=exc)
    warnings.warn(msg, RuntimeWarning, stacklevel=2)


# ===================================================================
# FlexAttention kernel_options fallback ladder (Ada/sm_89 shared-mem OOM)
# ===================================================================
#
# Symptom (verified on a live L40S run, `forced_block_selection` context,
# T=2048, d_head=64): flex_attention's Triton autotuner picks a forward
# config sized for A100 (~164KB shared memory/SM) or H100 (~228KB); Ada /
# sm_89 (L40S) only has ~101,376 bytes/SM. The autotuned config's
# shared-memory request (reported: 114,688 bytes) exceeds
# `Hardware limit: 101,376` for every candidate in the default search
# space -> "No valid triton configs." -> a torch._inductor
# `OutOfMemoryError` (a Triton *compile-time resource* error, distinct
# from a CUDA device-memory allocation failure -- no "CUDA out of memory"
# string appears). The previous code caught any runtime exception here and
# fell back to dense SDPA for the whole context, losing 100% of the
# block-sparsity FLOP/memory savings for that (context, shape).
#
# Fix: instead of degrading to dense on the first resource failure, retry
# with a small ladder of progressively smaller `kernel_options` tile sizes.
# BLOCK_M / BLOCK_N / num_stages are documented, stable keys on
# `torch.nn.attention.flex_attention.FlexKernelOptions` (verified against
# the installed torch build at
# .venv/lib/python3.10/site-packages/torch/nn/attention/flex_attention.py,
# lines ~88-219: plain dict, `total=False`, int-valued, "Common values:
# 16, 32, 64, 128").
#
# Approximate shared-memory model (standard Triton flash-attention-style
# fused kernel: a Q tile of BLOCK_M x d_head plus double-buffered K/V
# tiles of BLOCK_N x d_head each, pipelined over `num_stages`, bf16 =
# 2 bytes/elem):
#
#   smem_bytes ~= (BLOCK_M*d_head + 2*BLOCK_N*d_head) * 2 * num_stages
#
# With d_head=64 (production shape: d_model=1024, n_heads=16):
#
#   default (A100/H100-tuned; inferred from the failure) ~114,688 B  -> OOMs on Ada (limit 101,376 B)
#   {BLOCK_M:128, BLOCK_N:64, num_stages:2}: (128*64 + 2*64*64)*2*2 = 16,384*4 = 65,536 B  (35% headroom)
#   {BLOCK_M:64,  BLOCK_N:64, num_stages:2}: ( 64*64 + 2*64*64)*2*2 = 12,288*4 = 49,152 B  (52% headroom)
#   {BLOCK_M:64,  BLOCK_N:32, num_stages:1}: ( 64*64 + 2*32*64)*2*1 =  8,192*2 = 16,384 B  (84% headroom, most conservative)
#
# This model is an *approximation* of the real Triton-generated kernel's
# shared-memory layout (actual codegen also budgets for softmax
# accumulators/masks/padding, and the ladder deliberately does not try to
# reverse-engineer the exact 114,688-byte figure) -- it exists only to
# justify the *direction and margin* of the ladder (each step roughly
# halves the smem footprint of the failing default, with wide safety
# margin under the 101,376-byte limit). UNVERIFIED ON GPU: this machine
# has no CUDA (macOS/MPS only); whether each candidate actually compiles
# and how much speed is retained relative to dense can only be confirmed
# on an L40S.
#
# Only Triton *compile-time resource* errors ("No valid triton configs",
# "out of resource", "Hardware limit", "shared memory") are treated as
# recoverable and advance the ladder. A genuine CUDA device-memory OOM
# ("CUDA out of memory") or any other exception (shape bug, dynamo guard
# failure, etc.) is NOT retried here -- it propagates to the existing
# per-context try/except in `HierarchicalSparseAttention.forward`, which
# still falls back to dense SDPA loudly, exactly as before.

# IMPORTANT: per torch/nn/attention/flex_attention.py (FlexKernelOptions
# docstring), bare ``BLOCK_M``/``BLOCK_N`` constrain the FORWARD pass only --
# they are documented as "Thread block size for ... in forward pass". The
# BACKWARD kernel has its own separate tiles, set via the ``bwd_`` prefix:
# ``bwd_BLOCK_M1``/``bwd_BLOCK_N1`` (first backward kernel) and
# ``bwd_BLOCK_M2``/``bwd_BLOCK_N2`` (second). Constraining forward alone
# leaves backward at the same autotuned defaults that already exceeded Ada's
# 101,376-byte shared-memory limit -- and training obviously needs backward,
# so each rung sets both. (``num_stages`` is un-prefixed and applies to both.)
# Whether to use flex_attention for the `forced_block_selection` branch
# (the CLHR hard path and the closed-loop eval functions).
#
# Default False: measured on L40S at the real training shape, flex is 0.48x
# (2x SLOWER) and uses +10.7 GB on that branch, because it must build a
# BlockMask per call via create_block_mask(). The soft path, which reuses a
# cached BlockMask, is 4.38x FASTER with flex and stays enabled.
# Set True only to reproduce the A/B; see the comment at the branch itself.
_FLEX_ENABLE_FORCED_SELECTION = False

_FLEX_KERNEL_OPTION_LADDER: list[dict[str, int] | None] = [
    None,  # default: let FlexAttention's own autotuner choose (may OOM on Ada)
    {
        "BLOCK_M": 128, "BLOCK_N": 64,
        "bwd_BLOCK_M1": 64, "bwd_BLOCK_N1": 64,
        "bwd_BLOCK_M2": 64, "bwd_BLOCK_N2": 64,
        "num_stages": 2,
    },
    {
        "BLOCK_M": 64, "BLOCK_N": 64,
        "bwd_BLOCK_M1": 32, "bwd_BLOCK_N1": 64,
        "bwd_BLOCK_M2": 64, "bwd_BLOCK_N2": 32,
        "num_stages": 2,
    },
    {
        "BLOCK_M": 64, "BLOCK_N": 32,
        "bwd_BLOCK_M1": 32, "bwd_BLOCK_N1": 32,
        "bwd_BLOCK_M2": 32, "bwd_BLOCK_N2": 32,
        "num_stages": 1,
    },
]

# Cache of the ladder index that worked, keyed by
# (context, device capability, q-shape, q-dtype) -- so the ladder is
# walked (and the resulting Triton config compiled) once per distinct
# shape/device, not on every training step.
_FLEX_WORKING_LADDER_INDEX: dict[tuple, int] = {}
_FLEX_LADDER_ADVANCE_WARNED: set[tuple] = set()

_RECOVERABLE_FLEX_CONFIG_ERROR_MARKERS = (
    "no valid triton config",
    "out of resource",
    "hardware limit",
    "shared memory",
)


def _default_ladder_start(q: torch.Tensor) -> int:
    """First ladder rung to try for this device.

    Skips the ``None`` (full-autotune) rung on shared-memory-constrained
    GPUs.  Rationale: the ladder's retry loop wraps the FORWARD call, but a
    backward-pass Triton compile failure surfaces later, at
    ``loss.backward()`` -- outside this guard -- so it cannot be retried.
    The only way to preempt a backward OOM is to have already passed
    ``bwd_*`` tile sizes on the forward call.  On a GPU where the autotuned
    default is known to blow the shared-memory budget (Ada / sm_89, ~101 KB
    vs. the ~164 KB of A100 and ~228 KB of H100), starting at ``None`` risks
    a forward that happens to fit paired with a backward that does not,
    which would crash a training run instead of degrading.  So on Ada we
    begin at rung 1, which pins both forward and backward tiles.

    Ampere/Hopper keep the default rung, since autotune generally picks a
    good config there and we would rather not leave performance behind.
    """
    if not q.is_cuda:
        return 0
    try:
        major, minor = torch.cuda.get_device_capability(q.device)
    except Exception:
        return 0
    return 1 if (major, minor) == (8, 9) else 0


def _flex_config_cache_key(context: str, q: torch.Tensor) -> tuple:
    try:
        cap = torch.cuda.get_device_capability(q.device) if q.is_cuda else None
    except Exception:
        cap = None
    return (context, cap, tuple(q.shape), q.dtype)


def _is_recoverable_flex_config_error(exc: BaseException) -> bool:
    """True only for Triton compile-time resource errors (shared-memory /
    "No valid triton configs" -- the Ada-vs-A100/H100 tile-size mismatch
    this ladder exists to route around). False for a genuine CUDA
    device-memory OOM ("CUDA out of memory") or anything else, which the
    ladder cannot fix and which must still degrade to dense loudly rather
    than being silently retried.
    """
    seen: set[int] = set()
    node: BaseException | None = exc
    parts = []
    while node is not None and id(node) not in seen:
        seen.add(id(node))
        parts.append(f"{type(node).__name__}: {node}")
        node = node.__cause__ or node.__context__
    text = " | ".join(parts).lower()
    if "cuda out of memory" in text or "cuda error" in text:
        return False
    return any(marker in text for marker in _RECOVERABLE_FLEX_CONFIG_ERROR_MARKERS)


def _warn_flex_ladder_advance(context: str, idx: int,
                               cand: dict[str, int] | None,
                               exc: BaseException) -> None:
    """One-time-per-(context, idx) info log when the ladder advances past a
    candidate. Not a RuntimeWarning: this is expected, handled behavior
    (unlike `_warn_flex_runtime_failure`, which fires only once the ladder
    is exhausted and we are actually degrading to dense).
    """
    key = (context, idx)
    if key in _FLEX_LADDER_ADVANCE_WARNED:
        return
    _FLEX_LADDER_ADVANCE_WARNED.add(key)
    logger.info(
        "[sparse_attention_hierarchical] flex_attention kernel_options "
        f"candidate #{idx} ({cand}) hit a recoverable Triton resource "
        f"error in context={context!r}; advancing kernel_options ladder. "
        f"{type(exc).__name__}: {exc}"
    )


def _flex_attention_call_with_ladder(context: str, q: torch.Tensor,
                                      k: torch.Tensor, v: torch.Tensor, *,
                                      block_mask=None, score_mod=None):
    """Call the module-compiled `_FLEX_ATTN_COMPILED`, advancing through
    `_FLEX_KERNEL_OPTION_LADDER` on recoverable shared-memory / "No valid
    triton configs" Triton compile failures instead of letting the caller
    immediately fall back to dense SDPA.

    Deliberately does NOT call `torch.compile` here -- it only varies the
    `kernel_options=` argument passed into the single, module-scope
    compiled callable (see the module docstring, lines ~78-110, for why a
    fresh per-call `torch.compile` was the original root cause this file
    exists to avoid).

    Raises the last exception if every candidate in the ladder fails (the
    caller's existing try/except degrades to dense and warns loudly, same
    as before this ladder existed).
    """
    key = _flex_config_cache_key(context, q)
    start = _FLEX_WORKING_LADDER_INDEX.get(key, _default_ladder_start(q))

    last_exc: Exception | None = None
    for idx in range(start, len(_FLEX_KERNEL_OPTION_LADDER)):
        cand = _FLEX_KERNEL_OPTION_LADDER[idx]
        kwargs: dict = {}
        if block_mask is not None:
            kwargs["block_mask"] = block_mask
        if score_mod is not None:
            kwargs["score_mod"] = score_mod
        if cand is not None:
            kwargs["kernel_options"] = cand
        try:
            out = _FLEX_ATTN_COMPILED(q, k, v, **kwargs)
        except Exception as e:  # noqa: BLE001 -- classified below
            last_exc = e
            if _is_recoverable_flex_config_error(e):
                _warn_flex_ladder_advance(context, idx, cand, e)
                continue
            raise
        _FLEX_WORKING_LADDER_INDEX[key] = idx
        return out
    # Ladder exhausted: every candidate hit a recoverable resource error.
    assert last_exc is not None
    raise last_exc


CONDITIONS = ["standard", "contemporary_closedloop_hard", "dense"]

MODEL_CONFIGS = {
    "small": dict(vocab_size=50257, d_model=512, n_heads=8, n_layers=12,
                  d_ff=2048, d_gate=32),
    "medium": dict(vocab_size=50257, d_model=1024, n_heads=16, n_layers=20,
                   d_ff=4096, d_gate=32),
    "large": dict(vocab_size=50257, d_model=2048, n_heads=16, n_layers=24,
                  d_ff=8192, d_gate=64),
}

CHECKPOINT_TOKEN_MILESTONES = [
    250_000_000, 500_000_000, 1_000_000_000, 1_500_000_000,
    2_000_000_000, 2_500_000_000, 3_000_000_000, 3_500_000_000,
    4_000_000_000, 6_000_000_000,
]


# ===================================================================
# Rotary Positional Encoding
# ===================================================================

def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor,
                     sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embedding to x of shape (..., d_head)."""
    d_half = x.shape[-1] // 2
    x1 = x[..., :d_half]
    x2 = x[..., d_half:]
    return torch.cat([
        x1 * cos[..., :d_half] - x2 * sin[..., :d_half],
        x2 * cos[..., :d_half] + x1 * sin[..., :d_half],
    ], dim=-1)


class RotaryPositionalEncoding(nn.Module):
    """Standard RoPE with lazy cache expansion."""

    def __init__(self, d_head: int, max_seq_len: int = 8192,
                 base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, device=self.inv_freq.device,
                         dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)          # (T, d_head/2)
        emb = torch.cat([freqs, freqs], dim=-1)         # (T, d_head)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """x: (B, H, T, d_head).  Returns rotary-embedded tensor."""
        if seq_len > self.cos_cached.shape[0]:
            self._build_cache(seq_len)
        cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)  # (1,1,T,d)
        sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        return apply_rotary_emb(x, cos, sin)


class BlockRoPE(nn.Module):
    """RoPE for block-level representations.

    Block positions are 0, 1, 2, ..., n_blocks-1.  Applied to the pooled
    block representations so that the block routing scores are position-aware.
    """

    def __init__(self, d_head: int, max_blocks: int = 512,
                 base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_blocks)

    def _build_cache(self, n_blocks: int) -> None:
        t = torch.arange(n_blocks, device=self.inv_freq.device,
                         dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, n_blocks: int) -> torch.Tensor:
        """x: (B, H, n_blocks, d_gate).  Returns rotary-embedded tensor."""
        if n_blocks > self.cos_cached.shape[0]:
            self._build_cache(n_blocks)
        cos = self.cos_cached[:n_blocks].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[:n_blocks].unsqueeze(0).unsqueeze(0)
        return apply_rotary_emb(x, cos, sin)


# ===================================================================
# BlockGateModule -- learned attention-pooled block routing
# ===================================================================

class BlockGateModule(nn.Module):
    """Block-level routing with attention-pooled block representations.

    Unlike mean-pooling (which failed at 2048 context -- negative gate
    utility, CLHR couldn't rescue it), learned attention pooling lets the
    model select which tokens in a block are informative for routing.

    For each block, a learnable query (pool_query_q / pool_query_k) attends
    to per-token gate projections within that block.  The weighted sum becomes
    the block's representation for routing.
    """

    def __init__(self, d_model: int, n_heads: int, d_gate: int,
                 block_size: int, top_k_blocks: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_gate = d_gate
        self.block_size = block_size
        self.top_k_blocks = top_k_blocks

        # Per-token block-level projections
        self.W_bq = nn.Linear(d_model, n_heads * d_gate, bias=False)
        self.W_bk = nn.Linear(d_model, n_heads * d_gate, bias=False)

        # Learnable pooling queries -- one per head
        # pool_query_q pools tokens for block-query representations
        # pool_query_k pools tokens for block-key representations
        self.pool_query_q = nn.Parameter(torch.randn(1, n_heads, 1, d_gate) * 0.02)
        self.pool_query_k = nn.Parameter(torch.randn(1, n_heads, 1, d_gate) * 0.02)

        # Block-level RoPE on the gate dimension
        self.block_rope = BlockRoPE(d_gate, max_blocks=512)

    def _attention_pool(self, token_feats: torch.Tensor,
                        pool_query: torch.Tensor,
                        n_blocks: int) -> torch.Tensor:
        """Attention-pool per-token features into block representations.

        Args:
            token_feats: (B, H, T, d_gate) per-token gate projections.
            pool_query: (1, H, 1, d_gate) learnable pooling query.
            n_blocks: number of blocks.

        Returns:
            pooled: (B, H, n_blocks, d_gate) one vector per block.
        """
        B, H, T, dg = token_feats.shape
        bs = self.block_size

        # Pad to multiple of block_size
        pad_len = (bs - T % bs) % bs
        if pad_len > 0:
            token_feats = F.pad(token_feats, (0, 0, 0, pad_len))

        # Reshape: (B, H, n_blocks, block_size, d_gate)
        blocked = token_feats.view(B, H, n_blocks, bs, dg)

        # Attention logits: pool_query (1,H,1,dg) dot each token (B,H,nb,bs,dg)
        # -> (B, H, n_blocks, block_size)
        logits = torch.einsum("bhqd,bhnsd->bhns", pool_query, blocked)
        logits = logits / math.sqrt(dg)

        # Mask out padding positions (last block if padded)
        if pad_len > 0:
            # Only the last block has padding: mask last pad_len positions
            mask = torch.ones(bs, device=token_feats.device, dtype=torch.bool)
            mask[bs - pad_len:] = False
            # Expand mask to (1, 1, 1, block_size) and apply to last block
            pad_mask = mask.view(1, 1, 1, bs).expand(B, H, n_blocks, bs).clone()
            # Actually, only the last block is padded
            pad_mask[:, :, :-1, :] = True  # all non-last blocks: fully valid
            logits = logits.masked_fill(~pad_mask, float("-inf"))

        weights = F.softmax(logits, dim=-1)  # (B, H, n_blocks, block_size)
        weights = torch.nan_to_num(weights, nan=0.0)

        # Weighted sum: (B, H, n_blocks, d_gate)
        pooled = torch.einsum("bhns,bhnsd->bhnd", weights, blocked)
        return pooled

    def _strict_block_causal(self, n_blocks: int,
                             device: torch.device) -> torch.Tensor:
        """(n_blocks, n_blocks) bool: True iff key block j is strictly before
        query block i (j < i).

        Strict (``diagonal=-1``) rather than ``tril()`` because a block's
        attention-pooled summary spans its own future, so the diagonal block
        can never be routed to causally. Its past portion is served by the
        local window instead.
        """
        return torch.ones(
            n_blocks, n_blocks, device=device, dtype=torch.bool
        ).tril(diagonal=-1)

    def compute_block_scores(self, x: torch.Tensor,
                             n_blocks: int) -> torch.Tensor:
        """Compute block-to-block routing scores.

        Args:
            x: (B, T, D) hidden states (pre-norm output).
            n_blocks: number of blocks.

        Returns:
            block_scores: (B, H, n_blocks, n_blocks) raw routing scores
                (block-causal mask already applied, non-causal = -inf).
        """
        B, T, D = x.shape

        # Per-token gate projections
        bq = self.W_bq(x).view(B, T, self.n_heads, self.d_gate).transpose(1, 2)
        bk = self.W_bk(x).view(B, T, self.n_heads, self.d_gate).transpose(1, 2)

        # Attention-pool to block level
        pooled_bq = self._attention_pool(bq, self.pool_query_q, n_blocks)
        pooled_bk = self._attention_pool(bk, self.pool_query_k, n_blocks)

        # ---- CAUSALITY (see module docstring "Causal block routing") ----
        # Attention-pooling a block mixes ALL tokens in it, so block b's
        # summary contains tokens that are in the FUTURE of earlier queries
        # inside b.  Using it to route query block b leaked ~block_size-1
        # positions of lookahead and collapsed training loss to ~0.65
        # (PPL 1.9).  Two independent fixes are both required:
        #
        #   1. QUERY side -- shift the query summary back one block, so
        #      query block b routes using block b-1's summary, which lies
        #      entirely in its past.  Fixing only the diagonal is NOT
        #      sufficient: an unshifted query summary corrupts the choice
        #      among *past* key blocks too.
        #   2. KEY side -- restrict selection to strictly-past blocks
        #      (tril(diagonal=-1)), since the diagonal key block's summary
        #      spans its own future.
        #
        # The current block's past portion is covered by the local window,
        # which is independently causal (see _make_local_window_mask).
        pooled_bq = torch.cat(
            [torch.zeros_like(pooled_bq[:, :, :1]), pooled_bq[:, :, :-1]],
            dim=2,
        )

        # Block-level RoPE (applied after the shift so a query block keeps
        # its own positional index while carrying the previous block's
        # content -- relative distance to key block j stays b - j).
        pooled_bq = self.block_rope(pooled_bq, n_blocks)
        pooled_bk = self.block_rope(pooled_bk, n_blocks)

        # Block scores: (B, H, n_blocks, n_blocks)
        block_scores = (pooled_bq @ pooled_bk.transpose(-2, -1)
                        / math.sqrt(self.d_gate))

        # Strictly-past: query block i may attend key block j iff j < i.
        block_scores = block_scores.masked_fill(
            ~self._strict_block_causal(n_blocks, x.device), float("-inf")
        )

        return block_scores

    @torch.no_grad()
    def compute_hard_block_mask(self, x: torch.Tensor,
                                k_blocks: int | None = None) -> torch.Tensor:
        """Compute hard top-k block mask for deployment / CLHR.

        Args:
            x: (B, T, D) hidden states.
            k_blocks: override for top_k_blocks.

        Returns:
            hard_mask: (B, H, n_blocks, n_blocks) binary float.
        """
        if k_blocks is None:
            k_blocks = self.top_k_blocks
        B, T, D = x.shape
        n_blocks = (T + self.block_size - 1) // self.block_size

        block_scores = self.compute_block_scores(x, n_blocks)

        actual_k = min(k_blocks, n_blocks)
        _, topk_idx = torch.topk(block_scores, actual_k, dim=-1)
        hard_mask = torch.zeros_like(block_scores).scatter_(-1, topk_idx, 1.0)

        # Re-apply strict causality AFTER the scatter. This is load-bearing,
        # not defensive: query block 0 has no strictly-past key block, so its
        # entire score row is -inf and topk returns arbitrary indices. Without
        # this multiply those bogus picks would become real attention edges
        # pointing at the block itself or the future.
        hard_mask = hard_mask * self._strict_block_causal(
            n_blocks, x.device
        ).float()
        return hard_mask


# ===================================================================
# HierarchicalSparseAttention -- local window + block selection
# ===================================================================

def _materialize_causal_mask(causal_mask):
    """Resolve a `causal_mask` argument that may be an eager (1, 1, T, T)
    float tensor OR a zero-arg callable that lazily builds one on demand.

    2026-09-20 (dense-(T,T)-mask memory fix, 4th instance this session):
    `HierarchicalSparseTransformer.forward` and `_contemporary_clhr_forward`
    used to build this mask UNCONDITIONALLY via `_make_causal_mask` before
    the per-layer loop, even though it is provably unused whenever the
    `--attention-impl flex` hot path succeeds (see the branches in
    `HierarchicalSparseAttention.forward` below: the
    `forced_block_selection is not None and self.attention_impl == "flex"`
    branch and the soft-path's `_has_flex` success branch never reference
    `causal_mask` at all -- causality is already encoded in the flex
    BlockMask via `build_block_mask_direct`/`build_causal_block_mask_direct`
    in src/flex_block_mask.py). At T=32768 that eager tensor alone was 4.0
    GiB; at T=131072/262144 it OOM'd on the very first forward call, before
    any real attention math ran.

    Only the branches that actually READ `causal_mask` now call this to
    materialize it, lazily, at the point of use -- `dense_mode=True`, the
    `forced_block_selection`-without-flex dense fallback, the legacy
    `forced_hard_mask` fallback, and the soft-training-path's dense-SDPA
    fallback (taken only when flex is unavailable or raises). Eager
    (already-a-Tensor) callers -- e.g. `eval_closed_loop_hierarchical_hard`
    and `eval_random_block_hard`, which build the mask themselves and pass
    a concrete Tensor -- are unaffected: this is a no-op passthrough for
    them, so their behavior and values are unchanged bit-for-bit.
    """
    if callable(causal_mask):
        return causal_mask()
    return causal_mask


class HierarchicalSparseAttention(nn.Module):
    """Sparse attention combining local window with block-level routing.

    The effective attention mask is the UNION of:
      - Local window: banded mask where query i attends to
        keys in [max(0, i-w+1), i].  Always on, no routing.
      - Block selection: learned block-level routing (soft or hard).

    Soft training path:
      gate_bias = log(sigmoid(block_scores)) expanded to token level.
      Within local window, bias = 0.  Combined with causal mask and fed
      to scaled_dot_product_attention.

    Hard path (CLHR / deployment):
      top-k block selection -> binary mask -> expand to tokens.
      OR with local window -> combined hard mask.
    """

    def __init__(self, d_model: int, n_heads: int, d_gate: int,
                 block_size: int, top_k_blocks: int,
                 local_window: int = 256, dropout: float = 0.1,
                 attention_impl: str = "bias"):
        super().__init__()
        if attention_impl not in ATTENTION_IMPLS:
            raise ValueError(
                f"attention_impl must be one of {ATTENTION_IMPLS}, got "
                f"{attention_impl!r}"
            )
        self.attention_impl = attention_impl
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_gate = d_gate
        self.block_size = block_size
        self.top_k_blocks = top_k_blocks
        self.local_window = local_window

        # Standard QKV projections
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # Block gate module (attention-pooled routing)
        self.block_gate = BlockGateModule(
            d_model, n_heads, d_gate, block_size, top_k_blocks,
        )

        self.dropout = nn.Dropout(dropout)
        self.last_soft_block_scores: torch.Tensor | None = None

    # ----- mask helpers -----

    def _make_local_window_mask(self, T: int,
                                device: torch.device) -> torch.Tensor:
        """Create local window mask: (1, 1, T, T) bool.

        True where query i can attend to key j (within local window).
        """
        rows = torch.arange(T, device=device).unsqueeze(1)
        cols = torch.arange(T, device=device).unsqueeze(0)
        local_mask = (cols >= rows - self.local_window + 1) & (cols <= rows)
        return local_mask.unsqueeze(0).unsqueeze(0)

    def _expand_block_mask(self, block_mask: torch.Tensor,
                           T: int) -> torch.Tensor:
        """Expand (B, H, n_blocks_q, n_blocks_k) -> (B, H, T, T)."""
        token_mask = block_mask.repeat_interleave(
            self.block_size, dim=2
        ).repeat_interleave(self.block_size, dim=3)
        return token_mask[:, :, :T, :T]

    # ----- flex_attention helpers -----
    #
    # NOTE: an earlier version of this file had a second helper,
    # `_build_flex_block_mask_from_selection`, that tried to convert our
    # routing-level block selection (granularity = self.block_size, e.g.
    # 32/64) into FlexAttention's own kernel-block grid (granularity 128)
    # by unioning whichever kernel block *any* selected routing sub-block
    # fell into, then only re-enforcing plain causality (not the true
    # local-window / block-selection boundary) inside those kernel blocks.
    # That silently over-included up to ~127 extra tokens per query at
    # every kernel-block edge and was numerically WRONG relative to the
    # dense SDPA reference (~15% relative output error, verified on CPU in
    # tests/test_flex_attention_equivalence.py). It has been removed. The
    # helper below computes the mask_mod directly at TOKEN granularity
    # (via `q_idx // bs`), so it never conflates the two block sizes and
    # matches the dense path exactly (verified to float32 eps).

    def _create_hard_block_mask_flex(self, hard_block_selection, T, B, device):
        """Create a flex_attention BlockMask from hard block selection.

        Uses the block-sparse kernel — only computes attention for
        local window + selected blocks, skipping everything else.

        `hard_block_selection` indices are at ROUTING granularity
        (self.block_size). FlexAttention's BLOCK_SIZE=128 below is purely a
        kernel-tiling parameter for `create_block_mask`'s internal sparsity
        bookkeeping -- mask_mod is evaluated per (q_idx, kv_idx) TOKEN pair
        (via floor-division by self.block_size), so the two granularities
        never need to match and there is no approximation.
        """
        bs = self.block_size
        w = self.local_window
        n_h = self.n_heads

        # hard_block_selection: (B, H, n_blocks_q, n_blocks_k) binary
        sel = hard_block_selection

        def mask_mod(b, h, q_idx, kv_idx):
            # Causal
            causal = q_idx >= kv_idx
            # Local window
            local = (kv_idx >= q_idx - w + 1) & (kv_idx <= q_idx)
            # Block selection (routing granularity, exact token lookup)
            q_block = q_idx // bs
            k_block = kv_idx // bs
            selected = sel[b, h, q_block, k_block] > 0
            return causal & (local | selected)

        return _FLEX_CREATE_BLOCK_MASK(
            mask_mod, B, n_h, T, T, device=device,
            BLOCK_SIZE=_FLEX_KERNEL_BLOCK_SIZE,
        )

    def _get_causal_block_mask_flex(self, T: int, device: torch.device):
        """Cached plain-causal BlockMask for the soft training path.

        This mask_mod is a constant lambda (no data dependence -- the soft
        gate bias is applied via score_mod instead), so it is identical for
        every layer/step at a given (T, device) and does not need to be
        rebuilt every forward call.

        Built via `build_causal_block_mask_direct` (src/flex_block_mask.py),
        NOT `create_block_mask`: the latter evaluates its mask_mod via vmap
        over the FULL token index grid before any block compression,
        materializing a dense (1, H, T, T) bool tensor -- exactly 8.00 GiB
        at T=32768, H=8, which is the allocation that OOMs in the observed
        failure (see module-level `_FLEX_DENSE_FALLBACK_MAX_T` comment).
        `build_causal_block_mask_direct` computes the identical mask_mod /
        admitted-pair set (`q_idx >= kv_idx`) from a (num_blocks,
        num_blocks) boolean matrix instead, never touching token
        granularity at construction time. See its docstring for the full
        equivalence argument; see
        tests/test_flex_attention_equivalence.py::test_soft_path_mask_matches_previous_pattern
        for the elementwise verification against the old
        `create_block_mask` construction.
        """
        cache = getattr(self, "_causal_block_mask_cache", None)
        if cache is None:
            cache = {}
            self._causal_block_mask_cache = cache
        key = (T, str(device))
        bm = cache.get(key)
        if bm is not None:
            return bm

        bm = build_causal_block_mask_direct(
            T, _FLEX_KERNEL_BLOCK_SIZE, device,
        )
        cache[key] = bm
        return bm

    # ----- forward -----

    def forward(self, x: torch.Tensor, rope: RotaryPositionalEncoding,
                causal_mask: "torch.Tensor | Callable[[], torch.Tensor]",
                forced_hard_mask: torch.Tensor | None = None,
                forced_block_selection: torch.Tensor | None = None,
                dense_mode: bool = False,
                _use_flex: bool = True):
        """
        Args:
            x: (B, T, D) hidden states (pre-norm output).
            rope: RotaryPositionalEncoding for token-level Q/K.
            causal_mask: (1, 1, T, T) float mask (0 attend, -inf masked), OR
                a zero-arg callable that lazily builds one. Only the
                branches below that actually consume it call
                `_materialize_causal_mask` on it -- it is never touched on
                the `--attention-impl flex` hot path (both the
                forced_block_selection+flex branch and the soft path's
                flex-success branch encode causality via the flex
                BlockMask instead; see `_materialize_causal_mask`'s
                docstring for the full rationale).
            forced_hard_mask: (B, H, T, T) binary token-level hard mask
                (legacy dense-mask path, used as fallback).
            forced_block_selection: (B, H, n_blocks, n_blocks) binary block
                selection for flex_attention hard path (preferred over
                forced_hard_mask when _use_flex=True).
            dense_mode: if True, only causal mask (no routing).
            _use_flex: if True, use flex_attention for block-sparse kernels.

        Returns:
            output: (B, T, D)
            block_scores: (B, H, n_blocks, n_blocks) or None
        """
        B, T, D = x.shape
        n_h, d_h = self.n_heads, self.d_head
        n_blocks = (T + self.block_size - 1) // self.block_size

        q = self.W_q(x).view(B, T, n_h, d_h).transpose(1, 2)
        k = self.W_k(x).view(B, T, n_h, d_h).transpose(1, 2)
        v = self.W_v(x).view(B, T, n_h, d_h).transpose(1, 2)

        q = rope(q, T)
        k = rope(k, T)

        block_scores = None
        # NOTE: flex_attention is imported/compiled exactly once at module
        # scope (_init_flex_attention()), never inside this function. See
        # the module-level comment above CONDITIONS for why re-compiling
        # here on every call was the likely root cause of the silent
        # fallback to dense SDPA.
        _has_flex = _use_flex and x.is_cuda and _init_flex_attention()
        if _use_flex and x.is_cuda and not _has_flex:
            _warn_flex_unavailable()

        if dense_mode:
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=_materialize_causal_mask(causal_mask).to(q.dtype),
                dropout_p=self.dropout.p if self.training else 0.0,
            )
            self.last_soft_block_scores = None

        elif forced_block_selection is not None and self.attention_impl == "flex":
            # New, independent mechanism (--attention-impl flex): builds a
            # real BlockMask directly from the router's own top-k selection
            # via build_block_mask_direct (src/flex_block_mask.py) -- no
            # create_block_mask, no per-call token-grid mask_mod evaluation,
            # no dense (B,H,T,T) materialization anywhere in this branch.
            # Deliberately does NOT reuse the legacy `_FLEX_ENABLE_FORCED_
            # SELECTION` / `_create_hard_block_mask_flex` / kernel_options
            # ladder machinery below (that machinery builds its BlockMask via
            # create_block_mask, the exact construction this flag exists to
            # replace, and is independently gated off by design). Any
            # failure here (missing flex_attention, unsupported device,
            # compile error) is allowed to propagate -- silently falling
            # back to the dense path would defeat the entire point of this
            # flag, unlike the legacy branch's deliberate degrade-not-crash
            # behavior.
            #
            # top_k_blocks=self.top_k_blocks (2026-09-20, dynamo-recompile-
            # hazard fix): without this, build_block_mask_direct pads
            # kv_indices/q_indices to a width derived from the OBSERVED
            # per-row selection cardinality, which varies batch to batch
            # since routing is input-dependent -- this is exactly what was
            # OBSERVED to trigger a dynamo recompile of flex_attention every
            # time the width changed ("tensor 'block_mask.q_indices' size
            # mismatch"). Passing the router's actual top-k budget here lets
            # build_block_mask_direct compute a fixed, geometry-only width
            # instead (see its docstring's "Fixed-width index padding"
            # section), so the compiled graph's shape guard never fails on
            # this axis again, regardless of what any individual batch
            # actually routes to.
            if _FLEX_ATTN_COMPILED is None:
                _init_flex_attention()
            if _FLEX_ATTN_COMPILED is None:
                raise RuntimeError(
                    "attention_impl='flex' requires flex_attention, which is "
                    f"unavailable in this environment: {_FLEX_IMPORT_ERROR}"
                )
            bm = build_block_mask_direct(
                forced_block_selection, self.block_size, self.local_window,
                T, x.device, top_k_blocks=self.top_k_blocks,
            )
            out = _FLEX_ATTN_COMPILED(q, k, v, block_mask=bm)
            self.last_soft_block_scores = None

        elif forced_block_selection is not None:
            # CLHR / deployment with block selection.
            #
            # MEASURED on L40S (scripts/benchmark_flex_vs_sdpa.py, medium
            # config, T=2048, block_size=32, top_k=16, mb=8, bf16):
            #
            #   soft path : sdpa 3011 ms / 15.76 GB  ->  flex  687 ms / 12.67 GB
            #               = 4.38x FASTER, -3.1 GB   (flex is a clear win)
            #   CLHR path : sdpa 2415 ms / 11.76 GB  ->  flex 5007 ms / 22.44 GB
            #               = 0.48x, i.e. 2x SLOWER and +10.7 GB (flex LOSES)
            #
            # Reproduced at mb=16 (0.49x, 42.9 GB). Cause: this branch has to
            # build a BlockMask per call via create_block_mask(mask_mod, ...),
            # which evaluates the mask_mod closure over the index grid for
            # every layer on every microbatch. That construction costs more
            # than the dense SDPA it replaces -- and the +10.7 GB shows it
            # materializes the very thing flex exists to avoid. The soft path
            # wins precisely because it uses score_mod against a *cached*
            # causal BlockMask and pays no per-call construction.
            #
            # So: dense SDPA is the fast path here. Flex stays available
            # behind this flag for A/B, but is off by default.
            out = None
            if _has_flex and _FLEX_ENABLE_FORCED_SELECTION:
                try:
                    bm = self._create_hard_block_mask_flex(
                        forced_block_selection, T, B, x.device,
                    )
                    out = _flex_attention_call_with_ladder(
                        "forced_block_selection", q, k, v, block_mask=bm,
                    )
                except Exception as e:  # runtime failure: degrade, don't crash
                    _warn_flex_runtime_failure("forced_block_selection", e)
                    out = None
            if out is None:
                # Dense fallback (CPU, or flex runtime failure): expand to
                # dense token mask. Mathematically equivalent to the flex
                # path -- see tests/test_flex_attention_equivalence.py.
                token_mask = self._expand_block_mask(forced_block_selection, T)
                local_mask = self._make_local_window_mask(T, x.device)
                combined = token_mask + local_mask.float()
                gate_bias = torch.where(
                    combined > 0,
                    torch.zeros(1, device=x.device, dtype=q.dtype),
                    torch.tensor(float("-inf"), device=x.device, dtype=q.dtype),
                )
                attn_mask = gate_bias + _materialize_causal_mask(causal_mask)
                out = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=attn_mask.to(q.dtype),
                    dropout_p=self.dropout.p if self.training else 0.0,
                )
            self.last_soft_block_scores = None

        elif forced_hard_mask is not None:
            # Legacy dense-mask path (fallback for CPU or no flex_attention)
            gate_bias = torch.where(
                forced_hard_mask > 0,
                torch.zeros_like(forced_hard_mask),
                torch.full_like(forced_hard_mask, float("-inf")),
            )
            attn_mask = gate_bias + _materialize_causal_mask(causal_mask)
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask.to(q.dtype),
                dropout_p=self.dropout.p if self.training else 0.0,
            )
            self.last_soft_block_scores = None

        else:
            # Soft training path
            block_scores = self.block_gate.compute_block_scores(x, n_blocks)
            self.last_soft_block_scores = block_scores.detach()

            out = None
            if _has_flex:
                try:
                    # Flex with score_mod: never materializes the dense
                    # (B,H,T,T) bias/mask; still computes the FULL causal
                    # score matrix (score_mod can't skip blocks it might
                    # need for the soft bias), but at least avoids the
                    # dense-mask memory blowup. Real FLOP savings for the
                    # soft path would require a hard/approximate block
                    # mask too; out of scope for this fix (kept behavior
                    # identical to the dense reference).
                    soft_block_mask = torch.sigmoid(block_scores)
                    bs = self.block_size
                    w = self.local_window

                    def score_mod(score, b, h, q_idx, kv_idx):
                        q_blk = q_idx // bs
                        k_blk = kv_idx // bs
                        is_local = (kv_idx >= q_idx - w + 1) & (kv_idx <= q_idx)
                        bias = torch.where(
                            is_local,
                            torch.zeros_like(score),
                            torch.log(
                                soft_block_mask[b, h, q_blk, k_blk].clamp(min=1e-6)
                            ),
                        )
                        return score + bias

                    bm = self._get_causal_block_mask_flex(T, x.device)
                    out = _flex_attention_call_with_ladder(
                        "soft_path", q, k, v, score_mod=score_mod,
                        block_mask=bm,
                    )
                except Exception as e:
                    if T > _FLEX_DENSE_FALLBACK_MAX_T:
                        # Falling back to dense-mask SDPA here would just
                        # trade this clear error for a confusing OOM a few
                        # lines later: dense SDPA materializes a full
                        # (B, H, T, T) additive bias, which alone is 8 GiB
                        # at T=32768, H=8, bf16 (the exact allocation size
                        # observed to OOM in production at this shape). The
                        # likely root cause of the ORIGINAL flex failure at
                        # this scale is a mask/bias construction path that
                        # itself materializes dense (B, H, T, T) state (e.g.
                        # `create_block_mask`); see
                        # `build_block_mask_direct`/
                        # `build_causal_block_mask_direct` in
                        # src/flex_block_mask.py for the memory-safe
                        # alternative. Re-raise instead of degrading.
                        _warn_flex_runtime_failure("soft_path", e)
                        raise RuntimeError(
                            f"flex_attention failed at runtime in "
                            f"context='soft_path' with T={T} tokens, which "
                            f"exceeds _FLEX_DENSE_FALLBACK_MAX_T="
                            f"{_FLEX_DENSE_FALLBACK_MAX_T}. Refusing to fall "
                            "back to dense-mask SDPA: it would materialize a "
                            "full (B, H, T, T) additive bias tensor, which "
                            "alone is already multiple GiB at this sequence "
                            "length and will very likely OOM moments later "
                            "with a confusing, seemingly-unrelated CUDA "
                            "OutOfMemoryError. The likely underlying cause "
                            "is a flex mask/bias construction path that "
                            "itself materializes dense (B, H, T, T) state "
                            "(e.g. torch's create_block_mask) rather than "
                            "the memory-safe block-granularity construction "
                            "in src/flex_block_mask.py. Original error: "
                            f"{type(e).__name__}: {e}"
                        ) from e
                    _warn_flex_runtime_failure("soft_path", e)
                    out = None
            if out is None:
                # Training soft: use dense SDPA (grad flows through sigmoid)
                soft_block_mask = torch.sigmoid(block_scores)
                token_block_bias = self._expand_block_mask(soft_block_mask, T)
                block_bias = torch.log(token_block_bias.clamp(min=1e-6))
                local_mask = self._make_local_window_mask(T, x.device)
                gate_bias = torch.where(local_mask,
                                        torch.zeros_like(block_bias),
                                        block_bias)
                attn_mask = gate_bias + _materialize_causal_mask(causal_mask)
                out = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=attn_mask.to(q.dtype),
                    dropout_p=self.dropout.p if self.training else 0.0,
                )

        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.W_o(out), block_scores


# ===================================================================
# Transformer Block (pre-norm, hierarchical sparse attention + FFN)
# ===================================================================

class HierarchicalTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, d_gate: int,
                 block_size: int, top_k_blocks: int,
                 local_window: int = 256, dropout: float = 0.1,
                 attention_impl: str = "bias"):
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = HierarchicalSparseAttention(
            d_model, n_heads, d_gate, block_size, top_k_blocks,
            local_window=local_window, dropout=dropout,
            attention_impl=attention_impl,
        )
        self.ff_norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, rope: RotaryPositionalEncoding,
                causal_mask: "torch.Tensor | Callable[[], torch.Tensor]",
                forced_hard_mask: torch.Tensor | None = None,
                forced_block_selection: torch.Tensor | None = None,
                dense_mode: bool = False):
        # `causal_mask` may be an eager Tensor or a zero-arg callable that
        # lazily builds one -- this block only ever passes it through
        # unmodified to `self.attn`, which is the sole consumer that
        # decides (per branch) whether to materialize it. See
        # `_materialize_causal_mask` above for the full rationale.
        normed = self.attn_norm(x)
        attn_out, block_scores = self.attn(
            normed, rope, causal_mask,
            forced_hard_mask=forced_hard_mask,
            forced_block_selection=forced_block_selection,
            dense_mode=dense_mode,
        )
        x = x + attn_out
        x = x + self.ff(self.ff_norm(x))
        return x


# ===================================================================
# Hierarchical Sparse Transformer (full model)
# ===================================================================

class HierarchicalSparseTransformer(nn.Module):
    def __init__(self, vocab_size: int = 50257, d_model: int = 512,
                 n_heads: int = 8, n_layers: int = 12, d_ff: int = 2048,
                 d_gate: int = 32, block_size: int = 64,
                 top_k_blocks: int = 8, local_window: int = 256,
                 max_seq_len: int = 4096, dropout: float = 0.1,
                 attention_impl: str = "bias"):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_gate = d_gate
        self.block_size = block_size
        self.top_k_blocks = top_k_blocks
        self.local_window = local_window
        self.max_seq_len = max_seq_len

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.rope = RotaryPositionalEncoding(
            d_model // n_heads, max_seq_len=max_seq_len,
        )
        self.drop = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            HierarchicalTransformerBlock(
                d_model, n_heads, d_ff, d_gate, block_size, top_k_blocks,
                local_window=local_window, dropout=dropout,
                attention_impl=attention_impl,
            )
            for _ in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def _make_causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """Returns (1, 1, T, T) float mask: 0 for attend, -inf for masked."""
        if (not hasattr(self, "_causal_cache")
                or self._causal_cache.shape[-1] < T
                or self._causal_cache.device != device):
            mask = torch.zeros(T, T, device=device)
            mask.masked_fill_(
                ~torch.ones(T, T, device=device, dtype=torch.bool).tril(),
                float("-inf"),
            )
            self._causal_cache = mask.unsqueeze(0).unsqueeze(0)
        return self._causal_cache[:, :, :T, :T]

    def forward(self, input_ids: torch.Tensor,
                forced_hard_masks: list[torch.Tensor] | None = None,
                forced_block_selections: list[torch.Tensor] | None = None,
                use_checkpoint: bool = False,
                dense_mode: bool = False,
                skip_lm_head: bool = False):
        """
        Args:
            input_ids: (B, T) token ids.
            forced_hard_masks: optional per-layer list of
                (B, H, T, T) token-level hard masks (legacy dense path).
            forced_block_selections: optional per-layer list of
                (B, H, n_blocks, n_blocks) binary block selections.
                Preferred over forced_hard_masks — uses flex_attention
                for true block-sparse execution.
            use_checkpoint: gradient checkpointing.
            dense_mode: skip routing entirely.
            skip_lm_head: if True, return the post-final_norm hidden
                states instead of vocab logits, and never materialise the
                (B, T, vocab_size) logits tensor here.  Used by the
                chunked-loss path (see `chunked_lm_head_loss`) so the
                vocab projection can be fused with the loss and
                checkpointed per chunk instead of computed all at once.

        Returns:
            logits (or, if skip_lm_head, the (B, T, d_model) hidden
                states after final_norm): (B, T, vocab_size) normally.
            all_block_scores: list of (B, H, n_blocks, n_blocks) scores
                (empty when dense or forced).
        """
        B, T = input_ids.shape
        device = input_ids.device

        x = self.tok_emb(input_ids)
        x = self.drop(x)

        # Lazy: do NOT eagerly materialize the dense (1, 1, T, T) causal
        # mask here. It is provably unused whenever the `--attention-impl
        # flex` hot path succeeds (see `_materialize_causal_mask`'s
        # docstring) -- eagerly building it here was a 4.0 GiB allocation
        # at T=32768 and an outright OOM at T>=131072, on every forward
        # call, regardless of whether any consumer ever read it. Pass a
        # zero-arg thunk instead; only the branches inside
        # `HierarchicalSparseAttention.forward` that actually need a dense
        # mask call `_materialize_causal_mask` on it, which invokes this
        # thunk (and hits `self._make_causal_mask`'s own cache) on demand.
        causal_mask = lambda: self._make_causal_mask(T, device)  # noqa: E731

        all_block_scores: list[torch.Tensor] = []
        for i, layer in enumerate(self.layers):
            fm = (forced_hard_masks[i]
                  if forced_hard_masks is not None else None)
            fbs = (forced_block_selections[i]
                   if forced_block_selections is not None else None)

            if use_checkpoint and self.training:
                def _run_layer(_x, _layer=layer, _rope=self.rope,
                               _cm=causal_mask, _fm=fm, _fbs=fbs,
                               _dm=dense_mode):
                    return _layer(_x, _rope, _cm,
                                  forced_hard_mask=_fm,
                                  forced_block_selection=_fbs,
                                  dense_mode=_dm)
                x = grad_checkpoint(_run_layer, x, use_reentrant=False)
            else:
                x = layer(x, self.rope, causal_mask,
                          forced_hard_mask=fm,
                          forced_block_selection=fbs,
                          dense_mode=dense_mode)

            bs = layer.attn.last_soft_block_scores
            if bs is not None:
                all_block_scores.append(bs)

        x = self.final_norm(x)
        if skip_lm_head:
            return x, all_block_scores
        logits = self.lm_head(x)
        return logits, all_block_scores

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ===================================================================
# Chunked vocab-projection + loss (memory ceiling fix for long context)
# ===================================================================

def chunked_lm_head_loss(
    hidden: torch.Tensor,
    targets: torch.Tensor,
    lm_head: nn.Module,
    chunk_size: int,
) -> torch.Tensor:
    """Token-count-weighted mean cross-entropy loss without ever
    materialising the full (B, T, vocab_size) logits tensor.

    At long context, `lm_head(hidden)` followed by `F.cross_entropy`
    allocates a (B, T, vocab_size) tensor (and a second, fp32-upcast copy
    of it internally, since cross_entropy always computes its softmax in
    float32) -- this is the actual OOM ceiling, not the reduction over it.
    Chunking `F.cross_entropy` alone does NOT help: autograd would still
    keep every chunk's logits alive for backward. Instead, this fuses the
    vocab projection with the loss and wraps that pair in
    `torch.utils.checkpoint` per chunk, so backward recomputes and frees
    one chunk's logits at a time rather than holding all of them.

    Mathematically identical, for any chunk_size >= 1, to::

        F.cross_entropy(lm_head(hidden).reshape(-1, V), targets.reshape(-1))

    i.e. reduction='mean' over all B*T positions -- a token-count-weighted
    mean, NOT a mean of per-chunk means (those differ whenever the final
    chunk is short).

    Args:
        hidden: (B, T, d_model) hidden states AFTER final_norm and BEFORE
            lm_head, i.e. what `HierarchicalSparseTransformer.forward`
            returns when called with skip_lm_head=True.
        targets: (B, T) target token ids.
        lm_head: the vocab-projection module (e.g. raw_model.lm_head).
        chunk_size: sequence positions (flattened over B*T) per chunk.
            Values >= B*T collapse to a single chunk.
    """
    B, T, D = hidden.shape
    flat_hidden = hidden.reshape(B * T, D)
    flat_targets = targets.reshape(-1)
    n = flat_hidden.shape[0]
    eff_chunk = max(1, min(chunk_size, n))

    total = flat_hidden.new_zeros((), dtype=torch.float32)
    for start in range(0, n, eff_chunk):
        end = min(start + eff_chunk, n)
        h_chunk = flat_hidden[start:end]
        t_chunk = flat_targets[start:end]

        def _project_and_loss(_h, _t=t_chunk, _lm_head=lm_head):
            chunk_logits = _lm_head(_h)
            return F.cross_entropy(chunk_logits, _t, reduction="sum")

        if h_chunk.requires_grad:
            # use_reentrant=False: checkpoint intercepts saved tensors via
            # autograd's saved-tensor hooks rather than a custom Function,
            # so it recomputes _project_and_loss during backward instead
            # of keeping chunk_logits alive for the whole outer loop.
            chunk_sum = grad_checkpoint(
                _project_and_loss, h_chunk, use_reentrant=False,
            )
        else:
            chunk_sum = _project_and_loss(h_chunk)
        total = total + chunk_sum.float()

    return total / n


# REMOVED 2026-09-20: `_build_hard_token_mask` used to live here. It combined
# a hard block mask with a local window into a dense (B, H, T, T) binary
# token-level mask via `repeat_interleave` + `torch.where` -- the same
# O(T^2)-memory shape as the `_make_local_window_mask_bool` bug fixed
# elsewhere in this file (see the comments left at the old
# `eval_closed_loop_hierarchical_hard`/`eval_random_block_hard` call sites).
# It had ZERO callers among PRODUCTION code paths -- the live eval/CLHR path
# builds hard masks via `build_block_mask_direct` (src/flex_block_mask.py)
# instead, which never materializes a dense (T, T) tensor. It DID have one
# caller: an independent from-scratch equivalence oracle in
# tests/test_flex_attention_equivalence.py
# (test_contemporary_clhr_forward_runs_and_matches_dense_reimplementation),
# which deliberately never calls production mask-construction helpers so it
# can catch the flex path silently computing something different. That
# caller was NOT a reason to keep this in production -- it has been
# relocated into the test file itself as
# `_build_hard_token_mask_reference`, clearly marked as a test-only oracle.
# Removed here as inert dead-in-production code carrying the same landmine
# shape, rather than leaving it to be silently re-wired by a future refactor
# and reproduce this session's OOM.

_LOCAL_WINDOW_MASK_MAX_ELEMENTS = 256 * 1024 * 1024  # 256M bool elems (256 MiB)


def _make_local_window_mask_bool(T: int, local_window: int,
                                 device: torch.device) -> torch.Tensor:
    """(1, 1, T, T) dense bool local window mask.

    WARNING: this materializes a full (T, T) tensor -- O(T^2) memory (e.g.
    4.29 GB at T=65536). It is a small-T reference/testing utility only
    (kept because tests/test_flex_attention_equivalence.py uses it as an
    independent oracle); it must NEVER be called from a production
    training/eval hot path at long context. It used to be called (with its
    result unused -- dead code) from eval_closed_loop_hierarchical_hard and
    eval_random_block_hard, which is exactly what OOM'd at T=65536; both
    call sites were removed 2026-09-20. Production code that needs
    local-window membership at scale should use the pointwise trick instead
    (see `is_local = (kv_idx >= q_idx - w + 1) & (kv_idx <= q_idx)` in the
    soft-path score_mod above, or build_block_mask_direct /
    build_causal_block_mask_direct in src/flex_block_mask.py), never a
    materialized dense tensor.

    Raises RuntimeError instead of silently attempting a huge allocation
    if T is large enough that the dense mask alone would exceed
    _LOCAL_WINDOW_MASK_MAX_ELEMENTS.
    """
    if T * T > _LOCAL_WINDOW_MASK_MAX_ELEMENTS:
        raise RuntimeError(
            f"_make_local_window_mask_bool(T={T}) would allocate a dense "
            f"({T}, {T}) bool tensor ({T * T} elements), exceeding the "
            f"{_LOCAL_WINDOW_MASK_MAX_ELEMENTS}-element safety limit. This "
            "function is a small-T reference/testing utility only -- use "
            "the pointwise local-window predicate (see score_mod) or "
            "build_block_mask_direct instead at this scale."
        )
    rows = torch.arange(T, device=device).unsqueeze(1)
    cols = torch.arange(T, device=device).unsqueeze(0)
    mask = (cols >= rows - local_window + 1) & (cols <= rows)
    return mask.unsqueeze(0).unsqueeze(0)


# ===================================================================
# Evaluation helpers
# ===================================================================

@torch.no_grad()
def eval_closed_loop_hierarchical_hard(
    model: HierarchicalSparseTransformer,
    loader: DataLoader,
    device: torch.device,
    k_blocks: int,
    max_batches: int = 100,
    loss_chunk_size: int = 0,
) -> float:
    """Closed-loop hierarchical hard NLL: current gates, top-k blocks + local."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        if isinstance(batch, dict):
            batch = batch["input_ids"]
        batch = batch.to(device)
        x_ids, y = batch[:, :-1], batch[:, 1:]
        B, T = x_ids.shape

        # NOTE (2026-09-20): this function used to build
        # `local_mask = _make_local_window_mask_bool(T, local_window,
        # device)` here, a dense (1, 1, T, T) bool tensor -- 4.29 GB at
        # T=65536. That value was DEAD CODE: it was never read again in
        # this function. The local window is already applied inside
        # layer.attn(..., forced_block_selection=hard_block) below, which
        # (for attention_impl="flex") builds its mask via
        # build_block_mask_direct (src/flex_block_mask.py), never
        # materializing a dense (T, T) tensor. Removing this dead
        # allocation is a pure no-op on the returned loss -- see
        # tests/test_local_window_mask_memory.py -- and is what actually
        # fixes the OOM observed at T=65536 (traceback pointed at
        # `_make_local_window_mask_bool`'s `mask = (cols >= rows -
        # local_window + 1) & (cols <= rows)` line, called from here).

        # Embed
        x = model.tok_emb(x_ids)
        x = model.drop(x)
        # Lazy: causal_mask is unused whenever the flex hot path
        # succeeds (see _materialize_causal_mask's docstring above --
        # this eval loop passes forced_block_selection to layer.attn,
        # hitting the same never-reads-causal_mask branch). Eager
        # construction here was the 4th-instance bug's eval-path twin,
        # missed by the training-time fix's scope; T=131072 OOM'd here
        # too before this line was made lazy.
        causal_mask = lambda: model._make_causal_mask(T, device)  # noqa: E731

        for layer in model.layers:
            h = layer.attn_norm(x)

            # Current-gate hard block mask
            hard_block = layer.attn.block_gate.compute_hard_block_mask(
                h, k_blocks=k_blocks,
            )

            # Pass block selection directly — flex_attention handles expansion
            attn_out, _ = layer.attn(
                h, model.rope, causal_mask,
                forced_block_selection=hard_block,
            )
            x = x + attn_out
            x = x + layer.ff(layer.ff_norm(x))

        if loss_chunk_size > 0:
            hidden = model.final_norm(x)
            mean_loss = chunked_lm_head_loss(
                hidden, y, model.lm_head, loss_chunk_size,
            )
            loss = mean_loss * y.numel()
        else:
            logits = model.lm_head(model.final_norm(x))
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                reduction="sum",
            )
        total_loss += loss.item()
        total_tokens += y.numel()

    return total_loss / total_tokens


@torch.no_grad()
def eval_random_block_hard(
    model: HierarchicalSparseTransformer,
    loader: DataLoader,
    device: torch.device,
    k_blocks: int,
    max_batches: int = 50,
    loss_chunk_size: int = 0,
) -> float:
    """Random block selection + local window NLL (control for gate utility)."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    block_size = model.block_size
    n_heads = model.n_heads

    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        if isinstance(batch, dict):
            batch = batch["input_ids"]
        batch = batch.to(device)
        x_ids, y = batch[:, :-1], batch[:, 1:]
        B, T = x_ids.shape
        n_blocks = (T + block_size - 1) // block_size

        # NOTE (2026-09-20): see the identical note in
        # eval_closed_loop_hierarchical_hard above -- this used to build a
        # dense (1, 1, T, T) `local_mask` via `_make_local_window_mask_bool`
        # that was never read again (dead code, and the actual source of
        # the T=65536 eval-time OOM). The local window is applied inside
        # layer.attn(..., forced_block_selection=rand_block) below via the
        # memory-safe build_block_mask_direct path.

        x = model.tok_emb(x_ids)
        x = model.drop(x)
        # Lazy: causal_mask is unused whenever the flex hot path
        # succeeds (see _materialize_causal_mask's docstring above --
        # this eval loop passes forced_block_selection to layer.attn,
        # hitting the same never-reads-causal_mask branch). Eager
        # construction here was the 4th-instance bug's eval-path twin,
        # missed by the training-time fix's scope; T=131072 OOM'd here
        # too before this line was made lazy.
        causal_mask = lambda: model._make_causal_mask(T, device)  # noqa: E731

        for layer in model.layers:
            h = layer.attn_norm(x)

            # Random block mask (causal)
            rand_scores = torch.rand(
                B, n_heads, n_blocks, n_blocks, device=device,
            )
            block_causal = torch.ones(
                n_blocks, n_blocks, device=device, dtype=torch.bool,
            ).tril()
            rand_scores = rand_scores.masked_fill(~block_causal, float("-inf"))
            actual_k = min(k_blocks, n_blocks)
            _, topk_idx = torch.topk(rand_scores, actual_k, dim=-1)
            rand_block = torch.zeros_like(rand_scores).scatter_(
                -1, topk_idx, 1.0,
            )
            rand_block = rand_block * block_causal.float()

            attn_out, _ = layer.attn(
                h, model.rope, causal_mask,
                forced_block_selection=rand_block,
            )
            x = x + attn_out
            x = x + layer.ff(layer.ff_norm(x))

        if loss_chunk_size > 0:
            hidden = model.final_norm(x)
            mean_loss = chunked_lm_head_loss(
                hidden, y, model.lm_head, loss_chunk_size,
            )
            loss = mean_loss * y.numel()
        else:
            logits = model.lm_head(model.final_norm(x))
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                reduction="sum",
            )
        total_loss += loss.item()
        total_tokens += y.numel()

    return total_loss / total_tokens


@torch.no_grad()
def evaluate_nll(
    model: HierarchicalSparseTransformer,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 100,
    loss_chunk_size: int = 0,
) -> float:
    """Native soft NLL (soft block gating + local window).

    loss_chunk_size=0 (default) is BIT-IDENTICAL to the pre-2026-09-21
    behavior: full (B,T,vocab) logits + F.cross_entropy. At long context
    this OOMs (see evaluate_dense_nll's docstring for the exact incident:
    T=65536, 6.14 GiB allocation, observed on ssh2.vast.ai:13318,
    LONGCTX_64K_K4_S123 -- and this crash discards the ALREADY-COMPUTED
    native_nll/G_CL/gate_utility too, since run_full_evaluation builds one
    dict at the end and a later-stage crash never returns it). When >0,
    routes through the SAME chunked_lm_head_loss fusion already used by
    the training path (never materializes the full logits tensor).
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        if isinstance(batch, dict):
            batch = batch["input_ids"]
        batch = batch.to(device)
        x, y = batch[:, :-1], batch[:, 1:]
        if loss_chunk_size > 0:
            hidden, _ = model(x, use_checkpoint=False, dense_mode=False,
                              skip_lm_head=True)
            mean_loss = chunked_lm_head_loss(
                hidden, y, model.lm_head, loss_chunk_size,
            )
            loss_sum = mean_loss * y.numel()
        else:
            logits, _ = model(x, use_checkpoint=False, dense_mode=False)
            loss_sum = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                reduction="sum",
            )
        total_loss += loss_sum.item()
        total_tokens += y.numel()
    return total_loss / total_tokens


@torch.no_grad()
def evaluate_dense_nll(
    model: HierarchicalSparseTransformer,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 100,
    loss_chunk_size: int = 0,
) -> float:
    """Dense attention NLL (no routing, causal-only).

    OBSERVED INCIDENT (2026-09-21): this function's unchunked
    F.cross_entropy on the full (B,T,vocab) logits tensor OOM'd at
    T=65536 (`Tried to allocate 6.14 GiB`, ssh2.vast.ai:13318,
    LONGCTX_64K_K4_S123, at step 3800/3814 -- i.e. AFTER training
    completed, in run_full_evaluation's LAST call). Because
    run_full_evaluation builds one dict at the end and returns it only
    once, this crash discarded native_nll/G_CL/gate_utility too, even
    though those were already computed successfully earlier in the same
    call. dense_mode's own attention forward pass is more memory-hungry
    than the sparse/flex-routed paths used by the other eval functions
    (which is why THEY survived and this one, called last, did not) --
    the fix below (loss_chunk_size>0) is the same chunked_lm_head_loss
    fusion already used by the training path; loss_chunk_size=0 is
    bit-identical to the pre-fix behavior.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        if isinstance(batch, dict):
            batch = batch["input_ids"]
        batch = batch.to(device)
        x, y = batch[:, :-1], batch[:, 1:]
        if loss_chunk_size > 0:
            hidden, _ = model(x, use_checkpoint=False, dense_mode=True,
                              skip_lm_head=True)
            mean_loss = chunked_lm_head_loss(
                hidden, y, model.lm_head, loss_chunk_size,
            )
            loss_sum = mean_loss * y.numel()
        else:
            logits, _ = model(x, use_checkpoint=False, dense_mode=True)
            loss_sum = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                reduction="sum",
            )
        total_loss += loss_sum.item()
        total_tokens += y.numel()
    return total_loss / total_tokens


def run_full_evaluation(
    model: HierarchicalSparseTransformer,
    val_loader: DataLoader,
    device: torch.device,
    k_blocks: int = 8,
    loss_chunk_size: int = 0,
) -> dict:
    """Full evaluation suite: native, CL-hard, random, G_CL, utility, dense.

    loss_chunk_size threaded to all four sub-evals (see each function's
    docstring for the OOM incident this fixes) -- 0 is bit-identical to
    pre-2026-09-21 behavior. A crash in ANY of these four discards the
    whole dict (built once, returned once), including sub-evals that
    already completed successfully earlier in this same call -- this is
    why all four needed the fix, not just the one that happened to crash
    first at a given context length.
    """
    native_nll = evaluate_nll(
        model, val_loader, device, loss_chunk_size=loss_chunk_size,
    )

    cl_hard_nll = eval_closed_loop_hierarchical_hard(
        model, val_loader, device, k_blocks=k_blocks,
        loss_chunk_size=loss_chunk_size,
    )

    # Random block hard (average over 5 seeds for stability)
    rand_nlls = []
    for _ in range(5):
        rand_nlls.append(
            eval_random_block_hard(
                model, val_loader, device, k_blocks=k_blocks,
                loss_chunk_size=loss_chunk_size,
            )
        )
    rand_nll_mean = float(np.mean(rand_nlls))
    rand_nll_std = float(np.std(rand_nlls))

    # evaluate_dense_nll's dense_mode=True attention branch (unlike every
    # other branch here) eagerly materializes a full (T, T) fp32 causal
    # mask -- `_materialize_causal_mask`'s ONE genuinely-needed caller,
    # since dense attention has no block/local-window structure to exploit.
    # loss_chunk_size fixes this function's LOGITS/cross-entropy OOM but
    # does not and cannot fix this separate, attention-level allocation.
    # OBSERVED (2026-09-22): T=131072 needs 131072^2*4 bytes = 64.00 GiB for
    # this mask alone -- physically impossible on any single GPU in this
    # project's fleet, independent of any other memory optimization.
    # dense_nll is ALREADY established as scientifically invalid at long
    # context (off-distribution dense-mode eval on sparse-trained weights,
    # see this file's other dense_nll-invalidity notes) -- losing it at
    # extreme T costs nothing real, so failing gracefully here (instead of
    # letting the exception destroy native_nll/G_CL/gate_utility, which
    # ARE valid and already computed by this point) is the correct trade,
    # not a workaround.
    try:
        dense_nll = evaluate_dense_nll(
            model, val_loader, device, loss_chunk_size=loss_chunk_size,
        )
        dense_nll_oom = False
    except torch.OutOfMemoryError:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        dense_nll = None
        dense_nll_oom = True

    g_cl = cl_hard_nll - native_nll
    gate_utility = rand_nll_mean - cl_hard_nll

    result = {
        "native_nll": round(native_nll, 6),
        "native_ppl": round(math.exp(native_nll), 2),
        "closed_loop_hard_nll": round(cl_hard_nll, 6),
        "G_CL": round(g_cl, 6),
        "random_block_hard_nll_mean": round(rand_nll_mean, 6),
        "random_block_hard_nll_std": round(rand_nll_std, 6),
        "gate_utility": round(gate_utility, 6),
    }
    if dense_nll_oom:
        result["dense_nll"] = None
        result["dense_ppl"] = None
        result["dense_nll_skip_reason"] = (
            "CUDA OutOfMemoryError in evaluate_dense_nll's dense-attention "
            "causal-mask materialization (T*T*4 bytes, independent of "
            "loss_chunk_size) -- dense_nll is already known to be invalid "
            "at long context regardless, see file-level notes; skipped "
            "rather than letting it crash the whole evaluation."
        )
    else:
        result["dense_nll"] = round(dense_nll, 6)
        result["dense_ppl"] = round(math.exp(dense_nll), 2)
    return result


# ===================================================================
# Data loading
# ===================================================================

class CyclingTokenDataset(Dataset):
    """Fixed-length token sequences, cycling through the corpus."""

    def __init__(self, tokens: torch.Tensor, seq_len: int):
        self.tokens = tokens
        self.seq_len = seq_len
        self.n_seqs = (len(tokens) - 1) // seq_len

    def __len__(self) -> int:
        return self.n_seqs

    def __getitem__(self, idx: int) -> torch.Tensor:
        idx = idx % self.n_seqs
        start = idx * self.seq_len
        return self.tokens[start: start + self.seq_len + 1]


def _load_tokens_from_dir(data_dir: str,
                          split_names: list[str]) -> torch.Tensor | None:
    """Try to load tokens from various formats in *data_dir*."""
    data_path = Path(data_dir)
    for name in split_names:
        p = data_path / name
        if not p.exists():
            continue
        if name.endswith(".pt"):
            data = torch.load(p, weights_only=True)
            if isinstance(data, dict) and "input_ids" in data:
                return data["input_ids"].reshape(-1).long()
            return (data.reshape(-1) if data.dim() > 1 else data).long()
        if name.endswith(".npy"):
            return torch.from_numpy(np.load(p)).long()
    return None


def _detect_corpus(data_dir: str) -> str:
    """Auto-detect corpus type from directory contents."""
    p = Path(data_dir)
    if (p / "fineweb_edu_shards").exists() or (p / "fineweb_edu_eval").exists():
        return "fineweb-edu"
    return "wikitext-103"


def load_training_data(data_dir: str, seq_len: int,
                       seed: int = 0, rank: int = 0,
                       world_size: int = 1) -> CyclingTokenDataset:
    corpus = _detect_corpus(data_dir)
    if _HAS_DATA_LOADING:
        ds = load_corpus(corpus, data_dir, seq_len=seq_len,
                         split="train", cycling=True, seed=seed,
                         rank=rank, world_size=world_size)
        if ds is not None:
            n = len(ds) if hasattr(ds, '__len__') else '(streaming)'
            print(f"  Loaded training data via data_loading [{corpus}] "
                  f"({n} sequences)")
            return ds

    tokens = _load_tokens_from_dir(
        data_dir,
        ["wt103_train_tokens.pt", "wt103_train_tokens.npy", "train.pt"],
    )
    if tokens is None:
        raise FileNotFoundError(f"No training data found in {data_dir}")
    print(f"  Loaded {len(tokens):,} training tokens from {data_dir}")
    return CyclingTokenDataset(tokens, seq_len)


def load_val_data(data_dir: str, seq_len: int, rank: int = 0,
                  world_size: int = 1) -> CyclingTokenDataset:
    """Load the validation set.

    NOTE: rank/world_size are accepted for call-site symmetry with
    load_training_data but are deliberately NOT forwarded to load_corpus.
    Only the master rank ever calls this (see the `is_master` guard around
    load_val_data in train_experiment), so sharding the validation set
    across ranks would make rank 0 evaluate on a 1/world_size slice while
    reporting it as the full validation NLL -- silently changing what the
    metric means and breaking comparability with the existing
    world_size=1 numbers already in the paper. Validation therefore always
    loads the whole set (rank=0, world_size=1) regardless of the DDP
    world_size the caller is running under.
    """
    corpus = _detect_corpus(data_dir)
    if _HAS_DATA_LOADING:
        ds = load_corpus(corpus, data_dir, seq_len=seq_len,
                         split="validation", rank=0, world_size=1)
        if ds is not None:
            print(f"  Loaded validation data via data_loading "
                  f"({len(ds)} sequences)")
            return ds

    tokens = _load_tokens_from_dir(
        data_dir,
        ["wt103_val_tokens.pt", "wt103_val_tokens.npy", "validation.pt"],
    )
    if tokens is None:
        raise FileNotFoundError(f"No validation data found in {data_dir}")
    print(f"  Loaded {len(tokens):,} val tokens from {data_dir}")
    return CyclingTokenDataset(tokens, seq_len)


# ===================================================================
# Contemporary CLHR: inline hard forward with current gates (detached)
# ===================================================================

def _contemporary_clhr_forward(
    raw_model: HierarchicalSparseTransformer,
    x: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    loss_chunk_size: int = 0,
) -> torch.Tensor:
    """Contemporary closed-loop hard replay forward pass.

    Two-phase approach for speed:
      1. Compute per-layer hard block selections from the current hidden
         state (no grad, closed-loop: each layer's gate sees the hard-path
         hidden state from the previous layer).
      2. Pass the block selections through the model's own compiled forward
         method, which uses flex_attention for true block-sparse execution.

    This is mathematically equivalent to the manual unrolled version but
    runs through the same compiled code path as the soft forward, getting
    the same flex_attention speedup.
    """
    B, T = x.shape
    block_size = raw_model.block_size
    top_k = raw_model.top_k_blocks
    n_blocks = (T + block_size - 1) // block_size

    # Phase 1: compute per-layer block selections via closed-loop rollout
    # (each layer's gate sees the hard-path hidden state)
    per_layer_selections = []
    with torch.no_grad():
        h = raw_model.tok_emb(x)
        h = raw_model.drop(h)
        # Lazy, same reasoning as HierarchicalSparseTransformer.forward:
        # this rollout always passes forced_block_selection, which (for
        # attention_impl="flex") hits the flex-only branch in
        # HierarchicalSparseAttention.forward that never reads
        # causal_mask -- so building it eagerly here was a pure-waste
        # dense (1, 1, T, T) allocation on every training step.
        causal_mask = lambda: raw_model._make_causal_mask(T, device)  # noqa: E731

        block_causal = torch.ones(
            n_blocks, n_blocks, device=device, dtype=torch.bool,
        ).tril()

        for layer in raw_model.layers:
            normed = layer.attn_norm(h)
            attn = layer.attn

            # Gate from current hard-path hidden state
            block_scores = attn.block_gate.compute_block_scores(normed,
                                                                n_blocks)
            actual_k = min(top_k, n_blocks)
            _, topk_idx = torch.topk(block_scores, actual_k, dim=-1)
            hard_block = torch.zeros_like(block_scores).scatter_(
                -1, topk_idx, 1.0,
            ) * block_causal.float()

            per_layer_selections.append(hard_block)

            # Propagate hard-path hidden state for the next layer's gate.
            # Routed through attn.forward's forced_block_selection branch
            # (rather than a hand-rolled dense-mask SDPA call) so this
            # no-grad rollout also gets the flex_attention block-sparse
            # kernel -- this loop runs on every training step under CLHR
            # and was previously always-dense regardless of the flex fix
            # above.
            out, _ = attn(
                normed, raw_model.rope, causal_mask,
                forced_block_selection=hard_block,
            )
            h = h + out
            h = h + layer.ff(layer.ff_norm(h))

    # Phase 2: forward pass WITH gradients using the pre-computed block
    # selections, through the model's compiled forward (flex_attention path)
    if loss_chunk_size > 0:
        hidden, _ = raw_model(
            x,
            forced_block_selections=per_layer_selections,
            use_checkpoint=True,
            skip_lm_head=True,
        )
        hard_loss = chunked_lm_head_loss(
            hidden, y, raw_model.lm_head, loss_chunk_size,
        )
    else:
        logits, _ = raw_model(
            x,
            forced_block_selections=per_layer_selections,
            use_checkpoint=True,
        )
        hard_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1),
        )
    return hard_loss


def _contemporary_clhr_forward_checkpointed(
    raw_model: HierarchicalSparseTransformer,
    x: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    loss_chunk_size: int = 0,
) -> torch.Tensor:
    """Contemporary CLHR with two-phase design for flex_attention speedup.

    Phase 1 (no grad): closed-loop rollout to compute per-layer hard block
    selections from the current gate on the hard-path hidden state.

    Phase 2 (with grad): pass the pre-computed block selections through the
    model's own compiled forward, which uses flex_attention for block-sparse
    execution.  Gradient checkpointing is handled by the model forward.
    """
    return _contemporary_clhr_forward(
        raw_model, x, y, device, dtype, loss_chunk_size=loss_chunk_size,
    )


# ===================================================================
# Training
# ===================================================================

def train_experiment(
    condition: str,
    seed: int,
    data_dir: str,
    checkpoint_dir: str,
    output_path: str,
    model_size: str = "small",
    seq_len: int = 4096,
    block_size: int = 64,
    top_k_blocks: int = 8,
    local_window: int = 256,
    total_tokens: int = 2_000_000_000,
    micro_batch: int = 4,
    grad_accum: int | None = None,
    lr: float = 3e-4,
    lambda_rca: float = 1.0,
    attention_impl: str = "bias",
    loss_chunk_size: int = 0,
):
    # ---- DDP setup ----
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    is_master = rank == 0

    if is_master:
        print(f"Device: {device}, dtype: {dtype}, condition: {condition}, "
              f"seed: {seed}, model_size: {model_size}")
        print(f"  DDP: {ddp}, world_size: {world_size}")
        print(f"  seq_len: {seq_len}, block_size: {block_size}, "
              f"top_k_blocks: {top_k_blocks}, local_window: {local_window}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    is_dense = condition == "dense"
    use_rca = condition == "contemporary_closedloop_hard"

    # ---- grad accumulation ----
    if grad_accum is None:
        target_tokens = 65536
        grad_accum = max(1, target_tokens // (micro_batch * seq_len
                                              * world_size))
    tokens_per_step = micro_batch * seq_len * grad_accum * world_size
    total_steps = total_tokens // tokens_per_step

    if is_master:
        print(f"  micro_batch={micro_batch}, grad_accum={grad_accum}, "
              f"world_size={world_size}")
        print(f"  Tokens/step: {tokens_per_step:,}, "
              f"total steps: {total_steps:,}")

    # ---- data ----
    corpus = _detect_corpus(data_dir)
    data_dir_resolved = str(Path(data_dir).resolve())
    train_ds = load_training_data(data_dir, seq_len, seed=seed,
                                  rank=rank, world_size=world_size)
    # Cheap, best-effort token count: only taken when the dataset exposes
    # its underlying token tensor directly (both this module's
    # CyclingTokenDataset and data_loading's non-streaming TokenDataset do).
    # Streaming/sharded datasets have no such attribute, so the field is
    # simply omitted rather than guessed.
    train_tokens_available = None
    _train_tokens_attr = getattr(train_ds, "tokens", None)
    if _train_tokens_attr is not None and hasattr(_train_tokens_attr, "__len__"):
        train_tokens_available = len(_train_tokens_attr)
    is_iterable = isinstance(train_ds, IterableDataset)
    if is_iterable:
        sampler = None
        train_loader = DataLoader(
            train_ds, batch_size=micro_batch, shuffle=False,
            num_workers=0, pin_memory=True, drop_last=True,
        )
    else:
        sampler = (DistributedSampler(train_ds, num_replicas=world_size,
                                      rank=rank, shuffle=True)
                   if ddp else None)
        train_loader = DataLoader(
            train_ds, batch_size=micro_batch, shuffle=(sampler is None),
            sampler=sampler, num_workers=2, pin_memory=True, drop_last=True,
        )

    # ---- model ----
    cfg = MODEL_CONFIGS[model_size]
    model = HierarchicalSparseTransformer(
        **cfg, block_size=block_size, top_k_blocks=top_k_blocks,
        local_window=local_window, max_seq_len=seq_len, dropout=0.1,
        attention_impl=attention_impl,
    ).to(device).to(dtype)

    if is_master:
        print(f"  Parameters: {model.count_parameters() / 1e6:.1f}M")

    # ---- self-describing checkpoint config ----
    # See scripts/eval_hierarchical_checkpoint.py: it hard-fails on a
    # CLI/checkpoint geometry contradiction only when the checkpoint carries
    # this "config" dict. Without it, a wrong --block-size on eval silently
    # produces plausible-but-meaningless numbers.
    checkpoint_config = {
        "model_size": model_size,
        "seq_len": seq_len,
        "block_size": block_size,
        "top_k_blocks": top_k_blocks,
        "local_window": local_window,
        "condition": condition,
        "corpus": corpus,
        "d_model": cfg["d_model"],
        "n_heads": cfg["n_heads"],
        "n_layers": cfg["n_layers"],
        "d_ff": cfg["d_ff"],
        "d_gate": cfg["d_gate"],
    }

    raw_model = model
    use_compile = device.type == "cuda" and hasattr(torch, "compile")
    if use_compile:
        model = torch.compile(model)
        if is_master:
            print("  torch.compile enabled")

    if ddp:
        model = DDP(model, device_ids=[local_rank],
                    gradient_as_bucket_view=True)

    # ---- optimizer ----
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=0.1, betas=(0.9, 0.95),
    )
    warmup_steps = 2000

    def cosine_lr(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, cosine_lr)

    # ---- checkpointing ----
    tag = f"hierarchical_{model_size}_{condition}_s{seed}"
    ckpt_path = Path(checkpoint_dir) / tag
    ckpt_path.mkdir(parents=True, exist_ok=True)

    resume_file = ckpt_path / "latest.pt"
    global_step = 0
    tokens_seen = 0
    if resume_file.exists():
        ckpt = torch.load(resume_file, map_location=device)
        raw_model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        global_step = ckpt["step"]
        tokens_seen = ckpt.get("tokens_seen", global_step * tokens_per_step)
        if is_master:
            print(f"  Resumed from step {global_step}, "
                  f"{tokens_seen / 1e9:.2f}B tokens")
        # Advance the data stream past already-consumed sequences so a
        # resume does not re-train on the same shards.  Uses tokens_seen
        # rather than step count so it is invariant to micro_batch changes.
        if is_iterable and hasattr(train_ds, "skip_sequences"):
            train_ds.skip_sequences = tokens_seen // seq_len
            if is_master:
                print(f"  Skipping {train_ds.skip_sequences:,} already-seen "
                      f"sequences in the data stream")

    train_iter = iter(train_loader)
    t0 = time.time()
    running_loss = 0.0
    running_count = 0

    # ---- sparsity tracking ----
    target_sparsity = 0.875
    lambda_sparse = 1.0

    # ---- training loop ----
    while global_step < total_steps:
        model.train()
        optimizer.zero_grad()
        accum_loss = 0.0

        for _accum_step in range(grad_accum):
            try:
                batch = next(train_iter)
            except StopIteration:
                if sampler is not None:
                    sampler.set_epoch(global_step)
                train_iter = iter(train_loader)
                batch = next(train_iter)

            if isinstance(batch, dict):
                batch = batch["input_ids"]
            batch = batch.to(device)
            x, y = batch[:, :-1], batch[:, 1:]

            with torch.amp.autocast("cuda", dtype=dtype):
                # ---- primary (soft) forward ----
                if loss_chunk_size > 0:
                    hidden, block_scores_list = model(
                        x, use_checkpoint=True, dense_mode=is_dense,
                        skip_lm_head=True,
                    )
                    lm_loss = chunked_lm_head_loss(
                        hidden, y, raw_model.lm_head, loss_chunk_size,
                    )
                else:
                    logits, block_scores_list = model(
                        x, use_checkpoint=True, dense_mode=is_dense,
                    )
                    lm_loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                    )

                # ---- sparsity regularisation on block scores ----
                if is_dense:
                    sparsity_loss = torch.tensor(0.0, device=device)
                else:
                    sparsity_vals = []
                    for layer in raw_model.layers:
                        bs = layer.attn.last_soft_block_scores
                        if bs is not None:
                            # sigmoid(scores) gives the soft gate values
                            soft_vals = torch.sigmoid(bs)
                            sparsity_vals.append(1.0 - soft_vals.mean())
                    if sparsity_vals:
                        avg_sparsity = torch.stack(sparsity_vals).mean()
                        gap = F.relu(
                            torch.tensor(target_sparsity, device=device)
                            - avg_sparsity
                        )
                        sparsity_loss = gap ** 2
                    else:
                        sparsity_loss = torch.tensor(0.0, device=device)

                soft_loss = lm_loss + lambda_sparse * sparsity_loss

                # ---- Contemporary CLHR dual-loss ----
                if use_rca and lambda_rca > 0:
                    hard_loss = _contemporary_clhr_forward_checkpointed(
                        raw_model, x, y, device, dtype,
                        loss_chunk_size=loss_chunk_size,
                    )
                    # total = (soft + lambda * hard) / (1 + lambda)
                    loss = (soft_loss + lambda_rca * hard_loss) / (
                        1.0 + lambda_rca
                    )
                else:
                    loss = soft_loss

                loss = loss / grad_accum

            loss.backward()
            accum_loss += lm_loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        global_step += 1
        tokens_seen += tokens_per_step
        running_loss += accum_loss / grad_accum
        running_count += 1

        # ---- logging ----
        if is_master and global_step % 100 == 0:
            avg = running_loss / running_count
            elapsed = time.time() - t0
            tps = tokens_seen / elapsed if elapsed > 0 else 0
            mem = (torch.cuda.max_memory_allocated() / 1e9
                   if device.type == "cuda" else 0)
            print(f"  step {global_step:>6}/{total_steps}: loss={avg:.4f} "
                  f"tokens={tokens_seen / 1e9:.2f}B tps={tps:.0f} "
                  f"mem={mem:.1f}GB ({elapsed:.0f}s)", flush=True)
            running_loss = 0.0
            running_count = 0

        # ---- milestone checkpoints ----
        if is_master:
            for ct in CHECKPOINT_TOKEN_MILESTONES:
                if tokens_seen >= ct and tokens_seen - tokens_per_step < ct:
                    if ct >= 1_000_000_000:
                        label = f"{ct // 1_000_000_000}B"
                    else:
                        label = f"{ct // 1_000_000}M"
                    save_path = ckpt_path / f"tokens_{label}.pt"
                    torch.save({
                        "model": raw_model.state_dict(),
                        "step": global_step,
                        "tokens": tokens_seen,
                        "config": checkpoint_config,
                    }, save_path)
                    print(f"    [checkpoint saved: {save_path.name}]",
                          flush=True)
                    break

            # Periodic resume checkpoint. Tightened 1000 -> 100 (2026-09-21)
            # after two crashes each lost up to ~600-800 steps of progress
            # since the last save (LONGCTX_128K_K4 and LONGCTX_64K_K4_S123 --
            # see chat history / results/RUN_LEDGER.md). Overhead is one
            # extra ~400MB torch.save every 100 steps instead of every 1000,
            # negligible next to a multi-hour run.
            if global_step % 100 == 0:
                torch.save({
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "step": global_step,
                    "tokens_seen": tokens_seen,
                    "config": checkpoint_config,
                }, resume_file)

        if ddp:
            dist.barrier()

    # ---- final checkpoint + evaluation ----
    train_time = time.time() - t0

    if is_master:
        torch.save({
            "model": raw_model.state_dict(),
            "step": global_step,
            "tokens_seen": tokens_seen,
            "config": checkpoint_config,
        }, ckpt_path / "final.pt")

        print("\n  Final evaluation...")
        val_ds = load_val_data(data_dir, seq_len, rank=rank,
                               world_size=world_size)
        eval_batch = max(1, micro_batch // 4)
        val_loader = DataLoader(
            val_ds, batch_size=eval_batch, shuffle=False,
            num_workers=0, drop_last=True,
        )
        metrics = run_full_evaluation(
            raw_model, val_loader, device, k_blocks=top_k_blocks,
            loss_chunk_size=loss_chunk_size,
        )

        # ---- hardware info ----
        hw_info = {}
        if device.type == "cuda":
            hw_info["gpu"] = torch.cuda.get_device_name(device)
            hw_info["gpu_count"] = world_size
            hw_info["peak_memory_gb"] = round(
                torch.cuda.max_memory_allocated() / 1e9, 2,
            )

        results = {
            "condition": condition,
            "seed": seed,
            "model_size": model_size,
            "d_model": cfg["d_model"],
            "n_heads": cfg["n_heads"],
            "n_layers": cfg["n_layers"],
            "d_ff": cfg["d_ff"],
            "d_gate": cfg["d_gate"],
            "seq_len": seq_len,
            "block_size": block_size,
            "top_k_blocks": top_k_blocks,
            "local_window": local_window,
            "lambda_rca": lambda_rca,
            "model_params": raw_model.count_parameters(),
            "corpus": corpus,
            "data_dir": data_dir_resolved,
            "total_tokens": tokens_seen,
            "total_steps": global_step,
            "training_time_seconds": round(train_time, 1),
            **metrics,
            "hardware": hw_info,
        }
        if train_tokens_available is not None:
            results["train_tokens_available"] = train_tokens_available
            results["epochs_over_corpus"] = tokens_seen / train_tokens_available

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Saved to {output_path}")
        for k, v in metrics.items():
            print(f"    {k}: {v}")

    if ddp:
        dist.destroy_process_group()


# ===================================================================
# CLI
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Hierarchical sparse-attention transformer with CLHR",
    )
    parser.add_argument("--condition", choices=CONDITIONS,
                        default="standard")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output", type=str,
                        default="results/hierarchical.json")
    parser.add_argument("--checkpoint-dir", type=str,
                        default="./ckpts_hierarchical")
    parser.add_argument("--model-size",
                        choices=list(MODEL_CONFIGS.keys()),
                        default="small")
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--top-k-blocks", type=int, default=8)
    parser.add_argument("--local-window", type=int, default=256)
    parser.add_argument("--total-tokens", type=int,
                        default=2_000_000_000)
    parser.add_argument("--lambda-rca", type=float, default=1.0,
                        help="CLHR auxiliary loss weight")
    parser.add_argument("--micro-batch", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--attention-impl", choices=list(ATTENTION_IMPLS),
                        default="bias",
                        help="'bias': dense additive-mask SDPA (default, "
                             "unchanged production behavior). 'flex': "
                             "build_block_mask_direct + flex_attention "
                             "(src/flex_block_mask.py), no dense (B,H,T,T) "
                             "materialization; requires CUDA in practice.")
    parser.add_argument("--loss-chunk-size", type=int, default=0,
                        help="0 (default): unchanged behavior -- compute "
                             "the full (B,T,vocab) logits tensor and take "
                             "F.cross_entropy over it in one shot. >0: "
                             "fuse the vocab projection with the loss and "
                             "checkpoint it per chunk of this many "
                             "positions, so the full logits tensor (and "
                             "its fp32 upcast) is never materialised at "
                             "once. Mathematically identical loss/"
                             "gradients; fixes OOM at long context. "
                             "Applies to both the soft and CLHR hard "
                             "losses.")
    args = parser.parse_args()

    train_experiment(
        condition=args.condition,
        seed=args.seed,
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        output_path=args.output,
        model_size=args.model_size,
        seq_len=args.seq_len,
        block_size=args.block_size,
        top_k_blocks=args.top_k_blocks,
        local_window=args.local_window,
        total_tokens=args.total_tokens,
        micro_batch=args.micro_batch,
        grad_accum=args.grad_accum,
        lr=args.lr,
        lambda_rca=args.lambda_rca,
        attention_impl=args.attention_impl,
        loss_chunk_size=args.loss_chunk_size,
    )


if __name__ == "__main__":
    main()
