"""Block-level routed sparse-attention transformer for long-context CLHR.

Instead of computing per-token gate scores Q_g K_g^T in O(T^2), pools tokens
into blocks and routes at the block level in O((T/B_s)^2).  Uses RoPE instead
of learned positional embeddings for better long-context extrapolation.

Architecture:
  - BlockGatedSparseAttention: per-token W_gq/W_gk projections, mean-pooled to
    block level, soft sigmoid routing during training, hard top-k at eval.
  - RotaryPositionalEncoding: standard RoPE applied to Q and K.
  - BlockSparseTransformer: pre-norm decoder-only, tied embeddings, two sizes
    (small=~31M, medium=~300M).

Conditions:
  standard                  -- soft block-level gating
  coherent_closedloop_hard  -- CLHR with historical gate weights, block routing
  dense                     -- no routing, causal-only baseline

Usage:
    python src/sparse_attention_block_router.py \
        --condition standard --seed 42 \
        --data-dir ./wikitext103_cache \
        --checkpoint-dir ./ckpts_block_router \
        --output results/block_router_standard_s42.json \
        --model-size medium --seq-len 2048 --block-size 64 --top-k-blocks 16

    python src/sparse_attention_block_router.py \
        --condition coherent_closedloop_hard --seed 42 \
        --data-dir ./wikitext103_cache \
        --checkpoint-dir ./ckpts_block_router \
        --output results/block_router_clhr_s42.json \
        --model-size small --seq-len 2048 --block-size 64 --top-k-blocks 8 \
        --lambda-rca 1.0
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
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


CONDITIONS = ["standard", "coherent_closedloop_hard", "dense"]

MODEL_CONFIGS = {
    "small": dict(vocab_size=50257, d_model=512, n_heads=8, n_layers=12,
                  d_ff=2048, d_gate=32),
    "medium": dict(vocab_size=50257, d_model=1024, n_heads=16, n_layers=20,
                   d_ff=4096, d_gate=32),
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


# ===================================================================
# Block-Gated Sparse Attention
# ===================================================================

class BlockGatedSparseAttention(nn.Module):
    """Sparse attention with block-level routing.

    Per-token gate projections (W_gq, W_gk) are mean-pooled to block
    representations before computing the block-to-block routing matrix.
    During training the sigmoid scores are used as soft masks; at
    deployment / CLHR eval a hard top-k block mask is applied.
    """

    def __init__(self, d_model: int, n_heads: int, d_gate: int,
                 block_size: int, top_k_blocks: int, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_gate = d_gate
        self.block_size = block_size
        self.top_k_blocks = top_k_blocks

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.W_gq = nn.Linear(d_model, n_heads * d_gate, bias=False)
        self.W_gk = nn.Linear(d_model, n_heads * d_gate, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.last_soft_block_mask: torch.Tensor | None = None

    # ----- helpers -----

    def _expand_block_mask(self, block_mask: torch.Tensor,
                           T: int) -> torch.Tensor:
        """Expand (B, H, n_blocks_q, n_blocks_k) -> (B, H, T, T)."""
        token_mask = block_mask.repeat_interleave(
            self.block_size, dim=2
        ).repeat_interleave(self.block_size, dim=3)
        return token_mask[:, :, :T, :T]

    def _pool_gate_vectors(self, gq: torch.Tensor, gk: torch.Tensor,
                           n_blocks: int):
        """Mean-pool per-token gate vectors to block level.

        Args:
            gq, gk: (B, H, T, d_gate)
            n_blocks: number of blocks (after ceil-division)
        Returns:
            gq_blocks, gk_blocks: (B, H, n_blocks, d_gate)
        """
        B, H, T, dg = gq.shape
        pad_len = (self.block_size - T % self.block_size) % self.block_size
        if pad_len > 0:
            gq = F.pad(gq, (0, 0, 0, pad_len))
            gk = F.pad(gk, (0, 0, 0, pad_len))
        gq_blocks = gq.view(B, H, n_blocks, self.block_size, dg).mean(dim=3)
        gk_blocks = gk.view(B, H, n_blocks, self.block_size, dg).mean(dim=3)
        return gq_blocks, gk_blocks

    # ----- forward -----

    def forward(self, x: torch.Tensor, rope: RotaryPositionalEncoding,
                causal_mask: torch.Tensor,
                forced_block_mask: torch.Tensor | None = None,
                dense_mode: bool = False):
        """
        Args:
            x: (B, T, D) hidden states.
            rope: RotaryPositionalEncoding module (applied to Q/K).
            causal_mask: (1, 1, T, T) float mask (0 = attend, -inf = masked).
            forced_block_mask: optional (B, H, n_blocks, n_blocks) hard mask
                from a historical gate snapshot (CLHR training).
            dense_mode: if True, only causal mask is used (no routing).

        Returns:
            output: (B, T, D)
            soft_block_mask: (B, H, n_blocks, n_blocks) or None
        """
        B, T, D = x.shape
        n_h, d_h, d_g = self.n_heads, self.d_head, self.d_gate
        n_blocks = (T + self.block_size - 1) // self.block_size

        # Q / K / V
        q = self.W_q(x).view(B, T, n_h, d_h).transpose(1, 2)  # (B,H,T,d_h)
        k = self.W_k(x).view(B, T, n_h, d_h).transpose(1, 2)
        v = self.W_v(x).view(B, T, n_h, d_h).transpose(1, 2)

        # Apply RoPE to Q and K
        q = rope(q, T)
        k = rope(k, T)

        soft_block_mask = None

        if dense_mode:
            # Pure causal, no routing
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=causal_mask.to(q.dtype),
                dropout_p=self.dropout.p if self.training else 0.0,
            )
            self.last_soft_block_mask = None

        elif forced_block_mask is not None:
            # CLHR: use provided (historical) hard block mask
            token_mask = self._expand_block_mask(forced_block_mask, T)
            gate_bias = torch.where(
                token_mask > 0,
                torch.zeros_like(token_mask),
                torch.full_like(token_mask, float("-inf")),
            )
            attn_mask = gate_bias + causal_mask
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask.to(q.dtype),
                dropout_p=self.dropout.p if self.training else 0.0,
            )
            self.last_soft_block_mask = None

        else:
            # Standard soft block routing (training path)
            gq = self.W_gq(x).view(B, T, n_h, d_g).transpose(1, 2)
            gk = self.W_gk(x).view(B, T, n_h, d_g).transpose(1, 2)
            gq_blocks, gk_blocks = self._pool_gate_vectors(gq, gk, n_blocks)

            # Block routing scores: (B, H, n_blocks, n_blocks)
            block_scores = (gq_blocks @ gk_blocks.transpose(-2, -1)
                            / math.sqrt(d_g))

            # Block causal: query block i may attend to key block j iff j <= i
            block_causal = torch.ones(
                n_blocks, n_blocks, device=x.device, dtype=torch.bool
            ).tril()
            block_scores = block_scores.masked_fill(~block_causal, float("-inf"))

            soft_block_mask = torch.sigmoid(block_scores)
            self.last_soft_block_mask = soft_block_mask.detach()

            # Expand to token level and apply log-additive gating
            token_mask = self._expand_block_mask(soft_block_mask, T)
            gate_bias = torch.log(token_mask.clamp(min=1e-6))
            attn_mask = gate_bias + causal_mask
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask.to(q.dtype),
                dropout_p=self.dropout.p if self.training else 0.0,
            )

        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.W_o(out), soft_block_mask

    # ----- deployment helper -----

    @torch.no_grad()
    def compute_hard_block_mask(self, x: torch.Tensor,
                                k_blocks: int | None = None) -> torch.Tensor:
        """Compute hard top-k block mask for deployment / CLHR evaluation.

        Args:
            x: (B, T, D) hidden states (pre-norm output).
            k_blocks: number of key blocks each query block may attend to.

        Returns:
            hard_block_mask: (B, H, n_blocks, n_blocks) binary.
        """
        if k_blocks is None:
            k_blocks = self.top_k_blocks
        B, T, D = x.shape
        n_blocks = (T + self.block_size - 1) // self.block_size

        gq = self.W_gq(x).view(B, T, self.n_heads, self.d_gate).transpose(1, 2)
        gk = self.W_gk(x).view(B, T, self.n_heads, self.d_gate).transpose(1, 2)
        gq_blocks, gk_blocks = self._pool_gate_vectors(gq, gk, n_blocks)

        block_scores = (gq_blocks @ gk_blocks.transpose(-2, -1)
                        / math.sqrt(self.d_gate))
        block_causal = torch.ones(
            n_blocks, n_blocks, device=x.device, dtype=torch.bool
        ).tril()
        block_scores = block_scores.masked_fill(~block_causal, float("-inf"))

        actual_k = min(k_blocks, n_blocks)
        _, topk_idx = torch.topk(block_scores, actual_k, dim=-1)
        hard_block_mask = torch.zeros_like(block_scores).scatter_(
            -1, topk_idx, 1.0
        )
        hard_block_mask = hard_block_mask * block_causal.float()
        return hard_block_mask


# ===================================================================
# Transformer Block (pre-norm, block-sparse attention + FFN)
# ===================================================================

class BlockSparseTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, d_gate: int,
                 block_size: int, top_k_blocks: int, dropout: float = 0.1):
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = BlockGatedSparseAttention(
            d_model, n_heads, d_gate, block_size, top_k_blocks, dropout,
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
                causal_mask: torch.Tensor,
                forced_block_mask: torch.Tensor | None = None,
                dense_mode: bool = False):
        attn_out, soft_mask = self.attn(
            self.attn_norm(x), rope, causal_mask,
            forced_block_mask=forced_block_mask, dense_mode=dense_mode,
        )
        x = x + attn_out
        x = x + self.ff(self.ff_norm(x))
        return x


# ===================================================================
# Block-Sparse Transformer (full model)
# ===================================================================

class BlockSparseTransformer(nn.Module):
    def __init__(self, vocab_size: int = 50257, d_model: int = 1024,
                 n_heads: int = 16, n_layers: int = 20, d_ff: int = 4096,
                 d_gate: int = 32, block_size: int = 64,
                 top_k_blocks: int = 16, max_seq_len: int = 4096,
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_gate = d_gate
        self.block_size = block_size
        self.top_k_blocks = top_k_blocks
        self.max_seq_len = max_seq_len

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.rope = RotaryPositionalEncoding(
            d_model // n_heads, max_seq_len=max_seq_len,
        )
        self.drop = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            BlockSparseTransformerBlock(
                d_model, n_heads, d_ff, d_gate, block_size, top_k_blocks,
                dropout,
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
                forced_block_masks: list[torch.Tensor] | None = None,
                use_checkpoint: bool = False,
                dense_mode: bool = False):
        """
        Args:
            input_ids: (B, T) token ids.
            forced_block_masks: optional list (per layer) of
                (B, H, n_blocks, n_blocks) hard block masks.
            use_checkpoint: gradient checkpointing.
            dense_mode: skip routing entirely.

        Returns:
            logits: (B, T, vocab_size)
            all_block_masks: list of (B, H, n_blocks, n_blocks) soft masks
                (empty when dense or forced).
        """
        B, T = input_ids.shape
        device = input_ids.device

        x = self.tok_emb(input_ids)
        x = self.drop(x)

        causal_mask = self._make_causal_mask(T, device)

        all_block_masks: list[torch.Tensor] = []
        for i, layer in enumerate(self.layers):
            fm = forced_block_masks[i] if forced_block_masks is not None else None

            if use_checkpoint and self.training:
                # grad_checkpoint requires a function, not a method with
                # keyword-only args, so wrap in a closure.
                def _run_layer(_x, _layer=layer, _rope=self.rope,
                               _cm=causal_mask, _fm=fm, _dm=dense_mode):
                    return _layer(_x, _rope, _cm,
                                  forced_block_mask=_fm, dense_mode=_dm)
                x = grad_checkpoint(_run_layer, x, use_reentrant=False)
            else:
                x = layer(x, self.rope, causal_mask,
                          forced_block_mask=fm, dense_mode=dense_mode)

            m = layer.attn.last_soft_block_mask
            if m is not None:
                all_block_masks.append(m)

        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits, all_block_masks

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ===================================================================
# CLHR: closed-loop hard-replay forward
# ===================================================================

def forward_closed_loop_block_hard(
    model: BlockSparseTransformer,
    input_ids: torch.Tensor,
    gate_snapshots: list[dict],
    k_blocks: int,
) -> torch.Tensor:
    """CLHR forward with historical block-level gates.

    At each layer the *historical* W_gq / W_gk weights produce a hard block
    mask from the *current* hidden state.  The current model's Q/K/V/O
    weights receive gradients; the gate mask is stop-gradient.

    Args:
        model: BlockSparseTransformer (unwrapped).
        input_ids: (B, T).
        gate_snapshots: list of per-layer dicts with keys
            ``W_gq`` and ``W_gk`` (weight tensors).
        k_blocks: number of top-k blocks.

    Returns:
        logits: (B, T, vocab_size).
    """
    B, T = input_ids.shape
    device = input_ids.device
    n_heads = model.n_heads
    d_gate = model.d_gate
    block_size = model.block_size

    x = model.tok_emb(input_ids)
    x = model.drop(x)

    causal_mask = model._make_causal_mask(T, device)
    n_blocks = (T + block_size - 1) // block_size

    for layer_idx, layer in enumerate(model.layers):
        h = layer.attn_norm(x)

        # --- historical hard block mask (no grad) ---
        with torch.no_grad():
            hist_W_gq = gate_snapshots[layer_idx]["W_gq"]
            hist_W_gk = gate_snapshots[layer_idx]["W_gk"]

            gq = F.linear(h, hist_W_gq).view(B, T, n_heads, d_gate).transpose(1, 2)
            gk = F.linear(h, hist_W_gk).view(B, T, n_heads, d_gate).transpose(1, 2)

            # Pool into blocks
            pad_len = (block_size - T % block_size) % block_size
            if pad_len > 0:
                gq_padded = F.pad(gq, (0, 0, 0, pad_len))
                gk_padded = F.pad(gk, (0, 0, 0, pad_len))
            else:
                gq_padded = gq
                gk_padded = gk
            gq_blocks = gq_padded.view(
                B, n_heads, n_blocks, block_size, d_gate
            ).mean(dim=3)
            gk_blocks = gk_padded.view(
                B, n_heads, n_blocks, block_size, d_gate
            ).mean(dim=3)

            block_scores = (gq_blocks @ gk_blocks.transpose(-2, -1)
                            / math.sqrt(d_gate))
            block_causal = torch.ones(
                n_blocks, n_blocks, device=device, dtype=torch.bool,
            ).tril()
            block_scores = block_scores.masked_fill(~block_causal, float("-inf"))

            actual_k = min(k_blocks, n_blocks)
            _, topk_idx = torch.topk(block_scores, actual_k, dim=-1)
            hard_block_mask = torch.zeros_like(block_scores).scatter_(
                -1, topk_idx, 1.0,
            )
            hard_block_mask = hard_block_mask * block_causal.float()

        # --- forward with forced hard block mask (grads through Q/K/V) ---
        attn_out, _ = layer.attn(
            h, model.rope, causal_mask,
            forced_block_mask=hard_block_mask,
        )
        x = x + attn_out
        x = x + layer.ff(layer.ff_norm(x))

    logits = model.lm_head(model.final_norm(x))
    return logits


# ===================================================================
# Evaluation helpers
# ===================================================================

@torch.no_grad()
def eval_closed_loop_block_hard(
    model: BlockSparseTransformer,
    loader: DataLoader,
    device: torch.device,
    k_blocks: int,
    max_batches: int = 100,
) -> float:
    """Closed-loop block-hard NLL: model's own current gates, hard top-k."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    block_size = model.block_size
    n_heads = model.n_heads
    d_gate = model.d_gate

    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        if isinstance(batch, dict):
            batch = batch["input_ids"]
        batch = batch.to(device)
        x_ids, y = batch[:, :-1], batch[:, 1:]
        B, T = x_ids.shape

        # Embed
        x = model.tok_emb(x_ids)
        x = model.drop(x)
        causal_mask = model._make_causal_mask(T, device)
        n_blocks = (T + block_size - 1) // block_size

        for layer in model.layers:
            h = layer.attn_norm(x)

            # Current-gate hard block mask
            hard_block_mask = layer.attn.compute_hard_block_mask(
                h, k_blocks=k_blocks,
            )

            attn_out, _ = layer.attn(
                h, model.rope, causal_mask,
                forced_block_mask=hard_block_mask,
            )
            x = x + attn_out
            x = x + layer.ff(layer.ff_norm(x))

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
    model: BlockSparseTransformer,
    loader: DataLoader,
    device: torch.device,
    k_blocks: int,
    max_batches: int = 50,
) -> float:
    """Random block selection NLL (control for gate utility)."""
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

        x = model.tok_emb(x_ids)
        x = model.drop(x)
        causal_mask = model._make_causal_mask(T, device)
        n_blocks = (T + block_size - 1) // block_size

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
            rand_mask = torch.zeros_like(rand_scores).scatter_(
                -1, topk_idx, 1.0,
            )
            rand_mask = rand_mask * block_causal.float()

            attn_out, _ = layer.attn(
                h, model.rope, causal_mask,
                forced_block_mask=rand_mask,
            )
            x = x + attn_out
            x = x + layer.ff(layer.ff_norm(x))

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
    model: BlockSparseTransformer,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 100,
    dense_mode: bool = False,
) -> float:
    """Native soft NLL (or dense NLL when dense_mode=True)."""
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
        logits, _ = model(x, use_checkpoint=False, dense_mode=dense_mode)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1),
            reduction="sum",
        )
        total_loss += loss.item()
        total_tokens += y.numel()
    return total_loss / total_tokens


def run_full_evaluation(
    model: BlockSparseTransformer,
    val_loader: DataLoader,
    device: torch.device,
    k_blocks: int = 16,
) -> dict:
    """Full evaluation suite: native, closed-loop hard, random, G_CL, utility."""
    native_nll = evaluate_nll(model, val_loader, device)

    cl_hard_nll = eval_closed_loop_block_hard(
        model, val_loader, device, k_blocks=k_blocks,
    )

    # Random block hard (average over 5 seeds for stability)
    rand_nlls = []
    for _ in range(5):
        rand_nlls.append(
            eval_random_block_hard(
                model, val_loader, device, k_blocks=k_blocks,
            )
        )
    rand_nll_mean = float(np.mean(rand_nlls))
    rand_nll_std = float(np.std(rand_nlls))

    g_cl = cl_hard_nll - native_nll
    gate_utility = rand_nll_mean - cl_hard_nll

    return {
        "native_nll": round(native_nll, 6),
        "native_ppl": round(math.exp(native_nll), 2),
        "closed_loop_block_hard_nll": round(cl_hard_nll, 6),
        "G_CL": round(g_cl, 6),
        "random_block_hard_nll_mean": round(rand_nll_mean, 6),
        "random_block_hard_nll_std": round(rand_nll_std, 6),
        "gate_utility": round(gate_utility, 6),
    }


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


def load_training_data(data_dir: str, seq_len: int) -> CyclingTokenDataset:
    if _HAS_DATA_LOADING:
        ds = load_corpus("wikitext-103", data_dir, seq_len=seq_len, split="train", cycling=True)
        if ds is not None:
            print(f"  Loaded training data via data_loading ({len(ds)} sequences)")
            return ds

    tokens = _load_tokens_from_dir(
        data_dir,
        ["wt103_train_tokens.pt", "wt103_train_tokens.npy", "train.pt"],
    )
    if tokens is None:
        raise FileNotFoundError(f"No training data found in {data_dir}")
    print(f"  Loaded {len(tokens):,} training tokens from {data_dir}")
    return CyclingTokenDataset(tokens, seq_len)


def load_val_data(data_dir: str, seq_len: int) -> CyclingTokenDataset:
    if _HAS_DATA_LOADING:
        ds = load_corpus("wikitext-103", data_dir, seq_len=seq_len, split="validation")
        if ds is not None:
            print(f"  Loaded validation data via data_loading ({len(ds)} sequences)")
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
# Training
# ===================================================================

def train_experiment(
    condition: str,
    seed: int,
    data_dir: str,
    checkpoint_dir: str,
    output_path: str,
    model_size: str = "medium",
    seq_len: int = 2048,
    block_size: int = 64,
    top_k_blocks: int = 16,
    total_tokens: int = 2_000_000_000,
    micro_batch: int = 16,
    grad_accum: int | None = None,
    lr: float = 3e-4,
    target_sparsity: float = 0.875,
    lambda_rca: float = 1.0,
    lambda_sparse: float = 1.0,
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
              f"top_k_blocks: {top_k_blocks}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    is_dense = condition == "dense"

    # ---- grad accumulation ----
    if grad_accum is None:
        target_tokens = 65536
        grad_accum = max(1, target_tokens // (micro_batch * seq_len * world_size))
    tokens_per_step = micro_batch * seq_len * grad_accum * world_size
    total_steps = total_tokens // tokens_per_step

    if is_master:
        print(f"  micro_batch={micro_batch}, grad_accum={grad_accum}, "
              f"world_size={world_size}")
        print(f"  Tokens/step: {tokens_per_step:,}, total steps: {total_steps:,}")

    # ---- data ----
    train_ds = load_training_data(data_dir, seq_len)
    sampler = (DistributedSampler(train_ds, num_replicas=world_size,
                                  rank=rank, shuffle=True)
               if ddp else None)
    train_loader = DataLoader(
        train_ds, batch_size=micro_batch, shuffle=(sampler is None),
        sampler=sampler, num_workers=2, pin_memory=True, drop_last=True,
    )

    # ---- model ----
    cfg = MODEL_CONFIGS[model_size]
    model = BlockSparseTransformer(
        **cfg, block_size=block_size, top_k_blocks=top_k_blocks,
        max_seq_len=seq_len, dropout=0.1,
    ).to(device).to(dtype)

    if is_master:
        print(f"  Parameters: {model.count_parameters() / 1e6:.1f}M")

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

    # ---- gate snapshots (for CLHR) ----
    use_rca = condition == "coherent_closedloop_hard"
    gate_snapshots: list[dict] = []
    snapshot_interval = 1000
    n_snapshots = 5

    # ---- checkpointing ----
    tag = f"block_router_{model_size}_{condition}_s{seed}"
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
        gate_snapshots = ckpt.get("gate_snapshots", [])
        if is_master:
            print(f"  Resumed from step {global_step}, "
                  f"{tokens_seen / 1e9:.2f}B tokens")

    train_iter = iter(train_loader)
    t0 = time.time()
    running_loss = 0.0
    running_count = 0

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
                # ---- primary forward ----
                logits, block_masks = model(
                    x, use_checkpoint=True, dense_mode=is_dense,
                )
                lm_loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                )

                # ---- sparsity regularisation ----
                if is_dense:
                    sparsity_loss = torch.tensor(0.0, device=device)
                else:
                    sparsity_vals = []
                    for layer in raw_model.layers:
                        m = layer.attn.last_soft_block_mask
                        if m is not None:
                            sparsity_vals.append(1.0 - m.mean())
                    if sparsity_vals:
                        avg_sparsity = torch.stack(sparsity_vals).mean()
                        gap = F.relu(
                            torch.tensor(target_sparsity, device=device)
                            - avg_sparsity
                        )
                        sparsity_loss = gap ** 2
                    else:
                        sparsity_loss = torch.tensor(0.0, device=device)

                loss = lm_loss + lambda_sparse * sparsity_loss

                # ---- CLHR auxiliary loss ----
                if use_rca and gate_snapshots and lambda_rca > 0:
                    snap = min(gate_snapshots, key=lambda s: s["step"])

                    # Closed-loop block-hard replay
                    B_rca, T_rca = x.shape
                    h_rca = raw_model.tok_emb(x)
                    h_rca = raw_model.drop(h_rca)
                    causal_mask = raw_model._make_causal_mask(T_rca, device)
                    n_blocks = (T_rca + block_size - 1) // block_size

                    def closedloop_block_fn(
                        h_in, layer, gs, _causal_mask, _rope,
                        _n_heads=raw_model.n_heads,
                        _d_gate=raw_model.d_gate,
                        _block_size=block_size,
                        _n_blocks=n_blocks,
                        _k_blocks=top_k_blocks,
                    ):
                        normed = layer.attn_norm(h_in)
                        attn = layer.attn

                        # Q / K / V through current weights
                        _B, _T, _D = normed.shape
                        q = attn.W_q(normed).view(
                            _B, _T, _n_heads, attn.d_head,
                        ).transpose(1, 2)
                        k = attn.W_k(normed).view(
                            _B, _T, _n_heads, attn.d_head,
                        ).transpose(1, 2)
                        v = attn.W_v(normed).view(
                            _B, _T, _n_heads, attn.d_head,
                        ).transpose(1, 2)
                        q = _rope(q, _T)
                        k = _rope(k, _T)

                        # Historical gate -> hard block mask (no grad)
                        with torch.no_grad():
                            gq = F.linear(normed, gs["W_gq"]).view(
                                _B, _T, _n_heads, _d_gate,
                            ).transpose(1, 2)
                            gk = F.linear(normed, gs["W_gk"]).view(
                                _B, _T, _n_heads, _d_gate,
                            ).transpose(1, 2)
                            pad_len = (_block_size - _T % _block_size) % _block_size
                            if pad_len > 0:
                                gq_p = F.pad(gq, (0, 0, 0, pad_len))
                                gk_p = F.pad(gk, (0, 0, 0, pad_len))
                            else:
                                gq_p = gq
                                gk_p = gk
                            gq_b = gq_p.view(
                                _B, _n_heads, _n_blocks, _block_size, _d_gate,
                            ).mean(dim=3)
                            gk_b = gk_p.view(
                                _B, _n_heads, _n_blocks, _block_size, _d_gate,
                            ).mean(dim=3)
                            bscores = (gq_b @ gk_b.transpose(-2, -1)
                                       / math.sqrt(_d_gate))
                            bcausal = torch.ones(
                                _n_blocks, _n_blocks,
                                device=h_in.device, dtype=torch.bool,
                            ).tril()
                            bscores = bscores.masked_fill(
                                ~bcausal, float("-inf"),
                            )
                            ak = min(_k_blocks, _n_blocks)
                            _, topk_idx = torch.topk(bscores, ak, dim=-1)
                            hard_bm = torch.zeros_like(bscores).scatter_(
                                -1, topk_idx, 1.0,
                            )
                            hard_bm = hard_bm * bcausal.float()

                        # Expand to token mask
                        token_mask = hard_bm.repeat_interleave(
                            _block_size, dim=2,
                        ).repeat_interleave(_block_size, dim=3)
                        token_mask = token_mask[:, :, :_T, :_T]

                        gate_bias = torch.where(
                            token_mask > 0,
                            torch.zeros_like(token_mask),
                            torch.full_like(token_mask, float("-inf")),
                        )
                        attn_mask = gate_bias + _causal_mask
                        out = F.scaled_dot_product_attention(
                            q, k, v, attn_mask=attn_mask.to(q.dtype),
                        )
                        out = out.transpose(1, 2).contiguous().view(_B, _T, _D)
                        out = attn.W_o(out)

                        h_out = h_in + out
                        h_out = h_out + layer.ff(layer.ff_norm(h_out))
                        return h_out

                    for li, layer in enumerate(raw_model.layers):
                        gs = snap["gate_states"][li]
                        h_rca = grad_checkpoint(
                            closedloop_block_fn,
                            h_rca, layer, gs, causal_mask, raw_model.rope,
                            use_reentrant=False,
                        )

                    logits_rca = raw_model.lm_head(raw_model.final_norm(h_rca))
                    rca_loss = F.cross_entropy(
                        logits_rca.reshape(-1, logits_rca.size(-1)),
                        y.reshape(-1),
                    )
                    loss = loss + lambda_rca * rca_loss

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

        # ---- gate snapshots ----
        if use_rca and global_step % snapshot_interval == 0:
            gate_snapshots.append({
                "step": global_step,
                "gate_states": [{
                    "W_gq": layer.attn.W_gq.weight.clone(),
                    "W_gk": layer.attn.W_gk.weight.clone(),
                } for layer in raw_model.layers],
            })
            if len(gate_snapshots) > n_snapshots:
                gate_snapshots.pop(0)

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
                    }, save_path)
                    print(f"    [checkpoint saved: {save_path.name}]",
                          flush=True)
                    break

            # Periodic resume checkpoint
            if global_step % 1000 == 0:
                torch.save({
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "step": global_step,
                    "tokens_seen": tokens_seen,
                    "gate_snapshots": gate_snapshots,
                }, resume_file)

        if ddp:
            dist.barrier()

    # ---- final checkpoint + evaluation ----
    if is_master:
        torch.save({
            "model": raw_model.state_dict(),
            "step": global_step,
            "tokens_seen": tokens_seen,
        }, ckpt_path / "final.pt")

        print("\n  Final evaluation...")
        val_ds = load_val_data(data_dir, seq_len)
        eval_batch = max(1, micro_batch // 4)
        val_loader = DataLoader(
            val_ds, batch_size=eval_batch, shuffle=False,
            num_workers=0, drop_last=True,
        )
        metrics = run_full_evaluation(
            raw_model, val_loader, device, k_blocks=top_k_blocks,
        )

        results = {
            "condition": condition,
            "seed": seed,
            "model_size": model_size,
            "seq_len": seq_len,
            "block_size": block_size,
            "top_k_blocks": top_k_blocks,
            "lambda_rca": lambda_rca,
            "model_params": raw_model.count_parameters(),
            "total_tokens": tokens_seen,
            "total_steps": global_step,
            **metrics,
        }

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
        description="Block-level routed sparse-attention transformer (CLHR)",
    )
    parser.add_argument("--condition", choices=CONDITIONS, default="standard")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output", type=str,
                        default="results/block_router.json")
    parser.add_argument("--checkpoint-dir", type=str,
                        default="./ckpts_block_router")
    parser.add_argument("--model-size", choices=list(MODEL_CONFIGS.keys()),
                        default="medium")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--top-k-blocks", type=int, default=16)
    parser.add_argument("--total-tokens", type=int, default=2_000_000_000)
    parser.add_argument("--lambda-rca", type=float, default=1.0,
                        help="CLHR auxiliary loss weight")
    parser.add_argument("--micro-batch", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--lr", type=float, default=3e-4)
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
        total_tokens=args.total_tokens,
        micro_batch=args.micro_batch,
        grad_accum=args.grad_accum,
        lr=args.lr,
        lambda_rca=args.lambda_rca,
    )


if __name__ == "__main__":
    main()
