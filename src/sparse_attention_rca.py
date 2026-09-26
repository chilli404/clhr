"""Coherent historical-mask RCA for sparse attention.

Tests the frozen prediction from the routing-granularity reversal:
random masks fail in sparse attention because it is fine-grained internal
routing. Coherent masks from the model's own history should regularize
Q/K/V without destroying input-conditioned computational structure.

Conditions:
  learned_gate           — End-to-end learned sparse gate (published baseline)
  random_gate            — Frozen random projection gates (absorption baseline)
  contemporary_replay    — Dual-loss with current masks (compute control)
  shuffled_historical    — Historical masks with token permutation (incoherence control)
  coherent_oldest_first  — Historical masks from oldest snapshot (proposed method)

Architecture: 31M pre-norm transformer with soft gated sparse attention,
WikiText-103, sparsity target 87.5%.

Usage:
    python src/sparse_attention_rca.py \
        --condition coherent_oldest_first \
        --seed 42 \
        --data-dir ./wikitext103_cache \
        --checkpoint-dir /s3-data/ckpts_sparse_rca \
        --output /s3-data/results/sparse_rca_coherent_oldest_first_s42.json
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

RA_ROOT = Path(__file__).parent.parent.parent / "routing-absorption"
sys.path.insert(0, str(RA_ROOT))

from src.models.config import ModelConfig
from src.models.gated_attention import GatedAttentionConfig, GatedSparseAttention
from src.models.transformer import GatedTransformer

CONDITIONS = [
    "learned_gate",
    "random_gate",
    "contemporary_replay",
    "shuffled_historical",
    "coherent_oldest_first",
    "coherent_hard_oldest_first",
    "contemporary_hard_replay",
    "shuffled_hard_historical",
    "coherent_hard_annealed",
    "coherent_closedloop_hard",
    "contemporary_closedloop_hard",
    "shuffled_closedloop_hard",
    "ste_hard",
    "anneal_to_hard",
    "dual_ste_clhr",
    "dense",
    "hard_from_scratch",
]

CHECKPOINT_STEPS = [5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000]

DEFAULT_CONFIG = ModelConfig(
    vocab_size=50257,
    max_seq_len=512,
    n_layers=6,
    d_model=256,
    n_heads=4,
    d_ff=1024,
    dropout=0.1,
    d_gate=32,
    sparsity_mode="soft",
    sparsity_k=64,
    gate_temperature=1.0,
)


def extract_gate_masks(
    model: GatedTransformer,
    input_ids: torch.Tensor,
) -> list[torch.Tensor]:
    """Extract per-layer gate masks from a forward pass.

    Returns list of masks, one per layer, shape (batch, n_heads, seq, seq).
    """
    model.eval()
    with torch.no_grad():
        _ = model(input_ids)

    masks = []
    for layer in model.layers:
        attn = layer.attention
        if attn.last_mask is not None:
            masks.append(attn.last_mask.clone())
        else:
            B, T = input_ids.shape
            masks.append(torch.ones(B, attn.config.n_heads, T, T,
                                    device=input_ids.device))
    return masks


def forward_closed_loop_historical_hard(
    model: GatedTransformer,
    input_ids: torch.Tensor,
    historical_gate_model: GatedTransformer,
    k: int = 64,
) -> torch.Tensor:
    """Closed-loop hard forward using historical gate parameters on current Q/K/V.

    At each layer:
    1. Compute gate scores from HISTORICAL gate on the CURRENT hard-path hidden state
    2. Harden to top-k
    3. Apply immediately to CURRENT model's Q/K/V attention
    4. Propagate the resulting hard hidden state to the next layer

    Gradients flow through current model's Q/K/V but NOT through historical gate or top-k.
    """
    batch_size, seq_len = input_ids.shape
    device = input_ids.device
    n_heads = model.config.n_heads
    d_head = model.config.d_model // n_heads
    d_gate = model.config.d_gate

    positions = torch.arange(seq_len, device=device).unsqueeze(0)
    x = model.embedding(input_ids) + model.pos_embedding(positions)
    x = model.dropout(x)

    causal = torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0).unsqueeze(0)

    for layer_idx, (layer, hist_layer) in enumerate(
        zip(model.layers, historical_gate_model.layers)
    ):
        attn = layer.attention
        hist_attn = hist_layer.attention

        h = layer.attn_norm(x)

        # Current model's Q/K/V
        q = attn.W_q(h).view(batch_size, seq_len, n_heads, d_head).transpose(1, 2)
        kk = attn.W_k(h).view(batch_size, seq_len, n_heads, d_head).transpose(1, 2)
        v = attn.W_v(h).view(batch_size, seq_len, n_heads, d_head).transpose(1, 2)
        attn_scores = torch.matmul(q, kk.transpose(-2, -1)) / (d_head ** 0.5)

        # Historical gate scores on CURRENT hard-path hidden state (no grad through gate)
        with torch.no_grad():
            gq = hist_attn.W_gq(h).view(batch_size, seq_len, n_heads, d_gate).transpose(1, 2)
            gk = hist_attn.W_gk(h).view(batch_size, seq_len, n_heads, d_gate).transpose(1, 2)
            gate_scores = torch.matmul(gq, gk.transpose(-2, -1)) / (d_gate ** 0.5)
            gate_scores = gate_scores.masked_fill(causal == 0, float("-inf"))

            actual_k = min(k, seq_len)
            _, topk_idx = torch.topk(gate_scores, actual_k, dim=-1)
            hard_mask = torch.zeros_like(gate_scores)
            hard_mask.scatter_(-1, topk_idx, 1.0)
            hard_mask = hard_mask * causal

        # Apply hard mask to attention scores
        attn_scores = attn_scores.masked_fill(hard_mask == 0, float("-inf"))
        attn_scores = attn_scores.masked_fill(causal == 0, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = attn.dropout(attn_weights)

        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = attn.W_o(output)

        x = x + output
        x = x + layer.ff(layer.ff_norm(x))

    x = model.final_norm(x)
    logits = model.lm_head(x)
    return logits


def forward_closed_loop_contemporary_hard(
    model: GatedTransformer,
    input_ids: torch.Tensor,
    k: int = 64,
) -> torch.Tensor:
    """Closed-loop hard forward using the CURRENT gate (detached) on the hard-path state.

    Control for closed-loop coherent-hard: tests whether on-policy hard exposure
    alone (without historical routing) is sufficient.
    Gradients flow through current Q/K/V but not through the gate or top-k.
    """
    batch_size, seq_len = input_ids.shape
    device = input_ids.device
    n_heads = model.config.n_heads
    d_head = model.config.d_model // n_heads
    d_gate = model.config.d_gate

    positions = torch.arange(seq_len, device=device).unsqueeze(0)
    x = model.embedding(input_ids) + model.pos_embedding(positions)
    x = model.dropout(x)

    causal = torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0).unsqueeze(0)

    for layer in model.layers:
        attn = layer.attention
        h = layer.attn_norm(x)

        q = attn.W_q(h).view(batch_size, seq_len, n_heads, d_head).transpose(1, 2)
        kk = attn.W_k(h).view(batch_size, seq_len, n_heads, d_head).transpose(1, 2)
        v = attn.W_v(h).view(batch_size, seq_len, n_heads, d_head).transpose(1, 2)
        attn_scores = torch.matmul(q, kk.transpose(-2, -1)) / (d_head ** 0.5)

        with torch.no_grad():
            gq = attn.W_gq(h).view(batch_size, seq_len, n_heads, d_gate).transpose(1, 2)
            gk = attn.W_gk(h).view(batch_size, seq_len, n_heads, d_gate).transpose(1, 2)
            gate_scores = torch.matmul(gq, gk.transpose(-2, -1)) / (d_gate ** 0.5)
            gate_scores = gate_scores.masked_fill(causal == 0, float("-inf"))

            actual_k = min(k, seq_len)
            _, topk_idx = torch.topk(gate_scores, actual_k, dim=-1)
            hard_mask = torch.zeros_like(gate_scores)
            hard_mask.scatter_(-1, topk_idx, 1.0)
            hard_mask = hard_mask * causal

        attn_scores = attn_scores.masked_fill(hard_mask == 0, float("-inf"))
        attn_scores = attn_scores.masked_fill(causal == 0, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = attn.dropout(attn_weights)

        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = attn.W_o(output)

        x = x + output
        x = x + layer.ff(layer.ff_norm(x))

    x = model.final_norm(x)
    logits = model.lm_head(x)
    return logits


def forward_closed_loop_shuffled_hard(
    model: GatedTransformer,
    input_ids: torch.Tensor,
    historical_gate_model: GatedTransformer,
    k: int = 64,
) -> torch.Tensor:
    """Closed-loop hard forward with SHUFFLED historical gate on the hard-path state.

    Control for closed-loop coherent-hard: tests whether historical diversity
    without input-conditioned coherence suffices under on-policy exposure.
    Masks are generated from historical gate on the current hard state, then
    key-dimension shuffled to destroy input-conditioned structure.
    """
    batch_size, seq_len = input_ids.shape
    device = input_ids.device
    n_heads = model.config.n_heads
    d_head = model.config.d_model // n_heads
    d_gate = model.config.d_gate

    positions = torch.arange(seq_len, device=device).unsqueeze(0)
    x = model.embedding(input_ids) + model.pos_embedding(positions)
    x = model.dropout(x)

    causal = torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0).unsqueeze(0)

    for layer_idx, (layer, hist_layer) in enumerate(
        zip(model.layers, historical_gate_model.layers)
    ):
        attn = layer.attention
        hist_attn = hist_layer.attention
        h = layer.attn_norm(x)

        q = attn.W_q(h).view(batch_size, seq_len, n_heads, d_head).transpose(1, 2)
        kk = attn.W_k(h).view(batch_size, seq_len, n_heads, d_head).transpose(1, 2)
        v = attn.W_v(h).view(batch_size, seq_len, n_heads, d_head).transpose(1, 2)
        attn_scores = torch.matmul(q, kk.transpose(-2, -1)) / (d_head ** 0.5)

        with torch.no_grad():
            gq = hist_attn.W_gq(h).view(batch_size, seq_len, n_heads, d_gate).transpose(1, 2)
            gk = hist_attn.W_gk(h).view(batch_size, seq_len, n_heads, d_gate).transpose(1, 2)
            gate_scores = torch.matmul(gq, gk.transpose(-2, -1)) / (d_gate ** 0.5)
            gate_scores = gate_scores.masked_fill(causal == 0, float("-inf"))

            # Shuffle key dimension to destroy input-conditioned structure
            for b in range(batch_size):
                perm = torch.randperm(seq_len, device=device)
                gate_scores[b] = gate_scores[b, :, :, perm]

            actual_k = min(k, seq_len)
            _, topk_idx = torch.topk(gate_scores, actual_k, dim=-1)
            hard_mask = torch.zeros_like(gate_scores)
            hard_mask.scatter_(-1, topk_idx, 1.0)
            hard_mask = hard_mask * causal

        attn_scores = attn_scores.masked_fill(hard_mask == 0, float("-inf"))
        attn_scores = attn_scores.masked_fill(causal == 0, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = attn.dropout(attn_weights)

        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = attn.W_o(output)

        x = x + output
        x = x + layer.ff(layer.ff_norm(x))

    x = model.final_norm(x)
    logits = model.lm_head(x)
    return logits


def forward_hard_from_scratch(
    model: GatedTransformer,
    input_ids: torch.Tensor,
    k: int = 64,
) -> torch.Tensor:
    """`hard_from_scratch` condition: gate is discrete (top-k hardened) from
    step 0, no soft warm-up, no CLHR dual soft/hard loss mixing.

    STE combined at the level of the final ATTENTION WEIGHTS (softmax(hard)
    vs softmax(soft-bias)), not by multiplying a binary mask onto raw
    pre-softmax scores. The latter (this file's own `forward_ste_hard`, used
    by sparse_attention_ste_tuning.py) leaks attention mass onto masked-out
    positions instead of excluding them, because masked positions still
    contribute exp(0)=1 to the softmax denominator. Ported from
    sparse_attention_300m.py's GatedSparseAttention._forward_hard_ste, which
    exists specifically to avoid that bug -- see its docstring.
    """
    batch_size, seq_len = input_ids.shape
    device = input_ids.device
    n_heads = model.config.n_heads
    d_head = model.config.d_model // n_heads
    d_gate = model.config.d_gate

    positions = torch.arange(seq_len, device=device).unsqueeze(0)
    x = model.embedding(input_ids) + model.pos_embedding(positions)
    x = model.dropout(x)

    causal = torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0).unsqueeze(0)

    for layer in model.layers:
        attn = layer.attention
        h = layer.attn_norm(x)

        q = attn.W_q(h).view(batch_size, seq_len, n_heads, d_head).transpose(1, 2)
        kk = attn.W_k(h).view(batch_size, seq_len, n_heads, d_head).transpose(1, 2)
        v = attn.W_v(h).view(batch_size, seq_len, n_heads, d_head).transpose(1, 2)
        raw_scores = torch.matmul(q, kk.transpose(-2, -1)) / (d_head ** 0.5)

        gq = attn.W_gq(h).view(batch_size, seq_len, n_heads, d_gate).transpose(1, 2)
        gk = attn.W_gk(h).view(batch_size, seq_len, n_heads, d_gate).transpose(1, 2)
        gate_scores = torch.matmul(gq, gk.transpose(-2, -1)) / (d_gate ** 0.5)
        soft_mask = torch.sigmoid(gate_scores)

        with torch.no_grad():
            gs_causal = gate_scores.masked_fill(causal == 0, float("-inf"))
            actual_k = min(k, seq_len)
            _, topk_idx = torch.topk(gs_causal, actual_k, dim=-1)
            hard_mask = torch.zeros_like(gate_scores)
            hard_mask.scatter_(-1, topk_idx, 1.0)
            hard_mask = hard_mask * causal

        hard_scores = raw_scores.masked_fill(hard_mask == 0, float("-inf"))
        hard_scores = hard_scores.masked_fill(causal == 0, float("-inf"))
        w_hard = F.softmax(hard_scores, dim=-1)
        w_hard = torch.nan_to_num(w_hard, nan=0.0)

        soft_bias = torch.log(soft_mask.clamp(min=1e-6)).masked_fill(causal == 0, float("-inf"))
        soft_scores = raw_scores + soft_bias
        w_soft = F.softmax(soft_scores, dim=-1)
        w_soft = torch.nan_to_num(w_soft, nan=0.0)

        # Forward value is exactly w_hard (the soft.detach()-soft cancels to
        # 0); backward gradient flows entirely through w_soft.
        w = w_hard.detach() - w_soft.detach() + w_soft

        output = torch.matmul(w.to(v.dtype), v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = attn.W_o(output)

        attn.last_mask = hard_mask.detach()
        attn.last_hard_mask = hard_mask.detach()
        attn.last_mask_live = w

        x = x + output
        x = x + layer.ff(layer.ff_norm(x))

    x = model.final_norm(x)
    return model.lm_head(x)


def forward_ste_hard(
    model: GatedTransformer,
    input_ids: torch.Tensor,
    k: int = 64,
) -> torch.Tensor:
    """Single-path hard forward with straight-through estimator.

    Forward: hard top-k masks. Backward: gradients flow through soft sigmoid
    via STE (hard - soft.detach() + soft). Gate and Q/K/V both receive gradients.
    """
    batch_size, seq_len = input_ids.shape
    device = input_ids.device
    n_heads = model.config.n_heads
    d_head = model.config.d_model // n_heads
    d_gate = model.config.d_gate

    positions = torch.arange(seq_len, device=device).unsqueeze(0)
    x = model.embedding(input_ids) + model.pos_embedding(positions)
    x = model.dropout(x)

    causal = torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0).unsqueeze(0)

    for layer in model.layers:
        attn = layer.attention
        h = layer.attn_norm(x)

        q = attn.W_q(h).view(batch_size, seq_len, n_heads, d_head).transpose(1, 2)
        kk = attn.W_k(h).view(batch_size, seq_len, n_heads, d_head).transpose(1, 2)
        v = attn.W_v(h).view(batch_size, seq_len, n_heads, d_head).transpose(1, 2)
        attn_scores = torch.matmul(q, kk.transpose(-2, -1)) / (d_head ** 0.5)

        gq = attn.W_gq(h).view(batch_size, seq_len, n_heads, d_gate).transpose(1, 2)
        gk = attn.W_gk(h).view(batch_size, seq_len, n_heads, d_gate).transpose(1, 2)
        gate_scores = torch.matmul(gq, gk.transpose(-2, -1)) / (d_gate ** 0.5)
        soft_mask = torch.sigmoid(gate_scores)

        gate_scores_causal = gate_scores.masked_fill(causal == 0, float("-inf"))
        actual_k = min(k, seq_len)
        _, topk_idx = torch.topk(gate_scores_causal, actual_k, dim=-1)
        hard_mask = torch.zeros_like(gate_scores)
        hard_mask.scatter_(-1, topk_idx, 1.0)
        hard_mask = hard_mask * causal

        # Straight-through estimator: forward value is exactly hard_mask,
        # backward gradient flows through soft_mask. Multiplication is the
        # correct operator — masked_fill treats masks as boolean conditions
        # and severs gradients through the gate.
        ste_mask = hard_mask.detach() - soft_mask.detach() + soft_mask
        attn_scores = attn_scores * ste_mask
        attn_scores = attn_scores.masked_fill(causal == 0, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = attn.dropout(attn_weights)

        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = attn.W_o(output)

        attn.last_mask = hard_mask.detach()
        attn.last_mask_live = ste_mask

        x = x + output
        x = x + layer.ff(layer.ff_norm(x))

    x = model.final_norm(x)
    logits = model.lm_head(x)
    return logits


def forward_anneal_hard(
    model: GatedTransformer,
    input_ids: torch.Tensor,
    k: int = 64,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Single-path forward with temperature-annealed gating.

    At high temperature: soft sigmoid masks (like standard training).
    At low temperature: sigmoid sharpens toward hard binary masks.
    Anneal temperature from 1.0 → 0.01 over training to transition
    smoothly from soft to hard.
    """
    batch_size, seq_len = input_ids.shape
    device = input_ids.device
    n_heads = model.config.n_heads
    d_head = model.config.d_model // n_heads
    d_gate = model.config.d_gate

    positions = torch.arange(seq_len, device=device).unsqueeze(0)
    x = model.embedding(input_ids) + model.pos_embedding(positions)
    x = model.dropout(x)

    causal = torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0).unsqueeze(0)

    for layer in model.layers:
        attn = layer.attention
        h = layer.attn_norm(x)

        q = attn.W_q(h).view(batch_size, seq_len, n_heads, d_head).transpose(1, 2)
        kk = attn.W_k(h).view(batch_size, seq_len, n_heads, d_head).transpose(1, 2)
        v = attn.W_v(h).view(batch_size, seq_len, n_heads, d_head).transpose(1, 2)
        attn_scores = torch.matmul(q, kk.transpose(-2, -1)) / (d_head ** 0.5)

        gq = attn.W_gq(h).view(batch_size, seq_len, n_heads, d_gate).transpose(1, 2)
        gk = attn.W_gk(h).view(batch_size, seq_len, n_heads, d_gate).transpose(1, 2)
        gate_scores = torch.matmul(gq, gk.transpose(-2, -1)) / (d_gate ** 0.5)

        # Temperature-scaled sigmoid: low temp → near-binary
        soft_mask = torch.sigmoid(gate_scores / max(temperature, 1e-4))

        attn_scores = attn_scores * soft_mask
        attn_scores = attn_scores.masked_fill(causal == 0, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = attn.dropout(attn_weights)

        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = attn.W_o(output)

        attn.last_mask = soft_mask.detach()
        attn.last_mask_live = soft_mask

        x = x + output
        x = x + layer.ff(layer.ff_norm(x))

    x = model.final_norm(x)
    logits = model.lm_head(x)
    return logits


def forward_with_masks(
    model: GatedTransformer,
    input_ids: torch.Tensor,
    forced_masks: list[torch.Tensor],
) -> torch.Tensor:
    """Forward pass using externally provided gate masks instead of learned gates.

    Replaces the gate computation in each layer with the provided mask.
    """
    batch_size, seq_len = input_ids.shape
    device = input_ids.device

    positions = torch.arange(seq_len, device=device).unsqueeze(0)
    x = model.embedding(input_ids) + model.pos_embedding(positions)
    x = model.dropout(x)

    causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

    for layer_idx, layer in enumerate(model.layers):
        attn = layer.attention
        config = attn.config
        n_h = config.n_heads
        d_head = config.d_model // n_h

        h = layer.attn_norm(x)

        q = attn.W_q(h).view(batch_size, seq_len, n_h, d_head).transpose(1, 2)
        k = attn.W_k(h).view(batch_size, seq_len, n_h, d_head).transpose(1, 2)
        v = attn.W_v(h).view(batch_size, seq_len, n_h, d_head).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (d_head ** 0.5)

        gate_mask = forced_masks[layer_idx].to(device)
        if gate_mask.shape[-1] != seq_len:
            gate_mask = gate_mask[:, :, :seq_len, :seq_len]

        attn_scores = attn_scores * gate_mask
        attn_scores = attn_scores.masked_fill(causal_mask == 0, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = attn.dropout(attn_weights)

        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = attn.W_o(output)

        x = x + output
        x = x + layer.ff(layer.ff_norm(x))

    x = model.final_norm(x)
    logits = model.lm_head(x)
    return logits


def select_snapshot_oldest_first(
    snapshots: list[dict],
) -> dict:
    """Select the oldest available snapshot (strongest decorrelation)."""
    if not snapshots:
        raise ValueError("No snapshots available")
    return min(snapshots, key=lambda s: s["step"])


def harden_masks(
    masks: list[torch.Tensor],
    k: int = 64,
) -> list[torch.Tensor]:
    """Convert soft sigmoid masks to hard binary top-k masks.

    Exposes Q/K/V to the deployed mask FORM during training,
    not just different soft routing contexts.
    """
    hardened = []
    for m in masks:
        B, H, Tq, Tk = m.shape
        actual_k = min(k, Tk)
        _, topk_idx = torch.topk(m, actual_k, dim=-1)
        hard = torch.zeros_like(m)
        hard.scatter_(-1, topk_idx, 1.0)
        hardened.append(hard)
    return hardened


def shuffle_masks(
    masks: list[torch.Tensor],
    seed: int | None = None,
) -> list[torch.Tensor]:
    """Shuffle gate masks by permuting the key dimension across tokens.

    Preserves per-query sparsity level but destroys input-conditioned structure.
    """
    if seed is not None:
        torch.manual_seed(seed)

    shuffled = []
    for m in masks:
        s = m.clone()
        B, H, Tq, Tk = s.shape
        for b in range(B):
            perm = torch.randperm(Tk, device=s.device)
            s[b] = s[b, :, :, perm]
        shuffled.append(s)
    return shuffled


class TokenDataset(Dataset):
    def __init__(self, tokens: torch.Tensor, seq_len: int):
        self.tokens = tokens
        self.seq_len = seq_len
        self.n_seqs = (len(tokens) - 1) // seq_len

    def __len__(self):
        return self.n_seqs

    def __getitem__(self, idx):
        start = idx * self.seq_len
        chunk = self.tokens[start: start + self.seq_len + 1]
        return {"input_ids": chunk}


def load_wikitext(data_dir: str, seq_len: int = 512):
    data_path = Path(data_dir)
    cache_train_pt = data_path / "wt103_train_tokens.pt"
    cache_val_pt = data_path / "wt103_val_tokens.pt"
    cache_train_npy = data_path / "wt103_train_tokens.npy"
    cache_val_npy = data_path / "wt103_val_tokens.npy"

    if cache_train_pt.exists() and cache_val_pt.exists():
        train_tok = torch.load(cache_train_pt, weights_only=True).long()
        val_tok = torch.load(cache_val_pt, weights_only=True).long()
    elif cache_train_npy.exists() and cache_val_npy.exists():
        train_tok = torch.from_numpy(np.load(cache_train_npy)).long()
        val_tok = torch.from_numpy(np.load(cache_val_npy)).long()
    else:
        from datasets import load_dataset
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", cache_dir=data_dir)

        def tokenize_split(split_name):
            texts = [t for t in ds[split_name]["text"] if len(t.strip()) > 0]
            all_ids = []
            for t in texts:
                all_ids.extend(tokenizer.encode(t))
            return torch.tensor(all_ids, dtype=torch.long)

        train_tok = tokenize_split("train")
        val_tok = tokenize_split("validation")
        data_path.mkdir(parents=True, exist_ok=True)
        torch.save(train_tok, cache_train_pt)
        torch.save(val_tok, cache_val_pt)

    print(f"  WikiText-103: train={len(train_tok)} val={len(val_tok)} tokens")
    return TokenDataset(train_tok, seq_len), TokenDataset(val_tok, seq_len)


def evaluate(model, loader, device, max_batches=200):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            ids = batch["input_ids"].to(device)
            x, y = ids[:, :-1], ids[:, 1:]
            logits = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum"
            )
            total_loss += loss.item()
            total_tokens += y.numel()
    return total_loss / total_tokens


def evaluate_hard_from_scratch(model, loader, device, max_batches=200, k=64):
    """Native NLL for hard_from_scratch: must go through the same hard
    forward the model was trained/deployed with, not the default (soft)
    model(x) call `evaluate()` uses for every other condition."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            ids = batch["input_ids"].to(device)
            x, y = ids[:, :-1], ids[:, 1:]
            logits = forward_hard_from_scratch(model, x, k=k)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum"
            )
            total_loss += loss.item()
            total_tokens += y.numel()
    return total_loss / total_tokens


def evaluate_with_forced_masks(model, loader, device, gate_model, max_batches=200):
    """Evaluate using gate masks from gate_model applied to model's Q/K/V."""
    model.eval()
    gate_model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            ids = batch["input_ids"].to(device)
            x, y = ids[:, :-1], ids[:, 1:]
            masks = extract_gate_masks(gate_model, x)
            logits = forward_with_masks(model, x, masks)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum"
            )
            total_loss += loss.item()
            total_tokens += y.numel()
    return total_loss / total_tokens


def evaluate_with_random_masks(model, loader, device, max_batches=200, sparsity=0.875):
    """Evaluate with random gate masks (router ablation)."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    n_layers = len(model.layers)
    n_heads = model.config.n_heads

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            ids = batch["input_ids"].to(device)
            x, y = ids[:, :-1], ids[:, 1:]
            seq_len = x.shape[1]
            k = max(1, int(seq_len * (1 - sparsity)))

            masks = []
            for _ in range(n_layers):
                rand_scores = torch.rand(x.shape[0], n_heads, seq_len, seq_len, device=device)
                _, topk_idx = torch.topk(rand_scores, k, dim=-1)
                mask = torch.zeros_like(rand_scores)
                mask.scatter_(-1, topk_idx, 1.0)
                masks.append(mask)

            logits = forward_with_masks(model, x, masks)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum"
            )
            total_loss += loss.item()
            total_tokens += y.numel()
    return total_loss / total_tokens


def find_latest_checkpoint(ckpt_path: Path) -> Path | None:
    """Return the step_<N>.pt checkpoint with the highest N, or None if empty.

    Must sort by parsed step number, not filename string -- lexicographic
    order ranks "step_5000.pt" above "step_10000.pt".
    """
    ckpts = list(ckpt_path.glob("step_*.pt"))
    if not ckpts:
        return None
    return max(ckpts, key=lambda p: int(p.stem.split("_")[1]))


def train_experiment(
    condition: str,
    seed: int,
    data_dir: str,
    checkpoint_dir: str,
    output_path: str,
    max_steps: int = 50000,
    batch_size: int = 16,
    lr: float = 3e-4,
    lambda_rca: float = 0.3,
    lambda_sparse: float = 1.0,
    target_sparsity: float = 0.875,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, condition: {condition}, seed: {seed}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    train_ds, val_ds = load_wikitext(data_dir)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, drop_last=True)

    config = DEFAULT_CONFIG.model_copy()
    if condition == "random_gate":
        config.freeze_gates = True
    if condition == "dense":
        config.sparsity_mode = "dense"

    model = GatedTransformer(config).to(device)
    print(f"  Parameters: {model.num_parameters() / 1e6:.1f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1, betas=(0.9, 0.95))

    def cosine_lr(step):
        warmup = 1000
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, max_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, cosine_lr)

    use_rca = condition in (
        "contemporary_replay", "shuffled_historical", "coherent_oldest_first",
        "coherent_hard_oldest_first", "contemporary_hard_replay", "shuffled_hard_historical",
        "coherent_hard_annealed", "coherent_closedloop_hard",
        "contemporary_closedloop_hard", "shuffled_closedloop_hard",
        "dual_ste_clhr",
    )
    use_annealing = condition == "coherent_hard_annealed"
    gate_snapshots = []
    snapshot_interval = 2500
    n_snapshots = 5

    tag = f"sparse_rca_{condition}_s{seed}"
    ckpt_path = Path(checkpoint_dir) / tag
    ckpt_path.mkdir(parents=True, exist_ok=True)

    gate_model = None
    if use_rca:
        gate_model = GatedTransformer(config).to(device)
        gate_model.eval()

    global_step = 0
    latest_ckpt = find_latest_checkpoint(ckpt_path)
    if latest_ckpt is not None:
        ckpt = torch.load(latest_ckpt, weights_only=True)
        model.load_state_dict(ckpt["model"])
        global_step = ckpt["step"]
        scheduler.last_epoch = global_step - 1
        print(f"  Resumed from {latest_ckpt} at step {global_step} "
              f"(optimizer state not saved -- AdamW momentum restarts fresh)",
              flush=True)

    train_iter = iter(train_loader)
    t0 = time.time()
    running_loss = 0.0
    running_count = 0
    temp = 1.0

    while global_step < max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        ids = batch["input_ids"].to(device)
        x, y = ids[:, :-1], ids[:, 1:]

        model.train()
        optimizer.zero_grad()

        # Temperature annealing: 1.0 → 0.01 over training (log-linear)
        if use_annealing or condition == "anneal_to_hard":
            progress = global_step / max(1, max_steps)
            temp = math.exp(math.log(1.0) * (1 - progress) + math.log(0.01) * progress)
            if use_annealing:
                for layer in model.layers:
                    layer.attention.config.temperature = temp

        if condition == "ste_hard":
            logits = forward_ste_hard(model, x, k=64)
        elif condition == "anneal_to_hard":
            logits = forward_anneal_hard(model, x, k=64, temperature=temp)
        elif condition == "dual_ste_clhr":
            logits = forward_ste_hard(model, x, k=64)
        elif condition == "hard_from_scratch":
            logits = forward_hard_from_scratch(model, x, k=64)
        else:
            logits = model(x)
        lm_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        sparsity_loss = torch.tensor(0.0, device=device)
        for layer in model.layers:
            mask = layer.attention.last_mask_live
            if mask is not None:
                actual = 1.0 - (mask.sum() / mask.numel())
                gap = F.relu(torch.tensor(target_sparsity, device=device) - actual)
                sparsity_loss = sparsity_loss + gap ** 2

        loss = lm_loss + lambda_sparse * sparsity_loss

        rca_loss = torch.tensor(0.0, device=device)
        needs_snapshots = condition not in ("contemporary_replay", "contemporary_hard_replay",
                                            "contemporary_closedloop_hard", "dual_ste_clhr")
        if use_rca and (not needs_snapshots or gate_snapshots) and lambda_rca > 0:
            if condition in ("coherent_oldest_first", "coherent_hard_oldest_first",
                             "coherent_hard_annealed", "coherent_closedloop_hard",
                             "shuffled_closedloop_hard"):
                snap = select_snapshot_oldest_first(gate_snapshots)
            elif condition in ("contemporary_replay", "contemporary_hard_replay",
                               "contemporary_closedloop_hard", "dual_ste_clhr"):
                snap = None
            else:
                snap = gate_snapshots[global_step % len(gate_snapshots)]

            if condition == "coherent_closedloop_hard":
                gate_model.load_state_dict(snap["params"])
                gate_model.eval()
                logits_rca = forward_closed_loop_historical_hard(
                    model, x, gate_model, k=64
                )
                rca_loss = F.cross_entropy(
                    logits_rca.reshape(-1, logits_rca.size(-1)), y.reshape(-1)
                )
            elif condition == "shuffled_closedloop_hard":
                gate_model.load_state_dict(snap["params"])
                gate_model.eval()
                logits_rca = forward_closed_loop_shuffled_hard(
                    model, x, gate_model, k=64
                )
                rca_loss = F.cross_entropy(
                    logits_rca.reshape(-1, logits_rca.size(-1)), y.reshape(-1)
                )
            elif condition in ("contemporary_closedloop_hard", "dual_ste_clhr"):
                logits_rca = forward_closed_loop_contemporary_hard(
                    model, x, k=64
                )
                rca_loss = F.cross_entropy(
                    logits_rca.reshape(-1, logits_rca.size(-1)), y.reshape(-1)
                )
            else:
                # Open-loop RCA: extract masks from soft trajectory, apply in separate forward
                if snap is not None:
                    gate_model.load_state_dict(snap["params"])
                    gate_model.eval()
                    masks = extract_gate_masks(gate_model, x)
                else:
                    with torch.no_grad():
                        masks = extract_gate_masks(model, x)

                if condition in ("shuffled_historical", "shuffled_hard_historical"):
                    masks = shuffle_masks(masks)

                if condition in ("coherent_hard_oldest_first", "contemporary_hard_replay",
                                 "shuffled_hard_historical", "coherent_hard_annealed"):
                    masks = harden_masks(masks, k=64)

                logits_rca = forward_with_masks(model, x, masks)
                rca_loss = F.cross_entropy(
                    logits_rca.reshape(-1, logits_rca.size(-1)), y.reshape(-1)
                )

            loss = loss + lambda_rca * rca_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        global_step += 1
        running_loss += lm_loss.item()
        running_count += 1

        if use_rca and global_step % snapshot_interval == 0:
            gate_snapshots.append({
                "params": copy.deepcopy(model.state_dict()),
                "step": global_step,
            })
            if len(gate_snapshots) > n_snapshots:
                gate_snapshots.pop(0)

        if global_step % 500 == 0:
            avg = running_loss / running_count
            elapsed = time.time() - t0
            rca_str = f" rca={rca_loss.item():.3f}" if use_rca and gate_snapshots else ""
            temp_str = f" τ={temp:.4f}" if use_annealing else ""
            print(f"  step {global_step:6d}/{max_steps}: loss={avg:.4f}{rca_str}{temp_str} ({elapsed:.0f}s)",
                  flush=True)
            running_loss = 0.0
            running_count = 0

        if global_step in CHECKPOINT_STEPS:
            save_path = ckpt_path / f"step_{global_step}.pt"
            torch.save({"model": model.state_dict(), "step": global_step}, save_path)
            print(f"    [checkpoint saved]", flush=True)

    # Final evaluation
    print("\n  Final evaluation...")
    if condition == "hard_from_scratch":
        final_nll = evaluate_hard_from_scratch(model, val_loader, device, k=64)
    else:
        final_nll = evaluate(model, val_loader, device)
    final_ppl = math.exp(final_nll)
    print(f"  Learned-gate NLL: {final_nll:.4f} (PPL: {final_ppl:.2f})")

    random_nll = evaluate_with_random_masks(model, val_loader, device, sparsity=target_sparsity)
    random_ppl = math.exp(random_nll)
    print(f"  Random-gate NLL:  {random_nll:.4f} (PPL: {random_ppl:.2f})")

    delta_router = random_nll - final_nll
    print(f"  Δ_router (random - learned): {delta_router:.4f}")

    results = {
        "condition": condition,
        "seed": seed,
        "model_params": model.num_parameters(),
        "steps": max_steps,
        "learned_nll": round(final_nll, 6),
        "learned_ppl": round(final_ppl, 2),
        "random_nll": round(random_nll, 6),
        "random_ppl": round(random_ppl, 2),
        "delta_router": round(delta_router, 6),
        "lambda_rca": lambda_rca if use_rca else 0.0,
        "target_sparsity": target_sparsity,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {output_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Sparse-attention coherent historical-mask RCA")
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--lambda-rca", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    train_experiment(
        condition=args.condition,
        seed=args.seed,
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        output_path=args.output,
        max_steps=args.steps,
        lambda_rca=args.lambda_rca,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()
