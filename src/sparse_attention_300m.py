"""300M fully trainable sparse-attention transformer: scale confirmation.

Trains a ~300M decoder-only transformer from scratch with soft gated sparse
attention on all layers. Tests whether routing absorption and the coherent-RCA
remedy persist at 10× the 31M scale.

Architecture: 20 layers, d=1024, 16 heads, d_ff=4096, d_gate=32, ~300M params.
Training: 4B tokens on WikiText-103 (cycled), BF16, gradient checkpointing.

Conditions:
  standard             — normal sparse training
  contemporary_replay  — dual-loss with current masks
  shuffled_historical  — dual-loss with shuffled historical masks
  coherent_hard_oldest — dual-loss with hardened oldest historical masks
  hard_from_scratch    — the "why not just train hard the whole time?"
                         control: SAME gated-attention architecture, but the
                         gate is discrete (top-k hardened) from step 0 -- no
                         soft warm-up, no CLHR dual soft/hard loss mixing.
                         Gradient to the gate is via a straight-through
                         estimator (see GatedSparseAttention._forward_hard_ste).

Usage:
    python src/sparse_attention_300m.py \
        --condition standard --seed 42 \
        --data-dir ./wikitext103_cache \
        --checkpoint-dir /s3-data/ckpts_sparse_300m \
        --output /s3-data/results/sparse_300m_standard_s42.json \
        --total-tokens 4000000000

    python src/sparse_attention_300m.py --benchmark \
        --data-dir ./wikitext103_cache
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

CONDITIONS = ["standard", "contemporary_replay", "shuffled_historical", "coherent_hard_oldest",
              "coherent_closedloop_hard", "contemporary_closedloop_hard", "dense",
              "hard_from_scratch"]

CHECKPOINT_TOKEN_MILESTONES = [250_000_000, 500_000_000, 1_000_000_000,
                                1_500_000_000, 2_000_000_000, 2_500_000_000,
                                3_000_000_000, 3_500_000_000, 4_000_000_000,
                                6_000_000_000]

# Top-k cardinality for the `hard_from_scratch` condition's gate, matching the
# k=64 convention hardcoded everywhere else in this file (forward_closed_loop_
# gate_hard_300m's default, the CLHR closed-loop branches' actual_k=min(64,T),
# etc.) -- kept as one named constant here (rather than repeating the literal)
# since it must be IDENTICAL between training and the "native" eval call for
# the G_CL-near-zero-by-construction property to hold.
HARD_FROM_SCRATCH_K = 64


# ═══════════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════════

class GatedSparseAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_gate: int, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_gate = d_gate

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.W_gq = nn.Linear(d_model, n_heads * d_gate, bias=False)
        self.W_gk = nn.Linear(d_model, n_heads * d_gate, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.last_soft_mask = None
        self.last_hard_mask = None

    def forward(self, x, causal_mask, forced_mask=None, dense_mode=False,
                hard_ste_mode=False, hard_k=64):
        B, T, D = x.shape
        n_h, d_h, d_g = self.n_heads, self.d_head, self.d_gate

        q = self.W_q(x).view(B, T, n_h, d_h).transpose(1, 2)
        k = self.W_k(x).view(B, T, n_h, d_h).transpose(1, 2)
        v = self.W_v(x).view(B, T, n_h, d_h).transpose(1, 2)

        if hard_ste_mode:
            if dense_mode or forced_mask is not None:
                raise ValueError("hard_ste_mode is incompatible with dense_mode/forced_mask")
            return self._forward_hard_ste(x, q, k, v, causal_mask, hard_k)

        if dense_mode:
            attn_bias = causal_mask.masked_fill(causal_mask == 0, float("-inf")).masked_fill(causal_mask == 1, 0.0)
            self.last_soft_mask = None
        else:
            if forced_mask is not None:
                soft_mask = forced_mask
            else:
                gq = self.W_gq(x).view(B, T, n_h, d_g).transpose(1, 2)
                gk = self.W_gk(x).view(B, T, n_h, d_g).transpose(1, 2)
                gate_scores = torch.matmul(gq, gk.transpose(-2, -1)) / math.sqrt(d_g)
                soft_mask = torch.sigmoid(gate_scores)

            self.last_soft_mask = soft_mask.detach()

            gate_bias = torch.log(soft_mask.clamp(min=1e-6))
            attn_bias = gate_bias.masked_fill(causal_mask == 0, float("-inf"))

        if attn_bias.dtype != q.dtype:
            attn_bias = attn_bias.to(dtype=q.dtype)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_bias, dropout_p=self.dropout.p if self.training else 0.0,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.W_o(out)

    def _forward_hard_ste(self, x, q, k, v, causal_mask, hard_k):
        """`hard_from_scratch` condition: gate is discrete (top-k hardened)
        from step 0, no soft warm-up, no CLHR dual soft/hard loss mixing.

        Forward value is DELIBERATELY bit-identical to
        `forward_closed_loop_gate_hard_300m`'s closed-loop deployment
        computation (additive -inf masking of the final attention weights) --
        this is what makes G_CL ~0 "by construction" for this condition: there
        is no soft/hard mismatch to begin with, so the deployed (closed-loop)
        and trained (native) computations are the SAME function.

        Gradient to the gate (W_gq/W_gk) is obtained via a straight-through
        estimator applied at the level of the final attention WEIGHTS: the
        backward surrogate is this file's own additive log-bias soft-gate
        mechanism (the exact formula the `standard` condition trains with),
        not a multiplicative mask on the raw pre-softmax scores. Multiplying
        a binary mask onto raw scores and THEN softmaxing is the exact bug
        pattern `tests/test_sparse_300m_random_hard.py::
        test_random_hard_masking_is_real` guards against (it leaks attention
        mass onto masked-out positions instead of excluding them) -- and is
        also the literal convention `sparse_attention_rca.py::
        forward_ste_hard` (used by `sparse_attention_ste_tuning.py`) uses.
        That convention was intentionally NOT copied here: doing so would
        make this condition's own "native" forward diverge from its own
        closed-loop-hard deployment path, defeating the near-zero-G_CL-by-
        construction sanity check this baseline exists to provide. See
        tests/test_sparse_300m_hard_from_scratch.py for the regression tests
        enforcing bit-identity with forward_closed_loop_gate_hard_300m.
        """
        B, T, D = x.shape
        n_h, d_h, d_g = self.n_heads, self.d_head, self.d_gate

        gq = self.W_gq(x).view(B, T, n_h, d_g).transpose(1, 2)
        gk = self.W_gk(x).view(B, T, n_h, d_g).transpose(1, 2)
        gate_scores = torch.matmul(gq, gk.transpose(-2, -1)) / math.sqrt(d_g)
        soft_mask = torch.sigmoid(gate_scores)
        self.last_soft_mask = soft_mask.detach()

        with torch.no_grad():
            gs_causal = gate_scores.masked_fill(causal_mask == 0, float("-inf"))
            actual_k = min(hard_k, gs_causal.shape[-1])
            _, topk_idx = torch.topk(gs_causal, actual_k, dim=-1)
            hard_mask = torch.zeros_like(gate_scores)
            hard_mask.scatter_(-1, topk_idx, 1.0)
            hard_mask = hard_mask * causal_mask
        self.last_hard_mask = hard_mask.detach()

        raw_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_h)

        hard_scores = raw_scores.masked_fill(hard_mask == 0, float("-inf"))
        hard_scores = hard_scores.masked_fill(causal_mask == 0, float("-inf"))
        w_hard = F.softmax(hard_scores, dim=-1)
        w_hard = torch.nan_to_num(w_hard, nan=0.0)

        soft_bias = torch.log(soft_mask.clamp(min=1e-6)).masked_fill(causal_mask == 0, float("-inf"))
        soft_scores = raw_scores + soft_bias
        w_soft = F.softmax(soft_scores, dim=-1)
        w_soft = torch.nan_to_num(w_soft, nan=0.0)

        # Straight-through estimator: forward value is exactly w_hard (the
        # subtraction of soft.detach() from soft cancels to exactly 0 in
        # floating point); backward gradient flows entirely through w_soft.
        w = w_hard.detach() - w_soft.detach() + w_soft

        out = torch.matmul(w.to(v.dtype), v)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.W_o(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, d_gate, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = GatedSparseAttention(d_model, n_heads, d_gate, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, causal_mask, forced_mask=None, dense_mode=False,
                hard_ste_mode=False, hard_k=64):
        x = x + self.attn(self.ln1(x), causal_mask, forced_mask, dense_mode=dense_mode,
                           hard_ste_mode=hard_ste_mode, hard_k=hard_k)
        x = x + self.ff(self.ln2(x))
        return x


class SparseTransformer300M(nn.Module):
    def __init__(self, vocab_size=50257, d_model=1024, n_heads=16, n_layers=20,
                 d_ff=4096, d_gate=32, max_seq_len=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_gate = d_gate
        self.max_seq_len = max_seq_len

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, d_gate, dropout)
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, input_ids, forced_masks=None, use_checkpoint=False, dense_mode=False,
                hard_ste_mode=False, hard_k=64):
        B, T = input_ids.shape
        device = input_ids.device

        pos = torch.arange(T, device=device).unsqueeze(0)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        x = self.drop(x)

        if not hasattr(self, '_causal_cache') or self._causal_cache.shape[-1] != T or self._causal_cache.device != device:
            self._causal_cache = torch.tril(torch.ones(T, T, device=device)).unsqueeze(0).unsqueeze(0)
        causal_mask = self._causal_cache

        all_masks = []
        for i, block in enumerate(self.blocks):
            fm = forced_masks[i] if forced_masks is not None else None
            if use_checkpoint and self.training:
                x = grad_checkpoint(block, x, causal_mask, fm, dense_mode, hard_ste_mode, hard_k,
                                     use_reentrant=False)
            else:
                x = block(x, causal_mask, fm, dense_mode=dense_mode,
                          hard_ste_mode=hard_ste_mode, hard_k=hard_k)
            if block.attn.last_soft_mask is not None:
                all_masks.append(block.attn.last_soft_mask)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits, all_masks

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


# ═══════════════════════════════════════════════════════════════════════
# Mask utilities
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_oracle_masks(model, input_ids, k=64):
    model.eval()
    B, T = input_ids.shape
    device = input_ids.device

    pos = torch.arange(T, device=device).unsqueeze(0)
    x = model.tok_emb(input_ids) + model.pos_emb(pos)
    x = model.drop(x)
    causal = torch.tril(torch.ones(T, T, device=device)).unsqueeze(0).unsqueeze(0)

    masks = []
    for block in model.blocks:
        normed = block.ln1(x)
        attn = block.attn
        n_h, d_h = attn.n_heads, attn.d_head
        q = attn.W_q(normed).view(B, T, n_h, d_h).transpose(1, 2)
        kk = attn.W_k(normed).view(B, T, n_h, d_h).transpose(1, 2)
        scores = torch.matmul(q, kk.transpose(-2, -1)) / math.sqrt(d_h)
        scores = scores.masked_fill(causal == 0, float("-inf"))
        actual_k = min(k, T)
        _, topk_idx = torch.topk(scores, actual_k, dim=-1)
        mask = torch.zeros_like(scores)
        mask.scatter_(-1, topk_idx, 1.0)
        masks.append(mask * causal)
        x = block(x, causal)

    return masks


@torch.no_grad()
def compute_hardened_gate_masks(model, input_ids, k=64):
    model.eval()
    B, T = input_ids.shape
    device = input_ids.device

    pos = torch.arange(T, device=device).unsqueeze(0)
    x = model.tok_emb(input_ids) + model.pos_emb(pos)
    x = model.drop(x)
    causal = torch.tril(torch.ones(T, T, device=device)).unsqueeze(0).unsqueeze(0)

    masks = []
    for block in model.blocks:
        normed = block.ln1(x)
        attn = block.attn
        n_h, d_g = attn.n_heads, attn.d_gate
        gq = attn.W_gq(normed).view(B, T, n_h, d_g).transpose(1, 2)
        gk = attn.W_gk(normed).view(B, T, n_h, d_g).transpose(1, 2)
        gate_scores = torch.matmul(gq, gk.transpose(-2, -1)) / math.sqrt(d_g)
        gate_scores = gate_scores.masked_fill(causal == 0, float("-inf"))
        actual_k = min(k, T)
        _, topk_idx = torch.topk(gate_scores, actual_k, dim=-1)
        mask = torch.zeros_like(gate_scores)
        mask.scatter_(-1, topk_idx, 1.0)
        masks.append(mask * causal)
        x = block(x, causal)

    return masks


def harden_masks(masks, k=64):
    hardened = []
    for m in masks:
        actual_k = min(k, m.shape[-1])
        _, topk_idx = torch.topk(m, actual_k, dim=-1)
        hard = torch.zeros_like(m)
        hard.scatter_(-1, topk_idx, 1.0)
        hardened.append(hard)
    return hardened


def shuffle_masks(masks):
    shuffled = []
    for m in masks:
        s = m.clone()
        B = s.shape[0]
        Tk = s.shape[-1]
        for b in range(B):
            perm = torch.randperm(Tk, device=s.device)
            s[b] = s[b, :, :, perm]
        shuffled.append(s)
    return shuffled


# ═══════════════════════════════════════════════════════════════════════
# Data
# ═══════════════════════════════════════════════════════════════════════

class CyclingTokenDataset(Dataset):
    def __init__(self, tokens, seq_len):
        self.tokens = tokens
        self.seq_len = seq_len
        self.n_seqs = (len(tokens) - 1) // seq_len

    def __len__(self):
        return self.n_seqs

    def __getitem__(self, idx):
        idx = idx % self.n_seqs
        start = idx * self.seq_len
        return self.tokens[start: start + self.seq_len + 1]


def load_training_data(data_dir, seq_len=512):
    data_path = Path(data_dir)
    for name in ["wt103_train_tokens.pt", "wt103_train_tokens.npy"]:
        p = data_path / name
        if p.exists():
            if name.endswith(".pt"):
                tokens = torch.load(p, weights_only=True).long()
            else:
                tokens = torch.from_numpy(np.load(p)).long()
            print(f"  Loaded {len(tokens):,} tokens from {p}")
            return CyclingTokenDataset(tokens, seq_len)

    for name in ["train.pt"]:
        p = data_path / name
        if p.exists():
            data = torch.load(p, weights_only=True)
            if isinstance(data, dict) and "input_ids" in data:
                tokens = data["input_ids"].reshape(-1)
            else:
                tokens = data.reshape(-1) if data.dim() > 1 else data
            print(f"  Loaded {len(tokens):,} tokens from {p}")
            return CyclingTokenDataset(tokens.long(), seq_len)

    raise FileNotFoundError(f"No training data found in {data_dir}")


def load_val_data(data_dir, seq_len=512):
    data_path = Path(data_dir)
    for name in ["wt103_val_tokens.pt", "wt103_val_tokens.npy"]:
        p = data_path / name
        if p.exists():
            if name.endswith(".pt"):
                tokens = torch.load(p, weights_only=True).long()
            else:
                tokens = torch.from_numpy(np.load(p)).long()
            return CyclingTokenDataset(tokens, seq_len)

    for name in ["validation.pt"]:
        p = data_path / name
        if p.exists():
            data = torch.load(p, weights_only=True)
            if isinstance(data, dict) and "input_ids" in data:
                tokens = data["input_ids"].reshape(-1)
            else:
                tokens = data.reshape(-1) if data.dim() > 1 else data
            return CyclingTokenDataset(tokens.long(), seq_len)

    raise FileNotFoundError(f"No validation data found in {data_dir}")


# ═══════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_nll(model, loader, device, max_batches=100, forced_mask_fn=None,
                  hard_ste_mode=False, hard_k=64):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = batch.to(device)
        x, y = batch[:, :-1], batch[:, 1:]
        if forced_mask_fn is not None:
            masks = forced_mask_fn(model, x)
            logits, _ = model(x, forced_masks=masks, use_checkpoint=False)
        else:
            logits, _ = model(x, use_checkpoint=False, hard_ste_mode=hard_ste_mode, hard_k=hard_k)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum")
        total_loss += loss.item()
        total_tokens += y.numel()
    return total_loss / total_tokens


def forward_closed_loop_gate_hard_300m(model, input_ids, k=64):
    """Closed-loop one-pass hard deployment using the model's own learned gate.

    At each layer: compute gate scores from CURRENT hard-path hidden state,
    harden to top-k, apply immediately, propagate.
    """
    model.eval()
    B, T = input_ids.shape
    device = input_ids.device

    positions = torch.arange(T, device=device).unsqueeze(0)
    x = model.tok_emb(input_ids) + model.pos_emb(positions)
    x = model.drop(x)

    causal = torch.tril(torch.ones(T, T, device=device)).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        for block in model.blocks:
            h = block.ln1(x)
            attn = block.attn

            q = attn.W_q(h).view(B, T, attn.n_heads, attn.d_head).transpose(1, 2)
            kk = attn.W_k(h).view(B, T, attn.n_heads, attn.d_head).transpose(1, 2)
            v = attn.W_v(h).view(B, T, attn.n_heads, attn.d_head).transpose(1, 2)
            scores = torch.matmul(q, kk.transpose(-2, -1)) / math.sqrt(attn.d_head)

            gq = attn.W_gq(h).view(B, T, attn.n_heads, attn.d_gate).transpose(1, 2)
            gk = attn.W_gk(h).view(B, T, attn.n_heads, attn.d_gate).transpose(1, 2)
            gate_scores = torch.matmul(gq, gk.transpose(-2, -1)) / math.sqrt(attn.d_gate)
            gate_scores = gate_scores.masked_fill(causal == 0, float("-inf"))

            actual_k = min(k, T)
            _, topk_idx = torch.topk(gate_scores, actual_k, dim=-1)
            hard_mask = torch.zeros_like(gate_scores)
            hard_mask.scatter_(-1, topk_idx, 1.0)
            hard_mask = hard_mask * causal

            scores = scores.masked_fill(hard_mask == 0, float("-inf"))
            scores = scores.masked_fill(causal == 0, float("-inf"))
            w = F.softmax(scores, dim=-1)
            w = torch.nan_to_num(w, nan=0.0)

            out = torch.matmul(w, v)
            out = out.transpose(1, 2).contiguous().view(B, T, -1)
            out = attn.W_o(out)
            x = x + out

            x = x + block.ff(block.ln2(x))

    x = model.ln_f(x)
    return model.lm_head(x)


def forward_closed_loop_score_topk_300m(model, input_ids, k=64):
    """Closed-loop using score-top-k (Q@K^T) from current hard-path state."""
    model.eval()
    B, T = input_ids.shape
    device = input_ids.device

    positions = torch.arange(T, device=device).unsqueeze(0)
    x = model.tok_emb(input_ids) + model.pos_emb(positions)
    x = model.drop(x)

    causal = torch.tril(torch.ones(T, T, device=device)).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        for block in model.blocks:
            h = block.ln1(x)
            attn = block.attn

            q = attn.W_q(h).view(B, T, attn.n_heads, attn.d_head).transpose(1, 2)
            kk = attn.W_k(h).view(B, T, attn.n_heads, attn.d_head).transpose(1, 2)
            v = attn.W_v(h).view(B, T, attn.n_heads, attn.d_head).transpose(1, 2)
            scores = torch.matmul(q, kk.transpose(-2, -1)) / math.sqrt(attn.d_head)

            scores_for_topk = scores.clone()
            scores_for_topk = scores_for_topk.masked_fill(causal == 0, float("-inf"))
            actual_k = min(k, T)
            _, topk_idx = torch.topk(scores_for_topk, actual_k, dim=-1)
            hard_mask = torch.zeros_like(scores_for_topk)
            hard_mask.scatter_(-1, topk_idx, 1.0)
            hard_mask = hard_mask * causal

            scores = scores.masked_fill(hard_mask == 0, float("-inf"))
            scores = scores.masked_fill(causal == 0, float("-inf"))
            w = F.softmax(scores, dim=-1)
            w = torch.nan_to_num(w, nan=0.0)

            out = torch.matmul(w, v)
            out = out.transpose(1, 2).contiguous().view(B, T, -1)
            out = attn.W_o(out)
            x = x + out

            x = x + block.ff(block.ln2(x))

    x = model.ln_f(x)
    return model.lm_head(x)


@torch.no_grad()
def compute_random_hard_masks(model, input_ids, k=64, generator=None):
    """Per-layer uniformly-random hard masks with cardinality matching the
    learned gate-hard mask exactly (same causal-premask-then-topk recipe as
    forward_closed_loop_gate_hard_300m, so per-query-row counts agree by
    construction: both use actual_k = min(k, T) topk over causal-restricted
    scores). Respects causality: never selects j > i.
    """
    model.eval()
    B, T = input_ids.shape
    device = input_ids.device
    causal = torch.tril(torch.ones(T, T, device=device)).unsqueeze(0).unsqueeze(0)

    masks = []
    for block in model.blocks:
        n_h = block.attn.n_heads
        rand_scores = torch.rand(B, n_h, T, T, device=device, generator=generator)
        rand_scores = rand_scores.masked_fill(causal == 0, float("-inf"))
        actual_k = min(k, T)
        _, topk_idx = torch.topk(rand_scores, actual_k, dim=-1)
        mask = torch.zeros_like(rand_scores)
        mask.scatter_(-1, topk_idx, 1.0)
        masks.append(mask * causal)
    return masks


def forward_random_hard_300m(model, input_ids, k=64, generator=None):
    """Routing-utility baseline: one-pass hard deployment using a uniformly
    RANDOM per-layer selection of the same cardinality as the learned
    gate-hard mask (see compute_random_hard_masks). Masking is applied via
    additive -inf (masked_fill), never by multiplying a binary mask onto raw
    scores — multiplying zeroes negative scores toward 0 (i.e. *up*) instead
    of excluding them, leaking attention mass to masked-out positions.
    """
    model.eval()
    B, T = input_ids.shape
    device = input_ids.device

    positions = torch.arange(T, device=device).unsqueeze(0)
    x = model.tok_emb(input_ids) + model.pos_emb(positions)
    x = model.drop(x)

    causal = torch.tril(torch.ones(T, T, device=device)).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        for block in model.blocks:
            h = block.ln1(x)
            attn = block.attn

            q = attn.W_q(h).view(B, T, attn.n_heads, attn.d_head).transpose(1, 2)
            kk = attn.W_k(h).view(B, T, attn.n_heads, attn.d_head).transpose(1, 2)
            v = attn.W_v(h).view(B, T, attn.n_heads, attn.d_head).transpose(1, 2)
            scores = torch.matmul(q, kk.transpose(-2, -1)) / math.sqrt(attn.d_head)

            rand_scores = torch.rand(B, attn.n_heads, T, T, device=device, generator=generator)
            rand_scores = rand_scores.masked_fill(causal == 0, float("-inf"))
            actual_k = min(k, T)
            _, topk_idx = torch.topk(rand_scores, actual_k, dim=-1)
            hard_mask = torch.zeros_like(rand_scores)
            hard_mask.scatter_(-1, topk_idx, 1.0)
            hard_mask = hard_mask * causal

            scores = scores.masked_fill(hard_mask == 0, float("-inf"))
            scores = scores.masked_fill(causal == 0, float("-inf"))
            w = F.softmax(scores, dim=-1)
            w = torch.nan_to_num(w, nan=0.0)

            out = torch.matmul(w, v)
            out = out.transpose(1, 2).contiguous().view(B, T, -1)
            out = attn.W_o(out)
            x = x + out

            x = x + block.ff(block.ln2(x))

    x = model.ln_f(x)
    return model.lm_head(x)


def eval_closed_loop_300m(model, loader, device, mode, k=64, max_batches=100):
    """Evaluate with closed-loop hard deployment."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            ids = batch["input_ids"].to(device) if isinstance(batch, dict) else batch.to(device)
            x, y = ids[:, :-1], ids[:, 1:]
            if mode == "closed_gate":
                logits = forward_closed_loop_gate_hard_300m(model, x, k=k)
            elif mode == "closed_score":
                logits = forward_closed_loop_score_topk_300m(model, x, k=k)
            else:
                raise ValueError(f"Unknown mode: {mode}")
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   y.reshape(-1), reduction="sum")
            total_loss += loss.item()
            total_tokens += y.numel()
    return total_loss / total_tokens


def run_full_evaluation(model, val_loader, device, k=64, native_hard_ste=False):
    # Native. For every condition except hard_from_scratch this is the soft
    # (log-additive-gate) forward -- the model's own trained deployment mode.
    # For hard_from_scratch (native_hard_ste=True) there IS no soft mode: the
    # model was trained hard from step 0, so "native" must also be the hard
    # forward (hard_ste_mode=True), matching what it was actually trained on.
    native_nll = evaluate_nll(model, val_loader, device, hard_ste_mode=native_hard_ste, hard_k=k)

    # Open-loop (soft-state replay) — kept for comparison but labeled correctly
    ol_score_nll = evaluate_nll(model, val_loader, device,
                                forced_mask_fn=lambda m, x: compute_oracle_masks(m, x, k))
    ol_gate_nll = evaluate_nll(model, val_loader, device,
                                forced_mask_fn=lambda m, x: compute_hardened_gate_masks(m, x, k))

    # CLOSED-LOOP (true one-pass deployment) — the correct metrics
    cl_gate_nll = eval_closed_loop_300m(model, val_loader, device, "closed_gate", k=k)
    cl_score_nll = eval_closed_loop_300m(model, val_loader, device, "closed_score", k=k)

    # Random swap
    swap_nlls = []
    n_layers = model.n_layers
    n_heads = model.n_heads
    for _ in range(10):
        def random_mask_fn(m, x):
            B, T = x.shape
            masks = []
            for _ in range(n_layers):
                actual_k = min(k, T)
                rand = torch.rand(B, n_heads, T, T, device=x.device)
                _, idx = torch.topk(rand, actual_k, dim=-1)
                mask = torch.zeros_like(rand)
                mask.scatter_(-1, idx, 1.0)
                causal = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)
                masks.append(mask * causal)
            return masks
        swap_nlls.append(evaluate_nll(model, val_loader, device,
                                       forced_mask_fn=random_mask_fn, max_batches=50))

    return {
        "native_nll": round(native_nll, 6),
        "native_ppl": round(math.exp(native_nll), 2),
        # Closed-loop (correct deployment metrics)
        "closed_gate_nll": round(cl_gate_nll, 6),
        "closed_gate_excess": round(cl_gate_nll - native_nll, 6),
        "closed_score_nll": round(cl_score_nll, 6),
        "closed_score_excess": round(cl_score_nll - native_nll, 6),
        # Open-loop (soft-state replay, for reference only)
        "openloop_gate_nll": round(ol_gate_nll, 6),
        "openloop_gate_excess": round(ol_gate_nll - native_nll, 6),
        "openloop_score_nll": round(ol_score_nll, 6),
        "openloop_score_excess": round(ol_score_nll - native_nll, 6),
        # Swap
        "swap_nll_mean": round(float(np.mean(swap_nlls)), 6),
        "swap_nll_std": round(float(np.std(swap_nlls)), 6),
        "delta_swap": round(float(np.mean(swap_nlls)) - native_nll, 6),
    }


# ═══════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════

def train_experiment(
    condition: str,
    seed: int,
    data_dir: str,
    checkpoint_dir: str,
    output_path: str,
    total_tokens: int = 4_000_000_000,
    micro_batch: int = None,  # auto-selected based on condition
    grad_accum: int = None,   # auto-selected to maintain ~65K tokens/step
    seq_len: int = 512,
    lr: float = 3e-4,
    target_sparsity: float = 0.875,
    lambda_rca: float = 0.3,
    lambda_sparse: float = 1.0,
    normalize_loss: bool = False,
    benchmark: bool = False,
):
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
        print(f"Device: {device}, dtype: {dtype}, condition: {condition}, seed: {seed}")
        print(f"  DDP: {ddp}, world_size: {world_size}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    use_rca = condition not in ("standard", "dense", "hard_from_scratch")
    is_dense = condition == "dense"
    is_contemporary_cl = condition == "contemporary_closedloop_hard"
    is_hard_from_scratch = condition == "hard_from_scratch"
    if micro_batch is None:
        micro_batch = 32
    if grad_accum is None:
        target_tokens = 65536
        # Each GPU processes micro_batch per accum step; total = micro_batch * grad_accum * world_size
        grad_accum = max(1, target_tokens // (micro_batch * seq_len * world_size))
    if is_master:
        print(f"  micro_batch={micro_batch}, grad_accum={grad_accum}, world_size={world_size}")

    tokens_per_step = micro_batch * seq_len * grad_accum * world_size
    total_steps = total_tokens // tokens_per_step
    if is_master:
        print(f"  Tokens/step: {tokens_per_step:,}, total steps: {total_steps:,}")

    train_ds = load_training_data(data_dir, seq_len)
    sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True) if ddp else None
    train_loader = DataLoader(train_ds, batch_size=micro_batch, shuffle=(sampler is None),
                              sampler=sampler, num_workers=2, pin_memory=True, drop_last=True)

    model = SparseTransformer300M().to(device).to(dtype)
    if is_master:
        print(f"  Parameters: {model.count_parameters() / 1e6:.1f}M")

    use_compile = device.type == "cuda" and hasattr(torch, "compile")
    raw_model = model
    if use_compile:
        model = torch.compile(model)
        if is_master:
            print("  torch.compile enabled")

    if ddp:
        model = DDP(
            model,
            device_ids=[local_rank],
            gradient_as_bucket_view=True,
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1, betas=(0.9, 0.95))

    warmup_steps = 2000
    def cosine_lr(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, cosine_lr)

    use_rca = condition not in ("standard", "dense", "hard_from_scratch")
    gate_snapshots = []
    snapshot_interval = 1000
    n_snapshots = 5

    tag = f"sparse_300m_{condition}_s{seed}"
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
            print(f"  Resumed from step {global_step}, {tokens_seen/1e9:.2f}B tokens")

    train_iter = iter(train_loader)
    t0 = time.time()
    running_loss = 0.0
    running_count = 0
    next_ckpt_tokens = 0
    for m in CHECKPOINT_TOKEN_MILESTONES:
        if m > tokens_seen:
            next_ckpt_tokens = m
            break

    if benchmark:
        total_steps = 500
        if is_master:
            print(f"  BENCHMARK MODE: {total_steps} steps")

    while global_step < total_steps:
        model.train()
        optimizer.zero_grad()
        accum_loss = 0.0

        for accum_step in range(grad_accum):
            try:
                batch = next(train_iter)
            except StopIteration:
                if sampler is not None:
                    sampler.set_epoch(global_step)
                train_iter = iter(train_loader)
                batch = next(train_iter)

            batch = batch.to(device)
            x, y = batch[:, :-1], batch[:, 1:]

            with torch.amp.autocast("cuda", dtype=dtype):
                logits, masks = model(x, dense_mode=is_dense, hard_ste_mode=is_hard_from_scratch,
                                       hard_k=HARD_FROM_SCRATCH_K)
                lm_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

                if is_dense or is_hard_from_scratch:
                    # dense: no gate exists to be sparse. hard_from_scratch:
                    # cardinality is already exactly HARD_FROM_SCRATCH_K by
                    # construction (top-k hardening), so there is nothing for
                    # a soft sparsity-target loss to push toward -- and no
                    # CLHR dual soft/hard loss term either (this condition
                    # trains a single hard path, full stop). loss below
                    # therefore reduces to plain lm_loss.
                    sparsity_loss = torch.tensor(0.0, device=device)
                else:
                    sparsity_vals = []
                    for block in raw_model.blocks:
                        m = block.attn.last_soft_mask
                        if m is not None:
                            sparsity_vals.append(1.0 - m.mean())
                    if sparsity_vals:
                        avg_sparsity = torch.stack(sparsity_vals).mean()
                        gap = F.relu(torch.tensor(target_sparsity, device=device) - avg_sparsity)
                        sparsity_loss = gap ** 2
                    else:
                        sparsity_loss = torch.tensor(0.0, device=device)

                loss = lm_loss + lambda_sparse * sparsity_loss

                if use_rca and (gate_snapshots or is_contemporary_cl) and lambda_rca > 0:
                    snap = None
                    if condition in ("coherent_hard_oldest", "coherent_closedloop_hard"):
                        snap = min(gate_snapshots, key=lambda s: s["step"])
                    elif condition == "contemporary_replay":
                        snap = {"gate_states": [{
                            "W_gq.weight": b.attn.W_gq.weight.clone(),
                            "W_gk.weight": b.attn.W_gk.weight.clone(),
                        } for b in raw_model.blocks], "step": global_step}
                    elif condition != "contemporary_closedloop_hard" and gate_snapshots:
                        snap = gate_snapshots[global_step % len(gate_snapshots)]

                    if condition == "coherent_closedloop_hard":
                        # CLOSED-LOOP: historical gate computes masks ONLINE
                        # from the hard-path hidden state at each layer
                        from torch.utils.checkpoint import checkpoint as grad_ckpt

                        B, T = x.shape
                        pos = torch.arange(T, device=device).unsqueeze(0)
                        h_rca = raw_model.tok_emb(x) + raw_model.pos_emb(pos)
                        h_rca = raw_model.drop(h_rca)
                        causal = torch.tril(torch.ones(T, T, device=device)).unsqueeze(0).unsqueeze(0)

                        def closedloop_block(h_in, block, gs, causal_mask):
                            normed = block.ln1(h_in)
                            attn = block.attn
                            q = attn.W_q(normed).view(B, T, attn.n_heads, attn.d_head).transpose(1, 2)
                            kk = attn.W_k(normed).view(B, T, attn.n_heads, attn.d_head).transpose(1, 2)
                            v = attn.W_v(normed).view(B, T, attn.n_heads, attn.d_head).transpose(1, 2)
                            scores = torch.matmul(q, kk.transpose(-2, -1)) / math.sqrt(attn.d_head)
                            with torch.no_grad():
                                gq = F.linear(normed, gs["W_gq.weight"]).view(B, T, attn.n_heads, attn.d_gate).transpose(1, 2)
                                gk = F.linear(normed, gs["W_gk.weight"]).view(B, T, attn.n_heads, attn.d_gate).transpose(1, 2)
                                g_scores = torch.matmul(gq, gk.transpose(-2, -1)) / math.sqrt(attn.d_gate)
                                g_scores = g_scores.masked_fill(causal_mask == 0, float("-inf"))
                                actual_k = min(64, T)
                                _, topk_idx = torch.topk(g_scores, actual_k, dim=-1)
                                hard_mask = torch.zeros_like(g_scores)
                                hard_mask.scatter_(-1, topk_idx, 1.0)
                                hard_mask = hard_mask * causal_mask
                            scores = scores.masked_fill(hard_mask == 0, float("-inf"))
                            scores = scores.masked_fill(causal_mask == 0, float("-inf"))
                            w = F.softmax(scores, dim=-1)
                            w = torch.nan_to_num(w, nan=0.0)
                            out = torch.matmul(w, v)
                            out = out.transpose(1, 2).contiguous().view(B, T, -1)
                            out = attn.W_o(out)
                            h_out = h_in + out
                            h_out = h_out + block.ff(block.ln2(h_out))
                            return h_out

                        for li, block in enumerate(raw_model.blocks):
                            gs = snap["gate_states"][li]
                            h_rca = grad_ckpt(
                                closedloop_block, h_rca, block, gs, causal,
                                use_reentrant=False,
                            )

                        h_rca = raw_model.ln_f(h_rca)
                        logits_rca = raw_model.lm_head(h_rca)
                        rca_loss = F.cross_entropy(logits_rca.reshape(-1, logits_rca.size(-1)), y.reshape(-1))
                        loss = loss + lambda_rca * rca_loss
                    elif condition == "contemporary_closedloop_hard":
                        # CLOSED-LOOP with CURRENT gate (no history needed)
                        from torch.utils.checkpoint import checkpoint as grad_ckpt

                        B, T = x.shape
                        pos = torch.arange(T, device=device).unsqueeze(0)
                        h_rca = raw_model.tok_emb(x) + raw_model.pos_emb(pos)
                        h_rca = raw_model.drop(h_rca)
                        causal = torch.tril(torch.ones(T, T, device=device)).unsqueeze(0).unsqueeze(0)

                        def contemporary_cl_block(h_in, block, causal_mask):
                            normed = block.ln1(h_in)
                            attn = block.attn
                            q = attn.W_q(normed).view(B, T, attn.n_heads, attn.d_head).transpose(1, 2)
                            kk = attn.W_k(normed).view(B, T, attn.n_heads, attn.d_head).transpose(1, 2)
                            v = attn.W_v(normed).view(B, T, attn.n_heads, attn.d_head).transpose(1, 2)
                            scores = torch.matmul(q, kk.transpose(-2, -1)) / math.sqrt(attn.d_head)
                            with torch.no_grad():
                                gq = attn.W_gq(normed).view(B, T, attn.n_heads, attn.d_gate).transpose(1, 2)
                                gk = attn.W_gk(normed).view(B, T, attn.n_heads, attn.d_gate).transpose(1, 2)
                                g_scores = torch.matmul(gq, gk.transpose(-2, -1)) / math.sqrt(attn.d_gate)
                                g_scores = g_scores.masked_fill(causal_mask == 0, float("-inf"))
                                actual_k = min(64, T)
                                _, topk_idx = torch.topk(g_scores, actual_k, dim=-1)
                                hard_mask = torch.zeros_like(g_scores)
                                hard_mask.scatter_(-1, topk_idx, 1.0)
                                hard_mask = hard_mask * causal_mask
                            scores = scores.masked_fill(hard_mask == 0, float("-inf"))
                            scores = scores.masked_fill(causal_mask == 0, float("-inf"))
                            w = F.softmax(scores, dim=-1)
                            w = torch.nan_to_num(w, nan=0.0)
                            out = torch.matmul(w, v)
                            out = out.transpose(1, 2).contiguous().view(B, T, -1)
                            out = attn.W_o(out)
                            h_out = h_in + out
                            h_out = h_out + block.ff(block.ln2(h_out))
                            return h_out

                        for block in raw_model.blocks:
                            h_rca = grad_ckpt(
                                contemporary_cl_block, h_rca, block, causal,
                                use_reentrant=False,
                            )

                        h_rca = raw_model.ln_f(h_rca)
                        logits_rca = raw_model.lm_head(h_rca)
                        rca_loss = F.cross_entropy(logits_rca.reshape(-1, logits_rca.size(-1)), y.reshape(-1))
                        loss = loss + lambda_rca * rca_loss
                    else:
                        # OPEN-LOOP: extract masks from soft trajectory, apply in separate forward
                        hard_conditions = ("coherent_hard_oldest", "shuffled_historical")
                        hist_masks = []
                        with torch.no_grad():
                            pos = torch.arange(x.shape[1], device=device).unsqueeze(0)
                            h = raw_model.tok_emb(x) + raw_model.pos_emb(pos)
                            h = raw_model.drop(h)
                            causal = torch.tril(torch.ones(x.shape[1], x.shape[1], device=device)).unsqueeze(0).unsqueeze(0)
                            for li, block in enumerate(raw_model.blocks):
                                normed = block.ln1(h)
                                attn = block.attn
                                gs = snap["gate_states"][li]
                                gq = F.linear(normed, gs["W_gq.weight"]).view(
                                    x.shape[0], x.shape[1], attn.n_heads, attn.d_gate
                                ).transpose(1, 2)
                                gk = F.linear(normed, gs["W_gk.weight"]).view(
                                    x.shape[0], x.shape[1], attn.n_heads, attn.d_gate
                                ).transpose(1, 2)
                                g_scores = torch.matmul(gq, gk.transpose(-2, -1)) / math.sqrt(attn.d_gate)
                                mask = torch.sigmoid(g_scores)
                                if condition == "shuffled_historical":
                                    B_m, Tk = mask.shape[0], mask.shape[-1]
                                    for b in range(B_m):
                                        perm = torch.randperm(Tk, device=mask.device)
                                        mask[b] = mask[b, :, :, perm]
                                if condition in hard_conditions:
                                    actual_k = min(64, mask.shape[-1])
                                    _, topk_idx = torch.topk(mask, actual_k, dim=-1)
                                    hard = torch.zeros_like(mask)
                                    hard.scatter_(-1, topk_idx, 1.0)
                                    mask = hard
                                hist_masks.append(mask)
                                h = block(h, causal)

                        logits_rca, _ = model(x, forced_masks=hist_masks, use_checkpoint=False)
                        rca_loss = F.cross_entropy(logits_rca.reshape(-1, logits_rca.size(-1)), y.reshape(-1))
                        loss = loss + lambda_rca * rca_loss

                if normalize_loss and lambda_rca > 0 and use_rca:
                    loss = loss / (1.0 + lambda_rca)
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

        if use_rca and global_step % snapshot_interval == 0:
            gate_snapshots.append({
                "step": global_step,
                "gate_states": [{
                    "W_gq.weight": b.attn.W_gq.weight.clone(),
                    "W_gk.weight": b.attn.W_gk.weight.clone(),
                } for b in raw_model.blocks],
            })
            if len(gate_snapshots) > n_snapshots:
                gate_snapshots.pop(0)

        if is_master and global_step % 100 == 0:
            avg = running_loss / running_count
            elapsed = time.time() - t0
            tps = tokens_seen / elapsed if elapsed > 0 else 0
            mem = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0
            print(f"  step {global_step:>6}/{total_steps}: loss={avg:.4f} "
                  f"tokens={tokens_seen/1e9:.2f}B tps={tps:.0f} mem={mem:.1f}GB "
                  f"({elapsed:.0f}s)", flush=True)
            running_loss = 0.0
            running_count = 0

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
                    print(f"    [checkpoint saved: {save_path.name}]", flush=True)
                    break

            if benchmark and global_step >= 500:
                elapsed = time.time() - t0
                tps = tokens_seen / elapsed if elapsed > 0 else 0
                mem = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0
                print(f"\n  BENCHMARK RESULT:")
                print(f"    Tokens/second: {tps:.0f}")
                print(f"    Peak memory: {mem:.1f} GB")
                print(f"    Steps/second: {global_step / elapsed:.2f}")
                if ddp:
                    dist.destroy_process_group()
                return

            if tokens_seen >= next_ckpt_tokens and next_ckpt_tokens > 0:
                save_path = ckpt_path / f"tokens_{tokens_seen}.pt"
                torch.save({"model": raw_model.state_dict(), "step": global_step,
                             "tokens_seen": tokens_seen}, save_path)
                print(f"    [checkpoint at {tokens_seen/1e9:.2f}B tokens]", flush=True)
                for m in CHECKPOINT_TOKEN_MILESTONES:
                    if m > tokens_seen:
                        next_ckpt_tokens = m
                        break
                else:
                    next_ckpt_tokens = float("inf")

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

    if is_master:
        # Final checkpoint
        torch.save({"model": raw_model.state_dict(), "step": global_step,
                     "tokens_seen": tokens_seen},
                   ckpt_path / f"final.pt")

        # Evaluation — use smaller batch to avoid OOM from oracle mask accumulation
        print("\n  Final evaluation...")
        val_ds = load_val_data(data_dir, seq_len)
        eval_batch = max(1, micro_batch // 4)
        val_loader = DataLoader(val_ds, batch_size=eval_batch, shuffle=False,
                                num_workers=0, drop_last=True)
        metrics = run_full_evaluation(raw_model, val_loader, device, native_hard_ste=is_hard_from_scratch)

        results = {
            "condition": condition,
            "seed": seed,
            "lambda_rca": lambda_rca,
            "normalize_loss": normalize_loss,
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


def main():
    parser = argparse.ArgumentParser(description="300M sparse-attention scale confirmation")
    parser.add_argument("--condition", choices=CONDITIONS, default="standard")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--checkpoint-dir", type=str, default="./ckpts_sparse_300m")
    parser.add_argument("--output", type=str, default="results/sparse_300m.json")
    parser.add_argument("--total-tokens", type=int, default=4_000_000_000)
    parser.add_argument("--micro-batch", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--lambda-rca", type=float, default=0.3,
                        help="Hard-trajectory loss weight (default: 0.3)")
    parser.add_argument("--normalize-loss", action="store_true",
                        help="Normalize combined loss by 1/(1+lambda) for constant gradient scale")
    args = parser.parse_args()

    if args.benchmark:
        train_experiment("standard", 42, args.data_dir, args.checkpoint_dir, args.output,
                         total_tokens=args.total_tokens, benchmark=True)
    else:
        train_experiment(args.condition, args.seed, args.data_dir, args.checkpoint_dir,
                         args.output, total_tokens=args.total_tokens,
                         micro_batch=args.micro_batch, grad_accum=args.grad_accum,
                         lambda_rca=args.lambda_rca, normalize_loss=args.normalize_loss)


if __name__ == "__main__":
    main()
