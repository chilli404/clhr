"""Sparse-attention training on FineWeb-Edu / C4: corpus-transfer experiment.

Demonstrates that routing absorption and CLHR transfer beyond WikiText-103
by training the same gated sparse-attention architecture on larger, more
diverse corpora.

Architecture is self-contained (same as sparse_attention_300m.py):
  GatedSparseAttention -> TransformerBlock -> SparseTransformer
with log-additive gating via W_gq / W_gk projections.

Model sizes:
  100m: d=768,  12 layers, 12 heads, d_ff=3072  (~100M params)
  300m: d=1024, 20 layers, 16 heads, d_ff=4096  (~300M params)

Three conditions:
  standard                    -- normal sparse training (baseline)
  coherent_closedloop_hard    -- dual-loss with hardened historical gates
  contemporary_closedloop_hard -- dual-loss with current gate (detached) on hard path

Usage:
    # Prepare shards first (one-time)
    python -c "from src.data_loading import prepare_fineweb_shards; \
               prepare_fineweb_shards('./data', max_shards=20)"

    # Train 100M on FineWeb-Edu
    python src/sparse_attention_fineweb.py \
        --condition coherent_closedloop_hard \
        --corpus fineweb-edu \
        --model-size 100m \
        --seed 42 \
        --data-dir ./data \
        --output results/fineweb_clhr_100m_s42.json \
        --total-tokens 2000000000

    # Train 300M on C4
    python src/sparse_attention_fineweb.py \
        --condition standard \
        --corpus c4 \
        --model-size 300m \
        --seed 42 \
        --data-dir ./data \
        --output results/c4_standard_300m_s42.json
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.data.distributed import DistributedSampler

from data_loading import load_corpus

# ═══════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════

CONDITIONS = ["standard", "coherent_closedloop_hard", "contemporary_closedloop_hard", "native_hard"]

MODEL_CONFIGS = {
    "100m": dict(
        vocab_size=50257, d_model=768, n_heads=12, n_layers=12,
        d_ff=3072, d_gate=32, max_seq_len=512, dropout=0.1,
    ),
    "300m": dict(
        vocab_size=50257, d_model=1024, n_heads=16, n_layers=20,
        d_ff=4096, d_gate=32, max_seq_len=512, dropout=0.1,
    ),
    # ~1.05B params (tok_emb/lm_head tied): scaled proportionally from 300m,
    # not an exact 1.0B target -- see tests/test_sparse_attention_fineweb.py.
    "1b": dict(
        vocab_size=50257, d_model=2048, n_heads=16, n_layers=18,
        d_ff=8192, d_gate=32, max_seq_len=512, dropout=0.1,
    ),
}

CHECKPOINT_TOKEN_MILESTONES = [
    250_000_000, 500_000_000, 1_000_000_000,
    1_500_000_000, 2_000_000_000, 2_500_000_000,
    3_000_000_000, 3_500_000_000, 4_000_000_000,
    6_000_000_000,
]


# ═══════════════════════════════════════════════════════════════════════
# Model (self-contained, same architecture as sparse_attention_300m.py)
# ═══════════════════════════════════════════════════════════════════════

class GatedSparseAttention(nn.Module):
    """Sparse attention with learned log-additive gates (W_gq, W_gk)."""

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
            attn_bias = causal_mask.masked_fill(
                causal_mask == 0, float("-inf")
            ).masked_fill(causal_mask == 1, 0.0)
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
            q, k, v, attn_mask=attn_bias,
            dropout_p=self.dropout.p if self.training else 0.0,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.W_o(out)

    def _forward_hard_ste(self, x, q, k, v, causal_mask, hard_k):
        """`native_hard` condition: gate is discrete (top-k hardened) from
        step 0, no soft warm-up, no CLHR dual soft/hard loss mixing.

        Ported verbatim (mechanism, not just intent) from
        sparse_attention_300m.py's GatedSparseAttention._forward_hard_ste --
        see that docstring for the full rationale. Forward value is
        DELIBERATELY bit-identical to forward_closed_loop_gate_hard's
        closed-loop deployment computation (additive -inf masking of the
        final attention weights), which is what makes G_CL ~0 "by
        construction" for this condition. Gradient to W_gq/W_gk flows via a
        straight-through estimator on the final attention WEIGHTS (not a
        multiplicative mask on raw scores -- see the 300m.py docstring for
        why that convention would leak attention mass and defeat the
        near-zero-G_CL sanity check this baseline exists to provide).
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

        raw_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_h)

        hard_scores = raw_scores.masked_fill(hard_mask == 0, float("-inf"))
        hard_scores = hard_scores.masked_fill(causal_mask == 0, float("-inf"))
        w_hard = F.softmax(hard_scores, dim=-1)
        w_hard = torch.nan_to_num(w_hard, nan=0.0)

        soft_bias = torch.log(soft_mask.clamp(min=1e-6)).masked_fill(causal_mask == 0, float("-inf"))
        soft_scores = raw_scores + soft_bias
        w_soft = F.softmax(soft_scores, dim=-1)
        w_soft = torch.nan_to_num(w_soft, nan=0.0)

        # Straight-through: forward value is exactly w_hard, backward flows
        # entirely through w_soft.
        w = w_hard.detach() - w_soft.detach() + w_soft

        out = torch.matmul(w.to(v.dtype), v)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.W_o(out)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: LN -> Attn + residual -> LN -> FFN + residual."""

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


class SparseTransformer(nn.Module):
    """Gated sparse-attention decoder-only transformer.

    Identical architecture to SparseTransformer300M in sparse_attention_300m.py,
    but constructor accepts arbitrary sizes via MODEL_CONFIGS.
    """

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
        self.lm_head.weight = self.tok_emb.weight  # weight tying

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

        if (not hasattr(self, "_causal_cache")
                or self._causal_cache.shape[-1] != T
                or self._causal_cache.device != device):
            self._causal_cache = torch.tril(
                torch.ones(T, T, device=device)
            ).unsqueeze(0).unsqueeze(0)
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
# Evaluation helpers
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_nll(model, loader, device, max_batches=100, forced_mask_fn=None):
    """Evaluate native (soft) NLL, or with forced masks if provided."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        ids = batch["input_ids"].to(device) if isinstance(batch, dict) else batch.to(device)
        x, y = ids[:, :-1], ids[:, 1:]
        if forced_mask_fn is not None:
            masks = forced_mask_fn(model, x)
            logits, _ = model(x, forced_masks=masks, use_checkpoint=False)
        else:
            logits, _ = model(x, use_checkpoint=False)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum"
        )
        total_loss += loss.item()
        total_tokens += y.numel()
    return total_loss / total_tokens if total_tokens > 0 else float("inf")


def forward_closed_loop_gate_hard(model, input_ids, k=64):
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


def eval_closed_loop(model, loader, device, k=64, max_batches=100):
    """Evaluate with closed-loop gate-hard deployment."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            ids = batch["input_ids"].to(device) if isinstance(batch, dict) else batch.to(device)
            x, y = ids[:, :-1], ids[:, 1:]
            logits = forward_closed_loop_gate_hard(model, x, k=k)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum"
            )
            total_loss += loss.item()
            total_tokens += y.numel()
    return total_loss / total_tokens if total_tokens > 0 else float("inf")


@torch.no_grad()
def _evaluate_nll_hard_ste(model, loader, device, hard_k=64, max_batches=100):
    """Native-mode NLL for condition="native_hard" -- forward via hard_ste_mode."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        ids = batch["input_ids"].to(device) if isinstance(batch, dict) else batch.to(device)
        x, y = ids[:, :-1], ids[:, 1:]
        logits, _ = model(x, hard_ste_mode=True, hard_k=hard_k, use_checkpoint=False)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum"
        )
        total_loss += loss.item()
        total_tokens += y.numel()
    return total_loss / total_tokens if total_tokens > 0 else float("inf")


def run_full_evaluation(model, val_loader, device, k=64, native_hard_ste=False):
    """Full evaluation suite: native, closed-loop hard, random-hard, gate utility.

    native_hard_ste=True (for condition="native_hard"): there is no separate
    soft mode to evaluate, so native_nll uses the same hard_ste_mode forward
    the model was trained with. By design this is bit-identical to the
    closed-loop-hard eval below (see
    tests/test_sparse_attention_fineweb.py::
    test_hard_ste_forward_bit_identical_to_closed_loop_gate_hard), so G_CL~0
    here reflects "no mismatch to begin with," not a measurement artifact.
    """
    n_layers = model.n_layers
    n_heads = model.n_heads

    if native_hard_ste:
        native_nll = _evaluate_nll_hard_ste(model, val_loader, device, hard_k=k)
    else:
        native_nll = evaluate_nll(model, val_loader, device)

    # Closed-loop gate-hard NLL
    cl_gate_nll = eval_closed_loop(model, val_loader, device, k=k)

    # Random-hard NLL (multiple draws)
    swap_nlls = []
    for _ in range(10):
        def random_mask_fn(m, x, _n_layers=n_layers, _n_heads=n_heads, _k=k):
            B, T = x.shape
            masks = []
            for _ in range(_n_layers):
                actual_k = min(_k, T)
                rand = torch.rand(B, _n_heads, T, T, device=x.device)
                _, idx = torch.topk(rand, actual_k, dim=-1)
                mask = torch.zeros_like(rand)
                mask.scatter_(-1, idx, 1.0)
                causal = torch.tril(
                    torch.ones(T, T, device=x.device)
                ).unsqueeze(0).unsqueeze(0)
                masks.append(mask * causal)
            return masks

        swap_nlls.append(
            evaluate_nll(model, val_loader, device,
                         forced_mask_fn=random_mask_fn, max_batches=50)
        )

    random_hard_nll = float(np.mean(swap_nlls))

    # Derived metrics
    G_CL = cl_gate_nll - native_nll
    gate_utility = random_hard_nll - cl_gate_nll

    return {
        "native_nll": round(native_nll, 6),
        "native_ppl": round(math.exp(native_nll), 2),
        "closed_loop_hard_nll": round(cl_gate_nll, 6),
        "G_CL": round(G_CL, 6),
        "gate_utility": round(gate_utility, 6),
        "random_hard_nll_mean": round(random_hard_nll, 6),
        "random_hard_nll_std": round(float(np.std(swap_nlls)), 6),
    }


# ═══════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════

def train_experiment(
    condition: str,
    seed: int,
    corpus: str,
    model_size: str,
    data_dir: str,
    checkpoint_dir: str,
    output_path: str,
    total_tokens: int = 2_000_000_000,
    micro_batch: int = 32,
    seq_len: int = 512,
    lr: float = 3e-4,
    target_sparsity: float = 0.875,
    lambda_rca: float = 1.0,
    lambda_sparse: float = 1.0,
):
    # ── DDP setup ─────────────────────────────────────────────────────
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

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if is_master:
        print(f"Device: {device}, dtype: {dtype}")
        print(f"  condition={condition}, corpus={corpus}, model_size={model_size}, seed={seed}")
        print(f"  DDP: {ddp}, world_size={world_size}")

    # ── Model ─────────────────────────────────────────────────────────
    cfg = MODEL_CONFIGS[model_size].copy()
    cfg["max_seq_len"] = seq_len
    model = SparseTransformer(**cfg).to(device).to(dtype)
    raw_model = model

    if is_master:
        print(f"  Parameters: {model.count_parameters() / 1e6:.1f}M")

    use_compile = device.type == "cuda" and hasattr(torch, "compile")
    if use_compile:
        model = torch.compile(model)
        if is_master:
            print("  torch.compile enabled")

    if ddp:
        model = DDP(model, device_ids=[local_rank], gradient_as_bucket_view=True)

    # ── Data ──────────────────────────────────────────────────────────
    use_rca = condition not in ("standard", "native_hard")
    is_native_hard = condition == "native_hard"

    train_ds = load_corpus(
        corpus, data_dir, seq_len=seq_len, split="train",
        tokenizer_name="gpt2", cycling=True,
        rank=rank, world_size=world_size,
    )
    is_iterable = isinstance(train_ds, IterableDataset)

    if is_iterable:
        # ShardedTokenDataset handles DDP sharding internally
        sampler = None
        train_loader = DataLoader(
            train_ds, batch_size=micro_batch,
            num_workers=2, pin_memory=True, drop_last=True,
        )
    else:
        sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank, shuffle=True
        ) if ddp else None
        train_loader = DataLoader(
            train_ds, batch_size=micro_batch,
            shuffle=(sampler is None), sampler=sampler,
            num_workers=2, pin_memory=True, drop_last=True,
        )

    # ── Optimizer + scheduler ─────────────────────────────────────────
    # Gradient accumulation to maintain ~65K tokens/step
    target_tokens_per_step = 65536
    grad_accum = max(1, target_tokens_per_step // (micro_batch * seq_len * world_size))
    tokens_per_step = micro_batch * seq_len * grad_accum * world_size
    total_steps = total_tokens // tokens_per_step

    if is_master:
        print(f"  micro_batch={micro_batch}, grad_accum={grad_accum}")
        print(f"  Tokens/step: {tokens_per_step:,}, total steps: {total_steps:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=0.1, betas=(0.9, 0.95)
    )

    warmup_steps = 2000

    def cosine_lr(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, cosine_lr)

    # ── Gate snapshot buffer (for CLHR) ───────────────────────────────
    gate_snapshots = []
    snapshot_interval = 1000
    n_snapshots = 5

    # ── Checkpoint directory ──────────────────────────────────────────
    tag = f"fineweb_{corpus}_{condition}_{model_size}_s{seed}"
    ckpt_path = Path(checkpoint_dir) / tag
    ckpt_path.mkdir(parents=True, exist_ok=True)

    # ── Resume ────────────────────────────────────────────────────────
    final_file = ckpt_path / "final.pt"
    resume_file = ckpt_path / "latest.pt"
    global_step = 0
    tokens_seen = 0
    if final_file.exists():
        ckpt = torch.load(final_file, map_location=device, weights_only=False)
        raw_model.load_state_dict(ckpt["model"])
        global_step = total_steps
        tokens_seen = ckpt.get("tokens_seen", total_steps * tokens_per_step)
        if is_master:
            print(f"  Found final.pt — skipping training, proceeding to eval")
    elif resume_file.exists():
        ckpt = torch.load(resume_file, map_location=device, weights_only=False)
        raw_model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        global_step = ckpt["step"]
        tokens_seen = ckpt.get("tokens_seen", global_step * tokens_per_step)
        gate_snapshots = ckpt.get("gate_snapshots", [])
        if is_master:
            print(f"  Resumed from step {global_step}, {tokens_seen / 1e9:.2f}B tokens")

    # ── Training loop ─────────────────────────────────────────────────
    train_iter = iter(train_loader)
    t0 = time.time()
    running_loss = 0.0
    running_count = 0

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

            ids = batch["input_ids"].to(device) if isinstance(batch, dict) else batch.to(device)
            x, y = ids[:, :-1], ids[:, 1:]

            with torch.amp.autocast("cuda", dtype=dtype):
                logits, masks = model(x, hard_ste_mode=is_native_hard, hard_k=64)
                lm_loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), y.reshape(-1)
                )

                # Sparsity penalty
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

                # ── CLHR dual-loss ────────────────────────────────────
                use_historical = (condition == "coherent_closedloop_hard")
                can_run_rca = use_rca and lambda_rca > 0
                if use_historical:
                    can_run_rca = can_run_rca and len(gate_snapshots) > 0

                if can_run_rca:
                    snap = None
                    if use_historical:
                        snap = min(gate_snapshots, key=lambda s: s["step"])

                    B, T = x.shape
                    pos = torch.arange(T, device=device).unsqueeze(0)
                    h_rca = raw_model.tok_emb(x) + raw_model.pos_emb(pos)
                    h_rca = raw_model.drop(h_rca)
                    causal = torch.tril(
                        torch.ones(T, T, device=device)
                    ).unsqueeze(0).unsqueeze(0)

                    def closedloop_block(h_in, block, gs, causal_mask,
                                         _use_hist=use_historical):
                        normed = block.ln1(h_in)
                        attn = block.attn

                        q = attn.W_q(normed).view(B, T, attn.n_heads, attn.d_head).transpose(1, 2)
                        kk = attn.W_k(normed).view(B, T, attn.n_heads, attn.d_head).transpose(1, 2)
                        v = attn.W_v(normed).view(B, T, attn.n_heads, attn.d_head).transpose(1, 2)
                        scores = torch.matmul(q, kk.transpose(-2, -1)) / math.sqrt(attn.d_head)

                        with torch.no_grad():
                            if _use_hist:
                                gq = F.linear(normed, gs["W_gq.weight"]).view(
                                    B, T, attn.n_heads, attn.d_gate
                                ).transpose(1, 2)
                                gk = F.linear(normed, gs["W_gk.weight"]).view(
                                    B, T, attn.n_heads, attn.d_gate
                                ).transpose(1, 2)
                            else:
                                gq = attn.W_gq(normed).view(
                                    B, T, attn.n_heads, attn.d_gate
                                ).transpose(1, 2)
                                gk = attn.W_gk(normed).view(
                                    B, T, attn.n_heads, attn.d_gate
                                ).transpose(1, 2)
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
                        gs = snap["gate_states"][li] if snap else None
                        h_rca = grad_checkpoint(
                            closedloop_block, h_rca, block, gs, causal,
                            use_reentrant=False,
                        )

                    h_rca = raw_model.ln_f(h_rca)
                    logits_rca = raw_model.lm_head(h_rca)
                    rca_loss = F.cross_entropy(
                        logits_rca.reshape(-1, logits_rca.size(-1)), y.reshape(-1)
                    )

                    loss = (loss + lambda_rca * rca_loss) / (1.0 + lambda_rca)

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

        # ── Gate snapshot ─────────────────────────────────────────────
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

        # ── Logging ───────────────────────────────────────────────────
        if is_master and global_step % 100 == 0:
            avg = running_loss / running_count
            elapsed = time.time() - t0
            tps = tokens_seen / elapsed if elapsed > 0 else 0
            mem = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0
            print(
                f"  step {global_step:>6}/{total_steps}: loss={avg:.4f} "
                f"tokens={tokens_seen / 1e9:.2f}B tps={tps:.0f} mem={mem:.1f}GB "
                f"({elapsed:.0f}s)", flush=True,
            )
            running_loss = 0.0
            running_count = 0

        # ── Token-milestone checkpoints ───────────────────────────────
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

    # ══════════════════════════════════════════════════════════════════
    # Final evaluation
    # ══════════════════════════════════════════════════════════════════

    elapsed = time.time() - t0
    gpu_hours = (elapsed * world_size) / 3600.0

    if is_master:
        # Save final model
        torch.save(
            {"model": raw_model.state_dict(), "step": global_step, "tokens_seen": tokens_seen},
            ckpt_path / "final.pt",
        )

        print("\n  Final evaluation...")

        eval_batch = max(1, micro_batch // 4)

        # ── Primary evaluation on held-out FineWeb-Edu shard ─────────
        # Uses fineweb_edu_eval/ which contains a shard that was never
        # included in the training set (downloaded beyond the 25 training
        # shards from the FineWeb-Edu 10BT sample).
        metrics = {}
        try:
            val_ds = load_corpus(
                corpus, data_dir, seq_len=seq_len, split="validation",
                tokenizer_name="gpt2", cycling=False,
            )
            val_loader = DataLoader(val_ds, batch_size=eval_batch,
                                    shuffle=False, num_workers=0,
                                    drop_last=True)
            metrics = run_full_evaluation(raw_model, val_loader, device, k=64,
                                           native_hard_ste=is_native_hard)
        except Exception as e:
            print(f"  Primary eval failed: {e}")

        # ── Assemble results ──────────────────────────────────────────
        results = {
            "method": "corpus_transfer",
            "condition": condition,
            "corpus": corpus,
            "model_size": model_size,
            "seed": seed,
            "lambda_rca": lambda_rca,
            "total_tokens": tokens_seen,
            "total_steps": global_step,
            "model_params": raw_model.count_parameters(),
            **metrics,
            "training_elapsed_seconds": round(elapsed, 1),
            "training_gpu_hours": round(gpu_hours, 2),
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n  Saved to {output_path}")
        for k, v in metrics.items():
            print(f"    {k}: {v}")

    if ddp:
        dist.destroy_process_group()


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Sparse-attention corpus-transfer experiment (FineWeb-Edu / C4)"
    )
    parser.add_argument(
        "--condition", choices=CONDITIONS, default="standard",
        help="Training condition",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--corpus", choices=["wikitext-103", "fineweb-edu", "c4"], default="fineweb-edu",
        help="Training corpus",
    )
    parser.add_argument("--data-dir", type=str, required=True, help="Data cache directory")
    parser.add_argument("--output", type=str, default="results/corpus_transfer.json")
    parser.add_argument("--checkpoint-dir", type=str, default="./ckpts_fineweb")
    parser.add_argument(
        "--model-size", choices=list(MODEL_CONFIGS.keys()), default="100m",
        help="Model size (default: 100m)",
    )
    parser.add_argument(
        "--total-tokens", type=int, default=2_000_000_000,
        help="Total training tokens (default: 2B)",
    )
    parser.add_argument(
        "--lambda-rca", type=float, default=1.0,
        help="CLHR loss weight (default: 1.0)",
    )
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--micro-batch", type=int, default=32)

    args = parser.parse_args()

    train_experiment(
        condition=args.condition,
        seed=args.seed,
        corpus=args.corpus,
        model_size=args.model_size,
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        output_path=args.output,
        total_tokens=args.total_tokens,
        micro_batch=args.micro_batch,
        seq_len=args.seq_len,
        lambda_rca=args.lambda_rca,
    )


if __name__ == "__main__":
    main()
