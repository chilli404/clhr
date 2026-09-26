"""MoBA (Mixture of Block Attention) baseline for the CLHR reconciliation ladder.

Faithful from-scratch port of MoonshotAI/MoBA's block-sparse attention
(moba/moba_naive.py, verified against source) into this project's own
Transformer scaffold. Answers: does a REAL, widely-used sparse-attention
scheme (not this repo's own self-implemented gate) show the soft-train/
hard-deploy generalization gap CLHR targets, or does native block-hard
training avoid it by construction?

MoBA's gating is parameter-less: block relevance = query dot mean-pooled
block key (computed in fp32), no learned gate weights. Block selection is
hard top-k (no differentiable relaxation) with the query's own (current)
block always force-included via a +inf gate-score trick; a final,
unconditional token-level causal mask is the safety net that keeps
causality correct even when top_k_blocks >= n_blocks (dense-equivalent).

Three training conditions, mirroring this project's G_CL framework applied
to MoBA's own mechanism:
  standard      -- native block-hard training from step 0 (MoBA's own recipe)
  dense_switch  -- train dense (full causal attention), deploy block-hard
  clhr          -- mix dense and hard losses during training (the fix)
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

CONDITIONS = ["standard", "dense_switch", "clhr"]


# ─────────────────────────────────────────────────────────────────────────
# Rotary positional encoding
# ─────────────────────────────────────────────────────────────────────────

class RotaryPositionalEncoding(nn.Module):
    def __init__(self, d_head: int, max_seq_len: int = 8192, base: float = 10000.0):
        super().__init__()
        assert d_head % 2 == 0, "d_head must be even for RoPE"
        self.d_head = d_head
        inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device, dtype):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(device))
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)


def apply_rotary_emb(x: torch.Tensor, rope: RotaryPositionalEncoding) -> torch.Tensor:
    """x: [B, H, T, Dh] -> [B, H, T, Dh]."""
    _, _, seq_len, d_head = x.shape
    cos, sin = rope(seq_len, x.device, x.dtype)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    x1, x2 = x[..., : d_head // 2], x[..., d_head // 2 :]
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos + rotated * sin


# ─────────────────────────────────────────────────────────────────────────
# Core MoBA mechanism: parameter-less block gating
# ─────────────────────────────────────────────────────────────────────────

def moba_block_means(k: torch.Tensor, block_size: int) -> torch.Tensor:
    """Mean-pool k over contiguous blocks along the sequence dim.

    k: [B, H, T, Dh] any dtype. Returns fp32 [B, H, num_blocks, Dh].
    Ragged last block uses its true element count, not zero-padded.
    """
    _, _, seq_len, _ = k.shape
    k32 = k.float()
    num_blocks = math.ceil(seq_len / block_size)
    means = []
    for i in range(num_blocks):
        start = i * block_size
        end = min(seq_len, start + block_size)
        means.append(k32[:, :, start:end, :].mean(dim=2, keepdim=True))
    return torch.cat(means, dim=2)


def _blockwise_topk_bias(
    raw_scores: torch.Tensor, block_size: int, top_k_blocks: int, seq_len: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shared block-selection pipeline: force causal/own-block, top-k, expand.

    raw_scores: [B, H, T, N] arbitrary real-valued block-relevance scores.
    Returns (additive token-level bias [B,H,T,T] in {0,-inf}, block-level
    selection mask [B,H,T,N] bool).
    """
    num_blocks = raw_scores.shape[-1]
    scores = raw_scores.clone()
    for i in range(num_blocks):
        block_start = i * block_size
        block_end = min(seq_len, block_start + block_size)
        scores[:, :, :block_end, i] = float("-inf")
        scores[:, :, block_start:block_end, i] = float("inf")

    k_eff = min(top_k_blocks, num_blocks)
    topk_val, topk_idx = torch.topk(scores, k=k_eff, dim=-1, largest=True, sorted=False)
    topk_min, _ = topk_val.min(dim=-1)
    need_attend = scores >= topk_min.unsqueeze(-1)
    idx_mask = torch.zeros_like(need_attend, dtype=torch.bool)
    idx_mask.scatter_(-1, topk_idx, True)
    need_attend = need_attend & idx_mask

    bias = torch.where(need_attend, torch.zeros_like(scores), torch.full_like(scores, float("-inf")))
    bias = bias.repeat_interleave(block_size, dim=-1)[:, :, :, :seq_len]

    causal = torch.ones(seq_len, seq_len, dtype=torch.bool, device=raw_scores.device).tril()
    bias = bias.masked_fill(~causal.unsqueeze(0).unsqueeze(0), float("-inf"))
    return bias, need_attend


@torch.no_grad()
def compute_moba_gate_bias(
    q: torch.Tensor, k: torch.Tensor, block_size: int, top_k_blocks: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """q, k: [B, H, T, Dh]. Returns (bias [B,H,T,T], block_mask [B,H,T,N]), detached."""
    key_gate_weight = moba_block_means(k, block_size)  # [B,H,N,Dh] fp32
    q32 = q.float()
    raw_scores = torch.einsum("bhtd,bhnd->bhtn", q32, key_gate_weight)
    bias, block_mask = _blockwise_topk_bias(raw_scores, block_size, top_k_blocks, q.shape[2])
    return bias.detach(), block_mask.detach()


def _dense_causal_bias(batch: int, heads: int, seq_len: int, causal_mask: torch.Tensor) -> torch.Tensor:
    zeros = torch.zeros(batch, heads, seq_len, seq_len, device=causal_mask.device)
    neg_inf = torch.full((batch, heads, seq_len, seq_len), float("-inf"), device=causal_mask.device)
    return torch.where(causal_mask.bool(), zeros, neg_inf)


# ─────────────────────────────────────────────────────────────────────────
# MoBAAttention: parameter-less gating, standard Q/K/V/O projections only
# ─────────────────────────────────────────────────────────────────────────

class MoBAAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, block_size: int, top_k_blocks: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.block_size = block_size
        self.top_k_blocks = top_k_blocks
        self.dropout_p = dropout

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.last_block_mask: torch.Tensor | None = None

    def forward(
        self,
        x: torch.Tensor,
        rope: RotaryPositionalEncoding,
        causal_mask: torch.Tensor,
        dense_mode: bool = False,
        forced_token_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        q = self.W_q(x).view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = self.W_k(x).view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = self.W_v(x).view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        q = apply_rotary_emb(q, rope)
        k = apply_rotary_emb(k, rope)

        if forced_token_bias is not None:
            bias = forced_token_bias
            self.last_block_mask = None
        elif dense_mode:
            bias = _dense_causal_bias(batch, self.n_heads, seq_len, causal_mask)
            self.last_block_mask = None
        else:
            bias, block_mask = compute_moba_gate_bias(q, k, self.block_size, self.top_k_blocks)
            self.last_block_mask = block_mask

        dropout_p = self.dropout_p if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=dropout_p)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        return self.W_o(out)


class MoBATransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, block_size: int, top_k_blocks: int, dropout: float = 0.1):
        super().__init__()
        self.attn = MoBAAttention(d_model, n_heads, block_size, top_k_blocks, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        rope: RotaryPositionalEncoding,
        causal_mask: torch.Tensor,
        dense_mode: bool = False,
        forced_token_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attn_out = self.attn(self.ln1(x), rope, causal_mask, dense_mode=dense_mode, forced_token_bias=forced_token_bias)
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ff(self.ln2(x)))
        return x


class MoBATransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        block_size: int,
        top_k_blocks: int,
        max_seq_len: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [
                MoBATransformerBlock(d_model, n_heads, d_ff, block_size, top_k_blocks, dropout)
                for _ in range(n_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying
        self.rope = RotaryPositionalEncoding(d_model // n_heads, max_seq_len)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        dense_mode: bool = False,
        forced_token_biases: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor | None]]:
        batch, seq_len = x.shape
        h = self.dropout(self.tok_emb(x))
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).unsqueeze(0).unsqueeze(0)

        masks: list[torch.Tensor | None] = []
        for i, block in enumerate(self.blocks):
            fb = forced_token_biases[i] if forced_token_biases is not None else None
            h = block(h, self.rope, causal_mask, dense_mode=dense_mode, forced_token_bias=fb)
            masks.append(block.attn.last_block_mask)

        h = self.ln_f(h)
        logits = self.lm_head(h)
        return logits, masks

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ─────────────────────────────────────────────────────────────────────────
# Open-loop / closed-loop trajectory utilities
# ─────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def record_moba_biases_from_trajectory(
    model: MoBATransformer, x: torch.Tensor, trajectory_dense_mode: bool = True
) -> list[torch.Tensor]:
    """Record the per-layer attention bias that WOULD be computed under a
    given forward trajectory (dense or hard), for later frozen replay."""
    model.eval()
    batch, seq_len = x.shape
    h = model.tok_emb(x)
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).unsqueeze(0).unsqueeze(0)

    biases = []
    for block in model.blocks:
        attn = block.attn
        ln_h = block.ln1(h)
        q = attn.W_q(ln_h).view(batch, seq_len, attn.n_heads, attn.d_head).transpose(1, 2)
        k = attn.W_k(ln_h).view(batch, seq_len, attn.n_heads, attn.d_head).transpose(1, 2)
        q = apply_rotary_emb(q, model.rope)
        k = apply_rotary_emb(k, model.rope)
        if trajectory_dense_mode:
            bias = _dense_causal_bias(batch, attn.n_heads, seq_len, causal_mask)
        else:
            bias, _ = compute_moba_gate_bias(q, k, attn.block_size, attn.top_k_blocks)
        biases.append(bias)
        h = block(h, model.rope, causal_mask, dense_mode=trajectory_dense_mode)
    return biases


def forward_open_loop_moba(model: MoBATransformer, x: torch.Tensor, biases: list[torch.Tensor]) -> torch.Tensor:
    logits, _ = model(x, forced_token_biases=biases)
    return logits


def forward_closed_loop_moba(model: MoBATransformer, x: torch.Tensor) -> torch.Tensor:
    logits, _ = model(x, dense_mode=False)
    return logits


def _native_dense_mode(condition: str) -> bool:
    return condition == "dense_switch"


# ─────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────

def _mean_nll_over_loader(model, loader, device, forward_fn, max_batches=None):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            batch = batch.to(device)
            x, y = batch[:, :-1], batch[:, 1:]
            logits = forward_fn(x)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum"
            )
            total_loss += loss.item()
            total_tokens += y.numel()
    return total_loss / total_tokens


def evaluate_nll(model, loader, device, condition, max_batches=None):
    dense = _native_dense_mode(condition)
    return _mean_nll_over_loader(
        model, loader, device, lambda x: model(x, dense_mode=dense)[0], max_batches
    )


def eval_open_loop(model, loader, device, condition, max_batches=None):
    dense = _native_dense_mode(condition)

    def forward_fn(x):
        biases = record_moba_biases_from_trajectory(model, x, trajectory_dense_mode=dense)
        return forward_open_loop_moba(model, x, biases)

    return _mean_nll_over_loader(model, loader, device, forward_fn, max_batches)


def eval_closed_loop(model, loader, device, max_batches=None):
    return _mean_nll_over_loader(
        model, loader, device, lambda x: forward_closed_loop_moba(model, x), max_batches
    )


def eval_random_hard(model, loader, device, max_batches, generator):
    def forward_fn(x):
        batch, seq_len = x.shape
        biases = []
        for block in model.blocks:
            attn = block.attn
            num_blocks = math.ceil(seq_len / attn.block_size)
            rand_scores = torch.rand(
                batch, attn.n_heads, seq_len, num_blocks, generator=generator, device="cpu"
            ).to(x.device)
            bias, _ = _blockwise_topk_bias(rand_scores, attn.block_size, attn.top_k_blocks, seq_len)
            biases.append(bias)
        logits, _ = model(x, forced_token_biases=biases)
        return logits

    return _mean_nll_over_loader(model, loader, device, forward_fn, max_batches)


def run_full_evaluation(model, loader, device, condition, max_batches=None, n_random_draws=3, verbose=False):
    if verbose:
        print("native", flush=True)
    native_nll = evaluate_nll(model, loader, device, condition, max_batches)

    if verbose:
        print("open_loop", flush=True)
    open_loop_nll = eval_open_loop(model, loader, device, condition, max_batches)

    if verbose:
        print("closed_loop", flush=True)
    closed_loop_nll = eval_closed_loop(model, loader, device, max_batches)

    dense_mode_nll = _mean_nll_over_loader(
        model, loader, device, lambda x: model(x, dense_mode=True)[0], max_batches
    )

    if verbose:
        print("random_hard", flush=True)
    rand_vals = [
        eval_random_hard(model, loader, device, max_batches or 1_000_000, torch.Generator().manual_seed(1000 + i))
        for i in range(n_random_draws)
    ]
    random_hard_nll_mean = float(np.mean(rand_vals))
    random_hard_nll_std = float(np.std(rand_vals))

    native_ppl = math.exp(native_nll)
    G_CL = closed_loop_nll - native_nll
    G_OL = open_loop_nll - native_nll
    G_CL_vs_dense = closed_loop_nll - dense_mode_nll
    gate_utility = (random_hard_nll_mean - closed_loop_nll) / max(random_hard_nll_mean, 1e-9)

    return {
        "native_nll": native_nll,
        "native_ppl": native_ppl,
        "open_loop_nll": open_loop_nll,
        "closed_loop_nll": closed_loop_nll,
        "dense_mode_nll": dense_mode_nll,
        "G_CL": G_CL,
        "G_OL": G_OL,
        "G_CL_vs_dense": G_CL_vs_dense,
        "random_hard_nll_mean": random_hard_nll_mean,
        "random_hard_nll_std": random_hard_nll_std,
        "gate_utility": gate_utility,
    }


# ─────────────────────────────────────────────────────────────────────────
# Training-time loss per condition
# ─────────────────────────────────────────────────────────────────────────

def compute_condition_loss(model, x, y, condition, lambda_rca, normalize_loss=False):
    if condition == "standard":
        if lambda_rca != 0.0:
            raise ValueError(f"standard condition requires lambda_rca=0.0, got {lambda_rca}")
        logits, _ = model(x, dense_mode=False)
        native = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        return native, native.item()

    if condition == "dense_switch":
        if lambda_rca != 0.0:
            raise ValueError(f"dense_switch condition requires lambda_rca=0.0, got {lambda_rca}")
        logits, _ = model(x, dense_mode=True)
        native = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        return native, native.item()

    if condition == "clhr":
        logits_dense, _ = model(x, dense_mode=True)
        lm_dense = F.cross_entropy(logits_dense.reshape(-1, logits_dense.size(-1)), y.reshape(-1))
        native = lm_dense.item()
        if lambda_rca == 0.0:
            return lm_dense, native
        logits_hard, _ = model(x, dense_mode=False)
        lm_hard = F.cross_entropy(logits_hard.reshape(-1, logits_hard.size(-1)), y.reshape(-1))
        loss = lm_dense + lambda_rca * lm_hard
        if normalize_loss:
            loss = loss / (1.0 + lambda_rca)
        return loss, native

    raise ValueError(f"unknown condition: {condition}")


# ─────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────

def train_model(
    model,
    loader,
    device,
    condition,
    steps,
    lr,
    lambda_rca,
    log_every=100,
    grad_accum_steps=1,
    normalize_loss=False,
):
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1, betas=(0.9, 0.95))
    train_iter = iter(loader)
    t0 = time.time()
    step = 0
    accum_count = 0
    optimizer.zero_grad()

    while step < steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(loader)
            batch = next(train_iter)
        batch = batch.to(device)
        x, y = batch[:, :-1], batch[:, 1:]

        loss, native = compute_condition_loss(model, x, y, condition, lambda_rca, normalize_loss=normalize_loss)
        (loss / grad_accum_steps).backward()
        accum_count += 1
        if accum_count == grad_accum_steps:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            accum_count = 0

        step += 1
        if log_every and step % log_every == 0:
            print(f"[train] step {step}/{steps} loss={loss.item():.4f} native_nll={native:.4f}", flush=True)

    elapsed = time.time() - t0
    return step, elapsed


# ─────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────

class CyclingTokenDataset(Dataset):
    def __init__(self, tokens: torch.Tensor, seq_len: int):
        self.tokens = tokens
        self.seq_len = seq_len
        self._n = max(1, len(tokens) - seq_len)

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> torch.Tensor:
        i = idx % self._n
        return self.tokens[i : i + self.seq_len + 1]


def load_moba_data(data_dir: str, seq_len: int) -> tuple[CyclingTokenDataset, CyclingTokenDataset]:
    data_path = Path(data_dir)
    train_tokens = torch.from_numpy(np.load(data_path / "wt103_train_tokens.npy")).long()
    val_tokens = torch.from_numpy(np.load(data_path / "wt103_val_tokens.npy")).long()
    return CyclingTokenDataset(train_tokens, seq_len), CyclingTokenDataset(val_tokens, seq_len)


# ─────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────

def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MoBA baseline for the CLHR reconciliation ladder")
    parser.add_argument("--condition", choices=CONDITIONS, default="standard")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--checkpoint-dir", type=str, default="./ckpts_moba")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--lambda-rca", type=float, default=None)
    parser.add_argument("--normalize-loss", action="store_true")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--micro-batch", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--d-ff", type=int, default=1024)
    parser.add_argument("--vocab-size", type=int, default=50257)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--top-k-blocks", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--max-eval-batches", type=int, default=20)
    return parser


def main():
    args = build_argparser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    train_ds, val_ds = load_moba_data(args.data_dir, args.seq_len)
    train_loader = DataLoader(train_ds, batch_size=args.micro_batch, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.micro_batch, shuffle=False, num_workers=0, drop_last=True)

    lambda_rca = args.lambda_rca if args.lambda_rca is not None else (1.0 if args.condition == "clhr" else 0.0)
    if args.condition != "clhr" and lambda_rca != 0.0:
        raise ValueError(f"condition={args.condition} requires lambda_rca=0.0, got {lambda_rca}")

    model = MoBATransformer(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        block_size=args.block_size,
        top_k_blocks=args.top_k_blocks,
        max_seq_len=args.seq_len,
        dropout=args.dropout,
    ).to(device)

    print(f"Device: {device}, condition: {args.condition}, seed: {args.seed}", flush=True)
    print(f"  Parameters: {model.count_parameters() / 1e6:.1f}M", flush=True)

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    steps_completed, elapsed = train_model(
        model,
        train_loader,
        device,
        args.condition,
        steps=args.steps,
        lr=args.lr,
        lambda_rca=lambda_rca,
        log_every=args.log_every,
        grad_accum_steps=args.grad_accum,
        normalize_loss=args.normalize_loss,
    )
    print(f"Training done: {steps_completed} steps in {elapsed:.1f}s", flush=True)

    result = run_full_evaluation(
        model, val_loader, device, args.condition,
        max_batches=args.max_eval_batches, n_random_draws=3, verbose=True,
    )
    result.update(
        {
            "condition": args.condition,
            "seed": args.seed,
            "total_steps": steps_completed,
            "model_params": model.count_parameters(),
            "lambda_rca": lambda_rca,
            "elapsed_sec": elapsed,
            "block_size": args.block_size,
            "top_k_blocks": args.top_k_blocks,
        }
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
