"""Fair post-hoc KL comparison: distill gate onto mask-agnostic (dense) representation.

Reproduces the upstream routing-absorption paper's strongest remedy:
  1. Train a dense model (no gate masking) — the mask-agnostic representation
  2. Freeze all parameters
  3. Add fresh W_gq / W_gk gate projections
  4. Train only the gate with KL against softmax(QK^T / temp) for 1000 steps
  5. Deploy with the fresh gate's top-k in closed-loop mode

Compares against CLHR's original gate deployment on the same evaluation data.

Usage:
    python src/sparse_attention_posthoc_kl_maskagnostic.py \
        --dense-checkpoint-dir /s3-data/ckpts_sparse_rca \
        --clhr-checkpoint-dir /s3-data/ckpts_sparse_rca \
        --data-dir ./wikitext103 \
        --output /s3-data/results/posthoc_kl_maskagnostic.json
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

RA_ROOT = Path(__file__).parent.parent.parent / "routing-absorption"
sys.path.insert(0, str(RA_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from src.models.config import ModelConfig
from src.models.transformer import GatedTransformer
from sparse_attention_rca import DEFAULT_CONFIG, load_wikitext

DISTILL_STEPS = 1000
GATE_LR = 1e-3
GATE_TEMP = 1.0
K = 64
SEEDS = [42, 123, 456]


def eval_native(model, val_loader, device, max_batches=200):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break
            ids = batch["input_ids"].to(device)
            x, y = ids[:, :-1], ids[:, 1:]
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum")
            total_loss += loss.item()
            total_tokens += y.numel()
    return total_loss / total_tokens


def eval_closed_loop_hard(model, val_loader, device, k=64, max_batches=200):
    model.eval()
    n_heads = model.config.n_heads
    d_head = model.config.d_model // n_heads
    d_gate = model.config.d_gate
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break
            ids = batch["input_ids"].to(device)
            x_in, y = ids[:, :-1], ids[:, 1:]
            B, T = x_in.shape

            positions = torch.arange(T, device=device).unsqueeze(0)
            x = model.embedding(x_in) + model.pos_embedding(positions)
            x = model.dropout(x)
            causal = torch.tril(torch.ones(T, T, device=device)).unsqueeze(0).unsqueeze(0)

            for layer in model.layers:
                attn = layer.attention
                h = layer.attn_norm(x)
                q = attn.W_q(h).view(B, T, n_heads, d_head).transpose(1, 2)
                kk = attn.W_k(h).view(B, T, n_heads, d_head).transpose(1, 2)
                v = attn.W_v(h).view(B, T, n_heads, d_head).transpose(1, 2)
                attn_scores = torch.matmul(q, kk.transpose(-2, -1)) / (d_head ** 0.5)

                gq = attn.W_gq(h).view(B, T, n_heads, d_gate).transpose(1, 2)
                gk = attn.W_gk(h).view(B, T, n_heads, d_gate).transpose(1, 2)
                gs = torch.matmul(gq, gk.transpose(-2, -1)) / (d_gate ** 0.5)
                gs = gs.masked_fill(causal == 0, float("-inf"))
                _, idx = torch.topk(gs, min(k, T), dim=-1)
                hard_mask = torch.zeros_like(gs).scatter_(-1, idx, 1.0) * causal

                attn_scores = attn_scores.masked_fill(hard_mask == 0, float("-inf"))
                attn_scores = attn_scores.masked_fill(causal == 0, float("-inf"))
                w = torch.nan_to_num(F.softmax(attn_scores, dim=-1), nan=0.0)
                output = torch.matmul(w, v).transpose(1, 2).contiguous().view(B, T, -1)
                x = x + attn.W_o(output)
                x = x + layer.ff(layer.ff_norm(x))

            logits = model.lm_head(model.final_norm(x))
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum")
            total_loss += loss.item()
            total_tokens += y.numel()

    return total_loss / total_tokens


def distill_gate_kl(model, train_loader, device, steps=1000, lr=1e-3, temp=1.0, k=64):
    """Freeze all params, reinit gate, train gate with KL against oracle."""
    n_heads = model.config.n_heads
    d_head = model.config.d_model // n_heads
    d_gate = model.config.d_gate
    n_layers = len(model.layers)

    for p in model.parameters():
        p.requires_grad = False

    for layer in model.layers:
        attn = layer.attention
        nn.init.normal_(attn.W_gq.weight, std=0.02)
        nn.init.normal_(attn.W_gk.weight, std=0.02)
        attn.W_gq.weight.requires_grad = True
        attn.W_gk.weight.requires_grad = True

    gate_params = []
    for layer in model.layers:
        gate_params.extend([layer.attention.W_gq.weight, layer.attention.W_gk.weight])
    optimizer = torch.optim.AdamW(gate_params, lr=lr, weight_decay=0.0)

    model.train()
    train_iter = iter(train_loader)
    for step in range(1, steps + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        ids = batch["input_ids"].to(device)
        x_in = ids[:, :-1]
        B, T = x_in.shape

        optimizer.zero_grad()

        positions = torch.arange(T, device=device).unsqueeze(0)
        causal = torch.tril(torch.ones(T, T, device=device)).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            x = model.embedding(x_in) + model.pos_embedding(positions)
            x = model.dropout(x)

        total_kl = torch.tensor(0.0, device=device)
        for layer in model.layers:
            attn = layer.attention
            with torch.no_grad():
                h = layer.attn_norm(x)
                q = attn.W_q(h).view(B, T, n_heads, d_head).transpose(1, 2)
                kk = attn.W_k(h).view(B, T, n_heads, d_head).transpose(1, 2)
                qk_scores = torch.matmul(q, kk.transpose(-2, -1)) / (d_head ** 0.5)
                qk_scores = qk_scores.masked_fill(causal == 0, float("-inf"))
                oracle = F.softmax(qk_scores / temp, dim=-1)

            # Oracle top-k binary target
            with torch.no_grad():
                _, oracle_idx = torch.topk(qk_scores, min(k, T), dim=-1)
                oracle_mask = torch.zeros_like(qk_scores)
                oracle_mask.scatter_(-1, oracle_idx, 1.0)
                oracle_mask = oracle_mask * causal

            gq = attn.W_gq(h.detach()).view(B, T, n_heads, d_gate).transpose(1, 2)
            gk = attn.W_gk(h.detach()).view(B, T, n_heads, d_gate).transpose(1, 2)
            gate_scores = torch.matmul(gq, gk.transpose(-2, -1)) / (d_gate ** 0.5)

            # BCE against oracle top-k mask (numerically stable, avoids NaN from KL on masked positions)
            gate_probs = torch.sigmoid(gate_scores)
            bce = F.binary_cross_entropy(gate_probs * causal, oracle_mask, reduction="mean")
            total_kl = total_kl + bce

            with torch.no_grad():
                v = attn.W_v(h).view(B, T, n_heads, d_head).transpose(1, 2)
                attn_scores = qk_scores.masked_fill(causal == 0, float("-inf"))
                w = torch.nan_to_num(F.softmax(attn_scores, dim=-1), nan=0.0)
                output = torch.matmul(w, v).transpose(1, 2).contiguous().view(B, T, -1)
                x = x + attn.W_o(output)
                x = x + layer.ff(layer.ff_norm(x))

        avg_kl = total_kl / n_layers
        avg_kl.backward()
        torch.nn.utils.clip_grad_norm_(gate_params, 1.0)
        optimizer.step()

        if step % 250 == 0:
            print(f"    KL distill step {step}/{steps}: loss={avg_kl.item():.4f}")

    for p in model.parameters():
        p.requires_grad = True


def run_comparison(dense_ckpt_dir, clhr_ckpt_dir, data_dir, output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    _, val_ds = load_wikitext(data_dir)
    train_ds, _ = load_wikitext(data_dir)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0, drop_last=True)

    results = {}

    for seed in SEEDS:
        # --- Dense + post-hoc KL ---
        dense_tag = f"sparse_rca_dense_s{seed}"
        dense_ckpt = Path(dense_ckpt_dir) / dense_tag / "step_50000.pt"
        if dense_ckpt.exists():
            print(f"\n{'='*60}")
            print(f"DENSE + POST-HOC KL  seed={seed}")
            print(f"{'='*60}")

            config = DEFAULT_CONFIG.model_copy()
            config.sparsity_mode = "dense"
            model = GatedTransformer(config).to(device)
            ckpt = torch.load(dense_ckpt, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"])

            native_nll = eval_native(model, val_loader, device)
            print(f"  Dense native NLL: {native_nll:.4f} (PPL {math.exp(native_nll):.2f})")

            # Switch to soft mode and add gate projections for distillation
            for layer in model.layers:
                attn = layer.attention
                attn.config.sparsity_mode = "soft"
                if not hasattr(attn, 'W_gq') or attn.W_gq is None:
                    d_model = config.d_model
                    d_gate = config.d_gate
                    n_heads = config.n_heads
                    attn.W_gq = nn.Linear(d_model, n_heads * d_gate, bias=False).to(device)
                    attn.W_gk = nn.Linear(d_model, n_heads * d_gate, bias=False).to(device)

            distill_gate_kl(model, train_loader, device, steps=DISTILL_STEPS, lr=GATE_LR, temp=GATE_TEMP, k=K)

            posthoc_nll = eval_closed_loop_hard(model, val_loader, device, k=K)
            posthoc_excess = posthoc_nll - native_nll
            print(f"  Post-hoc KL deploy NLL: {posthoc_nll:.4f} (PPL {math.exp(posthoc_nll):.2f})")
            print(f"  Post-hoc excess: {posthoc_excess:.4f}")

            results[f"dense_posthoc_s{seed}"] = {
                "native_nll": round(native_nll, 6),
                "posthoc_deploy_nll": round(posthoc_nll, 6),
                "posthoc_excess": round(posthoc_excess, 6),
            }

        # --- CLHR original gate ---
        clhr_tag = f"sparse_rca_contemporary_closedloop_hard_s{seed}"
        clhr_ckpt = Path(clhr_ckpt_dir) / clhr_tag / "step_50000.pt"
        if clhr_ckpt.exists():
            print(f"\n  CLHR original gate  seed={seed}")

            config = DEFAULT_CONFIG.model_copy()
            model = GatedTransformer(config).to(device)
            ckpt = torch.load(clhr_ckpt, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"])

            native_nll = eval_native(model, val_loader, device)
            clhr_nll = eval_closed_loop_hard(model, val_loader, device, k=K)
            clhr_excess = clhr_nll - native_nll
            print(f"  CLHR native NLL: {native_nll:.4f}")
            print(f"  CLHR deploy NLL: {clhr_nll:.4f}")
            print(f"  CLHR excess: {clhr_excess:.4f}")

            results[f"clhr_original_s{seed}"] = {
                "native_nll": round(native_nll, 6),
                "deploy_nll": round(clhr_nll, 6),
                "deploy_excess": round(clhr_excess, 6),
            }

    # Summary
    print(f"\n{'='*80}")
    print("COMPARISON: Dense+PostHoc KL vs CLHR Original Gate")
    print(f"{'='*80}")
    print(f"{'Method':<30} {'Seed':>5} {'Native':>8} {'Deploy':>8} {'Excess':>8}")
    print('-' * 65)
    for key in sorted(results.keys()):
        val = results[key]
        native = val.get('native_nll', val.get('native_nll', 0))
        deploy = val.get('posthoc_deploy_nll', val.get('deploy_nll', 0))
        excess = val.get('posthoc_excess', val.get('deploy_excess', 0))
        seed_str = key.split('_s')[-1]
        method = 'Dense+PostHoc KL' if 'posthoc' in key else 'CLHR original gate'
        print(f"{method:<30} {seed_str:>5} {native:>8.4f} {deploy:>8.4f} {excess:>8.4f}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Mask-agnostic post-hoc KL comparison")
    parser.add_argument("--dense-checkpoint-dir", type=str, required=True)
    parser.add_argument("--clhr-checkpoint-dir", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    run_comparison(args.dense_checkpoint_dir, args.clhr_checkpoint_dir, args.data_dir, args.output)


if __name__ == "__main__":
    main()
