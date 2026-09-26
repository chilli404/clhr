"""Post-hoc KL gate distillation comparison.

Compares CLHR's original-gate deployment against the routing-absorption
paper's strongest remedy: freeze the representation, reinitialize fresh
gate projections, train them with KL divergence against softmax(QK^T/temp),
then deploy with top-k of the fresh gate.

Matches upstream protocol: 1000 steps, lr=1e-3, no weight decay, AdamW,
grad clip 1.0. KL(oracle || gate) where oracle = softmax(QK^T / temp).

Usage:
    python src/sparse_attention_posthoc_kl.py \
        --checkpoint-dir /s3-data/ckpts_sparse_rca \
        --data-dir ./wikitext103 \
        --output /s3-data/results/posthoc_kl_comparison.json
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
from torch.utils.data import DataLoader

RA_ROOT = Path(__file__).parent.parent.parent / "routing-absorption"
sys.path.insert(0, str(RA_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from src.models.config import ModelConfig
from src.models.transformer import GatedTransformer
from sparse_attention_rca import DEFAULT_CONFIG, load_wikitext

CONDITIONS = [
    "learned_gate",
    "contemporary_closedloop_hard",
    "dense",
]

DISTILL_STEPS = 1000
GATE_LR = 1e-3
K = 64
TEMPERATURE = 1.0


def eval_native_soft(model, val_loader, device, max_batches=200):
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
    """One-pass closed-loop hard deployment using model's own gate."""
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


def distill_and_deploy(model, train_loader, val_loader, device,
                       distill_steps=1000, gate_lr=1e-3, k=64, temperature=1.0):
    """Freeze model, reinitialize gates, train with KL, deploy with top-k."""
    n_heads = model.config.n_heads
    d_head = model.config.d_model // n_heads
    d_gate = model.config.d_gate
    n_layers = model.config.n_layers

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    gate_params = []
    for layer in model.layers:
        attn = layer.attention
        nn.init.normal_(attn.W_gq.weight, std=0.02)
        nn.init.normal_(attn.W_gk.weight, std=0.02)
        attn.W_gq.weight.requires_grad = True
        attn.W_gk.weight.requires_grad = True
        gate_params.extend([attn.W_gq.weight, attn.W_gk.weight])

    n_gate_params = sum(p.numel() for p in gate_params)
    print(f"    Gate params: {n_gate_params:,}")

    optimizer = torch.optim.AdamW(gate_params, lr=gate_lr, weight_decay=0.0)

    train_iter = iter(train_loader)

    for step in range(1, distill_steps + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        ids = batch["input_ids"].to(device)
        x_in = ids[:, :-1]
        B, T = x_in.shape

        positions = torch.arange(T, device=device).unsqueeze(0)
        causal = torch.tril(torch.ones(T, T, device=device)).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            x = model.embedding(x_in) + model.pos_embedding(positions)
            x = model.dropout(x)

        total_kl = 0.0
        for layer in model.layers:
            attn = layer.attention
            h = layer.attn_norm(x) if not x.requires_grad else layer.attn_norm(x.detach())

            with torch.no_grad():
                q = attn.W_q(h).view(B, T, n_heads, d_head).transpose(1, 2)
                kk = attn.W_k(h).view(B, T, n_heads, d_head).transpose(1, 2)
                scores = torch.matmul(q, kk.transpose(-2, -1)) / (d_head ** 0.5)
                scores = scores.masked_fill(causal == 0, float("-inf"))
                oracle_probs = F.softmax(scores / temperature, dim=-1)

            gq = attn.W_gq(h).view(B, T, n_heads, d_gate).transpose(1, 2)
            gk = attn.W_gk(h).view(B, T, n_heads, d_gate).transpose(1, 2)
            gate_logits = torch.matmul(gq, gk.transpose(-2, -1)) / (d_gate ** 0.5)
            gate_logits = gate_logits.masked_fill(causal == 0, float("-inf"))
            gate_log_probs = F.log_softmax(gate_logits / temperature, dim=-1)

            kl_pointwise = oracle_probs * (oracle_probs.clamp(min=1e-10).log() - gate_log_probs)
            kl_pointwise = torch.where(oracle_probs > 1e-10, kl_pointwise, torch.zeros_like(kl_pointwise))
            kl = kl_pointwise.sum(-1).mean()
            total_kl = total_kl + kl

            with torch.no_grad():
                x = x + attn(layer.attn_norm(x), causal)
                x = x + layer.ff(layer.ff_norm(x))

        avg_kl = total_kl / n_layers
        optimizer.zero_grad()
        avg_kl.backward()
        torch.nn.utils.clip_grad_norm_(gate_params, 1.0)
        optimizer.step()

        if step % 250 == 0:
            print(f"    KL distill step {step}/{distill_steps}: loss={avg_kl.item():.4f}")

    for p in gate_params:
        p.requires_grad = False

    posthoc_nll = eval_closed_loop_hard(model, val_loader, device, k=k)

    for p in model.parameters():
        p.requires_grad = True

    return posthoc_nll


def run_comparison(checkpoint_dir, data_dir, output_path, tag_prefix="sparse_rca"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ds, val_ds = load_wikitext(data_dir)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0, drop_last=True)

    results = {}
    seeds = [42, 123, 456]

    for cond in CONDITIONS:
        for seed in seeds:
            tag = f"{tag_prefix}_{cond}_s{seed}"
            ckpt_path = Path(checkpoint_dir) / tag / "step_50000.pt"
            if not ckpt_path.exists():
                continue

            key = f"{cond}_s{seed}"
            print(f"\n{'='*60}")
            print(f"CONDITION: {cond}  SEED: {seed}")
            print(f"{'='*60}")

            is_dense = cond == "dense"

            if is_dense:
                dense_config = DEFAULT_CONFIG.model_copy()
                dense_config.sparsity_mode = "dense"
                dense_model = GatedTransformer(dense_config).to(device)
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
                dense_model.load_state_dict(ckpt["model"])

                native_nll = eval_native_soft(dense_model, val_loader, device)
                print(f"  Native dense NLL: {native_nll:.4f} (PPL {math.exp(native_nll):.2f})")
                original_hard_nll = float('nan')
                original_excess = float('nan')
                print(f"  Original gate: N/A (dense model has no gate)")

                # For distillation: load into a soft-config model (which has W_gq/W_gk)
                # and load only the non-gate weights from the dense checkpoint
                config = DEFAULT_CONFIG.model_copy()
                model2 = GatedTransformer(config).to(device)
                dense_state = ckpt["model"]
                soft_state = model2.state_dict()
                for k_name in dense_state:
                    if k_name in soft_state:
                        soft_state[k_name] = dense_state[k_name]
                model2.load_state_dict(soft_state)
            else:
                config = DEFAULT_CONFIG.model_copy()
                model = GatedTransformer(config).to(device)
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
                model.load_state_dict(ckpt["model"])

                native_nll = eval_native_soft(model, val_loader, device)
                print(f"  Native soft NLL: {native_nll:.4f} (PPL {math.exp(native_nll):.2f})")

                original_hard_nll = eval_closed_loop_hard(model, val_loader, device, k=K)
                original_excess = original_hard_nll - native_nll
                print(f"  Original gate hard NLL: {original_hard_nll:.4f} (excess {original_excess:.4f})")

                model2 = GatedTransformer(config).to(device)
                model2.load_state_dict(ckpt["model"])
            posthoc_hard_nll = distill_and_deploy(
                model2, train_loader, val_loader, device,
                distill_steps=DISTILL_STEPS, gate_lr=GATE_LR, k=K, temperature=TEMPERATURE,
            )
            posthoc_excess = posthoc_hard_nll - native_nll
            print(f"  Post-hoc gate hard NLL: {posthoc_hard_nll:.4f} (excess {posthoc_excess:.4f})")

            results[key] = {
                "native_nll": round(native_nll, 6),
                "original_gate_hard_nll": round(original_hard_nll, 6),
                "posthoc_gate_hard_nll": round(posthoc_hard_nll, 6),
                "original_gate_excess": round(original_excess, 6),
                "posthoc_gate_excess": round(posthoc_excess, 6),
            }

    # Summary table
    print(f"\n{'='*80}")
    print("POST-HOC KL COMPARISON")
    print(f"{'='*80}")
    print(f"{'Condition':<40} {'Seed':>5} {'Native':>8} {'Orig_ex':>8} {'PostKL_ex':>9}")
    print('-' * 75)
    for key, val in sorted(results.items()):
        parts = key.rsplit('_s', 1)
        cond, seed = parts[0], parts[1]
        print(f"{cond:<40} {seed:>5} {val['native_nll']:>8.4f} "
              f"{val['original_gate_excess']:>8.4f} {val['posthoc_gate_excess']:>9.4f}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Post-hoc KL gate distillation comparison")
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--tag-prefix", type=str, default="sparse_rca")
    args = parser.parse_args()
    run_comparison(args.checkpoint_dir, args.data_dir, args.output, tag_prefix=args.tag_prefix)


if __name__ == "__main__":
    main()
