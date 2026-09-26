"""Qwen3-1.7B multi-layer sparse fine-tuning with RCA.

Sparsify a pretrained dense LLM: add soft bilinear gates to ALL 28 layers,
unfreeze ALL Q/K/V/O projections (~411M trainable params), keep FFN/embeddings/
norms frozen. Fine-tune with sparse attention for 5000 steps.

Unlike the single-layer version (qwen3_sparse_rca.py), this creates the
conditions for multi-layer routing absorption: Q/K/V across ALL layers
can co-adapt with their respective gates.

Conditions:
  learned_gate         — standard sparse fine-tuning (baseline)
  contemporary_replay  — dual-loss with current gate masks
  shuffled_historical  — historical gate masks with token permutation
  coherent_oldest_first — historical gate masks from oldest snapshot

Usage:
    python src/qwen3_multilayer_sparse_rca.py \
        --condition coherent_oldest_first --seed 42 \
        --output /s3-data/results/qwen3_multi_rca_coherent_oldest_first_s42.json \
        --model-name /s3-data/models/qwen3-1.7b-base
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
from torch.utils.data import DataLoader, TensorDataset

from qwen3_sparse_rca import (
    rotate_half, apply_rotary_pos_emb, repeat_kv,
    SingleLayerGate, select_oldest_snapshot, load_wikitext_qwen3,
)

CONDITIONS = [
    "learned_gate",
    "contemporary_replay",
    "shuffled_historical",
    "coherent_oldest_first",
]

D_GATE = 32
K_SPARSE = 64
TRAIN_STEPS = 5000
SNAPSHOT_INTERVAL = 500
N_SNAPSHOTS = 5
LAMBDA_RCA = 0.3
TARGET_SPARSITY = 0.875
LAMBDA_SPARSE = 0.1


class MultiLayerGates(nn.Module):
    def __init__(self, n_layers: int, n_heads: int, d_model: int, d_gate: int):
        super().__init__()
        self.gates = nn.ModuleList([
            SingleLayerGate(n_heads, d_model, d_gate)
            for _ in range(n_layers)
        ])

    def gate_scores(self, layer_idx: int, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.gates[layer_idx].gate_scores(hidden_states)


def shuffle_all_layer_masks(masks: list[torch.Tensor]) -> list[torch.Tensor]:
    result = []
    for m in masks:
        shuffled = m.clone()
        B, H, Tq, Tk = shuffled.shape
        for b in range(B):
            perm = torch.randperm(Tk, device=shuffled.device)
            shuffled[b] = shuffled[b, :, :, perm]
        result.append(shuffled)
    return result


def forward_gated_all_layers(
    model, input_ids, gates: MultiLayerGates,
    position_ids, causal_mask,
    forced_masks: list[torch.Tensor] | None = None,
):
    """Full forward pass with soft gating at EVERY layer."""
    cfg = model.config
    n_heads = cfg.num_attention_heads
    n_kv_heads = cfg.num_key_value_heads
    n_groups = n_heads // n_kv_heads
    head_dim = cfg.hidden_size // n_heads
    bsz, sl = input_ids.shape
    n_layers = cfg.num_hidden_layers

    hidden_states = model.model.embed_tokens(input_ids)
    position_embeddings = model.model.rotary_emb(hidden_states, position_ids)

    all_soft_masks = []

    for layer_idx in range(n_layers):
        layer = model.model.layers[layer_idx]
        residual = hidden_states
        normed = layer.input_layernorm(hidden_states)
        attn = layer.self_attn

        q = attn.q_proj(normed).view(bsz, sl, n_heads, head_dim)
        kk = attn.k_proj(normed).view(bsz, sl, n_kv_heads, head_dim)
        v = attn.v_proj(normed).view(bsz, sl, n_kv_heads, head_dim).transpose(1, 2)
        if hasattr(attn, "q_norm"):
            q = attn.q_norm(q)
            kk = attn.k_norm(kk)
        q, kk = q.transpose(1, 2), kk.transpose(1, 2)
        cos, sin = position_embeddings
        q, kk = apply_rotary_pos_emb(q, kk, cos, sin)
        kk, v = repeat_kv(kk, n_groups), repeat_kv(v, n_groups)

        aw = torch.matmul(q.float(), kk.float().transpose(2, 3)) / math.sqrt(head_dim)

        if forced_masks is not None:
            soft_mask = forced_masks[layer_idx]
        else:
            g_scores = gates.gate_scores(layer_idx, normed.float())
            g_scores = g_scores.masked_fill(causal_mask[:, :, :sl, :sl] < -1e4, -10.0)
            soft_mask = torch.sigmoid(g_scores)

        all_soft_masks.append(soft_mask.detach())

        aw = aw * soft_mask
        aw = aw + causal_mask[:, :, :sl, :sl].float()
        aw = F.softmax(aw, dim=-1)
        ao = torch.matmul(aw, v.float()).to(hidden_states.dtype)
        ao = ao.transpose(1, 2).contiguous().reshape(bsz, sl, -1)
        hidden_states = residual + attn.o_proj(ao)

        residual = hidden_states
        hidden_states = residual + layer.mlp(layer.post_attention_layernorm(hidden_states))

    hidden_states = model.model.norm(hidden_states)
    logits = model.lm_head(hidden_states)
    return logits, all_soft_masks


def extract_all_gate_masks(
    model, gates: MultiLayerGates, input_ids,
    position_ids, causal_mask,
) -> list[torch.Tensor]:
    """Extract soft masks from all layers without grad."""
    with torch.no_grad():
        _, masks = forward_gated_all_layers(
            model, input_ids, gates, position_ids, causal_mask,
        )
    return masks


def eval_nll_multilayer(
    model, loader, gates, position_ids_fn, causal_mask_fn,
    device, max_batches=30, forced_masks_fn=None,
):
    model.eval()
    gates.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for i, (input_ids,) in enumerate(loader):
            if i >= max_batches:
                break
            input_ids = input_ids.to(device)
            labels = input_ids[:, 1:].contiguous()
            inputs = input_ids[:, :-1].contiguous()
            bsz, sl = inputs.shape
            pos_ids = position_ids_fn(bsz, sl, device)
            cm = causal_mask_fn(sl, device)

            forced = None
            if forced_masks_fn is not None:
                forced = forced_masks_fn(model, gates, inputs, pos_ids, cm)

            logits, _ = forward_gated_all_layers(
                model, inputs, gates, pos_ids, cm, forced_masks=forced,
            )
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1), reduction="sum",
            )
            total_loss += loss.item()
            total_tokens += labels.numel()
    return total_loss / total_tokens


def make_oracle_masks_fn(k=64):
    def fn(model, gates, input_ids, position_ids, causal_mask):
        cfg = model.config
        n_heads = cfg.num_attention_heads
        n_kv_heads = cfg.num_key_value_heads
        n_groups = n_heads // n_kv_heads
        head_dim = cfg.hidden_size // n_heads
        bsz, sl = input_ids.shape

        hidden = model.model.embed_tokens(input_ids)
        pos_emb = model.model.rotary_emb(hidden, position_ids)
        causal_01 = torch.tril(torch.ones(sl, sl, device=input_ids.device)).unsqueeze(0).unsqueeze(0)
        masks = []

        for layer_idx in range(cfg.num_hidden_layers):
            layer = model.model.layers[layer_idx]
            normed = layer.input_layernorm(hidden)
            attn = layer.self_attn
            q = attn.q_proj(normed).view(bsz, sl, n_heads, head_dim)
            kk = attn.k_proj(normed).view(bsz, sl, n_kv_heads, head_dim)
            v = attn.v_proj(normed).view(bsz, sl, n_kv_heads, head_dim).transpose(1, 2)
            if hasattr(attn, "q_norm"):
                q = attn.q_norm(q)
                kk = attn.k_norm(kk)
            q, kk = q.transpose(1, 2), kk.transpose(1, 2)
            cos, sin = pos_emb
            q, kk = apply_rotary_pos_emb(q, kk, cos, sin)
            kk = repeat_kv(kk, n_groups)
            v = repeat_kv(v, n_groups)

            scores = torch.matmul(q.float(), kk.float().transpose(-2, -1)) / math.sqrt(head_dim)
            scores = scores.masked_fill(causal_01 == 0, float("-inf"))
            actual_k = min(k, sl)
            _, topk_idx = torch.topk(scores, actual_k, dim=-1)
            mask = torch.zeros_like(scores)
            mask.scatter_(-1, topk_idx, 1.0)
            masks.append(mask * causal_01)

            # Advance hidden through this layer (with oracle mask gating)
            aw = scores.clone()
            aw = aw.masked_fill(mask == 0, float("-inf"))
            aw = aw + causal_mask[:, :, :sl, :sl].float()
            aw = F.softmax(aw, dim=-1)
            ao = torch.matmul(aw, v.float()).to(hidden.dtype)
            ao = ao.transpose(1, 2).contiguous().reshape(bsz, sl, -1)
            hidden = hidden + attn.o_proj(ao)
            hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))

        return masks
    return fn


def make_random_masks_fn(n_heads, n_layers, k=64):
    def fn(model, gates, input_ids, position_ids, causal_mask):
        bsz, sl = input_ids.shape
        device = input_ids.device
        causal_01 = torch.tril(torch.ones(sl, sl, device=device)).unsqueeze(0).unsqueeze(0)
        masks = []
        for _ in range(n_layers):
            rand_scores = torch.rand(bsz, n_heads, sl, sl, device=device)
            actual_k = min(k, sl)
            _, topk_idx = torch.topk(rand_scores, actual_k, dim=-1)
            mask = torch.zeros_like(rand_scores)
            mask.scatter_(-1, topk_idx, 1.0)
            masks.append(mask * causal_01)
        return masks
    return fn


def train_experiment(
    condition: str,
    seed: int,
    output_path: str,
    data_dir: str = "./wikitext103_cache",
    model_name: str = "/s3-data/models/qwen3-1.7b-base",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, condition: {condition}, seed: {seed}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    from transformers import AutoModelForCausalLM

    print("Loading Qwen3-1.7B-Base...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16,
        device_map="auto", attn_implementation="eager",
        local_files_only=True,
    )
    model.eval()

    cfg = model.config
    n_heads = cfg.num_attention_heads
    n_layers = cfg.num_hidden_layers
    d_model = cfg.hidden_size

    for p in model.parameters():
        p.requires_grad = False

    unfrozen_params = []
    for layer_idx in range(n_layers):
        attn = model.model.layers[layer_idx].self_attn
        for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            for p in getattr(attn, name).parameters():
                p.requires_grad = True
                unfrozen_params.append(p)
        for name in ["q_norm", "k_norm"]:
            if hasattr(attn, name):
                for p in getattr(attn, name).parameters():
                    p.requires_grad = True
                    unfrozen_params.append(p)

    unfrozen_count = sum(p.numel() for p in unfrozen_params)
    total_count = sum(p.numel() for p in model.parameters())
    print(f"  Unfrozen attention params: {unfrozen_count:,} ({unfrozen_count/total_count*100:.1f}%)")

    train_ds, val_ds = load_wikitext_qwen3(data_dir)
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=0,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=2, num_workers=0)

    gates = MultiLayerGates(n_layers, n_heads, d_model, D_GATE).to(device)
    gate_params = list(gates.parameters())
    gate_count = sum(p.numel() for p in gate_params)
    print(f"  Gate params: {gate_count:,}")

    all_trainable = gate_params + unfrozen_params
    optimizer = torch.optim.AdamW(all_trainable, lr=1e-4, weight_decay=0.01)

    use_rca = condition in ("contemporary_replay", "shuffled_historical", "coherent_oldest_first")
    snapshots = []

    hist_gates = None
    if use_rca:
        hist_gates = MultiLayerGates(n_layers, n_heads, d_model, D_GATE).to(device)
        hist_gates.eval()

    def pos_ids_fn(bsz, sl, dev):
        return torch.arange(sl, device=dev).unsqueeze(0).expand(bsz, -1)

    def cm_fn(sl, dev):
        return torch.triu(
            torch.full((sl, sl), torch.finfo(torch.bfloat16).min, device=dev, dtype=torch.bfloat16),
            diagonal=1,
        ).unsqueeze(0).unsqueeze(0)

    model.eval()
    gates.train()
    train_iter = iter(train_loader)
    t0 = time.time()

    for step in range(1, TRAIN_STEPS + 1):
        try:
            (input_ids,) = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            (input_ids,) = next(train_iter)

        input_ids = input_ids.to(device)
        inputs = input_ids[:, :-1].contiguous()
        labels = input_ids[:, 1:].contiguous()
        bsz, sl = inputs.shape
        pos_ids = pos_ids_fn(bsz, sl, device)
        cm = cm_fn(sl, device)

        logits, soft_masks = forward_gated_all_layers(
            model, inputs, gates, pos_ids, cm,
        )
        lm_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

        sparsity_vals = []
        for m in soft_masks:
            sparsity_vals.append(1.0 - m.mean().item())
        mean_sparsity = np.mean(sparsity_vals)
        sparsity_gap = max(0.0, TARGET_SPARSITY - mean_sparsity)
        sparsity_loss = LAMBDA_SPARSE * (sparsity_gap ** 2)

        rca_loss = torch.tensor(0.0, device=device)
        if use_rca and snapshots:
            if condition == "coherent_oldest_first":
                snap = select_oldest_snapshot(snapshots)
            elif condition == "contemporary_replay":
                snap = {"step": step, "gate_state": copy.deepcopy(gates.state_dict())}
            else:
                snap = snapshots[step % len(snapshots)]

            hist_gates.load_state_dict(snap["gate_state"])
            hist_gates.eval()
            with torch.no_grad():
                hist_masks = extract_all_gate_masks(
                    model, hist_gates, inputs, pos_ids, cm,
                )

            if condition == "shuffled_historical":
                hist_masks = shuffle_all_layer_masks(hist_masks)

            logits_rca, _ = forward_gated_all_layers(
                model, inputs, gates, pos_ids, cm, forced_masks=hist_masks,
            )
            rca_loss = F.cross_entropy(
                logits_rca.reshape(-1, logits_rca.size(-1)), labels.reshape(-1),
            )

        loss = lm_loss + sparsity_loss + (LAMBDA_RCA * rca_loss if use_rca and snapshots else 0.0)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_trainable, 1.0)
        optimizer.step()

        if step % SNAPSHOT_INTERVAL == 0 and use_rca:
            snapshots.append({
                "step": step,
                "gate_state": copy.deepcopy(gates.state_dict()),
            })
            if len(snapshots) > N_SNAPSHOTS:
                snapshots.pop(0)

        if step % 100 == 0:
            rca_str = f" rca={rca_loss.item():.4f}" if use_rca and snapshots else ""
            print(f"  step {step}/{TRAIN_STEPS}: loss={lm_loss.item():.4f}{rca_str} "
                  f"sparsity={mean_sparsity:.3f} ({time.time()-t0:.0f}s)")

    # Evaluation
    print("\n  Final evaluation...")
    gates.eval()

    native_nll = eval_nll_multilayer(
        model, val_loader, gates, pos_ids_fn, cm_fn, device,
    )
    print(f"  Native NLL: {native_nll:.4f} (PPL: {math.exp(native_nll):.2f})")

    oracle_nll = eval_nll_multilayer(
        model, val_loader, gates, pos_ids_fn, cm_fn, device,
        forced_masks_fn=make_oracle_masks_fn(K_SPARSE),
    )
    print(f"  Oracle NLL: {oracle_nll:.4f} (PPL: {math.exp(oracle_nll):.2f})")

    swap_nlls = []
    for draw in range(20):
        nll = eval_nll_multilayer(
            model, val_loader, gates, pos_ids_fn, cm_fn, device,
            forced_masks_fn=make_random_masks_fn(n_heads, n_layers, K_SPARSE),
        )
        swap_nlls.append(nll)
    swap_mean = float(np.mean(swap_nlls))
    swap_std = float(np.std(swap_nlls))
    print(f"  Swap NLL: {swap_mean:.4f} ± {swap_std:.4f} (PPL: {math.exp(swap_mean):.2f})")

    oracle_excess = oracle_nll - native_nll
    delta_swap = swap_mean - native_nll
    print(f"  Oracle excess: {oracle_excess:.4f} nats")
    print(f"  Swap excess: {delta_swap:.4f} nats")

    results = {
        "condition": condition,
        "seed": seed,
        "experiment": "qwen3_multilayer",
        "n_layers_gated": n_layers,
        "train_steps": TRAIN_STEPS,
        "unfrozen_attn_params": unfrozen_count,
        "gate_params": gate_count,
        "native_nll": round(native_nll, 6),
        "native_ppl": round(math.exp(native_nll), 2),
        "oracle_nll": round(oracle_nll, 6),
        "oracle_ppl": round(math.exp(oracle_nll), 2),
        "oracle_excess_nll": round(oracle_excess, 6),
        "swap_nll_mean": round(swap_mean, 6),
        "swap_nll_std": round(swap_std, 6),
        "swap_ppl": round(math.exp(swap_mean), 2),
        "delta_swap": round(delta_swap, 6),
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-1.7B multi-layer sparse fine-tuning with RCA")
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="./wikitext103_cache")
    parser.add_argument("--model-name", type=str,
                        default="/s3-data/models/qwen3-1.7b-base")
    args = parser.parse_args()
    train_experiment(
        condition=args.condition, seed=args.seed,
        output_path=args.output, data_dir=args.data_dir,
        model_name=args.model_name,
    )


if __name__ == "__main__":
    main()
