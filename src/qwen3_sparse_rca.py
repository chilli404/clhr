"""Qwen3-1.7B single-layer sparse-attention RCA: scale confirmation.

Freeze the entire Qwen3-1.7B-Base except layer 14's Q/K/V/O projections.
Add soft bilinear gates and train for 5000 steps under four conditions:

  learned_gate         — standard end-to-end (baseline, matches published)
  contemporary_replay  — dual-loss with current gate masks (compute control)
  shuffled_historical  — historical gate masks with token permutation
  coherent_oldest_first — historical gate masks from oldest snapshot (proposed)

After training, evaluate:
  - Native NLL (model with its own trained gate)
  - Oracle hard-top-k NLL (top-k of Q@K^T, k=64)
  - 20-draw random-mask swap NLL
  - Pre-mask attention entropy and top-64 retained mass

Usage:
    python src/qwen3_sparse_rca.py \
        --condition coherent_oldest_first --seed 42 \
        --output /s3-data/results/qwen3_rca_coherent_oldest_first_s42.json
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

CONDITIONS = [
    "learned_gate",
    "contemporary_replay",
    "shuffled_historical",
    "coherent_oldest_first",
]

TARGET_LAYER = 14
D_GATE = 32
K_SPARSE = 64
TRAIN_STEPS = 5000
SNAPSHOT_INTERVAL = 500
N_SNAPSHOTS = 5
LAMBDA_RCA = 0.3


# ═══════════════════════════════════════════════════════════════════════
# Helpers (adapted from routing-absorption qwen3_absorption.py)
# ═══════════════════════════════════════════════════════════════════════

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def repeat_kv(hidden_states, n_rep):
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    return (
        hidden_states[:, :, None, :, :]
        .expand(batch, num_kv_heads, n_rep, slen, head_dim)
        .reshape(batch, num_kv_heads * n_rep, slen, head_dim)
    )


class SingleLayerGate(nn.Module):
    def __init__(self, n_heads, d_model, d_gate):
        super().__init__()
        self.n_heads = n_heads
        self.d_gate = d_gate
        self.W_gq = nn.Linear(d_model, n_heads * d_gate, bias=False)
        self.W_gk = nn.Linear(d_model, n_heads * d_gate, bias=False)

    def gate_scores(self, hidden_states):
        bs, seq, _ = hidden_states.shape
        g_q = self.W_gq(hidden_states).view(bs, seq, self.n_heads, self.d_gate).transpose(1, 2)
        g_k = self.W_gk(hidden_states).view(bs, seq, self.n_heads, self.d_gate).transpose(1, 2)
        return torch.matmul(g_q, g_k.transpose(-2, -1)) / (self.d_gate ** 0.5)


def select_oldest_snapshot(snapshots: list[dict]) -> dict:
    if not snapshots:
        raise ValueError("No snapshots available")
    return min(snapshots, key=lambda s: s["step"])


def shuffle_soft_mask(mask: torch.Tensor, seed: int | None = None) -> torch.Tensor:
    if seed is not None:
        torch.manual_seed(seed)
    shuffled = mask.clone()
    B, H, Tq, Tk = shuffled.shape
    for b in range(B):
        perm = torch.randperm(Tk, device=shuffled.device)
        shuffled[b] = shuffled[b, :, :, perm]
    return shuffled


# ═══════════════════════════════════════════════════════════════════════
# Forward pass (adapted from upstream, handles RoPE + GQA)
# ═══════════════════════════════════════════════════════════════════════

def forward_with_soft_gate(
    model, input_ids, gate, target_layer_idx,
    position_ids, causal_mask,
    forced_soft_mask=None,
):
    cfg = model.config
    n_heads = cfg.num_attention_heads
    n_kv_heads = cfg.num_key_value_heads
    n_groups = n_heads // n_kv_heads
    head_dim = cfg.hidden_size // n_heads
    bsz, sl = input_ids.shape

    with torch.no_grad():
        hidden_states = model.model.embed_tokens(input_ids)
        position_embeddings = model.model.rotary_emb(hidden_states, position_ids)

        for layer_idx in range(target_layer_idx):
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

            aw = torch.matmul(q, kk.transpose(2, 3)) / math.sqrt(head_dim)
            aw = aw + causal_mask
            aw = F.softmax(aw, dim=-1, dtype=torch.float32).to(q.dtype)
            ao = torch.matmul(aw, v).transpose(1, 2).contiguous().reshape(bsz, sl, -1)
            hidden_states = residual + attn.o_proj(ao)
            residual = hidden_states
            hidden_states = residual + layer.mlp(layer.post_attention_layernorm(hidden_states))

    hidden_states = hidden_states.detach().requires_grad_(True)

    layer = model.model.layers[target_layer_idx]
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

    if forced_soft_mask is not None:
        soft_mask = forced_soft_mask
    else:
        g_scores = gate.gate_scores(normed.float())
        g_scores = g_scores.masked_fill(causal_mask[:, :, :sl, :sl] < -1e4, -10.0)
        soft_mask = torch.sigmoid(g_scores)

    aw = aw * soft_mask
    aw = aw + causal_mask.float()
    aw = F.softmax(aw, dim=-1)
    ao = torch.matmul(aw, v.float()).to(hidden_states.dtype)
    ao = ao.transpose(1, 2).contiguous().reshape(bsz, sl, -1)
    hidden_states = residual + attn.o_proj(ao)
    residual = hidden_states
    hidden_states = residual + layer.mlp(layer.post_attention_layernorm(hidden_states))

    for layer_idx in range(target_layer_idx + 1, len(model.model.layers)):
        layer = model.model.layers[layer_idx]
        residual = hidden_states
        normed = layer.input_layernorm(hidden_states)
        attn_l = layer.self_attn
        q = attn_l.q_proj(normed).view(bsz, sl, n_heads, head_dim)
        kk = attn_l.k_proj(normed).view(bsz, sl, n_kv_heads, head_dim)
        v = attn_l.v_proj(normed).view(bsz, sl, n_kv_heads, head_dim).transpose(1, 2)
        if hasattr(attn_l, "q_norm"):
            q = attn_l.q_norm(q)
            kk = attn_l.k_norm(kk)
        q, kk = q.transpose(1, 2), kk.transpose(1, 2)
        cos, sin = position_embeddings
        q, kk = apply_rotary_pos_emb(q, kk, cos, sin)
        kk, v = repeat_kv(kk, n_groups), repeat_kv(v, n_groups)
        aw_l = torch.matmul(q, kk.transpose(2, 3)) / math.sqrt(head_dim)
        aw_l = aw_l + causal_mask
        aw_l = F.softmax(aw_l, dim=-1, dtype=torch.float32).to(q.dtype)
        ao = torch.matmul(aw_l, v).transpose(1, 2).contiguous().reshape(bsz, sl, -1)
        hidden_states = residual + attn_l.o_proj(ao)
        residual = hidden_states
        hidden_states = residual + layer.mlp(layer.post_attention_layernorm(hidden_states))

    hidden_states = model.model.norm(hidden_states)
    return model.lm_head(hidden_states)


def get_hidden_before_target(model, input_ids, target_layer_idx, position_ids, causal_mask):
    """Run frozen layers before target and return hidden states + position embeddings."""
    cfg = model.config
    n_heads = cfg.num_attention_heads
    n_kv_heads = cfg.num_key_value_heads
    n_groups = n_heads // n_kv_heads
    head_dim = cfg.hidden_size // n_heads
    bsz, sl = input_ids.shape

    with torch.no_grad():
        hidden_states = model.model.embed_tokens(input_ids)
        position_embeddings = model.model.rotary_emb(hidden_states, position_ids)
        for layer_idx in range(target_layer_idx):
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
            aw = torch.matmul(q, kk.transpose(2, 3)) / math.sqrt(head_dim)
            aw = aw + causal_mask
            aw = F.softmax(aw, dim=-1, dtype=torch.float32).to(q.dtype)
            ao = torch.matmul(aw, v).transpose(1, 2).contiguous().reshape(bsz, sl, -1)
            hidden_states = residual + attn.o_proj(ao)
            residual = hidden_states
            hidden_states = residual + layer.mlp(layer.post_attention_layernorm(hidden_states))
    return hidden_states, position_embeddings


# ═══════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════

def load_wikitext_qwen3(data_dir: str, seq_len: int = 512):
    cache_path = Path(data_dir) / "wikitext103_qwen3"
    train_pt = cache_path / "train.pt"
    val_pt = cache_path / "validation.pt"

    if train_pt.exists() and val_pt.exists():
        print("  Loading cached Qwen3-tokenized WikiText-103...")
        train_data = torch.load(train_pt, weights_only=False)
        val_data = torch.load(val_pt, weights_only=False)
    else:
        print("  Tokenizing WikiText-103 with Qwen3 tokenizer...")
        from datasets import load_dataset
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("/s3-data/models/qwen3-1.7b-base", local_files_only=True)
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")

        def tokenize_and_chunk(split_name):
            all_tokens = []
            for example in ds[split_name]:
                text = example["text"]
                if text.strip():
                    all_tokens.extend(tokenizer.encode(text))
            tokens = torch.tensor(all_tokens, dtype=torch.long)
            n_chunks = len(tokens) // seq_len
            tokens = tokens[: n_chunks * seq_len]
            return {"input_ids": tokens.view(n_chunks, seq_len)}

        cache_path.mkdir(parents=True, exist_ok=True)
        train_data = tokenize_and_chunk("train")
        val_data = tokenize_and_chunk("validation")
        torch.save(train_data, train_pt)
        torch.save(val_data, val_pt)
        print(f"  Cached: train={train_data['input_ids'].shape[0]}, val={val_data['input_ids'].shape[0]} chunks")

    train_ds = TensorDataset(train_data["input_ids"])
    val_ds = TensorDataset(val_data["input_ids"])
    return train_ds, val_ds


# ═══════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════

def eval_nll(model, loader, gate, target_layer, position_ids_fn, causal_mask_fn,
             device, max_batches=30, forced_soft_mask_fn=None):
    model.eval()
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
            if forced_soft_mask_fn is not None:
                hidden, _ = get_hidden_before_target(model, inputs, target_layer, pos_ids, cm)
                forced = forced_soft_mask_fn(model, hidden, inputs, sl, device)

            logits = forward_with_soft_gate(model, inputs, gate, target_layer, pos_ids, cm,
                                            forced_soft_mask=forced)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   labels.reshape(-1), reduction="sum")
            total_loss += loss.item()
            total_tokens += labels.numel()
    return total_loss / total_tokens


def make_oracle_mask_fn(k=64):
    def fn(model, hidden_before_target, input_ids, sl, device):
        cfg = model.config
        n_heads = cfg.num_attention_heads
        n_kv_heads = cfg.num_key_value_heads
        n_groups = n_heads // n_kv_heads
        head_dim = cfg.hidden_size // n_heads
        bsz = input_ids.shape[0]

        layer = model.model.layers[TARGET_LAYER]
        normed = layer.input_layernorm(hidden_before_target)
        attn = layer.self_attn
        q = attn.q_proj(normed).view(bsz, sl, n_heads, head_dim).transpose(1, 2)
        kk = attn.k_proj(normed).view(bsz, sl, n_kv_heads, head_dim).transpose(1, 2)
        if hasattr(attn, "q_norm"):
            q_3d = q.transpose(1, 2)
            kk_3d = kk.transpose(1, 2)
            q_3d = attn.q_norm(q_3d)
            kk_3d = attn.k_norm(kk_3d)
            q = q_3d.transpose(1, 2)
            kk = kk_3d.transpose(1, 2)
        kk = repeat_kv(kk, n_groups)

        scores = torch.matmul(q.float(), kk.float().transpose(-2, -1)) / math.sqrt(head_dim)
        causal = torch.tril(torch.ones(sl, sl, device=device)).unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(causal == 0, float("-inf"))
        actual_k = min(k, sl)
        _, topk_idx = torch.topk(scores, actual_k, dim=-1)
        mask = torch.zeros_like(scores)
        mask.scatter_(-1, topk_idx, 1.0)
        return mask * causal
    return fn


def make_random_mask_fn(n_heads, k=64):
    def fn(model, hidden_before_target, input_ids, sl, device):
        bsz = input_ids.shape[0]
        rand_scores = torch.rand(bsz, n_heads, sl, sl, device=device)
        actual_k = min(k, sl)
        _, topk_idx = torch.topk(rand_scores, actual_k, dim=-1)
        mask = torch.zeros_like(rand_scores)
        mask.scatter_(-1, topk_idx, 1.0)
        causal = torch.tril(torch.ones(sl, sl, device=device)).unsqueeze(0).unsqueeze(0)
        return mask * causal
    return fn


# ═══════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════

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
    d_model = cfg.hidden_size

    for p in model.parameters():
        p.requires_grad = False

    target_attn = model.model.layers[TARGET_LAYER].self_attn
    unfrozen_params = []
    for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
        for p in getattr(target_attn, name).parameters():
            p.requires_grad = True
            unfrozen_params.append(p)
    for name in ["q_norm", "k_norm"]:
        if hasattr(target_attn, name):
            for p in getattr(target_attn, name).parameters():
                p.requires_grad = True
                unfrozen_params.append(p)

    print(f"  Unfrozen params: {sum(p.numel() for p in unfrozen_params):,}")

    initial_attn_state = {}
    for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
        initial_attn_state[name] = copy.deepcopy(getattr(target_attn, name).state_dict())
    for name in ["q_norm", "k_norm"]:
        if hasattr(target_attn, name):
            initial_attn_state[name] = copy.deepcopy(getattr(target_attn, name).state_dict())

    train_ds, val_ds = load_wikitext_qwen3(data_dir)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=4, num_workers=0)

    gate = SingleLayerGate(n_heads, d_model, D_GATE).to(device)
    gate_params = list(gate.parameters())

    all_trainable = gate_params + unfrozen_params
    optimizer = torch.optim.AdamW(all_trainable, lr=1e-4, weight_decay=0.01)

    use_rca = condition in ("contemporary_replay", "shuffled_historical", "coherent_oldest_first")
    snapshots = []

    hist_gate = None
    if use_rca:
        hist_gate = SingleLayerGate(n_heads, d_model, D_GATE).to(device)
        hist_gate.eval()

    def pos_ids_fn(bsz, sl, dev):
        return torch.arange(sl, device=dev).unsqueeze(0).expand(bsz, -1)

    def cm_fn(sl, dev):
        return torch.triu(
            torch.full((sl, sl), torch.finfo(torch.bfloat16).min, device=dev, dtype=torch.bfloat16),
            diagonal=1,
        ).unsqueeze(0).unsqueeze(0)

    model.eval()
    gate.train()
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

        logits = forward_with_soft_gate(model, inputs, gate, TARGET_LAYER, pos_ids, cm)
        lm_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

        rca_loss = torch.tensor(0.0, device=device)
        if use_rca and snapshots:
            if condition == "coherent_oldest_first":
                snap = select_oldest_snapshot(snapshots)
            elif condition == "contemporary_replay":
                snap = {"step": step, "gate_state": copy.deepcopy(gate.state_dict())}
            else:
                snap = snapshots[step % len(snapshots)]

            hist_gate.load_state_dict(snap["gate_state"])
            hist_gate.eval()
            with torch.no_grad():
                hidden, _ = get_hidden_before_target(model, inputs, TARGET_LAYER, pos_ids, cm)
                normed = model.model.layers[TARGET_LAYER].input_layernorm(hidden)
                hist_scores = hist_gate.gate_scores(normed.float())
                hist_scores = hist_scores.masked_fill(cm[:, :, :sl, :sl] < -1e4, -10.0)
                hist_mask = torch.sigmoid(hist_scores)

            if condition == "shuffled_historical":
                hist_mask = shuffle_soft_mask(hist_mask)

            logits_rca = forward_with_soft_gate(model, inputs, gate, TARGET_LAYER, pos_ids, cm,
                                                forced_soft_mask=hist_mask)
            rca_loss = F.cross_entropy(logits_rca.reshape(-1, logits_rca.size(-1)), labels.reshape(-1))

        loss = lm_loss + LAMBDA_RCA * rca_loss if use_rca and snapshots else lm_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_trainable, 1.0)
        optimizer.step()

        if step % SNAPSHOT_INTERVAL == 0 and use_rca:
            snapshots.append({
                "step": step,
                "gate_state": copy.deepcopy(gate.state_dict()),
            })
            if len(snapshots) > N_SNAPSHOTS:
                snapshots.pop(0)

        if step % 100 == 0:
            rca_str = f" rca={rca_loss.item():.4f}" if use_rca and snapshots else ""
            print(f"  step {step}/{TRAIN_STEPS}: loss={lm_loss.item():.4f}{rca_str} ({time.time()-t0:.0f}s)")

    # ═══════════════════════════════════════════════════════════════
    # Evaluation
    # ═══════════════════════════════════════════════════════════════
    print("\n  Final evaluation...")
    gate.eval()

    native_nll = eval_nll(model, val_loader, gate, TARGET_LAYER, pos_ids_fn, cm_fn, device)
    print(f"  Native NLL: {native_nll:.4f} (PPL: {math.exp(native_nll):.2f})")

    oracle_nll = eval_nll(model, val_loader, gate, TARGET_LAYER, pos_ids_fn, cm_fn, device,
                          forced_soft_mask_fn=make_oracle_mask_fn(K_SPARSE))
    print(f"  Oracle NLL: {oracle_nll:.4f} (PPL: {math.exp(oracle_nll):.2f})")

    swap_nlls = []
    for draw in range(20):
        nll = eval_nll(model, val_loader, gate, TARGET_LAYER, pos_ids_fn, cm_fn, device,
                       forced_soft_mask_fn=make_random_mask_fn(n_heads, K_SPARSE))
        swap_nlls.append(nll)
    swap_mean = float(np.mean(swap_nlls))
    swap_std = float(np.std(swap_nlls))
    print(f"  Swap NLL: {swap_mean:.4f} ± {swap_std:.4f} (PPL: {math.exp(swap_mean):.2f})")

    oracle_excess = oracle_nll - native_nll
    print(f"  Oracle excess: {oracle_excess:.4f} nats")

    results = {
        "condition": condition,
        "seed": seed,
        "target_layer": TARGET_LAYER,
        "train_steps": TRAIN_STEPS,
        "native_nll": round(native_nll, 6),
        "native_ppl": round(math.exp(native_nll), 2),
        "oracle_nll": round(oracle_nll, 6),
        "oracle_ppl": round(math.exp(oracle_nll), 2),
        "oracle_excess_nll": round(oracle_excess, 6),
        "swap_nll_mean": round(swap_mean, 6),
        "swap_nll_std": round(swap_std, 6),
        "swap_ppl": round(math.exp(swap_mean), 2),
        "delta_swap": round(swap_mean - native_nll, 6),
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Qwen3-1.7B single-layer sparse-attention RCA")
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="./wikitext103_cache")
    parser.add_argument("--model-name", type=str, default="/s3-data/models/qwen3-1.7b-base")
    args = parser.parse_args()

    train_experiment(
        condition=args.condition, seed=args.seed,
        output_path=args.output, data_dir=args.data_dir,
        model_name=args.model_name,
    )


if __name__ == "__main__":
    main()
