"""Closed-loop (online) hard deployment evaluation for sparse attention.

The critical distinction:
  OPEN-LOOP (current): extract masks from soft forward, replay in hard forward
  CLOSED-LOOP (this): compute masks online from the hard-path hidden state at each layer

Only the closed-loop evaluation establishes true one-pass direct deployability.

Four evaluation modes on the same checkpoint:
1. Native soft — training-time quality
2. Open-loop gate hardening — masks from soft path, replayed on hard path (current result)
3. Closed-loop gate hardening — masks computed online from hard path (TRUE deployment)
4. Closed-loop score-top-k — Q@K^T top-k computed online from hard path

Also reports per-layer mask drift (Jaccard between open-loop and closed-loop masks).

Usage:
    python src/sparse_attention_closed_loop_eval.py \
        --checkpoint-dir /s3-data/ckpts_sparse_rca \
        --data-dir ./wikitext103 \
        --output /s3-data/results/sparse_rca_closed_loop.json
"""
from __future__ import annotations

import argparse
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
from sparse_attention_rca import DEFAULT_CONFIG, load_wikitext, forward_with_masks
from sparse_attention_exact_protocol import prepare_data_upstream

CONDITIONS = [
    "learned_gate", "coherent_oldest_first", "coherent_hard_oldest_first",
    "contemporary_replay", "shuffled_historical",
    "contemporary_hard_replay", "shuffled_hard_historical",
    "coherent_closedloop_hard", "coherent_hard_annealed",
    "contemporary_closedloop_hard", "shuffled_closedloop_hard",
    "contemporary_shuffled_closedloop_hard",
    "contemporary_hard_replay",
    "ste_hard", "anneal_to_hard", "dual_ste_clhr",
]


def forward_closed_loop_gate_hard(
    model: GatedTransformer,
    input_ids: torch.Tensor,
    k: int = 64,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """One-pass closed-loop hard deployment using the model's learned gate.

    At each layer: compute gate scores from the CURRENT hard-path hidden state,
    apply hard top-k, update hidden state, proceed to next layer.
    This is true direct deployment — no soft forward needed.
    """
    model.eval()
    batch_size, seq_len = input_ids.shape
    device = input_ids.device
    n_heads = model.config.n_heads
    d_head = model.config.d_model // n_heads
    d_gate = model.config.d_gate

    positions = torch.arange(seq_len, device=device).unsqueeze(0)
    x = model.embedding(input_ids) + model.pos_embedding(positions)
    x = model.dropout(x)

    causal = torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0).unsqueeze(0)
    masks_used = []

    with torch.no_grad():
        for layer in model.layers:
            h = layer.attn_norm(x)
            attn = layer.attention

            q = attn.W_q(h).view(batch_size, seq_len, n_heads, d_head).transpose(1, 2)
            kk = attn.W_k(h).view(batch_size, seq_len, n_heads, d_head).transpose(1, 2)
            v = attn.W_v(h).view(batch_size, seq_len, n_heads, d_head).transpose(1, 2)

            attn_scores = torch.matmul(q, kk.transpose(-2, -1)) / (d_head ** 0.5)

            # Compute gate mask from CURRENT hard-path hidden state
            if hasattr(attn, 'W_gq') and not attn.config.freeze_gates:
                gq = attn.W_gq(h).view(batch_size, seq_len, n_heads, d_gate).transpose(1, 2)
                gk = attn.W_gk(h).view(batch_size, seq_len, n_heads, d_gate).transpose(1, 2)
                gate_scores = torch.matmul(gq, gk.transpose(-2, -1)) / (d_gate ** 0.5)
                gate_scores = gate_scores.masked_fill(causal == 0, float("-inf"))

                actual_k = min(k, seq_len)
                _, topk_idx = torch.topk(gate_scores, actual_k, dim=-1)
                hard_mask = torch.zeros_like(gate_scores)
                hard_mask.scatter_(-1, topk_idx, 1.0)
                hard_mask = hard_mask * causal
            else:
                hard_mask = torch.ones(batch_size, n_heads, seq_len, seq_len, device=device)

            masks_used.append(hard_mask)

            # Apply hard mask: zero out unselected, then -inf for softmax
            attn_scores = attn_scores.masked_fill(hard_mask == 0, float("-inf"))
            attn_scores = attn_scores.masked_fill(causal == 0, float("-inf"))

            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

            output = torch.matmul(attn_weights, v)
            output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
            output = attn.W_o(output)

            x = x + output
            x = x + layer.ff(layer.ff_norm(x))

    x = model.final_norm(x)
    logits = model.lm_head(x)
    return logits, masks_used


def forward_closed_loop_score_topk(
    model: GatedTransformer,
    input_ids: torch.Tensor,
    k: int = 64,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """One-pass closed-loop using score-top-k (Q@K^T) from current hard-path state."""
    model.eval()
    batch_size, seq_len = input_ids.shape
    device = input_ids.device
    n_heads = model.config.n_heads
    d_head = model.config.d_model // n_heads

    positions = torch.arange(seq_len, device=device).unsqueeze(0)
    x = model.embedding(input_ids) + model.pos_embedding(positions)
    x = model.dropout(x)

    causal = torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0).unsqueeze(0)
    masks_used = []

    with torch.no_grad():
        for layer in model.layers:
            h = layer.attn_norm(x)
            attn = layer.attention

            q = attn.W_q(h).view(batch_size, seq_len, n_heads, d_head).transpose(1, 2)
            kk = attn.W_k(h).view(batch_size, seq_len, n_heads, d_head).transpose(1, 2)
            v = attn.W_v(h).view(batch_size, seq_len, n_heads, d_head).transpose(1, 2)

            attn_scores = torch.matmul(q, kk.transpose(-2, -1)) / (d_head ** 0.5)

            # Score-top-k from CURRENT hard-path Q@K^T
            scores_for_topk = attn_scores.clone()
            scores_for_topk = scores_for_topk.masked_fill(causal == 0, float("-inf"))

            actual_k = min(k, seq_len)
            _, topk_idx = torch.topk(scores_for_topk, actual_k, dim=-1)
            hard_mask = torch.zeros_like(scores_for_topk)
            hard_mask.scatter_(-1, topk_idx, 1.0)
            hard_mask = hard_mask * causal

            masks_used.append(hard_mask)

            attn_scores = attn_scores.masked_fill(hard_mask == 0, float("-inf"))
            attn_scores = attn_scores.masked_fill(causal == 0, float("-inf"))

            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

            output = torch.matmul(attn_weights, v)
            output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
            output = attn.W_o(output)

            x = x + output
            x = x + layer.ff(layer.ff_norm(x))

    x = model.final_norm(x)
    logits = model.lm_head(x)
    return logits, masks_used


def forward_partial_hard(
    model: GatedTransformer,
    input_ids: torch.Tensor,
    hard_layers: int,
    k: int = 64,
) -> torch.Tensor:
    """Single-pass forward: first `hard_layers` use closed-loop hard gate,
    remaining layers use native soft gating.

    This produces a progressive hardening curve showing where errors compound.
    """
    model.eval()
    batch_size, seq_len = input_ids.shape
    device = input_ids.device
    n_heads = model.config.n_heads
    d_head = model.config.d_model // n_heads
    d_gate = model.config.d_gate
    n_layers = len(model.layers)

    positions = torch.arange(seq_len, device=device).unsqueeze(0)
    x = model.embedding(input_ids) + model.pos_embedding(positions)
    x = model.dropout(x)

    causal = torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        for li, layer in enumerate(model.layers):
            if li < hard_layers:
                # Hard: compute gate from current state, harden to top-k
                h = layer.attn_norm(x)
                attn = layer.attention
                q = attn.W_q(h).view(batch_size, seq_len, n_heads, d_head).transpose(1, 2)
                kk = attn.W_k(h).view(batch_size, seq_len, n_heads, d_head).transpose(1, 2)
                v = attn.W_v(h).view(batch_size, seq_len, n_heads, d_head).transpose(1, 2)
                attn_scores = torch.matmul(q, kk.transpose(-2, -1)) / (d_head ** 0.5)

                if hasattr(attn, 'W_gq') and not attn.config.freeze_gates:
                    gq = attn.W_gq(h).view(batch_size, seq_len, n_heads, d_gate).transpose(1, 2)
                    gk = attn.W_gk(h).view(batch_size, seq_len, n_heads, d_gate).transpose(1, 2)
                    gate_scores = torch.matmul(gq, gk.transpose(-2, -1)) / (d_gate ** 0.5)
                    gate_scores = gate_scores.masked_fill(causal == 0, float("-inf"))
                    actual_k = min(k, seq_len)
                    _, topk_idx = torch.topk(gate_scores, actual_k, dim=-1)
                    hard_mask = torch.zeros_like(gate_scores)
                    hard_mask.scatter_(-1, topk_idx, 1.0)
                    hard_mask = hard_mask * causal
                else:
                    hard_mask = torch.ones(batch_size, n_heads, seq_len, seq_len, device=device)

                attn_scores = attn_scores.masked_fill(hard_mask == 0, float("-inf"))
                attn_scores = attn_scores.masked_fill(causal == 0, float("-inf"))
                attn_weights = F.softmax(attn_scores, dim=-1)
                attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
                output = torch.matmul(attn_weights, v)
                output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
                output = attn.W_o(output)
                x = x + output
                x = x + layer.ff(layer.ff_norm(x))
            else:
                # Soft: use native model forward for this layer
                x = x + layer.attention(layer.attn_norm(x), causal)
                x = x + layer.ff(layer.ff_norm(x))

    x = model.final_norm(x)
    return model.lm_head(x)


def measure_layerwise_drift(model, loader, device, k=64, max_batches=30):
    """Measure per-layer drift between soft and hard trajectories.

    Returns per-layer:
      delta_l: mean L2 distance between soft and hard hidden states
      cosine_l: mean cosine similarity between soft and hard hidden states
      jaccard_l: per-query-averaged Jaccard between soft and hard gate masks
      margin_l: mean gate score margin (k-th score minus (k+1)-th score)
      mass_l: fraction of full softmax mass retained by hard top-k
    """
    model.eval()
    n_layers = len(model.layers)
    n_heads = model.config.n_heads
    d_head = model.config.d_model // n_heads
    d_gate = model.config.d_gate

    deltas = [[] for _ in range(n_layers)]
    cosines = [[] for _ in range(n_layers)]
    jaccards = [[] for _ in range(n_layers)]
    margins = [[] for _ in range(n_layers)]
    masses = [[] for _ in range(n_layers)]
    post_ln_cosines = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            ids = batch["input_ids"].to(device)
            inp = ids[:, :-1]
            bs, sl = inp.shape

            positions = torch.arange(sl, device=device).unsqueeze(0)
            causal = torch.tril(torch.ones(sl, sl, device=device)).unsqueeze(0).unsqueeze(0)

            # Start both trajectories from the same embedding
            emb = model.embedding(inp) + model.pos_embedding(positions)
            emb = model.dropout(emb)
            x_soft = emb.clone()
            x_hard = emb.clone()

            for li, layer in enumerate(model.layers):
                attn = layer.attention

                # --- Soft trajectory: native forward ---
                h_s = layer.attn_norm(x_soft)
                x_soft = x_soft + layer.attention(h_s, causal)
                x_soft = x_soft + layer.ff(layer.ff_norm(x_soft))

                # --- Hard trajectory: closed-loop hard gate ---
                h_h = layer.attn_norm(x_hard)
                q = attn.W_q(h_h).view(bs, sl, n_heads, d_head).transpose(1, 2)
                kk = attn.W_k(h_h).view(bs, sl, n_heads, d_head).transpose(1, 2)
                v = attn.W_v(h_h).view(bs, sl, n_heads, d_head).transpose(1, 2)
                scores = torch.matmul(q, kk.transpose(-2, -1)) / (d_head ** 0.5)

                if hasattr(attn, 'W_gq') and not attn.config.freeze_gates:
                    gq = attn.W_gq(h_h).view(bs, sl, n_heads, d_gate).transpose(1, 2)
                    gk = attn.W_gk(h_h).view(bs, sl, n_heads, d_gate).transpose(1, 2)
                    gate_sc = torch.matmul(gq, gk.transpose(-2, -1)) / (d_gate ** 0.5)
                    gate_sc_causal = gate_sc.masked_fill(causal == 0, float("-inf"))
                    actual_k = min(k, sl)
                    topk_vals, topk_idx = torch.topk(gate_sc_causal, actual_k, dim=-1)
                    hard_mask_h = torch.zeros_like(gate_sc_causal)
                    hard_mask_h.scatter_(-1, topk_idx, 1.0)
                    hard_mask_h = hard_mask_h * causal

                    # Gate margin: k-th minus (k+1)-th score per query
                    # Use finite values only to avoid -inf contamination
                    if actual_k < sl:
                        finite_sc = gate_sc_causal.clone()
                        finite_sc[finite_sc == float("-inf")] = float("nan")
                        sorted_sc, _ = finite_sc.sort(dim=-1, descending=True)
                        kth = sorted_sc[:, :, :, actual_k - 1]
                        kp1 = sorted_sc[:, :, :, actual_k]
                        valid = torch.isfinite(kth) & torch.isfinite(kp1)
                        if valid.any():
                            margin = (kth[valid] - kp1[valid]).mean()
                            margins[li].append(margin.item())
                else:
                    hard_mask_h = torch.ones(bs, n_heads, sl, sl, device=device)

                scores_h = scores.masked_fill(hard_mask_h == 0, float("-inf"))
                scores_h = scores_h.masked_fill(causal == 0, float("-inf"))
                w_h = F.softmax(scores_h, dim=-1)
                w_h = torch.nan_to_num(w_h, nan=0.0)

                out_h = torch.matmul(w_h, v)
                out_h = out_h.transpose(1, 2).contiguous().view(bs, sl, -1)
                out_h = attn.W_o(out_h)
                x_hard = x_hard + out_h
                x_hard = x_hard + layer.ff(layer.ff_norm(x_hard))

                # --- Compute soft mask for Jaccard & mass ---
                h_s_for_gate = layer.attn_norm(x_soft - layer.ff(layer.ff_norm(x_soft - (x_soft))))
                # Simpler: recompute soft gate from original soft hidden
                # Actually use the soft trajectory's pre-attention state
                # We already have h_s from before the soft forward, but x_soft changed.
                # Recompute from the soft embedding lineage stored in x_soft:
                # Actually we need the mask the soft forward USED. The model stores it.
                soft_mask_li = attn.last_mask
                if soft_mask_li is not None:
                    soft_mask_bin = (soft_mask_li > 0.5).bool()
                    hard_mask_bin = hard_mask_h.bool()
                    # Per-query Jaccard: compute per (batch, head, query), then average
                    inter = (soft_mask_bin & hard_mask_bin).float().sum(dim=-1)  # (B, H, Q)
                    union = (soft_mask_bin | hard_mask_bin).float().sum(dim=-1)
                    jq = inter / union.clamp(min=1)
                    jaccards[li].append(jq.mean().item())

                # Retained mass: what fraction of full softmax mass does hard top-k keep?
                full_w = F.softmax(scores.masked_fill(causal == 0, float("-inf")), dim=-1)
                full_w = torch.nan_to_num(full_w, nan=0.0)
                retained = (full_w * hard_mask_h).sum(dim=-1).mean()
                masses[li].append(retained.item())

                # Hidden state drift
                diff = x_hard - x_soft
                delta = diff.norm(dim=-1).mean()  # mean over batch and seq
                deltas[li].append(delta.item())

                cos = F.cosine_similarity(
                    x_hard.reshape(-1, x_hard.shape[-1]),
                    x_soft.reshape(-1, x_soft.shape[-1]),
                    dim=-1,
                ).mean()
                cosines[li].append(cos.item())

            # Post-final-LayerNorm cosine (measured after all layers, per batch)
            if hasattr(model, 'final_norm'):
                ln_h = model.final_norm(x_hard)
                ln_s = model.final_norm(x_soft)
                post_cos = F.cosine_similarity(
                    ln_h.reshape(-1, ln_h.shape[-1]),
                    ln_s.reshape(-1, ln_s.shape[-1]),
                    dim=-1,
                ).mean()
                post_ln_cosines.append(post_cos.item())

    return {
        "layerwise_delta": [round(float(np.mean(d)), 4) if d else 0.0 for d in deltas],
        "layerwise_cosine": [round(float(np.mean(c)), 4) if c else 0.0 for c in cosines],
        "layerwise_jaccard": [round(float(np.mean(j)), 4) if j else 0.0 for j in jaccards],
        "layerwise_margin": [round(float(np.mean(m)), 4) if m else 0.0 for m in margins],
        "layerwise_mass": [round(float(np.mean(m)), 4) if m else 0.0 for m in masses],
        "post_ln_cosine_L": round(float(np.mean(post_ln_cosines)), 6) if post_ln_cosines else None,
    }


def eval_partial_hardening(model, loader, device, k=64, max_batches=100):
    """Evaluate NLL for progressive hardening: first r layers hard, rest soft."""
    model.eval()
    n_layers = len(model.layers)
    results = []

    for r in range(n_layers + 1):
        total_loss = 0.0
        total_tokens = 0
        with torch.no_grad():
            for i, batch in enumerate(loader):
                if i >= max_batches:
                    break
                ids = batch["input_ids"].to(device)
                x, y = ids[:, :-1], ids[:, 1:]
                logits = forward_partial_hard(model, x, hard_layers=r, k=k)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum"
                )
                total_loss += loss.item()
                total_tokens += y.numel()
        nll = total_loss / total_tokens
        results.append(round(nll, 6))

    return results


def eval_closed_loop(model, loader, device, mode, k=64, max_batches=200):
    """Evaluate with closed-loop hard deployment."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            ids = batch["input_ids"].to(device)
            x, y = ids[:, :-1], ids[:, 1:]

            if mode == "closed_gate":
                logits, _ = forward_closed_loop_gate_hard(model, x, k=k)
            elif mode == "closed_score":
                logits, _ = forward_closed_loop_score_topk(model, x, k=k)
            else:
                raise ValueError(f"Unknown mode: {mode}")

            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum"
            )
            total_loss += loss.item()
            total_tokens += y.numel()

    return total_loss / total_tokens


def compute_mask_drift(model, loader, device, k=64, max_batches=50):
    """Compare open-loop (soft-state) vs closed-loop (hard-state) masks per layer."""
    from sparse_attention_oracle_audit import (
        compute_direct_hardened_masks, compute_oracle_topk_masks,
    )

    model.eval()
    n_layers = len(model.layers)
    gate_jaccard_by_layer = [[] for _ in range(n_layers)]
    score_jaccard_by_layer = [[] for _ in range(n_layers)]

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            ids = batch["input_ids"].to(device)
            x = ids[:, :-1]

            # Open-loop masks (from soft trajectory)
            open_gate_masks = compute_direct_hardened_masks(model, x, k=k)
            open_score_masks = compute_oracle_topk_masks(model, x, k=k)

            # Closed-loop masks (from hard trajectory)
            _, closed_gate_masks = forward_closed_loop_gate_hard(model, x, k=k)
            _, closed_score_masks = forward_closed_loop_score_topk(model, x, k=k)

            for li in range(n_layers):
                og = open_gate_masks[li].bool()
                cg = closed_gate_masks[li].bool()
                intersection = (og & cg).float().sum()
                union = (og | cg).float().sum()
                if union > 0:
                    gate_jaccard_by_layer[li].append((intersection / union).item())

                os = open_score_masks[li].bool()
                cs = closed_score_masks[li].bool()
                intersection = (os & cs).float().sum()
                union = (os | cs).float().sum()
                if union > 0:
                    score_jaccard_by_layer[li].append((intersection / union).item())

    return {
        "gate_jaccard": [round(float(np.mean(j)), 4) if j else 0.0 for j in gate_jaccard_by_layer],
        "score_jaccard": [round(float(np.mean(j)), 4) if j else 0.0 for j in score_jaccard_by_layer],
    }


def eval_native(model, loader, device, max_batches=200):
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


def compute_gate_oracle_agreement(gate_mask: torch.Tensor, oracle_mask: torch.Tensor) -> float:
    """Compute F1 agreement between a gate's hard mask and the score-top-k oracle.

    Both inputs are binary tensors of shape (batch, heads, seq, seq).
    Returns macro-averaged F1 across all positions.
    """
    gate_bool = gate_mask.bool()
    oracle_bool = oracle_mask.bool()

    tp = (gate_bool & oracle_bool).float().sum().item()
    fp = (gate_bool & ~oracle_bool).float().sum().item()
    fn = (~gate_bool & oracle_bool).float().sum().item()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1


def forward_closed_loop_with_fresh_gate(
    model: GatedTransformer,
    input_ids: torch.Tensor,
    fresh_gq_weights: list[torch.Tensor],
    fresh_gk_weights: list[torch.Tensor],
    k: int = 64,
) -> torch.Tensor:
    """Closed-loop hard forward using FRESH (distilled) gate projections.

    Same as forward_closed_loop_gate_hard but uses externally provided
    gate weights instead of the model's own W_gq/W_gk.
    """
    model.eval()
    batch_size, seq_len = input_ids.shape
    device = input_ids.device
    n_heads = model.config.n_heads
    d_head = model.config.d_model // n_heads
    d_gate = model.config.d_gate

    positions = torch.arange(seq_len, device=device).unsqueeze(0)
    x = model.embedding(input_ids) + model.pos_embedding(positions)
    x = model.dropout(x)

    causal = torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        for li, layer in enumerate(model.layers):
            h = layer.attn_norm(x)
            attn = layer.attention

            q = attn.W_q(h).view(batch_size, seq_len, n_heads, d_head).transpose(1, 2)
            kk = attn.W_k(h).view(batch_size, seq_len, n_heads, d_head).transpose(1, 2)
            v = attn.W_v(h).view(batch_size, seq_len, n_heads, d_head).transpose(1, 2)
            attn_scores = torch.matmul(q, kk.transpose(-2, -1)) / (d_head ** 0.5)

            gq = F.linear(h, fresh_gq_weights[li]).view(batch_size, seq_len, n_heads, d_gate).transpose(1, 2)
            gk = F.linear(h, fresh_gk_weights[li]).view(batch_size, seq_len, n_heads, d_gate).transpose(1, 2)
            gate_scores = torch.matmul(gq, gk.transpose(-2, -1)) / (d_gate ** 0.5)
            gate_scores = gate_scores.masked_fill(causal == 0, float("-inf"))

            actual_k = min(k, seq_len)
            _, topk_idx = torch.topk(gate_scores, actual_k, dim=-1)
            hard_mask = torch.zeros_like(gate_scores)
            hard_mask.scatter_(-1, topk_idx, 1.0)
            hard_mask = hard_mask * causal

            attn_scores = attn_scores.masked_fill(hard_mask == 0, float("-inf"))
            attn_scores = attn_scores.masked_fill(causal == 0, float("-inf"))
            w = F.softmax(attn_scores, dim=-1)
            w = torch.nan_to_num(w, nan=0.0)

            output = torch.matmul(w, v)
            output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
            output = attn.W_o(output)

            x = x + output
            x = x + layer.ff(layer.ff_norm(x))

    x = model.final_norm(x)
    return model.lm_head(x)


def posthoc_kl_distillation(
    model, train_loader, val_loader, device,
    k=64, steps=1000, gate_lr=1e-3,
):
    """Post-hoc KL gate distillation: freeze Q/K/V, train fresh gate against oracle.

    Matches the routing-absorption paper's protocol. The gate is trained
    on oracle masks from the model's native soft forward. EVALUATION of
    the trained gate uses closed-loop hard deployment.
    """
    model.eval()
    config = model.config
    n_layers = len(model.layers)
    n_heads = config.n_heads
    d_model = config.d_model
    d_gate = config.d_gate
    d_head = d_model // n_heads

    for p in model.parameters():
        p.requires_grad = False

    fresh_gq = nn.ParameterList([
        nn.Parameter(torch.randn(n_heads * d_gate, d_model, device=device) * 0.02)
        for _ in range(n_layers)
    ])
    fresh_gk = nn.ParameterList([
        nn.Parameter(torch.randn(n_heads * d_gate, d_model, device=device) * 0.02)
        for _ in range(n_layers)
    ])

    optimizer = torch.optim.AdamW(
        list(fresh_gq.parameters()) + list(fresh_gk.parameters()),
        lr=gate_lr, weight_decay=0.0,
    )

    train_iter = iter(train_loader)
    causal_cache = {}

    for step in range(1, steps + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        ids = batch["input_ids"].to(device)
        x_in = ids[:, :-1]
        bs, sl = x_in.shape

        if sl not in causal_cache:
            causal_cache[sl] = torch.tril(torch.ones(sl, sl, device=device)).unsqueeze(0).unsqueeze(0)
        causal = causal_cache[sl]

        positions = torch.arange(sl, device=device).unsqueeze(0)
        h = model.embedding(x_in) + model.pos_embedding(positions)
        h = model.dropout(h)

        total_kl = torch.tensor(0.0, device=device)

        with torch.no_grad():
            layer_inputs = []
            for layer in model.layers:
                normed = layer.attn_norm(h)
                layer_inputs.append(normed.detach())
                h = h + layer.attention(normed, causal)
                h = h + layer.ff(layer.ff_norm(h))

        for li in range(n_layers):
            normed = layer_inputs[li]

            with torch.no_grad():
                attn = model.layers[li].attention
                q = attn.W_q(normed).view(bs, sl, n_heads, d_head).transpose(1, 2)
                kk = attn.W_k(normed).view(bs, sl, n_heads, d_head).transpose(1, 2)
                scores = torch.matmul(q, kk.transpose(-2, -1)) / (d_head ** 0.5)
                scores = scores.masked_fill(causal == 0, float("-inf"))
                actual_k = min(k, sl)
                _, topk_idx = torch.topk(scores, actual_k, dim=-1)
                oracle = torch.zeros_like(scores)
                oracle.scatter_(-1, topk_idx, 1.0)
                oracle = oracle * causal

            gq = F.linear(normed, fresh_gq[li]).view(bs, sl, n_heads, d_gate).transpose(1, 2)
            gk = F.linear(normed, fresh_gk[li]).view(bs, sl, n_heads, d_gate).transpose(1, 2)
            gate_logits = torch.matmul(gq, gk.transpose(-2, -1)) / (d_gate ** 0.5)

            mask_causal = causal.expand_as(gate_logits).bool()
            gate_probs = torch.sigmoid(gate_logits[mask_causal])
            oracle_flat = oracle[mask_causal]

            kl = oracle_flat * torch.log((oracle_flat + 1e-8) / (gate_probs + 1e-8)) + \
                 (1 - oracle_flat) * torch.log((1 - oracle_flat + 1e-8) / (1 - gate_probs + 1e-8))
            total_kl = total_kl + kl.mean()

        optimizer.zero_grad()
        total_kl.backward()
        optimizer.step()

        if step % 250 == 0:
            print(f"    KL distill step {step}/{steps}: loss={total_kl.item() / n_layers:.4f}")

    gq_weights = [p.data for p in fresh_gq]
    gk_weights = [p.data for p in fresh_gk]

    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= 100:
                break
            ids = batch["input_ids"].to(device)
            x_in, y = ids[:, :-1], ids[:, 1:]
            logits = forward_closed_loop_with_fresh_gate(model, x_in, gq_weights, gk_weights, k=k)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum")
            total_loss += loss.item()
            total_tokens += y.numel()

    deploy_nll = total_loss / total_tokens

    for p in model.parameters():
        p.requires_grad = True

    return {
        "deploy_nll": round(deploy_nll, 6),
        "deploy_ppl": round(math.exp(deploy_nll), 2),
        "final_kl": round(total_kl.item() / n_layers, 6),
        "steps": steps,
        "method": "kl",
    }


def posthoc_bce_distillation(
    model, train_loader, val_loader, device,
    k=64, steps=1000, gate_lr=1e-3,
):
    """Post-hoc BCE gate distillation. Same as KL but with BCE loss."""
    model.eval()
    config = model.config
    n_layers = len(model.layers)
    n_heads = config.n_heads
    d_model = config.d_model
    d_gate = config.d_gate
    d_head = d_model // n_heads

    for p in model.parameters():
        p.requires_grad = False

    fresh_gq = nn.ParameterList([
        nn.Parameter(torch.randn(n_heads * d_gate, d_model, device=device) * 0.02)
        for _ in range(n_layers)
    ])
    fresh_gk = nn.ParameterList([
        nn.Parameter(torch.randn(n_heads * d_gate, d_model, device=device) * 0.02)
        for _ in range(n_layers)
    ])

    optimizer = torch.optim.AdamW(
        list(fresh_gq.parameters()) + list(fresh_gk.parameters()),
        lr=gate_lr, weight_decay=0.0,
    )

    train_iter = iter(train_loader)
    causal_cache = {}

    for step in range(1, steps + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        ids = batch["input_ids"].to(device)
        x_in = ids[:, :-1]
        bs, sl = x_in.shape

        if sl not in causal_cache:
            causal_cache[sl] = torch.tril(torch.ones(sl, sl, device=device)).unsqueeze(0).unsqueeze(0)
        causal = causal_cache[sl]

        positions = torch.arange(sl, device=device).unsqueeze(0)
        h = model.embedding(x_in) + model.pos_embedding(positions)
        h = model.dropout(h)

        total_bce = torch.tensor(0.0, device=device)

        with torch.no_grad():
            layer_inputs = []
            for layer in model.layers:
                normed = layer.attn_norm(h)
                layer_inputs.append(normed.detach())
                h = h + layer.attention(normed, causal)
                h = h + layer.ff(layer.ff_norm(h))

        for li in range(n_layers):
            normed = layer_inputs[li]

            with torch.no_grad():
                attn = model.layers[li].attention
                q = attn.W_q(normed).view(bs, sl, n_heads, d_head).transpose(1, 2)
                kk = attn.W_k(normed).view(bs, sl, n_heads, d_head).transpose(1, 2)
                scores = torch.matmul(q, kk.transpose(-2, -1)) / (d_head ** 0.5)
                scores = scores.masked_fill(causal == 0, float("-inf"))
                actual_k = min(k, sl)
                _, topk_idx = torch.topk(scores, actual_k, dim=-1)
                oracle = torch.zeros_like(scores)
                oracle.scatter_(-1, topk_idx, 1.0)
                oracle = oracle * causal

            gq = F.linear(normed, fresh_gq[li]).view(bs, sl, n_heads, d_gate).transpose(1, 2)
            gk = F.linear(normed, fresh_gk[li]).view(bs, sl, n_heads, d_gate).transpose(1, 2)
            gate_logits = torch.matmul(gq, gk.transpose(-2, -1)) / (d_gate ** 0.5)

            mask_causal = causal.expand_as(gate_logits).bool()
            bce = F.binary_cross_entropy_with_logits(
                gate_logits[mask_causal], oracle[mask_causal]
            )
            total_bce = total_bce + bce

        optimizer.zero_grad()
        total_bce.backward()
        optimizer.step()

        if step % 250 == 0:
            print(f"    BCE distill step {step}/{steps}: loss={total_bce.item() / n_layers:.4f}")

    gq_weights = [p.data for p in fresh_gq]
    gk_weights = [p.data for p in fresh_gk]

    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= 100:
                break
            ids = batch["input_ids"].to(device)
            x_in, y = ids[:, :-1], ids[:, 1:]
            logits = forward_closed_loop_with_fresh_gate(model, x_in, gq_weights, gk_weights, k=k)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum")
            total_loss += loss.item()
            total_tokens += y.numel()

    deploy_nll = total_loss / total_tokens

    for p in model.parameters():
        p.requires_grad = True

    return {
        "deploy_nll": round(deploy_nll, 6),
        "deploy_ppl": round(math.exp(deploy_nll), 2),
        "final_bce": round(total_bce.item() / n_layers, 6),
        "steps": steps,
        "method": "bce",
    }


def run_audit(checkpoint_dir, data_dir, output_path, tag_prefix="sparse_rca", conditions=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if tag_prefix == "sparse_exact":
        train_ds, val_ds = prepare_data_upstream(data_dir)
    else:
        train_ds, val_ds = load_wikitext(data_dir)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True,
                              num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False,
                            num_workers=0, drop_last=True)

    results = {}
    seeds = [42, 123, 456, 789, 1001]

    eval_conditions = conditions if conditions is not None else CONDITIONS
    for cond in eval_conditions:
      for seed in seeds:
        tag = f"{tag_prefix}_{cond}_s{seed}"
        ckpt_path = Path(checkpoint_dir) / tag / "step_50000.pt"
        if not ckpt_path.exists():
            continue

        key = f"{cond}_s{seed}"
        print(f"\n{'='*60}")
        print(f"CONDITION: {cond}  SEED: {seed}")
        print(f"{'='*60}")

        config = DEFAULT_CONFIG.model_copy()
        if cond == "random_gate":
            config.freeze_gates = True

        model = GatedTransformer(config).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])

        # 1. Native soft
        native_nll = eval_native(model, val_loader, device)
        print(f"  Native soft NLL:         {native_nll:.4f} (PPL: {math.exp(native_nll):.2f})")

        # 2. Open-loop gate hardening (current method)
        from sparse_attention_oracle_audit import (
            evaluate_with_hardened_gate, evaluate_with_oracle_masks,
        )
        open_gate_nll = evaluate_with_hardened_gate(model, val_loader, device, k=64)
        print(f"  Open-loop gate hard NLL: {open_gate_nll:.4f} (PPL: {math.exp(open_gate_nll):.2f})")

        # 3. Closed-loop gate hardening (TRUE deployment)
        closed_gate_nll = eval_closed_loop(model, val_loader, device, "closed_gate", k=64)
        print(f"  CLOSED-LOOP gate hard:   {closed_gate_nll:.4f} (PPL: {math.exp(closed_gate_nll):.2f})")

        # 4. Open-loop score-top-k
        open_score_nll = evaluate_with_oracle_masks(model, val_loader, device, k=64)
        print(f"  Open-loop score-topk:    {open_score_nll:.4f} (PPL: {math.exp(open_score_nll):.2f})")

        # 5. Closed-loop score-top-k
        closed_score_nll = eval_closed_loop(model, val_loader, device, "closed_score", k=64)
        print(f"  CLOSED-LOOP score-topk:  {closed_score_nll:.4f} (PPL: {math.exp(closed_score_nll):.2f})")

        # 6. Layerwise drift (δ, J, margin, mass)
        print(f"  Computing layerwise drift...")
        drift_detail = measure_layerwise_drift(model, val_loader, device, k=64, max_batches=30)
        print(f"  Delta by layer:   {drift_detail['layerwise_delta']}")
        print(f"  Jaccard by layer: {drift_detail['layerwise_jaccard']}")
        print(f"  Cosine by layer:  {drift_detail['layerwise_cosine']}")
        print(f"  Margin by layer:  {drift_detail['layerwise_margin']}")
        print(f"  Mass by layer:    {drift_detail['layerwise_mass']}")

        # 7. Partial hardening curve
        print(f"  Computing partial hardening curve...")
        partial_nlls = eval_partial_hardening(model, val_loader, device, k=64, max_batches=50)
        partial_excess = [round(p - native_nll, 6) for p in partial_nlls]
        print(f"  Partial NLL (0..L hard): {partial_nlls}")
        print(f"  Partial excess:          {partial_excess}")

        # 8. Post-hoc KL distillation (closed-loop deployment)
        print(f"  Running post-hoc KL distillation (1000 steps)...")
        kl_result = posthoc_kl_distillation(model, train_loader, val_loader, device, k=64, steps=1000)
        print(f"  KL deploy NLL:    {kl_result['deploy_nll']:.4f} (PPL: {kl_result['deploy_ppl']:.2f})")

        # 9. Post-hoc BCE distillation (closed-loop deployment)
        print(f"  Running post-hoc BCE distillation (1000 steps)...")
        bce_result = posthoc_bce_distillation(model, train_loader, val_loader, device, k=64, steps=1000)
        print(f"  BCE deploy NLL:   {bce_result['deploy_nll']:.4f} (PPL: {bce_result['deploy_ppl']:.2f})")

        # 10. Gate-oracle F1 agreement (from closed-loop masks)
        print(f"  Computing gate-oracle agreement...")
        _, cl_gate_masks = forward_closed_loop_gate_hard(model, next(iter(val_loader))["input_ids"][:, :-1].to(device), k=64)
        _, cl_score_masks = forward_closed_loop_score_topk(model, next(iter(val_loader))["input_ids"][:, :-1].to(device), k=64)
        gate_f1s = []
        for li in range(len(cl_gate_masks)):
            f1 = compute_gate_oracle_agreement(cl_gate_masks[li], cl_score_masks[li])
            gate_f1s.append(round(f1, 4))
        mean_f1 = round(float(np.mean(gate_f1s)), 4)
        print(f"  Gate-oracle F1:   {mean_f1:.4f} (by layer: {gate_f1s})")

        results[key] = {
            "native_nll": round(native_nll, 6),
            "open_gate_nll": round(open_gate_nll, 6),
            "closed_gate_nll": round(closed_gate_nll, 6),
            "open_score_nll": round(open_score_nll, 6),
            "closed_score_nll": round(closed_score_nll, 6),
            "open_gate_excess": round(open_gate_nll - native_nll, 6),
            "closed_gate_excess": round(closed_gate_nll - native_nll, 6),
            "open_score_excess": round(open_score_nll - native_nll, 6),
            "closed_score_excess": round(closed_score_nll - native_nll, 6),
            "kl_deploy_nll": kl_result["deploy_nll"],
            "kl_deploy_excess": round(kl_result["deploy_nll"] - native_nll, 6),
            "bce_deploy_nll": bce_result["deploy_nll"],
            "bce_deploy_excess": round(bce_result["deploy_nll"] - native_nll, 6),
            "gate_oracle_f1_mean": mean_f1,
            "gate_oracle_f1_by_layer": gate_f1s,
            "layerwise_delta": drift_detail["layerwise_delta"],
            "layerwise_cosine": drift_detail["layerwise_cosine"],
            "layerwise_jaccard": drift_detail["layerwise_jaccard"],
            "layerwise_margin": drift_detail["layerwise_margin"],
            "layerwise_mass": drift_detail["layerwise_mass"],
            "post_ln_cosine_L": drift_detail.get("post_ln_cosine_L"),
            "partial_hardening_nll": partial_nlls,
            "partial_hardening_excess": partial_excess,
        }

    # Summary
    from collections import defaultdict
    by_cond = defaultdict(list)
    for key, r in results.items():
        cond = key.rsplit('_s', 1)[0]
        by_cond[cond].append(r)

    print(f"\n{'='*95}")
    print("CLOSED-LOOP vs OPEN-LOOP COMPARISON")
    print(f"{'='*95}")
    print(f"{'Condition':<35} {'Native':>7} {'OL-gate':>7} {'CL-gate':>7} {'OL-score':>7} {'CL-score':>7} {'J-gate':>7}")
    print(f"{'':35} {'NLL':>7} {'excess':>7} {'excess':>7} {'excess':>7} {'excess':>7} {'mean':>7}")
    print("-" * 90)
    for cond in eval_conditions:
        if cond not in by_cond:
            continue
        rs = by_cond[cond]
        n = np.mean([r['native_nll'] for r in rs])
        ol_g = np.mean([r['open_gate_excess'] for r in rs])
        cl_g = np.mean([r['closed_gate_excess'] for r in rs])
        ol_s = np.mean([r['open_score_excess'] for r in rs])
        cl_s = np.mean([r['closed_score_excess'] for r in rs])
        j = np.mean([np.mean(r['layerwise_jaccard']) for r in rs])
        print(f"{cond:<35} {n:>7.4f} {ol_g:>7.4f} {cl_g:>7.4f} {ol_s:>7.4f} {cl_s:>7.4f} {j:>7.4f}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Closed-loop hard deployment evaluation")
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--tag-prefix", type=str, default="sparse_rca")
    parser.add_argument("--conditions", type=str, default=None,
                        help="Comma-separated condition names to evaluate (default: all)")
    args = parser.parse_args()
    conds = args.conditions.split(",") if args.conditions else None
    run_audit(args.checkpoint_dir, args.data_dir, args.output, tag_prefix=args.tag_prefix, conditions=conds)


if __name__ == "__main__":
    main()
