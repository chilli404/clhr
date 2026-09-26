"""SSA-style alignment adapted to the exact 31M learned-router protocol.

This is a method-level control, not a reproduction of SSA's block-sparse model.
Following SSA, each batch propagates either the soft or sparse stream through
every layer, while both attention outputs are computed and aligned at each
layer. Here the two streams use the same learned token router: a differentiable
sigmoid mask for the soft stream and a detached, causally valid top-k mask for
the sparse stream.
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
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sparse_attention_exact_protocol import (
    eval_native,
    get_upstream_config,
    get_upstream_training_config,
    prepare_data_upstream,
)
from sparse_attention_routing_utility import (
    eval_closed_loop_gate_hard,
    eval_local_window,
    eval_random_hard,
)

import sys

RA_ROOT = Path(__file__).parent.parent.parent / "routing-absorption"
sys.path.insert(0, str(RA_ROOT))

from src.models.transformer import GatedTransformer
from src.training.losses import SparseAttentionLoss


def _paired_attention_outputs(attn, h, causal, topk):
    """Return soft and detached-top-k attention contexts before W_o."""
    batch_size, seq_len, _ = h.shape
    n_heads = attn.config.n_heads
    d_head = attn.config.d_model // n_heads
    d_gate = attn.config.d_gate

    q = attn.W_q(h).view(batch_size, seq_len, n_heads, d_head).transpose(1, 2)
    k = attn.W_k(h).view(batch_size, seq_len, n_heads, d_head).transpose(1, 2)
    v = attn.W_v(h).view(batch_size, seq_len, n_heads, d_head).transpose(1, 2)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_head)

    gq = attn.W_gq(h).view(batch_size, seq_len, n_heads, d_gate).transpose(1, 2)
    gk = attn.W_gk(h).view(batch_size, seq_len, n_heads, d_gate).transpose(1, 2)
    gate_scores = torch.matmul(gq, gk.transpose(-2, -1)) / math.sqrt(d_gate)
    soft_mask = torch.sigmoid(gate_scores / attn.config.temperature)

    soft_scores = (scores * soft_mask).masked_fill(causal == 0, float("-inf"))
    soft_weights = torch.nan_to_num(F.softmax(soft_scores, dim=-1), nan=0.0)
    soft_weights = attn.dropout(soft_weights)
    soft_context = torch.matmul(soft_weights, v)

    with torch.no_grad():
        eligible_scores = gate_scores.masked_fill(causal == 0, float("-inf"))
        indices = torch.topk(eligible_scores, min(topk, seq_len), dim=-1).indices
        hard_mask = torch.zeros_like(gate_scores).scatter_(-1, indices, 1.0) * causal

    hard_scores = scores.masked_fill(hard_mask == 0, float("-inf"))
    hard_weights = torch.nan_to_num(F.softmax(hard_scores, dim=-1), nan=0.0)
    hard_weights = attn.dropout(hard_weights)
    hard_context = torch.matmul(hard_weights, v)

    # Preserve the fields used by the exact-protocol sparsity regularizer.
    attn.last_gate_scores = gate_scores.detach()
    attn.last_mask = soft_mask.detach()
    attn.last_mask_live = soft_mask
    attn.last_attention = soft_weights.detach()

    return soft_context, hard_context


def ssa_style_forward(
    model, input_ids, *, propagate_soft=None, hard_probability=0.5, topk=64
):
    """Run the SSA-style layerwise paired computation.

    ``propagate_soft`` is sampled once per batch, as in the official SSA code,
    unless supplied explicitly for tests.
    """
    _, seq_len = input_ids.shape
    device = input_ids.device
    positions = torch.arange(seq_len, device=device).unsqueeze(0)
    hidden = model.embedding(input_ids) + model.pos_embedding(positions)
    hidden = model.dropout(hidden)
    causal = torch.tril(torch.ones(seq_len, seq_len, device=device))[None, None]

    if propagate_soft is None:
        propagate_soft = random.random() >= hard_probability

    alignment_terms = []
    for layer in model.layers:
        normed = layer.attn_norm(hidden)
        soft_context, hard_context = _paired_attention_outputs(
            layer.attention, normed, causal, topk
        )
        alignment_terms.append(
            F.smooth_l1_loss(soft_context, hard_context.detach())
            + F.smooth_l1_loss(soft_context.detach(), hard_context)
        )

        context = soft_context if propagate_soft else hard_context
        batch_size = input_ids.shape[0]
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        hidden = hidden + layer.attention.W_o(context)
        hidden = hidden + layer.ff(layer.ff_norm(hidden))

    logits = model.lm_head(model.final_norm(hidden))
    alignment = torch.stack(alignment_terms).mean()
    return logits, alignment, bool(propagate_soft)


def objective_matched_task_loss(losses, propagated_soft, hard_probability, clhr_weight):
    """Importance-weight one sampled task path to estimate the CLHR objective."""
    if not 0.0 < hard_probability < 1.0:
        raise ValueError("hard_probability must lie strictly between zero and one")
    if clhr_weight < 0.0:
        raise ValueError("clhr_weight must be nonnegative")
    if propagated_soft:
        return losses["loss_total"] / (1.0 - hard_probability)
    return clhr_weight * losses["loss_task"] / hard_probability


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    model_config = get_upstream_config()
    train_config = get_upstream_training_config()
    train_config.max_steps = args.max_steps
    train_ds, val_ds = prepare_data_upstream(args.data_dir)
    train_loader = DataLoader(
        train_ds, batch_size=train_config.batch_size, shuffle=True,
        num_workers=0, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=train_config.batch_size, shuffle=False, num_workers=0,
    )

    model = GatedTransformer(model_config).to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    criterion = SparseAttentionLoss(
        lambda_sparse=train_config.lambda_sparse,
        target_sparsity=train_config.target_sparsity,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay, betas=(0.9, 0.95),
    )

    def lr_lambda(step):
        if step < train_config.warmup_steps:
            return step / max(1, train_config.warmup_steps)
        progress = (step - train_config.warmup_steps) / max(
            1, train_config.max_steps - train_config.warmup_steps
        )
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    train_iter = iter(train_loader)
    started = time.time()
    task_sum = align_sum = 0.0
    soft_batches = 0

    for step in range(1, train_config.max_steps + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        ids = batch["input_ids"].to(device)
        inputs, labels = ids[:, :-1].contiguous(), ids[:, 1:].contiguous()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits, alignment, propagated_soft = ssa_style_forward(
            model, inputs, hard_probability=args.hard_probability, topk=args.topk
        )
        losses = criterion(logits, labels, model)
        if args.objective_clhr_weight is None:
            task_loss_for_backward = losses["loss_total"]
        else:
            task_loss_for_backward = objective_matched_task_loss(
                losses,
                propagated_soft,
                args.hard_probability,
                args.objective_clhr_weight,
            )
        loss = task_loss_for_backward + args.alignment_weight * alignment
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.max_grad_norm)
        optimizer.step()
        scheduler.step()

        task_sum += losses["loss_task"].item()
        align_sum += alignment.item()
        soft_batches += int(propagated_soft)
        if step % args.log_every == 0 or step == train_config.max_steps:
            count = args.log_every if step % args.log_every == 0 else step % args.log_every
            print(
                f"step {step:6d}/{train_config.max_steps}: "
                f"task={task_sum/count:.4f} align={align_sum/count:.6f} "
                f"soft_batches={soft_batches}/{count} elapsed={time.time()-started:.0f}s",
                flush=True,
            )
            task_sum = align_sum = 0.0
            soft_batches = 0

    training_elapsed_seconds = time.time() - started
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "step": train_config.max_steps,
            "alignment_weight": args.alignment_weight,
            "method": "ssa_style_learned_token_routing",
        },
        checkpoint_path,
    )

    native_nll = eval_native(model, val_loader, device, max_batches=args.eval_batches)
    hard_nll = eval_closed_loop_gate_hard(
        model, val_loader, device, k=args.topk, max_batches=args.eval_batches
    )
    random_nll = eval_random_hard(
        model, val_loader, device, k=args.topk,
        max_batches=args.eval_batches, n_draws=args.random_draws,
    )
    local_nll = eval_local_window(
        model, val_loader, device, k=args.topk, max_batches=args.eval_batches
    )
    result = {
        "method": "ssa_style_alignment_adapted_to_learned_token_routing",
        "protocol": "exact_upstream_31m",
        "seed": args.seed,
        "alignment_weight": args.alignment_weight,
        "stream_probability_soft": 1.0 - args.hard_probability,
        "stream_probability_hard": args.hard_probability,
        "objective_clhr_weight": args.objective_clhr_weight,
        "max_steps": train_config.max_steps,
        "native_nll": native_nll,
        "closed_loop_hard_nll": hard_nll,
        "closed_loop_excess_nll": hard_nll - native_nll,
        "random_hard_nll": random_nll,
        "local_window_nll": local_nll,
        "gate_utility": random_nll - hard_nll,
        "training_elapsed_seconds": training_elapsed_seconds,
        "elapsed_seconds": time.time() - started,
        "peak_cuda_memory_gb": (
            torch.cuda.max_memory_allocated(device) / 1024**3
            if device.type == "cuda" else None
        ),
        "checkpoint": str(checkpoint_path),
    }
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--alignment-weight", type=float, default=10.0)
    parser.add_argument("--hard-probability", type=float, default=0.5)
    parser.add_argument("--objective-clhr-weight", type=float)
    parser.add_argument("--topk", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=50000)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--eval-batches", type=int, default=200)
    parser.add_argument("--random-draws", type=int, default=5)
    parser.add_argument("--data-dir", default="./wikitext103_upstream")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
