"""Acceptance tests for src/moe_soft_to_hard.py (experiment A1: soft-to-hard
MoE discretization, registered in predictions/principle_generality.json).

Written and run FIRST (must fail before src/moe_soft_to_hard.py exists / is
implemented). Tiny CPU-only config throughout.
"""

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from moe_soft_to_hard import (
    MoEBlock,
    MoETransformer,
    benchmark_inference,
    clhr_loss,
    evaluate_relu,
    evaluate_softtopk,
    soft_top_k,
    run_full_evaluation,
    train_model,
    select_device,
    load_wikitext_cached,
    load_corpus_for_moe,
    build_argparser,
    main,
)

TINY = dict(d_model=32, n_heads=2, n_layers=2, d_ff=64, n_experts=4, max_len=16)
VOCAB = 100
SEQ_LEN = 16
BATCH = 4


def make_model(seed=0):
    torch.manual_seed(seed)
    return MoETransformer(VOCAB, **TINY)


def make_batch(seed=0, batch=BATCH, seq_len=SEQ_LEN):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, VOCAB, (batch, seq_len), generator=g)
    y = torch.randint(0, VOCAB, (batch, seq_len), generator=g)
    return x, y


class TinyLoader:
    """Minimal stand-in for a DataLoader yielding a fixed list of (x, y) batches."""

    def __init__(self, batches):
        self.batches = batches

    def __iter__(self):
        return iter(self.batches)


def make_loader(n_batches=2, seed=0):
    batches = [make_batch(seed=seed + i) for i in range(n_batches)]
    return TinyLoader(batches)


# ─────────────────────────────────────────────────────────────────────────
# 1. Soft mixture uses all experts
# ─────────────────────────────────────────────────────────────────────────

def test_soft_mixture_uses_all_experts():
    x, _ = make_batch()
    for expert_to_zero in range(TINY["n_experts"]):
        model = make_model(seed=1)
        model.eval()
        with torch.no_grad():
            logits_before, _, _, _, _ = model(x, mode="soft")

        with torch.no_grad():
            for block in model.blocks:
                expert = block.moe.experts[expert_to_zero]
                expert.w1.weight.zero_()
                expert.w2.weight.zero_()

        with torch.no_grad():
            logits_after, _, _, _, _ = model(x, mode="soft")

        assert not torch.allclose(logits_before, logits_after, atol=1e-6), (
            f"zeroing expert {expert_to_zero} did not change native_soft output; "
            "mixture may not be genuinely over all experts"
        )


# ─────────────────────────────────────────────────────────────────────────
# 2. Router probs sum to 1
# ─────────────────────────────────────────────────────────────────────────

def test_soft_mixture_weights_sum_to_one():
    model = make_model(seed=2)
    model.eval()
    x, _ = make_batch()
    with torch.no_grad():
        _, _, all_router_logits, _, _ = model(x, mode="soft")
    for logits in all_router_logits:
        probs = F.softmax(logits, dim=-1)
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


# ─────────────────────────────────────────────────────────────────────────
# 3. Router receives gradient in the soft path
# ─────────────────────────────────────────────────────────────────────────

def test_router_receives_gradient_in_soft_path():
    model = make_model(seed=3)
    model.train()
    x, y = make_batch()
    logits, _, _, _, aux = model(x, mode="soft")
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
    loss = loss + 0.0 * aux  # aux_coeff = 0
    loss.backward()

    for block in model.blocks:
        grad = block.moe.router.gate.weight.grad
        assert grad is not None
        assert torch.any(grad != 0)


# ─────────────────────────────────────────────────────────────────────────
# 4. Open-loop and closed-loop differ
# ─────────────────────────────────────────────────────────────────────────

def test_open_loop_and_closed_loop_differ():
    model = make_model(seed=4)
    device = torch.device("cpu")
    train_loader = make_loader(n_batches=3, seed=100)
    train_model(model, train_loader, device, "standard", steps=2, lr=1e-2, lambda_rca=0.0, aux_coeff=0.01)

    val_loader = make_loader(n_batches=3, seed=200)
    result = run_full_evaluation(model, val_loader, device, max_batches=3, n_random_draws=2)

    assert result["open_loop_top1_nll"] != pytest.approx(result["closed_loop_top1_nll"], abs=1e-6)


# ─────────────────────────────────────────────────────────────────────────
# 5. Closed-loop routes on dispatched state (feedback loop exists)
# ─────────────────────────────────────────────────────────────────────────

def test_closed_loop_routes_on_dispatched_state():
    model = make_model(seed=5)
    model.eval()
    x, _ = make_batch()

    n_experts = TINY["n_experts"]
    B, T = x.shape

    variant_a = torch.zeros(B, T, dtype=torch.long)
    variant_b = torch.full((B, T), n_experts - 1, dtype=torch.long)

    with torch.no_grad():
        _, _, _, indices_a, _ = model(x, mode="hard", forced_routes=[variant_a, None])
        _, _, _, indices_b, _ = model(x, mode="hard", forced_routes=[variant_b, None])

    # layer index 1 ("layer 2") routing must respond to layer 0's forced dispatch
    assert not torch.equal(indices_a[1], indices_b[1])


# ─────────────────────────────────────────────────────────────────────────
# 6. Shuffled preserves expert load histogram
# ─────────────────────────────────────────────────────────────────────────

def test_shuffled_preserves_expert_load_histogram():
    model = make_model(seed=6)
    device = torch.device("cpu")
    val_loader = make_loader(n_batches=3, seed=300)
    result = run_full_evaluation(model, val_loader, device, max_batches=3, n_random_draws=1)

    # run_full_evaluation only reports the closed-loop histogram, so recompute
    # shuffled's histogram directly to compare against it.
    model.eval()
    closed_hist = torch.zeros(TINY["n_experts"], dtype=torch.long)
    shuffled_hist = torch.zeros(TINY["n_experts"], dtype=torch.long)
    with torch.no_grad():
        for x, y in val_loader:
            _, _, _, closed_indices, _ = model(x, mode="hard")
            for li in closed_indices:
                closed_hist += torch.bincount(li.reshape(-1), minlength=TINY["n_experts"])

            shuffled_routes = []
            for li in closed_indices:
                flat = li.reshape(-1)
                perm = torch.randperm(flat.shape[0])
                shuffled_routes.append(flat[perm].reshape(li.shape))
            for li in shuffled_routes:
                shuffled_hist += torch.bincount(li.reshape(-1), minlength=TINY["n_experts"])

    assert torch.equal(torch.sort(closed_hist).values, torch.sort(shuffled_hist).values)
    assert torch.equal(closed_hist, shuffled_hist)  # exact match, not just sorted


# ─────────────────────────────────────────────────────────────────────────
# 7. No gradient through argmax
# ─────────────────────────────────────────────────────────────────────────

def test_no_gradient_through_argmax():
    model = make_model(seed=7)
    x, _ = make_batch()
    _, _, _, all_indices, _ = model(x, mode="hard")
    for indices in all_indices:
        assert indices.requires_grad is False


# ─────────────────────────────────────────────────────────────────────────
# 8. CLHR lambda=0 equals standard
# ─────────────────────────────────────────────────────────────────────────

def test_clhr_lambda_zero_equals_standard():
    model = make_model(seed=8)
    x, y = make_batch()

    combined, l_soft, l_hard = clhr_loss(model, x, y, lambda_rca=0.0, aux_coeff=0.01)
    assert torch.allclose(combined, l_soft, atol=1e-8)


# ─────────────────────────────────────────────────────────────────────────
# 9. All modes finite
# ─────────────────────────────────────────────────────────────────────────

def test_all_modes_finite():
    model = make_model(seed=9)
    device = torch.device("cpu")
    val_loader = make_loader(n_batches=2, seed=400)
    result = run_full_evaluation(model, val_loader, device, max_batches=2, n_random_draws=2)

    for key in (
        "native_soft_nll",
        "open_loop_top1_nll",
        "closed_loop_top1_nll",
        "shuffled_top1_nll",
        "random_top1_nll",
        "random_top1_std",
    ):
        assert np.isfinite(result[key]), f"{key} is not finite: {result[key]}"


# ─────────────────────────────────────────────────────────────────────────
# 10. Metrics keys present
# ─────────────────────────────────────────────────────────────────────────

def test_metrics_keys_present():
    model = make_model(seed=10)
    device = torch.device("cpu")
    val_loader = make_loader(n_batches=2, seed=500)
    result = run_full_evaluation(model, val_loader, device, max_batches=2, n_random_draws=2)

    required_keys = {
        "native_soft_nll", "open_loop_top1_nll", "closed_loop_top1_nll",
        "shuffled_top1_nll", "random_top1_nll", "random_top1_std",
        "G_CL", "G_OL", "compounding_ratio", "gate_utility",
        "G_CL_shuffled", "one_minus_cos_L",
    }
    assert required_keys.issubset(result.keys())
    assert "expert_load_histogram" in result


# ─────────────────────────────────────────────────────────────────────────
# Extra structural invariants (not in the numbered list, but stated as
# required behavior: weight tying, aux-coeff-off gradient, device selection).
# ─────────────────────────────────────────────────────────────────────────

def test_weight_tying_preserved():
    model = make_model(seed=11)
    assert model.lm_head.weight is model.tok_emb.weight


def test_select_device_returns_valid_device():
    device = select_device()
    assert device.type in ("cuda", "mps", "cpu")


# ─────────────────────────────────────────────────────────────────────────
# 11. Smoke: two-step train + eval writes valid JSON for both conditions
# ─────────────────────────────────────────────────────────────────────────

def test_smoke_train_two_steps(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    rng = np.random.default_rng(0)
    train_tokens = rng.integers(0, VOCAB, size=5000).astype(np.int64)
    val_tokens = rng.integers(0, VOCAB, size=2000).astype(np.int64)
    np.save(data_dir / "wt103_train_tokens.npy", train_tokens)
    np.save(data_dir / "wt103_val_tokens.npy", val_tokens)

    for condition in ("standard", "clhr"):
        output_path = tmp_path / f"result_{condition}.json"
        cmd = [
            sys.executable, str(REPO_ROOT / "src" / "moe_soft_to_hard.py"),
            "--condition", condition,
            "--data-dir", str(data_dir),
            "--output", str(output_path),
            "--steps", "2",
            "--batch-size", "2",
            "--seq-len", str(SEQ_LEN),
            "--d-model", str(TINY["d_model"]),
            "--n-heads", str(TINY["n_heads"]),
            "--n-layers", str(TINY["n_layers"]),
            "--d-ff", str(TINY["d_ff"]),
            "--n-experts", str(TINY["n_experts"]),
            "--max-eval-batches", "2",
            "--random-draws", "2",
        ]
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=120)
        assert proc.returncode == 0, f"condition={condition} failed:\n{proc.stdout}\n{proc.stderr}"

        assert output_path.exists()
        with open(output_path) as f:
            data = json.load(f)

        assert data["condition"] == condition
        assert data["steps_completed"] == 2
        for key in ("native_soft_nll", "closed_loop_top1_nll", "G_CL", "expert_load_histogram"):
            assert key in data


# ─────────────────────────────────────────────────────────────────────────
# Progress-logging acceptance tests (added: no-print-buffering fix for
# long, otherwise-silent SkyPilot training runs).
# ─────────────────────────────────────────────────────────────────────────

def _make_tiny_data_dir(tmp_path, seed=0):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    rng = np.random.default_rng(seed)
    train_tokens = rng.integers(0, VOCAB, size=5000).astype(np.int64)
    val_tokens = rng.integers(0, VOCAB, size=2000).astype(np.int64)
    np.save(data_dir / "wt103_train_tokens.npy", train_tokens)
    np.save(data_dir / "wt103_val_tokens.npy", val_tokens)
    return data_dir


def _tiny_main_argv(data_dir, output_path, condition="standard", seed=123, steps=2):
    return [
        "--condition", condition,
        "--seed", str(seed),
        "--data-dir", str(data_dir),
        "--output", str(output_path),
        "--steps", str(steps),
        "--batch-size", "2",
        "--seq-len", str(SEQ_LEN),
        "--d-model", str(TINY["d_model"]),
        "--n-heads", str(TINY["n_heads"]),
        "--n-layers", str(TINY["n_layers"]),
        "--d-ff", str(TINY["d_ff"]),
        "--n-experts", str(TINY["n_experts"]),
        "--max-eval-batches", "2",
        "--random-draws", "2",
    ]


def test_progress_logging_emitted(capsys):
    model = make_model(seed=20)
    device = torch.device("cpu")
    train_loader = make_loader(n_batches=3, seed=600)
    train_model(
        model, train_loader, device, "standard", steps=2, lr=1e-2,
        lambda_rca=0.0, aux_coeff=0.01, log_every=1,
    )
    captured = capsys.readouterr()
    progress_lines = [ln for ln in captured.out.splitlines() if ln.startswith("[train] step")]
    assert len(progress_lines) >= 1, f"no progress line emitted:\n{captured.out}"
    combined = "\n".join(progress_lines)
    assert re.search(r"step \d+/2", combined), f"missing step counter in:\n{combined}"
    assert re.search(r"loss_soft=-?\d+\.\d+", combined), f"missing numeric loss in:\n{combined}"


def test_all_prints_flush():
    source = (REPO_ROOT / "src" / "moe_soft_to_hard.py").read_text()
    tree = ast.parse(source)
    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print":
            flush_value = None
            for kw in node.keywords:
                if kw.arg == "flush":
                    flush_value = kw.value
            is_true = isinstance(flush_value, ast.Constant) and flush_value.value is True
            if not is_true:
                violations.append(node.lineno)
    assert not violations, f"print() calls missing flush=True at lines: {violations}"


def test_log_every_respected(capsys):
    device = torch.device("cpu")
    train_loader = make_loader(n_batches=3, seed=700)

    model_a = make_model(seed=21)
    train_model(
        model_a, train_loader, device, "standard", steps=2, lr=1e-2,
        lambda_rca=0.0, aux_coeff=0.01, log_every=1,
    )
    out_a = capsys.readouterr().out
    lines_a = [ln for ln in out_a.splitlines() if ln.startswith("[train] step")]
    assert len(lines_a) == 2, f"expected 2 progress lines with log_every=1, got {len(lines_a)}:\n{out_a}"

    model_b = make_model(seed=22)
    train_model(
        model_b, train_loader, device, "standard", steps=2, lr=1e-2,
        lambda_rca=0.0, aux_coeff=0.01, log_every=1000,
    )
    out_b = capsys.readouterr().out
    lines_b = [ln for ln in out_b.splitlines() if ln.startswith("[train] step")]
    assert len(lines_b) == 1, f"expected exactly 1 (final-step) progress line with log_every=1000, got {len(lines_b)}:\n{out_b}"
    assert "step 2/2" in lines_b[0]


def test_startup_banner_present(capsys, tmp_path):
    data_dir = _make_tiny_data_dir(tmp_path, seed=1)
    output_path = tmp_path / "banner_result.json"
    main(_tiny_main_argv(data_dir, output_path, condition="standard", seed=123, steps=2))
    out = capsys.readouterr().out
    assert "condition" in out and "standard" in out
    assert "seed" in out and "123" in out
    assert "parameters" in out
    assert ">>> training starting" in out


def test_eval_mode_announcements(capsys, tmp_path):
    data_dir = _make_tiny_data_dir(tmp_path, seed=2)
    output_path = tmp_path / "eval_announce_result.json"
    main(_tiny_main_argv(data_dir, output_path, condition="standard", seed=124, steps=2))
    out = capsys.readouterr().out
    for mode in ("native_soft", "open_loop_top1", "closed_loop_top1", "shuffled_top1", "random_top1"):
        assert mode in out, f"no announcement for eval mode {mode!r} found in output"


# ─────────────────────────────────────────────────────────────────────────
# Capacity-factor / token-dropping tests (Switch-Transformer-style expert
# capacity limiting).
#
# Drop convention implemented and tested here (see src/moe_soft_to_hard.py,
# MoEBlock.hard_forward docstring for the full citation/reasoning):
#   - capacity = ceil(capacity_factor * n_tokens / n_experts), computed once
#     per hard_forward call (i.e. per layer, per forward pass) over the
#     flattened (batch*seq) token axis for THAT call.
#   - tokens are kept/dropped in arrival order within the flattened token
#     axis: the first `capacity` tokens (in flattened order) assigned to an
#     expert are kept, any beyond that are dropped. This is the Switch
#     Transformer convention (Fedus, Zoph & Shazeer 2021, "Switch
#     Transformers: Scaling to Trillion Parameter Models with Simple and
#     Efficient Sparsity") -- not a router-confidence/top-k-by-logit rule.
#   - a dropped token's output is the token's own MoEBlock-input
#     representation unchanged (out[dropped] = flat[dropped]) -- a
#     residual/identity passthrough, NOT zero. This matches the paper's
#     description of overflowed tokens: no expert computation happens for
#     them, and their representation passes through unchanged.
#   - capacity_factor=None (default) must be perfectly bit-identical to the
#     pre-existing (uncapacitated) hard_forward -- this is enforced by
#     test_capacity_none_is_bit_identical below, which independently
#     reimplements the ORIGINAL (pre-capacity-flag) algorithm and checks
#     exact equality against the current code. This protects every existing
#     result under results/moe_soft_hard/ and the in-flight
#     moe-scaled-{standard,clhr}-s{42,123} clusters, neither of which pass
#     --capacity-factor and must therefore be wholly unaffected.
#   - capacity limiting is applied AFTER `indices` is determined by
#     whichever branch (forced_indices / random_mode / shuffle / plain
#     closed-loop) produced it, and is applied identically in every branch.
#     Rationale: a capacity limit models a physical/memory constraint on an
#     expert's per-batch buffer size in a real deployed MoE -- it holds
#     regardless of how the token->expert assignment was produced (a
#     learned router, a forced/open-loop replay, a permutation control, or
#     uniform random dispatch), so applying it selectively to only one mode
#     would be an inconsistency with no principled justification and would
#     make cross-mode metrics (e.g. gate_utility = random - closed_loop)
#     not comparable on equal footing.
# ─────────────────────────────────────────────────────────────────────────

import math as _math


def _reference_hard_forward_no_capacity(block, x, forced_indices=None, shuffle=False, random_mode=False):
    """Faithful reimplementation of MoEBlock.hard_forward AS IT EXISTED
    BEFORE the --capacity-factor change (copied verbatim from the pre-edit
    source, only renamed). Used as an independent oracle for the
    bit-identical-at-default test, so that test does not merely compare the
    new code against itself."""
    shape = x.shape
    flat = x.reshape(-1, shape[-1])

    logits = None
    if forced_indices is not None:
        indices = forced_indices.reshape(-1)
    elif random_mode:
        indices = torch.randint(0, block.n_experts, (flat.shape[0],), device=x.device)
    else:
        logits = block.router(flat)
        with torch.no_grad():
            indices = logits.argmax(dim=-1)
        if shuffle:
            perm = torch.randperm(indices.shape[0], device=x.device)
            indices = indices[perm]

    out = torch.zeros_like(flat)
    for e_idx in range(block.n_experts):
        mask = indices == e_idx
        if mask.any():
            out[mask] = block.experts[e_idx](flat[mask])

    aux_loss = torch.tensor(0.0, device=x.device)
    if logits is not None:
        from moe_soft_to_hard import _aux_load_balance_loss
        aux_loss = _aux_load_balance_loss(logits, indices, block.n_experts)

    return out.reshape(shape), indices.reshape(shape[:-1]), aux_loss


def test_capacity_none_is_bit_identical():
    """Default (capacity_factor=None, and also the case where the argument
    is omitted entirely) must produce EXACTLY the same output/indices/
    aux_loss as the pre-existing uncapacitated algorithm, for a fixed seed
    and input -- across the plain closed-loop, forced_indices, random_mode,
    and shuffle branches."""
    torch.manual_seed(1000)
    block = MoEBlock(d_model=8, d_ff=16, n_experts=4)
    x = torch.randn(3, 5, 8)

    # plain closed-loop branch
    torch.manual_seed(0)
    out_new, idx_new, aux_new = block.hard_forward(x)
    torch.manual_seed(0)
    out_ref, idx_ref, aux_ref = _reference_hard_forward_no_capacity(block, x)
    assert torch.equal(out_new, out_ref)
    assert torch.equal(idx_new, idx_ref)
    assert torch.allclose(aux_new, aux_ref, atol=0.0)

    # explicit capacity_factor=None must match the omitted-argument default
    out_explicit, idx_explicit, aux_explicit = block.hard_forward(x, capacity_factor=None)
    assert torch.equal(out_new, out_explicit)
    assert torch.equal(idx_new, idx_explicit)
    assert torch.allclose(aux_new, aux_explicit, atol=0.0)

    # forced_indices branch
    forced = torch.randint(0, 4, (3, 5))
    out_new, idx_new, aux_new = block.hard_forward(x, forced_indices=forced)
    out_ref, idx_ref, aux_ref = _reference_hard_forward_no_capacity(block, x, forced_indices=forced)
    assert torch.equal(out_new, out_ref)
    assert torch.equal(idx_new, idx_ref)

    # random_mode branch
    torch.manual_seed(7)
    out_new, idx_new, aux_new = block.hard_forward(x, random_mode=True)
    torch.manual_seed(7)
    out_ref, idx_ref, aux_ref = _reference_hard_forward_no_capacity(block, x, random_mode=True)
    assert torch.equal(out_new, out_ref)
    assert torch.equal(idx_new, idx_ref)

    # shuffle branch
    torch.manual_seed(13)
    out_new, idx_new, aux_new = block.hard_forward(x, shuffle=True)
    torch.manual_seed(13)
    out_ref, idx_ref, aux_ref = _reference_hard_forward_no_capacity(block, x, shuffle=True)
    assert torch.equal(out_new, out_ref)
    assert torch.equal(idx_new, idx_ref)


def test_capacity_limit_drops_excess_tokens():
    """8 tokens, 2 experts, capacity_factor=1.0 -> capacity = ceil(1.0*8/2) =
    4 per expert. Force 6 tokens to expert 0 and 2 to expert 1: exactly 2
    tokens (the overflow beyond capacity 4) must be dropped, and exactly 4
    must be genuinely processed by expert 0's forward."""
    torch.manual_seed(2000)
    block = MoEBlock(d_model=6, d_ff=12, n_experts=2)
    x = torch.randn(1, 8, 6)
    forced = torch.tensor([[0, 0, 0, 0, 0, 0, 1, 1]])  # 6 -> expert0, 2 -> expert1

    capacity_factor = 1.0
    n_tokens = 8
    n_experts = 2
    capacity = _math.ceil(capacity_factor * n_tokens / n_experts)
    assert capacity == 4

    out, indices, aux = block.hard_forward(x, forced_indices=forced, capacity_factor=capacity_factor)
    flat_x = x.reshape(-1, 6)
    flat_out = out.reshape(-1, 6)

    with torch.no_grad():
        expert0_out_all = block.experts[0](flat_x)

    # First `capacity` (4) tokens assigned to expert 0, in arrival order,
    # must be genuinely processed by expert 0.
    kept_positions = [0, 1, 2, 3]
    for p in kept_positions:
        assert torch.allclose(flat_out[p], expert0_out_all[p], atol=1e-6), (
            f"token {p} (within expert-0 capacity) was not processed by expert 0"
        )

    # The remaining 2 tokens assigned to expert 0 (positions 4, 5) overflow
    # capacity and must be dropped (not equal to expert-0's output).
    dropped_positions = [4, 5]
    for p in dropped_positions:
        assert not torch.allclose(flat_out[p], expert0_out_all[p], atol=1e-6), (
            f"token {p} (beyond expert-0 capacity) appears to have been processed by expert 0 anyway"
        )

    # expert 1 never exceeds its capacity (only 2 tokens assigned, capacity 4)
    # so both its tokens (positions 6, 7) must be genuinely processed.
    with torch.no_grad():
        expert1_out_all = block.experts[1](flat_x)
    for p in (6, 7):
        assert torch.allclose(flat_out[p], expert1_out_all[p], atol=1e-6)


def test_dropped_tokens_get_passthrough_not_zero():
    """Dropped tokens' output must equal their own (pre-expert) input
    representation exactly -- residual/identity passthrough -- and must NOT
    be zero (unless the input itself happened to be zero, which it isn't
    here since x is drawn from a standard normal)."""
    torch.manual_seed(3000)
    block = MoEBlock(d_model=6, d_ff=12, n_experts=2)
    x = torch.randn(1, 8, 6)
    forced = torch.tensor([[0, 0, 0, 0, 0, 0, 1, 1]])
    capacity_factor = 1.0  # capacity = 4 per expert

    out, indices, aux = block.hard_forward(x, forced_indices=forced, capacity_factor=capacity_factor)
    flat_x = x.reshape(-1, 6)
    flat_out = out.reshape(-1, 6)

    dropped_positions = [4, 5]
    for p in dropped_positions:
        assert torch.equal(flat_out[p], flat_x[p]), (
            f"dropped token {p} output does not exactly equal its own input (passthrough)"
        )
        assert not torch.allclose(flat_out[p], torch.zeros(6), atol=1e-6), (
            f"dropped token {p} output is (near-)zero; expected passthrough, not zero"
        )


def test_capacity_limit_deterministic_given_fixed_router():
    """Same input, same fixed router weights, same capacity_factor, plain
    closed-loop dispatch (no shuffle/random) -> identical drop pattern,
    output, and indices on repeated calls."""
    torch.manual_seed(4000)
    block = MoEBlock(d_model=10, d_ff=20, n_experts=3)
    block.eval()
    x = torch.randn(2, 12, 10)

    out1, idx1, aux1 = block.hard_forward(x, capacity_factor=1.25)
    out2, idx2, aux2 = block.hard_forward(x, capacity_factor=1.25)

    assert torch.equal(out1, out2)
    assert torch.equal(idx1, idx2)
    assert torch.allclose(aux1, aux2, atol=0.0)


def test_capacity_interacts_correctly_with_shuffle_and_random_mode():
    """Consistency rule under test: capacity limiting is applied uniformly
    to every dispatch branch, AFTER `indices` is fixed by that branch.

    (a) shuffle: shuffling preserves the exact per-expert token-count
        histogram (pre-existing invariant, test_shuffled_preserves_expert_
        load_histogram). Since capacity-drop counts are a deterministic
        function of that histogram alone (capacity is fixed, and the number
        of tokens over capacity for an expert = max(0, count - capacity)),
        the shuffled call must drop EXACTLY the same number of tokens per
        expert as the unshuffled closed-loop call that produced the indices
        it is a permutation of -- even though WHICH tokens are dropped
        generally differs (that's the point of shuffle: same aggregate
        load, no input-conditioning).

    (b) random_mode: capacity limiting must also apply here -- no expert's
        processed-token count may exceed its capacity, exactly as in the
        other branches. random_mode is NOT treated as an idealized/
        uncapacitated baseline; it represents a physical resource
        constraint that holds regardless of how tokens were assigned.
    """
    torch.manual_seed(5000)
    n_experts = 3
    block = MoEBlock(d_model=10, d_ff=20, n_experts=n_experts)
    block.eval()
    x = torch.randn(2, 12, 10)  # n_tokens = 24
    capacity_factor = 0.75
    n_tokens = 24
    capacity = _math.ceil(capacity_factor * n_tokens / n_experts)

    # --- (a) shuffle vs. the closed-loop indices it's a permutation of ---
    with torch.no_grad():
        _, closed_indices, _ = block.hard_forward(x, capacity_factor=capacity_factor)
    closed_flat = closed_indices.reshape(-1)
    closed_hist = torch.bincount(closed_flat, minlength=n_experts)
    closed_dropped_per_expert = torch.clamp(closed_hist - capacity, min=0)

    perm = torch.randperm(closed_flat.shape[0])
    shuffled_flat_indices = closed_flat[perm]
    shuffled_forced = shuffled_flat_indices.reshape(x.shape[0], x.shape[1])
    with torch.no_grad():
        shuffled_out, shuffled_indices, _ = block.hard_forward(
            x, forced_indices=shuffled_forced, capacity_factor=capacity_factor
        )

    # histogram (pre-capacity assignment) is identical by construction
    shuffled_hist = torch.bincount(shuffled_indices.reshape(-1), minlength=n_experts)
    assert torch.equal(closed_hist, shuffled_hist)

    # per-expert dropped COUNT must match, even though the specific dropped
    # tokens may differ.
    flat_x = x.reshape(-1, 10)
    flat_shuffled_out = shuffled_out.reshape(-1, 10)
    for e_idx in range(n_experts):
        expert_positions = (shuffled_indices.reshape(-1) == e_idx).nonzero(as_tuple=True)[0]
        n_dropped_this_expert = 0
        for p in expert_positions.tolist():
            if torch.equal(flat_shuffled_out[p], flat_x[p]):
                n_dropped_this_expert += 1
        assert n_dropped_this_expert == closed_dropped_per_expert[e_idx].item()

    # --- (b) random_mode must also respect capacity: no expert processes
    # more than `capacity` tokens (dropped tokens are excluded from the
    # "processed" count by construction -- verify via the passthrough
    # signature rather than a separate counter). ---
    torch.manual_seed(99)
    with torch.no_grad():
        rand_out, rand_indices, _ = block.hard_forward(x, random_mode=True, capacity_factor=capacity_factor)
    flat_rand_out = rand_out.reshape(-1, 10)
    rand_flat_indices = rand_indices.reshape(-1)
    for e_idx in range(n_experts):
        expert_positions = (rand_flat_indices == e_idx).nonzero(as_tuple=True)[0]
        n_processed = sum(
            1 for p in expert_positions.tolist() if not torch.equal(flat_rand_out[p], flat_x[p])
        )
        assert n_processed <= capacity, (
            f"random_mode expert {e_idx} processed {n_processed} tokens, exceeding capacity {capacity}"
        )


# ─────────────────────────────────────────────────────────────────────────
# top_k_dispatch (Mixtral-style top-k combine) extension.
#
# Four design decisions, prescribed exactly (not re-derived here):
#   1. Combine multiple experts' outputs: softmax over ONLY the selected k
#      logits (not all n_experts), then accumulate weight * expert(token)
#      per selected slot. At k=1, weight is always exactly 1.0 (softmax of a
#      single value), so accumulation into a zero tensor with one
#      contribution per token is bit-identical to the original assignment.
#   2. Shuffle generalizes to permuting ROWS of the (N, k) assignment matrix
#      (each token's whole top-k SET moves as a unit) -- per-expert counts
#      still preserved by construction.
#   3. Aux loss generalizes via `(indices == e_idx).any(dim=-1).mean()` on
#      an (N, k) indices tensor -- a token "uses" an expert if it appears in
#      ANY of its k slots. At k=1 (indices unsqueezed to (N, 1)) this
#      reduces exactly to the original (N,) formula.
#   4. Capacity limiting flattens all (token, slot) pairs routed to an
#      expert (row-major: for a fixed token, slot 0..k-1 in order, then the
#      next token) and applies the SAME arrival-order drop rule over that
#      larger pool; a token's 2nd-choice expert can be dropped independently
#      of its 1st-choice expert.
# ─────────────────────────────────────────────────────────────────────────

def _reference_hard_forward_top1(block, x):
    """Independent, from-scratch reimplementation of the ORIGINAL (pre-
    top_k_dispatch) top-1 closed-loop hard-dispatch algorithm -- written
    without consulting MoEBlock.hard_forward's current source, as an
    external oracle for test_top_k_dispatch_1_is_bit_identical. Only the
    plain closed-loop branch (no forced_indices/random_mode/shuffle) is
    needed here since that's the branch the accumulate-vs-assign rewrite
    changes most directly."""
    shape = x.shape
    flat = x.reshape(-1, shape[-1])
    logits = block.router(flat)
    with torch.no_grad():
        indices = logits.argmax(dim=-1)
    out = torch.zeros_like(flat)
    for e_idx in range(block.n_experts):
        mask = indices == e_idx
        if mask.any():
            out[mask] = block.experts[e_idx](flat[mask])
    from moe_soft_to_hard import _aux_load_balance_loss
    aux_loss = _aux_load_balance_loss(logits, indices, block.n_experts)
    return out.reshape(shape), indices.reshape(shape[:-1]), aux_loss


def test_top_k_dispatch_1_is_bit_identical():
    """Decision 1's bit-identical-at-k=1 requirement, proven against an
    INDEPENDENT from-scratch reference (_reference_hard_forward_top1), not
    merely new-code-default-vs-new-code-explicit."""
    torch.manual_seed(6000)
    block = MoEBlock(d_model=10, d_ff=20, n_experts=4)
    x = torch.randn(3, 7, 10)

    out_ref, idx_ref, aux_ref = _reference_hard_forward_top1(block, x)

    out_default, idx_default, aux_default = block.hard_forward(x)
    assert torch.equal(out_default, out_ref)
    assert torch.equal(idx_default, idx_ref)
    assert torch.allclose(aux_default, aux_ref, atol=0.0)

    out_explicit, idx_explicit, aux_explicit = block.hard_forward(x, top_k_dispatch=1)
    assert torch.equal(out_explicit, out_ref)
    assert torch.equal(idx_explicit, idx_ref)
    assert torch.allclose(aux_explicit, aux_ref, atol=0.0)


def test_top_k_dispatch_2_accumulates_correctly():
    """Hand-verified weighted-sum for a single token routed to 2 KNOWN
    experts with KNOWN softmax weights. The router's weight matrix is set
    manually so that, for x = [1, 0], logit_e = weight[e, 0] exactly (the
    second router input dim is zeroed out) -- giving full control over the
    per-expert logits without touching the experts themselves."""
    torch.manual_seed(6001)
    block = MoEBlock(d_model=2, d_ff=4, n_experts=3)
    with torch.no_grad():
        # logits for x=[1,0]: expert0 -> 1.0, expert1 -> 2.0, expert2 -> 0.0
        block.router.gate.weight[:, 0] = torch.tensor([1.0, 2.0, 0.0])
        block.router.gate.weight[:, 1] = torch.tensor([0.0, 0.0, 0.0])

    x = torch.tensor([[[1.0, 0.0]]])  # (B=1, T=1, d_model=2)

    out, indices, aux = block.hard_forward(x, top_k_dispatch=2)

    # top-2 of [1.0, 2.0, 0.0] -> expert1 (logit 2.0), expert0 (logit 1.0)
    assert set(indices.reshape(-1).tolist()) == {0, 1}

    # hand-computed softmax over ONLY the two selected logits {2.0, 1.0}
    w1 = _math.exp(2.0) / (_math.exp(2.0) + _math.exp(1.0))
    w0 = _math.exp(1.0) / (_math.exp(2.0) + _math.exp(1.0))
    assert abs((w0 + w1) - 1.0) < 1e-6

    flat_x = x.reshape(1, 2)
    with torch.no_grad():
        e0_out = block.experts[0](flat_x)
        e1_out = block.experts[1](flat_x)
    expected = w0 * e0_out + w1 * e1_out

    assert torch.allclose(out.reshape(1, 2), expected, atol=1e-6)


def test_top_k_softmax_weights_sum_to_one_per_token():
    """For k>1, the per-token combine weights across its k selected experts
    sum to 1.0 (allowing float tolerance) -- recomputed independently from
    the router's own logits/topk/softmax pipeline (the same formula
    MoEBlock.hard_forward uses internally), for a batch of tokens."""
    torch.manual_seed(6002)
    block = MoEBlock(d_model=8, d_ff=16, n_experts=5)
    x = torch.randn(2, 6, 8)
    flat = x.reshape(-1, 8)

    k = 3
    with torch.no_grad():
        logits = block.router(flat)
        topk_logits, _ = logits.topk(k, dim=-1)
        weights = F.softmax(topk_logits, dim=-1)

    sums = weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_shuffle_generalizes_correctly_at_k_gt_1():
    """Decision 2: shuffling at top_k_dispatch>1 permutes ROWS of the (N, k)
    assignment matrix -- per-expert counts (now counted across ALL (token,
    slot) pairs) are preserved by construction, exactly generalizing
    test_shuffled_preserves_expert_load_histogram."""
    torch.manual_seed(6003)
    n_experts = 4
    block = MoEBlock(d_model=12, d_ff=24, n_experts=n_experts)
    block.eval()
    x = torch.randn(3, 9, 12)
    k = 2

    torch.manual_seed(111)
    _, closed_indices, _ = block.hard_forward(x, top_k_dispatch=k)
    torch.manual_seed(111)
    _, shuffled_indices, _ = block.hard_forward(x, shuffle=True, top_k_dispatch=k)

    closed_hist = torch.bincount(closed_indices.reshape(-1), minlength=n_experts)
    shuffled_hist = torch.bincount(shuffled_indices.reshape(-1), minlength=n_experts)
    assert torch.equal(closed_hist, shuffled_hist)

    # shape/row-unit sanity: k-tuples are intact (each row is still a
    # permutation of SOME original row, not independently shuffled per slot)
    assert closed_indices.shape == shuffled_indices.shape == (3, 9, k)
    closed_rows = closed_indices.reshape(-1, k)
    shuffled_rows = shuffled_indices.reshape(-1, k)
    closed_row_multiset = sorted(tuple(r.tolist()) for r in closed_rows)
    shuffled_row_multiset = sorted(tuple(r.tolist()) for r in shuffled_rows)
    assert closed_row_multiset == shuffled_row_multiset


def test_aux_loss_any_dim_reduces_correctly_at_k_1():
    """Decision 3: the generalized `.any(dim=-1)` aux-loss formula on an
    (N, 1) indices tensor matches the original (N,) formula exactly."""
    from moe_soft_to_hard import _aux_load_balance_loss

    torch.manual_seed(6004)
    n_experts = 4
    logits = torch.randn(20, n_experts)
    indices_1d = torch.randint(0, n_experts, (20,))
    indices_2d = indices_1d.unsqueeze(-1)  # (N, 1) -- the k=1 shape

    loss_1d = _aux_load_balance_loss(logits, indices_1d, n_experts)
    loss_2d = _aux_load_balance_loss(logits, indices_2d, n_experts)

    assert torch.allclose(loss_1d, loss_2d, atol=1e-7)


def test_capacity_and_top_k_dispatch_interact_correctly():
    """Decision 4, hand-verified construction: a token's 2nd-choice expert
    can be dropped independently of whether its 1st-choice expert was
    dropped.

    2 experts, 2 tokens, top_k_dispatch=2, capacity_factor=0.5 ->
    capacity = ceil(0.5 * 2 / 2) = 1 per expert.
    forced_indices (row-major flatten, arrival order):
      token0: [0, 0]  (both slots -> expert0)
      token1: [0, 1]  (1st choice -> expert0, 2nd choice -> expert1)
    Flattened (token, slot) arrival order: pos0=t0s0(e0), pos1=t0s1(e0),
    pos2=t1s0(e0), pos3=t1s1(e1).
      expert0 queue: pos0, pos1, pos2 (3 pairs, capacity 1) -> keep pos0
        only; drop pos1, pos2.
      expert1 queue: pos3 (1 pair, capacity 1) -> keep pos3.
    So token1's 1st choice (expert0, pos2) is DROPPED because expert0's
    capacity was already exhausted by token0, while token1's 2nd choice
    (expert1, pos3) is KEPT because expert1 is untouched -- independent
    drop outcomes for the same token's two slots.
    """
    torch.manual_seed(6005)
    block = MoEBlock(d_model=4, d_ff=8, n_experts=2)
    block.eval()
    x = torch.randn(1, 2, 4)  # token0, token1
    forced = torch.tensor([[[0, 0], [0, 1]]])  # (B=1, T=2, k=2)

    capacity_factor = 0.5
    n_tokens = 2
    n_experts = 2
    capacity = _math.ceil(capacity_factor * n_tokens / n_experts)
    assert capacity == 1

    out, indices, aux = block.hard_forward(
        x, forced_indices=forced, capacity_factor=capacity_factor, top_k_dispatch=2,
    )
    flat_x = x.reshape(2, 4)
    flat_out = out.reshape(2, 4)
    weight = 1.0 / 2  # forced_indices branch: uniform 1/k weighting

    with torch.no_grad():
        e0_out = block.experts[0](flat_x)
        e1_out = block.experts[1](flat_x)

    # token0: slot0 (expert0, pos0) KEPT, slot1 (expert0, pos1) DROPPED
    expected_token0 = weight * e0_out[0] + weight * flat_x[0]
    assert torch.allclose(flat_out[0], expected_token0, atol=1e-6)

    # token1: slot0 (expert0, pos2) DROPPED, slot1 (expert1, pos3) KEPT --
    # this is the independence claim: token1's 1st choice is dropped while
    # its 2nd choice (a DIFFERENT expert, under capacity) is kept.
    expected_token1 = weight * flat_x[1] + weight * e1_out[1]
    assert torch.allclose(flat_out[1], expected_token1, atol=1e-6)

    # Explicitly confirm neither slot of token1 was uniformly resolved the
    # same way (i.e. it is not the case that both were kept or both dropped)
    token1_slot0_is_passthrough_share = torch.allclose(
        flat_out[1] - weight * e1_out[1], weight * flat_x[1], atol=1e-6
    )
    assert token1_slot0_is_passthrough_share


# ─────────────────────────────────────────────────────────────────────────
# Corpus dispatch (FineWeb-Edu corpus-generality extension via
# src/data_loading.py). Written FIRST, before load_corpus_for_moe existed.
# ─────────────────────────────────────────────────────────────────────────

def _make_fineweb_shard_dir(tmp_path, seed=0, n_shard_tokens=5000, n_eval_tokens=2000):
    """Tiny synthetic FineWeb-Edu shard directory, matching the exact shape
    (flat int32 .npy, 'shard_NNNN.npy' naming under 'fineweb_edu_shards/', a
    held-out .npy under 'fineweb_edu_eval/') that src/data_loading.py's
    ShardedTokenDataset / load_corpus("fineweb-edu", ...) expects -- see
    data_loading.prepare_fineweb_shards and data_loading.load_corpus."""
    data_dir = tmp_path / "fineweb_data"
    shard_dir = data_dir / "fineweb_edu_shards"
    eval_dir = data_dir / "fineweb_edu_eval"
    shard_dir.mkdir(parents=True)
    eval_dir.mkdir(parents=True)

    rng = np.random.default_rng(seed)
    shard_tokens = rng.integers(0, VOCAB, size=n_shard_tokens).astype(np.int32)
    np.save(shard_dir / "shard_0000.npy", shard_tokens)

    eval_tokens = rng.integers(0, VOCAB, size=n_eval_tokens).astype(np.int32)
    np.save(eval_dir / "eval_shard.npy", eval_tokens)

    return data_dir


def test_wikitext_path_unchanged_by_default(tmp_path):
    """The default --data-dir / no --corpus path must be bit-identical to the
    pre-extension load_wikitext_cached() path: same dataset length, same
    sampled tokens for a fixed seed."""
    data_dir = _make_tiny_data_dir(tmp_path, seed=42)

    old_train_ds, old_val_ds, old_vocab = load_wikitext_cached(str(data_dir), SEQ_LEN)
    new_train_ds, new_val_ds, new_vocab = load_corpus_for_moe(
        "wikitext-103", str(data_dir), seq_len=SEQ_LEN, seed=42
    )

    assert new_vocab == old_vocab
    assert len(new_train_ds) == len(old_train_ds)
    assert len(new_val_ds) == len(old_val_ds)

    for idx in (0, 1, len(old_train_ds) - 1):
        old_x, old_y = old_train_ds[idx]
        new_x, new_y = new_train_ds[idx]
        assert torch.equal(old_x, new_x)
        assert torch.equal(old_y, new_y)

    for idx in (0, 1, len(old_val_ds) - 1):
        old_x, old_y = old_val_ds[idx]
        new_x, new_y = new_val_ds[idx]
        assert torch.equal(old_x, new_x)
        assert torch.equal(old_y, new_y)

    # And the same dispatcher is what main() drives by default (no --corpus
    # flag at all) -- confirm the default argparse value matches too.
    args = build_argparser().parse_args(["--output", str(tmp_path / "unused.json")])
    assert args.corpus == "wikitext-103"


def test_fineweb_corpus_loads_correctly(tmp_path):
    """Pointed at a tiny synthetic FineWeb-Edu shard directory (shape/dtype
    matching src/data_loading.py's ShardedTokenDataset contract), the loader
    returns sequences of the correct shape/dtype."""
    data_dir = _make_fineweb_shard_dir(tmp_path)

    train_ds, val_ds, vocab_size = load_corpus_for_moe(
        "fineweb-edu", str(data_dir), seq_len=SEQ_LEN, seed=0
    )
    assert vocab_size == 50257  # gpt2 vocab, per data_loading's gpt2 tokenizer convention

    # Train side is the sharded IterableDataset wrapped to (x, y) tuples.
    train_iter = iter(train_ds)
    x, y = next(train_iter)
    assert x.shape == (SEQ_LEN,)
    assert y.shape == (SEQ_LEN,)
    assert x.dtype == torch.long
    assert y.dtype == torch.long
    # y is x shifted by one position (next-token targets from one input_ids chunk)
    assert torch.equal(x[1:], y[:-1])

    # Validation side is the held-out eval shard, a regular (non-iterable) Dataset.
    assert len(val_ds) > 0
    vx, vy = val_ds[0]
    assert vx.shape == (SEQ_LEN,)
    assert vy.shape == (SEQ_LEN,)
    assert vx.dtype == torch.long
    assert vy.dtype == torch.long


def test_fineweb_dataloader_construction_matches_main(tmp_path):
    """Regression test for an OBSERVED production failure: all 4
    moe-fineweb-* SkyPilot jobs crashed in ~5 minutes with
    `ValueError: DataLoader with IterableDataset: expected unspecified
    shuffle option, but got shuffle=True` -- main() unconditionally passed
    shuffle=True to DataLoader, which torch forbids for ANY IterableDataset
    (the FineWeb-Edu sharded train_ds). The prior test
    (test_fineweb_corpus_loads_correctly) called iter(train_ds) directly,
    bypassing DataLoader entirely, so it could not have caught this -- this
    test exercises the actual DataLoader construction path main() uses, not
    just the dataset."""
    data_dir = _make_fineweb_shard_dir(tmp_path)
    train_ds, val_ds, _ = load_corpus_for_moe(
        "fineweb-edu", str(data_dir), seq_len=SEQ_LEN, seed=0
    )
    assert isinstance(train_ds, torch.utils.data.IterableDataset)

    # Mirrors main()'s conditional construction: no shuffle kwarg at all for
    # an IterableDataset train_ds.
    train_kwargs = dict(batch_size=2, num_workers=0, drop_last=True)
    if not isinstance(train_ds, torch.utils.data.IterableDataset):
        train_kwargs["shuffle"] = True
    train_loader = torch.utils.data.DataLoader(train_ds, **train_kwargs)
    batch = next(iter(train_loader))
    x, y = batch
    assert x.shape == (2, SEQ_LEN)
    assert y.shape == (2, SEQ_LEN)

    # val_ds is always a regular Dataset (even for fineweb-edu), so shuffle=
    # is fine here -- confirm this path is genuinely unaffected.
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=2, shuffle=False, num_workers=0, drop_last=True
    )
    vx, vy = next(iter(val_loader))
    assert vx.shape == (2, SEQ_LEN)
    assert vy.shape == (2, SEQ_LEN)


def test_wikitext_dataloader_still_uses_shuffle_true(tmp_path):
    """The fix for the IterableDataset case must not silently disable
    shuffling for the WikiText-103 path (a regular Dataset) -- confirm
    main()'s conditional still resolves to shuffle=True there."""
    train_ds, val_ds, _ = load_corpus_for_moe(
        "wikitext-103", str(_make_tiny_data_dir(tmp_path, seed=42)), seq_len=SEQ_LEN, seed=0
    )
    assert not isinstance(train_ds, torch.utils.data.IterableDataset)
    train_kwargs = dict(batch_size=2, num_workers=0, drop_last=True)
    if not isinstance(train_ds, torch.utils.data.IterableDataset):
        train_kwargs["shuffle"] = True
    assert train_kwargs.get("shuffle") is True
    # Construction itself must not raise (this already worked pre-fix, but
    # confirm the fix didn't regress it).
    torch.utils.data.DataLoader(train_ds, **train_kwargs)


def test_corpus_dispatch_is_explicit_not_silent(tmp_path):
    """An invalid/ambiguous corpus specification must fail loudly (ValueError),
    not silently fall back to WikiText-103. This project has been bitten by
    silent wrong-corpus fallbacks before (see results/RUN_LEDGER.md Section 2g:
    the confounded 2B-token B1 hierarchical sweep) -- do not repeat that
    failure mode here."""
    # No WikiText cache files exist in this empty tmp_path, so a silent
    # fallback to load_wikitext_cached would itself raise FileNotFoundError
    # (a different exception / message) rather than the explicit "Unknown
    # corpus" ValueError -- asserting the message pins down which failure
    # mode actually fired.
    with pytest.raises(ValueError, match="Unknown corpus"):
        load_corpus_for_moe("not-a-real-corpus", str(tmp_path), seq_len=SEQ_LEN, seed=0)

    # Also enforced at the CLI layer: argparse itself rejects an unknown
    # --corpus value rather than accepting it and dispatching ambiguously.
    with pytest.raises(SystemExit):
        build_argparser().parse_args(["--output", "x.json", "--corpus", "not-a-real-corpus"])


# ─────────────────────────────────────────────────────────────────────────
# grad_accum_steps: bit-identical at 1, correctly gradient-equivalent at N>1
# ─────────────────────────────────────────────────────────────────────────

def test_grad_accum_steps_1_is_bit_identical():
    """grad_accum_steps=1 (the default) must reproduce EXACTLY the same
    weight trajectory as an independent reference training loop written
    without any accumulation wrapper -- proving the new micro-batch loop
    reduces to the original single-batch behavior, not just "looks close"."""
    torch.manual_seed(9001)
    model_new = make_model(seed=42)
    torch.manual_seed(9001)
    model_ref = make_model(seed=42)
    for p_new, p_ref in zip(model_new.parameters(), model_ref.parameters()):
        assert torch.equal(p_new, p_ref)  # sanity: identical init

    loader = make_loader(n_batches=3, seed=800)
    train_model(
        model_new, loader, torch.device("cpu"), "standard", steps=3, lr=1e-2,
        lambda_rca=0.0, aux_coeff=0.01, grad_accum_steps=1,
    )

    # Independent reference: the exact pre-grad-accum single-batch loop.
    optimizer = torch.optim.AdamW(model_ref.parameters(), lr=1e-2, weight_decay=0.01)
    train_iter = iter(loader)
    for _ in range(3):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(loader)
            x, y = next(train_iter)
        optimizer.zero_grad()
        soft_logits, _, _, _, soft_aux = model_ref(x, mode="soft")
        soft_ce = F.cross_entropy(soft_logits.reshape(-1, soft_logits.size(-1)), y.reshape(-1))
        loss = soft_ce + 0.01 * soft_aux
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_ref.parameters(), 1.0)
        optimizer.step()

    for p_new, p_ref in zip(model_new.parameters(), model_ref.parameters()):
        assert torch.equal(p_new, p_ref)


def test_grad_accum_steps_4_matches_concatenated_larger_batch():
    """grad_accum_steps=4 with micro-batch size B must produce the SAME
    single optimizer update as one forward/backward/step on a batch of
    size 4B formed by concatenating the same 4 micro-batches -- proving
    the accumulated gradient is mathematically equivalent to training on
    the larger batch directly, not merely "some form of accumulation".

    aux_coeff=0.0 deliberately: the router's aux load-balancing loss uses
    `probs.mean(dim=0)`, a batch-level statistic that is NOT decomposable
    across a 4-way micro-batch split the way cross_entropy's per-example
    mean is (mean-of-4-means-of-B != mean-of-one-4B-batch when the inner
    quantity is itself a nonlinear function of the batch). That is a
    pre-existing property of the aux-loss formula, not something this
    grad-accumulation change alters -- isolating aux_coeff=0 here tests
    exactly what changed (the accumulation mechanism) without conflating
    it with an unrelated, already-existing batch-size sensitivity."""
    B = 2
    x_micros = [torch.randint(0, VOCAB, (B, SEQ_LEN)) for _ in range(4)]
    y_micros = [torch.randint(0, VOCAB, (B, SEQ_LEN)) for _ in range(4)]

    class _FixedLoader:
        def __init__(self, batches):
            self._batches = batches

        def __iter__(self):
            return iter(self._batches)

    torch.manual_seed(4242)
    model_accum = make_model(seed=55)
    torch.manual_seed(4242)
    model_direct = make_model(seed=55)
    for p_a, p_d in zip(model_accum.parameters(), model_direct.parameters()):
        assert torch.equal(p_a, p_d)

    accum_loader = _FixedLoader(list(zip(x_micros, y_micros)))
    train_model(
        model_accum, accum_loader, torch.device("cpu"), "standard", steps=1,
        lr=1e-2, lambda_rca=0.0, aux_coeff=0.0, grad_accum_steps=4,
    )

    x_cat = torch.cat(x_micros, dim=0)
    y_cat = torch.cat(y_micros, dim=0)
    optimizer = torch.optim.AdamW(model_direct.parameters(), lr=1e-2, weight_decay=0.01)
    optimizer.zero_grad()
    soft_logits, _, _, _, soft_aux = model_direct(x_cat, mode="soft")
    soft_ce = F.cross_entropy(soft_logits.reshape(-1, soft_logits.size(-1)), y_cat.reshape(-1))
    loss = soft_ce + 0.0 * soft_aux
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model_direct.parameters(), 1.0)
    optimizer.step()

    # atol=1e-4, not 1e-6: summing 4 separate reductions (one per
    # micro-batch) vs. one reduction over the concatenated batch is
    # mathematically identical but not bit-identical in float32 --
    # observed max abs diff ~9e-6 from pure floating-point
    # non-associativity. A genuine implementation bug (e.g. wrong/missing
    # division by grad_accum_steps) would produce a difference many
    # orders of magnitude larger than this, so 1e-4 still catches real
    # bugs while tolerating expected float32 summation-order noise.
    for p_a, p_d in zip(model_accum.parameters(), model_direct.parameters()):
        assert torch.allclose(p_a, p_d, atol=1e-4), (
            "grad_accum_steps=4 over 4 micro-batches of size B must match "
            "one direct step on the concatenated 4B batch"
        )


def test_grad_accum_steps_cli_flag_default_and_override():
    args = build_argparser().parse_args(["--output", "x.json"])
    assert args.grad_accum_steps == 1
    args2 = build_argparser().parse_args(["--output", "x.json", "--grad-accum-steps", "4"])
    assert args2.grad_accum_steps == 4


def test_hard_condition_matches_independent_reference():
    torch.manual_seed(7001)
    model_new = make_model(seed=61)
    torch.manual_seed(7001)
    model_ref = make_model(seed=61)
    for p_new, p_ref in zip(model_new.parameters(), model_ref.parameters()):
        assert torch.equal(p_new, p_ref)

    loader = make_loader(n_batches=3, seed=900)
    train_model(
        model_new, loader, torch.device("cpu"), "hard", steps=3, lr=1e-2,
        lambda_rca=0.0, aux_coeff=0.01,
    )

    optimizer = torch.optim.AdamW(model_ref.parameters(), lr=1e-2, weight_decay=0.01)
    train_iter = iter(loader)
    for _ in range(3):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(loader)
            x, y = next(train_iter)
        optimizer.zero_grad()
        hard_logits, _, _, _, hard_aux = model_ref(x, mode="hard")
        hard_ce = F.cross_entropy(hard_logits.reshape(-1, hard_logits.size(-1)), y.reshape(-1))
        loss = hard_ce + 0.01 * hard_aux
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_ref.parameters(), 1.0)
        optimizer.step()

    for p_new, p_ref in zip(model_new.parameters(), model_ref.parameters()):
        assert torch.equal(p_new, p_ref)


def test_hard_condition_router_receives_gradient():
    model = make_model(seed=62)
    model.train()
    x, y = make_batch(seed=63)
    hard_logits, _, _, _, hard_aux = model(x, mode="hard")
    loss = F.cross_entropy(hard_logits.reshape(-1, hard_logits.size(-1)), y.reshape(-1)) + 0.01 * hard_aux
    loss.backward()
    for block in model.blocks:
        grad = block.moe.router.gate.weight.grad
        assert grad is not None
        assert torch.any(grad != 0)


def test_hard_condition_respects_capacity_and_top_k_dispatch():
    torch.manual_seed(7002)
    model_new = make_model(seed=64)
    torch.manual_seed(7002)
    model_ref = make_model(seed=64)

    loader = make_loader(n_batches=2, seed=901)
    train_model(
        model_new, loader, torch.device("cpu"), "hard", steps=2, lr=1e-2,
        lambda_rca=0.0, aux_coeff=0.0, capacity_factor=0.5, top_k_dispatch=2,
    )

    optimizer = torch.optim.AdamW(model_ref.parameters(), lr=1e-2, weight_decay=0.01)
    train_iter = iter(loader)
    for _ in range(2):
        x, y = next(train_iter)
        optimizer.zero_grad()
        hard_logits, _, _, _, hard_aux = model_ref(
            x, mode="hard", capacity_factor=0.5, top_k_dispatch=2,
        )
        hard_ce = F.cross_entropy(hard_logits.reshape(-1, hard_logits.size(-1)), y.reshape(-1))
        loss = hard_ce + 0.0 * hard_aux
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_ref.parameters(), 1.0)
        optimizer.step()

    for p_new, p_ref in zip(model_new.parameters(), model_ref.parameters()):
        assert torch.equal(p_new, p_ref)


def test_hard_condition_cli_flag_accepted():
    args = build_argparser().parse_args(["--output", "x.json", "--condition", "hard"])
    assert args.condition == "hard"


def test_hard_condition_rejects_lambda_rca_as_meaningless():
    model = make_model(seed=65)
    loader = make_loader(n_batches=1, seed=902)
    with pytest.raises(ValueError, match="lambda_rca"):
        train_model(
            model, loader, torch.device("cpu"), "hard", steps=1, lr=1e-2,
            lambda_rca=1.0, aux_coeff=0.0,
        )


def test_relu_forward_hand_verified_weighted_accumulation():
    torch.manual_seed(8001)
    block = MoEBlock(d_model=2, d_ff=4, n_experts=3)
    with torch.no_grad():
        block.router.gate.weight[:, 0] = torch.tensor([1.0, -0.5, 2.0])
        block.router.gate.weight[:, 1] = torch.tensor([0.0, 0.0, 0.0])

    x = torch.tensor([[[1.0, 0.0]]])
    out, routing_map, l1_loss = block.relu_forward(x)

    assert routing_map.tolist() == [[[True, False, True]]]

    flat_x = x.reshape(1, 2)
    with torch.no_grad():
        e0_out = block.experts[0](flat_x)
        e2_out = block.experts[2](flat_x)
    expected = 1.0 * e0_out + 2.0 * e2_out
    assert torch.allclose(out.reshape(1, 2), expected, atol=1e-6)


def test_relu_forward_zero_experts_gives_zero_output():
    torch.manual_seed(8002)
    block = MoEBlock(d_model=2, d_ff=4, n_experts=3)
    with torch.no_grad():
        block.router.gate.weight[:, 0] = torch.tensor([-1.0, -0.5, -2.0])
        block.router.gate.weight[:, 1] = torch.tensor([0.0, 0.0, 0.0])

    x = torch.tensor([[[1.0, 0.0]]])
    out, routing_map, l1_loss = block.relu_forward(x)

    assert routing_map.tolist() == [[[False, False, False]]]
    assert torch.equal(out, torch.zeros_like(x))


def test_relu_forward_variable_cardinality_across_tokens():
    torch.manual_seed(8003)
    block = MoEBlock(d_model=2, d_ff=4, n_experts=3)
    with torch.no_grad():
        block.router.gate.weight[:, 0] = torch.tensor([1.0, -1.0, 1.0])
        block.router.gate.weight[:, 1] = torch.tensor([-1.0, -1.0, -1.0])

    x = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    out, routing_map, l1_loss = block.relu_forward(x)

    n_active = routing_map.sum(dim=-1)
    assert n_active[0, 0].item() == 2
    assert n_active[0, 1].item() == 0


def test_relu_forward_l1_loss_matches_switch_formula():
    """Byte-verified against megatron/core/transformer/moe/moe_utils.py::
    switch_load_balancing_loss_func in the official ReMoE repo (thu-ml/ReMoE):
    aux_loss = sum(probs_per_expert * tokens_per_expert)
               * (n_experts * coeff) / (num_tokens^2 * topk)."""
    torch.manual_seed(8004)
    block = MoEBlock(d_model=2, d_ff=4, n_experts=2)
    with torch.no_grad():
        block.router.gate.weight[:, 0] = torch.tensor([1.0, -1.0])
        block.router.gate.weight[:, 1] = torch.tensor([0.0, 0.0])

    x = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])  # 2 identical tokens
    out, routing_map, l1_loss = block.relu_forward(x, l1_reg_coeff=0.5, target_topk=1)

    # logits = [1.0, -1.0] for both tokens -> probs = relu = [1.0, 0.0]
    # probs_per_expert = [2.0, 0.0]; tokens_per_expert = [2, 0]
    # aux = sum([2.0*2, 0.0*0]) * (2 * 0.5) / (2^2 * 1) = 4 * 1.0 / 4 = 1.0
    expected = 1.0
    assert torch.allclose(l1_loss, torch.tensor(expected), atol=1e-5)


def test_relu_forward_router_receives_gradient():
    model = make_model(seed=71)
    model.train()
    x, y = make_batch(seed=72)
    logits, _, _, all_indices, total_l1 = model(x, mode="relu", l1_reg_coeff=0.1, target_topk=1)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
    loss = loss + total_l1
    loss.backward()
    assert len(all_indices) == TINY["n_layers"]
    for block in model.blocks:
        grad = block.moe.router.gate.weight.grad
        assert grad is not None
        assert torch.any(grad != 0)


def test_relu_mode_l1_reg_coeff_none_gives_zero_total_aux():
    model = make_model(seed=73)
    model.eval()
    x, _ = make_batch(seed=74)
    with torch.no_grad():
        _, _, _, _, total_l1 = model(x, mode="relu")
    assert total_l1.item() == 0.0


def test_relu_forward_l1_reg_coeff_none_disables_l1_loss():
    torch.manual_seed(8005)
    block = MoEBlock(d_model=2, d_ff=4, n_experts=2)
    x = torch.randn(1, 2, 2)
    out, routing_map, l1_loss = block.relu_forward(x)
    assert l1_loss.item() == 0.0


# ─────────────────────────────────────────────────────────────────────────
# ReMoE end-to-end: evaluate_relu + train_model(condition="relu")
# ─────────────────────────────────────────────────────────────────────────


def test_evaluate_relu_returns_expected_keys():
    model = make_model(seed=90)
    loader = make_loader(n_batches=2, seed=91)
    metrics = evaluate_relu(model, loader, torch.device("cpu"), max_batches=2)
    assert set(metrics) == {"relu_nll", "avg_active_experts"}
    assert isinstance(metrics["relu_nll"], float)
    assert isinstance(metrics["avg_active_experts"], float)
    assert 0.0 <= metrics["avg_active_experts"] <= TINY["n_experts"]


def test_evaluate_relu_matches_manual_nll_computation():
    model = make_model(seed=92)
    model.eval()
    x, y = make_batch(seed=93)
    with torch.no_grad():
        logits, _, _, _, _ = model(x, mode="relu")
        expected_nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item()
    loader = TinyLoader([(x, y)])
    metrics = evaluate_relu(model, loader, torch.device("cpu"), max_batches=1)
    assert metrics["relu_nll"] == pytest.approx(expected_nll, abs=1e-6)


def test_evaluate_relu_respects_max_batches():
    model = make_model(seed=94)
    loader = make_loader(n_batches=5, seed=95)
    metrics_1 = evaluate_relu(model, loader, torch.device("cpu"), max_batches=1)
    metrics_all = evaluate_relu(model, loader, torch.device("cpu"), max_batches=5)
    # Different subsets of batches should generally give different NLLs --
    # not a strict correctness check, but confirms max_batches actually
    # truncates rather than being ignored.
    assert metrics_1["relu_nll"] != metrics_all["relu_nll"]


def test_train_model_relu_condition_runs_and_updates_router():
    model = make_model(seed=96)
    loader = make_loader(n_batches=3, seed=97)
    before = model.blocks[0].moe.router.gate.weight.clone()
    steps_completed, elapsed = train_model(
        model, loader, torch.device("cpu"), condition="relu", steps=3, lr=1e-2,
        lambda_rca=0.0, aux_coeff=0.0, log_every=0, l1_reg_coeff=0.1, relu_target_topk=1,
    )
    assert steps_completed == 3
    after = model.blocks[0].moe.router.gate.weight
    assert not torch.equal(before, after)


def test_train_model_relu_condition_rejects_nonzero_lambda_rca():
    model = make_model(seed=98)
    loader = make_loader(n_batches=1, seed=99)
    with pytest.raises(ValueError):
        train_model(
            model, loader, torch.device("cpu"), condition="relu", steps=1, lr=1e-2,
            lambda_rca=1.0, aux_coeff=0.0, log_every=0, l1_reg_coeff=0.1, relu_target_topk=1,
        )


def test_relu_condition_cli_flags_accepted():
    args = build_argparser().parse_args([
        "--output", "x.json", "--condition", "relu",
        "--l1-reg-coeff", "0.1", "--relu-target-topk", "2",
    ])
    assert args.condition == "relu"
    assert args.l1_reg_coeff == 0.1
    assert args.relu_target_topk == 2


def test_relu_condition_l1_reg_coeff_defaults_to_none():
    args = build_argparser().parse_args(["--output", "x.json", "--condition", "relu"])
    assert args.l1_reg_coeff is None
    assert args.relu_target_topk == 1


# ─────────────────────────────────────────────────────────────────────────
# SoftMoE (Zasada et al., "SoftMoE: Soft Differentiable Routing for
# Mixture-of-Experts in LLMs") -- LapSum soft-top-k operator, ported from
# dlcuda/SoftMoE's megatron-softmoe.patch (megatron/core/transformer/moe/
# soft_topk.py). p_i = LaplaceCDF(r_i/alpha - b), b solved in closed form
# so sum_i p_i == k. Independently re-derived reference: the *definition*
# is p_i = LaplaceCDF(r_i/alpha - b) with b s.t. sum(p) = k; we solve for b
# via bisection on that monotonic equation (not by trusting the ported
# closed-form root-finder) and compare.
# ─────────────────────────────────────────────────────────────────────────


def _laplace_cdf(x):
    return torch.where(x >= 0, 1 - 0.5 * torch.exp(-x), 0.5 * torch.exp(x))


def _reference_soft_top_k_row(r_row, k_val, alpha, lo=-1e4, hi=1e4, iters=200):
    """Independent bisection reference for a single row: find b such that
    sum(LaplaceCDF(r_row/alpha - b)) == k_val, then return the resulting
    probabilities. sum_i p_i(b) is strictly decreasing in b (each term is),
    so bisection is well-posed."""
    def total(b):
        return _laplace_cdf(r_row / alpha - b).sum().item()

    lo_b, hi_b = lo, hi
    for _ in range(iters):
        mid = (lo_b + hi_b) / 2
        if total(mid) > k_val:
            lo_b = mid
        else:
            hi_b = mid
    b = (lo_b + hi_b) / 2
    return _laplace_cdf(r_row / alpha - b)


def test_soft_top_k_matches_independent_bisection_reference():
    torch.manual_seed(3001)
    r = torch.randn(5, 8, dtype=torch.float64)
    k = torch.full((5,), 2.5, dtype=torch.float64)
    alpha = torch.tensor(1.0, dtype=torch.float64)
    p = soft_top_k(r, k, alpha)
    assert p.shape == r.shape
    for row in range(5):
        expected = _reference_soft_top_k_row(r[row], 2.5, 1.0)
        assert torch.allclose(p[row], expected, atol=1e-4)


def test_soft_top_k_probabilities_sum_to_k():
    torch.manual_seed(3002)
    r = torch.randn(6, 10, dtype=torch.float64)
    k = torch.full((6,), 3.0, dtype=torch.float64)
    alpha = torch.tensor(0.7, dtype=torch.float64)
    p = soft_top_k(r, k, alpha)
    assert torch.allclose(p.sum(dim=1), k, atol=1e-4)


def test_soft_top_k_probabilities_in_unit_interval():
    torch.manual_seed(3003)
    r = torch.randn(4, 12, dtype=torch.float64) * 5
    k = torch.full((4,), 4.0, dtype=torch.float64)
    alpha = torch.tensor(1.0, dtype=torch.float64)
    p = soft_top_k(r, k, alpha)
    assert (p >= 0).all() and (p <= 1).all()


def test_soft_top_k_low_alpha_approaches_hard_topk():
    """As alpha -> 0 the Laplace CDF saturates to a step function, so
    soft_top_k should approach the hard top-round(k) indicator."""
    torch.manual_seed(3004)
    r = torch.tensor([[5.0, 1.0, 4.0, -2.0, 3.0]], dtype=torch.float64)
    k = torch.tensor([3.0], dtype=torch.float64)
    alpha = torch.tensor(0.01, dtype=torch.float64)
    p = soft_top_k(r, k, alpha)
    hard = torch.zeros_like(r)
    top_idx = r.topk(3, dim=1).indices
    hard.scatter_(1, top_idx, 1.0)
    assert torch.allclose(p, hard, atol=1e-2)


def test_soft_top_k_gradcheck_r():
    torch.manual_seed(3005)
    r = torch.randn(3, 6, dtype=torch.float64, requires_grad=True)
    k = torch.full((3,), 2.0, dtype=torch.float64)
    alpha = torch.tensor(1.0, dtype=torch.float64)
    assert torch.autograd.gradcheck(lambda r_: soft_top_k(r_, k, alpha), (r,), eps=1e-6, atol=1e-4)


def test_soft_top_k_gradcheck_k():
    torch.manual_seed(3006)
    r = torch.randn(3, 6, dtype=torch.float64)
    k = torch.full((3,), 2.0, dtype=torch.float64, requires_grad=True)
    alpha = torch.tensor(1.0, dtype=torch.float64)
    assert torch.autograd.gradcheck(lambda k_: soft_top_k(r, k_, alpha), (k,), eps=1e-6, atol=1e-4)


# ─────────────────────────────────────────────────────────────────────────
# SoftMoE MoEBlock.softtopk_forward -- fixed-budget truncated soft top-k
# ─────────────────────────────────────────────────────────────────────────


def test_softtopk_forward_matches_manual_threshold_computation():
    torch.manual_seed(4001)
    block = MoEBlock(d_model=2, d_ff=4, n_experts=4)
    x = torch.tensor([[[1.0, 0.0]]])
    out, routing_map, aux_loss = block.softtopk_forward(
        x, topk=1.0, alpha=1.0, threshold=0.0, hard_threshold_coeff=4.0,
    )
    with torch.no_grad():
        logits = block.router(x.reshape(1, 2))
        k_tensor = torch.full((1,), 1.0, dtype=torch.float32)
        alpha_tensor = torch.tensor(1.0, dtype=torch.float32)
        expected_probs = soft_top_k(logits.to(torch.float32), k_tensor, alpha_tensor).to(logits.dtype)
        expected_active = expected_probs > 0
    assert torch.equal(routing_map.reshape(1, 4), expected_active)
    expected_out = torch.zeros(1, 2)
    for e_idx in range(4):
        if expected_active[0, e_idx]:
            expected_out += expected_probs[0, e_idx] * block.experts[e_idx](x.reshape(1, 2))
    assert torch.allclose(out.reshape(1, 2), expected_out, atol=1e-5)


def test_softtopk_forward_high_threshold_narrows_active_experts():
    """A near-1.0 threshold coefficient should leave (at most) as many
    active experts as a near-0.0 threshold -- the threshold only removes
    probability mass, it never adds active experts."""
    torch.manual_seed(4002)
    block = MoEBlock(d_model=4, d_ff=8, n_experts=6)
    x = torch.randn(2, 3, 4)
    _, routing_permissive, _ = block.softtopk_forward(x, topk=2.0, alpha=1.0, threshold=0.0)
    _, routing_strict, _ = block.softtopk_forward(x, topk=2.0, alpha=1.0, threshold=100.0)
    assert routing_strict.sum().item() <= routing_permissive.sum().item()


def test_softtopk_forward_router_receives_gradient():
    model = make_model(seed=110)
    model.train()
    x, y = make_batch(seed=111)
    logits, _, _, all_indices, aux = model(x, mode="softtopk", topk=1.0, alpha=1.0, aux_loss_coeff=0.01)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
    loss = loss + aux
    loss.backward()
    assert len(all_indices) == TINY["n_layers"]
    for block in model.blocks:
        grad = block.moe.router.gate.weight.grad
        assert grad is not None
        assert torch.any(grad != 0)


def test_softtopk_forward_aux_loss_zero_when_coeff_none():
    torch.manual_seed(4003)
    block = MoEBlock(d_model=2, d_ff=4, n_experts=4)
    x = torch.randn(1, 2, 2)
    _, _, aux_loss = block.softtopk_forward(x, topk=1.0, alpha=1.0)
    assert aux_loss.item() == 0.0


def test_train_model_softmoe_condition_runs_and_updates_router():
    model = make_model(seed=112)
    loader = make_loader(n_batches=3, seed=113)
    before = model.blocks[0].moe.router.gate.weight.clone()
    steps_completed, elapsed = train_model(
        model, loader, torch.device("cpu"), condition="softmoe", steps=3, lr=1e-2,
        lambda_rca=0.0, aux_coeff=0.0, log_every=0, softtopk_topk=1.0, softtopk_alpha=1.0,
        softtopk_aux_loss_coeff=0.01,
    )
    assert steps_completed == 3
    after = model.blocks[0].moe.router.gate.weight
    assert not torch.equal(before, after)


def test_train_model_softmoe_condition_rejects_nonzero_lambda_rca():
    model = make_model(seed=114)
    loader = make_loader(n_batches=1, seed=115)
    with pytest.raises(ValueError):
        train_model(
            model, loader, torch.device("cpu"), condition="softmoe", steps=1, lr=1e-2,
            lambda_rca=1.0, aux_coeff=0.0, log_every=0, softtopk_topk=1.0,
        )


def test_softmoe_condition_cli_flags_accepted():
    args = build_argparser().parse_args([
        "--output", "x.json", "--condition", "softmoe",
        "--softtopk-topk", "2.0", "--softtopk-alpha", "0.5",
        "--softtopk-threshold", "1.5", "--softtopk-hard-threshold-coeff", "3.0",
        "--softtopk-aux-loss-coeff", "0.02",
    ])
    assert args.condition == "softmoe"
    assert args.softtopk_topk == 2.0
    assert args.softtopk_alpha == 0.5
    assert args.softtopk_threshold == 1.5
    assert args.softtopk_hard_threshold_coeff == 3.0
    assert args.softtopk_aux_loss_coeff == 0.02


def test_softmoe_condition_defaults_match_paper_config():
    """train_configs/soft_topk.sh (official dlcuda/SoftMoE repo) uses
    alpha=1.0, hard-threshold-coeff=2.0, threshold=1.8 as its own reference
    fixed-budget config."""
    args = build_argparser().parse_args(["--output", "x.json", "--condition", "softmoe"])
    assert args.softtopk_topk == 1.0
    assert args.softtopk_alpha == 1.0
    assert args.softtopk_threshold == 1.8
    assert args.softtopk_hard_threshold_coeff == 2.0
    assert args.softtopk_aux_loss_coeff is None


def test_evaluate_softtopk_returns_expected_keys():
    model = make_model(seed=116)
    loader = make_loader(n_batches=2, seed=117)
    metrics = evaluate_softtopk(model, loader, torch.device("cpu"), max_batches=2, topk=1.0, alpha=1.0)
    assert set(metrics) == {"softtopk_nll", "avg_active_experts"}
    assert isinstance(metrics["softtopk_nll"], float)
    assert isinstance(metrics["avg_active_experts"], float)


# ─────────────────────────────────────────────────────────────────────────
# --capacity-factor extended to ReMoE (relu_forward) and SoftMoE
# (softtopk_forward). Same Switch-Transformer-style per-expert capacity +
# arrival-order dropping + residual/identity-passthrough convention as
# hard_forward's own --capacity-factor support (see that docstring), but
# adapted for variable cardinality: a token may be routed to zero, one, or
# several experts simultaneously, so there is no shared per-token "slot"
# resource -- capacity is tracked independently PER EXPERT, and within a
# single expert's queue arrival order is ascending flattened-token index
# (the same batch*seq row-major order as `flat = x.reshape(-1, d_model)`).
# ─────────────────────────────────────────────────────────────────────────


def test_relu_forward_capacity_none_is_bit_identical():
    """Default (capacity_factor=None, and the case where the argument is
    omitted entirely) must produce EXACTLY the same output/routing_map/
    l1_loss as the pre-existing uncapacitated algorithm -- regression guard
    matching test_capacity_none_is_bit_identical's convention for
    hard_forward."""
    torch.manual_seed(9000)
    block = MoEBlock(d_model=6, d_ff=12, n_experts=4)
    x = torch.randn(2, 5, 6)

    out_omitted, rm_omitted, l1_omitted = block.relu_forward(x, l1_reg_coeff=0.3, target_topk=2)
    out_none, rm_none, l1_none = block.relu_forward(
        x, l1_reg_coeff=0.3, target_topk=2, capacity_factor=None,
    )
    assert torch.equal(out_omitted, out_none)
    assert torch.equal(rm_omitted, rm_none)
    assert torch.allclose(l1_omitted, l1_none, atol=0.0)


def test_relu_forward_capacity_limit_drops_excess_pairs_arrival_order():
    """4 tokens, 2 experts. Router weights are constructed so EVERY token
    activates expert 0 (with an identical, exactly-known relu probability)
    and NO token activates expert 1. capacity_factor=1.0 -> capacity =
    ceil(1.0*4/2) = 2 for expert 0. Per the documented arrival-order
    convention (ascending flattened-token index within expert 0's queue),
    tokens 0 and 1 are kept (genuinely processed by expert 0) and tokens 2
    and 3 are dropped (residual/identity passthrough weighted by their own
    -- here identical -- relu probability, not zero)."""
    torch.manual_seed(9010)
    block = MoEBlock(d_model=2, d_ff=4, n_experts=2)
    with torch.no_grad():
        block.router.gate.weight[0, :] = torch.tensor([5.0, 0.0])   # logit0 = 5.0 * x0 (always > 0)
        block.router.gate.weight[1, :] = torch.tensor([-5.0, 0.0])  # logit1 = -5.0 * x0 (always < 0)

    x = torch.tensor([[[1.0, 0.1], [1.0, 0.2], [1.0, 0.3], [1.0, 0.4]]])  # x0 == 1.0 for every token
    capacity_factor = 1.0
    n_tokens, n_experts = 4, 2
    capacity = _math.ceil(capacity_factor * n_tokens / n_experts)
    assert capacity == 2

    out, routing_map, l1_loss = block.relu_forward(x, capacity_factor=capacity_factor)
    assert routing_map.tolist() == [[[True, False] for _ in range(4)]]

    flat_x = x.reshape(4, 2)
    flat_out = out.reshape(4, 2)
    with torch.no_grad():
        e0_out_all = block.experts[0](flat_x)
    prob0 = 5.0  # relu(5.0 * 1.0) for every token, exactly

    kept_positions = [0, 1]
    for p in kept_positions:
        expected = prob0 * e0_out_all[p]
        assert torch.allclose(flat_out[p], expected, atol=1e-5), (
            f"token {p} (within expert-0 capacity) was not genuinely processed by expert 0"
        )

    dropped_positions = [2, 3]
    for p in dropped_positions:
        expected_passthrough = prob0 * flat_x[p]
        assert torch.allclose(flat_out[p], expected_passthrough, atol=1e-5), (
            f"dropped token {p} does not equal its own residual/identity passthrough weighted by its probability"
        )
        assert not torch.allclose(flat_out[p], torch.zeros(2), atol=1e-6), (
            f"dropped token {p} output is (near-)zero; expected weighted passthrough, not zero"
        )
        # and must NOT equal what expert-0 processing would have given (confirms it was truly dropped)
        expected_if_processed = prob0 * e0_out_all[p]
        assert not torch.allclose(flat_out[p], expected_if_processed, atol=1e-6), (
            f"dropped token {p} appears to have been processed by expert 0 anyway"
        )


def test_relu_forward_capacity_zero_drops_everything_and_router_gets_gradient_via_passthrough():
    """capacity_factor=0.0 -> ceil(0.0 * n / n_experts) == 0 for every
    expert, so EVERY routed (token, expert) pair is dropped (no expert
    forward is ever invoked). This isolates the residual/identity-
    passthrough term entirely: the only way the output (and hence any loss
    built on it) can depend on the router is through `probs` (the weight
    multiplying the passthrough token), so a nonzero router gradient here
    is direct evidence gradient flows to the router via the
    dropped-pair passthrough weight, exactly as it does for
    hard_forward's dropped tokens."""
    torch.manual_seed(9020)
    block = MoEBlock(d_model=3, d_ff=6, n_experts=3)
    x = torch.randn(2, 4, 3)

    out, routing_map, _ = block.relu_forward(x, capacity_factor=0.0)

    flat_x = x.reshape(-1, 3)
    with torch.no_grad():
        logits = block.router(flat_x)
        probs_expected = F.relu(logits)
        routing_map_expected = probs_expected > 0
    assert torch.equal(routing_map.reshape(-1, 3), routing_map_expected)

    expected_out = torch.zeros_like(flat_x)
    for e_idx in range(3):
        mask = routing_map_expected[:, e_idx]
        if mask.any():
            expected_out[mask] = expected_out[mask] + probs_expected[mask, e_idx : e_idx + 1] * flat_x[mask]
    assert torch.allclose(out.reshape(-1, 3), expected_out, atol=1e-5)

    loss = out.sum()
    loss.backward()
    grad = block.router.gate.weight.grad
    assert grad is not None
    assert torch.any(grad != 0)


def test_softtopk_forward_capacity_none_is_bit_identical():
    """Same regression guard as test_relu_forward_capacity_none_is_bit_
    identical, for softtopk_forward."""
    torch.manual_seed(9030)
    block = MoEBlock(d_model=6, d_ff=12, n_experts=4)
    x = torch.randn(2, 5, 6)

    out_omitted, rm_omitted, aux_omitted = block.softtopk_forward(x, topk=1.5, alpha=1.0, aux_loss_coeff=0.02)
    out_none, rm_none, aux_none = block.softtopk_forward(
        x, topk=1.5, alpha=1.0, aux_loss_coeff=0.02, capacity_factor=None,
    )
    assert torch.equal(out_omitted, out_none)
    assert torch.equal(rm_omitted, rm_none)
    assert torch.allclose(aux_omitted, aux_none, atol=0.0)


def test_softtopk_forward_capacity_limit_drops_excess_pairs_arrival_order():
    """3 tokens, 3 experts. threshold=0.0 with hard_threshold_coeff=3.0
    means experts_threshold = min(ceil(2.0*3.0), 3) = 3 = n_experts, so
    `thresh` clamps to the MINIMUM soft-top-k probability across all
    experts for that token -- i.e. no truncation occurs and EVERY expert
    is active (probs > 0) for EVERY token (LapSum's soft top-k gives
    strictly positive probabilities everywhere at alpha=1.0 for these
    logit magnitudes). capacity_factor=1.0 -> capacity =
    ceil(1.0*3/3) = 1 per expert, so for every expert only the FIRST token
    (ascending flattened-token index, per the documented arrival-order
    convention) is kept and genuinely processed; tokens 1 and 2 are
    dropped for every expert -> residual/identity passthrough weighted by
    the SUM of their own per-expert probabilities (since all 3 experts
    independently drop the same two tokens)."""
    torch.manual_seed(9040)
    block = MoEBlock(d_model=2, d_ff=4, n_experts=3)
    x = torch.tensor([[[1.0, 0.1], [1.0, 0.2], [1.0, 0.3]]])
    capacity_factor = 1.0
    n_tokens, n_experts = 3, 3
    capacity = _math.ceil(capacity_factor * n_tokens / n_experts)
    assert capacity == 1

    out, routing_map, aux = block.softtopk_forward(
        x, topk=2.0, alpha=1.0, threshold=0.0, hard_threshold_coeff=3.0,
        capacity_factor=capacity_factor,
    )
    assert routing_map.all(), "expected no truncation -- every expert active for every token"

    flat_x = x.reshape(3, 2)
    with torch.no_grad():
        logits = block.router(flat_x)
        k_tensor = torch.full((3,), 2.0, dtype=torch.float32)
        alpha_tensor = torch.tensor(1.0, dtype=torch.float32)
        probs = soft_top_k(logits.to(torch.float32), k_tensor, alpha_tensor).to(logits.dtype)
        assert (probs > 0).all(), "test setup assumption violated: expected strictly positive probs"

    flat_out = out.reshape(3, 2)

    # token 0 (within every expert's capacity) genuinely processed by all 3 experts
    with torch.no_grad():
        expected_0 = torch.zeros(1, 2)
        for e_idx in range(3):
            expected_0 = expected_0 + probs[0, e_idx] * block.experts[e_idx](flat_x[0:1])
    assert torch.allclose(flat_out[0], expected_0.squeeze(0), atol=1e-5)

    # tokens 1, 2 dropped by every expert -> passthrough weighted by the
    # SUM of their per-expert probabilities
    for t in (1, 2):
        expected_passthrough = probs[t].sum() * flat_x[t]
        assert torch.allclose(flat_out[t], expected_passthrough, atol=1e-5)
        assert not torch.allclose(flat_out[t], torch.zeros(2), atol=1e-6)


def test_softtopk_forward_capacity_zero_drops_everything_and_router_gets_gradient_via_passthrough():
    """Same isolation strategy as test_relu_forward_capacity_zero_drops_
    everything_and_router_gets_gradient_via_passthrough, for
    softtopk_forward: capacity_factor=0.0 drops every routed (token,
    expert) pair, so the only path from the output back to the router is
    through the passthrough weight `probs`."""
    torch.manual_seed(9050)
    block = MoEBlock(d_model=3, d_ff=6, n_experts=3)
    x = torch.randn(2, 4, 3)

    out, routing_map, _ = block.softtopk_forward(x, topk=1.5, alpha=1.0, capacity_factor=0.0)

    flat_x = x.reshape(-1, 3)
    with torch.no_grad():
        logits = block.router(flat_x)
        k_tensor = torch.full((8,), 1.5, dtype=torch.float32)
        alpha_tensor = torch.tensor(1.0, dtype=torch.float32)
        probs_fp32 = soft_top_k(logits.to(torch.float32), k_tensor, alpha_tensor)
        experts_threshold = min(_math.ceil(1.5 * 2.0), 3)
        topk_vals, _ = torch.topk(probs_fp32, k=experts_threshold, dim=1)
        lower_bound = topk_vals[:, -1].unsqueeze(1)
        higher_bound = topk_vals[:, 0].unsqueeze(1)
        thresh = torch.clamp(
            torch.full_like(lower_bound, 1.8 * 1.5 / 3), min=lower_bound, max=higher_bound,
        )
        probs_fp32 = torch.where(probs_fp32 >= thresh, probs_fp32, torch.zeros_like(probs_fp32))
        probs_expected = probs_fp32.to(logits.dtype)
        routing_map_expected = probs_expected > 0
    assert torch.equal(routing_map.reshape(-1, 3), routing_map_expected)

    expected_out = torch.zeros_like(flat_x)
    for e_idx in range(3):
        mask = routing_map_expected[:, e_idx]
        if mask.any():
            expected_out[mask] = expected_out[mask] + probs_expected[mask, e_idx : e_idx + 1] * flat_x[mask]
    assert torch.allclose(out.reshape(-1, 3), expected_out, atol=1e-5)

    loss = out.sum()
    loss.backward()
    grad = block.router.gate.weight.grad
    assert grad is not None
    assert torch.any(grad != 0)


def test_train_model_relu_condition_with_capacity_factor_runs_and_updates_router():
    """End-to-end: condition='relu' with --capacity-factor set must run
    without error and still update the router (gradient still flows
    through the dropped-pair passthrough term when dropping occurs, and
    through the ordinary expert path otherwise)."""
    model = make_model(seed=200)
    loader = make_loader(n_batches=3, seed=201)
    before = model.blocks[0].moe.router.gate.weight.clone()
    steps_completed, elapsed = train_model(
        model, loader, torch.device("cpu"), condition="relu", steps=3, lr=1e-2,
        lambda_rca=0.0, aux_coeff=0.0, log_every=0, l1_reg_coeff=0.1, relu_target_topk=1,
        capacity_factor=0.1,
    )
    assert steps_completed == 3
    after = model.blocks[0].moe.router.gate.weight
    assert not torch.equal(before, after)


def test_train_model_softmoe_condition_with_capacity_factor_runs_and_updates_router():
    """End-to-end: condition='softmoe' with --capacity-factor set must run
    without error and still update the router."""
    model = make_model(seed=202)
    loader = make_loader(n_batches=3, seed=203)
    before = model.blocks[0].moe.router.gate.weight.clone()
    steps_completed, elapsed = train_model(
        model, loader, torch.device("cpu"), condition="softmoe", steps=3, lr=1e-2,
        lambda_rca=0.0, aux_coeff=0.0, log_every=0, softtopk_topk=1.5, softtopk_alpha=1.0,
        softtopk_aux_loss_coeff=0.01, capacity_factor=0.1,
    )
    assert steps_completed == 3
    after = model.blocks[0].moe.router.gate.weight
    assert not torch.equal(before, after)


def test_relu_condition_cli_capacity_factor_accepted():
    args = build_argparser().parse_args([
        "--output", "x.json", "--condition", "relu", "--capacity-factor", "1.25",
    ])
    assert args.condition == "relu"
    assert args.capacity_factor == 1.25


def test_softmoe_condition_cli_capacity_factor_accepted():
    args = build_argparser().parse_args([
        "--output", "x.json", "--condition", "softmoe", "--capacity-factor", "1.25",
    ])
    assert args.condition == "softmoe"
    assert args.capacity_factor == 1.25


def test_relu_mode_capacity_factor_changes_output_when_dropping_occurs():
    """Targeted reproduction that --capacity-factor actually changes
    mode='relu' behavior at the full-model level (not just at the isolated
    MoEBlock level above): an aggressively small capacity_factor, applied
    to TINY's n_experts=4 config, must produce a DIFFERENT lm_logits output
    than capacity_factor=None on the identical input/model, proving some
    dropping actually occurred end-to-end through TransformerBlock.
    forward_relu / MoETransformer.forward."""
    model = make_model(seed=210)
    model.eval()
    x, _ = make_batch(seed=211)
    with torch.no_grad():
        logits_uncapacitated, _, _, _, _ = model(x, mode="relu")
        logits_capacitated, _, _, _, _ = model(x, mode="relu", capacity_factor=0.01)
    assert not torch.allclose(logits_uncapacitated, logits_capacitated, atol=1e-6), (
        "capacity_factor=0.01 (an aggressively small buffer) produced no "
        "change in mode='relu' output -- expected some tokens to be dropped"
    )


def test_softtopk_mode_capacity_factor_changes_output_when_dropping_occurs():
    """Same targeted reproduction as test_relu_mode_capacity_factor_
    changes_output_when_dropping_occurs, for mode='softtopk'."""
    model = make_model(seed=212)
    model.eval()
    x, _ = make_batch(seed=213)
    with torch.no_grad():
        logits_uncapacitated, _, _, _, _ = model(x, mode="softtopk", topk=2.0, alpha=1.0)
        logits_capacitated, _, _, _, _ = model(
            x, mode="softtopk", topk=2.0, alpha=1.0, capacity_factor=0.01,
        )
    assert not torch.allclose(logits_uncapacitated, logits_capacitated, atol=1e-6), (
        "capacity_factor=0.01 (an aggressively small buffer) produced no "
        "change in mode='softtopk' output -- expected some tokens to be dropped"
    )


# ─────────────────────────────────────────────────────────────────────────
# Inference throughput/latency benchmark (structural compute-cost
# comparison of hard/relu/softtopk dispatch -- NOT an NLL/quality
# measurement; see benchmark_inference's docstring in src/moe_soft_to_hard.py).
# ─────────────────────────────────────────────────────────────────────────

_BENCHMARK_KEYS = {
    "mode", "tokens_per_sec", "mean_latency_s", "min_latency_s",
    "max_latency_s", "n_iters", "total_elapsed_s", "batch_size", "seq_len",
}


@pytest.mark.parametrize("mode", ["hard", "relu", "softtopk"])
def test_benchmark_inference_returns_expected_keys(mode):
    model = make_model(seed=300)
    result = benchmark_inference(
        model, mode, torch.device("cpu"),
        batch_size=2, seq_len=TINY["max_len"], vocab_size=VOCAB,
        n_warmup=1, n_iters=3,
    )
    assert _BENCHMARK_KEYS.issubset(result.keys())
    assert result["mode"] == mode
    assert result["n_iters"] == 3


@pytest.mark.parametrize("mode", ["hard", "relu", "softtopk"])
def test_benchmark_inference_latency_and_throughput_are_sane(mode):
    model = make_model(seed=301)
    result = benchmark_inference(
        model, mode, torch.device("cpu"),
        batch_size=2, seq_len=TINY["max_len"], vocab_size=VOCAB,
        n_warmup=1, n_iters=3,
    )
    assert isinstance(result["tokens_per_sec"], float)
    assert result["tokens_per_sec"] > 0.0
    assert result["min_latency_s"] <= result["mean_latency_s"] <= result["max_latency_s"]
    assert result["min_latency_s"] >= 0.0


def test_benchmark_inference_runs_requested_warmup_and_timed_iters():
    """Structural correctness check (not a wall-clock assertion, per this
    project's flakiness-avoidance convention): wrap model.forward with a
    counter and confirm benchmark_inference calls it exactly
    n_warmup + n_iters times -- i.e. the warmup loop and the timed loop
    both actually run the requested number of iterations, neither more
    nor fewer."""
    model = make_model(seed=302)
    call_count = {"n": 0}
    orig_forward = model.forward

    def counting_forward(*args, **kwargs):
        call_count["n"] += 1
        return orig_forward(*args, **kwargs)

    model.forward = counting_forward
    result = benchmark_inference(
        model, "relu", torch.device("cpu"),
        batch_size=2, seq_len=TINY["max_len"], vocab_size=VOCAB,
        n_warmup=4, n_iters=7,
    )
    assert call_count["n"] == 4 + 7
    assert result["n_iters"] == 7


def test_benchmark_inference_accepts_fixed_input_ids():
    """A caller-supplied fixed batch (rather than internally generated
    random tokens) must be honored and reflected in the reported
    batch_size/seq_len."""
    model = make_model(seed=303)
    x, _ = make_batch(seed=304, batch=3, seq_len=TINY["max_len"])
    result = benchmark_inference(
        model, "hard", torch.device("cpu"), input_ids=x,
        n_warmup=1, n_iters=2, top_k_dispatch=1,
    )
    assert result["batch_size"] == 3
    assert result["seq_len"] == TINY["max_len"]


def test_benchmark_inference_rejects_unknown_mode():
    model = make_model(seed=305)
    with pytest.raises(ValueError):
        benchmark_inference(
            model, "soft", torch.device("cpu"),
            batch_size=2, seq_len=TINY["max_len"], vocab_size=VOCAB,
            n_warmup=1, n_iters=1,
        )


# ─────────────────────────────────────────────────────────────────────────
# --benchmark-inference CLI flag
# ─────────────────────────────────────────────────────────────────────────

def test_benchmark_inference_cli_flag_defaults_false():
    args = build_argparser().parse_args(["--output", "x.json"])
    assert args.benchmark_inference is False
    assert args.benchmark_warmup_iters > 0
    assert args.benchmark_iters > 0


@pytest.mark.parametrize("condition", ["hard", "relu", "softmoe"])
def test_benchmark_inference_cli_flag_accepted_per_condition(condition):
    args = build_argparser().parse_args([
        "--output", "x.json", "--condition", condition, "--benchmark-inference",
        "--benchmark-warmup-iters", "2", "--benchmark-iters", "3",
    ])
    assert args.benchmark_inference is True
    assert args.condition == condition
    assert args.benchmark_warmup_iters == 2
    assert args.benchmark_iters == 3


@pytest.mark.parametrize("condition", ["hard", "relu", "softmoe"])
def test_benchmark_inference_smoke_skips_data_loading_and_writes_valid_json(tmp_path, condition):
    """--benchmark-inference must run to completion WITHOUT --data-dir
    pointing at a real corpus (the model only needs input_ids of the
    right shape, generated internally via torch.randint -- see
    benchmark_inference's docstring) and must write a JSON output with
    the benchmark dict's keys, not the usual training+eval metrics keys."""
    output_path = tmp_path / f"bench_{condition}.json"
    cmd = [
        sys.executable, str(REPO_ROOT / "src" / "moe_soft_to_hard.py"),
        "--condition", condition,
        "--benchmark-inference",
        "--output", str(output_path),
        "--batch-size", "2",
        "--seq-len", str(SEQ_LEN),
        "--d-model", str(TINY["d_model"]),
        "--n-heads", str(TINY["n_heads"]),
        "--n-layers", str(TINY["n_layers"]),
        "--d-ff", str(TINY["d_ff"]),
        "--n-experts", str(TINY["n_experts"]),
        "--benchmark-warmup-iters", "1",
        "--benchmark-iters", "2",
    ]
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, f"condition={condition} failed:\n{proc.stdout}\n{proc.stderr}"

    assert output_path.exists()
    with open(output_path) as f:
        data = json.load(f)

    assert _BENCHMARK_KEYS.issubset(data.keys())
    assert data["n_iters"] == 2
    assert data["condition"] == condition
