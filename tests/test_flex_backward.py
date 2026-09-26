"""Feasibility probe: does flex_attention's BACKWARD pass work with the
BlockMask produced by build_block_mask_direct (scripts/bench_flex_longcontext.py)?

Gating question for long-context (32K+) training: training currently
emulates block sparsity as a dense (B,H,T,T) additive bias, which costs
16 GB/layer at T=32768 and ~400h/run. If flex_attention's backward works
with our directly-assembled BlockMask, that cost collapses to the
selected-blocks cost. If backward does not work, 32K training via flex is
off the table and we report a boundary -- no workaround is attempted here.

This is a pure feasibility probe. It does NOT modify scripts/
bench_flex_longcontext.py, src/, or predictions/ -- it only imports
build_block_mask_direct and exercises flex_attention's backward path
directly.

Environment (OBSERVED, this run): torch 2.13.0, CPU-only (torch.cuda.
is_available() is False; MPS only). flex_attention on CPU uses an
unfused/eager path (torch warns about this) -- see the CPU-vs-CUDA caveat
in each test's docstring and in the module-level summary at the bottom of
this file. A CPU PASS here does NOT establish that the compiled Triton
backward kernel on CUDA works; only a GPU run can establish that. A CPU
FAIL, however, IS sufficient to establish that backward does not work
in general (a bug that reproduces on the simpler eager/CPU path will not
somehow vanish on GPU).

Run: uv run pytest tests/test_flex_backward.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

SCRIPTS = str(Path(__file__).resolve().parents[1] / "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

import bench_flex_longcontext as bfl  # noqa: E402

from torch.nn.attention.flex_attention import flex_attention  # noqa: E402

# ---------------------------------------------------------------------------
# Tiny fixed sizes so this runs on CPU in seconds (per task spec).
# ---------------------------------------------------------------------------
T = 128
BLOCK_SIZE = 32
LOCAL_WINDOW = 32
TOP_K = 2
N_HEADS = 2
D_HEAD = 16
BATCH = 1
NUM_BLOCKS = T // BLOCK_SIZE  # 4


def _selection(seed=0, n_heads=N_HEADS, num_blocks=NUM_BLOCKS, top_k=TOP_K):
    gen = torch.Generator().manual_seed(seed)
    return bfl.generate_block_selection(n_heads, num_blocks, num_blocks, top_k,
                                         generator=gen)


def _qkv(seed=0, requires_grad=True, n_heads=N_HEADS, seq_len=T, d_head=D_HEAD,
         batch=BATCH):
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(batch, n_heads, seq_len, d_head, generator=gen, dtype=torch.float32)
    k = torch.randn(batch, n_heads, seq_len, d_head, generator=gen, dtype=torch.float32)
    v = torch.randn(batch, n_heads, seq_len, d_head, generator=gen, dtype=torch.float32)
    if requires_grad:
        q.requires_grad_(True)
        k.requires_grad_(True)
        v.requires_grad_(True)
    return q, k, v


def _assert_finite_grad(name, grad):
    assert grad is not None, f"{name}.grad is None -- backward did not populate it"
    assert torch.isfinite(grad).all(), f"{name}.grad contains NaN/Inf: {grad}"


# ---------------------------------------------------------------------------
# CATEGORICAL FINDING (discovered running this file, torch 2.13.0, CPU):
# torch.nn.attention.flex_attention._validate_device (flex_attention.py,
# lines ~2075-2084) unconditionally raises
#   NotImplementedError("FlexAttention does not support backward on CPU.
#   Please set the input requires_grad to False or use another device.")
# whenever query/key/value are on device.type == "cpu" (or "mps") AND ANY
# of them requires_grad -- i.e. it fires on the FORWARD call itself, before
# _validate_no_nested_tensors's sibling checks reach anything BlockMask- or
# kernel-related. This is unconditional on the BlockMask: it fires
# regardless of whether build_block_mask_direct's output is valid, whether
# cardinality is uniform, or whether the call is eager or torch.compile'd
# (compile still executes the eager wrapper's validation first). Verified
# by reading the guard directly -- not inferred from a single exception
# string.
#
# Practical consequence: NO test of flex_attention's BACKWARD pass can run
# on this machine (CPU, no CUDA; torch.cuda.is_available() is False) or on
# any CPU/MPS-only machine, regardless of BlockMask correctness. The tests
# below assert that this exact, deterministic guard fires (documenting the
# boundary), and SKIP -- rather than fail -- because a raised
# NotImplementedError here is an expected, torch-source-confirmed property
# of this environment, not evidence about whether our BlockMask itself
# supports backward. That question (the one this file was written to
# answer) is UNRESOLVED and requires a CUDA machine.
# ---------------------------------------------------------------------------
_CPU_BACKWARD_MSG = (
    "FlexAttention does not support backward on CPU. Please set the input "
    "requires_grad to False or use another device."
)


def _run_and_classify(fn):
    """Run fn() (a zero-arg callable performing forward+backward). Returns
    ("cpu_device_guard", exc) if the exact, unconditional torch device
    guard fired; re-raises anything else (a genuine BlockMask/kernel
    failure, which IS this file's deliverable if it occurs)."""
    try:
        fn()
    except NotImplementedError as exc:
        if str(exc) == _CPU_BACKWARD_MSG:
            return "cpu_device_guard", exc
        raise
    return "ran", None


# ---------------------------------------------------------------------------
# 1. Does eager backward run at all? THE gating question.
# ---------------------------------------------------------------------------
def test_backward_runs_at_all():
    sel = _selection(seed=0)
    block_mask = bfl.build_block_mask_direct(sel, BLOCK_SIZE, LOCAL_WINDOW, T, "cpu")
    q, k, v = _qkv(seed=1)

    def _do():
        out = flex_attention(q, k, v, block_mask=block_mask)
        out.sum().backward()

    outcome, exc = _run_and_classify(_do)
    if outcome == "cpu_device_guard":
        pytest.skip(
            "torch's own _validate_device unconditionally blocks flex_attention "
            f"backward on CPU (verbatim: {exc}); this is a device restriction, "
            "not a BlockMask/kernel result -- CUDA is required to answer the "
            "gating question. See module docstring."
        )

    _assert_finite_grad("q", q.grad)
    _assert_finite_grad("k", k.grad)
    _assert_finite_grad("v", v.grad)


# ---------------------------------------------------------------------------
# 2. Do the gradients match a dense SDPA reference using the SAME boolean
#    mask (as an additive -inf bias)? A backward that runs but computes
#    wrong gradients is worse than one that fails loudly.
# ---------------------------------------------------------------------------
def test_gradients_match_dense_reference():
    sel = _selection(seed=0)
    block_mask = bfl.build_block_mask_direct(sel, BLOCK_SIZE, LOCAL_WINDOW, T, "cpu")
    mask_mod = block_mask.mask_mod

    # Same random loss-generating vector for both paths (identical scalar
    # loss functional: sum(out * w)), same q/k/v values, independent leaf
    # tensors so grad accumulation cannot leak between the two paths.
    q_flex, k_flex, v_flex = _qkv(seed=2)
    q_ref, k_ref, v_ref = _qkv(seed=2)
    assert torch.equal(q_flex, q_ref) and torch.equal(k_flex, k_ref) and torch.equal(v_flex, v_ref)

    gen = torch.Generator().manual_seed(3)
    w = torch.randn(BATCH, N_HEADS, T, D_HEAD, generator=gen)

    _flex_out_holder = {}

    def _do_flex():
        out = flex_attention(q_flex, k_flex, v_flex, block_mask=block_mask)
        (out * w).sum().backward()
        _flex_out_holder["out"] = out

    outcome, exc = _run_and_classify(_do_flex)
    if outcome == "cpu_device_guard":
        pytest.skip(
            "torch's own _validate_device unconditionally blocks flex_attention "
            f"backward on CPU (verbatim: {exc}); cannot compare gradients to the "
            "dense reference without a CUDA machine. See module docstring."
        )

    # Build the additive -inf bias from the SAME mask_mod the BlockMask
    # itself carries (block_mask.mask_mod), evaluated densely per head --
    # i.e. provably the same pattern, not a re-derived one.
    bias = torch.zeros(1, N_HEADS, T, T, dtype=torch.float32)
    for h in range(N_HEADS):
        dense = bfl.dense_mask_from_mask_mod(mask_mod, T, h)
        bias[0, h][~dense] = float("-inf")

    out_ref = F.scaled_dot_product_attention(q_ref, k_ref, v_ref, attn_mask=bias)
    (out_ref * w).sum().backward()

    max_abs = {}
    max_rel = {}
    for name, g_flex, g_ref in (("q", q_flex.grad, q_ref.grad),
                                 ("k", k_flex.grad, k_ref.grad),
                                 ("v", v_flex.grad, v_ref.grad)):
        _assert_finite_grad(f"flex.{name}", g_flex)
        _assert_finite_grad(f"ref.{name}", g_ref)
        diff = (g_flex - g_ref).abs()
        max_abs[name] = float(diff.max())
        max_rel[name] = float((diff / g_ref.abs().clamp_min(1e-8)).max())
        assert torch.allclose(g_flex, g_ref, atol=1e-3, rtol=1e-2), (
            f"{name}.grad mismatch: max_abs={max_abs[name]}, max_rel={max_rel[name]}"
        )

    # Surface the exact numbers regardless of pass/fail (captured by -s / on
    # failure by pytest's own capture).
    print(f"max_abs_deviation={max_abs} max_rel_deviation={max_rel}")


# ---------------------------------------------------------------------------
# 3. Are gradients zero for a key position the mask excludes for EVERY
#    query? Confirms sparsity is respected in backward, not just forward.
# ---------------------------------------------------------------------------
def test_gradients_zero_where_masked():
    # All-empty selection (no extra selected blocks) so the only allowed
    # pairs are causal + local-window. Key block NUM_BLOCKS - 1 (the last
    # block, tokens [T - BLOCK_SIZE, T - 1]) is excluded for every query
    # strictly outside the causal+local reach of that block -- but since
    # causal_block requires kv_idx <= q_idx, and local_window < block_size
    # doesn't hold here, easier: pick the FIRST key block for the LAST
    # query, then check the mask_mod on a token whose block distance is
    # provably outside both causal validity from earlier queries and the
    # local window for ALL queries: use a fully empty selection and a query
    # range confined to block 0, so key block NUM_BLOCKS - 1 is excluded
    # for every query (kv_idx > q_idx violates causality for those pairs).
    sel = torch.zeros(N_HEADS, NUM_BLOCKS, NUM_BLOCKS, dtype=torch.bool)
    block_mask = bfl.build_block_mask_direct(sel, BLOCK_SIZE, LOCAL_WINDOW, T, "cpu")
    mask_mod = block_mask.mask_mod

    # Independently confirm, via the mask itself, that key block
    # (NUM_BLOCKS - 1) is excluded for every query in query block 0 (purely
    # by causality: kv_idx in the last block is > any q_idx in block 0).
    excluded_key_block = NUM_BLOCKS - 1
    key_lo = excluded_key_block * BLOCK_SIZE
    key_hi = key_lo + BLOCK_SIZE
    for h in range(N_HEADS):
        dense = bfl.dense_mask_from_mask_mod(mask_mod, T, h)
        assert not dense[0:BLOCK_SIZE, key_lo:key_hi].any(), (
            "test setup invalid: excluded key block is not actually excluded "
            "for query block 0 under this mask"
        )

    q, k, v = _qkv(seed=4)

    def _do():
        out = flex_attention(q, k, v, block_mask=block_mask)
        # Loss depends only on queries in block 0, so any nonzero grad
        # reaching k/v in the excluded key block would have to flow through
        # an admitted (query, key) pair -- which, per the check above, does
        # not exist.
        out[:, :, 0:BLOCK_SIZE, :].sum().backward()

    outcome, exc = _run_and_classify(_do)
    if outcome == "cpu_device_guard":
        pytest.skip(
            "torch's own _validate_device unconditionally blocks flex_attention "
            f"backward on CPU (verbatim: {exc}); cannot check masked-gradient "
            "sparsity without a CUDA machine. See module docstring."
        )

    _assert_finite_grad("k", k.grad)
    _assert_finite_grad("v", v.grad)
    k_grad_excluded = k.grad[:, :, key_lo:key_hi, :]
    v_grad_excluded = v.grad[:, :, key_lo:key_hi, :]
    assert torch.equal(k_grad_excluded, torch.zeros_like(k_grad_excluded)), (
        f"k.grad nonzero at masked-out key block: max={k_grad_excluded.abs().max()}"
    )
    assert torch.equal(v_grad_excluded, torch.zeros_like(v_grad_excluded)), (
        f"v.grad nonzero at masked-out key block: max={v_grad_excluded.abs().max()}"
    )


# ---------------------------------------------------------------------------
# 4. Backward through torch.compile(flex_attention) -- the realistic
#    training path. Forward-only compile was already confirmed working
#    (test_flex_attention_accepts_direct_blockmask_cpu et al. in
#    tests/test_flex_block_mask.py); this checks backward through the
#    compiled graph.
# ---------------------------------------------------------------------------
def test_backward_with_compile():
    sel = _selection(seed=0)
    block_mask = bfl.build_block_mask_direct(sel, BLOCK_SIZE, LOCAL_WINDOW, T, "cpu")
    q, k, v = _qkv(seed=5)

    compiled_flex = torch.compile(flex_attention)

    def _do():
        out = compiled_flex(q, k, v, block_mask=block_mask)
        out.sum().backward()

    outcome, exc = _run_and_classify(_do)
    if outcome == "cpu_device_guard":
        pytest.skip(
            "torch's own _validate_device unconditionally blocks flex_attention "
            f"backward on CPU (verbatim: {exc}) -- torch.compile still runs the "
            "eager wrapper's validation first, so compiling does not bypass it. "
            "Cannot test compiled backward without a CUDA machine."
        )

    _assert_finite_grad("q", q.grad)
    _assert_finite_grad("k", k.grad)
    _assert_finite_grad("v", v.grad)


# ---------------------------------------------------------------------------
# 5. Non-uniform cardinality: query blocks selecting DIFFERENT numbers of
#    key blocks (the causal-boundary case that broke torch's own
#    _ordered_to_dense; see build_block_mask_direct's docstring, bug 1).
#    generate_block_selection already produces this naturally -- early
#    query blocks are clamped to n_valid = min(qb+1, num_kv_blocks) < top_k.
# ---------------------------------------------------------------------------
def test_non_uniform_cardinality_backward():
    # top_k=3 with NUM_BLOCKS=4: query block 0 has only 1 causally-valid key
    # block (itself), query block 1 has 2, query blocks 2 and 3 have 3 --
    # genuinely non-uniform per-row cardinality by construction.
    sel = _selection(seed=6, top_k=3)
    per_row_counts = sel.sum(dim=-1)
    assert per_row_counts.unique().numel() > 1, (
        "test setup invalid: selection has uniform cardinality, not the "
        "non-uniform case this test is meant to exercise"
    )

    block_mask = bfl.build_block_mask_direct(sel, BLOCK_SIZE, LOCAL_WINDOW, T, "cpu")
    q, k, v = _qkv(seed=7)

    def _do():
        out = flex_attention(q, k, v, block_mask=block_mask)
        out.sum().backward()

    outcome, exc = _run_and_classify(_do)
    if outcome == "cpu_device_guard":
        pytest.skip(
            "torch's own _validate_device unconditionally blocks flex_attention "
            f"backward on CPU (verbatim: {exc}); cannot test the non-uniform-"
            "cardinality (causal-boundary) case without a CUDA machine. Note "
            "this guard fires before any BlockMask-specific code runs, so it "
            "gives no signal either way about the from_kv_blocks-adjacent bugs "
            "build_block_mask_direct was written to avoid."
        )

    _assert_finite_grad("q", q.grad)
    _assert_finite_grad("k", k.grad)
    _assert_finite_grad("v", v.grad)
