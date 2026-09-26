"""Numerical-equivalence tests for the flex_attention integration in
sparse_attention_hierarchical.py.

Context: HierarchicalSparseAttention has two code paths for every attention
mode -- a flex_attention (block-sparse) path and a dense-SDPA fallback path.
The whole point of the flex path is speed/memory; if it does not compute the
SAME thing as the dense path it is worthless (or worse, silently wrong).
These tests force the flex path to execute even though we are on CPU (the
production `_has_flex = _use_flex and x.is_cuda` gate normally disables it
off-GPU) by calling the private mask/score-mod builders directly and running
`flex_attention` eagerly (uncompiled). torch.compile(flex_attention) itself
requires a GPU backend (verified: `torch.compile` on the CPU backend raises
NotImplementedError for the flex_attention Triton lowering), so:

  * Mask construction + eager numerical correctness: VERIFIED HERE, on CPU.
  * The compiled (torch.compile) kernel path, and therefore the actual
    speed/memory win and the specific silent-fallback failure mode this
    file's module-level flex_attention re-init fix addresses: UNVERIFIED
    locally -- there is no CUDA on this machine. Must be confirmed on a GPU
    box before trusting the speed claims.

Run: PYTHONPATH=src pytest tests/test_flex_attention_equivalence.py -q
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

SRC = str(Path(__file__).resolve().parents[1] / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import sparse_attention_hierarchical as sah  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

FLEX_OK = sah._init_flex_attention()
requires_flex = pytest.mark.skipif(
    not FLEX_OK,
    reason=f"flex_attention unavailable in this env: {sah._FLEX_IMPORT_ERROR}",
)


def _make_attn(block_size=32, local_window=64, top_k_blocks=4,
               d_model=64, n_heads=2, d_gate=16, seed=0):
    torch.manual_seed(seed)
    attn = sah.HierarchicalSparseAttention(
        d_model=d_model, n_heads=n_heads, d_gate=d_gate,
        block_size=block_size, top_k_blocks=top_k_blocks,
        local_window=local_window, dropout=0.0,
    )
    attn.eval()
    return attn


def _causal_mask(T):
    m = torch.zeros(1, 1, T, T)
    m.masked_fill_(~torch.ones(T, T, dtype=torch.bool).tril(), float("-inf"))
    return m


def _random_causal_block_selection(B, H, n_blocks, k, seed=1):
    torch.manual_seed(seed)
    sel = torch.zeros(B, H, n_blocks, n_blocks)
    for i in range(n_blocks):
        n_avail = i + 1
        perm = torch.randperm(n_avail)
        idx = perm[: min(k, n_avail)]
        sel[:, :, i, idx] = 1.0
    return sel


# ---------------------------------------------------------------------------
# 1. Hard / forced_block_selection path (CLHR + eval usage)
# ---------------------------------------------------------------------------

@requires_flex
@pytest.mark.parametrize("block_size,local_window,T,top_k", [
    (32, 64, 256, 4),
    (64, 128, 320, 3),   # T not a multiple of the flex kernel BLOCK_SIZE=128
    (32, 256, 512, 8),
])
def test_hard_block_selection_flex_matches_dense(block_size, local_window, T,
                                                  top_k):
    d_model, n_heads, d_head = 64, 2, 32
    attn = _make_attn(block_size=block_size, local_window=local_window,
                       top_k_blocks=top_k, d_model=d_model, n_heads=n_heads)
    B = 2
    n_blocks = (T + block_size - 1) // block_size
    sel = _random_causal_block_selection(B, n_heads, n_blocks, top_k)

    x = torch.randn(B, T, d_model)
    rope = sah.RotaryPositionalEncoding(d_head, max_seq_len=T)
    causal_mask = _causal_mask(T)

    with torch.no_grad():
        out_dense, _ = attn(x, rope, causal_mask, forced_block_selection=sel,
                             _use_flex=False)

        bm = attn._create_hard_block_mask_flex(sel, T, B, torch.device("cpu"))
        q = attn.W_q(x).view(B, T, n_heads, d_head).transpose(1, 2)
        k = attn.W_k(x).view(B, T, n_heads, d_head).transpose(1, 2)
        v = attn.W_v(x).view(B, T, n_heads, d_head).transpose(1, 2)
        q, k = rope(q, T), rope(k, T)
        out_flex_raw = sah._FLEX_ATTN_RAW(q, k, v, block_mask=bm)
        out_flex = attn.W_o(
            out_flex_raw.transpose(1, 2).contiguous().view(B, T, d_model)
        )

    torch.testing.assert_close(out_dense, out_flex, rtol=1e-4, atol=1e-5)


# ---------------------------------------------------------------------------
# 1b. Hard / forced_block_selection at (near-)production shape.
#
# The T=256/320/512 cases above are the *only* forced_block_selection
# coverage that existed before this fix, and small T picks small
# FlexAttention tile configs -- it never exercises the tile sizes that
# triggered the live-L40S shared-memory OOM ("No valid triton configs.",
# Required: 114,688 B > Ada's 101,376 B/SM limit) at T=2048. These two
# tests close that hole: one at the exact production shape
# (d_model=1024, n_heads=16, d_head=64, T=2048, block_size=32,
# top_k_blocks=16, local_window=256), and one at a smaller T that still
# uses production n_heads/d_head so tile selection is representative even
# on machines where a full T=2048 CUDA run is impractical.
#
# Both are GPU-only: `HierarchicalSparseAttention.forward` gates the flex
# branch on `x.is_cuda`, and torch.compile(flex_attention) itself requires
# a GPU backend, so there is nothing meaningful to test on CPU/MPS here
# (unlike the mask/score_mod builders above, which are exercised via the
# eager _FLEX_ATTN_RAW call and don't need torch.compile at all). Skipped
# cleanly without CUDA -- UNVERIFIED on this machine (macOS, no CUDA);
# the kernel_options ladder's actual GPU behavior can only be confirmed on
# an L40S.
# ---------------------------------------------------------------------------

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="flex_attention's compiled/kernel_options path requires CUDA; "
           "meaningless on CPU/MPS since forward() gates the flex branch "
           "on x.is_cuda.",
)


@requires_flex
@requires_cuda
def test_hard_block_selection_flex_matches_dense_production_shape():
    """Exact production training shape: T=2048, block_size=32,
    top_k_blocks=16, local_window=256, d_head=64 (d_model=1024,
    n_heads=16) -- the shape that OOM'd on a live L40S run before the
    kernel_options fallback ladder was added.
    """
    d_model, n_heads, d_head = 1024, 16, 64
    block_size, local_window, top_k, T = 32, 256, 16, 2048
    attn = _make_attn(block_size=block_size, local_window=local_window,
                       top_k_blocks=top_k, d_model=d_model, n_heads=n_heads,
                       d_gate=32).cuda()
    B = 2
    n_blocks = (T + block_size - 1) // block_size
    sel = _random_causal_block_selection(B, n_heads, n_blocks, top_k).cuda()

    x = torch.randn(B, T, d_model, device="cuda")
    rope = sah.RotaryPositionalEncoding(d_head, max_seq_len=T).to(x.device)
    causal_mask = _causal_mask(T).cuda()

    with torch.no_grad():
        out_dense, _ = attn(x, rope, causal_mask, forced_block_selection=sel,
                             _use_flex=False)
        out_flex, _ = attn(x, rope, causal_mask, forced_block_selection=sel,
                            _use_flex=True)

    torch.testing.assert_close(out_dense, out_flex, rtol=1e-4, atol=1e-5)


@requires_flex
@requires_cuda
def test_hard_block_selection_flex_matches_dense_production_heads_small_t():
    """Smaller T (512) but production n_heads/d_head (16 heads, d_head=64),
    so tile selection is representative of the real model even when a full
    T=2048 run is too slow/expensive for routine CI.
    """
    d_model, n_heads, d_head = 1024, 16, 64
    block_size, local_window, top_k, T = 32, 128, 8, 512
    attn = _make_attn(block_size=block_size, local_window=local_window,
                       top_k_blocks=top_k, d_model=d_model, n_heads=n_heads,
                       d_gate=32).cuda()
    B = 2
    n_blocks = (T + block_size - 1) // block_size
    sel = _random_causal_block_selection(B, n_heads, n_blocks, top_k).cuda()

    x = torch.randn(B, T, d_model, device="cuda")
    rope = sah.RotaryPositionalEncoding(d_head, max_seq_len=T).to(x.device)
    causal_mask = _causal_mask(T).cuda()

    with torch.no_grad():
        out_dense, _ = attn(x, rope, causal_mask, forced_block_selection=sel,
                             _use_flex=False)
        out_flex, _ = attn(x, rope, causal_mask, forced_block_selection=sel,
                            _use_flex=True)

    torch.testing.assert_close(out_dense, out_flex, rtol=1e-4, atol=1e-5)


# ---------------------------------------------------------------------------
# 1c. kernel_options ladder -- CPU-testable logic (classification + caching).
#
# The ladder's *effect* on an actual Triton compile is GPU-only (see above),
# but its decision logic -- which exceptions count as recoverable, and
# that a working config is cached and not re-derived every call -- is pure
# Python and fully testable on CPU by monkeypatching `_FLEX_ATTN_COMPILED`.
# ---------------------------------------------------------------------------

def test_flex_ladder_recoverable_error_classification():
    recoverable = [
        RuntimeError("No valid triton configs."),
        RuntimeError("out of resource: triton_tem_fused_flex_attention_0"),
        RuntimeError("Required: 114688  Hardware limit: 101376"),
        RuntimeError("ran out of shared memory"),
    ]
    fatal = [
        RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"),
        ValueError("mismatched shape"),
        RuntimeError("some unrelated dynamo guard failure"),
    ]
    for exc in recoverable:
        assert sah._is_recoverable_flex_config_error(exc), exc
    for exc in fatal:
        assert not sah._is_recoverable_flex_config_error(exc), exc


def test_flex_ladder_advances_on_recoverable_error_and_caches_result(
    monkeypatch,
):
    """First two candidates raise a simulated shared-memory error; the
    third succeeds. Verify the ladder advances (does not fall back to
    dense) and that the winning index is cached so a second call with the
    same shape/context goes straight to it instead of re-walking the
    ladder.
    """
    monkeypatch.setattr(sah, "_FLEX_WORKING_LADDER_INDEX", {})
    monkeypatch.setattr(sah, "_FLEX_LADDER_ADVANCE_WARNED", set())

    calls = []

    def fake_compiled(q, k, v, **kwargs):
        calls.append(kwargs.get("kernel_options"))
        if len(calls) <= 2:
            raise RuntimeError("No valid triton configs. out of resource")
        return q  # sentinel "success"

    monkeypatch.setattr(sah, "_FLEX_ATTN_COMPILED", fake_compiled)

    q = torch.randn(1, 2, 8, 4)
    k = torch.randn(1, 2, 8, 4)
    v = torch.randn(1, 2, 8, 4)

    out = sah._flex_attention_call_with_ladder("unit_test_ctx", q, k, v)
    assert out is q
    assert len(calls) == 3  # advanced past 2 failing candidates

    key = sah._flex_config_cache_key("unit_test_ctx", q)
    assert sah._FLEX_WORKING_LADDER_INDEX[key] == 2

    # Second call with the same shape/context must go straight to the
    # cached (working) index -- exactly one call, no re-walking.
    out2 = sah._flex_attention_call_with_ladder("unit_test_ctx", q, k, v)
    assert out2 is q
    assert len(calls) == 4


def test_flex_ladder_reraises_fatal_error_without_advancing(monkeypatch):
    monkeypatch.setattr(sah, "_FLEX_WORKING_LADDER_INDEX", {})
    monkeypatch.setattr(sah, "_FLEX_LADDER_ADVANCE_WARNED", set())

    def fake_compiled(q, k, v, **kwargs):
        raise ValueError("mismatched shape, not a resource error")

    monkeypatch.setattr(sah, "_FLEX_ATTN_COMPILED", fake_compiled)

    q = torch.randn(1, 2, 8, 4)
    k = torch.randn(1, 2, 8, 4)
    v = torch.randn(1, 2, 8, 4)

    with pytest.raises(ValueError, match="mismatched shape"):
        sah._flex_attention_call_with_ladder("unit_test_ctx2", q, k, v)


def test_flex_ladder_exhausted_raises_last_error(monkeypatch):
    monkeypatch.setattr(sah, "_FLEX_WORKING_LADDER_INDEX", {})
    monkeypatch.setattr(sah, "_FLEX_LADDER_ADVANCE_WARNED", set())

    def fake_compiled(q, k, v, **kwargs):
        raise RuntimeError("No valid triton configs. out of resource")

    monkeypatch.setattr(sah, "_FLEX_ATTN_COMPILED", fake_compiled)

    q = torch.randn(1, 2, 8, 4)
    k = torch.randn(1, 2, 8, 4)
    v = torch.randn(1, 2, 8, 4)

    with pytest.raises(RuntimeError, match="No valid triton configs"):
        sah._flex_attention_call_with_ladder("unit_test_ctx3", q, k, v)


# ---------------------------------------------------------------------------
# 2. Soft training path (score_mod)
# ---------------------------------------------------------------------------

@requires_flex
def test_soft_path_flex_matches_dense():
    d_model, n_heads, d_head, d_gate = 64, 2, 32, 16
    block_size, local_window, T = 32, 64, 256
    attn = _make_attn(block_size=block_size, local_window=local_window,
                       d_model=d_model, n_heads=n_heads, d_gate=d_gate)
    B = 2
    x = torch.randn(B, T, d_model)
    rope = sah.RotaryPositionalEncoding(d_head, max_seq_len=T)
    causal_mask = _causal_mask(T)

    with torch.no_grad():
        out_dense, block_scores = attn(x, rope, causal_mask, _use_flex=False)

        soft_block_mask = torch.sigmoid(block_scores)
        bs, w = block_size, local_window

        def score_mod(score, b, h, q_idx, kv_idx):
            q_blk, k_blk = q_idx // bs, kv_idx // bs
            is_local = (kv_idx >= q_idx - w + 1) & (kv_idx <= q_idx)
            bias = torch.where(
                is_local, torch.zeros_like(score),
                torch.log(soft_block_mask[b, h, q_blk, k_blk].clamp(min=1e-6)),
            )
            return score + bias

        bm = attn._get_causal_block_mask_flex(T, torch.device("cpu"))
        q = attn.W_q(x).view(B, T, n_heads, d_head).transpose(1, 2)
        k = attn.W_k(x).view(B, T, n_heads, d_head).transpose(1, 2)
        v = attn.W_v(x).view(B, T, n_heads, d_head).transpose(1, 2)
        q, k = rope(q, T), rope(k, T)
        out_flex_raw = sah._FLEX_ATTN_RAW(q, k, v, score_mod=score_mod,
                                          block_mask=bm)
        out_flex = attn.W_o(
            out_flex_raw.transpose(1, 2).contiguous().view(B, T, d_model)
        )

    torch.testing.assert_close(out_dense, out_flex, rtol=1e-4, atol=1e-5)


@requires_flex
def test_causal_block_mask_cache_is_broadcastable_across_batch():
    """_get_causal_block_mask_flex caches by (T, device) only -- verify the
    resulting BlockMask (B=H=None) actually broadcasts across arbitrary
    batch/head counts, since two different eval/train batch sizes reuse the
    same cache entry."""
    attn = _make_attn()
    T = 128
    bm = attn._get_causal_block_mask_flex(T, torch.device("cpu"))
    for B, H in [(1, 2), (4, 2), (3, 5)]:
        q = torch.randn(B, H, T, 8)
        k = torch.randn(B, H, T, 8)
        v = torch.randn(B, H, T, 8)
        with torch.no_grad():
            out = sah._FLEX_ATTN_RAW(q, k, v, block_mask=bm)
        assert out.shape == (B, H, T, 8)
    # same object returned on second call (actually cached, not rebuilt)
    assert attn._get_causal_block_mask_flex(T, torch.device("cpu")) is bm


# ---------------------------------------------------------------------------
# 3. Dense mode -- flex is never used regardless of _use_flex
# ---------------------------------------------------------------------------

def test_dense_mode_ignores_use_flex_flag():
    attn = _make_attn()
    T, d_model = 128, 64
    x = torch.randn(2, T, d_model)
    rope = sah.RotaryPositionalEncoding(d_model // attn.n_heads, max_seq_len=T)
    causal_mask = _causal_mask(T)
    with torch.no_grad():
        out_a, _ = attn(x, rope, causal_mask, dense_mode=True, _use_flex=True)
        out_b, _ = attn(x, rope, causal_mask, dense_mode=True, _use_flex=False)
    torch.testing.assert_close(out_a, out_b)


# ---------------------------------------------------------------------------
# 4. CLHR closed-loop forward -- integration smoke test (post-refactor)
# ---------------------------------------------------------------------------

def _build_hard_token_mask_reference(
    block_mask: torch.Tensor,
    local_mask: torch.Tensor,
    block_size: int,
    T: int,
) -> torch.Tensor:
    """OLD dense implementation kept for equivalence testing ONLY.

    Removed from production (`src/sparse_attention_hierarchical.py`,
    2026-09-20) as inert dead code: it built a dense (B, H, T, T) binary
    token-level mask via `repeat_interleave` + `torch.where` -- the same
    O(T^2)-memory shape as the `_make_local_window_mask_bool` bug fixed
    elsewhere in this session -- and had zero production callers. It is
    NOT used by any live code path; the production CLHR path builds hard
    masks via `build_block_mask_direct` (src/flex_block_mask.py) instead,
    which never materializes a dense (T, T) tensor. Kept here ONLY as an
    independent, from-scratch oracle for the equivalence check below, which
    intentionally never calls production mask-construction helpers so it can
    catch the flex path silently computing something different.

    Args:
        block_mask: (B, H, n_blocks, n_blocks) binary float.
        local_mask: (1, 1, T, T) bool -- local window.
        block_size: tokens per block.
        T: sequence length.

    Returns:
        token_mask: (B, H, T, T) binary float -- union of block + local.
    """
    token_block = block_mask.repeat_interleave(
        block_size, dim=2,
    ).repeat_interleave(block_size, dim=3)[:, :, :T, :T]
    combined = torch.where(
        local_mask | (token_block > 0),
        torch.ones_like(token_block),
        torch.zeros_like(token_block),
    )
    return combined


def test_contemporary_clhr_forward_runs_and_matches_dense_reimplementation():
    """After refactoring _contemporary_clhr_forward's Phase 1 rollout to call
    attn(...) instead of a hand-rolled dense SDPA, confirm the loss is
    unchanged relative to a from-scratch dense-only reimplementation of the
    same closed-loop logic (guards against the refactor changing semantics).
    """
    torch.manual_seed(0)
    cfg = dict(vocab_size=50, d_model=32, n_heads=2, n_layers=2, d_ff=64,
               d_gate=8)
    model = sah.HierarchicalSparseTransformer(
        **cfg, block_size=8, top_k_blocks=2, local_window=8, max_seq_len=32,
        dropout=0.0,
    )
    model.eval()
    B, T = 2, 32
    x = torch.randint(0, cfg["vocab_size"], (B, T))
    y = torch.randint(0, cfg["vocab_size"], (B, T))
    device = torch.device("cpu")

    loss = sah._contemporary_clhr_forward(model, x, y, device, torch.float32)
    assert torch.isfinite(loss)

    # Independent dense reimplementation of the same closed-loop rollout +
    # forced-mask forward, without going through attn.forward at all.
    block_size, top_k = model.block_size, model.top_k_blocks
    n_blocks = (T + block_size - 1) // block_size
    local_mask = sah._make_local_window_mask_bool(T, model.local_window, device)
    causal_mask = model._make_causal_mask(T, device)
    block_causal = torch.ones(n_blocks, n_blocks, dtype=torch.bool).tril()

    with torch.no_grad():
        h = model.drop(model.tok_emb(x))
        selections = []
        for layer in model.layers:
            normed = layer.attn_norm(h)
            attn = layer.attn
            bscores = attn.block_gate.compute_block_scores(normed, n_blocks)
            _, idx = torch.topk(bscores, min(top_k, n_blocks), dim=-1)
            hard_block = torch.zeros_like(bscores).scatter_(-1, idx, 1.0)
            hard_block = hard_block * block_causal.float()
            selections.append(hard_block)

            hard_token = _build_hard_token_mask_reference(hard_block, local_mask,
                                                           block_size, T)
            nh, dh = attn.n_heads, attn.d_head
            q = attn.W_q(normed).view(B, T, nh, dh).transpose(1, 2)
            k = attn.W_k(normed).view(B, T, nh, dh).transpose(1, 2)
            v = attn.W_v(normed).view(B, T, nh, dh).transpose(1, 2)
            q, k = model.rope(q, T), model.rope(k, T)
            gate_bias = torch.where(hard_token > 0, torch.zeros_like(hard_token),
                                     torch.full_like(hard_token, float("-inf")))
            attn_mask = gate_bias + causal_mask
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
            out = attn.W_o(out.transpose(1, 2).contiguous().view(B, T, -1))
            h = h + out
            h = h + layer.ff(layer.ff_norm(h))

        logits, _ = model(x, forced_block_selections=selections,
                           use_checkpoint=False)
        ref_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                    y.reshape(-1))

    torch.testing.assert_close(loss, ref_loss, rtol=1e-4, atol=1e-5)


# ---------------------------------------------------------------------------
# 5. Loud-fallback behavior (task requirement: never fail silently)
# ---------------------------------------------------------------------------

def test_flex_unavailable_warns_loudly_and_only_once(monkeypatch):
    monkeypatch.setattr(sah, "_FLEX_ATTN_COMPILED", None)
    monkeypatch.setattr(sah, "_FLEX_IMPORT_ERROR", RuntimeError("boom-import"))
    monkeypatch.setattr(sah, "_FLEX_FALLBACK_WARNED", False)

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        sah._warn_flex_unavailable()
        sah._warn_flex_unavailable()  # second call must be a no-op

    matching = [w for w in rec if issubclass(w.category, RuntimeWarning)]
    assert len(matching) == 1, "fallback warning must fire exactly once"
    assert "boom-import" in str(matching[0].message)
    assert "RuntimeError" in str(matching[0].message)


def test_flex_runtime_failure_warns_loudly_with_real_exception(monkeypatch):
    monkeypatch.setattr(sah, "_FLEX_RUNTIME_WARNED", set())
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        sah._warn_flex_runtime_failure("unit_test_ctx", ValueError("bad shape"))
        sah._warn_flex_runtime_failure("unit_test_ctx", ValueError("bad shape"))

    matching = [w for w in rec if issubclass(w.category, RuntimeWarning)]
    assert len(matching) == 1
    assert "bad shape" in str(matching[0].message)
    assert "unit_test_ctx" in str(matching[0].message)


@requires_flex
def test_forced_block_selection_falls_back_loudly_on_runtime_error(monkeypatch):
    """Simulate a flex runtime failure (e.g. a GPU-only edge case) inside
    forward()'s forced_block_selection branch and confirm it degrades to the
    dense path with a loud warning instead of crashing or silently
    computing something else.

    `x.is_cuda` is monkeypatched to True (globally, restored automatically
    by the `monkeypatch` fixture) purely to exercise the `_has_flex` branch
    of forward() on a CPU tensor -- this does not change device placement,
    it only flips the gate so we can unit-test the try/except wiring
    without a GPU.

    Also force-enables `_FLEX_ENABLE_FORCED_SELECTION`, which ships False
    because flex measured 0.48x (2x slower, +10.7 GB) on this branch. The
    fallback wiring still has to be correct for the A/B path, so this test
    opts in rather than being deleted.
    """
    monkeypatch.setattr(sah, "_FLEX_ENABLE_FORCED_SELECTION", True)
    attn = _make_attn()
    T, d_model, n_heads = 128, 64, attn.n_heads
    d_head = d_model // n_heads
    B = 1
    n_blocks = T // attn.block_size
    sel = _random_causal_block_selection(B, n_heads, n_blocks, attn.top_k_blocks)
    x = torch.randn(B, T, d_model)
    rope = sah.RotaryPositionalEncoding(d_head, max_seq_len=T)
    causal_mask = _causal_mask(T)

    monkeypatch.setattr(sah, "_FLEX_RUNTIME_WARNED", set())
    monkeypatch.setattr(
        attn, "_create_hard_block_mask_flex",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("simulated GPU fail")),
    )
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        with torch.no_grad():
            out, _ = attn(x, rope, causal_mask, forced_block_selection=sel)

    assert torch.isfinite(out).all()
    matching = [w for w in rec if issubclass(w.category, RuntimeWarning)]
    assert len(matching) == 1
    assert "simulated GPU fail" in str(matching[0].message)
    assert "forced_block_selection" in str(matching[0].message)

    # And it must match the plain dense computation (same sel, _use_flex=False)
    with torch.no_grad():
        out_dense, _ = attn(x, rope, causal_mask, forced_block_selection=sel,
                             _use_flex=False)
    torch.testing.assert_close(out, out_dense, rtol=1e-4, atol=1e-5)


# ---------------------------------------------------------------------------
# 6. `_get_causal_block_mask_flex` no longer uses `create_block_mask`
#    (which OOMs at long context -- see build_causal_block_mask_direct in
#    src/flex_block_mask.py), the fallback is loud, and re-raises instead of
#    falling back to dense SDPA above _FLEX_DENSE_FALLBACK_MAX_T.
# ---------------------------------------------------------------------------

@requires_flex
def test_soft_path_mask_matches_previous_pattern():
    """Correctness gate: the rewritten `_get_causal_block_mask_flex` (now
    built via `build_causal_block_mask_direct`) must produce a mask_mod that
    is elementwise identical, evaluated over the full token index grid, to
    what the OLD `create_block_mask(causal_local_mask, ...)` construction
    produced. A different pattern here would silently change the attention
    semantics of every soft-path training step.
    """
    T = 256  # small enough that the OLD create_block_mask path also fits

    def causal_local_mask(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    old_bm = sah._FLEX_CREATE_BLOCK_MASK(
        causal_local_mask, None, None, T, T, device=torch.device("cpu"),
        BLOCK_SIZE=sah._FLEX_KERNEL_BLOCK_SIZE,
    )

    attn = _make_attn()
    new_bm = attn._get_causal_block_mask_flex(T, torch.device("cpu"))

    q_idx = torch.arange(T).view(T, 1).expand(T, T)
    kv_idx = torch.arange(T).view(1, T).expand(T, T)
    old_grid = old_bm.mask_mod(None, None, q_idx, kv_idx)
    new_grid = new_bm.mask_mod(None, None, q_idx, kv_idx)

    assert torch.equal(old_grid, new_grid)
    assert torch.equal(new_grid, q_idx >= kv_idx)


def test_soft_path_does_not_call_create_block_mask(monkeypatch):
    """The rewritten soft path must build its causal BlockMask without ever
    calling `create_block_mask` -- monkeypatch it to a call-counter (NOT a
    raise: an exception here would just be swallowed by the soft path's own
    `except Exception` and turned into a silent dense fallback, defeating
    the point of the check) and confirm it is never invoked, then confirm
    the soft path still produces a finite, dense-matching output (forced
    onto the flex branch via the `is_cuda` gate, exactly as in
    test_forced_block_selection_falls_back_loudly_on_runtime_error above).
    """
    calls = {"n": 0}

    def _counting(*a, **k):
        calls["n"] += 1
        raise AssertionError("create_block_mask should never be called")
    monkeypatch.setattr(sah, "_FLEX_CREATE_BLOCK_MASK", _counting)

    attn = _make_attn()
    attn._causal_block_mask_cache = {}
    T, d_model, n_heads = 128, 64, attn.n_heads
    d_head = d_model // n_heads
    B = 1
    x = torch.randn(B, T, d_model)
    rope = sah.RotaryPositionalEncoding(d_head, max_seq_len=T)
    causal_mask = _causal_mask(T)

    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))

    with torch.no_grad():
        out, block_scores = attn(x, rope, causal_mask)

    assert calls["n"] == 0, "create_block_mask must not be called by the soft path"
    assert torch.isfinite(out).all()
    assert block_scores is not None

    with torch.no_grad():
        out_dense, _ = attn(x, rope, causal_mask, _use_flex=False)
    torch.testing.assert_close(out, out_dense, rtol=1e-4, atol=1e-5)


def test_long_context_fallback_reraises(monkeypatch):
    """Above `_FLEX_DENSE_FALLBACK_MAX_T`, a flex runtime failure in the
    soft path must re-raise instead of silently falling back to dense-mask
    SDPA (which would OOM moments later at real long-context shapes).
    Overrides the threshold to a tiny value so this stays a fast CPU test.
    """
    monkeypatch.setattr(sah, "_FLEX_DENSE_FALLBACK_MAX_T", 4)
    monkeypatch.setattr(sah, "_FLEX_RUNTIME_WARNED", set())
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))

    def _boom(*a, **k):
        raise RuntimeError("simulated flex failure")
    monkeypatch.setattr(sah, "_flex_attention_call_with_ladder", _boom)

    attn = _make_attn(block_size=2, local_window=2, top_k_blocks=1)
    T, d_model, n_heads = 8, 64, attn.n_heads  # T=8 > overridden threshold=4
    d_head = d_model // n_heads
    B = 1
    x = torch.randn(B, T, d_model)
    rope = sah.RotaryPositionalEncoding(d_head, max_seq_len=T)
    causal_mask = _causal_mask(T)

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        with pytest.raises(RuntimeError) as excinfo:
            with torch.no_grad():
                attn(x, rope, causal_mask)

    assert "simulated flex failure" in str(excinfo.value)
    assert "soft_path" in str(excinfo.value) or "soft_path" in str(
        [str(w.message) for w in rec]
    )
    # still logged loudly before re-raising
    matching = [w for w in rec if issubclass(w.category, RuntimeWarning)]
    assert len(matching) == 1
    assert "simulated flex failure" in str(matching[0].message)


def test_short_context_fallback_still_works(monkeypatch):
    """Below `_FLEX_DENSE_FALLBACK_MAX_T`, a flex runtime failure in the
    soft path must still degrade to dense-mask SDPA (existing behavior,
    unaffected by the new long-context guard).
    """
    monkeypatch.setattr(sah, "_FLEX_DENSE_FALLBACK_MAX_T", 8192)
    monkeypatch.setattr(sah, "_FLEX_RUNTIME_WARNED", set())
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))

    def _boom(*a, **k):
        raise RuntimeError("simulated flex failure")
    monkeypatch.setattr(sah, "_flex_attention_call_with_ladder", _boom)

    attn = _make_attn()
    T, d_model, n_heads = 128, 64, attn.n_heads  # T well below threshold
    d_head = d_model // n_heads
    B = 1
    x = torch.randn(B, T, d_model)
    rope = sah.RotaryPositionalEncoding(d_head, max_seq_len=T)
    causal_mask = _causal_mask(T)

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        with torch.no_grad():
            out, _ = attn(x, rope, causal_mask)

    assert torch.isfinite(out).all()
    matching = [w for w in rec if issubclass(w.category, RuntimeWarning)]
    assert len(matching) == 1
    assert "simulated flex failure" in str(matching[0].message)
    assert "soft_path" in str(matching[0].message)

    with torch.no_grad():
        out_dense, _ = attn(x, rope, causal_mask, _use_flex=False)
    torch.testing.assert_close(out, out_dense, rtol=1e-4, atol=1e-5)


@requires_flex
def test_soft_mask_cached(monkeypatch):
    """The causal BlockMask is built once per (T, device), not once per
    forward call -- monkeypatch `build_causal_block_mask_direct` to count
    invocations and drive two forward calls at the same T.
    """
    calls = {"n": 0}
    real = sah.build_causal_block_mask_direct

    def _counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)
    monkeypatch.setattr(sah, "build_causal_block_mask_direct", _counting)

    attn = _make_attn()
    attn._causal_block_mask_cache = {}
    T, d_model, n_heads = 128, 64, attn.n_heads
    d_head = d_model // n_heads
    B = 1
    x = torch.randn(B, T, d_model)
    rope = sah.RotaryPositionalEncoding(d_head, max_seq_len=T)
    causal_mask = _causal_mask(T)
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))

    with torch.no_grad():
        attn(x, rope, causal_mask)
        attn(x, rope, causal_mask)

    assert calls["n"] == 1, (
        "build_causal_block_mask_direct must be called once per (T, device), "
        f"not once per forward() call; got {calls['n']} calls"
    )
    # direct check on the helper too
    bm1 = attn._get_causal_block_mask_flex(T, torch.device("cpu"))
    bm2 = attn._get_causal_block_mask_flex(T, torch.device("cpu"))
    assert bm1 is bm2
    assert calls["n"] == 1
