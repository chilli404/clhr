"""Regression tests for the dense-(T,T)-causal-mask OOM fixed 2026-09-20
(4th instance of the dense-tensor bug found this session).

Root cause (confirmed by reading the code, not assumed):
`HierarchicalSparseTransformer.forward` and `_contemporary_clhr_forward`
both used to build

    causal_mask = self._make_causal_mask(T, device)   # / raw_model....

UNCONDITIONALLY, before the per-layer loop -- a dense float32 (1, 1, T, T)
tensor (4.0 GiB at T=32768, 16.0 GiB at T=65536, and an outright
`torch.OutOfMemoryError` at T=131072/262144, on the very first forward
call, regardless of `--attention-impl`).

Tracing `HierarchicalSparseAttention.forward` (the only consumer) shows
`causal_mask` is read in exactly four places:
  1. `dense_mode=True` branch (direct `attn_mask=causal_mask...`)
  2. the `forced_block_selection is not None` branch's dense fallback
     (taken only when `attention_impl != "flex"`, or flex is disabled/
     fails at runtime)
  3. the legacy `forced_hard_mask` fallback (CPU / no flex_attention)
  4. the soft-training-path's dense-SDPA fallback (taken only when
     `_has_flex` is False, i.e. flex unavailable, or a flex call raised)

On the actual production hot path -- `--attention-impl flex` with
`forced_block_selection` set, which takes the
`forced_block_selection is not None and self.attention_impl == "flex"`
branch -- causality is already encoded in the flex `BlockMask` built by
`build_block_mask_direct` (src/flex_block_mask.py), and `causal_mask` is
NEVER read. The same is true of the soft path's `_has_flex` success
branch (score_mod + `_get_causal_block_mask_flex`'s cached BlockMask).
So `causal_mask` is unused on every path any real production run
(zero fallbacks observed in any log this session) actually takes.

Fix: construction moved to a lazy zero-arg callable
(`causal_mask = lambda: self._make_causal_mask(T, device)`), materialized
only at the point of use via `_materialize_causal_mask`, which is a
no-op passthrough for callers that still pass an eager Tensor (every
pre-existing test in this repo, and `dense_mode`/legacy-fallback callers
that genuinely need an eager mask).

UPDATE (same day): `eval_closed_loop_hierarchical_hard` and
`eval_random_block_hard` were initially left eager -- scoped out of this
fix on the assumption a concurrent fix to those same functions (for a
DIFFERENT dense-(T,T) bug, in `_make_local_window_mask_bool`) covered
them. It didn't: that fix's scope was the local-window mask only. Both
functions still built `causal_mask = model._make_causal_mask(T, device)`
eagerly, then passed it into the identical `forced_block_selection`/
`attention_impl == "flex"` branch already shown above to never read it.
This is the eval-path twin of the training-time bug, closed the same
day by making both call sites lazy too -- see tests 8 below.

Tests below:
  1. no dense-(>=T*T)-element tensor is allocated when the flex hot path
     is taken (mocked to succeed, since flex_attention itself requires
     CUDA -- see the module comment; this IS what's CPU-testable).
  2. `dense_mode=True` is bit-identical to the pre-fix eager behavior.
  3. the legacy `forced_hard_mask` fallback is bit-identical to the
     pre-fix eager behavior.
  4. the soft-path's dense-SDPA fallback (the branch every existing CPU
     test exercises, since `_has_flex` is always False off-CUDA) is
     bit-identical to the pre-fix eager behavior.
  5. `_materialize_causal_mask` itself: passthrough for a Tensor,
     materializes for a callable.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

SRC = str(Path(__file__).resolve().parents[1] / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import sparse_attention_hierarchical as sah  # noqa: E402


def _tiny_attn(attention_impl="bias", seed=0):
    torch.manual_seed(seed)
    return sah.HierarchicalSparseAttention(
        d_model=32, n_heads=2, d_gate=8, block_size=8, top_k_blocks=2,
        local_window=8, dropout=0.0, attention_impl=attention_impl,
    ).eval()


def _old_eager_causal_mask(T: int, device: torch.device) -> torch.Tensor:
    """Verbatim copy of `_make_causal_mask`'s pre-fix body, kept here ONLY
    for equivalence testing -- never used in production. Production now
    builds this lazily; this reference always builds it eagerly."""
    mask = torch.zeros(T, T, device=device)
    mask.masked_fill_(
        ~torch.ones(T, T, device=device, dtype=torch.bool).tril(),
        float("-inf"),
    )
    return mask.unsqueeze(0).unsqueeze(0)


# ---------------------------------------------------------------------------
# 1. flex hot path never materializes a dense (T, T)-scale causal mask
# ---------------------------------------------------------------------------

def test_flex_path_never_builds_dense_causal_mask(monkeypatch):
    """`forced_block_selection is not None and attention_impl == "flex"` is
    the real production hot path. flex_attention itself requires CUDA (see
    the module-level comment block in sparse_attention_hierarchical.py), so
    a genuine forward+backward through the compiled kernel isn't
    CPU-testable. What IS CPU-testable: this branch does not check
    `x.is_cuda` before attempting flex, so mocking `_FLEX_ATTN_COMPILED`
    and `build_block_mask_direct` to "succeed" lets us exercise the exact
    same branch on CPU and assert no tensor of size >= T*T elements is
    allocated by `torch.zeros`/`torch.empty` anywhere during the call --
    NOT verified here: real flex_attention CUDA memory/numerics, which
    requires a GPU (see scripts/verify_causal_mask_memory.py).
    """
    T = 1024
    attn = _tiny_attn(attention_impl="flex")
    n_blocks = (T + attn.block_size - 1) // attn.block_size
    B, n_h = 2, attn.n_heads

    def _fake_flex_attn(q, k, v, block_mask=None, score_mod=None, **kw):
        return q  # shape-compatible stand-in; values are irrelevant here

    monkeypatch.setattr(sah, "_FLEX_ATTN_COMPILED", _fake_flex_attn)
    monkeypatch.setattr(sah, "build_block_mask_direct",
                         lambda *a, **k: object())

    big_allocations = []
    orig_zeros = torch.zeros
    orig_empty = torch.empty

    def _tracking_zeros(*args, **kwargs):
        t = orig_zeros(*args, **kwargs)
        if t.numel() >= T * T:
            big_allocations.append(("zeros", tuple(t.shape)))
        return t

    def _tracking_empty(*args, **kwargs):
        t = orig_empty(*args, **kwargs)
        if t.numel() >= T * T:
            big_allocations.append(("empty", tuple(t.shape)))
        return t

    monkeypatch.setattr(torch, "zeros", _tracking_zeros)
    monkeypatch.setattr(torch, "empty", _tracking_empty)

    def _boom():
        raise AssertionError(
            "causal_mask thunk was materialized on the flex hot path"
        )

    x = torch.randn(B, T, 32)
    rope = sah.RotaryPositionalEncoding(32 // n_h, max_seq_len=T)
    sel = torch.ones(B, n_h, n_blocks, n_blocks)

    out, block_scores = attn(
        x, rope, _boom, forced_block_selection=sel,
    )

    assert out.shape == (B, T, 32)
    assert block_scores is None
    assert not big_allocations, (
        f"dense-scale tensor(s) allocated on the flex hot path: "
        f"{big_allocations}"
    )


def test_soft_path_flex_success_never_builds_dense_causal_mask(monkeypatch):
    """The soft-training-path's `_has_flex` success branch (score_mod +
    a cached causal BlockMask) is the other real production hot path
    (used when no forced_block_selection/forced_hard_mask is supplied,
    i.e. plain `--condition standard` training). It must also never touch
    the dense causal_mask thunk. Unlike the forced_block_selection+flex
    branch, this one gates on `_has_flex = _use_flex and x.is_cuda and
    _init_flex_attention()` -- `x.is_cuda` cannot be forced True on
    CPU-only hardware without patching tensor internals, so this specific
    branch is NOT CPU-testable and is skipped rather than faked (avoiding
    a false-positive claim). It is by inspection identical in structure
    to the branch covered by test_flex_path_never_builds_dense_causal_mask
    above (score_mod + a cached BlockMask, no `causal_mask` reference in
    its body) -- see the module-level docstring for the line-level trace.
    Real verification requires a GPU; see scripts/verify_causal_mask_memory.py.
    """
    pytest.skip(
        "soft-path's _has_flex branch requires x.is_cuda=True (checked "
        "before _init_flex_attention()), so it cannot be forced on "
        "CPU-only hardware without patching tensor internals. Not "
        "verified here -- requires a GPU; see "
        "scripts/verify_causal_mask_memory.py. The forced_block_selection "
        "+ attention_impl='flex' branch (which does NOT check x.is_cuda) "
        "is covered by test_flex_path_never_builds_dense_causal_mask "
        "above and is the branch every real production flex run takes "
        "for the CLHR hard path."
    )


# ---------------------------------------------------------------------------
# 2. dense_mode=True: bit-identical to the pre-fix eager behavior
# ---------------------------------------------------------------------------

def test_dense_mode_causal_mask_unchanged():
    T = 48
    attn = _tiny_attn(attention_impl="bias")
    device = torch.device("cpu")
    x = torch.randn(2, T, 32)
    rope = sah.RotaryPositionalEncoding(32 // attn.n_heads, max_seq_len=T)

    eager_mask = _old_eager_causal_mask(T, device)

    out_eager, _ = attn(x, rope, eager_mask, dense_mode=True)
    out_lazy, _ = attn(x, rope, lambda: eager_mask, dense_mode=True)

    assert torch.equal(out_eager, out_lazy)

    # And bit-identical to a from-scratch reimplementation of the pre-fix
    # dense_mode branch (no _materialize_causal_mask involved at all).
    B, T_, D = x.shape
    n_h, d_h = attn.n_heads, attn.d_head
    q = attn.W_q(x).view(B, T_, n_h, d_h).transpose(1, 2)
    k = attn.W_k(x).view(B, T_, n_h, d_h).transpose(1, 2)
    v = attn.W_v(x).view(B, T_, n_h, d_h).transpose(1, 2)
    q, k = rope(q, T_), rope(k, T_)
    ref_out = F.scaled_dot_product_attention(
        q, k, v, attn_mask=eager_mask.to(q.dtype), dropout_p=0.0,
    )
    ref_out = attn.W_o(ref_out.transpose(1, 2).contiguous().view(B, T_, D))

    assert torch.equal(out_eager, ref_out)


# ---------------------------------------------------------------------------
# 3. legacy forced_hard_mask fallback: bit-identical to pre-fix eager
# ---------------------------------------------------------------------------

def test_legacy_fallback_causal_mask_unchanged():
    T = 48
    attn = _tiny_attn(attention_impl="bias")
    device = torch.device("cpu")
    x = torch.randn(2, T, 32)
    rope = sah.RotaryPositionalEncoding(32 // attn.n_heads, max_seq_len=T)

    torch.manual_seed(1)
    forced_hard_mask = (torch.rand(2, attn.n_heads, T, T) > 0.5).float()
    # Re-apply causality so this is a plausible mask (not load-bearing for
    # the equivalence check, but keeps values finite/sane).
    tril = torch.ones(T, T, dtype=torch.bool).tril()
    forced_hard_mask = forced_hard_mask * tril.float()

    eager_mask = _old_eager_causal_mask(T, device)

    out_eager, _ = attn(x, rope, eager_mask, forced_hard_mask=forced_hard_mask)
    out_lazy, _ = attn(x, rope, lambda: eager_mask,
                        forced_hard_mask=forced_hard_mask)
    assert torch.equal(out_eager, out_lazy)

    # From-scratch reimplementation of the pre-fix forced_hard_mask branch.
    gate_bias = torch.where(
        forced_hard_mask > 0,
        torch.zeros_like(forced_hard_mask),
        torch.full_like(forced_hard_mask, float("-inf")),
    )
    attn_mask = gate_bias + eager_mask
    B, T_, D = x.shape
    n_h, d_h = attn.n_heads, attn.d_head
    q = attn.W_q(x).view(B, T_, n_h, d_h).transpose(1, 2)
    k = attn.W_k(x).view(B, T_, n_h, d_h).transpose(1, 2)
    v = attn.W_v(x).view(B, T_, n_h, d_h).transpose(1, 2)
    q, k = rope(q, T_), rope(k, T_)
    ref_out = F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask.to(q.dtype), dropout_p=0.0,
    )
    ref_out = attn.W_o(ref_out.transpose(1, 2).contiguous().view(B, T_, D))

    assert torch.equal(out_eager, ref_out)


# ---------------------------------------------------------------------------
# 4. soft-path dense-SDPA fallback (the branch every CPU test takes, since
#    _has_flex is always False off-CUDA): bit-identical to pre-fix eager
# ---------------------------------------------------------------------------

def test_soft_path_dense_fallback_causal_mask_unchanged():
    T = 48
    attn = _tiny_attn(attention_impl="bias")
    device = torch.device("cpu")
    x = torch.randn(2, T, 32)
    rope = sah.RotaryPositionalEncoding(32 // attn.n_heads, max_seq_len=T)

    eager_mask = _old_eager_causal_mask(T, device)

    torch.manual_seed(2)
    out_eager, scores_eager = attn(x, rope, eager_mask)
    torch.manual_seed(2)
    out_lazy, scores_lazy = attn(x, rope, lambda: eager_mask)

    assert torch.equal(out_eager, out_lazy)
    assert torch.equal(scores_eager, scores_lazy)


# ---------------------------------------------------------------------------
# 5. _materialize_causal_mask itself
# ---------------------------------------------------------------------------

def test_materialize_causal_mask_passthrough_for_tensor():
    t = torch.zeros(4, 4)
    assert sah._materialize_causal_mask(t) is t


def test_materialize_causal_mask_calls_thunk():
    calls = []

    def thunk():
        calls.append(1)
        return torch.ones(2, 2)

    result = sah._materialize_causal_mask(thunk)
    assert calls == [1]
    assert torch.equal(result, torch.ones(2, 2))


# ---------------------------------------------------------------------------
# 6. integration: full-model forward() (now internally lazy) matches a
#    from-scratch per-layer reimplementation that builds an eager mask
# ---------------------------------------------------------------------------

def test_model_forward_soft_path_unchanged_vs_eager_reimplementation():
    torch.manual_seed(3)
    cfg = dict(vocab_size=50, d_model=32, n_heads=2, n_layers=2, d_ff=64,
               d_gate=8)
    model = sah.HierarchicalSparseTransformer(
        **cfg, block_size=8, top_k_blocks=2, local_window=8, max_seq_len=48,
        dropout=0.0,
    )
    model.eval()
    B, T = 2, 48
    x_ids = torch.randint(0, cfg["vocab_size"], (B, T))
    device = torch.device("cpu")

    torch.manual_seed(4)
    logits_new, _ = model(x_ids, use_checkpoint=False)

    # From-scratch reimplementation using the OLD eager causal_mask
    # construction, driving each layer directly.
    torch.manual_seed(4)
    causal_mask = _old_eager_causal_mask(T, device)
    h = model.drop(model.tok_emb(x_ids))
    for layer in model.layers:
        h = layer(h, model.rope, causal_mask)
    h = model.final_norm(h)
    logits_ref = model.lm_head(h)

    assert torch.equal(logits_new, logits_ref)


# ---------------------------------------------------------------------------
# 8. The eval-path twin of this bug: eval_closed_loop_hierarchical_hard and
#    eval_random_block_hard both build `causal_mask` eagerly, unconditionally,
#    then pass it to layer.attn(..., forced_block_selection=...) in a loop --
#    the same never-read-on-the-flex-hot-path branch as the training-time
#    bug. Missed by the training-time fix's deliberate scoping ("out of
#    scope, owned by a concurrent agent fixing the eval-side local-window
#    mask") and by that concurrent fix's own scope (it only touched
#    `_make_local_window_mask_bool`, not `_make_causal_mask`). Found and
#    fixed 2026-09-20, same day, same pattern: eager Tensor -> lazy
#    zero-arg callable, materialized only via `_materialize_causal_mask`
#    at the point of use inside `HierarchicalSparseAttention.forward`.
# ---------------------------------------------------------------------------

def _tiny_hier_model(seed=5, **overrides):
    torch.manual_seed(seed)
    cfg = dict(vocab_size=50, d_model=32, n_heads=2, n_layers=2, d_ff=64,
               d_gate=8, block_size=8, top_k_blocks=2, local_window=8,
               max_seq_len=2048, dropout=0.0)
    cfg.update(overrides)
    return sah.HierarchicalSparseTransformer(**cfg).eval()


def _one_batch_loader(B, T, vocab_size, seed=6):
    torch.manual_seed(seed)
    batch = torch.randint(0, vocab_size, (B, T + 1))
    return [batch]  # a bare list of one batch satisfies `for batch in loader`


def test_eval_closed_loop_hard_never_builds_dense_causal_mask(monkeypatch):
    """Same monkeypatch-and-track technique as
    test_flex_path_never_builds_dense_causal_mask, applied to the actual
    eval function rather than the attention module directly -- this is the
    integration-level check that the fix reaches all the way through
    eval_closed_loop_hierarchical_hard's loop, not just the lower-level
    helper."""
    T = 1024
    model = _tiny_hier_model(max_seq_len=T + 1)
    loader = _one_batch_loader(B=2, T=T, vocab_size=50)

    def _fake_flex_attn(q, k, v, block_mask=None, score_mod=None, **kw):
        return q

    monkeypatch.setattr(sah, "_FLEX_ATTN_COMPILED", _fake_flex_attn)
    monkeypatch.setattr(sah, "build_block_mask_direct",
                         lambda *a, **k: object())
    for layer in model.layers:
        layer.attn.attention_impl = "flex"

    big_allocations = []
    orig_zeros = torch.zeros

    def _tracking_zeros(*args, **kwargs):
        t = orig_zeros(*args, **kwargs)
        if t.numel() >= T * T:
            big_allocations.append(tuple(t.shape))
        return t

    monkeypatch.setattr(torch, "zeros", _tracking_zeros)

    loss = sah.eval_closed_loop_hierarchical_hard(
        model, loader, torch.device("cpu"), k_blocks=2, max_batches=1,
    )

    assert isinstance(loss, float)
    assert not big_allocations, (
        f"dense-scale tensor(s) allocated in eval_closed_loop_hierarchical_"
        f"hard's flex hot path: {big_allocations}"
    )


def test_eval_random_block_hard_never_builds_dense_causal_mask(monkeypatch):
    T = 1024
    model = _tiny_hier_model(max_seq_len=T + 1)
    loader = _one_batch_loader(B=2, T=T, vocab_size=50)

    def _fake_flex_attn(q, k, v, block_mask=None, score_mod=None, **kw):
        return q

    monkeypatch.setattr(sah, "_FLEX_ATTN_COMPILED", _fake_flex_attn)
    monkeypatch.setattr(sah, "build_block_mask_direct",
                         lambda *a, **k: object())
    for layer in model.layers:
        layer.attn.attention_impl = "flex"

    big_allocations = []
    orig_zeros = torch.zeros

    def _tracking_zeros(*args, **kwargs):
        t = orig_zeros(*args, **kwargs)
        if t.numel() >= T * T:
            big_allocations.append(tuple(t.shape))
        return t

    monkeypatch.setattr(torch, "zeros", _tracking_zeros)

    loss = sah.eval_random_block_hard(
        model, loader, torch.device("cpu"), k_blocks=2, max_batches=1,
    )

    assert isinstance(loss, float)
    assert not big_allocations, (
        f"dense-scale tensor(s) allocated in eval_random_block_hard's flex "
        f"hot path: {big_allocations}"
    )


def test_eval_closed_loop_hard_unchanged_vs_eager_reference():
    """Bit-identical check: eval_closed_loop_hierarchical_hard's returned
    loss must be identical whether causal_mask is built lazily (current
    production code) or eagerly (pre-fix behavior, reimplemented locally
    via _old_eager_causal_mask and driven through the same loop body)."""
    T, B = 40, 2
    model = _tiny_hier_model(max_seq_len=64)
    device = torch.device("cpu")
    loader = _one_batch_loader(B=B, T=T, vocab_size=50, seed=7)

    torch.manual_seed(8)
    loss_new = sah.eval_closed_loop_hierarchical_hard(
        model, loader, device, k_blocks=2, max_batches=1,
    )

    # Reference: verbatim pre-fix loop body, eager causal_mask.
    torch.manual_seed(8)
    batch = loader[0].to(device)
    x_ids, y = batch[:, :-1], batch[:, 1:]
    x = model.drop(model.tok_emb(x_ids))
    causal_mask = _old_eager_causal_mask(T, device)
    with torch.no_grad():
        for layer in model.layers:
            h = layer.attn_norm(x)
            hard_block = layer.attn.block_gate.compute_hard_block_mask(
                h, k_blocks=2,
            )
            attn_out, _ = layer.attn(
                h, model.rope, causal_mask, forced_block_selection=hard_block,
            )
            x = x + attn_out
            x = x + layer.ff(layer.ff_norm(x))
        logits = model.lm_head(model.final_norm(x))
        loss_ref = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1),
            reduction="sum",
        ).item() / y.numel()

    assert loss_new == pytest.approx(loss_ref, abs=1e-6)
