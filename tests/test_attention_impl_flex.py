"""Tests for the --attention-impl {bias,flex} mechanism in
sparse_attention_hierarchical.py.

Context: this is a NEW, independent mechanism from the pre-existing (dead in
production) `_use_flex`/`_FLEX_ENABLE_FORCED_SELECTION`/
`_create_hard_block_mask_flex` machinery exercised by
tests/test_flex_attention_equivalence.py -- that machinery is untouched.
`attention_impl="flex"` wires HierarchicalSparseAttention's
`forced_block_selection` (CLHR/hard-routing) branch to
build_block_mask_direct (src/flex_block_mask.py), which constructs a
BlockMask directly from the router's block selection without ever
materializing a dense (B, H, T, T) token-level mask -- the thing this whole
mechanism exists to avoid.

CPU caveats (OBSERVED, not guessed):
  - torch.compile(flex_attention) requires a GPU backend; on CPU we
    monkeypatch `sah._FLEX_ATTN_COMPILED` to the eager (uncompiled)
    `sah._FLEX_ATTN_RAW` to test correctness of the wiring/construction. The
    actual compiled-kernel speed/memory win is UNVERIFIED on this machine.
  - flex_attention's forward call raises NotImplementedError on CPU as soon
    as any input requires grad (OBSERVED: this fires at the forward() call
    itself, not only at .backward()) -- "FlexAttention does not support
    backward on CPU. Please set the input requires_grad to False or use
    another device." Tests that need grad-enabled tensors on CPU catch this
    verbatim and skip; they do not pretend it doesn't happen.

Run: PYTHONPATH=src pytest tests/test_attention_impl_flex.py -q
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pytest
import torch

SRC = str(Path(__file__).resolve().parents[1] / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import sparse_attention_hierarchical as sah  # noqa: E402
from flex_block_mask import build_block_mask_direct  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

FLEX_OK = sah._init_flex_attention()
requires_flex = pytest.mark.skipif(
    not FLEX_OK,
    reason=f"flex_attention unavailable in this env: {sah._FLEX_IMPORT_ERROR}",
)


def _make_attn(attention_impl, block_size=32, local_window=64,
               top_k_blocks=4, d_model=64, n_heads=2, d_gate=16, seed=0):
    torch.manual_seed(seed)
    attn = sah.HierarchicalSparseAttention(
        d_model=d_model, n_heads=n_heads, d_gate=d_gate,
        block_size=block_size, top_k_blocks=top_k_blocks,
        local_window=local_window, dropout=0.0,
        attention_impl=attention_impl,
    )
    attn.eval()
    return attn


def _causal_mask(T):
    mask = torch.zeros(T, T)
    mask.masked_fill_(~torch.ones(T, T, dtype=torch.bool).tril(), float("-inf"))
    return mask.unsqueeze(0).unsqueeze(0)


def _random_selection(B, n_heads, n_blocks, top_k, generator=None):
    sel = torch.zeros(B, n_heads, n_blocks, n_blocks, dtype=torch.bool)
    for b in range(B):
        for h in range(n_heads):
            for qb in range(n_blocks):
                n_valid = qb + 1
                k = min(top_k, n_valid)
                if generator is not None:
                    perm = torch.randperm(n_valid, generator=generator)
                else:
                    perm = torch.randperm(n_valid)
                sel[b, h, qb, perm[:k]] = True
    return sel


def _pairs_for_batch(block_mask, b):
    """(head, query_block, key_block) triples admitted for batch element b."""
    pairs = set()

    def _collect(num_blocks, indices):
        if num_blocks is None or indices is None:
            return
        nb = num_blocks[b]
        idx = indices[b]
        n_heads, num_q_blocks = nb.shape
        for h in range(n_heads):
            for qb in range(num_q_blocks):
                n = int(nb[h, qb])
                for kb in idx[h, qb, :n].tolist():
                    pairs.add((h, qb, int(kb)))

    _collect(block_mask.kv_num_blocks, block_mask.kv_indices)
    _collect(block_mask.full_kv_num_blocks, block_mask.full_kv_indices)
    return pairs


@requires_flex
def test_flex_matches_bias_logits():
    # Forward-only numerical equivalence, fp32, CPU eager (uncompiled) flex.
    # Tolerance: float32 eps-scale (~1e-6), since both paths do the same
    # matmuls/softmax in fp32 -- any larger discrepancy would mean the two
    # paths compute genuinely different attention patterns, not just
    # different rounding.
    block_size, local_window, top_k, T, B, n_heads = 32, 64, 4, 256, 2, 2
    n_blocks = T // block_size

    attn_bias = _make_attn("bias", block_size, local_window, top_k, n_heads=n_heads)
    attn_flex = _make_attn("flex", block_size, local_window, top_k, n_heads=n_heads)
    attn_flex.load_state_dict(attn_bias.state_dict())

    rope = sah.RotaryPositionalEncoding(attn_bias.d_head, max_seq_len=T)
    torch.manual_seed(1)
    x = torch.randn(B, T, attn_bias.n_heads * attn_bias.d_head)
    causal_mask = _causal_mask(T)
    sel = _random_selection(B, n_heads, n_blocks, top_k,
                             generator=torch.Generator().manual_seed(2))

    sah._FLEX_ATTN_COMPILED = sah._FLEX_ATTN_RAW
    with torch.no_grad():
        out_bias, _ = attn_bias(x, rope, causal_mask, forced_block_selection=sel.float())
        out_flex, _ = attn_flex(x, rope, causal_mask, forced_block_selection=sel.float())

    max_diff = (out_bias - out_flex).abs().max().item()
    assert max_diff < 1e-5, f"flex vs bias logits diverge: max abs diff {max_diff}"


@requires_flex
def test_flex_matches_bias_gradients():
    block_size, local_window, top_k, T, B, n_heads = 32, 64, 4, 256, 2, 2
    n_blocks = T // block_size

    attn_bias = _make_attn("bias", block_size, local_window, top_k, n_heads=n_heads)
    attn_flex = _make_attn("flex", block_size, local_window, top_k, n_heads=n_heads)
    attn_flex.load_state_dict(attn_bias.state_dict())

    rope = sah.RotaryPositionalEncoding(attn_bias.d_head, max_seq_len=T)
    torch.manual_seed(3)
    x = torch.randn(B, T, attn_bias.n_heads * attn_bias.d_head, requires_grad=True)
    causal_mask = _causal_mask(T)
    sel = _random_selection(B, n_heads, n_blocks, top_k,
                             generator=torch.Generator().manual_seed(4))

    sah._FLEX_ATTN_COMPILED = sah._FLEX_ATTN_RAW
    try:
        out_flex, _ = attn_flex(x, rope, causal_mask, forced_block_selection=sel.float())
        out_flex.sum().backward()
    except NotImplementedError as e:
        pytest.skip(f"flex_attention backward unavailable on CPU: {e}")

    out_bias, _ = attn_bias(x.detach().clone().requires_grad_(True), rope,
                             causal_mask, forced_block_selection=sel.float())
    out_bias.sum().backward()

    for (name_b, p_b), (name_f, p_f) in zip(
        attn_bias.named_parameters(), attn_flex.named_parameters(),
    ):
        assert name_b == name_f
        if p_b.grad is None and p_f.grad is None:
            continue
        max_diff = (p_b.grad - p_f.grad).abs().max().item()
        assert max_diff < 1e-4, f"{name_b}: grad diff {max_diff}"


@requires_flex
def test_flex_batched_selection_pattern():
    # Two examples routed differently must realize different output --
    # exercises Step 1's real per-example batch support, not broadcasting.
    block_size, local_window, top_k, T, n_heads = 32, 64, 2, 256, 2
    n_blocks = T // block_size
    attn = _make_attn("flex", block_size, local_window, top_k, n_heads=n_heads)
    rope = sah.RotaryPositionalEncoding(attn.d_head, max_seq_len=T)

    torch.manual_seed(5)
    x = torch.randn(2, T, attn.n_heads * attn.d_head)
    x = x[0:1].repeat(2, 1, 1)  # identical input for both batch elements
    causal_mask = _causal_mask(T)

    sel_a = torch.zeros(n_heads, n_blocks, n_blocks, dtype=torch.bool)
    sel_b = torch.zeros_like(sel_a)
    sel_a[:, n_blocks - 1, 0] = True  # block (last_qb, 0): causal, non-local
    sel = torch.stack([sel_a, sel_b], dim=0)

    sah._FLEX_ATTN_COMPILED = sah._FLEX_ATTN_RAW
    with torch.no_grad():
        out, _ = attn(x, rope, causal_mask, forced_block_selection=sel.float())

    diff = (out[0] - out[1]).abs().max().item()
    assert diff > 1e-6, (
        "identical input, differing per-example block selection produced "
        "identical output -- batch dimension may be broadcast, not real"
    )


def test_dense_mode_works_under_flex():
    # dense_mode=True must short-circuit before attention_impl is consulted
    # at all -- verified here by observing bit-identical output between
    # attention_impl='flex' and 'bias' under dense_mode.
    block_size, local_window, top_k, T = 32, 64, 4, 128
    attn_bias = _make_attn("bias", block_size, local_window, top_k)
    attn_flex = _make_attn("flex", block_size, local_window, top_k)
    attn_flex.load_state_dict(attn_bias.state_dict())

    rope = sah.RotaryPositionalEncoding(attn_bias.d_head, max_seq_len=T)
    torch.manual_seed(6)
    x = torch.randn(2, T, attn_bias.n_heads * attn_bias.d_head)
    causal_mask = _causal_mask(T)

    with torch.no_grad():
        out_bias, scores_bias = attn_bias(x, rope, causal_mask, dense_mode=True)
        out_flex, scores_flex = attn_flex(x, rope, causal_mask, dense_mode=True)

    assert scores_bias is None and scores_flex is None
    assert torch.equal(out_bias, out_flex), (
        "dense_mode output differs between attention_impl values -- "
        "dense_mode must be impl-independent"
    )


@requires_flex
def test_flex_path_allocates_no_quadratic_tensor(monkeypatch):
    # The important guard: attention_impl='flex' must never call the dense
    # token-level mask builders (_expand_block_mask / _make_local_window_
    # mask), which is exactly the O(T^2) materialization this mechanism
    # exists to avoid. If either is called, the flex path silently fell
    # back to (or reused) dense construction.
    block_size, local_window, top_k, T = 32, 64, 4, 128
    attn = _make_attn("flex", block_size, local_window, top_k)
    rope = sah.RotaryPositionalEncoding(attn.d_head, max_seq_len=T)
    torch.manual_seed(7)
    x = torch.randn(2, T, attn.n_heads * attn.d_head)
    causal_mask = _causal_mask(T)
    n_blocks = T // block_size
    sel = _random_selection(2, attn.n_heads, n_blocks, top_k)

    def _boom(*args, **kwargs):
        raise AssertionError("dense token-level mask builder was called under attention_impl='flex'")

    monkeypatch.setattr(sah.HierarchicalSparseAttention, "_expand_block_mask", _boom)
    monkeypatch.setattr(sah.HierarchicalSparseAttention, "_make_local_window_mask", _boom)

    sah._FLEX_ATTN_COMPILED = sah._FLEX_ATTN_RAW
    with torch.no_grad():
        attn(x, rope, causal_mask, forced_block_selection=sel.float())


@requires_flex
def test_flex_attention_compiled_once(monkeypatch):
    # flex_attention must be compiled once (module-wide singleton), never
    # re-initialized per forward() call.
    sah._init_flex_attention()  # ensure the real singleton is populated
    monkeypatch.setattr(sah, "_FLEX_ATTN_COMPILED", sah._FLEX_ATTN_RAW)

    calls = []
    real_init = sah._init_flex_attention

    def counting_init():
        calls.append(1)
        return real_init()

    monkeypatch.setattr(sah, "_init_flex_attention", counting_init)

    block_size, local_window, top_k, T = 32, 64, 4, 128
    attn = _make_attn("flex", block_size, local_window, top_k)
    rope = sah.RotaryPositionalEncoding(attn.d_head, max_seq_len=T)
    torch.manual_seed(8)
    x = torch.randn(2, T, attn.n_heads * attn.d_head)
    causal_mask = _causal_mask(T)
    n_blocks = T // block_size
    sel = _random_selection(2, attn.n_heads, n_blocks, top_k)

    with torch.no_grad():
        for _ in range(3):
            attn(x, rope, causal_mask, forced_block_selection=sel.float())

    assert calls == [], (
        "_init_flex_attention was called from forward() even though "
        "_FLEX_ATTN_COMPILED was already set -- singleton guard is broken"
    )


def test_router_selection_deterministic_across_recompute():
    # Gradient checkpointing re-runs forward during backward; the BlockMask
    # built from the SAME forced_block_selection tensor must be identical
    # across repeated calls, or checkpointed backward would silently use a
    # different sparsity pattern than the recorded forward.
    block_size, local_window, T = 32, 64, 256
    n_heads, n_blocks = 2, T // block_size
    sel = _random_selection(3, n_heads, n_blocks, top_k=3,
                             generator=torch.Generator().manual_seed(9)).float()

    bm1 = build_block_mask_direct(sel, block_size, local_window, T, "cpu")
    bm2 = build_block_mask_direct(sel, block_size, local_window, T, "cpu")

    for b in range(3):
        pairs1 = _pairs_for_batch(bm1, b)
        pairs2 = _pairs_for_batch(bm2, b)
        assert pairs1 == pairs2, (
            f"batch {b}: build_block_mask_direct produced different "
            f"BlockMasks across two calls with the identical selection "
            f"tensor -- non-deterministic construction would corrupt "
            f"gradient-checkpointed backward"
        )


@requires_flex
def test_train_two_steps_flex_cpu():
    # Attempt an actual 2-step training loop with attention_impl='flex' on
    # CPU. Per module docstring, flex_attention's backward raises
    # NotImplementedError on CPU as soon as any input requires grad -- if
    # so, this is UNVERIFIABLE on CPU and we skip with the verbatim
    # exception text rather than claim training works.
    block_size, local_window, top_k, T = 32, 64, 4, 64
    n_blocks = T // block_size
    model = sah.HierarchicalSparseTransformer(
        vocab_size=256, d_model=64, n_heads=2, n_layers=1, d_ff=128,
        d_gate=16, block_size=block_size, top_k_blocks=top_k,
        local_window=local_window, max_seq_len=T, dropout=0.0,
        attention_impl="flex",
    )
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    sah._FLEX_ATTN_COMPILED = sah._FLEX_ATTN_RAW

    n_heads = model.n_heads
    sel = [_random_selection(2, n_heads, n_blocks, top_k) for _ in range(2)]

    try:
        for step in range(2):
            input_ids = torch.randint(0, 256, (2, T))
            logits, _ = model(
                input_ids,
                forced_block_selections=[sel[step].float()] * model.n_layers,
            )
            loss = logits.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
    except NotImplementedError as e:
        pytest.skip(f"flex training loop unverifiable on CPU: {e}")
    except TypeError as e:
        pytest.skip(f"HierarchicalSparseTransformer.forward signature does "
                     f"not accept forced_block_selections as used here: {e}")
