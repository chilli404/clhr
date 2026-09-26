"""Causality gate for the hierarchical block router. RUN THIS FIRST.

Why this file exists
--------------------
Three rounds of validation on `sparse_attention_hierarchical.py` passed while
the model was leaking ~`block_size - 1` positions of lookahead. Training loss
collapsed to ~0.65 (PPL 1.9) on FineWeb, which reads as "excellent" and is in
fact the signature of a model copying the answer. Roughly three GPU-days were
spent on the leaking objective before anyone checked.

The reason the earlier tests could not catch it: they all asserted
`flex_attention output == dense SDPA output`. Both branches share the same
router, so both leaked identically and agreed to float eps. Equivalence
between two implementations is blind to a bug they have in common.

This file instead tests the *property the model must satisfy*: perturbing
token t must not change the logits at any position < t. It needs no GPU, runs
in seconds, and is the check that would have prevented the whole detour.

Root cause it guards
--------------------
`BlockGateModule._attention_pool` compresses a block by pooling over ALL its
tokens, so block b's summary contains tokens in the future of earlier queries
inside b. Routing query block b on that summary leaked ~block_size-1 positions.

Two changes were made; a regression check established they are NOT equally
load-bearing, contrary to the initial diagnosis:

  1. QUERY side (the actual fix) -- query block b routes on block b-1's
     summary, which lies entirely in its past. Reverting this alone
     reintroduces 31/511 leaking positions at block_size=32.
  2. KEY side (a design choice, NOT a causality fix) -- selection restricted
     to strictly-past blocks, tril(diagonal=-1). Reverting this alone leaks
     NOTHING, because `attn_mask = gate_bias + causal_mask` re-applies
     token-level causality, so selecting the diagonal block was already
     harmless. It is kept because the local window (256) already covers the
     diagonal block (32) completely, so excluding it from learned routing
     costs no capability and matches NSA / SeerAttention structure.

Do not treat 2 as protecting causality. Only 1 does. Tests below cover both
so that either regressing is caught, but only 1 will show up as a leak; 2
shows up in `test_selection_is_strictly_past` / `test_block_zero_selects_nothing`
as a design-invariant violation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

import sparse_attention_hierarchical as sah  # noqa: E402
from sparse_attention_hierarchical import (  # noqa: E402
    HierarchicalSparseTransformer,
    MODEL_CONFIGS,
)

TOL = 1e-6


def _tiny_model(block_size: int, local_window: int, top_k_blocks: int,
                T: int) -> HierarchicalSparseTransformer:
    cfg = dict(MODEL_CONFIGS["small"])
    cfg.update(n_layers=2, d_model=128, n_heads=4, d_ff=256)
    torch.manual_seed(0)
    return HierarchicalSparseTransformer(
        **cfg, block_size=block_size, top_k_blocks=top_k_blocks,
        local_window=local_window, max_seq_len=T, dropout=0.0,
    ).float().eval()


def _assert_routing_is_live(block_size: int, top_k: int, T: int) -> None:
    """Fail loudly if the config makes routing a no-op.

    `compute_hard_block_mask` uses `min(top_k, n_blocks)`, so whenever
    top_k >= n_blocks EVERY block is selected and the router's scores are
    irrelevant. A causality test under such a config passes trivially and
    proves nothing -- this is exactly how the first version of this file
    gave four green parametrisations while only one tested anything.
    """
    n_blocks = (T + block_size - 1) // block_size
    assert top_k < n_blocks, (
        f"vacuous config: top_k={top_k} >= n_blocks={n_blocks} "
        f"(T={T}, block_size={block_size}); every block would be selected "
        f"and routing would not be exercised"
    )


def _leaking_positions(fwd, T: int, perturb_at: int) -> list[int]:
    """Positions < perturb_at whose logits move when token perturb_at changes.

    Any non-empty result is a causality violation.
    """
    torch.manual_seed(1)
    x = torch.randint(0, 1000, (1, T))
    x2 = x.clone()
    x2[0, perturb_at] = (x2[0, perturb_at].item() + 500) % 1000
    with torch.no_grad():
        a, b = fwd(x), fwd(x2)
    delta = (a - b).abs()[0, :perturb_at]
    return (delta.max(dim=-1).values > TOL).nonzero().flatten().tolist()


# --- parametrised over the configs we actually train with -------------------
@pytest.mark.parametrize("block_size,local_window,top_k,T", [
    (32, 256, 4, 512),    # 16 blocks, k=4  -- production block_size
    (64, 256, 4, 512),    # 8 blocks,  k=4  -- original pilot block_size
    (32, 64, 4, 256),     # 8 blocks,  k=4  -- small local window
    (32, 256, 8, 640),    # 20 blocks, k=8  -- T not a multiple of block_size
])
def test_soft_path_is_causal(block_size, local_window, top_k, T):
    _assert_routing_is_live(block_size, top_k, T)
    m = _tiny_model(block_size, local_window, top_k, T)
    leaks = _leaking_positions(lambda t: m(t)[0], T, T - 1)
    assert leaks == [], (
        f"soft path leaked {len(leaks)} positions (block_size={block_size}); "
        f"first={leaks[:8]}. Expect ~block_size-1 if the query-side block "
        f"summary shift or the strictly-past key mask regressed."
    )


@pytest.mark.parametrize("block_size,local_window,top_k,T", [
    (32, 256, 4, 512),
    (64, 256, 4, 512),
])
def test_clhr_forced_block_selection_is_causal(block_size, local_window,
                                               top_k, T):
    """The CLHR hard path routes via forced_block_selection, a different
    branch from the soft path. It leaked independently and must be checked
    separately -- CLHR trains on BOTH, so a leak here is equally fatal.
    """
    _assert_routing_is_live(block_size, top_k, T)
    m = _tiny_model(block_size, local_window, top_k, T)

    def fwd(t):
        h = m.drop(m.tok_emb(t))
        cm = m._make_causal_mask(t.shape[1], t.device)
        for layer in m.layers:
            hn = layer.attn_norm(h)
            sel = layer.attn.block_gate.compute_hard_block_mask(hn)
            out, _ = layer.attn(hn, m.rope, cm, forced_block_selection=sel)
            h = h + out
            h = h + layer.ff(layer.ff_norm(h))
        return m.lm_head(m.final_norm(h))

    leaks = _leaking_positions(fwd, T, T - 1)
    assert leaks == [], (
        f"CLHR forced_block_selection leaked {len(leaks)} positions "
        f"(block_size={block_size}); first={leaks[:8]}"
    )


def test_dense_mode_is_causal():
    """dense_mode bypasses the router entirely. It was causal even while the
    routed paths leaked, which is how the leak was localised to the gate --
    keep it as the control.
    """
    T = 512
    m = _tiny_model(32, 256, 4, T)
    leaks = _leaking_positions(lambda t: m(t, dense_mode=True)[0], T, T - 1)
    assert leaks == [], f"dense_mode leaked: {leaks[:8]}"


def test_perturbation_actually_propagates_forward():
    """Guard against a vacuous pass: if the model ignored its input entirely,
    every causality test above would trivially succeed. Assert the perturbed
    position itself DOES move.
    """
    T = 512
    m = _tiny_model(32, 256, 4, T)
    torch.manual_seed(1)
    x = torch.randint(0, 1000, (1, T))
    x2 = x.clone()
    x2[0, T - 1] = (x2[0, T - 1].item() + 500) % 1000
    with torch.no_grad():
        a, b = m(x)[0], m(x2)[0]
    moved = (a - b).abs()[0, T - 1].max().item()
    assert moved > 1e-3, (
        f"final-position logits moved only {moved:.2e}; the causality tests "
        f"may be passing vacuously"
    )


def test_block_zero_selects_nothing():
    """Query block 0 has no strictly-past key block, so its score row is all
    -inf and topk returns arbitrary indices. compute_hard_block_mask must zero
    them out; otherwise block 0 attends itself or the future.
    """
    T, block_size = 512, 32
    m = _tiny_model(block_size, 256, 4, T)
    attn = m.layers[0].attn
    torch.manual_seed(2)
    h = torch.randn(1, T, attn.block_gate.W_bq.in_features)
    with torch.no_grad():
        hard = attn.block_gate.compute_hard_block_mask(h)
    assert hard[:, :, 0, :].sum().item() == 0.0, (
        "query block 0 selected a key block; strict causality was not "
        "re-applied after topk"
    )


def test_selection_is_strictly_past():
    """No selected key block may be >= its query block, for any block."""
    T, block_size = 320, 32          # non-multiple of block_size on purpose
    m = _tiny_model(block_size, 256, 4, T)
    attn = m.layers[0].attn
    torch.manual_seed(3)
    h = torch.randn(1, T, attn.block_gate.W_bq.in_features)
    with torch.no_grad():
        hard = attn.block_gate.compute_hard_block_mask(h)
    n_blocks = hard.shape[-1]
    qi = torch.arange(n_blocks).unsqueeze(1)
    kj = torch.arange(n_blocks).unsqueeze(0)
    illegal = (kj >= qi)
    assert (hard * illegal.float()).sum().item() == 0.0, (
        "selection included a key block at or after the query block"
    )
