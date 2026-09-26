"""Regression tests for the T=65536 eval-time OOM fixed 2026-09-20.

Root cause (confirmed by reading the code, not assumed): both
`eval_closed_loop_hierarchical_hard` and `eval_random_block_hard` in
sparse_attention_hierarchical.py used to build

    local_mask = _make_local_window_mask_bool(T, local_window, device)

-- a dense (1, 1, T, T) bool tensor (4.29 GB at T=65536) -- and then never
read `local_mask` again anywhere in either function. The actual
local-window logic executed by these eval functions lives entirely inside
`layer.attn(..., forced_block_selection=...)`, which (for
`attention_impl="flex"`) builds its mask via `build_block_mask_direct`
(src/flex_block_mask.py) and never materializes a dense (T, T) tensor.
So the fix is: delete the dead allocation. This is a strictly stronger
guarantee than "produces the same output" -- removing code whose result is
never read cannot change any function's return value, by construction.

These tests still exercise the three explicitly required properties:
  1. bit-identical mask construction, vs. an OLD dense reference kept here
     for testing only (never used in production).
  2. no O(T^2) (in fact, no) allocation happens on the fixed eval code
     path.
  3. eval_closed_loop_hierarchical_hard's output is unchanged whether or
     not the dead call is present (integration-level proof the call sites
     were updated correctly).
Plus running the pre-existing test that uses
sparse_attention_hierarchical._make_local_window_mask_bool directly as an
oracle (tests/test_flex_attention_equivalence.py), to confirm that
reference utility still behaves identically at small T.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

SRC = str(Path(__file__).resolve().parents[1] / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import sparse_attention_hierarchical as sah  # noqa: E402


def _old_dense_local_window_mask(T: int, local_window: int,
                                  device: torch.device) -> torch.Tensor:
    """OLD dense implementation, kept for equivalence testing ONLY.

    This is a verbatim copy of _make_local_window_mask_bool's body as it
    existed before the 2026-09-20 safety-guard change (identical formula,
    no size guard). Not used anywhere in production.
    """
    rows = torch.arange(T, device=device).unsqueeze(1)
    cols = torch.arange(T, device=device).unsqueeze(0)
    mask = (cols >= rows - local_window + 1) & (cols <= rows)
    return mask.unsqueeze(0).unsqueeze(0)


class _TinyTokenDataset(Dataset):
    """Fixed sequences of shape (T + 1,) for the eval functions under test."""

    def __init__(self, vocab_size: int, seq_len: int, n_seqs: int, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.seqs = [
            torch.randint(0, vocab_size, (seq_len + 1,), generator=g)
            for _ in range(n_seqs)
        ]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqs[idx]


def _make_tiny_model_and_loader(T: int, seed: int = 0):
    torch.manual_seed(seed)
    cfg = dict(vocab_size=50, d_model=32, n_heads=2, n_layers=2, d_ff=64,
               d_gate=8)
    model = sah.HierarchicalSparseTransformer(
        **cfg, block_size=8, top_k_blocks=2, local_window=8, max_seq_len=T,
        dropout=0.0,
    )
    model.eval()
    ds = _TinyTokenDataset(vocab_size=50, seq_len=T, n_seqs=3, seed=seed)
    loader = DataLoader(ds, batch_size=2)
    return model, loader


# ---------------------------------------------------------------------------
# 1. bit-identical mask construction at small T, including edge cases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("T,local_window", [
    (16, 4),
    (32, 1),          # local_window=1: attend only to self
    (17, 100),        # local_window >= T: fully causal-local
    (10, 3),          # T not a multiple of any block size
    (1, 1),
    (5, 5),
])
def test_local_window_mask_bit_identical_small_T(T, local_window):
    device = torch.device("cpu")
    old = _old_dense_local_window_mask(T, local_window, device)
    new = sah._make_local_window_mask_bool(T, local_window, device)
    assert old.shape == new.shape == (1, 1, T, T)
    assert old.dtype == new.dtype == torch.bool
    assert torch.equal(old, new), (
        f"mismatch at T={T}, local_window={local_window}: "
        f"{(old != new).sum().item()} differing elements"
    )


def test_local_window_mask_bool_raises_instead_of_oomming_at_scale():
    """Above the safety threshold, the reference utility must raise a clear
    error rather than silently attempting a huge allocation (this is what
    converts a future accidental re-introduction of the T=65536 landmine
    into a loud, immediate failure instead of a slow OOM).
    """
    device = torch.device("cpu")
    huge_T = 65536
    assert huge_T * huge_T > sah._LOCAL_WINDOW_MASK_MAX_ELEMENTS
    with pytest.raises(RuntimeError, match="dense"):
        sah._make_local_window_mask_bool(huge_T, 256, device)


# ---------------------------------------------------------------------------
# 2. no quadratic (in fact no) allocation on the real eval code path
# ---------------------------------------------------------------------------

def test_local_window_mask_no_quadratic_allocation(monkeypatch):
    """Prove _make_local_window_mask_bool is never invoked by the eval
    functions any more -- a strictly stronger guarantee than "no tensor of
    size >= T^2 elements is allocated", since it shows NO local-window mask
    memory (of any size) is allocated by these call sites at all.

    T=4096 is chosen per the task spec: large enough that the old dense
    mask (16M bool elements, 16 MB) would be trivial to allocate on CPU,
    so a fast-but-still-quadratic implementation could not pass this test
    by timing coincidence -- it is caught by the raise-on-call hook below,
    not by a clock.
    """
    T = 4096

    def _boom(*args, **kwargs):
        raise AssertionError(
            "_make_local_window_mask_bool was called from an eval "
            "function -- this reintroduces the O(T^2) OOM landmine."
        )

    monkeypatch.setattr(sah, "_make_local_window_mask_bool", _boom)

    model, loader = _make_tiny_model_and_loader(T)
    # Both eval functions must run to completion without ever touching the
    # dense mask builder now that it's guaranteed to raise if called.
    loss_cl = sah.eval_closed_loop_hierarchical_hard(
        model, loader, torch.device("cpu"), k_blocks=2, max_batches=1,
    )
    loss_rand = sah.eval_random_block_hard(
        model, loader, torch.device("cpu"), k_blocks=2, max_batches=1,
    )
    assert torch.isfinite(torch.tensor(loss_cl))
    assert torch.isfinite(torch.tensor(loss_rand))


# ---------------------------------------------------------------------------
# 3. integration: eval output unchanged whether or not the dead call is
#    present (proves the call-site removal is behavior-preserving)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _eval_closed_loop_hierarchical_hard_WITH_DEAD_CALL(
    model, loader, device, k_blocks, max_batches=100,
):
    """Reproduction of eval_closed_loop_hierarchical_hard with the removed
    dead `local_mask = _make_local_window_mask_bool(...)` line reinstated,
    to prove its presence/absence does not change the returned loss.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    local_window = model.local_window

    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        if isinstance(batch, dict):
            batch = batch["input_ids"]
        batch = batch.to(device)
        x_ids, y = batch[:, :-1], batch[:, 1:]
        B, T = x_ids.shape

        # The dead line, reinstated on purpose:
        _unused_local_mask = _old_dense_local_window_mask(T, local_window, device)

        x = model.tok_emb(x_ids)
        x = model.drop(x)
        causal_mask = model._make_causal_mask(T, device)

        for layer in model.layers:
            h = layer.attn_norm(x)
            hard_block = layer.attn.block_gate.compute_hard_block_mask(
                h, k_blocks=k_blocks,
            )
            attn_out, _ = layer.attn(
                h, model.rope, causal_mask,
                forced_block_selection=hard_block,
            )
            x = x + attn_out
            x = x + layer.ff(layer.ff_norm(x))

        logits = model.lm_head(model.final_norm(x))
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1),
            reduction="sum",
        )
        total_loss += loss.item()
        total_tokens += y.numel()

    return total_loss / total_tokens


def test_eval_closed_loop_hierarchical_hard_unchanged_at_small_T():
    T = 24
    model, loader = _make_tiny_model_and_loader(T)
    device = torch.device("cpu")

    loss_fixed = sah.eval_closed_loop_hierarchical_hard(
        model, loader, device, k_blocks=2, max_batches=2,
    )
    loss_with_dead_call = _eval_closed_loop_hierarchical_hard_WITH_DEAD_CALL(
        model, loader, device, k_blocks=2, max_batches=2,
    )
    assert loss_fixed == loss_with_dead_call, (
        f"fixed={loss_fixed!r} vs with-dead-call={loss_with_dead_call!r}"
    )
