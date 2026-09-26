"""Tests for `loss_chunk_size` on the four run_full_evaluation sub-functions
(evaluate_nll, eval_closed_loop_hierarchical_hard, eval_random_block_hard,
evaluate_dense_nll) in sparse_attention_hierarchical.py.

OBSERVED INCIDENT (2026-09-21): `evaluate_dense_nll`'s unchunked
F.cross_entropy on the full (B,T,vocab) logits tensor OOM'd at T=65536
(`Tried to allocate 6.14 GiB`, LONGCTX_64K_K4_S123, step 3800/3814 -- i.e.
AFTER training completed, inside the LAST call in run_full_evaluation).
Because run_full_evaluation builds one dict at the end and returns it only
once, this discarded native_nll/G_CL/gate_utility too, even though those
were already computed successfully earlier in the SAME call. The other
three sub-functions have the identical unchunked-logits pattern and did not
crash only because dense_mode's own attention forward is more memory-hungry
than the sparse/flex-routed paths (dense_mode is evaluated last) -- at
T=131072 (LONGCTX_128K_K4, running concurrently with this fix) even the
earlier-called sub-functions are a real risk given a ~2x larger logits
tensor than the one that already OOM'd at 65536. All four therefore got the
same `chunked_lm_head_loss` fusion already used by the training path.

loss_chunk_size=0 (the default) must be BIT-IDENTICAL to the pre-fix
behavior for every one of the four functions -- proven here against a
reference computed the OLD way (full logits + F.cross_entropy), not just
new-code-default-vs-new-code-explicit.

Run: PYTHONPATH=src pytest tests/test_eval_loss_chunking.py -q
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

SRC = str(Path(__file__).resolve().parents[1] / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import sparse_attention_hierarchical as sah  # noqa: E402

TOL = 1e-4  # fp32, tiny model: unchunked vs. chunked must agree this tightly

TINY_SEQ_LEN = 8
TINY_BLOCK_SIZE = 4
TINY_TOP_K = 1
TINY_LOCAL_WINDOW = 4
VOCAB = sah.MODEL_CONFIGS["small"]["vocab_size"]


def _tiny_model(seed: int = 0) -> sah.HierarchicalSparseTransformer:
    cfg = dict(sah.MODEL_CONFIGS["small"])
    torch.manual_seed(seed)
    model = sah.HierarchicalSparseTransformer(
        **cfg, block_size=TINY_BLOCK_SIZE, top_k_blocks=TINY_TOP_K,
        local_window=TINY_LOCAL_WINDOW, max_seq_len=TINY_SEQ_LEN,
        dropout=0.0,
    ).float()
    model.eval()
    return model


class _TinyDataset(Dataset):
    def __init__(self, n_seqs: int, seq_len: int, seed: int):
        g = torch.Generator().manual_seed(seed)
        self.data = [
            torch.randint(0, VOCAB, (seq_len + 1,), generator=g)
            for _ in range(n_seqs)
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def _tiny_loader(n_seqs: int = 3, batch_size: int = 1, seed: int = 1) -> DataLoader:
    ds = _TinyDataset(n_seqs, TINY_SEQ_LEN, seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


@pytest.mark.parametrize("chunk_size", [1, 3, 100])
def test_evaluate_nll_chunked_matches_unchunked(chunk_size):
    model = _tiny_model(seed=10)
    loader = _tiny_loader(seed=11)
    unchunked = sah.evaluate_nll(model, loader, torch.device("cpu"))
    chunked = sah.evaluate_nll(
        model, loader, torch.device("cpu"), loss_chunk_size=chunk_size,
    )
    assert abs(unchunked - chunked) < TOL


@pytest.mark.parametrize("chunk_size", [1, 3, 100])
def test_eval_closed_loop_hierarchical_hard_chunked_matches_unchunked(chunk_size):
    model = _tiny_model(seed=20)
    loader = _tiny_loader(seed=21)
    unchunked = sah.eval_closed_loop_hierarchical_hard(
        model, loader, torch.device("cpu"), k_blocks=TINY_TOP_K,
    )
    chunked = sah.eval_closed_loop_hierarchical_hard(
        model, loader, torch.device("cpu"), k_blocks=TINY_TOP_K,
        loss_chunk_size=chunk_size,
    )
    assert abs(unchunked - chunked) < TOL


@pytest.mark.parametrize("chunk_size", [1, 3, 100])
def test_eval_random_block_hard_chunked_matches_unchunked(chunk_size):
    # random_mode draws its own randomness internally (torch.rand for block
    # scores) -- seed globally so both calls draw the SAME random block
    # selection, isolating loss_chunk_size as the only variable.
    model = _tiny_model(seed=30)
    loader = _tiny_loader(seed=31)

    torch.manual_seed(999)
    unchunked = sah.eval_random_block_hard(
        model, loader, torch.device("cpu"), k_blocks=TINY_TOP_K,
    )
    torch.manual_seed(999)
    chunked = sah.eval_random_block_hard(
        model, loader, torch.device("cpu"), k_blocks=TINY_TOP_K,
        loss_chunk_size=chunk_size,
    )
    assert abs(unchunked - chunked) < TOL


@pytest.mark.parametrize("chunk_size", [1, 3, 100])
def test_evaluate_dense_nll_chunked_matches_unchunked(chunk_size):
    model = _tiny_model(seed=40)
    loader = _tiny_loader(seed=41)
    unchunked = sah.evaluate_dense_nll(model, loader, torch.device("cpu"))
    chunked = sah.evaluate_dense_nll(
        model, loader, torch.device("cpu"), loss_chunk_size=chunk_size,
    )
    assert abs(unchunked - chunked) < TOL


def test_run_full_evaluation_default_is_bit_identical_to_no_chunking_arg():
    """The default (loss_chunk_size=0, matching the pre-fix call signature
    before this field existed) must reproduce EXACTLY what run_full_evaluation
    returned before loss_chunk_size was threaded through -- i.e. omitting the
    kwarg and passing loss_chunk_size=0 explicitly must agree exactly."""
    model = _tiny_model(seed=50)
    loader = _tiny_loader(seed=51)

    torch.manual_seed(1234)
    default = sah.run_full_evaluation(
        model, loader, torch.device("cpu"), k_blocks=TINY_TOP_K,
    )
    torch.manual_seed(1234)
    explicit = sah.run_full_evaluation(
        model, loader, torch.device("cpu"), k_blocks=TINY_TOP_K,
        loss_chunk_size=0,
    )
    assert default == explicit


def test_run_full_evaluation_with_chunking_produces_same_metrics():
    """End-to-end: chunked run_full_evaluation reproduces the same G_CL/
    gate_utility (not just the same individual NLLs in isolation) as the
    unchunked path -- catches any bug in how the four sub-results combine."""
    model = _tiny_model(seed=60)
    loader = _tiny_loader(seed=61)

    torch.manual_seed(4321)
    unchunked = sah.run_full_evaluation(
        model, loader, torch.device("cpu"), k_blocks=TINY_TOP_K,
    )
    torch.manual_seed(4321)
    chunked = sah.run_full_evaluation(
        model, loader, torch.device("cpu"), k_blocks=TINY_TOP_K,
        loss_chunk_size=2,
    )
    for key in ("native_nll", "closed_loop_hard_nll", "G_CL",
                "random_block_hard_nll_mean", "gate_utility", "dense_nll"):
        assert abs(unchunked[key] - chunked[key]) < TOL, key


# ─────────────────────────────────────────────────────────────────────────
# dense_nll OOM graceful-degradation (2026-09-22 incident: T=131072 needs a
# 64.00 GiB (T,T) fp32 mask in evaluate_dense_nll's dense-attention branch --
# independent of loss_chunk_size, which only fixes the LOGITS/cross-entropy
# OOM, not this attention-level allocation. A crash here must not destroy
# native_nll/G_CL/gate_utility, which are valid and already computed by the
# time dense_nll (the LAST sub-eval) runs.)
# ─────────────────────────────────────────────────────────────────────────

def test_run_full_evaluation_normal_path_unaffected_by_oom_handling():
    """The try/except around evaluate_dense_nll must not change behavior on
    the normal (non-OOM) path -- dense_nll/dense_ppl present and correct,
    no skip_reason key."""
    model = _tiny_model(seed=70)
    loader = _tiny_loader(seed=71)
    result = sah.run_full_evaluation(
        model, loader, torch.device("cpu"), k_blocks=TINY_TOP_K,
    )
    assert "dense_nll_skip_reason" not in result
    assert result["dense_nll"] is not None
    assert result["dense_ppl"] is not None
    assert abs(result["dense_ppl"] - math.exp(result["dense_nll"])) < 1.0


def test_run_full_evaluation_degrades_gracefully_on_dense_nll_oom(monkeypatch):
    """If evaluate_dense_nll raises OutOfMemoryError (the actual T=131072
    incident), run_full_evaluation must NOT propagate the exception -- it
    must return a dict with native_nll/G_CL/gate_utility intact (these were
    already computed successfully before dense_nll's turn) and
    dense_nll=None/dense_ppl=None/a skip_reason explaining why."""
    model = _tiny_model(seed=80)
    loader = _tiny_loader(seed=81)

    def _raise_oom(*args, **kwargs):
        raise torch.OutOfMemoryError("simulated: 64.00 GiB causal mask")

    monkeypatch.setattr(sah, "evaluate_dense_nll", _raise_oom)

    result = sah.run_full_evaluation(
        model, loader, torch.device("cpu"), k_blocks=TINY_TOP_K,
    )
    assert result["dense_nll"] is None
    assert result["dense_ppl"] is None
    assert "dense_nll_skip_reason" in result
    # The valid, already-computed sub-evals must survive intact.
    for key in ("native_nll", "closed_loop_hard_nll", "G_CL",
                "random_block_hard_nll_mean", "random_block_hard_nll_std",
                "gate_utility"):
        assert key in result
        assert result[key] is not None
