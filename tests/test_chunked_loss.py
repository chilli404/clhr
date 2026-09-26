"""Tests for the chunked vocab-projection + loss path in
sparse_attention_hierarchical.py (`chunked_lm_head_loss`, `--loss-chunk-size`).

Context: at long context, `HierarchicalSparseTransformer.forward` computing
`logits = self.lm_head(x)` -- a (B, T, vocab_size) tensor -- followed by
`F.cross_entropy` (which upcasts that tensor to fp32 internally) is the OOM
ceiling, independent of and downstream from the block-sparse attention fix.
`chunked_lm_head_loss` fuses the vocab projection with the loss per chunk of
sequence positions and wraps that fused unit in `torch.utils.checkpoint`, so
backward recomputes and frees one chunk's logits at a time instead of
autograd retaining all of them (which is what a naive loop over
`F.cross_entropy` alone would still do).

These tests run on CPU in fp32 for exactness; CUDA/bf16/memory behavior is
verified separately by scripts/verify_chunked_loss_memory.py (no pytest
available on some cluster images).

Run: PYTHONPATH=src pytest tests/test_chunked_loss.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

SRC = str(Path(__file__).resolve().parents[1] / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import sparse_attention_hierarchical as sah  # noqa: E402

TOL = 1e-5  # fp32, tiny model: unchunked vs. chunked must agree this tightly


def _unchunked_loss(hidden, targets, lm_head):
    logits = lm_head(hidden)
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
    )


def _make_hidden_and_targets(B=2, T=13, D=8, V=37, seed=0, requires_grad=True):
    torch.manual_seed(seed)
    hidden = torch.randn(B, T, D, requires_grad=requires_grad)
    targets = torch.randint(0, V, (B, T))
    lm_head = nn.Linear(D, V, bias=False)
    return hidden, targets, lm_head


class _CountingLinear(nn.Module):
    """Wraps nn.Linear, counting forward() invocations -- used to verify the
    chunked path calls the vocab projection once per chunk, not once total
    or T times."""

    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.linear = linear
        self.call_count = 0

    def forward(self, x):
        self.call_count += 1
        return self.linear(x)


# ---------------------------------------------------------------------
# 1. Chunked loss value matches unchunked, for chunk sizes that do and
#    do not evenly divide T, and one larger than T.
# ---------------------------------------------------------------------

@pytest.mark.parametrize("T,chunk_size", [
    (13, 4),    # 13 % 4 != 0 -- ragged final chunk
    (13, 13),   # exactly one chunk
    (13, 1000),  # larger than T -- collapses to one chunk
    (16, 4),    # evenly divides
    (16, 5),    # ragged
])
def test_chunked_loss_matches_unchunked(T, chunk_size):
    hidden, targets, lm_head = _make_hidden_and_targets(T=T, requires_grad=False)
    ref = _unchunked_loss(hidden, targets, lm_head)
    got = sah.chunked_lm_head_loss(hidden, targets, lm_head, chunk_size)
    assert torch.allclose(got, ref, atol=TOL, rtol=TOL), (
        f"T={T} chunk_size={chunk_size}: chunked={got.item()} "
        f"ref={ref.item()}"
    )


# ---------------------------------------------------------------------
# 2. Gradients match, not just the scalar loss value.
# ---------------------------------------------------------------------

@pytest.mark.parametrize("chunk_size", [4, 13, 1000, 5])
def test_chunked_loss_gradients_match(chunk_size):
    T = 13
    hidden_ref, targets, lm_head_ref = _make_hidden_and_targets(
        T=T, requires_grad=True,
    )
    # Independent copies so gradients don't accumulate across the two paths.
    hidden_chunked = hidden_ref.detach().clone().requires_grad_(True)
    lm_head_chunked = nn.Linear(lm_head_ref.in_features,
                                 lm_head_ref.out_features, bias=False)
    lm_head_chunked.load_state_dict(lm_head_ref.state_dict())

    ref_loss = _unchunked_loss(hidden_ref, targets, lm_head_ref)
    ref_loss.backward()

    got_loss = sah.chunked_lm_head_loss(
        hidden_chunked, targets, lm_head_chunked, chunk_size,
    )
    got_loss.backward()

    assert torch.allclose(got_loss, ref_loss, atol=TOL, rtol=TOL)
    assert torch.allclose(hidden_chunked.grad, hidden_ref.grad,
                           atol=TOL, rtol=TOL)
    assert torch.allclose(lm_head_chunked.weight.grad,
                           lm_head_ref.weight.grad, atol=TOL, rtol=TOL)


# ---------------------------------------------------------------------
# 3. Ragged final chunk: result is the token-count-weighted mean, NOT the
#    mean of per-chunk means. These differ whenever the last chunk is
#    short, so this test would catch a later "simplification" to
#    averaging chunk means.
# ---------------------------------------------------------------------

def test_chunked_loss_weighting_with_ragged_final_chunk():
    # B*T = 10, chunk_size = 4 -> chunks of size 4, 4, 2 (ragged last chunk)
    B, T, D, V = 1, 10, 6, 5
    hidden, targets, lm_head = _make_hidden_and_targets(
        B=B, T=T, D=D, V=V, requires_grad=False,
    )
    chunk_size = 4

    got = sah.chunked_lm_head_loss(hidden, targets, lm_head, chunk_size)

    flat_hidden = hidden.reshape(-1, D)
    flat_targets = targets.reshape(-1)
    n = flat_hidden.shape[0]

    # Token-count-weighted mean: sum of per-chunk SUMS, divided by total N.
    chunk_means = []
    chunk_sums = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        logits_chunk = lm_head(flat_hidden[start:end])
        s = F.cross_entropy(logits_chunk, flat_targets[start:end],
                             reduction="sum")
        chunk_sums.append(s)
        chunk_means.append(s / (end - start))

    weighted_mean = torch.stack(chunk_sums).sum() / n
    mean_of_chunk_means = torch.stack(chunk_means).mean()

    assert torch.allclose(got, weighted_mean, atol=TOL, rtol=TOL)
    # Sanity: for this ragged split the two reductions actually differ,
    # otherwise the assertion below would be vacuous.
    assert not torch.allclose(weighted_mean, mean_of_chunk_means,
                               atol=1e-4, rtol=1e-4)
    assert not torch.allclose(got, mean_of_chunk_means,
                               atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------
# 4. loss_chunk_size == 0 is the exact existing code path (checked at the
#    model.forward level: skip_lm_head defaults to False and behaves
#    exactly as before -- full logits returned, unchanged shape/dtype).
# ---------------------------------------------------------------------

def test_chunk_size_zero_is_bit_identical():
    torch.manual_seed(0)
    model = sah.HierarchicalSparseTransformer(
        vocab_size=37, d_model=16, n_heads=2, n_layers=1, d_ff=32,
        d_gate=8, block_size=4, top_k_blocks=2, local_window=4,
        max_seq_len=16, dropout=0.0, attention_impl="bias",
    )
    model.eval()
    x = torch.randint(0, 37, (1, 12))

    torch.manual_seed(1)
    logits_a, scores_a = model(x, use_checkpoint=False, dense_mode=False)
    torch.manual_seed(1)
    logits_b, scores_b = model(x, use_checkpoint=False, dense_mode=False,
                                skip_lm_head=False)

    assert torch.equal(logits_a, logits_b)
    assert logits_a.shape == (1, 12, 37)

    # skip_lm_head=True must return the pre-lm_head hidden states, and
    # feeding them through lm_head + cross_entropy manually must equal the
    # loss_chunk_size=0 call chain used in the training loop.
    torch.manual_seed(1)
    hidden, _ = model(x, use_checkpoint=False, dense_mode=False,
                       skip_lm_head=True)
    reconstructed_logits = model.lm_head(hidden)
    assert torch.equal(reconstructed_logits, logits_a)


# ---------------------------------------------------------------------
# 5. The chunked path never materialises a single B*T*V-sized tensor: the
#    vocab projection is called once per chunk (not once for the whole
#    sequence). A clean "no single big allocation" check isn't reliably
#    instrumentable on CPU (no CUDA allocator stats), so this asserts the
#    number of lm_head/projection calls equals the expected chunk count,
#    which is what makes the memory-per-call bounded by chunk_size*V
#    rather than T*V.
# ---------------------------------------------------------------------

@pytest.mark.parametrize("T,chunk_size,expected_calls", [
    (13, 4, 4),     # ceil(13/4) = 4 chunks: 4,4,4,1
    (16, 4, 4),     # exact
    (13, 1000, 1),  # collapses to one chunk
])
def test_no_full_vocab_tensor_materialised(T, chunk_size, expected_calls):
    hidden, targets, lm_head = _make_hidden_and_targets(
        B=1, T=T, requires_grad=True,
    )
    counting_head = _CountingLinear(lm_head)

    loss = sah.chunked_lm_head_loss(hidden, targets, counting_head, chunk_size)
    loss.backward()

    # Each chunk is checkpointed, so the forward-projection call happens
    # once during the initial (memory-saving) pass and once more during
    # backward's recompute -- i.e. exactly 2x the chunk count, never a
    # single call over the full T (which would still be `expected_calls`
    # calls of full-T size instead of chunk-size).
    assert counting_head.call_count == 2 * expected_calls, (
        f"expected {2 * expected_calls} projection calls "
        f"(forward+recompute per chunk), got {counting_head.call_count}"
    )
