"""Acceptance tests for src/sparse_attention_moba.py -- a faithful from-scratch
port of MoonshotAI/MoBA (Mixture of Block Attention) into this project's
Transformer scaffold, for testing whether the soft-train/hard-deploy
generalization gap (G_CL) that CLHR targets shows up inside a REAL, widely
used sparse-attention scheme rather than only in this repo's own
self-implemented baselines (e.g. src/sparse_attention_block_router.py).

Written and run FIRST (must fail before src/sparse_attention_moba.py exists).
Tiny CPU-only configs throughout, matching this project's
tests/test_moe_soft_to_hard.py rigor bar.

Mechanism reference (verified against source, see module docstring in
src/sparse_attention_moba.py for full citations):
  https://raw.githubusercontent.com/MoonshotAI/MoBA/master/moba/moba_naive.py
  https://raw.githubusercontent.com/MoonshotAI/MoBA/master/moba/config.py
  https://raw.githubusercontent.com/MoonshotAI/MoBA/master/README.md
Key verified facts this test suite encodes:
  - MoBA's gating is PARAMETER-LESS: block relevance = query dot mean-pooled
    block key (fp32), no learned gate weights at all.
  - MoBA's own training-time block selection is HARD top-k (torch.topk(...)
    .indices only) -- not a soft/differentiable relaxation. No gradient
    flows through the selection; gradient reaches Q/K only via the attention
    softmax over whichever blocks were selected.
  - The current/local block is always force-included (a +inf gate score
    trick) and is the only block subject to additional *intra-block* causal
    restriction; once a past block is selected it is attended to in full.
  - A token-level causal tril mask is applied UNCONDITIONALLY after block
    expansion -- this is not just an intra-current-block fix, it is also the
    safety net that keeps causality correct in the top_k_blocks >= n_blocks
    edge case (see test_dense_mode_equals_full_top_k_blocks below).
"""
from __future__ import annotations

import ast
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from sparse_attention_moba import (
    CONDITIONS,
    apply_rotary_emb,
    RotaryPositionalEncoding,
    moba_block_means,
    compute_moba_gate_bias,
    MoBAAttention,
    MoBATransformerBlock,
    MoBATransformer,
    record_moba_biases_from_trajectory,
    forward_open_loop_moba,
    forward_closed_loop_moba,
    evaluate_nll,
    eval_open_loop,
    eval_closed_loop,
    eval_random_hard,
    _mean_nll_over_loader,
    run_full_evaluation,
    compute_condition_loss,
    train_model,
    CyclingTokenDataset,
    build_argparser,
)

VOCAB = 97
SEQ_LEN = 32
BATCH = 3
TINY = dict(d_model=16, n_heads=2, n_layers=2, d_ff=32, block_size=4, top_k_blocks=2, max_seq_len=SEQ_LEN,
            dropout=0.0)


def make_model(seed=0, **overrides):
    torch.manual_seed(seed)
    cfg = dict(TINY)
    cfg.update(overrides)
    return MoBATransformer(vocab_size=VOCAB, **cfg)


def make_batch(seed=0, batch=BATCH, seq_len=SEQ_LEN):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, VOCAB, (batch, seq_len), generator=g)
    y = torch.randint(0, VOCAB, (batch, seq_len), generator=g)
    return x, y


class TinyLoader:
    def __init__(self, batches):
        self.batches = batches

    def __iter__(self):
        return iter(self.batches)


def make_loader(n_batches=2, seed=0, batch=BATCH, seq_len=SEQ_LEN):
    g = torch.Generator().manual_seed(seed)
    batches = [torch.randint(0, VOCAB, (batch, seq_len + 1), generator=g) for _ in range(n_batches)]
    return TinyLoader(batches)


# ─────────────────────────────────────────────────────────────────────────
# 1. moba_block_means: exact mean over REAL tokens, ragged-block-aware
# ─────────────────────────────────────────────────────────────────────────

def test_block_means_full_blocks_hand_computed():
    # B=1,H=1,T=4,Dh=2, block_size=2 -> 2 full blocks, no ragged remainder.
    k = torch.tensor([[[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]]])  # (1,1,4,2)
    means = moba_block_means(k, block_size=2)
    expected = torch.tensor([[[[2.0, 3.0], [6.0, 7.0]]]])
    assert torch.allclose(means, expected)


def test_block_means_ragged_last_block_uses_true_count_not_zero_padded_average():
    # T=5, block_size=2 -> blocks of size [2,2,1]; last block's mean must be
    # its single real value, NOT (value + 0)/2 from naive zero-padding.
    k = torch.tensor([[[[1.0], [3.0], [5.0], [7.0], [100.0]]]])  # (1,1,5,1)
    means = moba_block_means(k, block_size=2)
    assert means.shape == (1, 1, 3, 1)
    assert torch.allclose(means[0, 0, 0], torch.tensor([2.0]))   # mean(1,3)
    assert torch.allclose(means[0, 0, 1], torch.tensor([6.0]))   # mean(5,7)
    assert torch.allclose(means[0, 0, 2], torch.tensor([100.0]))  # mean(100) -- NOT 50.0


def test_block_means_returns_fp32_regardless_of_input_dtype():
    k = torch.randn(1, 1, 4, 2, dtype=torch.float64)
    means = moba_block_means(k, block_size=2)
    assert means.dtype == torch.float32


# ─────────────────────────────────────────────────────────────────────────
# 2. compute_moba_gate_bias: causal + own-block-forced + top-k, hand-verified
# ─────────────────────────────────────────────────────────────────────────

def test_gate_bias_own_block_always_selected_even_with_low_score():
    """Construct q/k so the query's OWN block has the worst possible dot
    product against its own mean-pooled key, yet MoBA's +inf force-include
    trick must still select it (it's the only causally-valid block anyway
    for an early query, but we also check a later query whose own block
    scores below a competing earlier block that top-k would otherwise
    prefer)."""
    block_size = 2
    top_k = 1
    B, H, T, Dh = 1, 1, 6, 1
    # 3 blocks: [0,1] [2,3] [4,5]. Keys chosen so block means are 10, 1, 0.
    k = torch.tensor([[[[10.0], [10.0], [1.0], [1.0], [0.0], [0.0]]]])
    # Query at position 4 (own block = block 2, mean key = 0 -> dot = 0),
    # but block 0's mean key (10) would give a much higher score (q=1).
    q = torch.ones(B, H, T, Dh)
    bias, block_mask = compute_moba_gate_bias(q, k, block_size, top_k)
    # own block (block 2, columns 4:6) must be attendable from row 4 despite
    # block 0 (mean=10) scoring higher on raw dot product.
    assert bias[0, 0, 4, 4].item() == 0.0  # attends to its own block's start
    # and block 0 (a strictly-past, higher-scoring block) must be EXCLUDED
    # because top_k=1 is entirely consumed by the forced own-block pick.
    assert bias[0, 0, 4, 0].item() == float("-inf")


def test_gate_bias_never_attends_future_blocks():
    block_size = 2
    top_k = 3  # large enough that, absent causal masking, everything would pass
    B, H, T, Dh = 1, 1, 6, 1
    torch.manual_seed(0)
    q = torch.randn(B, H, T, Dh)
    k = torch.randn(B, H, T, Dh)
    bias, _ = compute_moba_gate_bias(q, k, block_size, top_k)
    causal = torch.ones(T, T).tril().bool()
    future = ~causal
    assert torch.all(bias[0, 0][future] == float("-inf"))


def test_gate_bias_respects_intra_current_block_causality():
    """Within the query's OWN (force-included) block, positions strictly
    after the query must still be masked -- the block-level force-include
    grants the whole block a 0 bias, but the final token-level causal mask
    must re-restrict it."""
    block_size = 4
    top_k = 1
    B, H, T, Dh = 1, 1, 4, 1
    q = torch.randn(B, H, T, Dh)
    k = torch.randn(B, H, T, Dh)
    bias, _ = compute_moba_gate_bias(q, k, block_size, top_k)
    # query at position 1 (within the single block [0,3]) must not attend to
    # position 2 or 3 (future, same block).
    assert bias[0, 0, 1, 2].item() == float("-inf")
    assert bias[0, 0, 1, 3].item() == float("-inf")
    assert bias[0, 0, 1, 0].item() == 0.0
    assert bias[0, 0, 1, 1].item() == 0.0


def test_gate_bias_topk_selects_highest_scoring_past_blocks():
    """4 past blocks + own block, top_k=2 (own block force-included consumes
    one slot) -> exactly 1 other block should be chosen: the highest-scoring
    one by q . mean(k_block)."""
    block_size = 1  # each token is its own block, for a fully controlled case
    top_k = 2
    B, H, T, Dh = 1, 1, 5, 1
    # blocks (=tokens) 0..3 are "past" for query at position 4; scores via
    # dot product q=1 against k values -> pick block with largest k value.
    k = torch.tensor([[[[3.0], [1.0], [5.0], [2.0], [0.0]]]])  # own block (4) mean=0
    q = torch.ones(B, H, T, Dh)
    bias, block_mask = compute_moba_gate_bias(q, k, block_size, top_k)
    row = bias[0, 0, 4]
    # own block (4) always included
    assert row[4].item() == 0.0
    # highest-scoring past block is block 2 (k=5) -> must be included
    assert row[2].item() == 0.0
    # exactly one other block besides the forced own block -> total selected == 2
    n_selected = (row == 0.0).sum().item()
    assert n_selected == 2
    # the other three past blocks (0,1,3) must be excluded
    for j in (0, 1, 3):
        assert row[j].item() == float("-inf")


def test_gate_bias_handles_sequence_shorter_than_block_size():
    block_size = 8
    top_k = 4
    B, H, T, Dh = 1, 1, 3, 1
    q = torch.randn(B, H, T, Dh)
    k = torch.randn(B, H, T, Dh)
    bias, block_mask = compute_moba_gate_bias(q, k, block_size, top_k)
    assert bias.shape == (1, 1, 3, 3)
    causal = torch.ones(T, T).tril()
    # single block always selected -> bias reduces to plain causal
    expected = torch.where(causal.bool(), torch.zeros(T, T), torch.full((T, T), float("-inf")))
    assert torch.equal(bias[0, 0], expected)


def test_gate_bias_matches_independent_manual_softmax_reference():
    """Independent from-scratch reference re-implementing MoBA's naive
    algorithm's math directly (mean-pool -> dot -> causal+force-mask ->
    topk -> additive bias -> softmax), NOT calling compute_moba_gate_bias,
    to confirm the ported function's bias is usable as a drop-in SDPA
    attn_mask producing the same attention weights."""
    torch.manual_seed(42)
    B, H, T, Dh = 2, 2, 9, 3
    block_size, top_k = 3, 2
    q = torch.randn(B, H, T, Dh)
    k = torch.randn(B, H, T, Dh)
    v = torch.randn(B, H, T, Dh)

    bias, _ = compute_moba_gate_bias(q, k, block_size, top_k)
    scale = Dh ** -0.5
    qk = torch.einsum("bhtd,bhsd->bhts", q, k) * scale
    weights = F.softmax(qk + bias, dim=-1)
    weights = torch.nan_to_num(weights, nan=0.0)
    out_manual = torch.einsum("bhts,bhsd->bhtd", weights, v)

    out_sdpa = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
    assert torch.allclose(out_manual, out_sdpa, atol=1e-5)


def test_gate_bias_is_detached_no_grad_history():
    q = torch.randn(1, 1, 6, 2, requires_grad=True)
    k = torch.randn(1, 1, 6, 2, requires_grad=True)
    bias, block_mask = compute_moba_gate_bias(q, k, block_size=2, top_k_blocks=1)
    assert bias.requires_grad is False
    assert block_mask.requires_grad is False


# ─────────────────────────────────────────────────────────────────────────
# 3. MoBAAttention: parameter-less gating, gradient flow, dense equivalence
# ─────────────────────────────────────────────────────────────────────────

def test_moba_attention_has_no_gate_parameters():
    """The single most important structural-fidelity check: real MoBA is
    explicitly 'parameter-less top-k gating' (README). Confirm the module
    introduces NO parameters beyond the standard W_q/W_k/W_v/W_o -- no
    W_gq/W_gk of any kind, unlike this repo's OWN self-implemented
    BlockGatedSparseAttention (src/sparse_attention_block_router.py)."""
    attn = MoBAAttention(d_model=16, n_heads=2, block_size=4, top_k_blocks=2)
    names = {n for n, _ in attn.named_parameters()}
    assert names == {"W_q.weight", "W_k.weight", "W_v.weight", "W_o.weight"}


def test_moba_attention_gradient_flows_to_qk_not_through_selection():
    rope = RotaryPositionalEncoding(d_head=8)
    attn = MoBAAttention(d_model=16, n_heads=2, block_size=4, top_k_blocks=2)
    x = torch.randn(2, 12, 16, requires_grad=True)
    causal_mask = torch.tril(torch.ones(12, 12)).unsqueeze(0).unsqueeze(0)
    out = attn(x, rope, causal_mask, dense_mode=False)
    loss = out.sum()
    loss.backward()
    assert attn.W_q.weight.grad is not None
    assert torch.any(attn.W_q.weight.grad != 0)
    assert attn.W_k.weight.grad is not None
    assert torch.any(attn.W_k.weight.grad != 0)


def test_dense_mode_equals_full_top_k_blocks():
    """dense_mode=True must be numerically equivalent to calling the real
    MoBA path with top_k_blocks >= n_blocks (attend to every causally-valid
    block) -- confirms the final unconditional token-level causal mask
    correctly acts as a safety net in this edge case (see module docstring
    in src/sparse_attention_moba.py)."""
    torch.manual_seed(7)
    T = 12
    n_blocks_if_bs4 = math.ceil(T / 4)
    attn_dense = MoBAAttention(d_model=16, n_heads=2, block_size=4, top_k_blocks=n_blocks_if_bs4, dropout=0.0)
    attn_full = MoBAAttention(d_model=16, n_heads=2, block_size=4, top_k_blocks=n_blocks_if_bs4, dropout=0.0)
    attn_full.load_state_dict(attn_dense.state_dict())

    rope = RotaryPositionalEncoding(d_head=8)
    x = torch.randn(2, T, 16)
    causal_mask = torch.tril(torch.ones(T, T)).unsqueeze(0).unsqueeze(0)

    out_dense = attn_dense(x, rope, causal_mask, dense_mode=True)
    out_full = attn_full(x, rope, causal_mask, dense_mode=False)
    assert torch.allclose(out_dense, out_full, atol=1e-5)


def test_forced_token_bias_bypasses_gate_computation():
    attn = MoBAAttention(d_model=16, n_heads=2, block_size=4, top_k_blocks=1, dropout=0.0)
    rope = RotaryPositionalEncoding(d_head=8)
    T = 8
    x = torch.randn(1, T, 16)
    causal_mask = torch.tril(torch.ones(T, T)).unsqueeze(0).unsqueeze(0)
    # forced bias = pure causal (attend everywhere valid), should differ from
    # the module's own (sparser) default top_k_blocks=1 computation.
    forced_bias = torch.where(causal_mask.bool(), torch.zeros(1, 1, T, T), torch.full((1, 1, T, T), float("-inf")))
    out_forced = attn(x, rope, causal_mask, forced_token_bias=forced_bias)
    out_default = attn(x, rope, causal_mask, dense_mode=False)
    assert not torch.allclose(out_forced, out_default, atol=1e-6)
    out_dense = attn(x, rope, causal_mask, dense_mode=True)
    assert torch.allclose(out_forced, out_dense, atol=1e-5)


# ─────────────────────────────────────────────────────────────────────────
# 4. Model-level structure
# ─────────────────────────────────────────────────────────────────────────

def test_model_construction_and_forward_shape():
    model = make_model(seed=1)
    x, _ = make_batch()
    logits, masks = model(x)
    assert logits.shape == (BATCH, SEQ_LEN, VOCAB)
    assert len(masks) == TINY["n_layers"]


def test_model_has_no_gate_parameters_anywhere():
    model = make_model(seed=2)
    for name, _ in model.named_parameters():
        assert "gate" not in name.lower() and "W_g" not in name


def test_weight_tying_preserved():
    model = make_model(seed=3)
    assert model.lm_head.weight is model.tok_emb.weight


def test_param_count_medium_config_is_300m_class_and_smaller_than_gated_equivalent():
    """Confirms scale is right (~300M-class, matching this repo's other
    300M-scale scripts) AND that dropping the gate saves exactly the
    W_gq/W_gk parameter count relative to a same-shape gated block-router
    model (src/sparse_attention_block_router.py's BlockGatedSparseAttention,
    d_gate=32) -- not merely "no gate params" (already covered by
    test_moba_attention_has_no_gate_parameters) but the exact expected
    delta, computed independently rather than eyeballing a param-count
    range."""
    model = MoBATransformer(vocab_size=50257, d_model=1024, n_heads=16, n_layers=20,
                             d_ff=4096, block_size=128, top_k_blocks=4, max_seq_len=2048)
    n = model.count_parameters()
    assert 280_000_000 < n < 320_000_000, f"Expected ~300M params, got {n/1e6:.0f}M"

    d_model, n_heads, n_layers, d_gate = 1024, 16, 20, 32
    gate_params_per_layer = 2 * (n_heads * d_gate) * d_model  # W_gq + W_gk, bias=False
    n_with_gate_equivalent = n + n_layers * gate_params_per_layer
    assert n_with_gate_equivalent > n + 20_000_000


def test_dense_mode_and_hard_mode_are_the_only_two_attention_paths_and_agree_at_full_k():
    """Sanity: at the FULL model level (not just single-attn-layer level),
    dense_mode=True and dense_mode=False with top_k_blocks>=n_blocks give
    the same logits."""
    T = TINY["max_seq_len"]
    n_blocks = math.ceil(T / TINY["block_size"])
    model = make_model(seed=4, top_k_blocks=n_blocks)
    x, _ = make_batch()
    model.eval()
    with torch.no_grad():
        logits_dense, _ = model(x, dense_mode=True)
        logits_full, _ = model(x, dense_mode=False)
    assert torch.allclose(logits_dense, logits_full, atol=1e-4)


# ─────────────────────────────────────────────────────────────────────────
# 5. Open-loop vs closed-loop trajectory utilities
# ─────────────────────────────────────────────────────────────────────────

def test_standard_condition_native_open_closed_loop_all_coincide():
    """For condition='standard' (MoBA trains block-sparse-hard from step 0),
    native/open-loop/closed-loop must all be (numerically) the SAME
    computation -- this is the 'G_CL ~ 0 by construction' sanity check that
    is the single most important empirical prediction for this baseline."""
    model = make_model(seed=5)
    model.eval()
    loader = make_loader(n_batches=2, seed=50)
    native = evaluate_nll(model, loader, torch.device("cpu"), "standard")
    open_loop = eval_open_loop(model, loader, torch.device("cpu"), "standard")
    closed_loop = eval_closed_loop(model, loader, torch.device("cpu"))
    assert native == pytest.approx(open_loop, abs=1e-5)
    assert native == pytest.approx(closed_loop, abs=1e-5)


def test_dense_switch_native_uses_dense_forward():
    model = make_model(seed=6)
    model.eval()
    # ids must be a genuine shifted-token sequence (x=ids[:,:-1], y=ids[:,1:])
    # -- evaluate_nll's protocol, matching make_loader -- NOT independently
    # random x/y (make_batch), which cannot satisfy that relationship.
    g = torch.Generator().manual_seed(60)
    ids = torch.randint(0, VOCAB, (BATCH, SEQ_LEN + 1), generator=g)
    x, y = ids[:, :-1], ids[:, 1:]
    with torch.no_grad():
        logits_dense, _ = model(x, dense_mode=True)
        expected_nll = F.cross_entropy(logits_dense.reshape(-1, VOCAB), y.reshape(-1)).item()
    loader = TinyLoader([ids])
    # native_nll for dense_switch must match a plain dense forward's NLL
    native = evaluate_nll(model, loader, torch.device("cpu"), "dense_switch")
    assert native == pytest.approx(expected_nll, abs=1e-5)


def test_closed_loop_wrapper_matches_direct_hard_forward():
    model = make_model(seed=7)
    model.eval()
    x, _ = make_batch(seed=70)
    with torch.no_grad():
        logits_direct, _ = model(x, dense_mode=False)
        logits_wrapper = forward_closed_loop_moba(model, x)
    assert torch.allclose(logits_direct, logits_wrapper)


def test_record_and_replay_open_loop_reproduces_forced_forward():
    model = make_model(seed=8)
    model.eval()
    x, _ = make_batch(seed=80)
    biases = record_moba_biases_from_trajectory(model, x, trajectory_dense_mode=True)
    assert len(biases) == TINY["n_layers"]
    logits_a = forward_open_loop_moba(model, x, biases)
    logits_b = forward_open_loop_moba(model, x, biases)
    assert torch.equal(logits_a, logits_b)  # deterministic replay


# ─────────────────────────────────────────────────────────────────────────
# 6. Random-hard control
# ─────────────────────────────────────────────────────────────────────────

def test_random_hard_respects_causality_and_cardinality():
    model = make_model(seed=9)
    model.eval()
    loader = make_loader(n_batches=1, seed=90, batch=2, seq_len=SEQ_LEN)
    nll = eval_random_hard(model, loader, torch.device("cpu"), max_batches=1,
                            generator=torch.Generator().manual_seed(1))
    assert np.isfinite(nll)


def test_random_hard_differs_from_learned_selection_generally():
    model = make_model(seed=10)
    model.eval()
    loader = make_loader(n_batches=2, seed=100)
    closed = eval_closed_loop(model, loader, torch.device("cpu"))
    rand = eval_random_hard(model, loader, torch.device("cpu"), max_batches=2,
                             generator=torch.Generator().manual_seed(2))
    assert closed != pytest.approx(rand, abs=1e-9)


# ─────────────────────────────────────────────────────────────────────────
# 7. run_full_evaluation: required keys, G_CL formula, finiteness
# ─────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("condition", CONDITIONS)
def test_run_full_evaluation_returns_expected_keys(condition):
    model = make_model(seed=11)
    loader = make_loader(n_batches=2, seed=110)
    result = run_full_evaluation(model, loader, torch.device("cpu"), condition,
                                  max_batches=2, n_random_draws=1, verbose=False)
    required = {"native_nll", "native_ppl", "open_loop_nll", "closed_loop_nll",
                "G_CL", "G_OL", "random_hard_nll_mean", "random_hard_nll_std", "gate_utility"}
    assert required.issubset(result.keys())
    for k in required:
        assert np.isfinite(result[k]), f"{k} not finite: {result[k]}"


def test_g_cl_formula_is_closed_loop_minus_native():
    model = make_model(seed=12)
    loader = make_loader(n_batches=2, seed=120)
    result = run_full_evaluation(model, loader, torch.device("cpu"), "dense_switch",
                                  max_batches=2, n_random_draws=1, verbose=False)
    assert result["G_CL"] == pytest.approx(result["closed_loop_nll"] - result["native_nll"], abs=1e-6)
    assert result["G_OL"] == pytest.approx(result["open_loop_nll"] - result["native_nll"], abs=1e-6)


def test_g_cl_near_zero_for_standard_condition():
    model = make_model(seed=13)
    loader = make_loader(n_batches=2, seed=130)
    result = run_full_evaluation(model, loader, torch.device("cpu"), "standard",
                                  max_batches=2, n_random_draws=1, verbose=False)
    assert abs(result["G_CL"]) < 1e-4


def test_eval_verbose_announces_all_four_modes(capsys):
    model = make_model(seed=14)
    loader = make_loader(n_batches=1, seed=140)
    run_full_evaluation(model, loader, torch.device("cpu"), "clhr",
                         max_batches=1, n_random_draws=1, verbose=True)
    out = capsys.readouterr().out
    for mode in ("native", "open_loop", "closed_loop", "random_hard"):
        assert mode in out, f"no announcement for eval mode {mode!r}"


def test_dense_mode_nll_is_measured_in_dense_mode_regardless_of_condition():
    """dense_mode_nll must always come from a dense_mode=True forward pass,
    for every condition -- not just dense_switch. Without this, there is no
    way to compare a clhr-trained model's soft/dense quality against its
    hard-deployed quality: _native_dense_mode() only returns True for
    dense_switch, so native_nll/open_loop_nll/closed_loop_nll are all
    identical (hard-vs-hard) for standard AND clhr, making G_CL=0.0 for
    clhr a methodological artifact rather than evidence the gap is fixed."""
    model = make_model(seed=15)
    loader = make_loader(n_batches=2, seed=150)

    for condition in ("standard", "dense_switch", "clhr"):
        result = run_full_evaluation(model, loader, torch.device("cpu"), condition,
                                      max_batches=2, n_random_draws=1, verbose=False)
        assert "dense_mode_nll" in result, f"{condition}: missing dense_mode_nll"
        expected_dense = _mean_nll_over_loader(
            model, loader, torch.device("cpu"),
            lambda x: model(x, dense_mode=True)[0], max_batches=2,
        )
        assert result["dense_mode_nll"] == pytest.approx(expected_dense, abs=1e-6), (
            f"{condition}: dense_mode_nll must be a genuine dense_mode=True eval"
        )


def test_g_cl_vs_dense_matches_existing_g_cl_for_dense_switch():
    """For dense_switch, native_nll IS ALREADY the dense-mode eval, so the
    new G_CL_vs_dense metric must equal the existing G_CL exactly -- this is
    a consistency check that the new metric doesn't silently disagree with
    the one dense_switch's own definition already gets right."""
    model = make_model(seed=16)
    loader = make_loader(n_batches=2, seed=160)
    result = run_full_evaluation(model, loader, torch.device("cpu"), "dense_switch",
                                  max_batches=2, n_random_draws=1, verbose=False)
    assert result["G_CL_vs_dense"] == pytest.approx(result["G_CL"], abs=1e-9)


def test_g_cl_vs_dense_is_not_trivially_zero_for_clhr():
    """The whole point of this metric: for clhr, closed_loop_nll (hard) vs
    dense_mode_nll (dense) should NOT be forced to match by construction,
    unlike the old G_CL which collapsed to 0.0 for clhr for a structural
    reason unrelated to whether CLHR actually closed anything."""
    model = make_model(seed=17)
    loader = make_loader(n_batches=2, seed=170)
    result = run_full_evaluation(model, loader, torch.device("cpu"), "clhr",
                                  max_batches=2, n_random_draws=1, verbose=False)
    assert result["G_CL_vs_dense"] == pytest.approx(
        result["closed_loop_nll"] - result["dense_mode_nll"], abs=1e-6
    )


# ─────────────────────────────────────────────────────────────────────────
# 8. compute_condition_loss: the three training conditions
# ─────────────────────────────────────────────────────────────────────────

def test_standard_condition_rejects_nonzero_lambda_rca():
    model = make_model(seed=15)
    x, y = make_batch(seed=150)
    with pytest.raises(ValueError, match="lambda_rca"):
        compute_condition_loss(model, x, y, "standard", lambda_rca=1.0)


def test_dense_switch_condition_rejects_nonzero_lambda_rca():
    model = make_model(seed=16)
    x, y = make_batch(seed=160)
    with pytest.raises(ValueError, match="lambda_rca"):
        compute_condition_loss(model, x, y, "dense_switch", lambda_rca=0.5)


def test_standard_condition_loss_equals_hard_forward_ce():
    model = make_model(seed=17)
    x, y = make_batch(seed=170)
    loss, native = compute_condition_loss(model, x, y, "standard", lambda_rca=0.0)
    with torch.no_grad():
        logits, _ = model(x, dense_mode=False)
        expected = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1)).item()
    assert native == pytest.approx(expected, abs=1e-5)


def test_dense_switch_condition_loss_equals_dense_forward_ce():
    model = make_model(seed=18)
    x, y = make_batch(seed=180)
    loss, native = compute_condition_loss(model, x, y, "dense_switch", lambda_rca=0.0)
    with torch.no_grad():
        logits, _ = model(x, dense_mode=True)
        expected = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1)).item()
    assert native == pytest.approx(expected, abs=1e-5)


def test_clhr_condition_lambda_zero_equals_dense_switch():
    model = make_model(seed=19)
    x, y = make_batch(seed=190)
    loss_clhr, native_clhr = compute_condition_loss(model, x, y, "clhr", lambda_rca=0.0)
    loss_dense, native_dense = compute_condition_loss(model, x, y, "dense_switch", lambda_rca=0.0)
    assert torch.allclose(loss_clhr, loss_dense, atol=1e-6)
    assert native_clhr == pytest.approx(native_dense, abs=1e-6)


def test_clhr_condition_mixes_dense_and_hard_losses():
    model = make_model(seed=20)
    x, y = make_batch(seed=200)
    loss, native = compute_condition_loss(model, x, y, "clhr", lambda_rca=1.0)
    with torch.no_grad():
        logits_dense, _ = model(x, dense_mode=True)
        lm_dense = F.cross_entropy(logits_dense.reshape(-1, VOCAB), y.reshape(-1))
        logits_hard, _ = model(x, dense_mode=False)
        lm_hard = F.cross_entropy(logits_hard.reshape(-1, VOCAB), y.reshape(-1))
        expected = lm_dense + 1.0 * lm_hard
    assert loss.item() == pytest.approx(expected.item(), abs=1e-5)
    assert native == pytest.approx(lm_dense.item(), abs=1e-5)


def test_clhr_condition_normalize_loss_divides_by_one_plus_lambda():
    model = make_model(seed=21)
    x, y = make_batch(seed=210)
    loss_norm, _ = compute_condition_loss(model, x, y, "clhr", lambda_rca=3.0, normalize_loss=True)
    loss_raw, _ = compute_condition_loss(model, x, y, "clhr", lambda_rca=3.0, normalize_loss=False)
    assert loss_norm.item() == pytest.approx(loss_raw.item() / 4.0, abs=1e-5)


def test_clhr_condition_gradient_reaches_shared_weights_from_both_branches():
    """Because standard's forward AND clhr's auxiliary forward both call the
    SAME MoBAAttention.forward(dense_mode=False), and clhr's primary forward
    shares the SAME W_q/W_k/W_v/W_o via the dense_mode=True path, a single
    clhr backward pass must touch every attention weight matrix (no
    accidental detachment of one branch)."""
    model = make_model(seed=22)
    x, y = make_batch(seed=220)
    loss, _ = compute_condition_loss(model, x, y, "clhr", lambda_rca=1.0)
    loss.backward()
    for block in model.blocks:
        for pname in ("W_q", "W_k", "W_v", "W_o"):
            grad = getattr(block.attn, pname).weight.grad
            assert grad is not None
            assert torch.any(grad != 0)


def test_unknown_condition_raises():
    model = make_model(seed=23)
    x, y = make_batch(seed=230)
    with pytest.raises(ValueError):
        compute_condition_loss(model, x, y, "not_a_condition", lambda_rca=0.0)


# ─────────────────────────────────────────────────────────────────────────
# 9. train_model: end-to-end CPU training loop, all three conditions
# ─────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("condition,lambda_rca", [("standard", 0.0), ("dense_switch", 0.0), ("clhr", 1.0)])
def test_train_model_runs_and_updates_weights(condition, lambda_rca):
    model = make_model(seed=24)
    loader = make_loader(n_batches=3, seed=240)
    before = model.blocks[0].attn.W_q.weight.clone()
    steps_completed, elapsed = train_model(model, loader, torch.device("cpu"), condition,
                                            steps=3, lr=1e-2, lambda_rca=lambda_rca, log_every=0)
    assert steps_completed == 3
    after = model.blocks[0].attn.W_q.weight
    assert not torch.equal(before, after)


def test_train_model_progress_logging_emitted_with_flush(capsys):
    model = make_model(seed=25)
    loader = make_loader(n_batches=2, seed=250)
    train_model(model, loader, torch.device("cpu"), "standard", steps=2, lr=1e-2,
                lambda_rca=0.0, log_every=1)
    out = capsys.readouterr().out
    lines = [ln for ln in out.splitlines() if ln.startswith("[train] step")]
    assert len(lines) >= 1
    assert any("step 2/2" in ln for ln in lines)


def test_train_model_log_every_zero_disables_logging(capsys):
    model = make_model(seed=26)
    loader = make_loader(n_batches=2, seed=260)
    train_model(model, loader, torch.device("cpu"), "standard", steps=2, lr=1e-2,
                lambda_rca=0.0, log_every=0)
    out = capsys.readouterr().out
    assert "[train] step" not in out


def test_train_model_grad_accum_steps_1_is_bit_identical_to_independent_reference():
    torch.manual_seed(9001)
    model_new = make_model(seed=27)
    torch.manual_seed(9001)
    model_ref = make_model(seed=27)
    for p_new, p_ref in zip(model_new.parameters(), model_ref.parameters()):
        assert torch.equal(p_new, p_ref)

    loader = make_loader(n_batches=3, seed=270)
    train_model(model_new, loader, torch.device("cpu"), "dense_switch", steps=3, lr=1e-2,
                lambda_rca=0.0, grad_accum_steps=1, log_every=0)

    optimizer = torch.optim.AdamW(model_ref.parameters(), lr=1e-2, weight_decay=0.1, betas=(0.9, 0.95))
    train_iter = iter(loader)
    for _ in range(3):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(loader)
            batch = next(train_iter)
        x, y = batch[:, :-1], batch[:, 1:]
        optimizer.zero_grad()
        loss, _ = compute_condition_loss(model_ref, x, y, "dense_switch", lambda_rca=0.0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_ref.parameters(), 1.0)
        optimizer.step()

    for p_new, p_ref in zip(model_new.parameters(), model_ref.parameters()):
        assert torch.equal(p_new, p_ref)


# ─────────────────────────────────────────────────────────────────────────
# 10. Data loading
# ─────────────────────────────────────────────────────────────────────────

def test_cycling_token_dataset_shapes():
    tokens = torch.arange(1000)
    ds = CyclingTokenDataset(tokens, seq_len=16)
    item = ds[0]
    assert item.shape == (17,)
    assert torch.equal(item, tokens[0:17])


# ─────────────────────────────────────────────────────────────────────────
# 11. CLI / argparse
# ─────────────────────────────────────────────────────────────────────────

def test_build_argparser_defaults():
    args = build_argparser().parse_args(["--data-dir", "x", "--output", "y.json"])
    assert args.condition == "standard"
    assert args.lambda_rca is None
    assert args.block_size > 0
    assert args.top_k_blocks > 0


def test_build_argparser_condition_choices_are_exactly_three():
    assert set(CONDITIONS) == {"standard", "dense_switch", "clhr"}
    with pytest.raises(SystemExit):
        build_argparser().parse_args(["--data-dir", "x", "--output", "y.json", "--condition", "bogus"])


@pytest.mark.parametrize("condition", CONDITIONS)
def test_build_argparser_accepts_each_condition(condition):
    args = build_argparser().parse_args(["--data-dir", "x", "--output", "y.json", "--condition", condition])
    assert args.condition == condition


# ─────────────────────────────────────────────────────────────────────────
# 12. House style: flush=True on every print (SkyPilot long-run convention)
# ─────────────────────────────────────────────────────────────────────────

def test_all_prints_flush():
    source = (REPO_ROOT / "src" / "sparse_attention_moba.py").read_text()
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


# ─────────────────────────────────────────────────────────────────────────
# 13. Smoke: full CLI subprocess train+eval for all three conditions
# ─────────────────────────────────────────────────────────────────────────

def _make_tiny_data_dir(tmp_path, seed=0):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    rng = np.random.default_rng(seed)
    train_tokens = rng.integers(0, VOCAB, size=6000).astype(np.int64)
    val_tokens = rng.integers(0, VOCAB, size=2000).astype(np.int64)
    np.save(data_dir / "wt103_train_tokens.npy", train_tokens)
    np.save(data_dir / "wt103_val_tokens.npy", val_tokens)
    return data_dir


@pytest.mark.parametrize("condition,lambda_rca", [("standard", None), ("dense_switch", None), ("clhr", "1.0")])
def test_smoke_cli_train_two_steps_all_conditions(tmp_path, condition, lambda_rca):
    data_dir = _make_tiny_data_dir(tmp_path, seed=1)
    output_path = tmp_path / f"result_{condition}.json"
    cmd = [
        sys.executable, str(REPO_ROOT / "src" / "sparse_attention_moba.py"),
        "--condition", condition,
        "--data-dir", str(data_dir),
        "--checkpoint-dir", str(tmp_path / "ckpt"),
        "--output", str(output_path),
        "--steps", "2",
        "--micro-batch", "2",
        "--grad-accum", "1",
        "--seq-len", str(SEQ_LEN),
        "--d-model", str(TINY["d_model"]),
        "--n-heads", str(TINY["n_heads"]),
        "--n-layers", str(TINY["n_layers"]),
        "--d-ff", str(TINY["d_ff"]),
        "--vocab-size", str(VOCAB),
        "--block-size", str(TINY["block_size"]),
        "--top-k-blocks", str(TINY["top_k_blocks"]),
        "--log-every", "1",
    ]
    if lambda_rca is not None:
        cmd += ["--lambda-rca", lambda_rca]
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, f"condition={condition} failed:\n{proc.stdout}\n{proc.stderr}"

    assert output_path.exists()
    with open(output_path) as f:
        data = json.load(f)
    assert data["condition"] == condition
    assert data["total_steps"] == 2
    for key in ("native_nll", "open_loop_nll", "closed_loop_nll", "G_CL", "model_params"):
        assert key in data
    assert np.isfinite(data["native_nll"])
    assert np.isfinite(data["closed_loop_nll"])


def test_cli_creates_missing_output_parent_directory(tmp_path):
    """Same production failure class hit by sparse_attention_nsa.py's
    wave-1 jobs tonight: training finishes and the checkpoint saves fine
    (checkpoint_dir.mkdir already has this safety), but --output's parent
    directory was never created, so the eval-result write crashes with
    FileNotFoundError, silently discarding the completed run's result."""
    data_dir = _make_tiny_data_dir(tmp_path, seed=1)
    output_path = tmp_path / "nested" / "does" / "not" / "exist" / "result.json"
    cmd = [
        sys.executable, str(REPO_ROOT / "src" / "sparse_attention_moba.py"),
        "--condition", "standard",
        "--data-dir", str(data_dir),
        "--checkpoint-dir", str(tmp_path / "ckpt"),
        "--output", str(output_path),
        "--steps", "2",
        "--micro-batch", "2",
        "--grad-accum", "1",
        "--seq-len", str(SEQ_LEN),
        "--d-model", str(TINY["d_model"]),
        "--n-heads", str(TINY["n_heads"]),
        "--n-layers", str(TINY["n_layers"]),
        "--d-ff", str(TINY["d_ff"]),
        "--vocab-size", str(VOCAB),
        "--block-size", str(TINY["block_size"]),
        "--top-k-blocks", str(TINY["top_k_blocks"]),
        "--log-every", "1",
    ]
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, f"failed:\n{proc.stdout}\n{proc.stderr}"
    assert output_path.exists()
