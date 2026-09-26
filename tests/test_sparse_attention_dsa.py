"""
Tests for src/sparse_attention_dsa.py -- a from-scratch port of DeepSeek
Sparse Attention (DSA)'s "lightning indexer" + top-k mechanism, with a
`standard` condition (dense warmup then an abrupt switch to indexer-gated
hard attention -- DSA's own reported training recipe, i.e. the soft/dense-
then-hard pattern CLHR targets) and a `clhr` condition (closed-loop mixing
of the dense and hard forward passes across a transition window straddling
the switch step).

Written and run FIRST, and expected to fail with an ImportError until
src/sparse_attention_dsa.py exists -- matches this repo's TDD convention
(see tests/test_moe_soft_to_hard.py).

MECHANISM PROVENANCE (see src/sparse_attention_dsa.py module docstring for
the full breakdown of verified-from-source vs. documentation vs. unknown):
the LightningIndexer's module composition (per-head query projection, a
single shared low-rank key projection + LayerNorm, per-head combination
weights, top-k hard masking added to the *main* attention's raw scores
before softmax -- not to the indexer's own scores) mirrors the verified
structure of the `Indexer` class and its use inside `MLA.forward` in
https://raw.githubusercontent.com/deepseek-ai/DeepSeek-V3.2-Exp/main/inference/model.py.
The exact scalar-combination arithmetic inside the real fp8 `fp8_index`
kernel (inference/kernel.py) was not retrievable (compiled kernel, not
visible Python), so the combination formula here is a standard
reconstruction consistent with the verified module composition, not a
byte-for-byte-verified formula. The two-phase "dense warmup then switch"
TRAINING RECIPE was not found anywhere in the accessible repo (inference-
only reference implementation; no training code; no arXiv/technical-report
link in the README) -- it is implemented here per this project's own task
specification of DSA's publicly reported behavior, not verified from
DeepSeek's source.
"""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from sparse_attention_dsa import (  # noqa: E402
    LightningIndexer,
    DSAAttention,
    DSATransformer,
    causal_valid_mask,
    topk_hard_mask,
    indexer_distillation_loss,
    step_mode,
    dense_loss,
    hard_loss,
    clhr_mixed_loss,
    native_nll,
    closed_loop_nll,
    open_loop_nll,
    run_full_evaluation,
    select_device,
    train_experiment,
)

TINY = dict(d_model=32, n_heads=2, n_layers=2, d_ff=64, index_n_heads=2, index_head_dim=8, max_seq_len=16)
VOCAB = 100
SEQ_LEN = 16
BATCH = 4


def make_model(seed=0):
    torch.manual_seed(seed)
    return DSATransformer(VOCAB, **TINY)


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


def make_loader(n_batches=2, seed=0):
    return TinyLoader([make_batch(seed=seed + i) for i in range(n_batches)])


# --- 1. LightningIndexer scoring module ---

def test_indexer_output_shape_and_finite():
    idx = LightningIndexer(d_model=TINY["d_model"], index_n_heads=2, index_head_dim=8)
    x = torch.randn(BATCH, SEQ_LEN, TINY["d_model"])
    score = idx(x)
    assert score.shape == (BATCH, SEQ_LEN, SEQ_LEN)
    assert torch.isfinite(score).all()


def test_indexer_params_receive_gradient_from_distillation_loss():
    # The hard top-k mask is non-differentiable w.r.t. index_score (it is a
    # constant 0/-inf additive mask keyed only on *which* indices were
    # selected), so the indexer receives NO gradient from the main LM loss
    # once attention is hardened. It must be trained via a separate signal.
    # This distillation loss (KL of indexer-softmax against detached dense-
    # attention-softmax) is that signal -- a design choice this port makes
    # explicit and tests directly, since it isn't visible in DeepSeek's
    # (training-code-free) reference repo.
    idx = LightningIndexer(d_model=TINY["d_model"], index_n_heads=2, index_head_dim=8)
    x = torch.randn(BATCH, SEQ_LEN, TINY["d_model"])
    score = idx(x)
    causal = causal_valid_mask(SEQ_LEN, score.device)
    dense_probs = F.softmax(
        torch.randn(BATCH, SEQ_LEN, SEQ_LEN).masked_fill(~causal, float("-inf")), dim=-1
    )
    loss = indexer_distillation_loss(score, dense_probs, causal)
    loss.backward()
    for p in idx.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()


# --- 2. top-k mask / dense-vs-sparse switch logic ---

def test_topk_mask_selects_exactly_k_when_enough_causal_history():
    torch.manual_seed(0)
    score = torch.randn(1, SEQ_LEN, SEQ_LEN)
    causal = causal_valid_mask(SEQ_LEN, score.device)
    k = 4
    mask = topk_hard_mask(score, k, causal)
    allowed = mask == 0
    for i in range(SEQ_LEN):
        expected = min(k, i + 1)
        assert allowed[0, i].sum().item() == expected


def test_topk_mask_never_leaks_future_positions():
    # k > seq_len deliberately stresses the tie-break-under-`-inf` path: a
    # naive torch.topk over causal-masked (-inf-padded) scores could, for
    # early rows with fewer than k valid positions, select some -inf-valued
    # *future* positions into the "top k" and incorrectly mark them allowed.
    torch.manual_seed(1)
    score = torch.randn(1, SEQ_LEN, SEQ_LEN)
    causal = causal_valid_mask(SEQ_LEN, score.device)
    k = SEQ_LEN + 8
    mask = topk_hard_mask(score, k, causal)
    allowed = mask == 0
    for i in range(SEQ_LEN):
        assert not allowed[0, i, i + 1:].any()
        assert allowed[0, i, : i + 1].sum().item() == min(k, i + 1)


def test_attention_is_genuinely_dense_before_switch():
    model = make_model()
    x, _ = make_batch()
    logits, aux = model(x, dense_mode=True, topk=2)
    causal = causal_valid_mask(SEQ_LEN, x.device)
    for layer_aux in aux:
        probs = layer_aux["dense_probs"]
        valid_probs = probs.masked_select(causal.unsqueeze(0).expand_as(probs))
        assert (valid_probs > 0).all()
    assert torch.isfinite(logits).all()


def test_attention_is_genuinely_sparse_after_switch():
    model = make_model()
    x, _ = make_batch()
    k = 3
    logits, aux = model(x, dense_mode=False, topk=k)
    for layer_aux in aux:
        probs = layer_aux["hard_probs"]
        nonzero = probs > 1e-12
        for i in range(SEQ_LEN):
            expected = min(k, i + 1)
            assert nonzero[0, i].sum().item() == expected
            assert not nonzero[0, i, i + 1:].any()
    assert torch.isfinite(logits).all()


# --- 3. standard-condition switch scheduling ---

def test_step_mode_standard_is_abrupt_cutover():
    assert step_mode(0, switch_step=20, transition_window=10, condition="standard") == "dense"
    assert step_mode(19, switch_step=20, transition_window=10, condition="standard") == "dense"
    assert step_mode(20, switch_step=20, transition_window=10, condition="standard") == "hard"
    assert step_mode(99, switch_step=20, transition_window=10, condition="standard") == "hard"


def test_step_mode_clhr_has_transition_window_at_the_switch():
    assert step_mode(19, switch_step=20, transition_window=10, condition="clhr") == "dense"
    assert step_mode(20, switch_step=20, transition_window=10, condition="clhr") == "mixed"
    assert step_mode(29, switch_step=20, transition_window=10, condition="clhr") == "mixed"
    assert step_mode(30, switch_step=20, transition_window=10, condition="clhr") == "hard"


def test_step_mode_rejects_unknown_condition():
    with pytest.raises(ValueError):
        step_mode(5, switch_step=20, transition_window=10, condition="bogus")


# --- 4. CLHR mixing formula (matches src/moe_soft_to_hard.py:clhr_loss's
#         (L_soft + lambda_rca * L_hard) / (1 + lambda_rca) convention --
#         there is no shared/reusable clhr module in this repo to import,
#         so this replicates that exact formula inline) ---

def test_dense_loss_and_hard_loss_are_finite_scalars():
    model = make_model()
    x, y = make_batch()
    ld, _ = dense_loss(model, x, y, indexer_aux_weight=0.1)
    lh, _ = hard_loss(model, x, y, topk=2, indexer_aux_weight=0.1)
    assert ld.dim() == 0 and torch.isfinite(ld)
    assert lh.dim() == 0 and torch.isfinite(lh)


def test_clhr_mixed_loss_lambda_zero_equals_dense_only():
    model = make_model()
    x, y = make_batch()
    combined, l_soft, l_hard = clhr_mixed_loss(
        model, x, y, lambda_rca=0.0, topk=2, indexer_aux_weight=0.0
    )
    assert torch.equal(combined, l_soft)
    assert torch.equal(l_hard, l_soft)


def test_clhr_mixed_loss_is_weighted_average():
    model = make_model()
    x, y = make_batch()
    combined, l_soft, l_hard = clhr_mixed_loss(
        model, x, y, lambda_rca=1.0, topk=2, indexer_aux_weight=0.0
    )
    expected = (l_soft + 1.0 * l_hard) / (1.0 + 1.0)
    assert torch.allclose(combined, expected)


# --- 5. G_CL-style eval metrics (G_CL = closed_loop_nll - native_nll,
#         matching src/moe_soft_to_hard.py and src/sparse_attention_fineweb.py) ---

def test_native_open_closed_nll_are_finite():
    model = make_model()
    loader = make_loader()
    device = torch.device("cpu")
    assert torch.isfinite(torch.tensor(native_nll(model, loader, device)))
    assert torch.isfinite(torch.tensor(open_loop_nll(model, loader, device, topk=2)))
    assert torch.isfinite(torch.tensor(closed_loop_nll(model, loader, device, topk=2)))


def test_metrics_dict_has_expected_keys_and_gcl_formula():
    model = make_model()
    loader = make_loader()
    device = torch.device("cpu")
    metrics = run_full_evaluation(model, loader, device, topk=2)
    for key in ("native_nll", "open_loop_nll", "closed_loop_nll", "G_CL", "G_OL", "compounding_ratio"):
        assert key in metrics
    assert metrics["G_CL"] == pytest.approx(
        metrics["closed_loop_nll"] - metrics["native_nll"], abs=1e-6
    )
    assert metrics["G_OL"] == pytest.approx(
        metrics["open_loop_nll"] - metrics["native_nll"], abs=1e-6
    )


def test_select_device_returns_valid_device():
    d = select_device()
    assert d.type in ("cuda", "mps", "cpu")


# --- 6. end-to-end smoke (synthetic data, CPU, tiny model) ---

def test_smoke_train_few_steps_standard(tmp_path):
    result = train_experiment(
        condition="standard", seed=0, total_steps=4, switch_step=2, transition_window=2,
        topk=2, indexer_aux_weight=0.1, lambda_rca=1.0, batch_size=BATCH, seq_len=SEQ_LEN,
        vocab_size=VOCAB, model_kwargs=TINY, device=torch.device("cpu"),
        checkpoint_dir=str(tmp_path / "ckpts"), output_path=str(tmp_path / "out.json"),
        synthetic_data=True, eval_max_batches=1, log_every=1,
    )
    assert "final_metrics" in result
    assert "at_switch_metrics" in result
    assert (tmp_path / "out.json").exists()


def test_smoke_train_few_steps_clhr(tmp_path):
    result = train_experiment(
        condition="clhr", seed=0, total_steps=4, switch_step=2, transition_window=2,
        topk=2, indexer_aux_weight=0.1, lambda_rca=1.0, batch_size=BATCH, seq_len=SEQ_LEN,
        vocab_size=VOCAB, model_kwargs=TINY, device=torch.device("cpu"),
        checkpoint_dir=str(tmp_path / "ckpts2"), output_path=str(tmp_path / "out2.json"),
        synthetic_data=True, eval_max_batches=1, log_every=1,
    )
    assert "final_metrics" in result
    assert "at_switch_metrics" in result
