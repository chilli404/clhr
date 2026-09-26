import sys
from pathlib import Path

import torch
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO.parent / "routing-absorption"))

from src.models.config import ModelConfig
from src.models.transformer import GatedTransformer
from sparse_attention_ssa_style import objective_matched_task_loss, ssa_style_forward


def tiny_model():
    config = ModelConfig(
        vocab_size=32,
        max_seq_len=16,
        n_layers=2,
        d_model=16,
        n_heads=2,
        d_ff=32,
        dropout=0.0,
        d_gate=4,
        sparsity_mode="soft",
        sparsity_k=4,
        gate_temperature=1.0,
    )
    return GatedTransformer(config)


def test_both_stream_choices_have_finite_outputs():
    model = tiny_model()
    tokens = torch.randint(0, 32, (2, 8))
    for propagate_soft in (True, False):
        logits, alignment, chosen = ssa_style_forward(
            model, tokens, propagate_soft=propagate_soft, topk=4
        )
        assert logits.shape == (2, 8, 32)
        assert torch.isfinite(logits).all()
        assert torch.isfinite(alignment)
        assert chosen is propagate_soft


def test_alignment_trains_router_and_backbone_when_hard_stream_propagates():
    model = tiny_model()
    tokens = torch.randint(0, 32, (2, 8))
    logits, alignment, _ = ssa_style_forward(
        model, tokens, propagate_soft=False, topk=4
    )
    (logits.square().mean() + 10.0 * alignment).backward()
    attn = model.layers[0].attention
    assert attn.W_gq.weight.grad is not None
    assert torch.isfinite(attn.W_gq.weight.grad).all()
    assert attn.W_q.weight.grad is not None
    assert torch.isfinite(attn.W_q.weight.grad).all()


def test_objective_matched_weights_equal_at_matched_probability():
    soft_total = torch.tensor(2.0)
    hard_task = torch.tensor(3.0)
    losses = {"loss_total": soft_total, "loss_task": hard_task}
    hard_probability = 0.3 / 1.3
    soft = objective_matched_task_loss(losses, True, hard_probability, 0.3)
    hard = objective_matched_task_loss(losses, False, hard_probability, 0.3)
    assert soft.item() == pytest.approx(1.3 * soft_total.item())
    assert hard.item() == pytest.approx(1.3 * hard_task.item())
