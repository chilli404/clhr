"""Tests for the 1B-scale preset and native_hard (hard-from-scratch STE)
condition added to src/sparse_attention_fineweb.py, ported from the
reference implementation in src/sparse_attention_300m.py's
GatedSparseAttention._forward_hard_ste.

Written and run FIRST (must fail before the 1b preset / native_hard
condition exist). Tiny overrides throughout for CPU speed; the "1b" preset
itself is only checked for parameter count, never instantiated in full.
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestOneBillionPreset:
    def test_1b_preset_exists(self):
        from sparse_attention_fineweb import MODEL_CONFIGS
        assert "1b" in MODEL_CONFIGS

    def test_1b_preset_param_count_near_one_billion(self):
        from sparse_attention_fineweb import MODEL_CONFIGS, SparseTransformer
        cfg = MODEL_CONFIGS["1b"]
        model = SparseTransformer(**cfg)
        n_params = model.count_parameters()
        assert 0.85e9 <= n_params <= 1.15e9, (
            f"1b preset has {n_params / 1e9:.3f}B params, expected ~1.0B +-15%"
        )

    def test_1b_preset_divisible_heads(self):
        from sparse_attention_fineweb import MODEL_CONFIGS
        cfg = MODEL_CONFIGS["1b"]
        assert cfg["d_model"] % cfg["n_heads"] == 0


class TestNativeHardCondition:
    def test_native_hard_in_conditions(self):
        from sparse_attention_fineweb import CONDITIONS
        assert "native_hard" in CONDITIONS

    def test_hard_ste_forward_shape(self):
        from sparse_attention_fineweb import SparseTransformer
        torch.manual_seed(0)
        model = SparseTransformer(
            vocab_size=97, d_model=32, n_heads=2, n_layers=2,
            d_ff=64, d_gate=8, max_seq_len=16, dropout=0.0,
        )
        x = torch.randint(0, 97, (2, 16))
        logits, masks = model(x, hard_ste_mode=True, hard_k=4)
        assert logits.shape == (2, 16, 97)

    def test_hard_ste_incompatible_with_dense_mode(self):
        from sparse_attention_fineweb import SparseTransformer
        model = SparseTransformer(
            vocab_size=97, d_model=32, n_heads=2, n_layers=2,
            d_ff=64, d_gate=8, max_seq_len=16, dropout=0.0,
        )
        x = torch.randint(0, 97, (2, 16))
        with pytest.raises(ValueError):
            model(x, hard_ste_mode=True, dense_mode=True)

    def test_hard_ste_forward_differs_from_dense(self):
        """A real top-k restriction must change the output vs full-dense
        attention -- if this fails, the STE branch is silently degenerating
        to dense (the same "no actual sparsity" bug class caught earlier
        tonight in the MoBA integration)."""
        from sparse_attention_fineweb import SparseTransformer
        torch.manual_seed(0)
        model = SparseTransformer(
            vocab_size=97, d_model=32, n_heads=2, n_layers=2,
            d_ff=64, d_gate=8, max_seq_len=16, dropout=0.0,
        )
        model.eval()
        x = torch.randint(0, 97, (2, 16))
        with torch.no_grad():
            logits_hard, _ = model(x, hard_ste_mode=True, hard_k=4)
            logits_dense, _ = model(x, dense_mode=True)
        assert not torch.allclose(logits_hard, logits_dense, atol=1e-4)

    def test_hard_ste_gradient_reaches_gate_weights(self):
        """Straight-through: forward is hard, but W_gq/W_gk must still get
        gradient via the soft surrogate, or the gate is dead weight."""
        from sparse_attention_fineweb import SparseTransformer
        torch.manual_seed(0)
        model = SparseTransformer(
            vocab_size=97, d_model=32, n_heads=2, n_layers=2,
            d_ff=64, d_gate=8, max_seq_len=16, dropout=0.0,
        )
        x = torch.randint(0, 97, (2, 16))
        logits, _ = model(x, hard_ste_mode=True, hard_k=4)
        loss = logits.sum()
        loss.backward()
        gate_param = model.blocks[0].attn.W_gq.weight
        assert gate_param.grad is not None
        assert torch.isfinite(gate_param.grad).all()
        assert gate_param.grad.abs().sum() > 0

    def test_hard_ste_forward_bit_identical_to_closed_loop_gate_hard(self):
        """The whole point of native_hard: native (STE) forward and
        closed-loop hard deployment must be the SAME function, so G_CL~0
        by construction -- not just numerically close."""
        from sparse_attention_fineweb import SparseTransformer, forward_closed_loop_gate_hard
        torch.manual_seed(0)
        model = SparseTransformer(
            vocab_size=97, d_model=32, n_heads=2, n_layers=2,
            d_ff=64, d_gate=8, max_seq_len=16, dropout=0.0,
        )
        model.eval()
        x = torch.randint(0, 97, (2, 16))
        with torch.no_grad():
            logits_native, _ = model(x, hard_ste_mode=True, hard_k=4)
            logits_closed_loop = forward_closed_loop_gate_hard(model, x, k=4)
        assert torch.allclose(logits_native, logits_closed_loop, atol=1e-4)
