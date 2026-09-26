"""Tests for the 300M sparse-attention scale confirmation experiment."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestSparseTransformer300M:
    def test_module_imports(self):
        from sparse_attention_300m import SparseTransformer300M, CONDITIONS

    def test_conditions_defined(self):
        from sparse_attention_300m import CONDITIONS
        assert "standard" in CONDITIONS
        assert "contemporary_replay" in CONDITIONS
        assert "shuffled_historical" in CONDITIONS
        assert "coherent_hard_oldest" in CONDITIONS

    def test_model_construction(self):
        from sparse_attention_300m import SparseTransformer300M
        model = SparseTransformer300M(
            vocab_size=1000, d_model=64, n_heads=4, n_layers=2,
            d_ff=256, d_gate=16, max_seq_len=32,
        )
        assert model is not None

    def test_param_count_small(self):
        from sparse_attention_300m import SparseTransformer300M
        model = SparseTransformer300M(
            vocab_size=1000, d_model=64, n_heads=4, n_layers=2,
            d_ff=256, d_gate=16, max_seq_len=32,
        )
        n = sum(p.numel() for p in model.parameters())
        assert n > 0

    def test_param_count_300m_config(self):
        from sparse_attention_300m import SparseTransformer300M
        model = SparseTransformer300M(
            vocab_size=50257, d_model=1024, n_heads=16, n_layers=20,
            d_ff=4096, d_gate=32, max_seq_len=512,
        )
        n = sum(p.numel() for p in model.parameters())
        assert 250_000_000 < n < 400_000_000, f"Expected ~300M params, got {n/1e6:.0f}M"

    def test_forward_pass_small(self):
        from sparse_attention_300m import SparseTransformer300M
        model = SparseTransformer300M(
            vocab_size=1000, d_model=64, n_heads=4, n_layers=2,
            d_ff=256, d_gate=16, max_seq_len=32,
        )
        x = torch.randint(0, 1000, (2, 16))
        logits, masks = model(x)
        assert logits.shape == (2, 16, 1000)
        assert len(masks) == 2

    def test_forward_with_forced_masks(self):
        from sparse_attention_300m import SparseTransformer300M
        model = SparseTransformer300M(
            vocab_size=1000, d_model=64, n_heads=4, n_layers=2,
            d_ff=256, d_gate=16, max_seq_len=32,
        )
        x = torch.randint(0, 1000, (2, 16))
        forced = [torch.ones(2, 4, 16, 16) for _ in range(2)]
        logits, masks = model(x, forced_masks=forced)
        assert logits.shape == (2, 16, 1000)

    def test_oracle_masks(self):
        from sparse_attention_300m import SparseTransformer300M, compute_oracle_masks
        model = SparseTransformer300M(
            vocab_size=1000, d_model=64, n_heads=4, n_layers=2,
            d_ff=256, d_gate=16, max_seq_len=32,
        )
        model.eval()
        x = torch.randint(0, 1000, (2, 16))
        masks = compute_oracle_masks(model, x, k=4)
        assert len(masks) == 2
        for m in masks:
            assert m.shape == (2, 4, 16, 16)

    def test_hardened_masks(self):
        from sparse_attention_300m import SparseTransformer300M, compute_hardened_gate_masks
        model = SparseTransformer300M(
            vocab_size=1000, d_model=64, n_heads=4, n_layers=2,
            d_ff=256, d_gate=16, max_seq_len=32,
        )
        model.eval()
        x = torch.randint(0, 1000, (2, 16))
        masks = compute_hardened_gate_masks(model, x, k=4)
        assert len(masks) == 2
        for m in masks:
            assert m.shape == (2, 4, 16, 16)
            for b in range(2):
                for h in range(4):
                    for q in range(16):
                        assert m[b, h, q].sum().item() <= 4

    def test_weight_tying(self):
        from sparse_attention_300m import SparseTransformer300M
        model = SparseTransformer300M(
            vocab_size=1000, d_model=64, n_heads=4, n_layers=2,
            d_ff=256, d_gate=16, max_seq_len=32,
        )
        assert model.lm_head.weight is model.tok_emb.weight


class TestBenchmarkMode:
    def test_benchmark_flag_exists(self):
        from sparse_attention_300m import CONDITIONS
        # Just verify the module loads; actual benchmark needs GPU


class TestDataLoading:
    def test_data_loader_function_exists(self):
        from sparse_attention_300m import load_training_data
