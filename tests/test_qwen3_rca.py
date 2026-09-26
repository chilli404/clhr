"""Tests for the Qwen3-1.7B single-layer sparse-attention RCA experiment.

Tests module structure, conditions, snapshot logic, and mask manipulation.
No GPU required — uses pure logic tests.
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestQwen3RCAModule:
    """Test module structure and imports."""

    def test_module_imports(self):
        from qwen3_sparse_rca import CONDITIONS, select_oldest_snapshot  # noqa

    def test_conditions_defined(self):
        from qwen3_sparse_rca import CONDITIONS
        expected = [
            "learned_gate",
            "contemporary_replay",
            "shuffled_historical",
            "coherent_oldest_first",
        ]
        for c in expected:
            assert c in CONDITIONS, f"Missing condition: {c}"

    def test_exactly_four_conditions(self):
        from qwen3_sparse_rca import CONDITIONS
        assert len(CONDITIONS) == 4


class TestSnapshotLogic:
    """Test snapshot selection and management."""

    def test_select_oldest_snapshot(self):
        from qwen3_sparse_rca import select_oldest_snapshot
        snapshots = [
            {"step": 1000, "gate_state": "a", "attn_state": "x"},
            {"step": 500, "gate_state": "b", "attn_state": "y"},
            {"step": 1500, "gate_state": "c", "attn_state": "z"},
        ]
        oldest = select_oldest_snapshot(snapshots)
        assert oldest["step"] == 500

    def test_select_oldest_single(self):
        from qwen3_sparse_rca import select_oldest_snapshot
        snapshots = [{"step": 1000, "gate_state": "a", "attn_state": "x"}]
        oldest = select_oldest_snapshot(snapshots)
        assert oldest["step"] == 1000

    def test_select_oldest_empty_raises(self):
        from qwen3_sparse_rca import select_oldest_snapshot
        with pytest.raises(ValueError):
            select_oldest_snapshot([])


class TestMaskShuffling:
    """Test that shuffled masks permute correctly."""

    def test_shuffle_changes_mask(self):
        from qwen3_sparse_rca import shuffle_soft_mask
        mask = torch.zeros(1, 2, 4, 4)
        mask[0, 0, 0, 0] = 5.0
        mask[0, 0, 1, 1] = 3.0
        mask[0, 1, 2, 3] = 7.0
        shuffled = shuffle_soft_mask(mask, seed=42)
        assert not torch.equal(mask, shuffled)

    def test_shuffle_preserves_values(self):
        from qwen3_sparse_rca import shuffle_soft_mask
        mask = torch.randn(2, 4, 8, 8)
        shuffled = shuffle_soft_mask(mask, seed=123)
        for b in range(2):
            for h in range(4):
                orig_sorted = mask[b, h].flatten().sort().values
                shuf_sorted = shuffled[b, h].flatten().sort().values
                assert torch.allclose(orig_sorted, shuf_sorted, atol=1e-6)

    def test_shuffle_is_deterministic(self):
        from qwen3_sparse_rca import shuffle_soft_mask
        mask = torch.randn(1, 2, 4, 4)
        s1 = shuffle_soft_mask(mask, seed=99)
        s2 = shuffle_soft_mask(mask, seed=99)
        assert torch.equal(s1, s2)


class TestDataLoading:
    """Test WikiText loading with Qwen3 tokenizer."""

    def test_load_function_exists(self):
        from qwen3_sparse_rca import load_wikitext_qwen3  # noqa
