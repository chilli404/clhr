"""Tests for the multi-layer Qwen3-1.7B sparse fine-tuning with RCA.

All 28 layers get gates and trainable Q/K/V/O, unlike the single-layer
version which only modifies layer 14.
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestMultiLayerGates:
    def test_module_imports(self):
        from qwen3_multilayer_sparse_rca import MultiLayerGates, CONDITIONS

    def test_conditions_defined(self):
        from qwen3_multilayer_sparse_rca import CONDITIONS
        expected = ["learned_gate", "contemporary_replay",
                    "shuffled_historical", "coherent_oldest_first"]
        for c in expected:
            assert c in CONDITIONS

    def test_multi_gate_construction(self):
        from qwen3_multilayer_sparse_rca import MultiLayerGates
        gates = MultiLayerGates(n_layers=4, n_heads=2, d_model=64, d_gate=16)
        assert len(gates.gates) == 4

    def test_multi_gate_scores_shape(self):
        from qwen3_multilayer_sparse_rca import MultiLayerGates
        gates = MultiLayerGates(n_layers=4, n_heads=2, d_model=64, d_gate=16)
        hidden = torch.randn(2, 8, 64)
        scores = gates.gate_scores(0, hidden)
        assert scores.shape == (2, 2, 8, 8)

    def test_multi_gate_state_dict_roundtrip(self):
        from qwen3_multilayer_sparse_rca import MultiLayerGates
        gates = MultiLayerGates(n_layers=4, n_heads=2, d_model=64, d_gate=16)
        sd = gates.state_dict()
        gates2 = MultiLayerGates(n_layers=4, n_heads=2, d_model=64, d_gate=16)
        gates2.load_state_dict(sd)
        for p1, p2 in zip(gates.parameters(), gates2.parameters()):
            assert torch.equal(p1, p2)

    def test_multi_gate_param_count(self):
        from qwen3_multilayer_sparse_rca import MultiLayerGates
        gates = MultiLayerGates(n_layers=28, n_heads=16, d_model=2048, d_gate=32)
        total = sum(p.numel() for p in gates.parameters())
        expected_per_layer = 2 * 2048 * (16 * 32)
        assert total == 28 * expected_per_layer


class TestSnapshotLogic:
    def test_select_oldest(self):
        from qwen3_multilayer_sparse_rca import select_oldest_snapshot
        snaps = [{"step": 1000, "x": "a"}, {"step": 500, "x": "b"}, {"step": 1500, "x": "c"}]
        assert select_oldest_snapshot(snaps)["step"] == 500


class TestShuffleAllLayers:
    def test_shuffle_list_of_masks(self):
        from qwen3_multilayer_sparse_rca import shuffle_all_layer_masks
        masks = [torch.randn(1, 2, 4, 4) for _ in range(3)]
        shuffled = shuffle_all_layer_masks(masks)
        assert len(shuffled) == 3
        for orig, shuf in zip(masks, shuffled):
            assert orig.shape == shuf.shape
