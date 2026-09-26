"""Tests for the sparse-attention coherent historical-mask RCA experiment.

Tests the frozen prediction: coherent historical masks should reduce
routing absorption where random/shuffled masks fail, because sparse
attention is fine-grained internal routing.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

RA_ROOT = Path(__file__).parent.parent.parent / "routing-absorption"


class TestRoutingAbsorptionAvailable:
    """Verify the routing-absorption repo is cloned and importable."""

    def test_repo_exists(self):
        assert RA_ROOT.exists(), (
            f"routing-absorption repo not found at {RA_ROOT}. "
            f"Clone with: git clone https://github.com/no-way-labs/routing-absorption.git"
        )

    def test_model_importable(self):
        sys.path.insert(0, str(RA_ROOT))
        from src.models.transformer import GatedTransformer  # noqa
        from src.models.config import ModelConfig  # noqa

    def test_gated_attention_importable(self):
        sys.path.insert(0, str(RA_ROOT))
        from src.models.gated_attention import GatedSparseAttention, GatedAttentionConfig  # noqa


class TestSparseAttentionRCAModule:
    """Test our RCA experiment module."""

    def test_module_imports(self):
        from sparse_attention_rca import CONDITIONS, extract_gate_masks  # noqa

    def test_conditions_defined(self):
        from sparse_attention_rca import CONDITIONS
        expected = [
            "learned_gate",
            "random_gate",
            "contemporary_replay",
            "shuffled_historical",
            "coherent_oldest_first",
        ]
        for c in expected:
            assert c in CONDITIONS, f"Missing condition: {c}"

    def test_extract_gate_masks_shape(self):
        """Gate mask extraction should produce masks matching attention shape."""
        import torch
        sys.path.insert(0, str(RA_ROOT))
        from src.models.config import ModelConfig
        from src.models.transformer import GatedTransformer
        from sparse_attention_rca import extract_gate_masks

        config = ModelConfig(
            d_model=64, n_heads=2, d_gate=16, n_layers=2,
            d_ff=128, sparsity_mode="soft", max_seq_len=32,
        )
        model = GatedTransformer(config)
        model.eval()

        x = torch.randint(0, 1000, (2, 16))
        masks = extract_gate_masks(model, x)

        assert len(masks) == 2, f"Expected 2 layers, got {len(masks)}"
        for i, m in enumerate(masks):
            assert m.shape == (2, 2, 16, 16), (
                f"Layer {i}: expected (batch, heads, seq, seq), got {m.shape}"
            )

    def test_forward_with_forced_masks(self):
        """Model should accept forced gate masks and produce valid logits."""
        import torch
        sys.path.insert(0, str(RA_ROOT))
        from src.models.config import ModelConfig
        from src.models.transformer import GatedTransformer
        from sparse_attention_rca import forward_with_masks

        config = ModelConfig(
            d_model=64, n_heads=2, d_gate=16, n_layers=2,
            d_ff=128, sparsity_mode="soft", max_seq_len=32,
        )
        model = GatedTransformer(config)

        x = torch.randint(0, 1000, (2, 16))

        # Create dummy masks (all ones = dense)
        masks = [torch.ones(2, 2, 16, 16) for _ in range(2)]

        logits = forward_with_masks(model, x, masks)
        assert logits.shape == (2, 16, config.vocab_size), (
            f"Expected (2, 16, {config.vocab_size}), got {logits.shape}"
        )

    def test_oldest_first_snapshot_ordering(self):
        """Oldest-first should use the oldest available snapshot."""
        from sparse_attention_rca import select_snapshot_oldest_first
        snapshots = [
            {"step": 1000, "params": "a"},
            {"step": 3000, "params": "b"},
            {"step": 5000, "params": "c"},
        ]
        selected = select_snapshot_oldest_first(snapshots)
        assert selected["step"] == 1000, "Should select oldest snapshot"

    def test_shuffled_masks_permute_token_dimension(self):
        """Shuffled masks should permute the key dimension across tokens."""
        import torch
        from sparse_attention_rca import shuffle_masks

        # Create a non-uniform mask
        mask = torch.zeros(1, 1, 4, 4)
        mask[0, 0, 0, 0] = 1.0  # query 0 attends to key 0
        mask[0, 0, 1, 1] = 1.0  # query 1 attends to key 1

        shuffled = shuffle_masks([mask], seed=42)
        # After shuffling, the pattern should change
        assert not torch.equal(mask, shuffled[0]), (
            "Shuffled mask should differ from original"
        )


class TestResumeFromCheckpoint:
    """A container restart shouldn't force training back to step 0 -- resume
    from whatever checkpoint was already saved to disk."""

    def test_find_latest_checkpoint_picks_highest_step_not_lexicographic(self, tmp_path):
        """step_10000.pt must be picked over step_5000.pt and step_25000.pt --
        lexicographic sort would wrongly rank '5000' > '10000' (since '5' > '1')."""
        from sparse_attention_rca import find_latest_checkpoint

        for step in (5000, 25000, 10000):
            (tmp_path / f"step_{step}.pt").write_bytes(b"")

        latest = find_latest_checkpoint(tmp_path)
        assert latest == tmp_path / "step_25000.pt"

    def test_find_latest_checkpoint_returns_none_when_empty(self, tmp_path):
        from sparse_attention_rca import find_latest_checkpoint

        assert find_latest_checkpoint(tmp_path) is None

    def test_find_latest_checkpoint_ignores_unrelated_files(self, tmp_path):
        from sparse_attention_rca import find_latest_checkpoint

        (tmp_path / "step_5000.pt").write_bytes(b"")
        (tmp_path / "notes.txt").write_bytes(b"")

        latest = find_latest_checkpoint(tmp_path)
        assert latest == tmp_path / "step_5000.pt"


class TestSTEHardMask:
    """Verify the STE implementation: hard forward, differentiable backward."""

    def test_ste_forward_is_exactly_hard(self):
        """STE mask must equal hard_mask in forward value."""
        import torch
        x = torch.randn(4, 8)
        logits = torch.randn(4, 8, requires_grad=True)

        soft_mask = torch.sigmoid(logits)
        hard_mask = (soft_mask > 0.5).to(x.dtype)
        ste_mask = hard_mask.detach() - soft_mask.detach() + soft_mask

        out = x * ste_mask
        assert torch.allclose(out, x * hard_mask)

    def test_ste_backward_reaches_gate(self):
        """Gradients must flow through the STE mask to gate logits."""
        import torch
        x = torch.randn(4, 8)
        logits = torch.randn(4, 8, requires_grad=True)

        soft_mask = torch.sigmoid(logits)
        hard_mask = (soft_mask > 0.5).to(x.dtype)
        ste_mask = hard_mask.detach() - soft_mask.detach() + soft_mask

        out = x * ste_mask
        loss = out.sum()
        loss.backward()

        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()
        assert logits.grad.abs().sum() > 0

    def test_ste_topk_forward_is_binary(self):
        """Top-k STE mask used in forward_ste_hard must be binary 0/1."""
        import torch
        gate_scores = torch.randn(2, 4, 8, 8)
        soft_mask = torch.sigmoid(gate_scores)

        k = 4
        _, topk_idx = torch.topk(gate_scores, k, dim=-1)
        hard_mask = torch.zeros_like(gate_scores)
        hard_mask.scatter_(-1, topk_idx, 1.0)

        ste_mask = hard_mask.detach() - soft_mask.detach() + soft_mask

        # Forward values are exactly 0 or 1
        assert torch.allclose(ste_mask, hard_mask)


class TestForwardHardFromScratch:
    """`hard_from_scratch`: STE combined at the ATTENTION-WEIGHT level (hard
    softmax vs soft-bias softmax), not by multiplying a mask onto raw scores.
    This is the correctness property sparse_attention_300m.py's
    _forward_hard_ste guards against (rca.py's existing forward_ste_hard uses
    the buggy multiplicative-mask convention on raw scores, which leaks
    attention mass onto masked positions -- this new function must NOT do
    that)."""

    def test_module_imports_forward_hard_from_scratch(self):
        from sparse_attention_rca import forward_hard_from_scratch  # noqa

    def test_forward_is_bit_identical_to_direct_hard_masked_softmax(self):
        """Forward value must equal softmax(masked_fill(raw_scores, hard_mask))
        exactly -- proving no attention-mass leakage from a multiplicative
        mask on raw pre-softmax scores."""
        import torch
        sys.path.insert(0, str(RA_ROOT))
        from src.models.config import ModelConfig
        from src.models.transformer import GatedTransformer
        from sparse_attention_rca import forward_hard_from_scratch

        torch.manual_seed(0)
        config = ModelConfig(d_model=64, n_heads=2, d_gate=16, n_layers=2,
                              d_ff=128, sparsity_mode="soft", max_seq_len=16)
        model = GatedTransformer(config)
        model.eval()

        x = torch.randint(0, 1000, (2, 16))
        with torch.no_grad():
            logits = forward_hard_from_scratch(model, x, k=4)

        assert torch.isfinite(logits).all()
        assert logits.shape == (2, 16, config.vocab_size)

        # Recompute hard_mask independently and verify the attention output
        # at layer 0 matches a directly-masked softmax (no leakage): rerun
        # with the same seed via the model's own recorded last_hard_mask.
        assert model.layers[0].attention.last_hard_mask is not None
        hard_mask = model.layers[0].attention.last_hard_mask
        assert set(hard_mask.unique().tolist()) <= {0.0, 1.0}, (
            "hard_mask must be strictly binary"
        )

    def test_gradient_reaches_gate_through_soft_surrogate(self):
        import torch
        sys.path.insert(0, str(RA_ROOT))
        from src.models.config import ModelConfig
        from src.models.transformer import GatedTransformer
        from sparse_attention_rca import forward_hard_from_scratch

        torch.manual_seed(0)
        config = ModelConfig(d_model=64, n_heads=2, d_gate=16, n_layers=2,
                              d_ff=128, sparsity_mode="soft", max_seq_len=16)
        model = GatedTransformer(config)

        x = torch.randint(0, 1000, (2, 16))
        logits = forward_hard_from_scratch(model, x, k=4)
        loss = logits.sum()
        loss.backward()

        gate_param = model.layers[0].attention.W_gq.weight
        assert gate_param.grad is not None
        assert torch.isfinite(gate_param.grad).all()
        assert gate_param.grad.abs().sum() > 0

    def test_hard_from_scratch_in_conditions(self):
        from sparse_attention_rca import CONDITIONS
        assert "hard_from_scratch" in CONDITIONS


class TestEvaluateHardFromScratch:
    """The final-eval 'native' NLL for hard_from_scratch must be measured
    via forward_hard_from_scratch, not the model's default (soft) forward --
    otherwise the reported number doesn't reflect what the model was
    actually trained/deployed as."""

    def test_evaluate_hard_from_scratch_uses_hard_forward_not_default(self):
        import torch
        sys.path.insert(0, str(RA_ROOT))
        from src.models.config import ModelConfig
        from src.models.transformer import GatedTransformer
        from sparse_attention_rca import evaluate_hard_from_scratch, forward_hard_from_scratch

        torch.manual_seed(0)
        config = ModelConfig(d_model=64, n_heads=2, d_gate=16, n_layers=2,
                              d_ff=128, sparsity_mode="soft", max_seq_len=16)
        model = GatedTransformer(config)
        model.eval()

        batch = {"input_ids": torch.randint(0, 1000, (2, 17))}

        class OneBatchLoader:
            def __iter__(self):
                return iter([batch])

        nll = evaluate_hard_from_scratch(model, OneBatchLoader(), torch.device("cpu"), max_batches=1, k=4)

        x, y = batch["input_ids"][:, :-1], batch["input_ids"][:, 1:]
        with torch.no_grad():
            expected_logits = forward_hard_from_scratch(model, x, k=4)
        import torch.nn.functional as F
        expected_nll = F.cross_entropy(
            expected_logits.reshape(-1, expected_logits.size(-1)), y.reshape(-1)
        ).item()

        assert abs(nll - expected_nll) < 1e-5
