"""Tests for SeerAttention-style post-hoc block-sparse gate distillation.

New baseline requested by external review: block-level gate learning via
self-distillation from a frozen model's dense attention pattern (SeerAttention
mechanism), with a `clhr` mode that mixes CLHR's closed-loop hard-deployment
signal into the gate-distillation objective itself.

Genuinely new relative to existing repo code (see src/sparse_attention_seer_post.py
module docstring for the full justification):
  - block-level (not per-token) gate + ground truth
  - categorical softmax-KL distillation target (not elementwise Bernoulli KL/BCE
    as in sparse_attention_closed_loop_eval.py's posthoc_kl_distillation)
  - a closed-loop training-time term (existing posthoc_* functions are open-loop
    only; closed-loop is only ever used for post-training eval there)

Verifies:
1. BlockImportanceGate produces correctly-shaped block-level logits (incl.
   non-block-divisible sequence lengths).
2. Dense-attention block pooling produces a valid per-query-block distribution.
3. Block-causal mask is correct (no future-block leakage).
4. Hard block-mask selection respects causality and the requested top-k budget.
5. Self-distillation KL loss is lower for matching vs. mismatched distributions.
6. The STE hard-selection indicator has a forward value identical to the true
   hard mask, but a nonzero gradient (so gradient reaches the gate through a
   frozen backbone).
7. SeerPostHocGateModel freezes the backbone and only exposes gate params as
   trainable.
8. `standard` mode backprop touches only gate params.
9. `clhr` mode backprop touches only gate params but produces a real
   closed_loop_lm_loss term.
10. target-sparsity -> top-k-blocks conversion is correct at the boundaries.
"""
import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sparse_attention_seer_post import (  # noqa: E402
    BlockImportanceGate,
    SelfDistillationLoss,
    SeerPostHocGateModel,
    Qwen3SeerPostHocGateModel,
    block_causal_mask,
    block_gate_ste_indicator,
    harden_block_mask,
    n_blocks_for,
    pool_dense_attention_to_blocks,
    topk_blocks_from_sparsity,
    build_frozen_dense_backbone,
)


class TestBlockImportanceGate:
    def test_output_shape_divisible(self):
        gate = BlockImportanceGate(d_head=8, block_size=4)
        q = torch.randn(2, 3, 16, 8)  # B=2,H=3,T=16,d_head=8
        k = torch.randn(2, 3, 16, 8)
        logits = gate(q, k)
        assert logits.shape == (2, 3, 4, 4)  # 16/4 = 4 blocks

    def test_output_shape_non_divisible(self):
        gate = BlockImportanceGate(d_head=8, block_size=5)
        q = torch.randn(1, 2, 12, 8)  # 12/5 -> 3 blocks (padded)
        k = torch.randn(1, 2, 12, 8)
        logits = gate(q, k)
        assert logits.shape == (1, 2, 3, 3)

    def test_gate_has_learnable_params(self):
        gate = BlockImportanceGate(d_head=8, block_size=4)
        params = list(gate.parameters())
        assert len(params) > 0
        assert all(p.requires_grad for p in params)


class TestBlockPooling:
    def test_n_blocks_for(self):
        assert n_blocks_for(16, 4) == 4
        assert n_blocks_for(15, 4) == 4
        assert n_blocks_for(1, 4) == 1

    def test_pool_dense_attention_to_blocks_is_distribution(self):
        B, H, T, block_size = 2, 2, 8, 4
        # random causal dense attention (rows sum to 1 over valid positions)
        raw = torch.rand(B, H, T, T)
        causal = torch.tril(torch.ones(T, T))
        raw = raw.masked_fill(causal.unsqueeze(0).unsqueeze(0) == 0, float("-inf"))
        attn_probs = torch.softmax(raw, dim=-1)
        pooled = pool_dense_attention_to_blocks(attn_probs, block_size)
        assert pooled.shape == (B, H, 2, 2)
        row_sums = pooled.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4)
        assert (pooled >= 0).all()


class TestBlockCausalMask:
    def test_lower_triangular(self):
        mask = block_causal_mask(seq_len=16, block_size=4, device=torch.device("cpu"))
        assert mask.shape == (4, 4)
        expected = torch.tril(torch.ones(4, 4))
        assert torch.equal(mask, expected)


class TestHardenBlockMask:
    def test_respects_causality(self):
        n_qb, n_kb = 4, 4
        logits = torch.randn(1, 1, n_qb, n_kb)
        causal = block_causal_mask(16, 4, torch.device("cpu"))
        hard = harden_block_mask(logits, topk_blocks=4, causal_block_mask=causal)
        # No selection above the block diagonal (future blocks) anywhere.
        illegal = hard * (1 - causal)
        assert illegal.sum().item() == 0.0

    def test_topk_count_matches_available_causal_budget(self):
        n_qb, n_kb = 4, 4
        logits = torch.randn(1, 1, n_qb, n_kb)
        causal = block_causal_mask(16, 4, torch.device("cpu"))
        hard = harden_block_mask(logits, topk_blocks=2, causal_block_mask=causal)
        counts = hard.sum(dim=-1).squeeze()
        # row 0 has only 1 causal block available, rows 1-3 have >= 2
        assert counts[0].item() == 1.0
        assert counts[1].item() == 2.0
        assert counts[3].item() == 2.0


class TestSelfDistillationLoss:
    def test_lower_for_matching_distribution(self):
        causal = block_causal_mask(16, 4, torch.device("cpu"))
        target = torch.softmax(
            torch.randn(1, 1, 4, 4).masked_fill(causal == 0, float("-inf")), dim=-1
        )
        loss_fn = SelfDistillationLoss()

        # logits that reproduce the target distribution exactly: log_softmax of
        # log(target) (masked to the same causal support) recovers log(target)
        # itself, since target already sums to 1 over that support.
        matching_logits = torch.log(target.clamp_min(1e-8))
        matched_loss = loss_fn(matching_logits, target, causal)

        # logits uncorrelated with the target
        random_logits = torch.randn_like(matching_logits)
        random_loss = loss_fn(random_logits, target, causal)

        assert matched_loss.item() < random_loss.item()
        # True KL divergence is >= 0; allow float32 rounding noise near 0.
        assert matched_loss.item() >= -1e-6


class TestSTEIndicator:
    def test_forward_equals_hard_mask(self):
        n_qb, n_kb = 4, 4
        causal = block_causal_mask(16, 4, torch.device("cpu"))
        logits = torch.randn(1, 1, n_qb, n_kb, requires_grad=True)
        ste = block_gate_ste_indicator(logits, causal, topk_blocks=2)
        with torch.no_grad():
            hard = harden_block_mask(logits, topk_blocks=2, causal_block_mask=causal)
        # `hard + soft - soft.detach()` is only exactly hard in exact arithmetic;
        # (a+b)-b can differ from a by ~1 float32 ULP due to rounding in the
        # intermediate sum. allclose at 1e-5 still tightly checks the STE
        # property (forward value == hard mask) without over-specifying bits.
        assert torch.allclose(ste.detach(), hard, atol=1e-5)

    def test_gradient_flows_to_logits(self):
        n_qb, n_kb = 4, 4
        causal = block_causal_mask(16, 4, torch.device("cpu"))
        logits = torch.randn(1, 1, n_qb, n_kb, requires_grad=True)
        ste = block_gate_ste_indicator(logits, causal, topk_blocks=2)
        ste.sum().backward()
        assert logits.grad is not None
        assert logits.grad.abs().sum().item() > 0.0


class TestTopkBlocksFromSparsity:
    def test_boundaries(self):
        assert topk_blocks_from_sparsity(n_kb_total=8, target_sparsity=0.5) == 4
        assert topk_blocks_from_sparsity(n_kb_total=8, target_sparsity=0.0) == 8
        assert topk_blocks_from_sparsity(n_kb_total=8, target_sparsity=1.0) == 1  # clipped to >=1
        assert topk_blocks_from_sparsity(n_kb_total=8, target_sparsity=0.99) == 1


class TestSeerPostHocGateModel:
    def _tiny_model(self, seq_len=16, block_size=4):
        backbone = build_frozen_dense_backbone(
            vocab_size=32, d_model=16, n_heads=2, n_layers=2, d_ff=32, max_seq_len=seq_len
        )
        model = SeerPostHocGateModel(backbone, block_size=block_size, gate_dim=8)
        return model

    def test_backbone_frozen_gates_trainable(self):
        model = self._tiny_model()
        assert all(not p.requires_grad for p in model.backbone.parameters())
        assert all(p.requires_grad for p in model.gates.parameters())

    def test_standard_mode_backprop_only_touches_gates(self):
        torch.manual_seed(0)
        model = self._tiny_model()
        input_ids = torch.randint(0, 32, (2, 16))
        targets = torch.randint(0, 32, (2, 16))
        loss, info = model.compute_loss(
            input_ids, targets, mode="standard", topk_blocks=2, lambda_rca=1.0
        )
        loss.backward()
        assert info["closed_loop_lm_loss"] is None
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.gates.parameters())
        assert all(p.grad is None for p in model.backbone.parameters())

    def test_clhr_mode_backprop_and_closed_loop_loss_present(self):
        torch.manual_seed(0)
        model = self._tiny_model()
        input_ids = torch.randint(0, 32, (2, 16))
        targets = torch.randint(0, 32, (2, 16))
        loss, info = model.compute_loss(
            input_ids, targets, mode="clhr", topk_blocks=2, lambda_rca=1.0
        )
        loss.backward()
        assert isinstance(info["closed_loop_lm_loss"], float)
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.gates.parameters())
        assert all(p.grad is None for p in model.backbone.parameters())

    def test_invalid_mode_raises(self):
        model = self._tiny_model()
        input_ids = torch.randint(0, 32, (2, 16))
        targets = torch.randint(0, 32, (2, 16))
        with pytest.raises(ValueError):
            model.compute_loss(input_ids, targets, mode="bogus", topk_blocks=2)


class TestQwen3SeerPostHocGateModel:
    """Real HF Qwen3 architecture adapter, exercised against a TINY,
    randomly-initialized Qwen3Config (no weight download -- this only
    instantiates the real transformers Qwen3ForCausalLM class with random
    weights, verifying the adapter's RoPE/QK-norm/GQA-repeat plumbing is
    correct, independent of any real pretrained checkpoint). GQA is
    deliberately exercised here (num_attention_heads=4, num_key_value_heads=2)
    since that is the part most likely to silently break.
    """

    def _tiny_qwen(self, seq_len=16, block_size=4):
        from transformers import Qwen3Config, Qwen3ForCausalLM

        config = Qwen3Config(
            vocab_size=32,
            hidden_size=16,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=32,
            max_position_embeddings=seq_len,
            head_dim=4,
            attn_implementation="eager",
        )
        qwen = Qwen3ForCausalLM(config)
        model = Qwen3SeerPostHocGateModel(qwen, block_size=block_size, gate_dim=8)
        return model

    def test_backbone_frozen_gates_trainable(self):
        model = self._tiny_qwen()
        assert all(not p.requires_grad for p in model.backbone.parameters())
        assert all(p.requires_grad for p in model.gates.parameters())

    def test_teacher_forward_shapes_respect_gqa(self):
        model = self._tiny_qwen(seq_len=16)
        input_ids = torch.randint(0, 32, (2, 16))
        records, logits = model.teacher_forward(input_ids)
        assert logits.shape == (2, 16, 32)
        n_heads = model.backbone.config.num_attention_heads
        head_dim = model.backbone.config.head_dim
        for rec in records:
            # q/k are expanded to num_attention_heads (post GQA-repeat), not
            # num_key_value_heads, matching what the gate expects.
            assert rec["q"].shape == (2, n_heads, 16, head_dim)
            assert rec["k"].shape == (2, n_heads, 16, head_dim)
            assert rec["attn_probs"].shape == (2, n_heads, 16, 16)
            row_sums = rec["attn_probs"].sum(dim=-1)
            assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3)

    def test_standard_mode_backprop_only_touches_gates(self):
        torch.manual_seed(0)
        model = self._tiny_qwen()
        input_ids = torch.randint(0, 32, (2, 16))
        targets = torch.randint(0, 32, (2, 16))
        loss, info = model.compute_loss(
            input_ids, targets, mode="standard", topk_blocks=2, lambda_rca=1.0
        )
        loss.backward()
        assert info["closed_loop_lm_loss"] is None
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.gates.parameters())
        assert all(p.grad is None for p in model.backbone.parameters())

    def test_clhr_mode_backprop_and_closed_loop_loss_present(self):
        torch.manual_seed(0)
        model = self._tiny_qwen()
        input_ids = torch.randint(0, 32, (2, 16))
        targets = torch.randint(0, 32, (2, 16))
        loss, info = model.compute_loss(
            input_ids, targets, mode="clhr", topk_blocks=2, lambda_rca=1.0
        )
        loss.backward()
        assert isinstance(info["closed_loop_lm_loss"], float)
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.gates.parameters())
        assert all(p.grad is None for p in model.backbone.parameters())
