"""Tests for scripts/lm_eval_adapter.py -- verifies SparseTransformerLM
produces sensible loglikelihoods against a real (tiny, untrained) checkpoint,
before trusting it against real 300M FineWeb checkpoints.
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from eval_downstream import SparseTransformer  # noqa: E402


@pytest.fixture
def tiny_checkpoint(tmp_path):
    torch.manual_seed(0)
    model = SparseTransformer(
        vocab_size=50257, d_model=64, n_heads=1, n_layers=2, d_ff=128,
        d_gate=16, max_seq_len=32, dropout=0.0,
    )
    path = tmp_path / "tiny.pt"
    torch.save(model.state_dict(), path)
    return str(path)


class TestSparseTransformerLM:
    def test_loads_and_infers_config(self, tiny_checkpoint):
        from lm_eval_adapter import SparseTransformerLM
        lm = SparseTransformerLM(tiny_checkpoint, mode="soft", device="cpu")
        assert lm.cfg["d_model"] == 64
        assert lm.cfg["n_layers"] == 2
        assert lm.max_seq_len == 32

    def test_loglikelihood_returns_finite_negative_values(self, tiny_checkpoint):
        from lm_eval_adapter import SparseTransformerLM

        class FakeReq:
            def __init__(self, args):
                self.args = args

        lm = SparseTransformerLM(tiny_checkpoint, mode="soft", device="cpu")
        reqs = [FakeReq(("The cat sat on the", " mat"))]
        results = lm.loglikelihood(reqs)
        assert len(results) == 1
        logprob, is_greedy = results[0]
        assert torch.isfinite(torch.tensor(logprob))
        assert logprob < 0, "log-probability of a token sequence must be negative"
        assert isinstance(is_greedy, bool)

    def test_hard_mode_differs_from_soft_mode(self, tiny_checkpoint):
        """soft and hard forward paths take different code paths in
        get_logits -- confirm the adapter actually respects the mode switch
        (an untrained random model's soft vs hard logits will differ)."""
        from lm_eval_adapter import SparseTransformerLM

        class FakeReq:
            def __init__(self, args):
                self.args = args

        lm_soft = SparseTransformerLM(tiny_checkpoint, mode="soft", device="cpu")
        lm_hard = SparseTransformerLM(tiny_checkpoint, mode="hard", k=4, device="cpu")
        reqs = [FakeReq(("The cat sat on the", " mat"))]
        soft_result = lm_soft.loglikelihood(reqs)[0][0]
        hard_result = lm_hard.loglikelihood(reqs)[0][0]
        assert soft_result != pytest.approx(hard_result, abs=1e-9), (
            "soft and hard modes should score differently -- if identical, "
            "the mode switch isn't actually wired through to get_logits"
        )

    def test_context_truncated_from_left_not_continuation(self, tiny_checkpoint):
        """If context+continuation exceeds max_seq_len, the CONTEXT must be
        truncated (from the left), never the continuation -- matching
        eval_downstream.py's own build_example convention."""
        from lm_eval_adapter import SparseTransformerLM

        lm = SparseTransformerLM(tiny_checkpoint, mode="soft", device="cpu")
        long_context_ids = list(range(100, 140))  # 40 tokens, > max_seq_len=32
        continuation_ids = [7, 8, 9]
        per_token = lm._logprobs_for_continuation(long_context_ids, continuation_ids)
        assert len(per_token) == 3, "must score all 3 continuation tokens despite truncation"

    def test_loglikelihood_rolling_handles_long_text_in_chunks(self, tiny_checkpoint):
        from lm_eval_adapter import SparseTransformerLM

        class FakeReq:
            def __init__(self, args):
                self.args = args

        lm = SparseTransformerLM(tiny_checkpoint, mode="soft", device="cpu")
        text = "word " * 200  # tokenizes to well over max_seq_len=32
        result = lm.loglikelihood_rolling([FakeReq((text,))])
        assert len(result) == 1
        assert torch.isfinite(torch.tensor(result[0]))

    def test_is_greedy_uses_single_forward_pass_not_one_per_token(self, tiny_checkpoint, monkeypatch):
        """is_greedy must be derived from the SAME forward pass already used
        for loglikelihood scoring -- not one extra forward pass per
        continuation token (the old _argmax_continuation approach), which
        made real multi-choice-task runs (HellaSwag/ARC-e/WinoGrande, ~4
        continuations x 200 examples) far too slow to finish."""
        from lm_eval_adapter import SparseTransformerLM
        import lm_eval_adapter

        class FakeReq:
            def __init__(self, args):
                self.args = args

        lm = SparseTransformerLM(tiny_checkpoint, mode="soft", device="cpu")

        call_count = {"n": 0}
        real_get_logits = lm_eval_adapter.get_logits

        def counting_get_logits(*args, **kwargs):
            call_count["n"] += 1
            return real_get_logits(*args, **kwargs)

        monkeypatch.setattr(lm_eval_adapter, "get_logits", counting_get_logits)

        # 3-token continuation: the buggy version would make 1 (scoring) + 3
        # (one per token via _argmax_continuation) = 4 calls. The fix must
        # make exactly 1.
        reqs = [FakeReq(("The cat sat on the", " mat and slept"))]
        results = lm.loglikelihood(reqs)

        assert call_count["n"] == 1, (
            f"expected exactly 1 forward pass (shared for scoring + greedy check), "
            f"got {call_count['n']} -- is_greedy is still doing extra passes"
        )
        logprob, is_greedy = results[0]
        assert torch.isfinite(torch.tensor(logprob))
        assert isinstance(is_greedy, bool)

    def test_is_greedy_correctness_against_known_logits(self, tiny_checkpoint, monkeypatch):
        """Hand-verify is_greedy against fabricated logits where the correct
        answer is known by construction, not just 'doesn't crash'. Exercises
        the shared internal scoring method directly with known token ids,
        bypassing the tokenizer for full control."""
        from lm_eval_adapter import SparseTransformerLM
        import lm_eval_adapter

        lm = SparseTransformerLM(tiny_checkpoint, mode="soft", device="cpu")
        vocab = 50257

        # context (2 toks) + continuation (2 toks) -> full length 4.
        # logits[pos] predicts token at pos+1. Continuation starts at index
        # len(context_ids)=2 in `full`, so continuation tok 0 is predicted by
        # position 1, continuation tok 1 is predicted by position 2.
        context_ids = [10, 11]
        continuation_ids = [20, 21]
        fake_logits = torch.full((1, 4, vocab), -10.0)
        fake_logits[0, 1, continuation_ids[0]] = 10.0   # position 1 correctly predicts tok 20 (match)
        fake_logits[0, 2, 999] = 10.0                    # position 2 predicts 999, not 21 (mismatch)

        def fake_get_logits(model, input_ids, mode, k=64):
            return fake_logits

        monkeypatch.setattr(lm_eval_adapter, "get_logits", fake_get_logits)

        per_token, is_greedy = lm._score_continuation(context_ids, continuation_ids)
        assert len(per_token) == 2
        for lp in per_token:
            assert torch.isfinite(torch.tensor(lp))
        # One mismatch (position 2) is enough to make the whole continuation non-greedy.
        assert is_greedy is False, "position 2 predicts 999, not the real continuation token 21"

        # Now make BOTH positions match -> is_greedy must flip to True.
        fake_logits[0, 2, continuation_ids[1]] = 10.0
        fake_logits[0, 2, 999] = -10.0
        per_token2, is_greedy2 = lm._score_continuation(context_ids, continuation_ids)
        assert is_greedy2 is True, "both positions now correctly predict the continuation"
