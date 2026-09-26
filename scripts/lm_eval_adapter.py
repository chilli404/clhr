"""lm-eval-harness adapter for this project's self-contained SparseTransformer
(scripts/eval_downstream.py), so standard lm-eval-harness tasks (LAMBADA,
HellaSwag, PIQA, ARC-e, WinoGrande, RULER) can run soft-vs-hard, alongside
(not replacing) the custom eval_downstream.py harness already used tonight.

Not a HuggingFace model -- loglikelihood is computed by hand against
SparseTransformer.get_logits(model, input_ids, mode, k), matching exactly
how eval_downstream.py already scores soft vs hard.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

from eval_downstream import get_logits, load_checkpoint  # noqa: E402

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model


class SparseTransformerLM(LM):
    """mode: "soft" or "hard" -- selects which deployment path is scored."""

    def __init__(self, checkpoint: str, mode: str, k: int = 64, device: str = None):
        super().__init__()
        self._device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model, self.cfg, _ = load_checkpoint(checkpoint, self._device)
        self.mode = mode
        self.k = k
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.max_seq_len = self.cfg["max_seq_len"]

    @property
    def device(self):
        return self._device

    def _score_continuation(self, context_ids: list[int], continuation_ids: list[int]):
        """Single forward pass: returns (per_token_logprobs, is_greedy).

        is_greedy is derived from the SAME pass's logits (argmax at each
        continuation position) rather than a second round of incremental
        forward passes -- one pass per continuation token was the original
        (correct but far too slow) design, unworkable for multi-choice tasks
        (HellaSwag/ARC-e/WinoGrande: ~4 continuations x 200 examples each).
        """
        full = context_ids + continuation_ids
        # Truncate the CONTEXT from the left if needed -- never the continuation
        # (matches scripts/eval_downstream.py's build_example convention).
        if len(full) > self.max_seq_len:
            overflow = len(full) - self.max_seq_len
            context_ids = context_ids[overflow:]
            full = context_ids + continuation_ids
        input_ids = torch.tensor([full], dtype=torch.long, device=self.device)
        with torch.no_grad():
            logits = get_logits(self.model, input_ids, self.mode, k=self.k)
        logprobs = F.log_softmax(logits.float(), dim=-1)
        # logits[i] predicts token[i+1] -- continuation tokens start at
        # position len(context_ids) in `full`.
        start = len(context_ids)
        per_token = []
        is_greedy = True
        for i, tok in enumerate(continuation_ids):
            pos = start + i - 1
            per_token.append(logprobs[0, pos, tok].item())
            predicted = int(logits[0, pos].argmax().item())
            if predicted != tok:
                is_greedy = False
        return per_token, is_greedy

    def _logprobs_for_continuation(self, context_ids: list[int], continuation_ids: list[int]):
        return self._score_continuation(context_ids, continuation_ids)[0]

    def loglikelihood(self, requests):
        results = []
        for req in requests:
            context, continuation = req.args
            context_ids = self.tokenizer.encode(context)
            continuation_ids = self.tokenizer.encode(continuation)
            per_token, is_greedy = self._score_continuation(context_ids, continuation_ids)
            results.append((sum(per_token), is_greedy))
        return results

    def loglikelihood_rolling(self, requests):
        results = []
        for req in requests:
            (text,) = req.args
            ids = self.tokenizer.encode(text)
            total = 0.0
            for start in range(0, len(ids), self.max_seq_len):
                chunk = ids[start:start + self.max_seq_len]
                if len(chunk) < 2:
                    continue
                context_ids, continuation_ids = chunk[:1], chunk[1:]
                total += sum(self._logprobs_for_continuation(context_ids, continuation_ids))
            results.append(total)
        return results

    def generate_until(self, requests):
        raise NotImplementedError("generate_until not needed for the loglikelihood-based tasks used here")
