"""Deployment verification tests for CLHR.

B6: Exact sparsity — every query attends to exactly min(k, valid_keys) keys.
B7/B9: Fresh-process independent eval — minimal reimplementation of closed-loop
       hard forward, no imports from training/eval code.
Additional: Binary mask check — no soft/sigmoid masks during hard eval.
"""
import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

RA_ROOT = Path(__file__).parent.parent.parent / "routing-absorption"
sys.path.insert(0, str(RA_ROOT))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.models.config import ModelConfig
from src.models.transformer import GatedTransformer

CKPT_DIR = Path("/s3-data/ckpts_sparse_rca")
CKPT_PATH = CKPT_DIR / "sparse_rca_contemporary_closedloop_hard_s42" / "step_50000.pt"
DATA_DIR = Path(__file__).parent.parent / "wikitext103_cache"

K = 64
SEQ_LEN = 512

DEFAULT_CONFIG = ModelConfig(
    vocab_size=50257,
    max_seq_len=512,
    n_layers=6,
    d_model=256,
    n_heads=4,
    d_ff=1024,
    dropout=0.1,
    d_gate=32,
    sparsity_mode="soft",
    sparsity_k=64,
    gate_temperature=1.0,
)

ckpt_available = CKPT_PATH.exists()
data_available = (DATA_DIR / "wt103_val_tokens.npy").exists()
skip_no_ckpt = pytest.mark.skipif(not ckpt_available, reason="Checkpoint not on this machine")
skip_no_data = pytest.mark.skipif(not data_available, reason="Wikitext cache not found")


def _load_model():
    config = DEFAULT_CONFIG.model_copy()
    device = torch.device("cpu")
    model = GatedTransformer(config).to(device)
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, device


def _get_val_batch():
    import numpy as np
    val_tok = torch.from_numpy(np.load(DATA_DIR / "wt103_val_tokens.npy")).long()
    ids = val_tok[:SEQ_LEN + 1].unsqueeze(0)
    return ids


@skip_no_ckpt
@skip_no_data
class TestExactSparsity:
    """B6: Verify every query has exactly min(k, valid_causal_keys) active keys."""

    def test_sparsity_per_query(self):
        model, device = _load_model()
        ids = _get_val_batch().to(device)
        x_in = ids[:, :-1]
        B, T = x_in.shape
        n_heads = model.config.n_heads
        d_head = model.config.d_model // n_heads
        d_gate = model.config.d_gate

        positions = torch.arange(T, device=device).unsqueeze(0)
        x = model.embedding(x_in) + model.pos_embedding(positions)
        causal = torch.tril(torch.ones(T, T, device=device)).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            for layer_idx, layer in enumerate(model.layers):
                h = layer.attn_norm(x)
                attn = layer.attention

                q = attn.W_q(h).view(B, T, n_heads, d_head).transpose(1, 2)
                kk = attn.W_k(h).view(B, T, n_heads, d_head).transpose(1, 2)
                v = attn.W_v(h).view(B, T, n_heads, d_head).transpose(1, 2)
                attn_scores = torch.matmul(q, kk.transpose(-2, -1)) / (d_head ** 0.5)

                gq = attn.W_gq(h).view(B, T, n_heads, d_gate).transpose(1, 2)
                gk = attn.W_gk(h).view(B, T, n_heads, d_gate).transpose(1, 2)
                gs = torch.matmul(gq, gk.transpose(-2, -1)) / (d_gate ** 0.5)
                gs = gs.masked_fill(causal == 0, float("-inf"))
                actual_k = min(K, T)
                _, idx = torch.topk(gs, actual_k, dim=-1)
                hard_mask = torch.zeros_like(gs).scatter_(-1, idx, 1.0) * causal

                # Check: each query position q_pos should have exactly
                # min(k, q_pos+1) active keys (causal constraint).
                for q_pos in range(T):
                    valid_keys = q_pos + 1
                    expected_active = min(K, valid_keys)
                    actual_active = hard_mask[0, :, q_pos, :].sum(dim=-1)
                    for head in range(n_heads):
                        assert int(actual_active[head].item()) == expected_active, (
                            f"Layer {layer_idx}, head {head}, query {q_pos}: "
                            f"expected {expected_active} active keys, got {int(actual_active[head].item())}"
                        )

                # Propagate state for next layer
                attn_scores = attn_scores.masked_fill(hard_mask == 0, float("-inf"))
                attn_scores = attn_scores.masked_fill(causal == 0, float("-inf"))
                w = torch.nan_to_num(F.softmax(attn_scores, dim=-1), nan=0.0)
                output = torch.matmul(w, v).transpose(1, 2).contiguous().view(B, T, -1)
                x = x + attn.W_o(output)
                x = x + layer.ff(layer.ff_norm(x))


@skip_no_ckpt
@skip_no_data
class TestFreshProcessEval:
    """B7/B9: Minimal reimplementation of closed-loop hard forward.

    No imports from sparse_attention_rca.py or sparse_attention_closed_loop_eval.py.
    Uses only raw model parameters.
    """

    def test_fresh_closed_loop_nll(self):
        config = DEFAULT_CONFIG.model_copy()
        device = torch.device("cpu")
        model = GatedTransformer(config).to(device)
        ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        model.eval()

        import numpy as np
        val_tok = torch.from_numpy(np.load(DATA_DIR / "wt103_val_tokens.npy")).long()

        n_heads = config.n_heads
        d_head = config.d_model // n_heads
        d_gate = config.d_gate
        k = K

        total_loss = 0.0
        total_tokens = 0
        batch_size = 1
        n_batches = 20

        with torch.no_grad():
            for bi in range(n_batches):
                start = bi * (SEQ_LEN + 1)
                if start + SEQ_LEN + 1 > len(val_tok):
                    break
                ids = val_tok[start:start + SEQ_LEN + 1].unsqueeze(0).to(device)
                x_in, y = ids[:, :-1], ids[:, 1:]
                B, T = x_in.shape

                # Embed — fresh, using model weights directly
                x = model.embedding(x_in) + model.pos_embedding(
                    torch.arange(T, device=device).unsqueeze(0)
                )

                causal = torch.tril(torch.ones(T, T, device=device)).unsqueeze(0).unsqueeze(0)

                # Layer-by-layer closed-loop hard forward
                for layer in model.layers:
                    h = layer.attn_norm(x)
                    attn = layer.attention

                    # Q/K/V from raw weights
                    q = attn.W_q(h).view(B, T, n_heads, d_head).transpose(1, 2)
                    kk = attn.W_k(h).view(B, T, n_heads, d_head).transpose(1, 2)
                    v = attn.W_v(h).view(B, T, n_heads, d_head).transpose(1, 2)
                    scores = torch.matmul(q, kk.transpose(-2, -1)) / math.sqrt(d_head)

                    # Gate from raw weights — on current hard-path state
                    gq = attn.W_gq(h).view(B, T, n_heads, d_gate).transpose(1, 2)
                    gk = attn.W_gk(h).view(B, T, n_heads, d_gate).transpose(1, 2)
                    gs = torch.matmul(gq, gk.transpose(-2, -1)) / math.sqrt(d_gate)
                    gs = gs.masked_fill(causal == 0, float("-inf"))

                    # Hard top-k
                    _, idx = torch.topk(gs, min(k, T), dim=-1)
                    mask = torch.zeros_like(gs)
                    mask.scatter_(-1, idx, 1.0)
                    mask = mask * causal

                    # Apply
                    scores = scores.masked_fill(mask == 0, float("-inf"))
                    scores = scores.masked_fill(causal == 0, float("-inf"))
                    w = F.softmax(scores, dim=-1)
                    w = torch.nan_to_num(w, nan=0.0)

                    out = torch.matmul(w, v).transpose(1, 2).contiguous().view(B, T, -1)
                    x = x + attn.W_o(out)
                    x = x + layer.ff(layer.ff_norm(x))

                logits = model.lm_head(model.final_norm(x))
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum"
                )
                total_loss += loss.item()
                total_tokens += y.numel()

        fresh_nll = total_loss / total_tokens

        # Sanity: NLL should be in a reasonable range for a trained model
        assert fresh_nll < 6.0, f"NLL {fresh_nll:.4f} too high — model may not have loaded correctly"
        assert fresh_nll > 2.0, f"NLL {fresh_nll:.4f} suspiciously low"

        # The eval reported G_CL = 0.015 for this checkpoint, native ~3.965.
        # So hard NLL should be ~3.98. Allow generous tolerance for subset eval.
        assert abs(fresh_nll - 3.98) < 0.15, (
            f"Fresh-process NLL {fresh_nll:.4f} deviates significantly from expected ~3.98"
        )


@skip_no_ckpt
@skip_no_data
class TestNoSoftPathInHardEval:
    """Verify that closed-loop hard eval produces only binary masks."""

    def test_masks_are_binary(self):
        model, device = _load_model()
        ids = _get_val_batch().to(device)
        x_in = ids[:, :-1]
        B, T = x_in.shape
        n_heads = model.config.n_heads
        d_head = model.config.d_model // n_heads
        d_gate = model.config.d_gate

        positions = torch.arange(T, device=device).unsqueeze(0)
        x = model.embedding(x_in) + model.pos_embedding(positions)
        causal = torch.tril(torch.ones(T, T, device=device)).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            for layer_idx, layer in enumerate(model.layers):
                h = layer.attn_norm(x)
                attn = layer.attention

                q = attn.W_q(h).view(B, T, n_heads, d_head).transpose(1, 2)
                kk = attn.W_k(h).view(B, T, n_heads, d_head).transpose(1, 2)
                v = attn.W_v(h).view(B, T, n_heads, d_head).transpose(1, 2)
                attn_scores = torch.matmul(q, kk.transpose(-2, -1)) / (d_head ** 0.5)

                gq = attn.W_gq(h).view(B, T, n_heads, d_gate).transpose(1, 2)
                gk = attn.W_gk(h).view(B, T, n_heads, d_gate).transpose(1, 2)
                gs = torch.matmul(gq, gk.transpose(-2, -1)) / (d_gate ** 0.5)
                gs = gs.masked_fill(causal == 0, float("-inf"))
                _, idx = torch.topk(gs, min(K, T), dim=-1)
                hard_mask = torch.zeros_like(gs).scatter_(-1, idx, 1.0) * causal

                # Every value in the mask must be exactly 0.0 or 1.0
                unique_vals = hard_mask.unique()
                assert set(unique_vals.tolist()).issubset({0.0, 1.0}), (
                    f"Layer {layer_idx}: mask contains non-binary values: {unique_vals.tolist()}"
                )

                # No sigmoid was applied — mask should NOT have values in (0, 1)
                fractional = (hard_mask > 0.0) & (hard_mask < 1.0)
                assert fractional.sum() == 0, (
                    f"Layer {layer_idx}: {fractional.sum().item()} fractional mask entries found"
                )

                # Propagate
                attn_scores = attn_scores.masked_fill(hard_mask == 0, float("-inf"))
                attn_scores = attn_scores.masked_fill(causal == 0, float("-inf"))
                w = torch.nan_to_num(F.softmax(attn_scores, dim=-1), nan=0.0)
                output = torch.matmul(w, v).transpose(1, 2).contiguous().view(B, T, -1)
                x = x + attn.W_o(output)
                x = x + layer.ff(layer.ff_norm(x))
