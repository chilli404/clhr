"""Acceptance tests for scripts/eval_hierarchical_checkpoint.py.

These tests must fail before the driver script exists (there is nothing to
import / invoke yet). They are written first, per the task's TDD contract.

CRITICAL CORRECTNESS CONTEXT (see driver docstring / final report for the
full writeup): the real production checkpoints written by
src/sparse_attention_hierarchical.py (tokens_*.pt, final.pt, latest.pt) do
NOT store block_size/top_k_blocks/local_window anywhere -- verified by
reading the torch.save() call sites in that module, which only ever save
{"model", "step", "tokens"/"tokens_seen"}. Nor does any persistent tensor
shape in HierarchicalSparseTransformer depend on those three values (the
block-level RoPE cache buffers are registered with persistent=False and are
excluded from state_dict; the gate-projection shapes depend only on
n_heads/d_gate, which are tied to --model-size, not to block_size/top_k/
local_window).

So this test file exercises BOTH of the driver's two config paths:

  * "checkpoint" path (self-configuring, strongly preferred): the
    checkpoint payload carries a "config" dict. This is the ONLY
    deterministic mechanism available for catching a block_size /
    top_k_blocks / local_window mismatch, since no tensor shape encodes
    them. test_driver_rejects_geometry_mismatch and
    test_driver_reads_config_from_checkpoint_when_present exercise it.

  * "cli" path (legacy / real production checkpoint format): no "config"
    key present, so block_size/top_k_blocks/local_window/seq_len are
    required CLI args. test_runs_end_to_end_on_tiny_checkpoint exercises
    this path against a checkpoint shaped exactly like a real
    tokens_*.pt/final.pt file.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from sparse_attention_hierarchical import (  # noqa: E402
    HierarchicalSparseTransformer,
    MODEL_CONFIGS,
    CyclingTokenDataset,
)

DRIVER_PATH = _REPO / "scripts" / "eval_hierarchical_checkpoint.py"

# Geometry shared by the tiny end-to-end fixtures below.
TINY_SEQ_LEN = 8
TINY_BLOCK_SIZE = 4
TINY_TOP_K = 1
TINY_LOCAL_WINDOW = 4
VOCAB = MODEL_CONFIGS["small"]["vocab_size"]


def _load_driver_module():
    """Import the driver script as a module, for unit-testing helpers
    (e.g. state-dict prefix stripping) without paying subprocess cost."""
    if not DRIVER_PATH.exists():
        pytest.fail(f"driver script not found at {DRIVER_PATH}")
    spec = importlib.util.spec_from_file_location(
        "eval_hierarchical_checkpoint", DRIVER_PATH,
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _tiny_model(seq_len: int = TINY_SEQ_LEN,
                block_size: int = TINY_BLOCK_SIZE,
                top_k_blocks: int = TINY_TOP_K,
                local_window: int = TINY_LOCAL_WINDOW) -> HierarchicalSparseTransformer:
    """A model built with the real 'small' preset (so the driver's own
    MODEL_CONFIGS["small"]-based reconstruction matches), but a very short
    seq_len so forward passes stay cheap."""
    cfg = dict(MODEL_CONFIGS["small"])
    torch.manual_seed(0)
    model = HierarchicalSparseTransformer(
        **cfg, block_size=block_size, top_k_blocks=top_k_blocks,
        local_window=local_window, max_seq_len=seq_len, dropout=0.0,
    ).float()
    return model


def _write_tiny_token_file(path: Path, n_tokens: int, seed: int) -> None:
    g = torch.Generator().manual_seed(seed)
    tokens = torch.randint(0, VOCAB, (n_tokens,), generator=g, dtype=torch.long)
    torch.save(tokens, path)


def _make_data_dir(tmp_path: Path, n_tokens: int = 200) -> Path:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_tiny_token_file(data_dir / "wt103_val_tokens.pt", n_tokens, seed=1)
    _write_tiny_token_file(data_dir / "wt103_train_tokens.pt", n_tokens, seed=2)
    return data_dir


def _run_driver(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(DRIVER_PATH), *args],
        capture_output=True, text=True, timeout=120,
    )


def _train_two_steps(model: HierarchicalSparseTransformer,
                      seq_len: int) -> None:
    """Two tiny optimizer steps, just enough to make the checkpoint
    non-random / exercise the same forward path used at eval time."""
    tokens = torch.randint(0, VOCAB, (seq_len * 4 + 1,), dtype=torch.long)
    ds = CyclingTokenDataset(tokens, seq_len)
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    it = iter(loader)
    for _ in range(2):
        batch = next(it)
        x, y = batch[:, :-1], batch[:, 1:]
        logits, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()


# ---------------------------------------------------------------------------
# Fixtures: build checkpoints once, reuse across the tests that only assert
# on different facets of the same run's output.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def data_dir(tmp_path_factory) -> Path:
    return _make_data_dir(tmp_path_factory.mktemp("hier_eval_data"))


@pytest.fixture(scope="module")
def checkpoint_with_config(tmp_path_factory) -> Path:
    """Mimics a self-describing checkpoint (the 'strongly preferred' case).
    Also doubles as the training-step fixture used by the config-source
    test, so it carries real (if tiny) trained weights."""
    model = _tiny_model()
    _train_two_steps(model, TINY_SEQ_LEN)
    ckpt_path = tmp_path_factory.mktemp("hier_eval_ckpt") / "with_config.pt"
    torch.save({
        "model": model.state_dict(),
        "step": 2,
        "tokens_seen": 2 * TINY_SEQ_LEN,
        "config": {
            "block_size": TINY_BLOCK_SIZE,
            "top_k_blocks": TINY_TOP_K,
            "local_window": TINY_LOCAL_WINDOW,
            "seq_len": TINY_SEQ_LEN,
            "model_size": "small",
        },
    }, ckpt_path)
    return ckpt_path


@pytest.fixture(scope="module")
def checkpoint_legacy_no_config(tmp_path_factory) -> Path:
    """Mimics the REAL production checkpoint format written by
    src/sparse_attention_hierarchical.py's milestone/final save calls:
    only {"model", "step", "tokens_seen"} -- no architecture config."""
    model = _tiny_model()
    _train_two_steps(model, TINY_SEQ_LEN)
    ckpt_path = tmp_path_factory.mktemp("hier_eval_ckpt_legacy") / "final.pt"
    torch.save({
        "model": model.state_dict(),
        "step": 2,
        "tokens_seen": 2 * TINY_SEQ_LEN,
    }, ckpt_path)
    return ckpt_path


@pytest.fixture(scope="module")
def config_source_run_output(checkpoint_with_config, data_dir, tmp_path_factory) -> dict:
    """Runs the driver once against the self-describing checkpoint with NO
    geometry CLI flags at all, so the driver must read them from the
    checkpoint. Shared by the tests that only inspect different keys of
    the same successful run's JSON output."""
    out_path = tmp_path_factory.mktemp("hier_eval_out") / "result.json"
    result = _run_driver([
        "--checkpoint", str(checkpoint_with_config),
        "--data-dir", str(data_dir),
        "--output", str(out_path),
        "--model-size", "small",
    ])
    assert result.returncode == 0, (
        f"driver failed unexpectedly.\nstdout={result.stdout}\n"
        f"stderr={result.stderr}"
    )
    assert out_path.exists()
    return json.loads(out_path.read_text())


# ---------------------------------------------------------------------------
# 1. THE MOST IMPORTANT TEST
# ---------------------------------------------------------------------------

def test_driver_rejects_geometry_mismatch(checkpoint_with_config, data_dir, tmp_path):
    """A CLI-supplied --block-size that contradicts the checkpoint's own
    stored config must be a hard, loud, non-zero-exit failure -- never a
    silent wrong-geometry evaluation."""
    out_path = tmp_path / "result.json"
    wrong_block_size = TINY_BLOCK_SIZE * 2  # 32 vs 64 style contradiction
    result = _run_driver([
        "--checkpoint", str(checkpoint_with_config),
        "--data-dir", str(data_dir),
        "--output", str(out_path),
        "--model-size", "small",
        "--block-size", str(wrong_block_size),
    ])
    assert result.returncode != 0, (
        f"driver must reject a block_size contradiction, but exited 0.\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )
    combined = (result.stdout + result.stderr).lower()
    assert "block_size" in combined or "block-size" in combined
    assert str(TINY_BLOCK_SIZE) in combined
    assert str(wrong_block_size) in combined
    assert not out_path.exists()


# ---------------------------------------------------------------------------
# 2. Self-configuring from checkpoint
# ---------------------------------------------------------------------------

def test_driver_reads_config_from_checkpoint_when_present(config_source_run_output):
    out = config_source_run_output
    assert out["config_source"] == "checkpoint"
    assert out["block_size"] == TINY_BLOCK_SIZE
    assert out["top_k_blocks"] == TINY_TOP_K
    assert out["local_window"] == TINY_LOCAL_WINDOW
    assert out["seq_len"] == TINY_SEQ_LEN


# ---------------------------------------------------------------------------
# 3. state_dict prefix stripping (pure unit test, no subprocess)
# ---------------------------------------------------------------------------

def test_state_dict_prefix_stripping():
    driver = _load_driver_module()
    raw = {
        "module.tok_emb.weight": torch.zeros(2, 2),
        "module._orig_mod.final_norm.weight": torch.ones(2),
        "_orig_mod.lm_head.weight": torch.zeros(2, 2),
        "layers.0.attn_norm.weight": torch.ones(2),
    }
    cleaned = driver._strip_prefixes(raw)
    assert set(cleaned.keys()) == {
        "tok_emb.weight",
        "final_norm.weight",
        "lm_head.weight",
        "layers.0.attn_norm.weight",
    }


# ---------------------------------------------------------------------------
# 4. Required output keys
# ---------------------------------------------------------------------------

def test_output_json_has_required_keys(config_source_run_output):
    out = config_source_run_output
    required = {
        "native_nll", "closed_loop_hierarchical_hard_nll", "G_CL",
        "gate_utility", "random_block_hard_nll", "dense_nll",
        "checkpoint", "block_size", "top_k_blocks", "local_window",
        "seq_len", "tokens_seen", "step", "config_source",
    }
    missing = required - out.keys()
    assert not missing, f"missing required keys: {missing}"


# ---------------------------------------------------------------------------
# 5. G_CL arithmetic consistency
# ---------------------------------------------------------------------------

def test_G_CL_equals_closed_loop_minus_native(config_source_run_output):
    out = config_source_run_output
    expected = out["closed_loop_hierarchical_hard_nll"] - out["native_nll"]
    assert out["G_CL"] == pytest.approx(expected, abs=1e-4)


# ---------------------------------------------------------------------------
# 6. End-to-end on a tiny checkpoint shaped like the real production format
# ---------------------------------------------------------------------------

def test_runs_end_to_end_on_tiny_checkpoint(checkpoint_legacy_no_config, data_dir, tmp_path):
    out_path = tmp_path / "result.json"
    result = _run_driver([
        "--checkpoint", str(checkpoint_legacy_no_config),
        "--data-dir", str(data_dir),
        "--output", str(out_path),
        "--model-size", "small",
        "--seq-len", str(TINY_SEQ_LEN),
        "--block-size", str(TINY_BLOCK_SIZE),
        "--top-k-blocks", str(TINY_TOP_K),
        "--local-window", str(TINY_LOCAL_WINDOW),
    ])
    assert result.returncode == 0, (
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )
    out = json.loads(out_path.read_text())
    assert out["config_source"] == "cli"
    assert isinstance(out["native_nll"], float)
    assert isinstance(out["G_CL"], float)
    assert out["step"] == 2
    assert out["tokens_seen"] == 2 * TINY_SEQ_LEN
