"""Acceptance tests for the provenance fixes in
src/sparse_attention_hierarchical.py:

  1. Result JSONs now record which corpus/data_dir a run trained on, plus
     (when cheaply obtainable) train_tokens_available / epochs_over_corpus.
     Motivation: a wrong-corpus config was inferred from NLL magnitudes
     instead of read, because the corpus was nowhere in the output JSON.

  2. Checkpoints now carry a "config" dict describing the routing geometry
     they were trained with (block_size, top_k_blocks, local_window, ...).
     Motivation: nothing in HierarchicalSparseTransformer's persistent
     state (tensor shapes) depends on block_size/top_k_blocks/local_window
     -- the block-RoPE caches are persistent=False -- so evaluating a
     checkpoint with the wrong --block-size silently produced plausible
     but meaningless numbers. scripts/eval_hierarchical_checkpoint.py
     already has a self-describing path that reads a "config" key and
     hard-fails on CLI/checkpoint contradiction; it just never had a
     checkpoint that carried one.

Both directions are tested with a real (tiny) call into
train_experiment() -- not a hand-built fake -- so the tests fail if the
actual training loop stops populating these fields.

Backwards compatibility: a legacy checkpoint lacking "config" must still
load via the eval driver's CLI-required path exactly as before
(test_legacy_checkpoint_without_config_still_loads).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

import sparse_attention_hierarchical as sah  # noqa: E402

DRIVER_PATH = _REPO / "scripts" / "eval_hierarchical_checkpoint.py"

VOCAB = sah.MODEL_CONFIGS["small"]["vocab_size"]

# Tiny geometry shared by every run in this file -- keeps the real "small"
# preset (d_model=512, n_layers=12, ...) cheap by using a minuscule seq_len
# and a 2-step run, matching the pattern already used in
# tests/test_hierarchical_eval_driver.py's end-to-end driver tests.
SEQ_LEN = 8
BLOCK_SIZE = 4
TOP_K = 1
LOCAL_WINDOW = 4
N_TRAIN_TOKENS = 200
TOTAL_TOKENS = 16  # micro_batch(1) * seq_len(8) * grad_accum(1) -> 2 steps


def _write_tiny_pt(path: Path, n_tokens: int, seed: int) -> None:
    g = torch.Generator().manual_seed(seed)
    tokens = torch.randint(0, VOCAB, (n_tokens,), generator=g, dtype=torch.long)
    torch.save(tokens, path)


def _make_wikitext_data_dir(base: Path, n_tokens: int = N_TRAIN_TOKENS) -> Path:
    data_dir = base / "wikitext_data"
    data_dir.mkdir()
    _write_tiny_pt(data_dir / "wt103_train_tokens.pt", n_tokens, seed=1)
    _write_tiny_pt(data_dir / "wt103_val_tokens.pt", n_tokens, seed=2)
    return data_dir


def _make_fineweb_data_dir(base: Path, n_tokens: int = N_TRAIN_TOKENS) -> Path:
    """A fineweb-edu-shaped data dir whose train dataset is a streaming
    ShardedTokenDataset (IterableDataset) with no `.tokens` attribute --
    the case where train_tokens_available must be *omitted*, not guessed.
    """
    data_dir = base / "fineweb_data"
    (data_dir / "fineweb_edu_shards").mkdir(parents=True)
    (data_dir / "fineweb_edu_eval").mkdir(parents=True)
    rng = np.random.default_rng(3)
    shard = rng.integers(0, VOCAB, size=n_tokens, dtype=np.int32)
    np.save(data_dir / "fineweb_edu_shards" / "shard_0000.npy", shard)
    eval_rng = np.random.default_rng(4)
    eval_tokens = eval_rng.integers(0, VOCAB, size=n_tokens, dtype=np.int32)
    np.save(data_dir / "fineweb_edu_eval" / "eval0.npy", eval_tokens)
    return data_dir


def _run_train_experiment(data_dir: Path, checkpoint_dir: Path,
                          output_path: Path, condition: str = "standard",
                          seed: int = 0) -> dict:
    sah.train_experiment(
        condition=condition,
        seed=seed,
        data_dir=str(data_dir),
        checkpoint_dir=str(checkpoint_dir),
        output_path=str(output_path),
        model_size="small",
        seq_len=SEQ_LEN,
        block_size=BLOCK_SIZE,
        top_k_blocks=TOP_K,
        local_window=LOCAL_WINDOW,
        total_tokens=TOTAL_TOKENS,
        micro_batch=1,
        grad_accum=1,
        lr=1e-3,
        lambda_rca=1.0,
    )
    return json.loads(Path(output_path).read_text())


def _run_driver(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(DRIVER_PATH), *args],
        capture_output=True, text=True, timeout=180,
    )


# ---------------------------------------------------------------------------
# Fixtures: run train_experiment once per corpus type, reused across the
# tests that only inspect different facets of the same run's output.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def wikitext_run(tmp_path_factory) -> dict:
    base = tmp_path_factory.mktemp("wikitext_run")
    data_dir = _make_wikitext_data_dir(base)
    checkpoint_dir = base / "ckpt"
    output_path = base / "result.json"
    result = _run_train_experiment(data_dir, checkpoint_dir, output_path)
    return {
        "result": result,
        "data_dir": data_dir,
        "checkpoint_dir": checkpoint_dir,
        "final_ckpt": checkpoint_dir / "hierarchical_small_standard_s0" / "final.pt",
    }


@pytest.fixture(scope="module")
def fineweb_run(tmp_path_factory) -> dict:
    base = tmp_path_factory.mktemp("fineweb_run")
    data_dir = _make_fineweb_data_dir(base)
    checkpoint_dir = base / "ckpt"
    output_path = base / "result.json"
    result = _run_train_experiment(data_dir, checkpoint_dir, output_path)
    return {"result": result, "data_dir": data_dir}


# ---------------------------------------------------------------------------
# Hole 1: result JSON provenance
# ---------------------------------------------------------------------------

def test_result_json_records_corpus_and_data_dir(wikitext_run):
    result = wikitext_run["result"]
    assert result["corpus"] == "wikitext-103"
    assert result["data_dir"] == str(wikitext_run["data_dir"].resolve())


def test_epochs_over_corpus_computed_when_available(wikitext_run):
    result = wikitext_run["result"]
    assert result["train_tokens_available"] == N_TRAIN_TOKENS
    assert "epochs_over_corpus" in result
    expected = result["total_tokens"] / N_TRAIN_TOKENS
    assert result["epochs_over_corpus"] == pytest.approx(expected, abs=1e-9)


def test_epochs_over_corpus_absent_when_unavailable(fineweb_run):
    result = fineweb_run["result"]
    assert result["corpus"] == "fineweb-edu"
    # ShardedTokenDataset is a streaming IterableDataset with no `.tokens`
    # attribute -- the token count is not cheaply obtainable, so both keys
    # must be omitted rather than present-but-null/wrong.
    assert "train_tokens_available" not in result
    assert "epochs_over_corpus" not in result


# ---------------------------------------------------------------------------
# Hole 2: self-describing checkpoints
# ---------------------------------------------------------------------------

def test_checkpoint_carries_config(wikitext_run):
    ckpt = torch.load(wikitext_run["final_ckpt"], map_location="cpu",
                      weights_only=False)
    assert "config" in ckpt
    config = ckpt["config"]
    expected = {
        "model_size": "small",
        "seq_len": SEQ_LEN,
        "block_size": BLOCK_SIZE,
        "top_k_blocks": TOP_K,
        "local_window": LOCAL_WINDOW,
        "condition": "standard",
        "corpus": "wikitext-103",
        "d_model": sah.MODEL_CONFIGS["small"]["d_model"],
        "n_heads": sah.MODEL_CONFIGS["small"]["n_heads"],
        "n_layers": sah.MODEL_CONFIGS["small"]["n_layers"],
        "d_ff": sah.MODEL_CONFIGS["small"]["d_ff"],
        "d_gate": sah.MODEL_CONFIGS["small"]["d_gate"],
    }
    for key, value in expected.items():
        assert config.get(key) == value, f"config[{key!r}] wrong: {config.get(key)!r}"
    # Existing keys must be untouched.
    assert ckpt["step"] == 2
    assert ckpt["tokens_seen"] == TOTAL_TOKENS


def test_eval_driver_reads_new_config(wikitext_run):
    out_path = wikitext_run["checkpoint_dir"] / "eval_result.json"
    result = _run_driver([
        "--checkpoint", str(wikitext_run["final_ckpt"]),
        "--data-dir", str(wikitext_run["data_dir"]),
        "--output", str(out_path),
        "--model-size", "small",
    ])
    assert result.returncode == 0, (
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )
    out = json.loads(out_path.read_text())
    assert out["config_source"] == "checkpoint"
    assert out["block_size"] == BLOCK_SIZE
    assert out["top_k_blocks"] == TOP_K
    assert out["local_window"] == LOCAL_WINDOW
    assert out["seq_len"] == SEQ_LEN


def test_eval_driver_hard_fails_on_contradiction(wikitext_run, tmp_path):
    out_path = tmp_path / "should_not_exist.json"
    wrong_block_size = BLOCK_SIZE * 2
    result = _run_driver([
        "--checkpoint", str(wikitext_run["final_ckpt"]),
        "--data-dir", str(wikitext_run["data_dir"]),
        "--output", str(out_path),
        "--model-size", "small",
        "--block-size", str(wrong_block_size),
    ])
    assert result.returncode != 0
    combined = (result.stdout + result.stderr).lower()
    assert "block_size" in combined or "block-size" in combined
    assert str(BLOCK_SIZE) in combined
    assert str(wrong_block_size) in combined
    assert not out_path.exists()


def test_legacy_checkpoint_without_config_still_loads(wikitext_run, tmp_path):
    """A checkpoint shaped exactly like the pre-fix format (no "config"
    key) must still load through the driver's CLI-required path."""
    ckpt = torch.load(wikitext_run["final_ckpt"], map_location="cpu",
                      weights_only=False)
    legacy_ckpt = {k: v for k, v in ckpt.items() if k != "config"}
    assert "config" not in legacy_ckpt
    legacy_path = tmp_path / "legacy_final.pt"
    torch.save(legacy_ckpt, legacy_path)

    out_path = tmp_path / "legacy_result.json"
    result = _run_driver([
        "--checkpoint", str(legacy_path),
        "--data-dir", str(wikitext_run["data_dir"]),
        "--output", str(out_path),
        "--model-size", "small",
        "--seq-len", str(SEQ_LEN),
        "--block-size", str(BLOCK_SIZE),
        "--top-k-blocks", str(TOP_K),
        "--local-window", str(LOCAL_WINDOW),
    ])
    assert result.returncode == 0, (
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )
    out = json.loads(out_path.read_text())
    assert out["config_source"] == "cli"
    assert isinstance(out["native_nll"], float)
