"""Acceptance tests for the DDP data-duplication fix in
src/sparse_attention_hierarchical.py.

Bug: load_training_data / load_val_data called data_loading.load_corpus()
without rank/world_size, so every DDP rank read the identical shard set
and identical stream -- every multi-GPU run trained on 1/world_size of the
intended unique data while reporting the full token count.

Fix: thread rank/world_size from train_experiment's DDP setup into both
loader functions and on into load_corpus.

Decisions under test:
  1. Validation stays whole (rank=0, world_size=1) regardless of the
     global world_size, because only the master rank ever evaluates
     (see the `if is_master:` guard around load_val_data in
     train_experiment) and the reported NLL must stay comparable to the
     existing world_size=1 results in the paper.
  2. The non-sharded WikiText path (CyclingTokenDataset/TokenDataset via
     load_corpus) ignores rank/world_size entirely -- data_loading.py's
     load_corpus never forwards them for name in ("wikitext-103",
     "wikitext-103-v1"). So DDP WikiText runs still duplicate data; this
     is a second, latent instance of the same class of bug that this
     patch does NOT fix (fixing it would require changing
     CyclingTokenDataset/TokenDataset in data_loading.py, which is out of
     scope here). The test below documents this explicitly rather than
     leaving it silent.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

import sparse_attention_hierarchical as sah  # noqa: E402
from data_loading import ShardedTokenDataset, CyclingTokenDataset  # noqa: E402

VOCAB = sah.MODEL_CONFIGS["small"]["vocab_size"]
SEQ_LEN = 8


def _make_fineweb_shard_dir(base: Path, n_shards: int = 8) -> Path:
    data_dir = base / "fineweb_data"
    shard_dir = data_dir / "fineweb_edu_shards"
    shard_dir.mkdir(parents=True)
    for i in range(n_shards):
        tokens = np.random.randint(0, VOCAB, (200,), dtype=np.int64)
        np.save(str(shard_dir / f"shard_{i:03d}.npy"), tokens)
    eval_dir = data_dir / "fineweb_edu_eval"
    eval_dir.mkdir()
    eval_tokens = np.random.randint(0, VOCAB, (200,), dtype=np.int64)
    np.save(str(eval_dir / "eval_0.npy"), eval_tokens)
    return data_dir


def _shard_names(ds: ShardedTokenDataset) -> set[str]:
    files = sorted(Path(ds.shard_dir).glob("shard_*.npy"))
    return {
        s.name for i, s in enumerate(files)
        if i % ds.world_size == ds.rank
    }


class TestDdpRanksGetDisjointShards:
    def test_ddp_ranks_get_disjoint_shards(self, tmp_path):
        data_dir = _make_fineweb_shard_dir(tmp_path, n_shards=8)
        world_size = 4
        per_rank_shards = []
        for rank in range(world_size):
            ds = sah.load_training_data(
                str(data_dir), SEQ_LEN, seed=0,
                rank=rank, world_size=world_size,
            )
            assert isinstance(ds, ShardedTokenDataset)
            per_rank_shards.append(_shard_names(ds))

        # Pairwise disjoint.
        for i in range(world_size):
            for j in range(i + 1, world_size):
                assert per_rank_shards[i].isdisjoint(per_rank_shards[j]), (
                    f"rank {i} and rank {j} share shards: "
                    f"{per_rank_shards[i] & per_rank_shards[j]}"
                )

        # Union is the whole shard set.
        all_shards = {p.name for p in
                      (data_dir / "fineweb_edu_shards").glob("shard_*.npy")}
        union = set().union(*per_rank_shards)
        assert union == all_shards


class TestSingleRankUnchanged:
    def test_single_rank_unchanged(self, tmp_path):
        data_dir = _make_fineweb_shard_dir(tmp_path, n_shards=8)
        # Default call (no rank/world_size) must match explicit world_size=1.
        ds_default = sah.load_training_data(str(data_dir), SEQ_LEN, seed=0)
        ds_explicit = sah.load_training_data(
            str(data_dir), SEQ_LEN, seed=0, rank=0, world_size=1,
        )
        assert isinstance(ds_default, ShardedTokenDataset)
        assert ds_default.rank == ds_explicit.rank == 0
        assert ds_default.world_size == ds_explicit.world_size == 1
        assert _shard_names(ds_default) == _shard_names(ds_explicit)
        # And it's the full shard set -- single-rank behaviour is
        # bit-identical to pre-fix (world_size defaulted to 1 before too).
        all_shards = {p.name for p in
                      (data_dir / "fineweb_edu_shards").glob("shard_*.npy")}
        assert _shard_names(ds_default) == all_shards


class TestRankWorldSizeReachLoadCorpus:
    def test_load_training_data_forwards_rank_world_size(self, tmp_path, monkeypatch):
        data_dir = _make_fineweb_shard_dir(tmp_path, n_shards=2)
        captured = {}

        def fake_load_corpus(*args, **kwargs):
            captured.update(kwargs)
            captured["args"] = args
            return None  # forces fallback path, which we don't care about here

        monkeypatch.setattr(sah, "load_corpus", fake_load_corpus)
        monkeypatch.setattr(sah, "_HAS_DATA_LOADING", True)
        with pytest.raises(FileNotFoundError):
            # fallback raises because there's no wt103_*.pt in this dir;
            # what we care about is that load_corpus was called with the
            # rank/world_size we passed in.
            sah.load_training_data(
                str(data_dir), SEQ_LEN, seed=7, rank=3, world_size=8,
            )
        assert captured.get("rank") == 3
        assert captured.get("world_size") == 8
        assert captured.get("seed") == 7


class TestValidationStaysWhole:
    def test_load_val_data_ignores_global_world_size(self, tmp_path, monkeypatch):
        """Only rank 0 ever calls load_val_data (see the `is_master` guard
        in train_experiment). Sharding validation would silently redefine
        the reported NLL to be over a 1/world_size slice, breaking
        comparability with the existing world_size=1 numbers in the paper.
        So load_val_data must always request the whole validation set from
        load_corpus, regardless of the DDP world_size passed to it.
        """
        captured = {}
        real_load_corpus = sah.load_corpus

        def spy_load_corpus(*args, **kwargs):
            captured.update(kwargs)
            return real_load_corpus(*args, **kwargs)

        monkeypatch.setattr(sah, "load_corpus", spy_load_corpus)

        data_dir = _make_fineweb_shard_dir(tmp_path, n_shards=4)
        sah.load_val_data(str(data_dir), SEQ_LEN, rank=2, world_size=4)

        assert captured.get("rank") == 0
        assert captured.get("world_size") == 1


class TestNonShardedPathStillDuplicates:
    def test_wikitext_cycling_dataset_ignores_rank_world_size(self):
        """Documents a second, NOT-fixed instance of the same bug class:
        for WikiText, load_corpus returns a CyclingTokenDataset (a flat
        in-memory token array), and data_loading.load_corpus never
        forwards rank/world_size for name in ("wikitext-103",
        "wikitext-103-v1") -- see src/data_loading.py's load_corpus body.
        So even after threading rank/world_size through
        load_training_data/load_val_data, a DDP WikiText run still has
        every rank iterate over the identical CyclingTokenDataset with no
        rank-based filtering. This is a real, currently-unfixed gap;
        fixing it would require changing CyclingTokenDataset/TokenDataset
        in data_loading.py, which is out of scope for this patch.
        """
        tokens = torch.arange(1000)
        ds_rank0 = CyclingTokenDataset(tokens, seq_len=32)
        ds_rank1 = CyclingTokenDataset(tokens, seq_len=32)
        # No rank/world_size parameters exist on this class at all --
        # confirmed by inspecting its constructor.
        import inspect
        sig = inspect.signature(CyclingTokenDataset.__init__)
        assert "rank" not in sig.parameters
        assert "world_size" not in sig.parameters
        # Every "rank" sees the identical sequence at the identical index.
        assert torch.equal(ds_rank0[0]["input_ids"], ds_rank1[0]["input_ids"])
        assert len(ds_rank0) == len(ds_rank1)
