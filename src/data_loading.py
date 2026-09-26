"""Shared data loading module for tokenized corpora.

Consolidates the inline load_wikitext() patterns used across the codebase
and adds support for FineWeb-Edu and C4.

Two tokenizer backends are supported:
  - "gpt2": HuggingFace AutoTokenizer (Pattern A, sparse_attention_rca.py style)
  - "tiktoken_gpt2": tiktoken gpt2 encoding (Pattern B, sparse_attention_exact_protocol.py style)

Three dataset classes handle different corpus scales:
  - TokenDataset: fixed-length slicing of a flat 1D token tensor
  - CyclingTokenDataset: same, but allows cycling for multi-epoch training on small corpora
  - ShardedTokenDataset: memory-mapped IterableDataset for large corpora (FineWeb-Edu, C4)

Usage:
    from data_loading import load_corpus, prepare_fineweb_shards

    # WikiText-103 (small, fits in RAM)
    train_ds = load_corpus("wikitext-103", data_dir="./data", seq_len=512, split="train")
    val_ds   = load_corpus("wikitext-103", data_dir="./data", seq_len=512, split="validation")

    # FineWeb-Edu (large, sharded on disk)
    prepare_fineweb_shards("./data", tokenizer_name="gpt2", max_shards=10)
    train_ds = load_corpus("fineweb-edu", data_dir="./data", seq_len=512, split="train")
"""
from __future__ import annotations

import hashlib
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset


# ═══════════════════════════════════════════════════════════════════════
# Tokenizer helpers
# ═══════════════════════════════════════════════════════════════════════

def _get_tokenizer(tokenizer_name: str):
    """Return a tokenizer object with an .encode() method.

    For "gpt2": returns a HuggingFace AutoTokenizer.
    For "tiktoken_gpt2": returns a tiktoken encoding wrapped to present
    the same .encode(text) interface (uses encode_ordinary, no special tokens).
    """
    if tokenizer_name == "tiktoken_gpt2":
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")

        class _TiktokenWrapper:
            """Thin wrapper so tiktoken has the same .encode() signature."""
            def encode(self, text: str) -> list[int]:
                return enc.encode_ordinary(text)

        return _TiktokenWrapper()

    # Default: HuggingFace
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(tokenizer_name)


# ═══════════════════════════════════════════════════════════════════════
# Dataset classes
# ═══════════════════════════════════════════════════════════════════════

class TokenDataset(Dataset):
    """Fixed-length sequences from a flat 1D token tensor."""

    def __init__(self, tokens: torch.Tensor, seq_len: int):
        self.tokens = tokens
        self.seq_len = seq_len
        self.n_seqs = (len(tokens) - 1) // seq_len

    def __len__(self) -> int:
        return self.n_seqs

    def __getitem__(self, idx: int) -> dict:
        idx = idx % self.n_seqs
        start = idx * self.seq_len
        chunk = self.tokens[start : start + self.seq_len + 1]
        return {"input_ids": chunk}


class CyclingTokenDataset(TokenDataset):
    """Same as TokenDataset but cycles (for multi-epoch training on small corpora).

    Reports a virtual length 100x the real sequence count so that a DataLoader
    with ``drop_last=True`` keeps yielding batches across many epochs without
    requiring manual epoch management.
    """

    def __getitem__(self, idx: int) -> dict:
        idx = idx % self.n_seqs
        return super().__getitem__(idx)

    def __len__(self) -> int:
        return self.n_seqs * 100  # allow cycling


class ShardedTokenDataset(IterableDataset):
    """Memory-mapped sharded dataset for large corpora like FineWeb-Edu.

    Shards are ``.npy`` files in *shard_dir*, each containing ~100M token IDs
    stored as ``int32``.

    Iteration: shuffle shard order, memory-map each shard, yield fixed-length
    sequences.  DDP-aware: each rank processes every ``world_size``-th shard.
    """

    def __init__(
        self,
        shard_dir: str,
        seq_len: int,
        shuffle: bool = True,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 0,
        skip_sequences: int = 0,
    ):
        super().__init__()
        self.shard_dir = Path(shard_dir)
        self.seq_len = seq_len
        self.shuffle = shuffle
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        # Set to resume mid-stream; cleared automatically once consumed.
        self.skip_sequences = skip_sequences

    def _shard_rng(self, shard_name: str) -> random.Random:
        """Per-shard RNG that does not depend on traversal order.

        Uses sha256 rather than hash() so ordering is stable across
        processes regardless of PYTHONHASHSEED.
        """
        digest = hashlib.sha256(f"{self.seed}:{shard_name}".encode()).hexdigest()
        return random.Random(int(digest[:16], 16))

    def __iter__(self):
        shard_files = sorted(self.shard_dir.glob("shard_*.npy"))
        if not shard_files:
            raise FileNotFoundError(
                f"No shard_*.npy files found in {self.shard_dir}"
            )

        # Shard-level DDP: each rank takes every world_size-th shard
        shard_files = [
            s for i, s in enumerate(shard_files)
            if i % self.world_size == self.rank
        ]

        if self.shuffle:
            random.Random(self.seed).shuffle(shard_files)

        to_skip = self.skip_sequences
        # Consume the skip budget once, so re-iterating starts a fresh pass.
        self.skip_sequences = 0
        skipped = 0

        for shard_path in shard_files:
            tokens = np.load(str(shard_path), mmap_mode="r")
            n_seqs = (len(tokens) - 1) // self.seq_len
            if n_seqs == 0:
                continue

            # Whole-shard fast skip: reads only the .npy header.
            if skipped + n_seqs <= to_skip:
                skipped += n_seqs
                continue

            indices = list(range(n_seqs))
            if self.shuffle:
                self._shard_rng(shard_path.name).shuffle(indices)

            for idx in indices:
                if skipped < to_skip:
                    skipped += 1
                    continue
                start = idx * self.seq_len
                chunk = torch.from_numpy(
                    tokens[start : start + self.seq_len + 1].copy()
                ).long()
                yield {"input_ids": chunk}


# ═══════════════════════════════════════════════════════════════════════
# WikiText tokenization + caching
# ═══════════════════════════════════════════════════════════════════════

def _wikitext_hf_name(name: str) -> str:
    """Map short corpus name to HuggingFace dataset config name."""
    if name == "wikitext-103":
        return "wikitext-103-raw-v1"
    elif name == "wikitext-103-v1":
        return "wikitext-103-v1"
    else:
        raise ValueError(f"Unknown wikitext variant: {name}")


def _tokenize_wikitext(
    name: str,
    data_dir: str,
    split: str,
    tokenizer_name: str,
) -> torch.Tensor:
    """Tokenize WikiText-103 and cache as .npy.  Returns flat token tensor."""
    from datasets import load_dataset

    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    # Deterministic cache filename
    safe_tok = tokenizer_name.replace("/", "_")
    cache_name = f"{name}_{split}_{safe_tok}.npy"
    cache_path = data_path / cache_name

    if cache_path.exists():
        print(f"  [data_loading] Loading cached tokens from {cache_path}")
        arr = np.load(str(cache_path))
        return torch.from_numpy(arr).long()

    # Also check legacy names used by other scripts in the repo
    legacy_names = {
        ("wikitext-103", "train"): ["wt103_train_tokens.npy", "wt103_train_tokens.pt"],
        ("wikitext-103", "validation"): ["wt103_val_tokens.npy", "wt103_val_tokens.pt"],
        ("wikitext-103-v1", "train"): ["wt103_train_tokens.npy", "wt103_train_tokens.pt"],
        ("wikitext-103-v1", "validation"): ["wt103_val_tokens.npy", "wt103_val_tokens.pt"],
    }
    for legacy in legacy_names.get((name, split), []):
        lp = data_path / legacy
        if lp.exists():
            print(f"  [data_loading] Loading legacy cached tokens from {lp}")
            if legacy.endswith(".pt"):
                tokens = torch.load(str(lp), weights_only=True)
                if isinstance(tokens, dict) and "input_ids" in tokens:
                    tokens = tokens["input_ids"].reshape(-1)
                elif tokens.dim() > 1:
                    tokens = tokens.reshape(-1)
                return tokens.long()
            else:
                return torch.from_numpy(np.load(str(lp))).long()

    print(f"  [data_loading] Tokenizing {name} ({split}) with {tokenizer_name}...")
    hf_config = _wikitext_hf_name(name)
    ds = load_dataset(
        "Salesforce/wikitext",
        hf_config,
        cache_dir=str(data_path / "hf_cache"),
    )

    tokenizer = _get_tokenizer(tokenizer_name)
    all_ids: list[int] = []
    for item in ds[split]:
        text = item["text"]
        if text.strip():
            all_ids.extend(tokenizer.encode(text))

    tokens = torch.tensor(all_ids, dtype=torch.long)
    np.save(str(cache_path), tokens.numpy())
    print(f"  [data_loading] Cached {len(tokens):,} tokens to {cache_path}")
    return tokens


# ═══════════════════════════════════════════════════════════════════════
# FineWeb-Edu shard preparation
# ═══════════════════════════════════════════════════════════════════════

def prepare_fineweb_shards(
    data_dir: str,
    tokenizer_name: str = "gpt2",
    shard_size: int = 100_000_000,
    max_shards: Optional[int] = None,
) -> Path:
    """Download and tokenize FineWeb-Edu into shards.

    Creates ``data_dir/fineweb_edu_shards/shard_0000.npy``, ``shard_0001.npy``, etc.
    Each shard is a flat numpy array of token IDs (~100M tokens, ~400 MB).
    Uses the 10BT sample to keep download manageable.

    Returns:
        Path to the shard directory.
    """
    from datasets import load_dataset

    shard_dir = Path(data_dir) / "fineweb_edu_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    # Skip if shards already exist
    existing = sorted(shard_dir.glob("shard_*.npy"))
    if existing:
        print(f"  [data_loading] Found {len(existing)} existing FineWeb-Edu shards in {shard_dir}")
        if max_shards is not None and len(existing) >= max_shards:
            return shard_dir

    tokenizer = _get_tokenizer(tokenizer_name)

    print(f"  [data_loading] Streaming FineWeb-Edu (sample-10BT) and tokenizing...")
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
        revision="v1.0.0",
    )

    shard_buffer: list[int] = []
    shard_idx = len(existing)  # resume from last shard

    for example in ds:
        ids = tokenizer.encode(example["text"])
        shard_buffer.extend(ids)

        while len(shard_buffer) >= shard_size:
            arr = np.array(shard_buffer[:shard_size], dtype=np.int32)
            out_path = shard_dir / f"shard_{shard_idx:04d}.npy"
            np.save(str(out_path), arr)
            print(f"    Saved {out_path.name} ({shard_size:,} tokens)")
            shard_buffer = shard_buffer[shard_size:]
            shard_idx += 1

            if max_shards is not None and shard_idx >= max_shards:
                print(f"  [data_loading] Reached max_shards={max_shards}, stopping.")
                return shard_dir

    # Save remainder if substantial
    if len(shard_buffer) > shard_size // 10:
        arr = np.array(shard_buffer, dtype=np.int32)
        out_path = shard_dir / f"shard_{shard_idx:04d}.npy"
        np.save(str(out_path), arr)
        print(f"    Saved {out_path.name} ({len(shard_buffer):,} tokens, partial)")

    print(f"  [data_loading] FineWeb-Edu sharding complete: {shard_idx + 1} shards in {shard_dir}")
    return shard_dir


# ═══════════════════════════════════════════════════════════════════════
# C4 shard preparation
# ═══════════════════════════════════════════════════════════════════════

def prepare_c4_shards(
    data_dir: str,
    tokenizer_name: str = "gpt2",
    shard_size: int = 100_000_000,
    max_shards: Optional[int] = None,
) -> Path:
    """Download and tokenize C4 (English) into shards.

    Creates ``data_dir/c4_shards/shard_0000.npy``, ``shard_0001.npy``, etc.
    Each shard is a flat numpy array of token IDs (~100M tokens, ~400 MB).

    Returns:
        Path to the shard directory.
    """
    from datasets import load_dataset

    shard_dir = Path(data_dir) / "c4_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(shard_dir.glob("shard_*.npy"))
    if existing:
        print(f"  [data_loading] Found {len(existing)} existing C4 shards in {shard_dir}")
        if max_shards is not None and len(existing) >= max_shards:
            return shard_dir

    tokenizer = _get_tokenizer(tokenizer_name)

    print(f"  [data_loading] Streaming C4 (en) and tokenizing...")
    ds = load_dataset(
        "allenai/c4",
        "en",
        split="train",
        streaming=True,
    )

    shard_buffer: list[int] = []
    shard_idx = len(existing)

    for example in ds:
        ids = tokenizer.encode(example["text"])
        shard_buffer.extend(ids)

        while len(shard_buffer) >= shard_size:
            arr = np.array(shard_buffer[:shard_size], dtype=np.int32)
            out_path = shard_dir / f"shard_{shard_idx:04d}.npy"
            np.save(str(out_path), arr)
            print(f"    Saved {out_path.name} ({shard_size:,} tokens)")
            shard_buffer = shard_buffer[shard_size:]
            shard_idx += 1

            if max_shards is not None and shard_idx >= max_shards:
                print(f"  [data_loading] Reached max_shards={max_shards}, stopping.")
                return shard_dir

    if len(shard_buffer) > shard_size // 10:
        arr = np.array(shard_buffer, dtype=np.int32)
        out_path = shard_dir / f"shard_{shard_idx:04d}.npy"
        np.save(str(out_path), arr)
        print(f"    Saved {out_path.name} ({len(shard_buffer):,} tokens, partial)")

    print(f"  [data_loading] C4 sharding complete: {shard_idx + 1} shards in {shard_dir}")
    return shard_dir


# ═══════════════════════════════════════════════════════════════════════
# Main entry point: load_corpus()
# ═══════════════════════════════════════════════════════════════════════

def load_corpus(
    name: str,
    data_dir: str,
    seq_len: int = 512,
    split: str = "train",
    tokenizer_name: str = "gpt2",
    cycling: bool = True,
    rank: int = 0,
    world_size: int = 1,
    seed: int = 0,
) -> Dataset:
    """Load a tokenized corpus as a PyTorch Dataset.

    Args:
        name: ``"wikitext-103"`` | ``"wikitext-103-v1"`` | ``"fineweb-edu"`` | ``"c4"``
        data_dir: directory for caching tokenized data
        seq_len: sequence length (returns ``seq_len + 1`` tokens per sample)
        split: ``"train"`` or ``"validation"``
        tokenizer_name: ``"gpt2"`` (HuggingFace AutoTokenizer) or
            ``"tiktoken_gpt2"`` (tiktoken)
        cycling: if True and corpus is small (WikiText), use CyclingTokenDataset
        rank: DDP rank (only relevant for sharded datasets)
        world_size: DDP world size (only relevant for sharded datasets)

    Returns:
        Dataset where each item is ``{"input_ids": tensor of shape (seq_len+1,)}``
    """
    if name in ("wikitext-103", "wikitext-103-v1"):
        tokens = _tokenize_wikitext(name, data_dir, split, tokenizer_name)
        if cycling and split == "train":
            return CyclingTokenDataset(tokens, seq_len)
        else:
            return TokenDataset(tokens, seq_len)

    elif name == "fineweb-edu":
        shard_dir = Path(data_dir) / "fineweb_edu_shards"
        if not shard_dir.exists() or not list(shard_dir.glob("shard_*.npy")):
            raise FileNotFoundError(
                f"No FineWeb-Edu shards found in {shard_dir}. "
                f"Run prepare_fineweb_shards('{data_dir}') first."
            )
        if split == "validation":
            eval_dir = Path(data_dir) / "fineweb_edu_eval"
            eval_files = sorted(eval_dir.glob("*.npy")) if eval_dir.exists() else []
            if eval_files:
                tokens = torch.from_numpy(np.load(str(eval_files[0]))).long()
                return TokenDataset(tokens, seq_len)
            raise FileNotFoundError(
                f"No FineWeb-Edu eval data in {eval_dir}. "
                f"Place a held-out .npy shard there (tokens never seen during training)."
            )
        return ShardedTokenDataset(
            str(shard_dir), seq_len, shuffle=(split == "train"),
            rank=rank, world_size=world_size, seed=seed,
        )

    elif name == "c4":
        shard_dir = Path(data_dir) / "c4_shards"
        if split == "validation":
            # C4 validation is small; tokenize and return as TokenDataset
            from datasets import load_dataset as _load_dataset

            cache_path = Path(data_dir) / f"c4_val_{tokenizer_name.replace('/', '_')}.npy"
            if cache_path.exists():
                tokens = torch.from_numpy(np.load(str(cache_path))).long()
            else:
                print("  [data_loading] Tokenizing C4 validation split...")
                tokenizer = _get_tokenizer(tokenizer_name)
                ds = _load_dataset("allenai/c4", "en", split="validation")
                all_ids: list[int] = []
                for item in ds:
                    text = item["text"]
                    if text.strip():
                        all_ids.extend(tokenizer.encode(text))
                tokens = torch.tensor(all_ids, dtype=torch.long)
                np.save(str(cache_path), tokens.numpy())
                print(f"  [data_loading] Cached {len(tokens):,} C4 val tokens")
            return TokenDataset(tokens, seq_len)

        if not shard_dir.exists() or not list(shard_dir.glob("shard_*.npy")):
            raise FileNotFoundError(
                f"No C4 shards found in {shard_dir}. "
                f"Run prepare_c4_shards('{data_dir}') first."
            )
        return ShardedTokenDataset(
            str(shard_dir), seq_len, shuffle=(split == "train"),
            rank=rank, world_size=world_size,
        )

    else:
        raise ValueError(
            f"Unknown corpus: {name!r}. "
            f"Choose from: wikitext-103, wikitext-103-v1, fineweb-edu, c4"
        )
