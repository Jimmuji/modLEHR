"""Prepared splits on disk, and batches made from them.

A prepared split is a directory with one .npy file per array, read memory-mapped, so a split
larger than memory can still be trained on:

    subject_ids  [n_subjects]       offsets      [n_subjects + 1]  (events of subject i are offsets[i]:offsets[i+1])
    tokens ages values next_target dt loss_mask   [n_events]       (see prepare.timeline)
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import torch

from modlehr.prepare.timeline import ARRAYS

FILES = ("subject_ids", "offsets") + ARRAYS
DTYPES = {"subject_ids": np.int64, "offsets": np.int64, "tokens": np.int32, "ages": np.float32, "values": np.float32,
          "next_target": np.int32, "dt": np.float32, "loss_mask": np.bool_}


class PreparedSplit:
    def __init__(self, path: str | Path):
        path = Path(path)
        if path.suffix == ".npz":  # layout written before shard-by-shard prepare
            with np.load(path) as z:
                self.arrays = {k: z[k] for k in z.files}
        else:
            self.arrays = {k: np.load(path / f"{k}.npy", mmap_mode="r") for k in FILES}
        self.offsets = np.asarray(self.arrays["offsets"])
        self.subject_ids = np.asarray(self.arrays["subject_ids"])

    def __len__(self) -> int:
        return len(self.subject_ids)

    def lengths(self) -> np.ndarray:
        return np.diff(self.offsets)

    def subject(self, i: int, start: int = 0, stop: int | None = None) -> dict[str, np.ndarray]:
        a, b = self.offsets[i], self.offsets[i + 1]
        stop = (b - a) if stop is None else stop
        return {k: np.asarray(self.arrays[k][a + start:a + stop]) for k in ARRAYS}


def open_split(prepared_dir: str | Path, split: str) -> PreparedSplit:
    return PreparedSplit(Path(prepared_dir) / split)


class SplitWriter:
    """Appends one shard's arrays at a time, then joins them into one memory-mappable split."""

    def __init__(self, out_dir: Path):
        self.out_dir = Path(out_dir)
        self.chunk_dir = self.out_dir / "_chunks"
        if self.out_dir.exists():
            shutil.rmtree(self.out_dir)
        self.chunk_dir.mkdir(parents=True)
        self.n_chunks = 0

    def append(self, arrays: dict[str, np.ndarray]) -> None:
        if len(arrays["subject_ids"]) == 0:
            return
        for k in FILES:
            np.save(self.chunk_dir / f"{self.n_chunks:05d}_{k}.npy", arrays[k])
        self.n_chunks += 1

    def close(self) -> dict[str, int]:
        """Join the chunks into one split, subjects in ascending subject_id order.

        The order does not depend on how the MEDS data was sharded, so the same data always gives
        the same arrays, and the same seed draws the same training windows.
        """
        load = lambda i, k: np.load(self.chunk_dir / f"{i:05d}_{k}.npy", mmap_mode="r")  # noqa: E731
        chunks = range(self.n_chunks)
        sids = [np.asarray(load(i, "subject_ids")) for i in chunks]
        offs = [np.asarray(load(i, "offsets")) for i in chunks]
        subject_ids = np.concatenate(sids) if sids else np.zeros(0, dtype=np.int64)
        chunk_of = np.concatenate([np.full(len(x), i) for i, x in enumerate(sids)]) if sids else np.zeros(0, int)
        start = np.concatenate([o[:-1] for o in offs]) if offs else np.zeros(0, np.int64)
        stop = np.concatenate([o[1:] for o in offs]) if offs else np.zeros(0, np.int64)
        if len(np.unique(subject_ids)) != len(subject_ids):
            raise ValueError("a subject appears in more than one MEDS shard; MEDS keeps each subject in one shard")
        order = np.argsort(subject_ids, kind="stable")
        offsets = np.concatenate([[0], np.cumsum(stop[order] - start[order])]).astype(np.int64)

        for k in ARRAYS:
            parts = [load(i, k) for i in chunks]
            out = np.lib.format.open_memmap(self.out_dir / f"{k}.npy", mode="w+", dtype=DTYPES[k], shape=(int(offsets[-1]),))
            for j, s in enumerate(order):
                out[offsets[j]:offsets[j + 1]] = parts[chunk_of[s]][start[s]:stop[s]]
            out.flush()
            del out, parts
        np.save(self.out_dir / "subject_ids.npy", subject_ids[order].astype(np.int64))
        np.save(self.out_dir / "offsets.npy", offsets)
        shutil.rmtree(self.chunk_dir)
        return {"subjects": int(len(subject_ids)), "events": int(offsets[-1])}


def collate(rows: list[dict[str, np.ndarray]]) -> dict[str, torch.Tensor]:
    """Right-pad to the longest row. Padding has token 0 and no loss."""
    length = max(len(r["tokens"]) for r in rows)
    batch = len(rows)
    tokens = np.zeros((batch, length), dtype=np.int64)
    ages = np.zeros((batch, length), dtype=np.float32)
    values = np.zeros((batch, length), dtype=np.float32)
    has_value = np.zeros((batch, length), dtype=bool)
    next_target = np.full((batch, length), -1, dtype=np.int64)
    dt = np.zeros((batch, length), dtype=np.float32)
    loss_mask = np.zeros((batch, length), dtype=bool)
    for j, r in enumerate(rows):
        n = len(r["tokens"])
        tokens[j, :n] = r["tokens"]
        ages[j, :n] = r["ages"]
        v = r["values"]
        has_value[j, :n] = ~np.isnan(v)
        values[j, :n] = np.nan_to_num(v)
        next_target[j, :n] = r["next_target"]
        dt[j, :n] = r["dt"]
        loss_mask[j, :n] = r["loss_mask"]
    out = dict(tokens=tokens, ages=ages, values=values, has_value=has_value, next_target=next_target, dt=dt, loss_mask=loss_mask)
    return {k: torch.from_numpy(v) for k, v in out.items()}


def random_windows(split: PreparedSplit, batch_size: int, max_len: int, rng: np.random.Generator) -> dict[str, torch.Tensor]:
    """A training batch: random subjects, each cropped to a random window of at most max_len events."""
    lengths = split.lengths()
    rows = []
    for i in rng.integers(0, len(split), batch_size):
        n = lengths[i]
        start = int(rng.integers(0, n - max_len + 1)) if n > max_len else 0
        rows.append(split.subject(int(i), start, start + min(n, max_len)))
    return collate(rows)


def fixed_windows(split: PreparedSplit, batch_size: int, max_len: int, n_batches: int, seed: int = 0) -> list[dict[str, torch.Tensor]]:
    """The same windows every time, for comparable tuning losses during training."""
    rng = np.random.default_rng(seed)
    return [random_windows(split, batch_size, max_len, rng) for _ in range(n_batches)]
