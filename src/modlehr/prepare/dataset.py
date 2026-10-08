"""Load a prepared split and turn timelines into padded batches."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from modlehr.prepare.timeline import ARRAYS


class PreparedSplit:
    def __init__(self, path: str | Path):
        with np.load(path) as z:
            self.arrays = {k: z[k] for k in z.files}
        self.offsets = self.arrays["offsets"]
        self.subject_ids = self.arrays["subject_ids"]

    def __len__(self) -> int:
        return len(self.subject_ids)

    def lengths(self) -> np.ndarray:
        return np.diff(self.offsets)

    def subject(self, i: int, start: int = 0, stop: int | None = None) -> dict[str, np.ndarray]:
        a, b = self.offsets[i], self.offsets[i + 1]
        stop = (b - a) if stop is None else stop
        return {k: self.arrays[k][a + start:a + stop] for k in ARRAYS}


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
