"""Read modLEHR's prepared data with numpy alone, for model code of your own.

Copy this file next to your code; it needs numpy and nothing from modLEHR. docs/reading_prepared_data.md says what
each array means.

    from prepared_data import PreparedData
    data = PreparedData("runs/prepared/mimic_iv_10pct")
    train = data.split("train")
    batch = train.windows(batch_size=32, max_len=1024, rng=np.random.default_rng(0))   # next-token training
    task = data.task("admission_icu", "train")                                         # a binary task
    inputs = data.task_inputs(task, "train", max_len=1024, rows=range(32))              # its inputs, padded

Run it to check a copy of the data (prints counts only, never a patient's events):

    python examples/prepared_data.py runs/prepared/mimic_iv_10pct
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

EVENT_ARRAYS = ("tokens", "ages", "values", "target", "next_target", "dt", "loss_mask")  # one entry per event
SUBJECT_ARRAYS = ("subject_ids", "follow_up_start", "follow_up_end")                    # one entry per subject


class Split:
    """One split (train, tuning or held_out). The events of subject i are rows offsets[i]:offsets[i + 1] of every
    event array, in time order. Arrays are memory-mapped, so opening a split reads nothing yet."""

    def __init__(self, path: str | Path):
        path = Path(path)
        self.arrays = {p.stem: np.load(p, mmap_mode="r") for p in path.glob("*.npy")}
        self.offsets = np.asarray(self.arrays["offsets"])
        self.subject_ids = np.asarray(self.arrays["subject_ids"])

    def __len__(self) -> int:
        return len(self.subject_ids)

    def lengths(self) -> np.ndarray:
        return np.diff(self.offsets)

    def events(self, i: int, start: int = 0, stop: int | None = None) -> dict[str, np.ndarray]:
        """Events start:stop of subject i (all of them by default), one array per name in EVENT_ARRAYS."""
        a, b = int(self.offsets[i]), int(self.offsets[i + 1])
        stop = b - a if stop is None else stop
        return {k: np.asarray(self.arrays[k][a + start:a + stop]) for k in EVENT_ARRAYS if k in self.arrays}

    def windows(self, batch_size: int, max_len: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        """A training batch as modLEHR draws it: random subjects, each cut to a random window of at most max_len
        events, right-padded with token 0. For next-token training, predict tokens[:, 1:] from tokens[:, :-1]
        and skip positions where the next token is 0."""
        lengths = self.lengths()
        rows = []
        for i in rng.integers(0, len(self), batch_size):
            n = int(lengths[i])
            start = int(rng.integers(0, n - max_len + 1)) if n > max_len else 0
            rows.append(self.events(int(i), start, start + min(n, max_len)))
        return pad(rows)


def pad(rows: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Right-pad rows of events to the longest one: tokens 0, ages 0, values NaN, and a boolean `mask` of real events."""
    length = max(1, max(len(r["tokens"]) for r in rows))
    out = {"tokens": np.zeros((len(rows), length), np.int64), "ages": np.zeros((len(rows), length), np.float32),
           "values": np.full((len(rows), length), np.nan, np.float32), "mask": np.zeros((len(rows), length), bool)}
    for j, r in enumerate(rows):
        n = len(r["tokens"])
        out["tokens"][j, :n], out["ages"][j, :n], out["values"][j, :n] = r["tokens"], r["ages"], r["values"]
        out["mask"][j, :n] = True
    return out


class PreparedData:
    def __init__(self, prepared_dir: str | Path):
        self.dir = Path(prepared_dir)
        vocab = json.loads((self.dir / "vocab.json").read_text())
        self.tokens: list[str] = vocab["tokens"]    # token id -> name; 0 is padding
        self.targets: list[str] = vocab["targets"]  # target id -> name (the `target` and `next_target` arrays)
        self.meta = json.loads((self.dir / "prepare.json").read_text())
        self._splits: dict[str, Split] = {}

    def split(self, name: str) -> Split:
        if name not in self._splits:
            self._splits[name] = Split(self.dir / name)
        return self._splits[name]

    def tasks(self) -> list[str]:
        return sorted(p.name for p in (self.dir / "tasks").iterdir()) if (self.dir / "tasks").is_dir() else []

    def task(self, name: str, split: str) -> dict[str, np.ndarray]:
        """The samples of one binary task in one split, one entry per sample: subject_id, prediction_time, label,
        index (the subject's position in the split, -1 if it has no events) and n (how many of its events come at or
        before the prediction time, so the input is events 0:n)."""
        with np.load(self.dir / "tasks" / name / f"{split}.npz") as z:
            return {k: z[k] for k in ("subject_id", "prediction_time", "label", "index", "n")}

    def task_inputs(self, task: dict[str, np.ndarray], split: str, max_len: int, rows) -> dict[str, np.ndarray]:
        """Padded inputs of the samples `rows` of a task: each sample's last max_len events up to its prediction
        time, and its label. A model reads the row and predicts the label from its last real event. A sample with
        n == 0 has no events by then and comes as an empty row (mask all False); modLEHR gives those the
        training prevalence."""
        s, rows = self.split(split), list(rows)
        empty = {k: np.zeros(0, np.float32) for k in ("tokens", "ages", "values")}
        out = []
        for r in rows:
            i, n = int(task["index"][r]), int(task["n"][r])
            out.append(s.events(i, max(0, n - max_len), n) if i >= 0 and n > 0 else empty)
        batch = pad(out)
        batch["label"] = task["label"][rows]
        return batch


def main(prepared_dir: str) -> None:
    data = PreparedData(prepared_dir)
    subset = data.meta.get("subset")
    print(f"{prepared_dir}: {len(data.tokens):,} tokens (0 is padding), {len(data.targets):,} targets"
          + (f"; {subset['fraction']:g} of the subjects of {subset['of']}" if subset else ""))
    for name in ("train", "tuning", "held_out"):
        s = data.split(name)
        lengths = s.lengths()
        print(f"  {name:9} {len(s):>9,} subjects {int(lengths.sum()):>12,} events  "
              f"(per subject: median {int(np.median(lengths)):,}, 90th percentile {int(np.percentile(lengths, 90)):,})")
    batch = data.split("train").windows(8, 1024, np.random.default_rng(0))
    print(f"  a batch of 8 training windows: tokens {batch['tokens'].shape}, {int(batch['mask'].sum()):,} real events")
    for t in data.tasks():
        counts = []
        for name in ("train", "tuning", "held_out"):
            if (data.dir / "tasks" / t / f"{name}.npz").exists():
                task = data.task(t, name)
                counts.append(f"{name} {len(task['label']):,} ({int(task['label'].sum()):,} positive)")
        print(f"  task {t}: " + ", ".join(counts))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "runs/prepared/mimic_iv_10pct")
