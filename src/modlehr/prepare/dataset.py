"""Prepared splits on disk, and batches made from them.

A prepared split is a directory with one .npy file per array, read memory-mapped, so a split
larger than memory can still be trained on:

    subject_ids  [n_subjects]       offsets      [n_subjects + 1]  (events of subject i are offsets[i]:offsets[i+1])
    follow_up_start follow_up_end [n_subjects]
    tokens ages values target next_target dt loss_mask   [n_events]   (see prepare.timeline)

Splits written before target and the follow-up arrays existed still load: they get one label per
time and follow-up from the first event to the last.

Splits of data with text events also hold text_ref, note_offsets and note_words (see prepare.text).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import torch

from modlehr.prepare.text import TEXT_FILES
from modlehr.prepare.timeline import ARRAYS, SUBJECT_ARRAYS, next_target_sets

FILES = ("subject_ids", "offsets") + ARRAYS + SUBJECT_ARRAYS
DTYPES = {"subject_ids": np.int64, "offsets": np.int64, "tokens": np.int32, "ages": np.float32, "values": np.float32,
          "target": np.int32, "next_target": np.int32, "dt": np.float32, "loss_mask": np.bool_,
          "follow_up_start": np.float32, "follow_up_end": np.float32}


class PreparedSplit:
    def __init__(self, path: str | Path):
        path = Path(path)
        if path.suffix == ".npz":  # layout written before shard-by-shard prepare
            with np.load(path) as z:
                self.arrays = {k: z[k] for k in z.files}
        else:
            names = FILES + (TEXT_FILES if (path / "text_ref.npy").exists() else ())
            self.arrays = {k: np.load(path / f"{k}.npy", mmap_mode="r") for k in names if (path / f"{k}.npy").exists()}
        self.offsets = np.asarray(self.arrays["offsets"])
        self.subject_ids = np.asarray(self.arrays["subject_ids"])
        if "follow_up_end" in self.arrays:
            self.follow_up_end = np.asarray(self.arrays["follow_up_end"])
        else:  # written before follow-up was tracked: it ran to the last event
            self.follow_up_end = np.asarray(self.arrays["ages"])[self.offsets[1:] - 1] if len(self.offsets) > 1 \
                else np.zeros(0, np.float32)
        if "follow_up_start" in self.arrays:
            self.follow_up_start = np.asarray(self.arrays["follow_up_start"])
        else:  # written before follow-up could start late: from the first event
            self.follow_up_start = np.asarray(self.arrays["ages"])[self.offsets[:-1]] if len(self.offsets) > 1 \
                else np.zeros(0, np.float32)
        self.has_text = "text_ref" in self.arrays
        if self.has_text:
            self.note_offsets = np.asarray(self.arrays["note_offsets"])

    def __len__(self) -> int:
        return len(self.subject_ids)

    def lengths(self) -> np.ndarray:
        return np.diff(self.offsets)

    def subject(self, i: int, start: int = 0, stop: int | None = None, labels: bool = True) -> dict:
        """Events start:stop of subject i. row["next_set"] [n, K] holds every target of the next target
        time (when the split has `target` and labels is true; a model's forward pass does not need it).
        With text, row["text"] lists each event's note words (or None)."""
        a, b = self.offsets[i], self.offsets[i + 1]
        stop = (b - a) if stop is None else stop
        row = {k: np.asarray(self.arrays[k][a + start:a + stop]) for k in ARRAYS if k in self.arrays}
        if labels and "target" in self.arrays:
            row["next_set"] = next_target_sets(np.asarray(self.arrays["ages"][a:b]), np.asarray(self.arrays["target"][a:b]),
                                               float(self.follow_up_end[i]), start, stop)
        if self.has_text:
            refs = np.asarray(self.arrays["text_ref"][a + start:a + stop])
            words = self.arrays["note_words"]
            row["text"] = [None if r < 0 else np.asarray(words[self.note_offsets[r]:self.note_offsets[r + 1]])
                           for r in refs]
        return row


def open_split(prepared_dir: str | Path, split: str) -> PreparedSplit:
    return PreparedSplit(Path(prepared_dir) / split)


class SplitWriter:
    """Appends one shard's arrays at a time, then joins them into one memory-mappable split."""

    def __init__(self, out_dir: Path, with_text: bool = False):
        self.out_dir = Path(out_dir)
        self.chunk_dir = self.out_dir / "_chunks"
        self.with_text = with_text
        if self.out_dir.exists():
            shutil.rmtree(self.out_dir)
        self.chunk_dir.mkdir(parents=True)
        self.n_chunks = 0

    def append(self, arrays: dict[str, np.ndarray]) -> None:
        if len(arrays["subject_ids"]) == 0:
            return
        names = FILES + (TEXT_FILES if self.with_text else ())
        for k in names:
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

        for k in SUBJECT_ARRAYS:
            per_subject = np.concatenate([np.asarray(load(i, k)) for i in chunks]) if sids else np.zeros(0)
            np.save(self.out_dir / f"{k}.npy", per_subject[order].astype(DTYPES[k]))
        for k in ARRAYS:
            parts = [load(i, k) for i in chunks]
            out = np.lib.format.open_memmap(self.out_dir / f"{k}.npy", mode="w+", dtype=DTYPES[k], shape=(int(offsets[-1]),))
            for j, s in enumerate(order):
                out[offsets[j]:offsets[j + 1]] = parts[chunk_of[s]][start[s]:stop[s]]
            out.flush()
            del out, parts
        if self.with_text:
            self._join_text(load, order, chunk_of, start, stop, int(offsets[-1]))
        np.save(self.out_dir / "subject_ids.npy", subject_ids[order].astype(np.int64))
        np.save(self.out_dir / "offsets.npy", offsets)
        shutil.rmtree(self.chunk_dir)
        return {"subjects": int(len(subject_ids)), "events": int(offsets[-1])}


    def _join_text(self, load, order, chunk_of, start, stop, n_events: int) -> None:
        """Renumber notes in the joined event order and copy their words."""
        refs = [np.asarray(load(i, "text_ref")) for i in range(self.n_chunks)]
        note_offs = [np.asarray(load(i, "note_offsets")) for i in range(self.n_chunks)]
        words = [load(i, "note_words") for i in range(self.n_chunks)]
        n_words = sum(len(w) for w in words)
        text_ref = np.lib.format.open_memmap(self.out_dir / "text_ref.npy", mode="w+", dtype=np.int32, shape=(n_events,))
        note_words = np.lib.format.open_memmap(self.out_dir / "note_words.npy", mode="w+", dtype=np.int32,
                                               shape=(n_words,))
        new_offsets, pos, event, note = [0], 0, 0, 0
        for s in order:
            c = chunk_of[s]
            local = refs[c][start[s]:stop[s]]
            out = np.full(len(local), -1, dtype=np.int32)
            for j in np.flatnonzero(local >= 0):
                a, b = note_offs[c][local[j]], note_offs[c][local[j] + 1]
                note_words[pos:pos + (b - a)] = words[c][a:b]
                pos += b - a
                new_offsets.append(pos)
                out[j] = note
                note += 1
            text_ref[event:event + len(local)] = out
            event += len(local)
        text_ref.flush()
        note_words.flush()
        del text_ref, note_words
        np.save(self.out_dir / "note_offsets.npy", np.asarray(new_offsets, dtype=np.int64))


PAD_TO = 64


def collate(rows: list[dict]) -> dict[str, torch.Tensor]:
    """Right-pad to the longest row, rounded up to a multiple of PAD_TO. Padding has token 0 and no loss.

    Rounding keeps the number of distinct batch shapes small. On an Apple GPU some operations keep a
    compiled graph for every shape they meet, and the mixture-of-experts model, whose router uses
    them, grew past the machine's memory when every batch had its own length. Padding comes after
    a row's events, which attend only backwards, so it changes no prediction.
    """
    length = -(-max(len(r["tokens"]) for r in rows) // PAD_TO) * PAD_TO
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
    if all("next_set" in r for r in rows):  # every target of the next target time, -1 padded
        k = max(r["next_set"].shape[1] for r in rows)
        next_set = np.full((batch, length, k), -1, dtype=np.int64)
        for j, r in enumerate(rows):
            next_set[j, :len(r["next_set"]), :r["next_set"].shape[1]] = r["next_set"]
        out["next_set"] = next_set
    if any("text" in r for r in rows):
        # flat word ids, where each text event's words start, and that event's position in [batch * length]
        ids, starts, pos = [], [], []
        n = 0
        for j, r in enumerate(rows):
            for i, words in enumerate(r.get("text") or []):
                if words is not None:
                    starts.append(n), pos.append(j * length + i), ids.append(words)
                    n += len(words)
        out["text_ids"] = np.concatenate(ids).astype(np.int64) if ids else np.zeros(0, dtype=np.int64)
        out["text_offsets"] = np.asarray(starts, dtype=np.int64)
        out["text_pos"] = np.asarray(pos, dtype=np.int64)
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
