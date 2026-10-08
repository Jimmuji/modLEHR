"""Per-subject timelines as flat arrays, with the training labels attached to every event.

For each event i at age t_i the label is the next target event strictly later in time: its
target id and the years until it. If no target follows, the event is censored at the end of
the record (next_target = -1, dt = years until the last event). Only the last event of each
distinct time carries a loss (loss_mask), so a prediction is always made after everything
recorded at that moment has been seen.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from modlehr.prepare.vocabulary import Vocabulary

ARRAYS = ("tokens", "ages", "values", "next_target", "dt", "loss_mask")


def build_split(frame: pd.DataFrame, vocab: Vocabulary) -> dict[str, np.ndarray]:
    """frame: one split, sorted by subject_id then age, with columns subject_id, token_id,
    age, value."""
    subject = frame["subject_id"].to_numpy()
    subjects, starts = np.unique(subject, return_index=True)
    offsets = np.append(starts, len(frame)).astype(np.int64)
    tokens = frame["token_id"].to_numpy(np.int32)
    ages = frame["age"].to_numpy(np.float32)
    values = frame["value"].to_numpy(np.float32)
    target_of = vocab.input_to_target[tokens]

    n = len(frame)
    next_target = np.full(n, -1, dtype=np.int32)
    dt = np.zeros(n, dtype=np.float32)
    loss_mask = np.zeros(n, dtype=bool)
    for s in range(len(subjects)):
        a, b = offsets[s], offsets[s + 1]
        age, tgt = ages[a:b], target_of[a:b]
        end = age[-1]
        idx = np.flatnonzero(tgt >= 0)
        if len(idx):
            t_age = age[idx]
            pos = np.searchsorted(t_age, age, side="right")
            has = pos < len(idx)
            p = np.minimum(pos, len(idx) - 1)
            next_target[a:b] = np.where(has, tgt[idx][p], -1)
            dt[a:b] = np.where(has, t_age[p] - age, end - age)
        else:
            dt[a:b] = end - age
        last_of_time = np.ones(b - a, dtype=bool)
        last_of_time[:-1] = age[1:] != age[:-1]
        loss_mask[a:b] = last_of_time
    return {"subject_ids": subjects.astype(np.int64), "offsets": offsets, "tokens": tokens, "ages": ages,
            "values": values, "next_target": next_target, "dt": dt, "loss_mask": loss_mask}


def first_target_ages(tokens: np.ndarray, ages: np.ndarray, offsets: np.ndarray, vocab: Vocabulary) -> np.ndarray:
    """[n_subjects, n_targets] age at the first occurrence of each target, inf if never."""
    out = np.full((len(offsets) - 1, len(vocab.targets)), np.inf, dtype=np.float32)
    target_of = vocab.input_to_target[tokens]
    for s in range(len(offsets) - 1):
        a, b = offsets[s], offsets[s + 1]
        t = target_of[a:b]
        rows = t >= 0
        np.minimum.at(out[s], t[rows], ages[a:b][rows])
    return out
