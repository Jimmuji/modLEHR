"""Per-subject timelines as flat arrays, with the training labels attached to every event.

For each event i at age t_i the label is the next target time strictly later: the targets recorded
then and the years until it. Targets recorded at the same time are all labels of that time: in
MIMIC-IV every diagnosis of an admission is timed at its discharge, so nearly all first diagnoses
share their time with others. `next_target` keeps the first of them (for code that wants one) and
`PreparedSplit.subject` rebuilds the full set from `target`. If no target follows, the event is
censored at the end of follow-up (next_target = -1, dt = years until then).

Follow-up starts at the subject's first event, or with data.follow_up_from: first_contact at the first
event of a follow_up rule that is not static; events before it carry no loss, since nothing could have
been recorded then. It ends at the subject's last event of a rule with follow_up: true (all rules by default),
and no later than death, counting events that prepare drops later (thinned, repeated or rare ones),
since they still show the subject was followed then. Events after it (an outpatient measurement after the last admission, say)
stay in the input but carry no loss, since no outcome could have been recorded then. Only the last
event of each distinct time carries a loss (loss_mask), so a prediction is always made after
everything recorded at that moment has been seen.
"""

from __future__ import annotations

import meds
import numpy as np
import pandas as pd

from modlehr.prepare.vocabulary import Vocabulary

ARRAYS = ("tokens", "ages", "values", "target", "next_target", "dt", "loss_mask")  # one per event
SUBJECT_ARRAYS = ("follow_up_start", "follow_up_end")  # one per subject


def follow_up_ends(ages: np.ndarray, follows: np.ndarray, target: np.ndarray, offsets: np.ndarray,
                   death: int | None) -> np.ndarray:
    """Per subject: the age of the last follow-up event, capped at death."""
    end = np.zeros(len(offsets) - 1, dtype=np.float32)
    for s in range(len(end)):
        a, b = offsets[s], offsets[s + 1]
        f = follows[a:b]
        end[s] = ages[a:b][f].max() if f.any() else ages[b - 1]
        if death is not None:
            d = np.flatnonzero(target[a:b] == death)
            if len(d):
                end[s] = min(end[s], ages[a + d[0]])
    return end


def build_split(frame: pd.DataFrame, vocab: Vocabulary) -> dict[str, np.ndarray]:
    """frame: one split, sorted by subject_id then age, with columns subject_id, token_id, age, value,
    and optionally follow_up_start and follow_up_end (per event, its subject's; prepare computes them
    before dropping any event) or else follow_up (bool, default true), from which the end is computed
    here, with follow-up from the first event."""
    subject = frame["subject_id"].to_numpy()
    subjects, starts = np.unique(subject, return_index=True)
    offsets = np.append(starts, len(frame)).astype(np.int64)
    tokens = frame["token_id"].to_numpy(np.int32)
    ages = frame["age"].to_numpy(np.float32)
    values = frame["value"].to_numpy(np.float32)
    target_of = vocab.input_to_target[tokens].astype(np.int32)
    follows = frame["follow_up"].to_numpy(bool) if "follow_up" in frame else np.ones(len(frame), bool)
    death = vocab.targets.index(meds.death_code) if meds.death_code in vocab.targets else None
    if "follow_up_end" in frame:
        end_of = frame["follow_up_end"].to_numpy(np.float32)[starts]
    else:
        end_of = follow_up_ends(ages, follows, target_of, offsets, death)
    start_of = frame["follow_up_start"].to_numpy(np.float32)[starts] if "follow_up_start" in frame else ages[starts]

    n = len(frame)
    next_target = np.full(n, -1, dtype=np.int32)
    dt = np.zeros(n, dtype=np.float32)
    loss_mask = np.zeros(n, dtype=bool)
    for s in range(len(subjects)):
        a, b = offsets[s], offsets[s + 1]
        age, tgt, end, begin = ages[a:b], target_of[a:b], end_of[s], start_of[s]
        idx = np.flatnonzero((tgt >= 0) & (age <= end))
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
        within = age <= end
        loss_mask[a:b] = last_of_time & within & (age >= begin)
        next_target[a:b][~within] = -1
        dt[a:b][~within] = 0.0
    return {"subject_ids": subjects.astype(np.int64), "offsets": offsets, "tokens": tokens, "ages": ages,
            "values": values, "target": target_of, "next_target": next_target, "dt": dt, "loss_mask": loss_mask,
            "follow_up_start": start_of, "follow_up_end": end_of}


def next_target_sets(ages: np.ndarray, target: np.ndarray, end: float, start: int, stop: int) -> np.ndarray:
    """[stop - start, K]: for each of events start:stop of one subject, every target recorded at the
    next target time within follow-up (-1 padding; all -1 when censored)."""
    m = (target >= 0) & (ages <= end)
    t_age, t_id = ages[m], target[m]
    if not len(t_age):
        return np.full((stop - start, 1), -1, dtype=np.int32)
    times, first, counts = np.unique(t_age, return_index=True, return_counts=True)
    groups = np.full((len(times) + 1, counts.max()), -1, dtype=np.int32)  # last row: censored
    for g, (f, c) in enumerate(zip(first, counts)):
        groups[g, :c] = t_id[f:f + c]
    g = np.searchsorted(times, ages[start:stop], side="right")
    return groups[g]


def first_target_ages(tokens: np.ndarray, ages: np.ndarray, offsets: np.ndarray, vocab: Vocabulary,
                      end: np.ndarray | None = None) -> np.ndarray:
    """[n_subjects, n_targets] age at the first occurrence of each target within follow-up, inf if never."""
    out = np.full((len(offsets) - 1, len(vocab.targets)), np.inf, dtype=np.float32)
    target_of = vocab.input_to_target[np.asarray(tokens)]
    ages = np.asarray(ages)
    subject = np.repeat(np.arange(len(offsets) - 1, dtype=np.int32), np.diff(offsets))
    rows = target_of >= 0
    if end is not None:
        rows &= ages <= np.asarray(end)[subject]
    np.minimum.at(out, (subject[rows], target_of[rows]), ages[rows])
    return out
