"""The landmark prediction task, shared by the model evaluation and the classical baselines.

For a landmark age L, a horizon H and a target k, among subjects whose record covers L (first
event at or before L, last event after L):

    left out   subjects who already had k at or before L
    case       the first k falls in (L, L + H]

Who counts as a control depends on `evaluate.controls`:

    "at_risk" (default)   no k by L + H, and the record reaches L + H; subjects whose record
                          ends before L + H without k are left out as censored. A control here
                          may still develop k later, which is the clinically useful question
                          and the harder one.

    "never"               no k anywhere in the record, censored or not. This is the Delphi-2M
                          convention, where controls are people who never have the disease.
                          Easier to separate, since never-havers differ systematically from
                          cases, so the AUCs run higher and are not comparable with "at_risk".

Inputs are every event at or before L.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from modlehr.prepare.dataset import PreparedSplit
from modlehr.prepare.timeline import first_target_ages
from modlehr.prepare.vocabulary import Vocabulary


CONTROLS = ("at_risk", "never")


@dataclass
class LandmarkTask:
    landmark: float
    horizon: float
    subjects: np.ndarray  # indices into the split of the subjects in the record at the landmark
    first: np.ndarray     # [n, n_targets] age at the first occurrence of each target, inf if never
    end: np.ndarray       # [n] age at the last event
    controls: str = "at_risk"

    def labels(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        """(keep, case) for target k: keep marks the subjects in the task, case the cases among all."""
        f = self.first[:, k]
        lo, hi = self.landmark, self.landmark + self.horizon
        case = (f > lo) & (f <= hi)
        if self.controls == "never":
            keep = (f > lo) & (case | np.isinf(f))
        else:
            keep = (f > lo) & (case | (self.end >= hi))
        return keep, case


def landmark_task(split: PreparedSplit, vocab: Vocabulary, landmark: float, horizon: float,
                  first: np.ndarray | None = None, controls: str = "at_risk") -> LandmarkTask:
    """The task at one landmark. Pass `first` (from first_target_ages) to reuse it across landmarks."""
    if controls not in CONTROLS:
        raise ValueError(f"evaluate.controls must be one of {CONTROLS}, got {controls!r}")
    ages, offsets = split.arrays["ages"], split.offsets
    start, end = np.asarray(ages)[offsets[:-1]], split.follow_up_end  # end of follow-up, not of the record
    subjects = np.flatnonzero((start <= landmark) & (end > landmark))
    if first is None:
        first = split_first_ages(split, vocab)
    return LandmarkTask(landmark, horizon, subjects, first[subjects], end[subjects], controls)


def split_first_ages(split: PreparedSplit, vocab: Vocabulary) -> np.ndarray:
    """first_target_ages of a split, counting only targets within each subject's follow-up."""
    return first_target_ages(split.arrays["tokens"], split.arrays["ages"], split.offsets, vocab, split.follow_up_end)


def history_before(split: PreparedSplit, i: int, landmark: float, max_len: int | None = None) -> dict[str, np.ndarray]:
    """Subject i's events at or before the landmark; the last max_len of them if max_len is set. Inputs
    only: no label sets."""
    a, b = split.offsets[i], split.offsets[i + 1]
    n = int(np.searchsorted(split.arrays["ages"][a:b], landmark, side="right"))
    start = 0 if max_len is None else max(0, n - max_len)
    return split.subject(int(i), start, n, labels=False)


def summarize(result: pd.DataFrame) -> dict:
    """Mean AUC over (target, landmark) pairs with an approximate 95% interval.

    The interval comes from the spread of the pair AUCs (normal approximation). Pairs share
    subjects, so they are not independent; read the interval as a guide, not a test.
    """
    if not len(result):
        return {"pairs": 0, "mean_auc": None, "mean_auc_ci95": None, "median_auc": None}
    auc = result["auc"]
    half = 1.96 * auc.std(ddof=1) / math.sqrt(len(auc)) if len(auc) > 1 else float("nan")
    return {"pairs": int(len(auc)), "mean_auc": float(auc.mean()),
            "mean_auc_ci95": [float(auc.mean() - half), float(auc.mean() + half)],
            "median_auc": float(auc.median())}
