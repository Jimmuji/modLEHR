"""Numeric values: scaled per measurement for the value encoding, or cut into bins.

Statistics come from the training split only, so nothing leaks from tuning or held-out data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CLIP = 5.0


def fit_scaler(tokens: pd.Series, values: pd.Series) -> dict[str, tuple[float, float]]:
    """token → (mean, std) over training values."""
    frame = pd.DataFrame({"token": tokens, "value": values}).dropna()
    stats = frame.groupby("token")["value"].agg(["mean", "std"]).fillna({"std": 1.0})
    stats.loc[stats["std"] < 1e-6, "std"] = 1.0
    return {t: (float(m), float(s)) for t, m, s in stats.itertuples()}


def scale(tokens: pd.Series, values: pd.Series, stats: dict[str, tuple[float, float]]) -> np.ndarray:
    """(value - mean) / std, clipped to ±5; NaN where there is no value or no statistics."""
    mean = tokens.map({t: m for t, (m, _) in stats.items()})
    std = tokens.map({t: s for t, (_, s) in stats.items()})
    z = ((values - mean) / std).clip(-CLIP, CLIP)
    return z.to_numpy(dtype=np.float32)


def fit_bins(codes: pd.Series, values: pd.Series, n_bins: int) -> dict[str, list[float]]:
    """code → inner quantile cut points over training values (n_bins - 1 of them)."""
    frame = pd.DataFrame({"code": codes, "value": values}).dropna()
    qs = np.linspace(0, 1, n_bins + 1)[1:-1]
    return {c: [float(x) for x in g["value"].quantile(qs)] for c, g in frame.groupby("code")}


def apply_bins(codes: pd.Series, values: pd.Series, cuts: dict[str, list[float]]) -> pd.Series:
    """code//Q1 … code//Qn; the bare code where the value or the cut points are missing."""
    out = codes.astype(str).to_numpy(dtype=object, copy=True)
    # one pass over the rows grouped by code, not one pass per code: unthinned shards have millions of rows
    v = values.to_numpy(dtype=np.float64)
    pos = np.flatnonzero(~np.isnan(v) & codes.isin(list(cuts)).to_numpy())
    group, names = pd.factorize(out[pos])
    order = np.argsort(group, kind="stable")
    bounds = np.searchsorted(group[order], np.arange(len(names) + 1))
    for g, code in enumerate(names):
        rows = pos[order[bounds[g]:bounds[g + 1]]]
        edges = np.asarray(cuts[code])
        b = np.searchsorted(edges, v[rows], side="left")
        out[rows] = np.array([f"{code}//Q{i}" for i in range(1, len(edges) + 2)], dtype=object)[b]
    return pd.Series(out, index=codes.index)
