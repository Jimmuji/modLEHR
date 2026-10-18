"""`modlehr compare`: mean AUC of several runs, and the paired difference against the first.

    modlehr compare runs/synthea_gpt_s1 runs/synthea_gpt_s2 runs/synthea_gpt_rope_age_s1 runs/synthea_gpt_logistic

Run directories in the same parent directory whose names differ only by a seed suffix (_s1, _s2,
...) form one group: each (target, landmark) AUC is averaged over the group's seeds before anything
is compared. Runs of the same name in different directories are different groups, shown by their
path. The first group is the reference. For every other group, the difference is taken pair by pair
over the pairs both groups scored, with two 95% intervals:

    over pairs   from the spread of the pair differences (normal approximation). It leaves out how
                 much a whole run moves with its seed, and pairs share subjects, so it is too narrow
                 for deciding whether one model beats another.
    over seeds   from the runs' mean AUCs (over the shared pairs), with a t interval: seed by seed
                 when both groups have the same seeds, one-sample when one group is a single
                 deterministic run (a baseline). Wide with few seeds, but it is the one to read.

Held-out negative log-likelihood is shown when the runs recorded it (lower is better).
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pandas as pd
from scipy import stats

SEED_SUFFIX = re.compile(r"_s\d+$")


def group_runs(paths: list[str | Path]) -> dict[str, list[Path]]:
    """Run name without its seed suffix → run directories, in the order given. Directories of the same
    name under different parents are different runs; those groups are named by their path."""
    keyed: dict[Path, list[Path]] = {}
    for p in map(Path, paths):
        keyed.setdefault(p.parent / SEED_SUFFIX.sub("", p.name), []).append(p)
    names = [key.name for key in keyed]
    return {(key.name if names.count(key.name) == 1 else str(key)): dirs for key, dirs in keyed.items()}


def load_group(dirs: list[Path]) -> tuple[pd.Series, float | None]:
    """(AUC per (target, landmark) averaged over seeds, mean held-out NLL or None)."""
    tables, nlls = [], []
    for d in dirs:
        csv = d / "evaluate.csv"
        if not csv.exists():
            raise FileNotFoundError(f"{csv} not found; run `modlehr evaluate` (or `modlehr baseline`) first")
        tables.append(pd.read_csv(csv).set_index(["target", "landmark"])["auc"])
        summary = d / "evaluate.json"
        if summary.exists():
            nll = json.loads(summary.read_text()).get("likelihood", {}).get("nll")
            if nll is not None:
                nlls.append(nll)
    auc = pd.concat(tables, axis=1).mean(axis=1)
    return auc, (sum(nlls) / len(nlls) if len(nlls) == len(dirs) else None)


def seed_interval(ref_dirs: list[Path], dirs: list[Path], pairs: pd.Index) -> tuple[float, float, int] | None:
    """95% t interval of the difference in mean AUC over `pairs`, from the runs' own means."""
    mean = lambda d: pd.read_csv(d / "evaluate.csv").set_index(["target", "landmark"])["auc"].reindex(pairs).mean()  # noqa: E731
    seed = lambda d: SEED_SUFFIX.search(d.name).group(0) if SEED_SUFFIX.search(d.name) else None  # noqa: E731
    a, b = {seed(d): mean(d) for d in ref_dirs}, {seed(d): mean(d) for d in dirs}
    if len(a) > 1 and set(a) == set(b):
        diffs = [b[k] - a[k] for k in a]  # same seed, same training windows
    elif len(a) == 1 and len(b) > 1:
        diffs = [v - next(iter(a.values())) for v in b.values()]
    elif len(b) == 1 and len(a) > 1:
        diffs = [next(iter(b.values())) - v for v in a.values()]
    else:
        return None
    n = len(diffs)
    m = sum(diffs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in diffs) / (n - 1))
    half = stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n)
    return m - half, m + half, n


def compare(paths: list[str | Path]) -> pd.DataFrame:
    groups = group_runs(paths)
    loaded = {name: load_group(dirs) for name, dirs in groups.items()}
    ref_name = next(iter(loaded))
    ref = loaded[ref_name][0]
    rows = []
    for name, (auc, nll) in loaded.items():
        row = {"run": name, "seeds": len(groups[name]), "pairs": len(auc), "mean_auc": auc.mean(), "nll": nll,
               "diff": None, "ci_low": None, "ci_high": None, "shared_pairs": None,
               "seed_ci_low": None, "seed_ci_high": None, "seed_n": None}
        if name != ref_name:
            both = pd.concat([ref, auc], axis=1, join="inner").dropna()
            d = both.iloc[:, 1] - both.iloc[:, 0]
            half = 1.96 * d.std(ddof=1) / math.sqrt(len(d)) if len(d) > 1 else float("nan")
            row.update(diff=d.mean(), ci_low=d.mean() - half, ci_high=d.mean() + half, shared_pairs=len(d))
            seeds = seed_interval(groups[ref_name], groups[name], both.index)
            if seeds:
                row.update(seed_ci_low=seeds[0], seed_ci_high=seeds[1], seed_n=seeds[2])
        rows.append(row)
    return pd.DataFrame(rows)


def print_comparison(table: pd.DataFrame, log=print) -> None:
    ref = table.iloc[0]["run"]
    log(f"{'run':30} {'seeds':>5} {'pairs':>5} {'mean AUC':>9} {'held-out NLL':>13}   difference from {ref}, "
        "95% CI over pairs; over seeds")
    for r in table.itertuples():
        nll = f"{r.nll:13.4f}" if r.nll is not None and not pd.isna(r.nll) else f"{'':13}"
        diff = (f"   {r.diff:+.4f} ({r.ci_low:+.4f} to {r.ci_high:+.4f}) over {int(r.shared_pairs)} pairs"
                if r.diff is not None and not pd.isna(r.diff) else "")
        if diff and r.seed_n is not None and not pd.isna(r.seed_n):
            diff += f"; ({r.seed_ci_low:+.4f} to {r.seed_ci_high:+.4f}) over {int(r.seed_n)} seeds"
        log(f"{r.run:30} {r.seeds:5d} {r.pairs:5d} {r.mean_auc:9.3f} {nll}{diff}")
