"""`modlehr compare`: mean AUC of several runs, and the paired difference against the first.

    modlehr compare runs/synthea_gpt_s1 runs/synthea_gpt_s2 runs/synthea_gpt_rope_age_s1 runs/synthea_gpt_logistic

Run directories whose names differ only by a seed suffix (_s1, _s2, ...) form one group: each
(target, landmark) AUC is averaged over the group's seeds before anything is compared. The
first group is the reference. For every other group, the difference is taken pair by pair over
the pairs both groups scored, with a 95% interval from the spread of those differences (normal
approximation; pairs share subjects, so read the interval as a guide). Held-out negative
log-likelihood is shown when the runs recorded it (lower is better).
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pandas as pd

SEED_SUFFIX = re.compile(r"_s\d+$")


def group_runs(paths: list[str | Path]) -> dict[str, list[Path]]:
    """Run name without its seed suffix → run directories, in the order given."""
    groups: dict[str, list[Path]] = {}
    for p in map(Path, paths):
        groups.setdefault(SEED_SUFFIX.sub("", p.name), []).append(p)
    return groups


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


def compare(paths: list[str | Path]) -> pd.DataFrame:
    groups = group_runs(paths)
    loaded = {name: load_group(dirs) for name, dirs in groups.items()}
    ref_name = next(iter(loaded))
    ref = loaded[ref_name][0]
    rows = []
    for name, (auc, nll) in loaded.items():
        row = {"run": name, "seeds": len(groups[name]), "pairs": len(auc), "mean_auc": auc.mean(), "nll": nll,
               "diff": None, "ci_low": None, "ci_high": None, "shared_pairs": None}
        if name != ref_name:
            both = pd.concat([ref, auc], axis=1, join="inner").dropna()
            d = both.iloc[:, 1] - both.iloc[:, 0]
            half = 1.96 * d.std(ddof=1) / math.sqrt(len(d)) if len(d) > 1 else float("nan")
            row.update(diff=d.mean(), ci_low=d.mean() - half, ci_high=d.mean() + half, shared_pairs=len(d))
        rows.append(row)
    return pd.DataFrame(rows)


def print_comparison(table: pd.DataFrame, log=print) -> None:
    ref = table.iloc[0]["run"]
    log(f"{'run':30} {'seeds':>5} {'pairs':>5} {'mean AUC':>9} {'held-out NLL':>13}   difference from {ref} (95% CI)")
    for r in table.itertuples():
        nll = f"{r.nll:13.4f}" if r.nll is not None and not pd.isna(r.nll) else f"{'':13}"
        diff = (f"   {r.diff:+.4f} ({r.ci_low:+.4f} to {r.ci_high:+.4f}) over {int(r.shared_pairs)} pairs"
                if r.diff is not None and not pd.isna(r.diff) else "")
        log(f"{r.run:30} {r.seeds:5d} {r.pairs:5d} {r.mean_auc:9.3f} {nll}{diff}")
