"""Compare evaluation results of several runs against a reference run.

    python scripts/compare_runs.py runs/synthea_gpt runs/synthea_gpt_rope_age runs/synthea_gpt_gqa

The first run is the reference. For each other run: mean AUC, and the paired difference over the
(target, landmark) pairs both runs scored, with a 95% interval (normal approximation).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd


def main(runs: list[str]) -> None:
    tables = {Path(r).name: pd.read_csv(Path(r) / "evaluate.csv").set_index(["target", "landmark"])["auc"] for r in runs}
    names = list(tables)
    ref = names[0]
    print(f"{'run':28} {'pairs':>5} {'mean AUC':>9} {'vs ' + ref:>32}")
    for name in names:
        auc = tables[name]
        line = f"{name:28} {len(auc):5d} {auc.mean():9.3f}"
        if name != ref:
            both = pd.concat([tables[ref], auc], axis=1, join="inner").dropna()
            d = both.iloc[:, 1] - both.iloc[:, 0]
            half = 1.96 * d.std() / math.sqrt(len(d))
            line += f"   {d.mean():+.4f} (95% CI {d.mean() - half:+.4f} to {d.mean() + half:+.4f})"
        print(line)


if __name__ == "__main__":
    main(sys.argv[1:])
