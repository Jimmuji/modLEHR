"""Model size against training data on MIMIC-IV diagnoses (the runs of scripts/fir/scaling.sh and study.sh):
for each size of the plain GPT and each share of the training subjects, mean AUC, held-out negative
log-likelihood and calibration, over seeds, with logistic regression on the same share. Writes a table and
a figure.

    python scripts/analysis/scaling.py runs
    python scripts/analysis/scaling.py runs/fir_v4 --out runs/scaling

Whether a larger model or more data helps more is the question: if the curves for larger models rise with
data and keep apart, the model is limited by data; if they lie on top of each other, by neither.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PLAIN = "mimic_iv_core_gpt"
SIZES = {"xs": "XS (1.1M)", "s": "S (5.3M)", "m": "M (27M)", "l": "L (87M)"}
SHARES = {10: 0.10, 25: 0.25, 50: 0.50, 100: 1.00}


def run_name(size: str, share: int) -> str:
    base = PLAIN if size == "s" else f"{PLAIN}_{size}"
    return base if share == 100 else f"{base}_frac{share}"


def runs(root: Path, name: str) -> list[dict]:
    """evaluate.json of each seed of a run (or of the single baseline run of that name)."""
    dirs = sorted(root.glob(f"{name}_s[0-9]*")) or ([root / name] if (root / name).is_dir() else [])
    return [json.loads((d / "evaluate.json").read_text()) for d in dirs if (d / "evaluate.json").exists()]


def mean_ci(values: list[float]) -> tuple[float, float]:
    values = [v for v in values if v is not None and not math.isnan(v)]
    if not values:
        return float("nan"), float("nan")
    if len(values) < 2:
        return float(np.mean(values)), float("nan")
    return float(np.mean(values)), float(stats.t.ppf(0.975, len(values) - 1) * np.std(values, ddof=1) / math.sqrt(len(values)))


def table(root: Path) -> pd.DataFrame:
    rows = []
    for size, label in SIZES.items():
        for share, fraction in SHARES.items():
            found = runs(root, run_name(size, share))
            if not found:
                continue
            auc, auc_ci = mean_ci([r.get("mean_auc") for r in found])
            nll, nll_ci = mean_ci([r.get("likelihood", {}).get("nll") for r in found])
            eo, _ = mean_ci([r.get("calibration", {}).get("expected_over_observed") for r in found])
            rows.append({"model": label, "size": size, "share": fraction, "seeds": len(found), "mean_auc": auc,
                         "auc_ci95": auc_ci, "nll": nll, "nll_ci95": nll_ci, "e_over_o": eo})
    for share, fraction in SHARES.items():
        found = runs(root, f"{run_name('s', share)}_logistic")
        if found:
            rows.append({"model": "Logistic regression", "size": "logistic", "share": fraction, "seeds": 1,
                         "mean_auc": found[0]["mean_auc"], "auc_ci95": float("nan"), "nll": float("nan"),
                         "nll_ci95": float("nan"), "e_over_o": float("nan")})
    return pd.DataFrame(rows)


def figure(t: pd.DataFrame, path: Path) -> None:
    import matplotlib  # only for the figure
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"xs": "#93C5FD", "s": "#3B82F6", "m": "#1D4ED8", "l": "#1E3A8A", "logistic": "#9CA3AF"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for size, part in t.groupby("size", sort=False):
        part = part.sort_values("share")
        label = part["model"].iloc[0]
        style = dict(color=colors[size], marker="o", lw=2, ms=6, label=label, ls="--" if size == "logistic" else "-")
        axes[0].errorbar(part["share"] * 100, part["mean_auc"], yerr=part["auc_ci95"].fillna(0), capsize=3, **style)
        if size != "logistic":
            axes[1].errorbar(part["share"] * 100, part["nll"], yerr=part["nll_ci95"].fillna(0), capsize=3, **style)
    for ax, title in [(axes[0], "Mean AUC, held-out (higher is better)"),
                      (axes[1], "Held-out negative log-likelihood (lower is better)")]:
        ax.set_xscale("log")
        ax.set_xticks([10, 25, 50, 100])
        ax.set_xticklabels(["10%", "25%", "50%", "100%"])
        ax.set_xlabel("Share of the training subjects")
        ax.set_title(title, fontsize=11)
        ax.grid(color="#E5E7EB")
        ax.set_axisbelow(True)
    axes[0].legend(fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("root", type=Path, help="directory of evaluated runs")
    parser.add_argument("--out", type=Path, help="output stem (default: <root>/scaling)")
    args = parser.parse_args()
    t = table(args.root)
    out = args.out or args.root / "scaling"
    out.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(out.with_suffix(".csv"), index=False)
    try:
        figure(t, out.with_suffix(".png"))
    except ModuleNotFoundError:  # no matplotlib on Fir: draw it where the table is copied to
        print("matplotlib is not installed here; table only")
    print(t.round(4).to_string(index=False))
    print(f"→ {out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
