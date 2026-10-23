"""How much each choice moves the results on MIMIC-IV diagnoses: model parts against data and evaluation
settings (the runs of scripts/fir/study.sh). Reads evaluated runs, writes a table and a figure.

    python scripts/analysis/settings_study.py runs                       # on Fir, after study.sh
    python scripts/analysis/settings_study.py runs/fir_v3 --previous runs/fir --out runs/settings_study

Every row is one choice changed from the plain GPT (configs/experiment/mimic_iv_core_gpt.yaml), in one
of four kinds:

    model part      a registered part swapped; the landmark tasks are the plain GPT's
    training data   what the model learns from changes, the tasks do not ("no event" tokens, a share of
                    the training subjects)
    task            who or what is scored changes: where follow-up starts, a washout, the target codes,
                    the horizon, the controls, the diseases averaged over
    readout         the same model and task, a different score read from the model

Columns, each a mean over seeds with a 95% t interval over the seeds both runs have:

    change     reported mean AUC minus the plain GPT's. For a model part or training data, over the pairs
               both score; for a task or readout, over each one's own pairs, since the pairs differ:
               this is how far the number a paper would report moves.
    over_lr    the model's mean AUC minus logistic regression's on the same task (shared pairs): what the
               model adds, which a task change can move too.
    e_over_o   calibration, expected over observed new diagnoses (1 is calibrated).

--previous DIR fills in model parts missing from the main directory with runs from DIR, each compared
with the plain GPT of DIR (results from an earlier version of the code, marked as such).
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PLAIN = "mimic_iv_core_gpt"
PARTS = {"rope_age": "Rotary position over age", "alibi_age": "ALiBi over age", "local_age": "Window over the last 5 years",
         "gqa": "Grouped-query attention", "mla": "Multi-head latent attention", "qknorm": "Query and key normalization",
         "rmsnorm": "RMSNorm", "swiglu": "SwiGLU feed-forward", "moe": "Mixture of experts"}
# (kind, label, experiment, evaluate stem, logistic run, reference for "change")
SETTINGS = [
    ("training data", '"No event" tokens every 5 years', "no_event", "evaluate", "no_event_logistic", PLAIN),
    ("training data", "50% of training subjects", "frac50", "evaluate", "frac50_logistic", PLAIN),
    ("training data", "25% of training subjects", "frac25", "evaluate", "frac25_logistic", PLAIN),
    ("training data", "10% of training subjects", "frac10", "evaluate", "frac10_logistic", PLAIN),
    ("task", "Follow-up from the first visit", "first_contact", "evaluate", "first_contact_logistic", PLAIN),
    ("task", 'First visit + "no event" tokens', "no_event_first_contact", "evaluate", "no_event_first_contact_logistic", PLAIN),
    ("task", "First visit + 1-year washout", "washout", "evaluate", "washout_logistic", PLAIN),
    ("task", "ICD-9 and ICD-10 kept apart", "icd3char", "evaluate", "icd3char_logistic", PLAIN),
    ("task", "Phecodes as targets", "phecode", "evaluate", "phecode_logistic", PLAIN),
    ("task", "Horizon 1 year (not 2)", "", "evaluate_h1", "h1_logistic", PLAIN),
    ("task", "Horizon 5 years (not 2)", "", "evaluate_h5", "h5_logistic", PLAIN),
    ("task", "Controls: never diagnosed", "", "evaluate_never", "never_logistic", PLAIN),
    ("task", "Delphi-2M's diseases only (A-Q, death)", "", "evaluate", "logistic", PLAIN),  # filtered below
    ("readout", "Score: next target, not own risk", "", "evaluate_next_event", "logistic", PLAIN),
]
SUBSET_LABEL = "Delphi-2M's diseases only (A-Q, death)"
SEED = re.compile(r"_s(\d+)$")


@dataclass
class Run:
    seed: int | None
    auc: pd.Series   # per (target, landmark)
    summary: dict


def name(exp: str) -> str:
    return f"{PLAIN}_{exp}" if exp else PLAIN


def load(root: Path, run: str, stem: str = "evaluate", targets: set[str] | None = None) -> list[Run]:
    """The seeds of `run` under root (or the single baseline run of that name) scored as <stem>.csv."""
    dirs = sorted(d for d in root.glob(f"{run}_s*") if SEED.search(d.name) and SEED.sub("", d.name) == run)
    if not dirs and (root / run).is_dir():
        dirs = [root / run]
    out = []
    for d in dirs:
        if not (d / f"{stem}.csv").exists():
            continue
        auc = pd.read_csv(d / f"{stem}.csv").set_index(["target", "landmark"])["auc"]
        if targets is not None:
            auc = auc[auc.index.get_level_values("target").isin(targets)]
        summary = json.loads((d / f"{stem}.json").read_text()) if (d / f"{stem}.json").exists() else {}
        out.append(Run(int(SEED.search(d.name).group(1)) if SEED.search(d.name) else None, auc, summary))
    return out


def interval(diffs: list[float]) -> tuple[float, float, float]:
    """Mean and 95% t interval (nan without two values)."""
    m = float(np.mean(diffs)) if diffs else float("nan")
    if len(diffs) < 2:
        return m, float("nan"), float("nan")
    half = stats.t.ppf(0.975, len(diffs) - 1) * float(np.std(diffs, ddof=1)) / math.sqrt(len(diffs))
    return m, m - half, m + half


def paired(ref: list[Run], runs: list[Run], shared_pairs: bool) -> tuple[float, float, float]:
    """runs minus ref in mean AUC, seed by seed over the seeds both have; a single baseline run on one side
    is set against every seed of the other. shared_pairs: over the pairs every run scores."""
    if not ref or not runs:
        return float("nan"), float("nan"), float("nan")
    index = None
    if shared_pairs:
        for r in ref + runs:
            index = r.auc.index if index is None else index.intersection(r.auc.index)
    mean = lambda r: float((r.auc.reindex(index) if index is not None else r.auc).mean())  # noqa: E731
    if len(ref) == 1 and ref[0].seed is None:
        return interval([mean(r) - mean(ref[0]) for r in runs])
    if len(runs) == 1 and runs[0].seed is None:
        return interval([mean(runs[0]) - mean(r) for r in ref])
    by_seed = {r.seed: r for r in ref}
    return interval([mean(r) - mean(by_seed[r.seed]) for r in runs if r.seed in by_seed])


def calibration(runs: list[Run]) -> float:
    values = [r.summary["calibration"]["expected_over_observed"] for r in runs if "calibration" in r.summary]
    return float(np.mean(values)) if values else float("nan")


def study(root: Path, previous: Path | None, targets_file: Path) -> pd.DataFrame:
    subset = {line.strip() for line in targets_file.read_text().splitlines() if line.strip() and not line.startswith("#")}
    plain = load(root, PLAIN)
    rows = []

    def row(kind, label, runs, ref, lr, shared, source="this code", calibrated=True):
        change, lo, hi = paired(ref, runs, shared)
        adv, alo, ahi = paired(lr, runs, True)
        rows.append({"kind": kind, "setting": label, "seeds": len(runs), "pairs": int(np.mean([len(r.auc) for r in runs])),
                     "mean_auc": float(np.mean([r.auc.mean() for r in runs])), "change": change, "change_low": lo,
                     "change_high": hi, "over_lr": adv, "over_lr_low": alo, "over_lr_high": ahi,
                     # calibration counts every target, so a subset of them has none of its own
                     "e_over_o": calibration(runs) if calibrated else float("nan"),
                     "reference_e_over_o": calibration(ref), "source": source})

    lr_plain = load(root, f"{PLAIN}_logistic")
    if plain:
        row("reference", "Plain GPT", plain, plain, lr_plain, True)
    for part, label in PARTS.items():
        runs, ref, source = load(root, name(part)), plain, "this code"
        if not runs and previous is not None:
            runs, ref, source = load(previous, name(part)), load(previous, PLAIN), f"earlier code ({previous.name})"
        if runs and ref:
            row("model part", label, runs, ref, load(root if source == "this code" else previous, f"{PLAIN}_logistic"),
                True, source)
    for kind, label, exp, stem, lr, _ in SETTINGS:
        targets = subset if label == SUBSET_LABEL else None
        runs = load(root, name(exp), stem, targets)
        if not runs or not plain:
            continue
        row(kind, label, runs, plain, load(root, f"{PLAIN}_{lr}", "evaluate", targets), kind == "training data",
            calibrated=targets is None)
    return pd.DataFrame(rows)


COLORS = {"reference": "#111827", "model part": "#9CA3AF", "training data": "#2563EB", "task": "#D97706", "readout": "#7C3AED"}


def figure(table: pd.DataFrame, path: Path) -> None:
    import matplotlib  # only for the figure
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter

    t = table[table["kind"] != "reference"].iloc[::-1].reset_index(drop=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 0.42 * len(t) + 2.2), sharey=True,
                             gridspec_kw={"width_ratios": [1.3, 1, 0.8]})
    y = np.arange(len(t))
    earlier = t["source"] != "this code"
    for ax, col, title in [(axes[0], "change", "Reported mean AUC minus the plain GPT's"),
                           (axes[1], "over_lr", "Model minus logistic regression, same task")]:
        for yi, r in t.iterrows():
            c = COLORS[r["kind"]]
            if not np.isnan(r[f"{col}_low"]):
                ax.plot([r[f"{col}_low"], r[f"{col}_high"]], [yi, yi], color=c, lw=2)
            ax.plot(r[col], yi, "o", color=c, mfc="white" if earlier[yi] else c, ms=7, zorder=3)
        ax.axvline(0, color="#6B7280", lw=1)
        ax.set_title(title, fontsize=12)
    ref = table[table["kind"] == "reference"]
    if len(ref):
        axes[1].axvline(ref["over_lr"].iloc[0], color="#111827", lw=1, ls="--")
        axes[2].axvline(ref["e_over_o"].iloc[0], color="#111827", lw=1, ls="--")
    if earlier.any():  # the earlier code's plain GPT, which those rows are compared with
        axes[2].axvline(t.loc[earlier, "reference_e_over_o"].iloc[0], color="#9CA3AF", lw=1, ls="--")
    for yi, r in t.iterrows():
        if not np.isnan(r["e_over_o"]):
            c = COLORS[r["kind"]]
            axes[2].plot(r["e_over_o"], yi, "s", color=c, mfc="white" if earlier[yi] else c, ms=7, zorder=3)
    axes[2].axvline(1, color="#6B7280", lw=1)
    axes[2].set_xscale("log")
    ticks = [0.8, 1, 1.5, 2, 3, 4]
    lo, hi = np.nanmin(t["e_over_o"].tolist() + [1.0]), np.nanmax(t["e_over_o"].tolist() + [1.0])
    axes[2].set_xlim(lo / 1.15, hi * 1.15)
    axes[2].xaxis.set_major_locator(FixedLocator([v for v in ticks if lo / 1.15 <= v <= hi * 1.15]))
    axes[2].xaxis.set_minor_locator(NullLocator())
    axes[2].xaxis.set_major_formatter(ScalarFormatter())
    axes[2].set_title("Calibration, expected / observed", fontsize=12)
    for ax in axes:
        ax.grid(axis="x", color="#E5E7EB")
        ax.set_axisbelow(True)
        for yi in range(1, len(t)):  # a line between kinds
            if t["kind"].iloc[yi] != t["kind"].iloc[yi - 1]:
                ax.axhline(yi - 0.5, color="#D1D5DB", lw=0.8)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(t["setting"], fontsize=11)
    axes[0].set_ylim(-0.6, len(t) - 0.4)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=k) for k, c in COLORS.items()
               if k != "reference" and (t["kind"] == k).any()]
    if earlier.any():
        handles.append(plt.Line2D([], [], marker="o", ls="", color="#9CA3AF", mfc="white",
                                  label="earlier code, against its own plain GPT"))
    fig.legend(handles=handles, loc="lower left", ncol=len(handles), fontsize=10, frameon=False, bbox_to_anchor=(0.01, 0.0))
    fig.text(0.99, 0.012, "Dashed: the plain GPT. Lines: 95% intervals over seeds.", ha="right", fontsize=9, color="#6B7280")
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("root", type=Path, help="directory of evaluated runs")
    parser.add_argument("--previous", type=Path, help="earlier runs, for model parts missing from root")
    parser.add_argument("--targets", type=Path, default=Path("configs/targets/delphi2m.txt"))
    parser.add_argument("--out", type=Path, help="output stem (default: <root>/settings_study)")
    args = parser.parse_args()
    table = study(args.root, args.previous, args.targets)
    out = args.out or args.root / "settings_study"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out.with_suffix(".csv"), index=False)
    try:
        figure(table, out.with_suffix(".png"))
    except ModuleNotFoundError:  # Fir's environment has no matplotlib: draw it where the table is copied to
        print("matplotlib is not installed here; table only (draw the figure from the .csv elsewhere)")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(table.drop(columns=["change_low", "change_high", "over_lr_low", "over_lr_high"]).round(4).to_string(index=False))
    print(f"→ {out.with_suffix('.csv')}, {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
