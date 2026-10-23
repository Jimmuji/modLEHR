"""Where a model beats a baseline: mean AUC by ICD-10 chapter, by landmark age and by how many cases a
pair has, for a model's seeds against a baseline on the same pairs. The overall mean AUC weighs every
(target, landmark) pair the same, so a thousand uncommon targets decide it; this shows what it is made of.

    python scripts/analysis/by_chapter.py runs/mimic_iv_core_gpt_s1 runs/mimic_iv_core_gpt_s2 \\
        runs/mimic_iv_core_gpt_s3 --baseline runs/mimic_iv_core_gpt_logistic --out runs/by_chapter.csv

Targets are modLEHR target names (ICD10//I50, MEDS_DEATH); anything else is grouped as "other".
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CHAPTERS = [  # (first code, last code, chapter), ICD-10 three-character ranges
    ("A00", "B99", "I Infections"), ("C00", "D49", "II Neoplasms"), ("D50", "D89", "III Blood and immune"),
    ("E00", "E89", "IV Endocrine and metabolic"), ("F01", "F99", "V Mental"), ("G00", "G99", "VI Nervous system"),
    ("H00", "H59", "VII Eye"), ("H60", "H95", "VIII Ear"), ("I00", "I99", "IX Circulatory"),
    ("J00", "J99", "X Respiratory"), ("K00", "K95", "XI Digestive"), ("L00", "L99", "XII Skin"),
    ("M00", "M99", "XIII Musculoskeletal"), ("N00", "N99", "XIV Genitourinary"), ("O00", "O9A", "XV Pregnancy"),
    ("P00", "P96", "XVI Perinatal"), ("Q00", "Q99", "XVII Congenital"), ("R00", "R99", "XVIII Symptoms and findings"),
    ("S00", "T98", "XIX Injury and poisoning"), ("V00", "Y99", "XX External causes"), ("Z00", "Z99", "XXI Health services"),
]


def chapter(target: str) -> str:
    if target == "MEDS_DEATH":
        return "Death"
    code = target.split("//", 1)[1] if target.startswith("ICD10//") else None
    if not code:
        return "other"
    for lo, hi, name in CHAPTERS:
        if lo <= code[:3] <= hi:
            return name
    return "other"


def pair_aucs(runs: list[Path]) -> pd.DataFrame:
    """AUC per (target, landmark) averaged over the runs, with the cases of the first."""
    tables = [pd.read_csv(Path(r) / "evaluate.csv").set_index(["target", "landmark"]) for r in runs]
    out = pd.concat([t["auc"] for t in tables], axis=1).mean(axis=1).rename("auc").to_frame()
    return out.join(tables[0]["cases"])


def breakdown(model: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    both = model.join(baseline["auc"].rename("baseline_auc"), how="inner").reset_index()
    both["chapter"] = both["target"].map(chapter)
    both["case_quartile"] = pd.qcut(both["cases"], 4).astype(str)
    parts = []
    for by in ("chapter", "landmark", "case_quartile"):
        g = both.groupby(by).agg(pairs=("auc", "size"), model=("auc", "mean"), baseline=("baseline_auc", "mean"))
        parts.append(g.reset_index().rename(columns={by: "group"}).assign(by=by))
    total = pd.DataFrame([{"by": "all", "group": "all pairs", "pairs": len(both), "model": both["auc"].mean(),
                           "baseline": both["baseline_auc"].mean()},
                          {"by": "all", "group": "weighted by cases", "pairs": len(both),
                           "model": np.average(both["auc"], weights=both["cases"]),
                           "baseline": np.average(both["baseline_auc"], weights=both["cases"])}])
    out = pd.concat([total, *parts], ignore_index=True)
    out["difference"] = out["model"] - out["baseline"]
    return out[["by", "group", "pairs", "model", "baseline", "difference"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("runs", nargs="+", type=Path, help="the model's evaluated runs (its seeds)")
    parser.add_argument("--baseline", type=Path, required=True, help="an evaluated baseline run on the same task")
    parser.add_argument("--out", type=Path, help="write the table here as CSV")
    args = parser.parse_args()
    table = breakdown(pair_aucs(args.runs), pair_aucs([args.baseline]))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.out, index=False)
    for by, part in table.groupby("by", sort=False):
        part = part.sort_values("pairs", ascending=False) if by == "chapter" else part
        print(part.drop(columns="by").round(3).to_string(index=False), end="\n\n")


if __name__ == "__main__":
    main()
