"""Tabular baseline on the Synthea onset tasks (scripts/synthea_tasks.py), as Pham et al. (arXiv 2604.01841) score
Synthea with tabular models: L2 logistic regression on what is recorded by the prediction time, namely the count
of each condition code (SNOMED CT, log(1 + n)), the last value of the 13 measurements the GPT reads
(configs/data/synthea.yaml) with a missing flag, age and sex. C is picked by tuning-split AUROC; held-out AUROC
and AUPRC are reported with a bootstrap 95% interval. Synthea is synthetic, so this runs anywhere in minutes.

  python scripts/synthea_logistic.py data/synthea_meds data/synthea_tasks runs/synthea_logistic
"""

import json
import sys
from pathlib import Path

import meds
import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

SPLITS = (meds.train_split, meds.tuning_split, meds.held_out_split)
YEAR_S = 365.25 * 86400


def measurement_codes() -> list[str]:
    cfg = yaml.safe_load(Path("configs/data/synthea.yaml").read_text())["data"]
    return next(e["codes"] for e in cfg["events"] if e["name"] == "measurements")


def features(meds_dir: Path, samples: pd.DataFrame, split: str, codes: list[str], loinc: list[str]):
    ev = pd.concat(pd.read_parquet(f, columns=["subject_id", "time", "code", "numeric_value"])
                   for f in (meds_dir / "data" / split).glob("*.parquet"))
    ev = ev.loc[ev["subject_id"].isin(set(samples["subject_id"]))]
    birth = ev.loc[ev["code"] == meds.birth_code].groupby("subject_id")["time"].min()
    female = ev.loc[ev["code"] == "GENDER//F", "subject_id"].unique()
    s = samples.reset_index(drop=True).assign(row=lambda d: np.arange(len(d)))
    dense = pd.DataFrame({"age": (s["prediction_time"] - s["subject_id"].map(birth)).dt.total_seconds() / YEAR_S,
                          "female": s["subject_id"].isin(female).astype(float)})
    snomed = ev.loc[ev["code"].str.startswith("SNOMEDCT//") & ev["time"].notna(), ["subject_id", "time", "code"]]
    j = s[["row", "subject_id", "prediction_time"]].merge(snomed, on="subject_id")
    j = j.loc[j["time"] <= j["prediction_time"]]
    counts = j.groupby(["row", "code"]).size().reset_index(name="n")
    col = {c: i for i, c in enumerate(codes)}
    counts = counts.loc[counts["code"].isin(col)]
    X_codes = sp.csr_matrix((np.log1p(counts["n"].to_numpy(float)), (counts["row"], counts["code"].map(col))),
                            shape=(len(s), len(codes)))
    left = s[["row", "subject_id", "prediction_time"]].sort_values("prediction_time")
    for code in loinc:
        m = ev.loc[(ev["code"] == code) & ev["time"].notna() & ev["numeric_value"].notna(),
                   ["subject_id", "time", "numeric_value"]].sort_values("time")
        last = pd.merge_asof(left, m, left_on="prediction_time", right_on="time", by="subject_id", direction="backward")
        dense[code] = last.set_index("row")["numeric_value"].reindex(s["row"]).to_numpy()
    return X_codes, dense, s["boolean_value"].to_numpy(bool)


def bootstrap(y, p, fn, n=500, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].any() and not y[i].all():
            vals.append(fn(y[i], p[i]))
    return [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))]


def main(meds_dir: Path, tasks_dir: Path, out_dir: Path) -> None:
    loinc = measurement_codes()
    train_codes = pd.concat(pd.read_parquet(f, columns=["code"]) for f in (meds_dir / "data" / meds.train_split).glob("*.parquet"))
    codes = sorted(c for c in train_codes["code"].unique() if c.startswith("SNOMEDCT//"))
    results = {}
    for task_dir in sorted(p for p in tasks_dir.iterdir() if p.is_dir()):
        data = {}
        for split in SPLITS:
            samples = pd.concat(pd.read_parquet(f) for f in (task_dir / split).glob("*.parquet"))
            samples["prediction_time"] = samples["prediction_time"].astype("datetime64[us]")
            data[split] = features(meds_dir, samples, split, codes, loinc)
        med = data[meds.train_split][1].median()
        mean = sd = None
        mats = {}
        for split, (Xc, dense, y) in data.items():
            miss = dense[loinc].isna().astype(float).add_suffix("_missing")
            d = pd.concat([dense.fillna(med), miss], axis=1)
            if mean is None:
                mean, sd = d.mean(), d.std().replace(0, 1)
            mats[split] = (sp.hstack([Xc, sp.csr_matrix(((d - mean) / sd).to_numpy())]).tocsr(), y)
        (Xtr, ytr), (Xtu, ytu), (Xte, yte) = (mats[s] for s in SPLITS)
        best = None
        for C in (0.01, 0.03, 0.1, 0.3, 1.0):
            clf = LogisticRegression(C=C, max_iter=3000, solver="liblinear").fit(Xtr, ytr)
            a = roc_auc_score(ytu, clf.predict_proba(Xtu)[:, 1])
            if best is None or a > best[0]:
                best = (a, C, clf)
        tune_auroc, C, clf = best
        p = clf.predict_proba(Xte)[:, 1]
        results[task_dir.name] = {
            "held_out_samples": int(len(yte)), "held_out_positives": int(yte.sum()), "C": C,
            "tuning_auroc": float(tune_auroc), "auroc": float(roc_auc_score(yte, p)),
            "auroc_ci95": bootstrap(yte, p, roc_auc_score), "auprc": float(average_precision_score(yte, p)),
            "prevalence": float(yte.mean())}
        r = results[task_dir.name]
        print(f"{task_dir.name:26s} held_out AUROC {r['auroc']:.3f} ({r['auroc_ci95'][0]:.3f} to {r['auroc_ci95'][1]:.3f}), "
              f"AUPRC {r['auprc']:.3f}, {r['held_out_positives']} of {r['held_out_samples']:,}, C {C}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
