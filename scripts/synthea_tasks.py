"""Disease-onset tasks on Synthea in the MEDS label format, for `modlehr finetune` and the logistic baseline
(scripts/synthea_logistic.py): at each age L in 40, 45, ..., 80, does the disorder (or death) first occur in
(L, L + 5 years]? Synthea is synthetic, so rows can be looked at. Pham et al. (arXiv 2604.01841) report stroke and
lung cancer near AUROC 1 on Synthea with tabular models; the other tasks are the commonest late-life disorders.

A subject gives a sample at L when it was born, has an event before L, is alive at L, has not had the disorder by
L, and is followed to L + 5 (a later event or death). Death in the window without the disorder is a negative.

  python scripts/synthea_tasks.py data/synthea_meds data/synthea_tasks
"""

import sys
from pathlib import Path

import meds
import pandas as pd

TASKS = {
    "stroke_5y": ["SNOMEDCT//230690007"],
    "lung_cancer_5y": ["SNOMEDCT//254637007", "SNOMEDCT//254632001"],
    "myocardial_infarction_5y": ["SNOMEDCT//22298006", "SNOMEDCT//401303003", "SNOMEDCT//401314000"],
    "heart_failure_5y": ["SNOMEDCT//88805009", "SNOMEDCT//84114007"],
    "type2_diabetes_5y": ["SNOMEDCT//44054006"],
    "ckd_stage3_5y": ["SNOMEDCT//433144002"],
    "death_5y": [meds.death_code],
}
AGES = range(40, 81, 5)
YEAR = pd.Timedelta(days=365.25)


def main(meds_dir: Path, out: Path) -> None:
    for split_dir in sorted((meds_dir / "data").iterdir()):
        ev = pd.concat(pd.read_parquet(f, columns=["subject_id", "time", "code"]) for f in split_dir.glob("*.parquet"))
        ev = ev.dropna(subset=["time"])
        birth = ev.loc[ev["code"] == meds.birth_code].groupby("subject_id")["time"].min()
        death = ev.loc[ev["code"] == meds.death_code].groupby("subject_id")["time"].min()
        other = ev.loc[ev["code"] != meds.birth_code]
        first_event, last_event = other.groupby("subject_id")["time"].min(), other.groupby("subject_id")["time"].max()
        subj = pd.DataFrame({"birth": birth}).join(first_event.rename("first")).join(last_event.rename("last")) \
            .join(death.rename("death"))
        for task, codes in TASKS.items():
            onset = ev.loc[ev["code"].isin(codes)].groupby("subject_id")["time"].min()
            s = subj.join(onset.rename("onset"))
            rows = []
            for L in AGES:
                at = s["birth"] + L * YEAR
                end = at + 5 * YEAR
                ok = s["first"].lt(at) & ~(s["death"].le(at)) & ~(s["onset"].le(at))
                ok &= s["last"].ge(end) | s["death"].le(end) | s["onset"].le(end)
                pos = s["onset"].gt(at) & s["onset"].le(end)
                rows.append(pd.DataFrame({"subject_id": s.index[ok].astype("int64"),
                                          "prediction_time": at[ok].astype("datetime64[us]").to_numpy(),
                                          "boolean_value": pos[ok].to_numpy()}))
            df = pd.concat(rows, ignore_index=True).sort_values(["subject_id", "prediction_time"])
            d = out / task / split_dir.name
            d.mkdir(parents=True, exist_ok=True)
            df.to_parquet(d / "0.parquet", index=False)
            print(f"{task:26s} {split_dir.name:9s} samples {len(df):6,} subjects {df.subject_id.nunique():6,} "
                  f"positives {int(df.boolean_value.sum()):5,} ({df.boolean_value.mean():.3%})")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
