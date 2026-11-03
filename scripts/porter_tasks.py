"""PORTER's three outcome tasks on MIMIC-IV (Guo et al., arXiv 2606.24102), on the official MIMIC_IV_MEDS ETL, as
labels in the MEDS label format: <out>/<task>/<split>/<shard>.parquet, one file per data shard (MEDS-Tab pairs
label files with data shards by name), rows sorted by subject_id and prediction_time.

PORTER predicts at a calendar time, 23:59 of a day, which ACES windows (offsets from an event) cannot express, so
these are not ACES configs. Per hospital admission (HOSPITAL_ADMISSION, matched to the next HOSPITAL_DISCHARGE),
of a patient 18 or older at the prediction time:
  day1_hospital_mortality       at 23:59 of the admission day: death before the discharge (discharge to DIED, or
                                a death record up to the discharge)
  day1_long_stay_7d             at 23:59 of the admission day: discharge 7 days or more after the admission
  predischarge_readmission_30d  at 23:59 of the day before the discharge day: another hospital admission within
                                30 days after the discharge
An admission is left out when the discharge or a death comes at or before the prediction time, or, for
readmission, when the prediction time comes before the admission (a discharge on the admission day). As in PORTER,
a death in hospital after the prediction time is a negative for readmission, not left out. The two day-1 tasks have
the same samples in the same order.

PORTER's own cohort (MIMIC-IV 1.0 through the OHDSI OMOP ETL, 58,513 admissions, split by calendar time) is not
public; this applies its rules to every admission of our patient split.

  python scripts/porter_tasks.py <meds_dir> <out_dir>
"""

import sys
import warnings
from pathlib import Path

import polars as pl

warnings.filterwarnings("ignore", "Sortedness of columns cannot be checked")  # join_asof by subject, sorted within it

TASKS = ("day1_hospital_mortality", "day1_long_stay_7d", "predischarge_readmission_30d")
ADULT_SECONDS = 18 * 365.25 * 86400


def stays(shard: Path) -> pl.DataFrame:
    ev = (pl.scan_parquet(shard).select("subject_id", "time", "code").filter(pl.col("time").is_not_null())
          .filter(pl.col("code").str.starts_with("HOSPITAL_") | pl.col("code").is_in(["MEDS_DEATH", "MEDS_BIRTH"]))
          .collect())
    first = lambda code, name: (ev.filter(pl.col("code") == code).group_by("subject_id")  # noqa: E731
                                .agg(pl.col("time").min().alias(name)))
    adm = (ev.filter(pl.col("code").str.starts_with("HOSPITAL_ADMISSION//"))
           .select("subject_id", pl.col("time").alias("admit")).unique().sort("subject_id", "admit"))
    dis = (ev.filter(pl.col("code").str.starts_with("HOSPITAL_DISCHARGE//")).group_by("subject_id", "time")
           .agg((pl.col("code") == "HOSPITAL_DISCHARGE//DIED").any().alias("died")).rename({"time": "discharge"})
           .sort("subject_id", "discharge"))
    nxt = adm.rename({"admit": "next_admit"})
    return (adm.join_asof(dis, left_on="admit", right_on="discharge", by="subject_id", strategy="forward")
            .filter(pl.col("discharge").is_not_null())
            .sort("subject_id", "discharge")
            .join_asof(nxt, left_on="discharge", right_on="next_admit", by="subject_id", strategy="forward",
                       allow_exact_matches=False)
            .join(first("MEDS_BIRTH", "birth"), on="subject_id", how="left")
            .join(first("MEDS_DEATH", "death"), on="subject_id", how="left"))


def eligible(t: pl.Expr) -> pl.Expr:
    """Alive, still in hospital and adult at prediction time t."""
    return ((pl.col("discharge") > t) & (pl.col("death").is_null() | (pl.col("death") > t))
            & pl.col("birth").is_not_null() & ((t - pl.col("birth")).dt.total_seconds() >= ADULT_SECONDS))


def labels(s: pl.DataFrame) -> dict[str, pl.DataFrame]:
    day1 = pl.col("admit").dt.truncate("1d") + pl.duration(hours=23, minutes=59)
    pre = pl.col("discharge").dt.truncate("1d") - pl.duration(minutes=1)
    died = pl.col("died") | (pl.col("death").is_not_null() & (pl.col("death") <= pl.col("discharge")))
    a = s.filter(eligible(day1)).select("subject_id", day1.alias("prediction_time"), died.alias("mortality"),
                                        (pl.col("discharge") - pl.col("admit") >= pl.duration(days=7)).alias("long"))
    b = s.filter(eligible(pre) & (pl.col("admit") <= pre)).select(
        "subject_id", pre.alias("prediction_time"),
        (pl.col("next_admit").is_not_null() & (pl.col("next_admit") - pl.col("discharge") <= pl.duration(days=30)))
        .alias("boolean_value"))
    key = ["subject_id", "prediction_time"]
    a = a.unique(key, keep="first").sort(key)  # two admissions on one day give one sample
    return {"day1_hospital_mortality": a.select(*key, pl.col("mortality").alias("boolean_value")),
            "day1_long_stay_7d": a.select(*key, pl.col("long").alias("boolean_value")),
            "predischarge_readmission_30d": b.unique(key, keep="first").sort(key)}


def main(meds: Path, out: Path) -> None:
    counts = {}
    for shard in sorted((meds / "data").glob("*/*.parquet")):
        split = shard.parent.name
        for task, df in labels(stays(shard)).items():
            df = df.with_columns(pl.col("subject_id").cast(pl.Int64), pl.col("boolean_value").cast(pl.Boolean))
            (out / task / split).mkdir(parents=True, exist_ok=True)
            df.write_parquet(out / task / split / shard.name)
            c = counts.setdefault((task, split), [0, 0, set()])
            c[0] += len(df)
            c[1] += int(df["boolean_value"].sum())
            c[2].update(df["subject_id"].to_list())
    for (task, split), (n, pos, subjects) in sorted(counts.items()):
        print(f"{split} {task}: samples {n} from {len(subjects)} subjects, positives {pos} ({pos / max(n, 1):.4f})")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
