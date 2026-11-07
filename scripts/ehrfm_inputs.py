"""Inputs for PORTER's training code, ehr-fm (github.com/sungresearch/ehr-fm), on the official MIMIC_IV_MEDS ETL:
the same patients, split and task labels as our GPT and MEDS-Tab.

  python scripts/ehrfm_inputs.py meds <meds_dir> <out_meds_dir>
      A copy of the MEDS data that ehr-fm's reader accepts: each subject's events without a time (one static
      code per subject in this ETL) placed at its MEDS_BIRTH, as an OMOP ETL places demographics (ehr-fm compares
      every event time with the index time), and a null `description` column, which meds_reader databases must
      have; only the columns ehr-fm reads are kept. A subject without MEDS_BIRTH keeps no timeless events (ehr-fm
      skips such subjects anyway).
  python scripts/ehrfm_inputs.py cohorts <meds_dir> <labels_dir> <out_dir>
      <out_dir>/pretraining.parquet: one row per train and tuning subject (id, index_t = its last event, split
      train or validation), as PORTER's pretraining cohort. <out_dir>/<group>.parquet: the labels of the tasks
      that list the same samples (id, index_t = prediction time, split train / val / test for our train / tuning /
      held_out, one label_<task> column each), as PORTER's task cohorts, for its linear probe.

Prints counts only.
"""

import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import polars as pl

warnings.filterwarnings("ignore", "Sortedness of columns cannot be checked")

GROUPS = {
    "admission": ("admission_hospital_mortality", "admission_icu", "admission_long_stay_10d"),
    "day1": ("day1_hospital_mortality", "day1_long_stay_7d"),
    "predischarge": ("predischarge_readmission_30d",),
}
SPLITS = {"train": "train", "tuning": "val", "held_out": "test"}


def copy_shard(args: tuple[Path, Path]) -> dict[str, int]:
    src, dst = args
    ev = pl.read_parquet(src, columns=["subject_id", "time", "code", "numeric_value", "text_value", "unit"])
    birth = ev.filter(pl.col("code") == "MEDS_BIRTH").group_by("subject_id").agg(pl.col("time").min().alias("birth"))
    timeless = ev["time"].is_null()
    ev = (ev.join(birth, on="subject_id", how="left")
          .with_columns(pl.col("time").fill_null(pl.col("birth")))
          .filter(pl.col("time").is_not_null())
          .sort("subject_id", "time", maintain_order=True)
          .select("subject_id", "time", "code", "numeric_value", pl.col("text_value").cast(pl.Utf8),
                  pl.col("unit").cast(pl.Utf8), pl.lit(None, pl.Utf8).alias("description")))
    dst.parent.mkdir(parents=True, exist_ok=True)
    ev.write_parquet(dst)
    return {"events": len(ev), "timeless": int(timeless.sum()), "subjects": ev["subject_id"].n_unique(),
            "without_birth": ev["subject_id"].n_unique() - len(birth)}


def meds(src: Path, dst: Path) -> None:
    shards = sorted((src / "data").glob("*/*.parquet"))
    os.environ["POLARS_MAX_THREADS"] = "1"
    with ProcessPoolExecutor(int(os.environ.get("SLURM_CPUS_PER_TASK", 4))) as pool:
        counts = list(pool.map(copy_shard, [(f, dst / "data" / f.parent.name / f.name) for f in shards]))
    (dst / "metadata").mkdir(parents=True, exist_ok=True)
    for f in (src / "metadata").iterdir():
        (dst / "metadata" / f.name).write_bytes(f.read_bytes())
    total = {k: sum(c[k] for c in counts) for k in counts[0]}
    print(f"{len(shards)} shards: {total['events']:,} events of {total['subjects']:,} subjects; {total['timeless']:,} "
          f"events without a time placed at birth; {total['without_birth']:,} subjects without MEDS_BIRTH")


def cohorts(meds_dir: Path, labels: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    splits = pl.read_parquet(meds_dir / "metadata" / "subject_splits.parquet").select("subject_id", "split")
    last = (pl.scan_parquet(meds_dir / "data" / "*" / "*.parquet").group_by("subject_id")
            .agg(pl.col("time").max().alias("index_t"), (pl.col("code") == "MEDS_BIRTH").any().alias("birth"))
            .collect())
    pre = (last.join(splits, on="subject_id").filter(pl.col("split").is_in(["train", "tuning"]))
           .select(pl.col("subject_id").cast(pl.Int64).alias("id"), pl.col("index_t").cast(pl.Datetime("us")),
                   pl.col("split").replace({"tuning": "validation"})).sort("id"))
    pre.write_parquet(out / "pretraining.parquet")
    for split, g in pre.group_by("split", maintain_order=True):
        print(f"pretraining {split[0]}: {len(g):,} subjects")
    no_birth = set(last.filter(~pl.col("birth"))["subject_id"].to_list())

    key = ["id", "index_t"]
    for group, tasks in GROUPS.items():
        df = None
        for task in tasks:
            parts = []
            for ours, theirs in SPLITS.items():
                lab = pl.read_parquet(labels / task / ours / "*.parquet", columns=["subject_id", "prediction_time",
                                                                                   "boolean_value"])
                parts.append(lab.select(pl.col("subject_id").cast(pl.Int64).alias("id"),
                                        pl.col("prediction_time").cast(pl.Datetime("us")).alias("index_t"),
                                        pl.lit(theirs).alias("split"),
                                        pl.col("boolean_value").cast(pl.Int8).alias(f"label_{task}")))
            t = pl.concat(parts)
            dup = len(t) - t.select(key).n_unique()
            if dup:
                print(f"{task}: {dup} samples share a subject and time with another; the first is kept")
                t = t.unique(key, keep="first", maintain_order=True)
            if df is None:
                df = t
            else:
                n = len(df)
                df = df.join(t.drop("split"), on=key, how="inner")
                assert len(df) == n == len(t), f"{task} lists other samples than {tasks[0]}"
        df = df.sort(key)
        df.write_parquet(out / f"{group}.parquet")
        sub = df.with_columns(pl.col("index_t").dt.microsecond().alias("us"))
        for split in ("train", "val", "test"):
            s = sub.filter(pl.col("split") == split)
            pos = ", ".join(f"{c[6:]} {int(s[c].sum()):,}" for c in df.columns if c.startswith("label_"))
            print(f"{group} {split}: {len(s):,} samples of {s['id'].n_unique():,} subjects ({pos}); "
                  f"{int(s['id'].is_in(no_birth).sum())} from subjects without MEDS_BIRTH; "
                  f"{int((s['us'] != 0).sum())} with a fractional second")


if __name__ == "__main__":
    if sys.argv[1] == "meds":
        meds(Path(sys.argv[2]), Path(sys.argv[3]))
    elif sys.argv[1] == "cohorts":
        cohorts(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))
    else:
        raise SystemExit(__doc__)
