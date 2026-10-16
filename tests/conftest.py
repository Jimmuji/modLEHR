"""A tiny synthetic MEDS dataset, written with the official meds schemas."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import meds
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from modlehr.config import Config, DataConfig, EvaluateConfig, EventRule, ModelConfig, TrainConfig

DISORDERS = {"SNOMEDCT//1001": "Disease A (disorder)", "SNOMEDCT//1002": "Disease B (disorder)",
             "SNOMEDCT//1003": "Disease C (disorder)"}
FINDING = {"SNOMEDCT//2001": "Something noted (finding)"}
HBA1C = "LOINC//4548-4"


NOTE = "NOTE//DISCHARGE"


def note_text(has_a: bool) -> str:
    return "Patient reports polyuria and fatigue." if has_a else "Routine visit, no complaints."


def make_meds(root: Path, n: int = 160, seed: int = 0, notes: bool = False) -> Path:
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n):
        birth = dt.datetime(1940, 1, 1) + dt.timedelta(days=int(rng.integers(0, 20 * 365)))
        rows.append((s, birth, meds.birth_code, None, None))
        rows.append((s, None, "GENDER//" + ("F" if s % 2 else "M"), None, None))
        level = rng.normal(5.8, 0.6)
        for year in range(20, 80, 3):
            when = birth + dt.timedelta(days=365 * year + int(rng.integers(0, 300)))
            rows.append((s, when, HBA1C, float(level + 0.03 * year + rng.normal(0, 0.2)), None))
        has_a = False
        for code in DISORDERS:
            if rng.random() < 0.5:
                rows.append((s, birth + dt.timedelta(days=int(rng.integers(30, 75) * 365)), code, None, None))
                has_a = has_a or code == "SNOMEDCT//1001"
        if notes:
            rows.append((s, birth + dt.timedelta(days=45 * 365), NOTE, None, note_text(has_a)))
        rows.append((s, birth + dt.timedelta(days=40 * 365), next(iter(FINDING)), None, None))
        if rng.random() < 0.3:
            rows.append((s, birth + dt.timedelta(days=int(rng.integers(70, 85) * 365)), meds.death_code, None, None))
    df = pd.DataFrame(rows, columns=["subject_id", "time", "code", "numeric_value", "text_value"])
    df["time"] = pd.to_datetime(df["time"]).astype("datetime64[us]")
    split = np.where(np.arange(n) % 10 < 7, meds.train_split, np.where(np.arange(n) % 10 == 7, meds.tuning_split, meds.held_out_split))
    splits = pd.DataFrame({"subject_id": np.arange(n, dtype="int64"), "split": split})
    df = df.merge(splits, on="subject_id").sort_values(["subject_id", "time"], na_position="first")
    for name, group in df.groupby("split"):
        path = root / "data" / name / "0.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pandas(group.drop(columns="split").reset_index(drop=True), preserve_index=False)
        pq.write_table(table.cast(meds.DataSchema.schema()), path)
    codes = pd.DataFrame({"code": list(DISORDERS) + list(FINDING) + [HBA1C, meds.birth_code, meds.death_code, "GENDER//F", "GENDER//M", NOTE],
                          "description": list(DISORDERS.values()) + list(FINDING.values()) + ["HbA1c", "Birth", "Death", "F", "M", "Discharge note"]})
    codes["parent_codes"] = None
    (root / "metadata").mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(codes, preserve_index=False).cast(meds.CodeMetadataSchema.schema()), root / meds.code_metadata_filepath)
    pq.write_table(pa.Table.from_pandas(splits, preserve_index=False).cast(meds.SubjectSplitSchema.schema()), root / meds.subject_splits_filepath)
    return root


@pytest.fixture
def tiny_meds(tmp_path: Path) -> Path:
    return make_meds(tmp_path / "meds")


@pytest.fixture
def tiny_config(tmp_path: Path, tiny_meds: Path) -> Config:
    data = DataConfig(
        meds_dir=str(tiny_meds), prepared_dir=str(tmp_path / "prepared"), min_subjects_per_token=2, value_mode="continuous",
        events=[EventRule(name="sex", role="static", prefix="GENDER//"),
                EventRule(name="death", role="target", codes=["MEDS_DEATH"]),
                EventRule(name="disorders", role="target", prefix="SNOMEDCT//", description_regex=r"\(disorder\)"),
                EventRule(name="hba1c", role="measurement", codes=[HBA1C], thin_years=5)])
    model = ModelConfig(d_model=32, n_layers=2, n_heads=2, dropout=0.0, max_len=64, value_encoding="mlp")
    train = TrainConfig(out_dir=str(tmp_path / "run"), batch_size=16, steps=30, warmup=5, eval_every=10, eval_batches=2, device="cpu")
    evaluate = EvaluateConfig(landmarks=[50.0, 60.0], horizon_years=10.0, min_cases=2)
    return Config(data=data, model=model, train=train, evaluate=evaluate)
