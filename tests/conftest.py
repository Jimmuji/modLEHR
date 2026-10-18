"""Fixtures: the toy MEDS dataset (modlehr.data.toy) and a small config that reads it, and a writer
for small MEDS datasets built row by row."""

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
from modlehr.data.toy import DISORDERS, FINDING, HBA1C, NOTE, make_meds, note_text  # noqa: F401


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


def write_meds(root: Path, rows: list[tuple], shards: int = 1) -> Path:
    """A MEDS dataset from (subject_id, age in years or None, code, numeric_value) rows, every subject born
    on 1 January 1950 and in the training split, spread over `shards` shards by subject."""
    birth = dt.datetime(1950, 1, 1)
    subjects = sorted({r[0] for r in rows})
    events = [(s, birth, meds.birth_code, None) for s in subjects]
    events += [(s, None if age is None else birth + dt.timedelta(days=365.25 * age), code, value)
               for s, age, code, value in rows]
    df = pd.DataFrame(events, columns=["subject_id", "time", "code", "numeric_value"]).assign(text_value=None)
    df["time"] = pd.to_datetime(df["time"]).astype("datetime64[us]")
    df = df.sort_values(["subject_id", "time"], na_position="first", kind="stable")
    for i in range(shards):
        part = df[df["subject_id"].map({s: j % shards for j, s in enumerate(subjects)}) == i]
        path = root / "data" / "train" / f"{i}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pandas(part.reset_index(drop=True), preserve_index=False)
                       .cast(meds.DataSchema.schema()), path)
    codes = pd.DataFrame({"code": sorted(set(df["code"])), "description": None, "parent_codes": None})
    (root / "metadata").mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(codes, preserve_index=False).cast(meds.CodeMetadataSchema.schema()),
                   root / meds.code_metadata_filepath)
    splits = pd.DataFrame({"subject_id": np.asarray(subjects, dtype="int64"), "split": meds.train_split})
    pq.write_table(pa.Table.from_pandas(splits, preserve_index=False).cast(meds.SubjectSplitSchema.schema()),
                   root / meds.subject_splits_filepath)
    return root
