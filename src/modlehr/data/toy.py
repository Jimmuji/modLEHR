"""A small synthetic MEDS dataset for trying modLEHR in a minute, and for the tests.

    modlehr convert toy --out data/toy_meds
    modlehr run --config configs/experiment/toy_gpt.yaml

Subjects born 1940 to 1960, with sex, HbA1c measured every three years and drifting upward with
age, three diseases each present in about half of the subjects, a finding that no rule selects, an
optional discharge note, and death for about 30%. Nothing here is real data or meant to be learned
from; it only exercises every stage.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import meds
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DISORDERS = {"SNOMEDCT//1001": "Disease A (disorder)", "SNOMEDCT//1002": "Disease B (disorder)",
             "SNOMEDCT//1003": "Disease C (disorder)"}
FINDING = {"SNOMEDCT//2001": "Something noted (finding)"}
HBA1C = "LOINC//4548-4"


NOTE = "NOTE//DISCHARGE"


def note_text(has_a: bool) -> str:
    return "Patient reports polyuria and fatigue." if has_a else "Routine visit, no complaints."


def make_meds(root: str | Path, n: int = 160, seed: int = 0, notes: bool = False) -> Path:
    root = Path(root)
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


def convert(out_dir: str | Path, n: int = 2000, seed: int = 0, notes: bool = True) -> dict:
    root = make_meds(out_dir, n=n, seed=seed, notes=notes)
    (root / meds.dataset_metadata_filepath).write_text(json.dumps(
        {"dataset_name": "modLEHR toy", "etl_name": "modlehr.data.toy", "etl_version": "0.1.0",
         "meds_version": meds.__version__}, indent=2))
    return {"subjects": n, "notes": notes}
