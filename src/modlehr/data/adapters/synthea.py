"""Synthea → MEDS.

Synthea writes FHIR R4 by default; this adapter reads its CSV export (patients.csv,
conditions.csv, observations.csv), as written with `--exporter.csv.export true`, either as
one directory or as chunk directories that each contain a `csv/` folder.

Every source row becomes one MEDS event. Nothing is binned, deduplicated or filtered for a
model here; those choices belong to `modlehr prepare`.

    subject_id     int64, assigned in order of the Synthea patient UUID
    time           event datetime; null for static rows (sex)
    code           MEDS_BIRTH, MEDS_DEATH, GENDER//F|M, SNOMEDCT//<code>, LOINC//<code>
    numeric_value  numeric observations (labs, vitals, BMI)
    text_value     text observations (for example smoking status)
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

from modlehr import __version__


def _csv_dirs(raw_dir: Path) -> list[Path]:
    if (raw_dir / "patients.csv").exists():
        return [raw_dir]
    dirs = sorted(p / "csv" for p in raw_dir.iterdir() if (p / "csv" / "patients.csv").exists())
    if not dirs:
        raise FileNotFoundError(f"no Synthea CSV export found under {raw_dir}")
    return dirs


def _time(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True).dt.tz_localize(None).astype("datetime64[us]")


def _events(csv: Path, uuid_to_id: dict) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    p = pd.read_csv(csv / "patients.csv", usecols=["Id", "BIRTHDATE", "DEATHDATE", "GENDER"])
    sid = p["Id"].map(uuid_to_id)
    parts = [
        pd.DataFrame({"subject_id": sid, "time": _time(p["BIRTHDATE"]), "code": meds.birth_code}),
        pd.DataFrame({"subject_id": sid, "time": pd.NaT, "code": "GENDER//" + p["GENDER"]}),
    ]
    d = p.dropna(subset=["DEATHDATE"])
    parts.append(pd.DataFrame({"subject_id": d["Id"].map(uuid_to_id), "time": _time(d["DEATHDATE"]), "code": meds.death_code}))

    c = pd.read_csv(csv / "conditions.csv", usecols=["START", "PATIENT", "CODE", "DESCRIPTION"], dtype={"CODE": str})
    parts.append(pd.DataFrame({"subject_id": c["PATIENT"].map(uuid_to_id), "time": _time(c["START"]), "code": "SNOMEDCT//" + c["CODE"]}))
    codes = [pd.DataFrame({"code": "SNOMEDCT//" + c["CODE"], "description": c["DESCRIPTION"], "unit": None})]

    o = pd.read_csv(csv / "observations.csv", usecols=["DATE", "PATIENT", "CODE", "DESCRIPTION", "VALUE", "UNITS", "TYPE"],
                    dtype={"CODE": str, "VALUE": str})
    numeric = o["TYPE"] == "numeric"
    parts.append(pd.DataFrame({
        "subject_id": o["PATIENT"].map(uuid_to_id), "time": _time(o["DATE"]), "code": "LOINC//" + o["CODE"],
        "numeric_value": pd.to_numeric(o["VALUE"].where(numeric), errors="coerce").astype("float32"),
        "text_value": o["VALUE"].where(~numeric),
    }))
    codes.append(pd.DataFrame({"code": "LOINC//" + o["CODE"], "description": o["DESCRIPTION"], "unit": o["UNITS"]}))
    return pd.concat(parts, ignore_index=True), codes


def _write(df: pd.DataFrame, path: Path) -> None:
    if "numeric_value" not in df:
        df["numeric_value"] = np.nan
    if "text_value" not in df:
        df["text_value"] = None
    df = df[["subject_id", "time", "code", "numeric_value", "text_value"]]
    table = pa.Table.from_pandas(df, preserve_index=False).cast(meds.DataSchema.schema())
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def convert(raw_dir: str | Path, out_dir: str | Path, held_out: float = 0.2, tuning: float = 0.1, seed: int = 1413) -> dict:
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    dirs = _csv_dirs(raw_dir)
    uuids = sorted(pd.concat([pd.read_csv(d / "patients.csv", usecols=["Id"])["Id"] for d in dirs]))
    uuid_to_id = {u: i for i, u in enumerate(uuids)}

    rng = np.random.default_rng(seed)
    is_held = rng.random(len(uuids)) < held_out
    is_tune = (~is_held) & (np.random.default_rng(seed + 1).random(len(uuids)) < tuning)
    split = np.where(is_held, meds.held_out_split, np.where(is_tune, meds.tuning_split, meds.train_split))
    splits = pd.DataFrame({"subject_id": np.arange(len(uuids), dtype="int64"), "split": split})

    all_codes, n_events = [], 0
    for i, csv in enumerate(dirs):
        events, codes = _events(csv, uuid_to_id)
        all_codes += codes
        events = events.merge(splits, on="subject_id")
        events = events.sort_values(["subject_id", "time"], na_position="first", kind="stable")
        for name, group in events.groupby("split"):
            _write(group.drop(columns="split").reset_index(drop=True), out_dir / meds.data_subdirectory / name / f"{i}.parquet")
        n_events += len(events)

    codes = pd.concat(all_codes).drop_duplicates("code")
    codes = pd.concat([codes, pd.DataFrame({
        "code": [meds.birth_code, meds.death_code, "GENDER//F", "GENDER//M"],
        "description": ["Birth", "Death", "Sex: female", "Sex: male"],
    })]).assign(parent_codes=None).reset_index(drop=True)
    schema = pa.schema(list(meds.CodeMetadataSchema.schema()) + [pa.field("unit", pa.string())])
    (out_dir / "metadata").mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(codes[["code", "description", "parent_codes", "unit"]], preserve_index=False).cast(schema),
                   out_dir / meds.code_metadata_filepath)
    pq.write_table(pa.Table.from_pandas(splits, preserve_index=False).cast(meds.SubjectSplitSchema.schema()),
                   out_dir / meds.subject_splits_filepath)
    summary = {"subjects": len(uuids), "events": int(n_events), "codes": int(len(codes))}
    (out_dir / meds.dataset_metadata_filepath).write_text(json.dumps({
        "dataset_name": "Synthea", "etl_name": "modlehr.data.adapters.synthea", "etl_version": __version__,
        "meds_version": meds.__version__, "created_at": dt.datetime.now().isoformat(timespec="seconds"),
    }, indent=2))
    return summary
