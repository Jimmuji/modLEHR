"""MIMIC-IV-Note → events merged into an existing MIMIC-IV MEDS dataset.

MIMIC-IV-Note is a separate PhysioNet module (discharge summaries and radiology reports), and the
MIMIC_IV_MEDS ETL does not read it. This adapter adds each note as one event to the MEDS output of
that ETL:

    subject_id   the MIMIC subject_id (the ETL keeps it)
    time         the note's charttime
    code         NOTE//DISCHARGE or NOTE//RADIOLOGY
    text_value   the note text

MEDS keeps all of a subject's events in one shard, so notes are merged into the shard that already
holds the subject, and every shard is rewritten to `out_dir` sorted by subject and time. Notes of
subjects absent from the MEDS data are dropped. Metadata is copied, with the two note codes added.

    modlehr convert mimic-iv-note --raw-dir ~/mimic_dl/physionet.org/files/mimic-iv-note/2.2/note \\
        --meds-dir data/mimic_iv_meds --out data/mimic_iv_meds_notes
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import meds
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

KINDS = {"discharge": ("NOTE//DISCHARGE", "Discharge summary"),
         "radiology": ("NOTE//RADIOLOGY", "Radiology report")}


def _read_notes(raw_dir: Path, kinds: tuple[str, ...]) -> pd.DataFrame:
    frames = []
    for kind in kinds:
        path = next((p for p in (raw_dir / f"{kind}.csv.gz", raw_dir / f"{kind}.csv") if p.exists()), None)
        if path is None:
            raise FileNotFoundError(f"{raw_dir} has no {kind}.csv.gz")
        notes = pd.read_csv(path, usecols=["subject_id", "charttime", "text"])
        frames.append(pd.DataFrame({"subject_id": notes["subject_id"].astype("int64"),
                                    "time": pd.to_datetime(notes["charttime"], format="ISO8601").astype("datetime64[us]"),
                                    "code": KINDS[kind][0], "numeric_value": np.float32("nan"),
                                    "text_value": notes["text"].astype(str)}))
    return pd.concat(frames, ignore_index=True)


def convert(raw_dir: str | Path, meds_dir: str | Path, out_dir: str | Path,
            kinds: tuple[str, ...] = ("discharge", "radiology")) -> dict:
    raw_dir, meds_dir, out_dir = Path(raw_dir), Path(meds_dir), Path(out_dir)
    if out_dir.resolve() == meds_dir.resolve():
        raise ValueError("write the merged dataset to a new directory, not over the input")
    notes = _read_notes(raw_dir, kinds)
    by_subject = notes.groupby("subject_id")
    noted = set(by_subject.groups)

    shards = sorted((meds_dir / meds.data_subdirectory).rglob("*.parquet"))
    added = 0
    for shard in shards:
        table = pq.read_table(shard)
        frame = table.to_pandas()
        subjects = [s for s in frame["subject_id"].unique() if s in noted]
        if subjects:
            extra = pd.concat([by_subject.get_group(s) for s in subjects], ignore_index=True)
            frame = pd.concat([frame, extra], ignore_index=True)
            frame = frame.sort_values(["subject_id", "time"], na_position="first", kind="stable")
            added += len(extra)
        target = out_dir / shard.relative_to(meds_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pandas(frame, preserve_index=False, schema=table.schema), target)

    (out_dir / "metadata").mkdir(parents=True, exist_ok=True)
    for item in (meds_dir / "metadata").iterdir():
        if item.name not in ("codes.parquet", "dataset.json"):
            (shutil.copytree if item.is_dir() else shutil.copy2)(item, out_dir / "metadata" / item.name)
    codes = pq.read_table(meds_dir / meds.code_metadata_filepath)
    present = set(codes["code"].to_pylist())
    new_kinds = [k for k in kinds if KINDS[k][0] not in present]
    extra_codes = pd.DataFrame({"code": [KINDS[k][0] for k in new_kinds], "description": [KINDS[k][1] for k in new_kinds]})
    merged_codes = pa.concat_tables([codes, _pad(pa.Table.from_pandas(extra_codes, preserve_index=False), codes.schema)])
    pq.write_table(merged_codes, out_dir / meds.code_metadata_filepath)
    meta_path = meds_dir / meds.dataset_metadata_filepath
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta["dataset_name"] = f"{meta.get('dataset_name', 'MIMIC-IV')} with MIMIC-IV-Note"
    (out_dir / meds.dataset_metadata_filepath).write_text(json.dumps(meta, indent=2))
    return {"notes_read": int(len(notes)), "notes_added": int(added), "shards": len(shards)}


def _pad(table: pa.Table, schema: pa.Schema) -> pa.Table:
    """Give `table` every column of `schema`, nulls where it has none, in schema order."""
    cols = [table[f.name].cast(f.type) if f.name in table.column_names else pa.nulls(len(table), f.type) for f in schema]
    return pa.Table.from_arrays(cols, schema=schema)
