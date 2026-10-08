"""Read a MEDS dataset (meds 0.4): sharded event tables plus codes, splits and dataset metadata.

The core of modLEHR reads nothing but MEDS. Differences between data sources live in the
adapters that write MEDS, not here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import meds
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq


@dataclass
class MedsDataset:
    root: Path
    codes: pd.DataFrame
    splits: pd.DataFrame
    metadata: dict

    def shards(self) -> list[Path]:
        return sorted((self.root / meds.data_subdirectory).rglob("*.parquet"))

    def read_shard(self, path: Path, codes: set[str] | None = None) -> pd.DataFrame:
        """One shard as a DataFrame, optionally keeping only the given codes."""
        table = pq.read_table(path)
        if codes is not None:
            table = table.filter(pc.is_in(table["code"], value_set=_as_array(codes)))
        frame = table.to_pandas()
        for col in ("numeric_value", "text_value"):
            if col not in frame:
                frame[col] = None
        return frame[["subject_id", "time", "code", "numeric_value", "text_value"]]


def _as_array(values: set[str]):
    import pyarrow as pa

    return pa.array(sorted(values), type=pa.string())


def open_meds(root: str | Path, validate: bool = False) -> MedsDataset:
    root = Path(root)
    if not (root / meds.data_subdirectory).is_dir():
        raise FileNotFoundError(f"{root} has no '{meds.data_subdirectory}/' directory; is it a MEDS dataset?")
    codes_path = root / meds.code_metadata_filepath
    codes = pq.read_table(codes_path).to_pandas() if codes_path.exists() else pd.DataFrame({"code": [], "description": []})
    if "description" not in codes:
        codes["description"] = None
    splits_path = root / meds.subject_splits_filepath
    splits = pq.read_table(splits_path).to_pandas() if splits_path.exists() else pd.DataFrame({"subject_id": [], "split": []})
    meta_path = root / meds.dataset_metadata_filepath
    metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    ds = MedsDataset(root=root, codes=codes, splits=splits, metadata=metadata)
    if validate:
        validate_meds(ds)
    return ds


def validate_meds(ds: MedsDataset) -> None:
    """Check every data shard, the codes table and the splits against the official meds schemas."""
    for shard in ds.shards():
        meds.DataSchema.validate(pq.read_table(shard))
    meds.CodeMetadataSchema.validate(pq.read_table(ds.root / meds.code_metadata_filepath))
    meds.SubjectSplitSchema.validate(pq.read_table(ds.root / meds.subject_splits_filepath))
