"""The Synthea adapter, on a few rows in the layout Synthea's CSV export writes."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from modlehr.data.adapters.synthea import convert
from modlehr.data.meds_io import open_meds


def write_export(base: Path) -> None:
    """Two patients, as Synthea writes them under --exporter.baseDirectory: base/csv/*.csv."""
    csv = base / "csv"
    csv.mkdir(parents=True)
    pd.DataFrame({"Id": ["a", "b"], "BIRTHDATE": ["1950-01-01", "1960-06-01"], "DEATHDATE": ["2015-03-01", None],
                  "GENDER": ["F", "M"]}).to_csv(csv / "patients.csv", index=False)
    pd.DataFrame({"START": ["2000-01-01", "2010-05-05"], "STOP": [None, None], "PATIENT": ["a", "b"],
                  "CODE": ["44054006", "38341003"], "DESCRIPTION": ["Diabetes mellitus type 2 (disorder)",
                                                                  "Hypertension (disorder)"]}).to_csv(csv / "conditions.csv", index=False)
    pd.DataFrame({"DATE": ["2001-01-01T08:00:00Z", "2011-01-01T08:00:00Z"], "PATIENT": ["a", "b"],
                  "CODE": ["4548-4", "72166-2"], "DESCRIPTION": ["Hemoglobin A1c", "Tobacco smoking status"],
                  "VALUE": ["6.1", "Never smoked tobacco (finding)"], "UNITS": ["%", None],
                  "TYPE": ["numeric", "text"]}).to_csv(csv / "observations.csv", index=False)


@pytest.mark.parametrize("where", ["base", "csv"])
def test_convert_reads_the_folder_synthea_writes(tmp_path, where):
    """`--raw-dir` may be the base directory given to Synthea (the README's command) or its csv/ folder."""
    write_export(tmp_path / "raw")
    summary = convert(tmp_path / "raw" if where == "base" else tmp_path / "raw" / "csv", tmp_path / "meds")
    assert summary["subjects"] == 2
    ds = open_meds(tmp_path / "meds", validate=True)
    events = pd.concat([ds.read_shard(s) for s in ds.shards()])
    assert {"MEDS_BIRTH", "MEDS_DEATH", "GENDER//F", "SNOMEDCT//44054006", "LOINC//4548-4"} <= set(events["code"])
    assert events.loc[events["code"] == "LOINC//4548-4", "numeric_value"].iloc[0] == pytest.approx(6.1)
    assert events.loc[events["code"] == "LOINC//72166-2", "text_value"].iloc[0] == "Never smoked tobacco (finding)"


def test_convert_says_what_it_looked_for(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="patients.csv"):
        convert(tmp_path / "empty", tmp_path / "meds")
