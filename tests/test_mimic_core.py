"""The MIMIC-IV core adapter: the three hosp tables that carry the diagnosis history → MEDS."""

from __future__ import annotations

import copy

import meds
import pandas as pd
import pytest

from modlehr.config import EventRule
from modlehr.data.adapters.mimic_iv_core import convert
from modlehr.data.meds_io import open_meds
from modlehr.prepare.dataset import open_split
from modlehr.prepare.pipeline import run_prepare

QUIET = lambda *_: None  # noqa: E731


def write_raw(raw):
    """Three patients in MIMIC-IV's own column layout: one dies after discharge, one dies in hospital,
    one has no admissions."""
    hosp = raw / "hosp"
    hosp.mkdir(parents=True)
    pd.DataFrame([
        {"subject_id": 1, "gender": "F", "anchor_age": 50, "anchor_year": 2150, "anchor_year_group": "2014 - 2016",
         "dod": "2155-03-01"},
        {"subject_id": 2, "gender": "M", "anchor_age": 60, "anchor_year": 2160, "anchor_year_group": "2014 - 2016",
         "dod": "2161-02-02"},
        {"subject_id": 3, "gender": "M", "anchor_age": 40, "anchor_year": 2140, "anchor_year_group": "2014 - 2016",
         "dod": None},
    ]).to_csv(hosp / "patients.csv.gz", index=False)
    pd.DataFrame([
        {"subject_id": 1, "hadm_id": 11, "admittime": "2151-01-01 08:00:00", "dischtime": "2151-01-05 12:00:00",
         "deathtime": None, "admission_type": "URGENT", "admission_location": "TRANSFER FROM HOSPITAL",
         "edregtime": "2151-01-01 02:00:00"},
        {"subject_id": 1, "hadm_id": 12, "admittime": "2153-06-01 08:00:00", "dischtime": "2153-06-03 12:00:00",
         "deathtime": None, "admission_type": "EW EMER.", "admission_location": None, "edregtime": None},
        {"subject_id": 2, "hadm_id": 21, "admittime": "2161-02-01 08:00:00", "dischtime": "2161-02-02 12:00:00",
         "deathtime": "2161-02-02 12:00:00", "admission_type": "ELECTIVE", "admission_location": "PHYSICIAN REFERRAL",
         "edregtime": None},
    ]).to_csv(hosp / "admissions.csv.gz", index=False)
    pd.DataFrame([
        {"subject_id": 1, "hadm_id": 11, "seq_num": 1, "icd_code": "4280", "icd_version": 9},
        {"subject_id": 1, "hadm_id": 12, "seq_num": 1, "icd_code": "E119", "icd_version": 10},
        {"subject_id": 2, "hadm_id": 21, "seq_num": 1, "icd_code": "I10", "icd_version": 10},
    ]).to_csv(hosp / "diagnoses_icd.csv.gz", index=False)
    pd.DataFrame([{"icd_code": "4280", "icd_version": 9, "long_title": "Congestive heart failure, unspecified"}
                  ]).to_csv(hosp / "d_icd_diagnoses.csv.gz", index=False)
    return raw


def test_core_adapter_writes_valid_meds(tmp_path):
    out = tmp_path / "core_meds"
    summary = convert(write_raw(tmp_path / "raw"), out, held_out=0.0, tuning=0.0)
    assert summary["subjects"] == 3 and summary["events_without_a_time"] == 0

    ds = open_meds(out, validate=True)
    events = pd.concat([ds.read_shard(s) for s in ds.shards()])
    codes = dict(zip(ds.codes["code"], ds.codes["description"]))
    assert codes["DIAGNOSIS//ICD//9//4280"] == "Congestive heart failure, unspecified"

    one = events[events["subject_id"] == 1].set_index("code")
    # birth is anchor_year - anchor_age, at the start of that year, as in the official ETL
    assert one.loc[meds.birth_code, "time"] == pd.Timestamp("2100-01-01")
    assert one.loc[meds.death_code, "time"] == pd.Timestamp("2155-03-01")
    assert pd.isna(one.loc["GENDER//F", "time"])
    # a diagnosis is timed at the discharge of its admission, not at its admission
    assert one.loc["DIAGNOSIS//ICD//9//4280", "time"] == pd.Timestamp("2151-01-05 12:00:00")
    assert one.loc["ED_REGISTRATION", "time"] == pd.Timestamp("2151-01-01 02:00:00")
    assert "HOSPITAL_ADMISSION//EW EMER.//UNKNOWN" in set(one.index)  # a missing location still makes a code

    assert set(events[events["subject_id"] == 3]["code"]) == {meds.birth_code, "GENDER//M"}

    # an in-hospital death takes the admission's deathtime, not midnight of dod, so the diagnoses
    # of the admission it ended do not come after it
    two = events[events["subject_id"] == 2].set_index("code")
    assert two.loc[meds.death_code, "time"] == pd.Timestamp("2161-02-02 12:00:00")
    assert two.loc["DIAGNOSIS//ICD//10//I10", "time"] <= two.loc[meds.death_code, "time"]


def test_diagnoses_can_be_timed_at_admission(tmp_path):
    """diagnosis_time="admission" moves a stay's diagnoses to its admission and nothing else."""
    raw = write_raw(tmp_path / "raw")
    convert(raw, tmp_path / "admit", held_out=0.0, tuning=0.0, diagnosis_time="admission")
    ds = open_meds(tmp_path / "admit", validate=True)
    one = pd.concat([ds.read_shard(s) for s in ds.shards()]).query("subject_id == 1").set_index("code")
    assert one.loc["DIAGNOSIS//ICD//9//4280", "time"] == pd.Timestamp("2151-01-01 08:00:00")
    assert one.loc["DIAGNOSIS//ICD//10//E119", "time"] == pd.Timestamp("2153-06-01 08:00:00")
    assert one.loc[meds.death_code, "time"] == pd.Timestamp("2155-03-01")
    with pytest.raises(ValueError, match="diagnosis_time"):
        convert(raw, tmp_path / "bad", diagnosis_time="midnight")


def test_core_adapter_needs_the_three_tables(tmp_path):
    raw = write_raw(tmp_path / "raw")
    (raw / "hosp/diagnoses_icd.csv.gz").unlink()
    with pytest.raises(FileNotFoundError, match="diagnoses_icd"):
        convert(raw, tmp_path / "core_meds")


def test_core_dataset_prepares(tmp_path, tiny_config):
    out = tmp_path / "core_meds"
    convert(write_raw(tmp_path / "raw"), out, held_out=0.0, tuning=0.0)
    cfg = copy.deepcopy(tiny_config)
    cfg.data.meds_dir, cfg.data.prepared_dir = str(out), str(tmp_path / "prepared")
    cfg.data.value_mode, cfg.data.min_subjects_per_token = "none", 1
    cfg.data.events = [EventRule(name="sex", role="static", prefix="GENDER//"),
                       EventRule(name="death", role="target", codes=["MEDS_DEATH"]),
                       EventRule(name="diagnoses", role="target", prefix="DIAGNOSIS//ICD//"),
                       EventRule(name="admissions", role="context", prefix="HOSPITAL_ADMISSION//")]
    run_prepare(cfg.data, log=QUIET)
    split = open_split(cfg.data.prepared_dir, "train")
    assert len(split.offsets) - 1 == 3


def test_core_adapter_adds_outpatient_measurements(tmp_path):
    """With hosp/omr.csv.gz present, its rows come in as the official ETL writes them (code = result_name,
    raw value as text), with numeric_value set when the value is a number."""
    raw = write_raw(tmp_path / "raw")
    pd.DataFrame([
        {"subject_id": 1, "chartdate": "2152-03-01", "seq_num": 1, "result_name": "BMI (kg/m2)", "result_value": "27.5"},
        {"subject_id": 1, "chartdate": "2152-03-01", "seq_num": 1, "result_name": "Blood Pressure", "result_value": "120/80"},
        {"subject_id": 2, "chartdate": "2160-06-01", "seq_num": 1, "result_name": "Weight (Lbs)", "result_value": "180"},
    ]).to_csv(raw / "hosp/omr.csv.gz", index=False)
    out = tmp_path / "core_meds"
    summary = convert(raw, out, held_out=0.0, tuning=0.0)
    assert summary["omr_rows"] == 3

    ds = open_meds(out, validate=True)
    events = pd.concat([ds.read_shard(s) for s in ds.shards()]).set_index("code")
    bmi, bp = events.loc["BMI (kg/m2)"], events.loc["Blood Pressure"]
    assert bmi["numeric_value"] == 27.5 and bmi["time"] == pd.Timestamp("2152-03-01")
    assert pd.isna(bp["numeric_value"]) and bp["text_value"] == "120/80"
    assert pd.isna(events.loc["DIAGNOSIS//ICD//10//I10", "numeric_value"])  # other events keep no value
