"""MIMIC-IV core tables → MEDS: patients, admissions and diagnoses only.

The official MIMIC_IV_MEDS ETL (scripts/run_mimic_iv.sh) reads every table in the release and is
what a full run should use. This adapter reads the three tables that carry the diagnosis history
(`hosp/patients.csv.gz`, `hosp/admissions.csv.gz`, `hosp/diagnoses_icd.csv.gz`) and writes the same
code families, so the rules in configs/data/mimic_iv.yaml keep matching. It exists for two reasons:
the diagnosis sequence is the part Delphi-2M models (sex, disease codes, death and nothing else),
and those three tables are a small download next to labevents and chartevents.

What it does not produce: LAB//, MEDICATION//, PROCEDURE// and ICU_ADMISSION// events. A config
written for this dataset should leave those rules out.

When `hosp/omr.csv.gz` is there too (the outpatient measurements: BMI, weight, height, blood
pressure), its rows are added the way the official ETL writes them: the code is `result_name`
(e.g. `BMI (kg/m2)`), the time `chartdate`, the raw value `text_value`; and `numeric_value` holds
the value when it is a number, so a measurement rule can use it (blood pressure, "120/80", stays
text only). These are the closest MIMIC-IV has to Delphi-2M's lifestyle inputs (BMI, smoking,
alcohol); smoking appears only as diagnosis codes.

`extra` adds inputs that are not diagnoses, from tables of the same release (all in hosp/, so they are
there before labevents and chartevents are):
  procedures    procedures_icd.csv.gz → PROCEDURE//ICD//<9|10>//<icd_code>, at its chartdate
  medications   prescriptions.csv.gz  → MEDICATION//START//<DRUG>, at its starttime (the drug name
                upper-cased, so "Insulin" and "INSULIN" are one code)
They are model inputs only (role context in the config); configs/data/mimic_iv_core_inputs.yaml uses them.

    subject_id     MIMIC-IV's own subject_id
    time           event datetime; null for sex
    code           MEDS_BIRTH, MEDS_DEATH, GENDER//F|M,
                   DIAGNOSIS//ICD//<9|10>//<icd_code>   at hospital discharge (or admission, below)
                   HOSPITAL_ADMISSION//<type>//<location>, ED_REGISTRATION
                   PROCEDURE//ICD//<9|10>//<icd_code>, MEDICATION//START//<DRUG>   with `extra`
    numeric_value  the omr value when it is a number; otherwise empty
    text_value     the omr value as written

Death is timed as in the official ETL: the earliest in-hospital `deathtime` from admissions when
there is one, else the date-only `dod` from patients. `dod` alone would put an in-hospital death at
midnight, before the diagnoses of the admission it ended, which are timed at discharge.

MIMIC-IV records a stay's diagnoses without a time of their own; the official ETL, and this adapter,
time them at the stay's discharge, when they are known for certain. `diagnosis_time="admission"`
(`modlehr convert mimic-iv-core --diagnosis-time admission`) times them at the admission instead, as if
each were there when the patient arrived: a model then cannot learn that an admission is followed within
days by its own diagnoses, which says that the patient is in hospital rather than what will happen later.

Ages come from MIMIC's date shift: a subject's birth is `anchor_year - anchor_age`, taken at the
start of that year, which is what the official ETL does. Ages are therefore accurate to about a
year, and the shift keeps each subject's own intervals exact.
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

TABLES = ("hosp/patients.csv.gz", "hosp/admissions.csv.gz", "hosp/diagnoses_icd.csv.gz")
OMR = "hosp/omr.csv.gz"  # optional
EXTRA = {"procedures": "hosp/procedures_icd.csv.gz", "medications": "hosp/prescriptions.csv.gz"}
SUBJECTS_PER_SHARD = 50_000


def _time(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").astype("datetime64[us]")


def _patient_events(raw: Path, deathtime: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    p = pd.read_csv(raw / "hosp/patients.csv.gz",
                    usecols=["subject_id", "gender", "anchor_age", "anchor_year", "dod"])
    birth = pd.to_datetime(dict(year=p["anchor_year"] - p["anchor_age"], month=1, day=1)).astype("datetime64[us]")
    parts = [
        pd.DataFrame({"subject_id": p["subject_id"], "time": birth, "code": meds.birth_code}),
        pd.DataFrame({"subject_id": p["subject_id"], "time": pd.NaT, "code": "GENDER//" + p["gender"]}),
    ]
    death = p["subject_id"].map(deathtime).fillna(_time(p["dod"]))  # in-hospital time first, as the official ETL
    d = death.notna()
    parts.append(pd.DataFrame({"subject_id": p["subject_id"][d], "time": death[d], "code": meds.death_code}))
    return pd.concat(parts, ignore_index=True), p["subject_id"]


def _admission_events(raw: Path) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    a = pd.read_csv(raw / "hosp/admissions.csv.gz",
                    usecols=["subject_id", "hadm_id", "admittime", "dischtime", "deathtime", "edregtime",
                             "admission_type", "admission_location"])
    location = a["admission_location"].fillna("UNKNOWN")
    parts = [pd.DataFrame({"subject_id": a["subject_id"], "time": _time(a["admittime"]),
                           "code": "HOSPITAL_ADMISSION//" + a["admission_type"] + "//" + location})]
    ed = a.dropna(subset=["edregtime"])
    parts.append(pd.DataFrame({"subject_id": ed["subject_id"], "time": _time(ed["edregtime"]),
                               "code": "ED_REGISTRATION"}))
    discharge = a.set_index("hadm_id")["dischtime"].pipe(_time)
    admission = a.set_index("hadm_id")["admittime"].pipe(_time)
    deathtime = _time(a["deathtime"]).groupby(a["subject_id"]).min().dropna()
    return pd.concat(parts, ignore_index=True), discharge, admission, deathtime


def _diagnosis_events(raw: Path, discharge: pd.Series) -> pd.DataFrame:
    d = pd.read_csv(raw / "hosp/diagnoses_icd.csv.gz",
                    usecols=["subject_id", "hadm_id", "icd_code", "icd_version"], dtype={"icd_code": str})
    code = "DIAGNOSIS//ICD//" + d["icd_version"].astype(str) + "//" + d["icd_code"].str.strip()
    return pd.DataFrame({"subject_id": d["subject_id"], "time": d["hadm_id"].map(discharge), "code": code})


def _omr_events(raw: Path) -> pd.DataFrame:
    o = pd.read_csv(raw / OMR, usecols=["subject_id", "chartdate", "result_name", "result_value"],
                    dtype={"result_name": str, "result_value": str})
    value = o["result_value"].str.strip()
    return pd.DataFrame({"subject_id": o["subject_id"], "time": _time(o["chartdate"]), "code": o["result_name"].str.strip(),
                         "numeric_value": pd.to_numeric(value, errors="coerce"), "text_value": value})


def _procedure_events(raw: Path) -> pd.DataFrame:
    p = pd.read_csv(raw / EXTRA["procedures"], usecols=["subject_id", "chartdate", "icd_code", "icd_version"],
                    dtype={"icd_code": str})
    code = "PROCEDURE//ICD//" + p["icd_version"].astype(str) + "//" + p["icd_code"].str.strip()
    return pd.DataFrame({"subject_id": p["subject_id"], "time": _time(p["chartdate"]), "code": code})


def _medication_events(raw: Path) -> pd.DataFrame:
    m = pd.read_csv(raw / EXTRA["medications"], usecols=["subject_id", "starttime", "drug"], dtype={"drug": str})
    m = m.dropna(subset=["drug"])
    return pd.DataFrame({"subject_id": m["subject_id"], "time": _time(m["starttime"]),
                         "code": "MEDICATION//START//" + m["drug"].str.strip().str.upper()})


def _code_metadata(raw: Path, codes: pd.Series) -> pd.DataFrame:
    titles = [pd.DataFrame({"code": [], "description": []})]
    for kind, family in (("diagnoses", "DIAGNOSIS"), ("procedures", "PROCEDURE")):
        path = raw / f"hosp/d_icd_{kind}.csv.gz"
        if path.exists():
            t = pd.read_csv(path, dtype={"icd_code": str})
            titles.append(pd.DataFrame({"code": f"{family}//ICD//" + t["icd_version"].astype(str) + "//"
                                        + t["icd_code"].str.strip(), "description": t["long_title"]}))
    titles = pd.concat(titles, ignore_index=True)
    known = pd.DataFrame({"code": [meds.birth_code, meds.death_code, "GENDER//F", "GENDER//M", "ED_REGISTRATION"],
                          "description": ["Birth", "Death", "Sex: female", "Sex: male", "Emergency department registration"]})
    out = pd.concat([titles, known], ignore_index=True)
    rest = pd.DataFrame({"code": sorted(set(codes) - set(out["code"])), "description": None})
    return pd.concat([out, rest], ignore_index=True).drop_duplicates("code").reset_index(drop=True)


def _write(df: pd.DataFrame, path: Path) -> None:
    for column, empty in (("numeric_value", np.nan), ("text_value", None)):
        if column not in df:
            df = df.assign(**{column: empty})
    df = df[["subject_id", "time", "code", "numeric_value", "text_value"]]
    table = pa.Table.from_pandas(df, preserve_index=False).cast(meds.DataSchema.schema())
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


DIAGNOSIS_TIMES = ("discharge", "admission")


def convert(raw_dir: str | Path, out_dir: str | Path, held_out: float = 0.2, tuning: float = 0.1,
            seed: int = 1413, diagnosis_time: str = "discharge", extra: tuple[str, ...] = ()) -> dict:
    """diagnosis_time: when a stay's diagnoses are timed, at its discharge (the official ETL) or admission.
    extra: inputs to add besides diagnoses, any of EXTRA ("procedures", "medications")."""
    if diagnosis_time not in DIAGNOSIS_TIMES:
        raise ValueError(f"diagnosis_time must be one of {DIAGNOSIS_TIMES}, not {diagnosis_time!r}")
    unknown = sorted(set(extra) - set(EXTRA))
    if unknown:
        raise ValueError(f"extra must be among {sorted(EXTRA)}, not {unknown}")
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    missing = [t for t in TABLES + tuple(EXTRA[e] for e in extra) if not (raw_dir / t).exists()]
    if missing:
        raise FileNotFoundError(f"{raw_dir} is missing {', '.join(missing)}")

    admissions, discharge, admission, deathtime = _admission_events(raw_dir)
    patients, subject_ids = _patient_events(raw_dir, deathtime)
    diagnoses = _diagnosis_events(raw_dir, discharge if diagnosis_time == "discharge" else admission)
    parts = [patients, admissions, diagnoses]
    n_omr = 0
    if (raw_dir / OMR).exists():
        parts.append(_omr_events(raw_dir))
        n_omr = len(parts[-1])
    extra_rows = {}
    for name, read in (("procedures", _procedure_events), ("medications", _medication_events)):
        if name in extra:
            parts.append(read(raw_dir))
            extra_rows[name] = len(parts[-1])
    events = pd.concat(parts, ignore_index=True)
    dropped = int(events["time"].isna().sum() - (events["code"].str.startswith("GENDER//")).sum())
    events = events[events["time"].notna() | events["code"].str.startswith("GENDER//")]

    subjects = np.sort(subject_ids.unique())
    rng = np.random.default_rng(seed)
    is_held = rng.random(len(subjects)) < held_out
    is_tune = (~is_held) & (np.random.default_rng(seed + 1).random(len(subjects)) < tuning)
    split = np.where(is_held, meds.held_out_split, np.where(is_tune, meds.tuning_split, meds.train_split))
    splits = pd.DataFrame({"subject_id": subjects.astype("int64"), "split": split})
    shard = pd.Series(np.arange(len(subjects)) // SUBJECTS_PER_SHARD, index=subjects)

    events = events.merge(splits, on="subject_id")
    events["shard"] = events["subject_id"].map(shard)
    events = events.sort_values(["subject_id", "time"], na_position="first", kind="stable")
    for (name, i), group in events.groupby(["split", "shard"]):
        _write(group.drop(columns=["split", "shard"]).reset_index(drop=True),
               out_dir / meds.data_subdirectory / name / f"{i}.parquet")

    codes = _code_metadata(raw_dir, events["code"]).assign(parent_codes=None)
    schema = pa.schema(list(meds.CodeMetadataSchema.schema()) + [pa.field("unit", pa.string())])
    codes["unit"] = None
    (out_dir / "metadata").mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(codes[["code", "description", "parent_codes", "unit"]],
                                        preserve_index=False).cast(schema),
                   out_dir / meds.code_metadata_filepath)
    pq.write_table(pa.Table.from_pandas(splits, preserve_index=False).cast(meds.SubjectSplitSchema.schema()),
                   out_dir / meds.subject_splits_filepath)
    (out_dir / meds.dataset_metadata_filepath).write_text(json.dumps({
        "dataset_name": "MIMIC-IV 3.1 core (patients, admissions, diagnoses" + "".join(f", {e}" for e in extra) + ")",
        "etl_name": "modlehr.data.adapters.mimic_iv_core", "etl_version": __version__, "diagnosis_time": diagnosis_time,
        "extra": list(extra),
        "meds_version": meds.__version__, "created_at": dt.datetime.now().isoformat(timespec="seconds"),
    }, indent=2))
    return {"subjects": len(subjects), "events": int(len(events)), "codes": int(len(codes)),
            "omr_rows": n_omr, **{f"{k}_rows": v for k, v in extra_rows.items()}, "events_without_a_time": dropped}
