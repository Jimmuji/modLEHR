# Data dictionary

Back to the [README](../README.md).

Every variable of the standardized data, in the order the data is made:

| Layer | What it is | Made by | Where |
|---|---|---|---|
| 1. MEDS | One row per event, the same five columns for every source | a converter (`modlehr convert`, or the official MIMIC-IV ETL) | `data/<source>_meds/` |
| 2. Model input | One token per kept event, with its age, as numpy arrays | `modlehr prepare` | `runs/prepared/<name>/` |
| 3. Labels | One row per prediction: patient, time, yes or no | ACES configs and task scripts | `data/<source>_tasks/<task>/` |

Synthea is public at all three layers on [Hugging Face](https://huggingface.co/datasets/Jimmu1215/modlehr-synthea).
MIMIC-IV is credentialed: anyone with PhysioNet access rebuilds the same three layers with `scripts/run_mimic_iv.sh`,
and only summary numbers are shared.

## 1. MEDS

The [Medical Event Data Standard](https://github.com/Medical-Event-Data-Standard/meds), version 0.4.

### Files

| File | Contents |
|---|---|
| `data/<split>/<n>.parquet` | events; `<split>` is `train`, `tuning` or `held_out`, and a patient's rows are all in one file |
| `metadata/codes.parquet` | one row per code: `code`, `description`, `parent_codes`, and from some converters `unit` (Synthea) or `code_template` (MIMIC-IV) |
| `metadata/subject_splits.parquet` | `subject_id`, `split` |
| `metadata/dataset.json` | dataset name, converter name and version, MEDS version |

### Event columns

| Column | Type | Meaning |
|---|---|---|
| `subject_id` | int64 | the patient |
| `time` | timestamp (µs) | when it happened; empty for facts without a time, such as sex |
| `code` | string | what happened: the source's coding system, `//`, then the source's code |
| `numeric_value` | float32 | a measured number (a lab result, a vital sign); empty otherwise |
| `text_value` | string | a text answer (smoking status, a note); empty otherwise |

The meaning of every code is its `description` in `metadata/codes.parquet`.

### Codes in Synthea

Converted by `modlehr convert synthea` from three files of Synthea's CSV export. 1,229 codes.

| Code | From | Time |
|---|---|---|
| `GENDER//F`, `GENDER//M` | `patients.csv` GENDER | none |
| `MEDS_BIRTH` | `patients.csv` BIRTHDATE | birth |
| `MEDS_DEATH` | `patients.csv` DEATHDATE | death |
| `SNOMEDCT//<code>` | `conditions.csv` CODE (a SNOMED CT condition) | START |
| `LOINC//<code>` | `observations.csv` CODE; a numeric VALUE goes to `numeric_value`, any other to `text_value` | DATE |

### Codes in MIMIC-IV

Written by the official ETL, [MIMIC_IV_MEDS](https://github.com/Medical-Event-Data-Standard/MIMIC_IV_MEDS). The
templates below come from its event configuration (`code_template` in `metadata/codes.parquet`). Item IDs are
MIMIC-IV's own: `d_labitems` names the `LAB//RESULT` items and `d_items` the ICU ones, described in the
[MIMIC-IV documentation](https://mimic.mit.edu/docs/iv/).

| Code | From | Time |
|---|---|---|
| `GENDER//F`, `GENDER//M` | `hosp/patients` gender | none |
| `MEDS_BIRTH` | `hosp/patients` anchor_year minus anchor_age (MIMIC-IV gives no birth date) | birth year |
| `MEDS_DEATH` | `hosp/admissions` deathtime, else `hosp/patients` dod | death |
| `HOSPITAL_ADMISSION//<type>//<location>` | `hosp/admissions` | admittime |
| `HOSPITAL_DISCHARGE//<location>` | `hosp/admissions` | dischtime |
| `ED_REGISTRATION`, `ED_OUT` | `hosp/admissions` | edregtime, edouttime |
| `ICU_ADMISSION//<unit>` | `icu/icustays` | intime |
| `DIAGNOSIS//ICD//<9 or 10>//<code>` | `hosp/diagnoses_icd` | the admission's discharge (the table has no time) |
| `PROCEDURE//ICD//<9 or 10>//<code>` | `hosp/procedures_icd` | chartdate |
| `LAB//RESULT//<itemid>//<unit>` | `hosp/labevents`, value in `numeric_value` | charttime |
| `LAB//<itemid>//<unit>` | `icu/chartevents` (vital signs, scores), value in `numeric_value` | charttime |
| `MEDICATION//START//<drug>//<NDC>` | `hosp/prescriptions` | starttime |

The ETL writes some more families (`MEDICATION//STOP`, administrations from `emar`, `LAB//SPECIMEN_COLLECTED`, `DRG`,
`TRANSFER_TO`, `PROCEDURE//START` and `//END`, infusions, outputs, `HCPCS`). The data configs leave them out.

## 2. Model input

Written by `modlehr prepare --config configs/data/<name>.yaml` into `runs/prepared/<name>/`. Statistics (bins,
which tokens are kept) come from the training split only.
[docs/reading_prepared_data.md](reading_prepared_data.md) shows how to load it.

### Files

| File | Contents |
|---|---|
| `vocab.json` | `tokens`: token id to name, id 0 is padding; `targets`: target id to name |
| `prepare.json` | `data_config` (the rules used), `meds_fingerprint`, `bins` (cut points per measurement code), `rules` (codes, tokens and events per rule), `summary` (patients and events per split), `target_counts` and `exposure_years` (training split), `vocab_size`, `n_targets` |
| `<split>/<array>.npy` | the arrays below |
| `tasks/<task>/<split>.npz` | the samples of one task (below) |

### Tokens

| Token | Meaning |
|---|---|
| the MEDS code, e.g. `ICU_ADMISSION//<unit>` | an event without a value |
| `<code>//Q1`, `//Q2`, `//Q3` | a measurement in the lowest, middle or highest third of its training values |
| `<code>//<text>`, e.g. `LOINC//72166-2//Never smoked tobacco (finding)` | a categorical answer |
| `ICD10//<3 characters>`, e.g. `ICD10//E11` | MIMIC-IV diagnoses grouped to ICD-10 categories; ICD-9 codes are mapped through the CMS GEMs |
| `SNOMEDCT//<code>` | a Synthea condition, kept as is |
| `MEDS_DEATH` | death |

A token is kept if at least `min_subjects_per_token` training patients have it (100 on MIMIC-IV, 25 on Synthea).
Diagnoses and death are prediction targets; every target is also an input token.

### Arrays of a split

The events of patient `i` are rows `offsets[i]:offsets[i + 1]` of every event array, in time order.

| Array | Shape | Type | Meaning |
|---|---|---|---|
| `subject_ids` | patients | int64 | MEDS subject_id, ascending |
| `offsets` | patients + 1 | int64 | where each patient's events start and end |
| `tokens` | events | int32 | token id |
| `ages` | events | float32 | age in years at the event (sex at 0) |
| `values` | events | float32 | the number, when values are kept as numbers (`value_mode: continuous`); NaN otherwise |
| `target` | events | int32 | target id if the event is a target, else -1 |
| `next_target` | events | int32 | the next target after this event, -1 if none before follow-up ends |
| `dt` | events | float32 | years until that next target, or until follow-up ends |
| `loss_mask` | events | bool | true at the last event of each time point within follow-up |
| `follow_up_start`, `follow_up_end` | patients | float32 | ages between which the record is followed |

### Task samples

`tasks/<task>/<split>.npz`, written by `modlehr finetune --samples-only`, one entry per label row:

| Field | Meaning |
|---|---|
| `subject_id`, `prediction_time`, `label` | the label row |
| `index` | the patient's position in the split, -1 if they have no events |
| `n` | how many of the patient's events come by the prediction time; the input is events `0:n` |
| `strict` | true when an event just after the prediction time has the same float32 age, so the input stops before that age |

### Which events are kept

MIMIC-IV, `configs/data/mimic_iv.yaml` (9,576 tokens, 1,047 targets):

| Events | Codes | Kept |
|---|---|---|
| Sex | `GENDER//` | once |
| Death | `MEDS_DEATH` | once |
| Diagnoses | `DIAGNOSIS//ICD//` | first time each, as ICD-10 categories |
| Hospital labs | `LAB//RESULT//` | 3 value bins, once per code per quarter year |
| ICU charts | other `LAB//` | 3 value bins, once per code per quarter year |
| Medications | `MEDICATION//START//` | once per code per quarter year |
| Procedures | `PROCEDURE//` | once per code per quarter year |
| Admissions, ICU stays, ED visits | `HOSPITAL_ADMISSION//`, `ICU_ADMISSION//`, `ED_REGISTRATION` | every event |

Synthea, `configs/data/synthea.yaml` (209 tokens, 164 targets):

| Events | Codes | Kept |
|---|---|---|
| Sex | `GENDER//` | once |
| Death | `MEDS_DEATH` | once |
| Conditions | `SNOMEDCT//` whose description ends in "(disorder)" | first time each |
| Smoking status | `LOINC//72166-2` | first answer |
| 13 measurements: BMI, systolic and diastolic blood pressure, heart rate, body weight, total, LDL and HDL cholesterol, triglycerides, glucose, HbA1c, creatinine, eGFR | their `LOINC//` codes | 3 value bins, once per code per 2 years |

Everything else in MEDS is left out. For example, synthetic patient 296 has 1,830 MEDS rows and 220 tokens: 1,379
rows match no rule (1,308 other LOINC observations, 71 SNOMED CT findings that are not disorders), the birth row
only sets the ages, 42 condition rows of 20 conditions give 20 tokens, 51 smoking answers give 1, and 355
measurements of the 13 codes give 197.

## 3. Labels

The MEDS label format, one row per sample, in `<task>/<split>/<n>.parquet`:

| Column | Type | Meaning |
|---|---|---|
| `subject_id` | int64 | the patient |
| `prediction_time` | timestamp (µs) | when the prediction is made; the input is events at or before it |
| `boolean_value` | bool | the outcome |

### MIMIC-IV tasks

| Task | One sample per | Predicted at | Yes if | Source | Defined in |
|---|---|---|---|---|---|
| `admission_hospital_mortality` | hospital admission | admission | death before discharge | ETHOS-ARES | `configs/tasks/mimic_iv/` (ACES) |
| `admission_icu` | hospital admission, not straight into an ICU | admission | an ICU admission during the stay | ETHOS-ARES | `configs/tasks/mimic_iv/` (ACES) |
| `admission_long_stay_10d` | hospital admission | admission | discharge or death 10 days or more later | ETHOS-ARES | `configs/tasks/mimic_iv/` (ACES) |
| `day1_hospital_mortality` | adult admission | 23:59 of the admission day | death before discharge | PORTER | `scripts/porter_tasks.py` |
| `day1_long_stay_7d` | adult admission | 23:59 of the admission day | discharge 7 days or more after admission | PORTER | `scripts/porter_tasks.py` |
| `predischarge_readmission_30d` | adult admission | 23:59 of the day before discharge | another admission within 30 days of discharge | PORTER | `scripts/porter_tasks.py` |
| `readmission_30d` | discharge alive | discharge | another admission within 30 days | Pollet and McDermott (MC2) | `configs/tasks/mimic_iv/` (ACES) |
| `mortality_post_discharge_90d` | discharge alive | discharge | death within 90 days | Pollet and McDermott (MC10) | `configs/tasks/mimic_iv/` (ACES) |

Each file's header comment and the script's docstring give the full rules (what counts as a death, which samples
are left out).

### Synthea tasks

`scripts/synthea_tasks.py`. At each age 40, 45, ..., 80: does the condition (or death) first occur within the next
5 years? A patient gives a sample at an age when they are alive, have an earlier event, have not had the condition
yet, and are followed to the end of the 5 years (a later event or death). A death within the 5 years without the
condition is a no.

| Task | Codes | Held-out samples | Held-out yes |
|---|---|---|---|
| `ckd_stage3_5y` | `SNOMEDCT//433144002` | 8,885 | 205 |
| `death_5y` | `MEDS_DEATH` | 9,265 | 431 |
| `heart_failure_5y` | `SNOMEDCT//88805009`, `SNOMEDCT//84114007` | 9,247 | 112 |
| `lung_cancer_5y` | `SNOMEDCT//254637007`, `SNOMEDCT//254632001` | 9,244 | 44 |
| `myocardial_infarction_5y` | `SNOMEDCT//22298006`, `SNOMEDCT//401303003`, `SNOMEDCT//401314000` | 8,964 | 225 |
| `stroke_5y` | `SNOMEDCT//230690007` | 9,198 | 39 |
| `type2_diabetes_5y` | `SNOMEDCT//44054006` | 8,532 | 251 |
