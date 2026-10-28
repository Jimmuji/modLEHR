# Data sources and the MEDS format

Back to the [README](../README.md).

## Data format: MEDS

The core reads nothing but MEDS (meds 0.4). Everything specific to a source lives in the code that
converts it. MEDS fixes the storage, not the vocabulary: codes keep their source coding (SNOMED
CT, ICD-10, LOINC), and the choice of prediction targets is made in `prepare`.

A MEDS dataset is a directory:

```
data/<split>/<n>.parquet          events, one row each
metadata/codes.parquet            code, description, parent_codes (one row per code)
metadata/subject_splits.parquet   subject_id, split (train, tuning, held_out)
metadata/dataset.json             dataset name, converter name and version, MEDS version
```

Event columns:

| Column | Type | Meaning |
|---|---|---|
| `subject_id` | int64 | The patient |
| `time` | datetime | When the event happened; empty for facts without a time |
| `code` | string | What happened, prefixed by its vocabulary |
| `numeric_value` | float32 | A measured number, when there is one |
| `text_value` | string | A text answer, when there is one |

Some events of the same patient as above, as stored:

| subject_id | time | code | numeric_value | text_value |
|---|---|---|---|---|
| 296 | | `GENDER//F` | | |
| 296 | 1963-01-16 | `MEDS_BIRTH` | | |
| 296 | 1963-01-16 08:55 | `LOINC//72166-2` (smoking status) | | Never smoked tobacco (finding) |
| 296 | 1965-11-07 | `SNOMEDCT//444814009` (viral sinusitis) | | |
| 296 | 2011-02-09 08:55 | `LOINC//4548-4` (HbA1c) | 5.9 | |
| 296 | 2014-01-22 | `SNOMEDCT//44054006` (type 2 diabetes) | | |
| 296 | 2021-05-05 | `MEDS_DEATH` | | |

## Other data sources

**MIMIC-IV.** The official ETL ([MIMIC_IV_MEDS](https://github.com/Medical-Event-Data-Standard/MIMIC_IV_MEDS))
writes MEDS, and `configs/data/mimic_iv.yaml` reads it: diagnoses (ICD-9 and ICD-10, grouped to
three-character categories) and death as targets; hospital lab results and ICU charted items as
measurements; medications, procedures, admissions, ICU stays and emergency visits as context.
`scripts/run_mimic_iv.sh` runs everything, from the raw files to the comparison with the classical
baselines:

```bash
RAW=~/mimic_dl/physionet.org/files/mimiciv/3.1 scripts/run_mimic_iv.sh
```

It checks that the raw download is complete first, and prints the command to resume it if not. The
ETL takes about 32 minutes and 20 GB of memory. MIMIC-IV is credentialed data: run it in your own
terminal and share summary numbers only, never rows. `configs/experiment/mimic_iv_gpt.yaml` is a
starting point (256 wide, 6 layers, 512-event windows, 20,000 steps; landmarks 40 to 80 with a
2-year horizon, since a MIMIC record ends at the last hospital contact); on a CUDA GPU add
`--set train.precision=bf16`.

**MIMIC-IV diagnoses only.** The full ETL reads every table in the release, labevents and
chartevents among them. `modlehr convert mimic-iv-core` reads the three tables that carry the diagnosis history
(`hosp/patients.csv.gz`, `hosp/admissions.csv.gz`, `hosp/diagnoses_icd.csv.gz`, 56 MB) and
writes the same code families, so `configs/data/mimic_iv_core.yaml` is `mimic_iv.yaml` with the
measurement rules removed. This is the Delphi-2M data shape on MIMIC: sex, disease codes, death,
and the hospital contacts that time them. A stay's diagnoses are timed at its discharge, as in the
official ETL; `--diagnosis-time admission` times them at its admission instead
(`configs/data/mimic_iv_core_admission_time.yaml`). Targets are ICD-10 categories, with ICD-9 codes mapped
through the CMS GEMs (`icd_gems`, 1,018 targets), which `scripts/fetch_code_maps.sh` downloads; see
[docs/disease_codes.md](disease_codes.md). `scripts/run_mimic_iv_core.sh` runs it end to end
and evaluates under both control conventions:

```bash
RAW=~/mimic_dl/physionet.org/files/mimiciv/3.1 scripts/run_mimic_iv_core.sh
```

It is the comparison that says what the measurements are worth, not a replacement for the full run.
`scripts/compare_mimic_iv_core.sh` trains the plain GPT and nine variants on it over seeds 1 to 3
and compares them with logistic regression, all variants at one seed before the next seed; each
run takes about half an hour on an Apple M-series GPU. Runs already evaluated are skipped and
interrupted ones resume, so the script can be stopped and started again.
`scripts/fetch_mimic_iv.sh` resumes the raw download with a PhysioNet account, checking each file
against the release's `SHA256SUMS.txt`.

**Clinical notes.** MIMIC-IV-Note is a separate PhysioNet module and the ETL above does not read
it. `modlehr convert mimic-iv-note` merges its discharge summaries and radiology reports into the
MIMIC-IV MEDS data as `NOTE//DISCHARGE` and `NOTE//RADIOLOGY` events with the text in `text_value`,
each into the shard that already holds the patient. `configs/experiment/mimic_iv_gpt_notes.yaml` adds a
`text` rule for them and turns on the `hashed_bag` text encoding; `scripts/run_mimic_iv.sh` does both
when `NOTES` points at the note folder:

```bash
RAW=~/mimic_dl/physionet.org/files/mimiciv/3.1 NOTES=~/mimic_dl/physionet.org/files/mimic-iv-note/2.2/note \
  scripts/run_mimic_iv.sh
```

Each note keeps its first 256 words (`data.text_max_words`); a note can only be read by its own event
and later ones, like every other input.

The open demo (100 patients) checks the same rules end to end; in it, 1,474 diagnosis codes
(734 ICD-10, 740 ICD-9) become 704 three-character categories, of which 142 plus death are
frequent enough to be targets. It is too small to score:

```bash
uv venv --python 3.11 .venv-etl && uv pip install --python .venv-etl/bin/python MIMIC_IV_MEDS
PATH="$PWD/.venv-etl/bin:$PATH" meds-extract-run spec=MIMIC-IV output_dir=$PWD/data/mimic_iv_demo_meds dataset_key=demo
modlehr run --config configs/experiment/mimic_iv_demo_gpt.yaml
```

**Anything else.** If a MEDS converter exists, run it and write a data config. Otherwise write an
adapter; [docs/extending.md](extending.md) lists what the output must contain.
