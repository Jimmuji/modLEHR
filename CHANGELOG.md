# Changelog

## Unreleased

Data
- `modlehr convert mimic-iv-core`: MIMIC-IV's patients, admissions and diagnoses tables → MEDS,
  the Delphi-2M data shape on MIMIC, with `configs/data/mimic_iv_core.yaml` and
  `scripts/run_mimic_iv_core.sh`. Death is timed as in the official ETL (in-hospital `deathtime`
  first, then `dod`)
- `scripts/fetch_mimic_iv.sh` resumes the raw MIMIC-IV download and checks it against SHA256SUMS.txt

- `modlehr run` prepares again when the MEDS files have changed, not only the data config:
  `prepare.json` records a fingerprint of the files (path, size, modification time)

Models
- The variant files in `configs/model/` list only the part they swap, so they layer onto a model of
  any size. Before, each one pulled in `gpt_baseline.yaml` and reset the width to 128. `gpt_gqa`
  and `gpt_mla` take their sizes from the model (half the query heads; a latent of `d_model` / 4),
  which on the Synthea baseline are the values they had written out
- `configs/experiment/mimic_iv_core_gpt_*.yaml` and `scripts/compare_mimic_iv_core.sh`: every
  variant on MIMIC-IV diagnoses over three seeds, against logistic regression

Evaluation
- `evaluate.controls`: `at_risk` (the default) or `never`, Delphi-2M's definition, where controls
  are subjects who never have the disease. The convention is recorded in `evaluate.json` and the
  classical baselines use the same one

## 0.2.0 (October 2026)

Data
- `prepare` reads one MEDS shard at a time; splits are memory-mapped, in subject order whatever the sharding
- Full MIMIC-IV configs and `scripts/run_mimic_iv.sh`, from the raw files to the comparison with baselines
- Clinical notes: the `text` role, `modlehr convert mimic-iv-note`
- Disease groupings `icd_gems` (ICD-9 through the CMS GEMs) and `phecode`, configured with `target_grouping_options`
- `thin_years` for context and categorical events
- `modlehr convert toy`: a synthetic dataset to try every command in a minute

Model
- Attention: `alibi_age`, `mla`, `local_age`, and `qk_norm` for any attention
- Feed-forward: `moe`, with a load-balancing loss; `ffn_options`
- Text encoding: `hashed_bag`

Training
- `--resume` and `--stop-after`; `precision: bf16`, `grad_accum`, `compile`

Evaluation
- Held-out negative log-likelihood (event and time parts) and calibration in `evaluate`
- `modlehr baseline` (logistic regression, gradient boosting on the same task), `modlehr compare` (averages seeds)
- `modlehr simulate`: sampled futures against observed incidence by age

Usability
- `--set section.key=value` and `--seed N` on every config command; `modlehr run`, `modlehr info`
- README walks every stage with real output; docs/configuration.md and docs/extending.md
- Tests on GitHub Actions

## 0.1.0 (October 2026)

MEDS in, a plain GPT baseline with swappable parts, Synthea adapter, landmark AUC.
