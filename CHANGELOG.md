# Changelog

## Unreleased

Fixes from an external code review (all six of its hypotheses held up)
- Targets recorded at the same time are all labels of that time. Before, only the first (in token
  order) was, so on MIMIC-IV, where every diagnosis of an admission is timed at its discharge,
  85% of first diagnoses never supervised the model, and a death tied with diagnoses never did.
  The loss is now -sum over the tied targets of log lambda_k + Lambda dt. Needs retraining
- Follow-up ends at the last event of a rule with `follow_up: true` (all by default) and no later
  than death. A later outpatient BMI no longer turns a censored subject into a control: the BMI
  runs had 6.8% more controls than the diagnoses-only runs they were compared with. Events after
  follow-up carry no loss. Needs re-preparing and retraining
- Calibration weights observed outcomes by the inverse probability of still being followed
  (Kaplan-Meier of the censoring times) instead of dropping censored subjects, which read 0.82 for
  perfect predictions under heavy censoring
- `modlehr compare` adds a 95% interval over seeds; the interval over pairs leaves out seed-to-seed
  variation and was too narrow to decide between models
- `--resume` on an Apple GPU restores its random state too


Results
- MIMIC-IV diagnoses, ten architectures over three seeds on the Alliance cluster Fir: every
  transformer beats logistic regression by about 0.012 AUC; no architecture moves AUC by more than
  0.003 against the plain GPT (README, Results so far)

Data
- `modlehr convert mimic-iv-core`: MIMIC-IV's patients, admissions and diagnoses tables → MEDS,
  the Delphi-2M data shape on MIMIC, with `configs/data/mimic_iv_core.yaml` and
  `scripts/run_mimic_iv_core.sh`. Death is timed as in the official ETL (in-hospital `deathtime`
  first, then `dod`)
- `scripts/fetch_mimic_iv.sh` resumes the raw MIMIC-IV download and checks it against SHA256SUMS.txt

- The MIMIC-IV configs predict ICD-10 categories with ICD-9 mapped through the CMS GEMs
  (`icd_gems`): 1,018 targets on MIMIC-IV diagnoses, against 1,398 with the eras apart
  (`mimic_iv_core_icd3char.yaml`) and 1,361 phecodes (`mimic_iv_core_phecode.yaml`).
  `scripts/fetch_code_maps.sh` downloads the GEMs and the phecode map and checks their hashes
- `phecode` reads the combined v1.2 map (ICD-9-CM and ICD-10-CM in one file, by its `Flag` column);
  a code listed with several phecodes takes the first
- A config file that sets `data.target_grouping` replaces the inherited `target_grouping_options`
- `modlehr run` prepares again when the MEDS files have changed, not only the data config:
  `prepare.json` records a fingerprint of the files (path, size, modification time)

Models
- Batches are padded to a multiple of 64 events, so a model sees at most a few batch shapes. With
  every batch its own length, the mixture of experts grew without bound on an Apple GPU (some
  operations keep a compiled graph per shape); with it, a MIMIC-IV MoE run levels off at 19 GB.
  Padding follows a row's events, so no prediction changes (tested); training differs only in the
  dropout masks drawn, and is the same to float precision with dropout off. About 10% slower
- The mixture of experts takes its load-balancing loss over real events; it used to count padding,
  most of a batch on MIMIC-IV
- `moe` computes every expert and weights the ones not chosen by zero (`dispatch: dense`, the
  default): the same outputs and gradients with fixed tensor shapes. Dispatching events to their
  experts (`dispatch: sparse`) changes shapes every step, and on an Apple GPU the memory cache grew
  a MIMIC-IV run to 47 GB and stalled it; dense runs at 160 ms per step and holds 11 to 13 GB
- The variant files in `configs/model/` list only the part they swap, so they layer onto a model of
  any size. Before, each one pulled in `gpt_baseline.yaml` and reset the width to 128. `gpt_gqa`
  and `gpt_mla` take their sizes from the model (half the query heads; a latent of `d_model` / 4),
  which on the Synthea baseline are the values they had written out
- `configs/experiment/mimic_iv_core_gpt_*.yaml` and `scripts/compare_mimic_iv_core.sh`: every
  variant on MIMIC-IV diagnoses over three seeds, against logistic regression

Tracking
- `train.wandb_project` (and `wandb_entity`) tracks a run in Weights & Biases: losses, learning
  rate, steps per second and peak GPU memory at every check, then the `evaluate` and `baseline`
  summaries in the same run (one id per run directory, kept in `wandb_id`). Off by default;
  works offline on compute nodes, with `wandb sync` afterwards

Evaluation
- Calibration counts the observed window from the landmark, (L, L + H]. It used to count H years
  from the last event before L, which on MIMIC-IV is often years earlier (the sex token at age 0
  for most subjects), so the window was mostly already known to be empty: expected over observed
  read 4.3 on MIMIC-IV diagnoses and is about 0.7 counted from L. `evaluate.json` records
  `"window": "landmark"`
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
