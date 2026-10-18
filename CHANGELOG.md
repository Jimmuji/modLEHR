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
- Each target's risk is its own probability within the horizon, 1 − exp(−λ_k H) (`evaluate.risk: marginal`).
  The old score, P(k is the next target within the horizon), spreads one event's probability over
  every target of a discharge once targets share their time; it stays as `risk: next_event`
- Calibration compares predicted and observed numbers of new targets in the window. With targets
  sharing their time, P(any target) from the summed hazards read 3.4 times too high

Fixes from a second review
- Calibration counts a target as expected only if it comes before death, since nothing is recorded
  after it: (λ_k / (λ_k + λ_d))(1 − exp(−(λ_k + λ_d) H)). Each target on its own hazard read 1.17
  for the true hazards of a synthetic cohort with a death rate of 0.4 a year (about 1.10 at 0.1 a
  year). Calibration no longer depends on `evaluate.risk`, which picks the AUC score only; with
  `risk: next_event` it compared P(any target) with a count of all of them. Needs re-evaluating
- A config with an unknown section (`evalute:`) is an error; it used to leave that section at its
  defaults without a word
- `modlehr compare` keeps runs of the same name in different directories apart
  (`diagnoses/gpt_s1`, `bmi/gpt_s1`); it averaged them as two seeds of one run and compared nothing
- The interval over seeds pairs the seeds two groups share (seeds 1 to 3 against 1 and 2 used to
  get none), and is one-sample only against a classical baseline. A single seeded model run was
  taken as a fixed number, which left out its own seed-to-seed variation
- Follow-up ends at the last follow-up event the rules match, before `prepare` drops any. It used to
  be computed after thinning, first-occurrence and rare-token filtering, so a thinned last
  measurement ended follow-up up to `thin_years` early (2 years on Synthea, 3 months on full
  MIMIC-IV), and on MIMIC-IV diagnoses a last stay that brought no new diagnosis ended it at
  admission rather than discharge. Needs re-preparing
- `prepare` skips a MEDS shard with no `MEDS_BIRTH` rows, or no rows at all; it stopped with a pandas
  TypeError instead of dropping those subjects as documented
- Configs are checked when they load: `evaluate.risk`, `evaluate.controls`, `evaluate.landmarks`,
  `evaluate.horizon_years`, `data.value_mode` and `train.precision`. A misspelt `evaluate.risk` used
  to fail only after `modlehr run` had trained. A bad event rule names the rule and its unknown or
  missing keys
- `modlehr delphi` sorts cases and controls by sex and age band once instead of masking every row
  for each (disease, sex, band): at MIMIC-IV size (1,018 targets, a million case rows) the AUC table
  takes 83 seconds instead of about 15 minutes. Same table (tested against the plain loop)
- Histories read for a forward pass (landmark scores, baselines, `delphi`) skip the tied-target label
  sets, which made up most of the time: 1.0 instead of 3.6 seconds per landmark for 39,000 subjects
- README, docs/design.md, docs/extending.md and docs/configuration.md describe the code as it is:
  tied labels and follow-up, the marginal risk, calibration, both comparison intervals, the head
  interface and `modlehr delphi`. `scipy`, imported directly, is a declared dependency

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
- `modlehr delphi`: Delphi-2M's own AUC (`evaluate_auc_pipeline`): cases predicted from just before
  their diagnosis, controls who never have the disease, by sex and 5-year age band from 40 to 80,
  with gaps of 0.1 day and one year, to set a model next to Delphi-2M's published numbers
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
