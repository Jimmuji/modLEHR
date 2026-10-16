# Changelog

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
