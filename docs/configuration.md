# Configuration reference

An experiment is one YAML file with up to four sections: `data`, `model`, `train` and
`evaluate`. Every key has a default, so a section only lists what it changes.

## How a config is assembled

```yaml
# configs/experiment/synthea_gpt_rope_age.yaml
defaults: [synthea_gpt.yaml, ../model/gpt_rope_age.yaml]   # merged first, left to right
train:
  out_dir: runs/synthea_gpt_rope_age                         # then this file's own keys
```

1. Each file in `defaults` is read (with its own `defaults`, recursively) and merged in order.
   Paths in `defaults` are relative to the file that lists them.
2. The file's own keys are merged on top. Nested mappings merge key by key; lists and plain
   values replace what was there.
3. Unknown keys are an error, so a typo fails at once instead of being ignored.
4. Paths *inside* a section (`meds_dir`, `prepared_dir`, `out_dir`) are relative to the directory
   the command runs from. Run every command from the repository root.

The repository splits configs by role, so experiments reuse the pieces:

| Directory | Holds | Example |
|---|---|---|
| `configs/data/` | which MEDS dataset, which events, how to tokenize them | `synthea.yaml` |
| `configs/model/` | which registered parts make up the model, and its size | `gpt_baseline.yaml` |
| `configs/experiment/` | data + model + training + evaluation for one run | `synthea_gpt.yaml` |

## Changing a value without a new file

Every command that takes `--config` also takes `--set section.key=value`, repeatable. The value
is parsed as YAML, so numbers, lists and mappings work:

```bash
modlehr train --config configs/experiment/synthea_gpt.yaml \
  --set model.attention=gqa --set "model.attention_options={n_kv_heads: 2}" \
  --set train.steps=6000 --set train.out_dir=runs/synthea_gpt_gqa_6k
```

`--seed N` sets `train.seed` and appends `_sN` to `train.out_dir`, so repeated seeds land in
separate directories that `modlehr compare` groups together. `data.events` cannot be set this
way; edit the data config.

## `data`

| Key | Default | Meaning |
|---|---|---|
| `meds_dir` | `data/meds` | The MEDS dataset to read |
| `prepared_dir` | `runs/prepared` | Where `prepare` writes timelines. Give each data config its own directory; experiments that use the same data config share it |
| `events` | `[]` | Event rules, see below. Codes no rule matches are left out |
| `target_grouping` | `identity` | Registered function that maps a target code to the outcome the model predicts: `identity`, `icd10_3char`, `icd_3char`, `icd_gems`, `phecode` |
| `target_grouping_options` | `{}` | Passed to groupings that read files: `icd_gems` takes `gems_path` (CMS `2018_I9gem.txt`); `phecode` takes `icd9_map`, `icd10_map` (PheWAS catalog CSVs) and `unmapped` (`keep` or `drop`) |
| `min_subjects_per_token` | `25` | A token is kept only if at least this many training subjects have it. Applies to inputs and targets alike, so it also caps the number of prediction targets |
| `value_mode` | `continuous` | How measurement values enter: `continuous` (scaled number, read by the value encoding), `binned` (the value becomes a token `code//Q1..Qn`), `none` (the measurement is an event without a value) |
| `n_bins` | `3` | Number of quantile bins when `value_mode: binned`. Cut points come from the training split |
| `text_buckets` | `65536` | Text events: words are hashed (CRC-32) into this many ids |
| `text_max_words` | `256` | Text events: words kept from the start of each note |

### Event rules (`data.events`)

Each rule selects MEDS codes and gives their events a role. A code goes to the **first** rule
that matches it, and every condition listed in a rule must hold.

| Key | Default | Meaning |
|---|---|---|
| `name` | required | A label for the rule, shown in `prepare` and `info` |
| `role` | required | `target`, `static`, `measurement`, `categorical`, `context` or `text` (table below) |
| `prefix` | none | Code starts with this string, e.g. `"SNOMEDCT//"` |
| `codes` | none | Code is one of these |
| `description_regex` | none | The code's description in `metadata/codes.parquet` matches this regular expression |
| `exclude_regex` | none | Codes matching this regular expression are left out of the rule |
| `thin_years` | none | Keep each subject's first event of each code in every window of this many years. For measurements, context events (medications, procedures) and categorical answers; frequent events such as ICU vital signs or repeated drug administrations otherwise dominate the sequence |
| `first_occurrence` | true for targets, false otherwise | Keep only each subject's first event of each token |

| Role | What it is | How it is tokenized |
|---|---|---|
| `target` | An outcome the model predicts (diseases, death) | The grouped code, e.g. `ICD10//E11`. Also an input token, so a predicted outcome can be fed back when sampling |
| `static` | A fact without a time (sex) | The code, placed at age 0 |
| `measurement` | A numeric value (labs, vitals) | The code, with its value kept, binned or dropped by `value_mode` |
| `categorical` | A text answer (smoking status) | `code//answer`, e.g. `LOINC//72166-2//Never smoked tobacco (finding)` |
| `context` | Any other input (medications, procedures) | The code |
| `text` | A note (discharge summary, radiology report) | The code, plus the note's words for the text encoding |

Example, from `configs/data/synthea.yaml`:

```yaml
events:
  - {name: sex, role: static, prefix: "GENDER//"}
  - {name: death, role: target, codes: ["MEDS_DEATH"]}
  - {name: disorders, role: target, prefix: "SNOMEDCT//", description_regex: "\\(disorder\\)"}
  - {name: smoking, role: categorical, codes: ["LOINC//72166-2"], first_occurrence: true}
  - {name: measurements, role: measurement, thin_years: 2, codes: ["LOINC//39156-5", "LOINC//4548-4"]}
```

Synthea writes SNOMED CT codes for disorders, findings and administrative situations alike;
the `description_regex` keeps only disorders as targets.

## `model`

| Key | Default | Meaning |
|---|---|---|
| `d_model` | `128` | Width of every event embedding and layer |
| `n_layers` | `6` | Transformer blocks |
| `n_heads` | `4` | Attention heads; `d_model` must divide by it |
| `dropout` | `0.1` | Dropout in the input head, attention and feed-forward layers |
| `max_len` | `256` | Events per training window, and the most recent events the model sees at evaluation |
| `time_encoding` | `sinusoidal_age` | `sinusoidal_age`, `none` |
| `value_encoding` | `none` | `none`, `mlp` (use with `data.value_mode: continuous`) |
| `text_encoding` | `none` | `none`, `hashed_bag` (needs a `text` rule in the data config) |
| `attention` | `mha` | `mha`, `gqa`, `rope`, `rope_age`, `alibi_age`, `mla`, `local_age` |
| `attention_options` | `{}` | Passed to the attention; see below |
| `norm` | `layernorm` | `layernorm`, `rmsnorm` |
| `ffn` | `gelu_mlp` | `gelu_mlp` (hidden width 4 x `d_model`), `swiglu` (gated, hidden width 8/3 x `d_model`), `moe` (mixture of experts) |
| `ffn_options` | `{}` | Passed to the feed-forward part; see below |
| `head` | `competing_risk` | `competing_risk` |

`attention_options` that every attention accepts:

| Option | Default | Meaning |
|---|---|---|
| `n_kv_heads` | `n_heads` (`gqa`: half) | Key and value heads; fewer than `n_heads` shares them between query heads |
| `rope_base` | `10000` | Rotary frequency base (`rope`, `rope_age`) |
| `age_scale` | `1.0` | Multiplies ages, in years, before they set rotations (`rope_age`) or penalties (`alibi_age`) |
| `max_slope_exponent` | `8.0` | `alibi_age`: head h has slope 2^(-8h/n_heads) per year |
| `qk_norm` | `false` | RMSNorm on each head's queries and keys before their dot product |
| `kv_rank` | none (`mla`: `d_model` / 4) | Compute keys and values from a latent of this size |
| `window_years` | none (`local_age`: 5) | Attend only to events in this many years before the current one |

`ffn_options` for `moe`:

| Option | Default | Meaning |
|---|---|---|
| `n_experts` | `4` | SwiGLU experts |
| `top_k` | `2` | Experts each event is routed to |
| `mult` | `4/3` | Expert hidden width as a multiple of `d_model`; with `top_k: 2` the active width matches the GELU MLP |
| `aux_weight` | `0.01` | Weight of the load-balancing loss added during training |

`gelu_mlp` and `swiglu` take `mult` too (4 and 8/3 by default).

`modlehr train` prints the parameter count. The Synthea baseline (128 wide, 6 layers) has 1.25M.

## `train`

| Key | Default | Meaning |
|---|---|---|
| `out_dir` | `runs/experiment` | Where the run writes; see below |
| `batch_size` | `64` | Timelines per step |
| `steps` | `3000` | Optimizer steps |
| `lr`, `min_lr` | `6e-4`, `6e-5` | Peak learning rate and the floor of the cosine decay |
| `warmup` | `200` | Linear warm-up steps |
| `weight_decay` | `0.1` | AdamW weight decay on weight matrices (not embeddings, norms or biases) |
| `grad_clip` | `1.0` | Gradient norm clipping |
| `eval_every` | `500` | Steps between tuning-loss checks |
| `eval_batches` | `20` | Fixed tuning batches per check, the same windows every time |
| `seed` | `1` | Seeds batch sampling and initialization |
| `device` | `auto` | `auto` picks CUDA, then Apple MPS, then CPU; or name one |
| `precision` | `fp32` | `bf16` runs the forward pass in bfloat16; CUDA only |
| `grad_accum` | `1` | Batches whose gradients are added up per optimizer step, for a larger effective batch |
| `compile` | `false` | `torch.compile` the model; worthwhile on CUDA for long runs |

Each training step draws `batch_size` random subjects and a random window of at most `max_len`
events from each. The run keeps the checkpoint with the lowest tuning loss.

`last.pt` holds the optimizer and random-number state as well, and is rewritten at every check:
`modlehr train --resume` continues from it exactly, and `--stop-after STEP` stops a run on purpose.

`out_dir` receives `config.yaml` (the full resolved config), `log.jsonl` (one line per check:
step, train and tuning loss, the loss parts, seconds), `best.pt`, `last.pt`, and later
`evaluate.csv` and `evaluate.json`.

## `evaluate`

| Key | Default | Meaning |
|---|---|---|
| `landmarks` | `[40, 50, 60, 70]` | Ages, in years, at which predictions are made |
| `horizon_years` | `5.0` | A case has the outcome within this many years after the landmark |
| `min_cases` | `20` | A (target, landmark) pair is scored only with at least this many cases and this many controls |
| `controls` | `at_risk` | Who counts as a control: `at_risk` keeps subjects with no outcome by L + H whose record reaches L + H, and leaves out the rest as censored; `never` keeps every subject with no outcome anywhere in the record, which is Delphi-2M's definition. AUCs under the two are not comparable, so the one used is written into `evaluate.json` |
| `split` | `held_out` | Which split to score |
| `batch_size` | `256` | Timelines per forward pass |
| `likelihood` | `true` | Also report the negative log-likelihood per prediction on the split |

`modlehr baseline` uses the same `evaluate` section, so its pairs line up with the model's.
`--models` picks `logistic`, `gbm` or both; `--gbm-features` (default 2,000) caps the columns
gradient boosting sees, keeping those non-empty for the most training subjects.
