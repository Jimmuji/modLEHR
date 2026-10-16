# modLEHR

modLEHR is a framework for modeling longitudinal electronic health records. Each patient's
record is a sequence of dated events (diagnoses, lab results, vital signs, death), and a model
learns to predict which outcome comes next and when. Records from any source are converted to
one format, [MEDS](https://github.com/Medical-Event-Data-Standard/meds); one code base turns them
into timelines, trains a model, and evaluates it; and every part of the model is chosen by
name in a config file. That makes it cheap to ask whether an idea from recent language models
helps on health records: change one name, train, and compare against the baseline on the same
data with the same evaluation.

The baseline is a plain GPT over events: each event is a code plus the patient's age at that
event, and an output head gives every outcome a rate per year.

> Private repository for now. Do not make it public.

**Contents:**
[What it does](#what-it-does) ·
[Installation](#installation) ·
[Quick start on Synthea](#quick-start-on-synthea) ·
[The pipeline, stage by stage](#the-pipeline-stage-by-stage) ·
[Running and comparing experiments](#running-and-comparing-experiments) ·
[Data format](#data-format-meds) ·
[The model](#the-model) ·
[Results so far](#results-so-far) ·
[Other data sources](#other-data-sources) ·
[Extending](#extending) ·
[Repository layout](#repository-layout) ·
[Troubleshooting](#troubleshooting) ·
[Roadmap](#roadmap)

## What it does

```
raw records ──convert──> MEDS ──prepare──> timelines ──train──> model ──evaluate──> AUC, likelihood
                                                                  │         baseline──> the same task, classical models
                                                                  └─────────generate──> sampled futures
```

| Command | What it does |
|---|---|
| `modlehr convert synthea` | Synthea CSV export → MEDS. Other sources use their existing MEDS converters |
| `modlehr validate` | Check a MEDS dataset against the official schemas |
| `modlehr prepare` | Select events, compute ages, group disease codes, build vocabularies, scale or bin values, attach labels |
| `modlehr info` | Summarize a prepared dataset: subjects, events, tokens and targets per rule |
| `modlehr train` | Fit the model; keep the checkpoint with the lowest tuning loss |
| `modlehr evaluate` | AUC for each outcome at landmark ages, and the held-out negative log-likelihood |
| `modlehr baseline` | Logistic regression and gradient boosting on exactly the same prediction task |
| `modlehr generate` | Sample future outcomes for one patient |
| `modlehr simulate` | Sample one future per held-out patient, cut at the end of the real record, and compare incidence by age with the records |
| `modlehr run` | `prepare` (only if the data config changed), `train` and `evaluate` in one command |
| `modlehr compare` | Mean AUC of several runs and the paired difference against the first, averaging over seeds |

## Installation

Needs Python 3.11 or newer and PyTorch 2.4 or newer. Developed on macOS with Python 3.13;
training uses CUDA if available, then Apple MPS, then the CPU.

With [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/Jimmuji/modLEHR.git && cd modLEHR
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python -m pytest          # 34 tests, about 7 seconds
```

With plain pip: `python -m venv .venv && .venv/bin/pip install -e ".[dev]"`.

Run every command from the repository root; paths in the configs are relative to it. The
examples below write `modlehr`, meaning `.venv/bin/modlehr` (or `.venv/bin/python -m modlehr`).

## Quick start on Synthea

[Synthea](https://github.com/synthetichealth/synthea) simulates complete patient records from
published statistics. It is open and free to redistribute, which makes it the reference dataset
for checking that everything runs. It is not real data; see [Results so far](#results-so-far)
for what it can and cannot show.

**1. Generate patients.** Download `synthea-with-dependencies.jar` from the
[Synthea releases](https://github.com/synthetichealth/synthea/releases) (tested with 3.4.0 on
Java 11), then:

```bash
java -jar synthea-with-dependencies.jar -p 20000 -s 20260919 \
  --exporter.baseDirectory data/synthea_raw \
  --exporter.csv.export true --exporter.fhir.export false \
  --exporter.csv.included_files patients.csv,conditions.csv,observations.csv \
  --exporter.years_of_history 0 --generate.only_alive_patients false
```

Three of these flags change what the model can learn:

| Flag | Synthea's default | Why it is overridden |
|---|---|---|
| `--exporter.csv.export true` | FHIR R4 JSON only | The adapter reads the CSV export |
| `--exporter.years_of_history 0` | only the last 10 years of each record | Keep the whole life, so ages and long histories are real |
| `--generate.only_alive_patients false` | only patients alive at the end | Keep people who die, so death can be predicted |

Asking for 20,000 living patients gives about 23,000 records once deaths are kept.

**2. Convert to MEDS and check it** (74 seconds for 22,962 patients):

```console
$ modlehr convert synthea --raw-dir data/synthea_raw --out data/synthea_meds
{'subjects': 22962, 'events': 47494155, 'codes': 1229}
$ modlehr validate --meds-dir data/synthea_meds
data/synthea_meds: 30 shards, 1,229 codes, 22,962 subjects; valid
```

**3. Prepare timelines** (20 seconds):

```console
$ modlehr prepare --config configs/experiment/synthea_gpt.yaml
[prepare] 10,284,208 matched events, 22,962 subjects
[prepare] train: 16,538 subjects, 2,534,854 events
[prepare] tuning: 1,860 subjects, 291,593 events
[prepare] held_out: 4,564 subjects, 701,948 events
[prepare] rule               role            codes   tokens     kept       events
[prepare] sex                static              2        2        2       22,962
[prepare] death              target              1        1        1        2,962
[prepare] disorders          target            210      210      163      323,337
[prepare] smoking            categorical         1        3        3       33,084
[prepare] measurements       measurement        13       39       39    3,146,050
[prepare] vocabulary 209 tokens, 164 targets → runs/prepared/synthea_binned
```

Read the table row by row: 210 SNOMED CT disorder codes matched, 163 of them occur in at least
25 training patients and become prediction targets, plus death. The 13 measurements become 39
tokens because this baseline bins each value into 3 quantiles.

**4. Train** (3,000 steps, 5 minutes on Apple MPS, 1.25M parameters):

```console
$ modlehr train --config configs/experiment/synthea_gpt.yaml
[train] 1.25M parameters, 209 tokens, 164 targets, device mps
[train] step      0  train 5.9511  tune 5.9111  (1.7s)
...
[train] best tuning loss 4.7806 → runs/synthea_gpt/best.pt
```

**5. Evaluate, and check against classical models** (7 and 30 seconds):

```console
$ modlehr evaluate --config configs/experiment/synthea_gpt.yaml
[evaluate] mean AUC 0.589 (95% CI 0.564 to 0.614) over 124 target and landmark pairs
[evaluate] negative log-likelihood per prediction 4.781 (event 3.852, time 0.929) over 160,013 predictions
$ modlehr baseline --config configs/experiment/synthea_gpt.yaml
[baseline] logistic: mean AUC 0.612 (95% CI 0.586 to 0.637) over 124 pairs → runs/synthea_gpt_logistic/evaluate.csv
[baseline] gbm: mean AUC 0.606 (95% CI 0.579 to 0.633) over 124 pairs → runs/synthea_gpt_gbm/evaluate.csv
```

**6. Sample futures for one patient:**

```console
$ modlehr generate --config configs/experiment/synthea_gpt.yaml --subject 8 --start-age 55 --horizon 15
subject 61 (held_out), sampling from age 55 for 15 years
sample 0:
   57.89  SNOMEDCT//68496003             Polyp of colon (disorder)
   62.84  SNOMEDCT//403190006            Epidermal burn of skin (disorder)
   69.25  SNOMEDCT//414545008            Ischemic heart disease (disorder)
sample 1:
   56.15  SNOMEDCT//368581000119106      Neuropathy due to type 2 diabetes mellitus (disorder)
   ...
sample 2:
   57.01  MEDS_DEATH                     Death
```

## The pipeline, stage by stage

### convert

Turns a source's files into MEDS without making any modeling choice: every source row becomes
one event, nothing is binned, deduplicated or filtered. The Synthea adapter
(`modlehr/data/adapters/synthea.py`) reads `patients.csv`, `conditions.csv` and
`observations.csv`, either from one directory or from chunk directories that each contain a
`csv/` folder, and writes:

| Synthea | MEDS code | Value |
|---|---|---|
| birth date | `MEDS_BIRTH` | |
| death date | `MEDS_DEATH` | |
| sex | `GENDER//F`, `GENDER//M` (no time) | |
| condition | `SNOMEDCT//<code>` at its start date | |
| numeric observation | `LOINC//<code>` | `numeric_value` |
| text observation | `LOINC//<code>` | `text_value`, e.g. `Never smoked tobacco (finding)` |

Subjects are split once, here: 20% held out, 10% of the rest for tuning, the rest for
training. Every later stage uses these splits.

### prepare

Reads MEDS and writes, to `data.prepared_dir`, one timeline per subject with a label on every
event. It does five things, all set in the data config (full reference in
[docs/configuration.md](docs/configuration.md)):

1. **Select events by rules.** Each rule picks codes (by prefix, list or description) and gives
   them a role: `target` (an outcome the model predicts), `static`, `measurement`, `categorical` or
   `context`. Codes no rule picks are left out.
2. **Compute ages.** Each event's age is its time minus the subject's `MEDS_BIRTH`, in years.
   Static facts sit at age 0.
3. **Group disease codes into targets.** Inputs keep the source's detail; targets go through a
   grouping (`identity`, `icd10_3char`, `icd_3char`) so the prediction target stays a manageable
   set. See [docs/disease_codes.md](docs/disease_codes.md). Each target keeps its first occurrence
   per patient.
4. **Handle values.** `value_mode: binned` turns each measurement into a token for its training
   quantile (`LOINC//4548-4//Q1`); `continuous` keeps the number, scaled per measurement, for the
   value encoding to read; `none` drops it.
5. **Build vocabularies and labels.** A token is kept if at least `min_subjects_per_token`
   training patients have it. Every event is labeled with the next target that comes strictly
   later, and the years until it; if none follows, the event is censored at the end of the
   record. Only the last event at each distinct time carries a loss, so the model predicts after
   seeing everything recorded at that moment.

What the labels look like for one Synthea patient (subject 296, 220 events after prepare):

| Token | Age | Next target | In (years) | Loss here |
|---|---|---|---|---|
| `GENDER//F` | 0.00 | Viral sinusitis | 2.81 | yes |
| `LOINC//4548-4//Q1` (HbA1c, lowest third) | 48.07 | Type 2 diabetes | 2.95 | no, more events at this time |
| `LOINC//4548-4//Q1` | 50.00 | Type 2 diabetes | 1.01 | no, more events at this time |
| `SNOMEDCT//44054006` (type 2 diabetes) | 51.02 | Sleep disorder | 0.95 | yes |
| `MEDS_DEATH` | 58.30 | censored | 0.00 | yes |

It reads one MEDS shard at a time, in three passes (training values, training subjects per
token, then every shard written to its split), so memory grows with the largest shard rather than
with the dataset. Statistics such as bin cut points come from the training split only.

Output: `vocab.json` (input and target vocabularies), `prepare.json` (the data config, per-rule
counts, value statistics or bin cut points, target counts, training exposure) and one directory
per split holding one `.npy` file per array, read memory-mapped, with subjects in `subject_id`
order whatever the sharding. `modlehr info` prints a summary, including the most frequent targets.

### train

Each step draws `batch_size` random patients and a random window of at most `max_len` events
from each, and minimizes the competing-risks negative log-likelihood (see
[The model](#the-model)). Every `eval_every` steps it scores the same fixed tuning windows and
keeps the checkpoint with the lowest tuning loss. The run directory gets `config.yaml` (the full
resolved config), `log.jsonl` (one line per check), `best.pt` and `last.pt`.

For long runs: `last.pt` is rewritten at every check with the optimizer and random-number state,
so `modlehr train --resume` continues a stopped run exactly where it was, and `--stop-after STEP`
ends a run early on purpose (a job's time limit). On a GPU, `train.precision: bf16` runs the forward
pass in bfloat16 and `train.grad_accum: k` adds up the gradients of k batches per step.

### evaluate

**AUC at landmark ages.** For a landmark age L, a horizon H and a target k, among patients whose
record covers L: a case has a first k in (L, L + H]; a control has no k by L + H and a record
that reaches L + H; patients who already had k, and patients whose record ends earlier without k,
are left out. The score is the model's probability of k within H from everything up to L. One
AUC per (target, landmark) pair with at least `min_cases` cases and controls; the summary is the
mean over pairs, with an approximate 95% interval. The defaults are landmarks 40, 50, 60 and 70
and a 5-year horizon.

**Calibration.** At each landmark, the head's probability that the next outcome is k and comes
within the horizon is compared with what came next in the record; patients whose record ends
sooner with no outcome are left out. Reported as expected over observed outcomes (1 is calibrated),
and as predicted against observed by risk bin in `evaluate_calibration.csv`.

**Likelihood.** The mean negative log-likelihood per prediction on the held-out patients,
split into an event part (which target comes next) and a time part (how long until it). AUC
only ranks patients within an age group, so it cannot tell whether the model gets timing right;
the time part can. Lower is better.

Output: `evaluate.csv` (one row per pair: target, landmark, cases, controls, AUC),
`evaluate_calibration.csv` and `evaluate.json` (summary) in the run directory.

### baseline

Fits two classical models on exactly the same pairs: an L2-regularized logistic regression and
histogram gradient boosting. Features are one column per input token (present by the landmark
or not), plus the latest scaled value of each measurement when `value_mode: continuous`, stored
sparse so large vocabularies fit; gradient boosting uses the 2,000 most common columns
(`--gbm-features`). Results go to `<out_dir>_logistic/` and `<out_dir>_gbm/` in the same format as
`evaluate`.

Run it for every new dataset. If the logistic regression matches the transformer, the
transformer has learned nothing the inputs did not already say. If gradient boosting does no
better than the logistic regression, the data carry little signal beyond single features, and a
larger model will not find more.

### generate

Starts from a patient's history up to `--start-age`, draws the next outcome and the time until it
from the head, appends it, and repeats until the horizon or death. Targets already in the
history, or already drawn, are not drawn again, since targets are first occurrences.

### simulate

Samples one future from `--start-age` for up to 1,000 held-out patients whose record covers that
age, and cuts each sampled future where that patient's real record ends. Without the cut, a
sampled future runs on to death while a real record stops when follow-up stops, and the model
looks several times too active. Both sides are counted as outcomes per 1,000 person-years in
5-year age bands (`simulate.csv`), and per target (`simulate_targets.csv`).

```console
$ modlehr simulate --config configs/experiment/synthea_gpt.yaml --start-age 50
[simulate] 1,000 of 1,867 held_out patients in the record at age 50
[simulate] outcomes per 1,000 person-years: observed 349.8, sampled 235.5 (ratio 0.67)
```

This checks what AUC cannot: whether sampled futures look like real ones. Here the model samples
281 deaths where the records hold 223, so sampled follow-up ends early, and it generates too few
of every other outcome.

## Running and comparing experiments

```bash
# one command: prepare (skipped when the prepared data matches the data config), train, evaluate
modlehr run --config configs/experiment/synthea_gpt_rope_age.yaml --seed 1

# change any value without writing a config
modlehr run --config configs/experiment/synthea_gpt.yaml --set model.attention=alibi_age \
  --set train.out_dir=runs/synthea_gpt_alibi_age --seed 1

# compare runs; directories that differ only by _s<seed> are averaged first
modlehr compare runs/synthea_gpt_s1 runs/synthea_gpt_s2 runs/synthea_gpt_s3 \
  runs/synthea_gpt_rope_age_s1 runs/synthea_gpt_rope_age_s2 runs/synthea_gpt_rope_age_s3 \
  runs/synthea_gpt_logistic
```

`--seed N` sets `train.seed` and adds `_sN` to the run directory. `modlehr compare` prints, for
each run against the first: mean AUC, held-out negative log-likelihood, and the mean paired AUC
difference over the pairs both scored, with a 95% interval.

`scripts/compare_synthea.sh` trains the baseline and every variant over seeds 1 to 3, fits the
classical baselines, and prints the comparison. Each run takes about 7 minutes on Apple MPS, so the
15 runs take close to two hours; `SEEDS="1" scripts/compare_synthea.sh` runs one seed.

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

## The model

```
tokens, ages, values → input head → N transformer blocks → final norm → output head
```

**Input head.** Each event's vector is `E_code(c) + E_age(t) + m · F(z)`: a learned embedding of
its token, an encoding of the patient's age at the event (in place of a position), and, when the
event has a value and the value encoding is on, a small network applied to the scaled value.
An optional `extra` tensor is added here; it is where other modalities (clinical notes, images)
will plug in.

**Blocks.** Pre-norm transformer blocks with causal attention: an event attends to itself and
to earlier events, never to later ones, and never to padding.

**Output head (`competing_risk`).** One rate per target per year. Between events each target k
has a constant rate λ_k; the next target is k after Δt years with likelihood
λ_k · exp(−Λ Δt), where Λ is the sum of all rates, and a stretch with no target until the record
ends has likelihood exp(−Λ Δt). The loss is the negative log of these. It is the same likelihood
as Delphi-2M's next-event and waiting-time losses (Shmatko et al., *Nature* 2025), written as one
expression, with the end of the record handled as censoring. Rates start at each target's
training-set rate. The risk of k within H years is (λ_k / Λ)(1 − exp(−Λ H)).

Every part is registered by name and picked in `configs/model/`:

| Kind | Registered | Notes |
|---|---|---|
| `attention` | `mha`, `gqa`, `rope`, `rope_age`, `alibi_age` | multi-head (GPT-2); grouped-query (Llama, Qwen); rotary over event order; rotary over age; ALiBi over age (BLOOM, MPT use it over positions) |
| `norm` | `layernorm`, `rmsnorm` | |
| `ffn` | `gelu_mlp`, `swiglu` | GELU MLP (GPT-2); gated SwiGLU (Llama) |
| `time_encoding` | `sinusoidal_age`, `none` | |
| `value_encoding` | `none`, `mlp` | `mlp` reads values kept with `value_mode: continuous` |
| `head` | `competing_risk` | |
| `target_grouping` | `identity`, `icd10_3char`, `icd_3char` | `icd_3char` handles ICD-9 and ICD-10 mixed, as in MIMIC-IV |

Options go in the config, for example `attention_options: {n_kv_heads: 2}`. All keys and
defaults: [docs/configuration.md](docs/configuration.md).

## Results so far

Synthea, 22,962 patients; 164 targets; landmarks 40 to 70; 5-year horizon; 124 (target, landmark)
pairs scored on 4,564 held-out patients. Each transformer has 1.25M parameters (GQA 1.16M) and
3,000 training steps, and is trained with seeds 1, 2 and 3; each pair's AUC is averaged over the
seeds before comparing. The difference is the mean paired AUC difference against the plain GPT,
with its 95% interval.

| Run | Mean AUC | Difference | Held-out NLL |
|---|---|---|---|
| Plain GPT (binned values) | 0.590 | | 4.778 |
| Rotary over age (`rope_age`) | 0.594 | +0.004 (+0.000 to +0.008) | 4.773 |
| ALiBi over age (`alibi_age`) | 0.588 | −0.002 (−0.006 to +0.001) | 4.761 |
| Grouped-query attention (`gqa`) | 0.589 | −0.001 (−0.005 to +0.003) | 4.770 |
| Values read directly (`value_encoding: mlp`) | 0.588 | −0.002 (−0.006 to +0.003) | 4.771 |
| Logistic regression, same inputs | 0.612 | +0.022 (+0.011 to +0.034) | |
| Gradient boosting, same inputs | 0.606 | +0.016 (+0.005 to +0.027) | |

How to read it:

- **Only rotary attention over age moves AUC**, and only just: +0.004, with the interval touching
  zero. With one seed the same comparison gave +0.007; averaging seeds roughly halved it, which is
  why `modlehr compare` averages them.
- **The likelihoods do not separate the variants.** Across the three seeds of one variant the
  held-out NLL varies by up to 0.03; the differences between variants are 0.01 to 0.02.
- **The classical models beat every transformer.** At 3,000 steps the transformers are
  undertrained, so architecture comparisons should wait until the baseline at least matches the
  logistic regression.
- **Rates are off in ways AUC does not show.** The plain GPT predicts 17% more next outcomes than
  occur (expected over observed 1.17), and simulated futures from age 50 hold 0.67 of the observed
  outcomes per person-year, with too many deaths.
- **Synthea limits what can be learned.** It generates diseases from a small number of rules, so
  the signal it holds is limited. It is the place to check that the pipeline runs, not where an
  architecture should be judged; model comparisons belong on real records (MIMIC-IV).

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

The open demo (100 patients) checks the same rules end to end; in it, 1,474 diagnosis codes
(734 ICD-10, 740 ICD-9) become 704 three-character categories, of which 142 plus death are
frequent enough to be targets. It is too small to score:

```bash
uv venv --python 3.11 .venv-etl && uv pip install --python .venv-etl/bin/python MIMIC_IV_MEDS
PATH="$PWD/.venv-etl/bin:$PATH" meds-extract-run spec=MIMIC-IV output_dir=$PWD/data/mimic_iv_demo_meds dataset_key=demo
modlehr run --config configs/experiment/mimic_iv_demo_gpt.yaml
```

**Anything else.** If a MEDS converter exists, run it and write a data config. Otherwise write an
adapter; [docs/extending.md](docs/extending.md) lists what the output must contain.

## Extending

A new model part is one class or function, registered under a name, and one config line; a new
data source is an adapter that writes MEDS and a data config. Step-by-step instructions, the
interface each kind of part must follow, and a worked example (how `alibi_age` was added):
[docs/extending.md](docs/extending.md).

## Repository layout

```
configs/
  data/            which MEDS dataset, which events, how to tokenize them
  model/           which registered parts make up the model
  experiment/      data + model + training + evaluation for one run
src/modlehr/
  data/            MEDS reader and validation; adapters/ convert sources to MEDS
  prepare/         event selection, target grouping, vocabularies, values, labels, batches
  model/           input head, swappable layers, output heads, the assembled model
  train/           training loop
  evaluate/        landmark task, AUC and likelihood, classical baselines, run comparison
  generate/        sampling futures
  registry.py      name → part, for everything that can be swapped
  config.py        experiment configuration and command-line overrides
  cli.py           the commands
scripts/           compare_synthea.sh: every Synthea variant over three seeds; run_mimic_iv.sh: full MIMIC-IV
tests/             unit tests and end-to-end tests on a tiny synthetic MEDS dataset
docs/              design notes, configuration reference, extension guide, disease codes
```

## Troubleshooting

**`import modlehr` fails after the editable install (macOS, Python 3.13).** The install's `.pth`
file is created with the hidden flag, and Python 3.13 skips hidden `.pth` files. Clear it:
`chflags nohidden .venv/lib/python3.13/site-packages/*.pth`. If it keeps coming back, run commands as
`PYTHONPATH=src .venv/bin/python -m modlehr ...`; the tests and `scripts/compare_synthea.sh` already do.

**`no target had enough cases and controls`.** Each (target, landmark) pair needs `min_cases` cases
and controls in the evaluation split. Lower `evaluate.min_cases`, add landmarks, or use a longer
horizon. Small datasets such as the MIMIC-IV demo need `min_cases` of 2 or so.

**`unknown attention 'x'; available: [...]`.** The name is not registered, or the module that
registers it is not imported; see [docs/extending.md](docs/extending.md).

**A rerun of Synthea with the same seed gives a different population.** Synthea generates patients
on several threads, and in our runs the same seed produced 23,056 and then 22,962 patients. Keep the
converted MEDS directory rather than regenerating it, and compare models on the same copy.

**Memory.** `prepare` holds one MEDS shard at a time, plus the training values of every
measurement (pass 1). Peak memory on Synthea is 1.5 GB. If a source's shards are very large, reshard
it with the MEDS tools rather than narrowing the event rules.

## Roadmap

- Disease codes: ICD-9 to ICD-10 through the CMS General Equivalence Mappings, and a phecode grouping
- More architectures from recent language models, compared over several seeds
- Clinical notes as an input, through `text_value` and the input head's `extra` hook
- Prediction tasks in the MEDS-DEV label format, for comparison with other MEDS models

Design notes: [docs/design.md](docs/design.md). Disease codes and prediction targets:
[docs/disease_codes.md](docs/disease_codes.md).
