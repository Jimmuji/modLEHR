# The pipeline, stage by stage

What each command does, what it reads and what it writes. Back to the [README](../README.md).

## convert

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

## prepare

Reads MEDS and writes, to `data.prepared_dir`, one timeline per subject with a label on every
event. It does five things, all set in the data config (full reference in
[docs/configuration.md](configuration.md)):

1. **Select events by rules.** Each rule picks codes (by prefix, list or description) and gives
   them a role: `target` (an outcome the model predicts), `static`, `measurement`, `categorical`,
   `context` or `text` (a note, whose words are kept for the text encoding). Codes no rule picks
   are left out.
2. **Compute ages.** Each event's age is its time minus the subject's `MEDS_BIRTH`, in years.
   Static facts sit at age 0.
3. **Group disease codes into targets.** Inputs keep the source's detail; targets go through a
   grouping (`identity`, `icd10_3char`, `icd_3char`) so the prediction target stays a manageable
   set. See [docs/disease_codes.md](disease_codes.md). Each target keeps its first occurrence
   per patient.
4. **Handle values.** `value_mode: binned` turns each measurement into a token for its training
   quantile (`LOINC//4548-4//Q1`); `continuous` keeps the number, scaled per measurement, for the
   value encoding to read; `none` drops it.
5. **Build vocabularies and labels.** A token is kept if at least `min_subjects_per_token`
   training patients have it. Every event is labeled with the next time strictly later at which a
   target is recorded: all the targets recorded then (a discharge often brings several diagnoses
   at once), and the years until it. If none follows, the event is censored at the end of
   follow-up: the subject's last event of a rule with `follow_up: true` (every rule by default),
   and no later than death. Events after follow-up stay inputs but carry no loss. Follow-up starts
   at the first event; with `follow_up_from: first_contact` it starts at the first event of a
   follow-up rule that is not static, for records that begin at a first hospital contact, and
   events before it carry no loss either; `washout_years` moves the start later still, so the
   diagnoses of a first stay are history rather than new cases. Only the last
   event at each distinct time carries a loss, so the model predicts after seeing everything
   recorded at that moment.

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

## train

Each step draws `batch_size` random patients and a random window of at most `max_len` events
from each, and minimizes the competing-risks negative log-likelihood (see
[the model](model.md)). Every `eval_every` steps it scores the same fixed tuning windows and
keeps the checkpoint with the lowest tuning loss. The run directory gets `config.yaml` (the full
resolved config), `log.jsonl` (one line per check), `best.pt` and `last.pt`.

For long runs: `last.pt` is rewritten at every check with the optimizer and random-number state,
so `modlehr train --resume` continues a stopped run exactly where it was, and `--stop-after STEP`
ends a run early on purpose (a job's time limit). On a GPU, `train.precision: bf16` runs the forward
pass in bfloat16 and `train.grad_accum: k` adds up the gradients of k batches per step.
`train.patience: n` stops a run after n checks without a new lowest tuning loss; the schedule still
follows `train.steps`, so the stopped run is the start of the full one, and resuming it with
`train.patience: null` continues it on the same path. The tuning loss is an estimate from
`eval_batches` batches, and an early low value can stop a run that would have improved: on MIMIC-IV,
patience 5 stopped one run at an AUC of 0.53 where its other seeds reached 0.63, so the MIMIC-IV
configs train every step. `train.data_fraction: 0.25`
trains on a fixed quarter of the training subjects (the same for every seed, inside the half), for
a learning curve.

## evaluate

**AUC at landmark ages.** For a landmark age L, a horizon H and a target k, among patients whose
record starts by L and who are followed past L: a case has a first k in (L, L + H]; a control has
no k by L + H and is followed to L + H; patients who already had k, and patients whose follow-up
ends earlier without k, are left out. The score is the model's probability of k within H from everything up to L. One
AUC per (target, landmark) pair with at least `min_cases` cases and controls; the summary is the
mean over pairs, with an approximate 95% interval. The defaults are landmarks 40, 50, 60 and 70
and a 5-year horizon.

`evaluate.controls` picks who the controls are. The default `at_risk` is the paragraph above: a
control must still be followed at L + H, so a control may develop the disease later, and
censored subjects are dropped. `controls: never` keeps every subject with no record of the disease
at all, censored or not, which is how Delphi-2M defines controls. Which one is in force changes the
numbers, so it is written into `evaluate.json` and printed with the mean, and `modlehr baseline`
reads the same setting.

**"No event" tokens.** By default the score at a landmark is read at the last event before it,
which can be years earlier, with the rates held constant since. `data.no_event_years: 5` adds
Delphi-2M's answer: training inserts a "no event" token at a random age every five years on
average, labelled with the next target from that age, and evaluation reads each landmark at one
placed at the landmark age.

**Calibration.** At each landmark, the predicted number of targets first occurring in (L, L + H] is
compared with the number observed. The prediction sums, over targets the patient has not had yet,
the probability that each is recorded in the window, which means before death, since nothing is
recorded after it. It comes from the model's rates whatever `evaluate.risk` is.
Patients whose follow-up ends inside the window are not dropped: the fully observed ones are weighted
by 1 / P(still followed), from a Kaplan-Meier fit of the follow-up end times. Reported as expected
over observed (1 is calibrated), and per (patient, target) by risk bin in `evaluate_calibration.csv`.
`evaluate.recalibrate: true` first scales the rates at each landmark by one factor fit on the tuning
split, which leaves AUCs as they are and shows how much of the error is the overall level.

**Likelihood.** The mean negative log-likelihood per prediction on the held-out patients,
split into an event part (which target comes next) and a time part (how long until it). AUC
only ranks patients within an age group, so it cannot tell whether the model gets timing right;
the time part can. Lower is better.

Output: `evaluate.csv` (one row per pair: target, landmark, cases, controls, AUC),
`evaluate_calibration.csv` and `evaluate.json` (summary) in the run directory.

## baseline

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

## generate

Starts from a patient's history up to `--start-age`, draws the next outcome and the time until it
from the head, appends it, and repeats until the horizon or death. Targets already in the
history, or already drawn, are not drawn again, since targets are first occurrences.

## simulate

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

## delphi

Scores a model the way Delphi-2M scores itself (`evaluate_auc_pipeline` in its code), so its
numbers can be set next to the ones Delphi-2M published. For each disease, every patient whose
first record of it falls within follow-up is a case, predicted from the last event strictly before
the diagnosis minus a gap; controls are patients who never have the disease, each predicted from
the last event before one of their own event times minus the gap. AUCs are taken by sex and by
5-year band of the age at prediction, from 40 to 80, with one prediction per patient per band, and
averaged over bands per disease and sex. The gaps are 0.1 day and one year by default
(`--gap-days`), Delphi-2M's two settings.

```console
$ modlehr delphi --config configs/experiment/toy_gpt.yaml
[delphi] gap 0.1 days, query at event: female 0.611 over 4 diseases, male 0.639 over 4 diseases → runs/toy_gpt/evaluate_delphi_gap0.1d.csv
```

This is not the main task: a case's prediction point is chosen by when the diagnosis happened, and
with a short gap it is usually the admission of the same hospital stay, so the score mixes
forecasting with recognising a patient already in hospital. Report it next to the landmark AUC,
not in its place.

To compare models, score them all on one set of prediction points with `--points-dir DIR`. The
first model to need them draws them from the data (with seed 0, whatever the model's seed) and
writes them there; every later model reads them, and is refused if its data differ. Every model is
then scored on the same diseases, cases, controls and times, which a model on its own prepared data
(one trained with "no event" tokens, or Delphi-2M's own code) would not otherwise guarantee. The
points name patients and ages, so keep the directory with the data, not with the results you share.
A model with a "no event" token can also be asked at one placed a gap before each diagnosis, after
the same history (`--query gap`), to see what knowing the current age adds.

Delphi-2M predicts only ICD-10 chapters I to XVII (A00 to Q99) and death; MIMIC-IV's hospital codes
also hold symptoms (R), injuries (S, T), external causes (V to Y) and Z codes. `--targets
configs/targets/delphi2m.txt` adds the means over Delphi-2M's diseases to the summary, for a
comparison over the diseases both predict.

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
difference over the pairs both scored, with two 95% intervals. The one over pairs leaves out how
much a run moves with its seed and is too narrow to decide between models. The one over seeds is
the one to read: it pairs the seeds both runs have (at least two), or, against a classical
baseline, uses the other run's seeds alone. A single seeded model run gets none.

### Every variant over seeds

`scripts/compare_synthea.sh` trains the baseline and every variant over seeds 1 to 3, fits the
classical baselines, and prints the comparison. Each run takes about 7 minutes on Apple MPS, so the
33 runs (11 variants, 3 seeds) take about four hours; `SEEDS="1" scripts/compare_synthea.sh` runs one
seed.

### What moves the results

The settings study asks how far each choice moves the numbers: the nine model parts, "no event"
tokens, a share of the training subjects (10%, 25%, 50%), and the choices that define the task (where
follow-up starts, a washout, the target codes, the horizon, the controls), each over seeds and against
logistic regression on its own task, plus gradient boosting. Once those runs are evaluated,
`scripts/analysis/settings_study.py runs` writes the table and figure: for each choice, the change
in the reported mean AUC, the model's lead over logistic regression on the same task, and calibration.
`scripts/analysis/by_chapter.py` splits a model's lead over a baseline by ICD-10 chapter, landmark age
and number of cases.

### Delphi-2M's own code on the same data

```bash
scripts/compare_delphi_code.sh --set model.text_encoding=none     # toy data, CPU, about a minute
CONFIG=configs/experiment/mimic_iv_core_gpt.yaml DEVICE=cuda scripts/compare_delphi_code.sh --seed 1
```

The script clones Delphi-2M's code at a fixed commit into `external/delphi`, trains modLEHR, writes
the same prepared data in Delphi's format (`modlehr export-delphi`, with a config of the same size and
schedule), trains Delphi's `train.py` on it, scores its checkpoint with `modlehr evaluate` and
`modlehr delphi` (`--delphi-repo`), and compares the two. What the two models share, where they
differ, and how one format maps to the other: [docs/delphi.md](delphi.md).
