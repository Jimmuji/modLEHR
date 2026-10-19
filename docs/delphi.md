# modLEHR and Delphi-2M

Delphi-2M (Shmatko et al., *Nature* 2025; code at
[github.com/gerstung-lab/Delphi](https://github.com/gerstung-lab/Delphi)) is a GPT trained on the
disease histories of 400,000 UK Biobank participants that predicts the rate of over 1,000 diseases
and death. It is a reference point for modLEHR, not its template: modLEHR scores any model with
Delphi-2M's AUC, and can run Delphi-2M's own code on a prepared dataset, so the two can be compared
on identical subjects, events and splits.

## What is the same

| | Delphi-2M | modLEHR's plain GPT |
|---|---|---|
| Records | First occurrence of each ICD-10 category, sex, BMI, smoking and alcohol levels, death | First occurrence of each target, plus whatever inputs a data config selects |
| Time | Age at each token, in days | Age at each event, in years |
| Age encoding | Sinusoidal encoding of age in place of positions | `sinusoidal_age`, the same idea |
| Blocks | GPT-2: pre-norm LayerNorm, causal attention, GELU MLP | `gpt_baseline.yaml`: the same |
| Output | A log-rate per token per day: the next token by softmax, the wait by an exponential | A log-rate per target per year, in one competing-risks likelihood. Up to Delphi's time floor (below) the two are the same likelihood |
| Sampling | Each token's wait drawn from its rate, the first one taken | `modlehr generate`: one wait from the total rate and the target in proportion to its rate, the same distribution |
| Delphi-2M's AUC | `evaluate_auc_pipeline` in `evaluate_auc.py` | `modlehr delphi`, the same definition, checked against that code |

## What differs

| | Delphi-2M | modLEHR |
|---|---|---|
| End of a record | The last token | The end of follow-up, set by the data config's `follow_up` rules and death; the stretch from the last event to it is censored in the likelihood |
| "No event" tokens | Inserted at random, one every five years on average, up to the last token; they are predicted like any other token | Off by default: a prediction at a landmark comes from the last event before it. With `data.no_event_years: 5` training inserts them (as inputs that carry a loss, never predicted) and a landmark is read at one |
| Targets recorded together | Attention between them is masked in training; the wait counts from the last untied token | All of them are labels of that time; only the last event of each time carries a loss |
| Inputs that are not targets | Sex and lifestyle tokens, left out of the loss; lifestyle ages shifted at random in training | Any role (static, measurement, categorical, context, text), never predicted |
| Measurements | Levels as tokens (BMI low, normal, high) | Binned tokens, a scaled number (`value_encoding: mlp`) or none |
| Notes | None | `text` events and the `hashed_bag` text encoding |
| Training windows | The first `block_size` tokens of each record | A random window of `max_len` events |
| Starting rates | None | Each target's training-set rate |
| Time floor | `t_min`: 0.1 day added to every wait | None |
| Main evaluation | AUC by sex and 5-year age band against patients who never have the disease | AUC at landmark ages against patients followed past the window (or `controls: never`), calibration, held-out likelihood |

The landmark AUC and calibration (see the README) are modLEHR's main task; `modlehr delphi` is
there to set a model next to Delphi-2M's published numbers, 0.76 (women) and 0.75 (men) on UK
Biobank with a gap of 0.1 day. Those numbers come from a cohort with self-reported history and
registry follow-up; MIMIC-IV is one hospital's admissions, whose records start at the first stay and
end at the last, so the same model is not expected to score the same.

## Scoring several models on the same points

`modlehr delphi --points-dir DIR` shares one set of prediction points between models: diseases,
cases, controls, each point's event and its age band. They are drawn once, from the data alone, and
checked against each model's split by a fingerprint of the subjects, event ages, follow-up and first
diagnoses (token ids are left out, so data prepared with a "no event" token still matches). So a
difference between two models' scores comes from the models, not from who was scored.

On the same points, `--query gap` reads a model with a "no event" token (a modLEHR model trained
with `data.no_event_years`, or Delphi-2M's own code) at one placed at (diagnosis - gap), or for a
control at (its anchor event - gap), after the same history. That is close to how Delphi-2M itself
is read: its "no event" tokens fill the time since the last event, so the model knows how long ago
that was. `--query event` (the default) reads every model at the last event.

## Delphi-2M's own code on modLEHR's data

```bash
scripts/compare_delphi_code.sh                                # the toy data, on a CPU, in a minute
CONFIG=configs/experiment/mimic_iv_core_gpt.yaml DEVICE=cuda scripts/compare_delphi_code.sh --seed 1
```

The script clones Delphi-2M's code at a fixed commit into `external/delphi`, then:

1. `modlehr run` prepares the data, trains modLEHR and scores it; `modlehr delphi` adds Delphi-2M's AUC.
2. `modlehr export-delphi` writes the prepared data in Delphi's format to `<train.out_dir>_delphi_data`:
   `train.bin`, `val.bin` and `test.bin` (the train, tuning and evaluation splits), `labels.csv`,
   `mapping.json`, and `delphi_config.py`, a config for Delphi's `train.py` with this vocabulary and
   the size and schedule of the modLEHR run (layers, heads, width, window, batch, steps, learning
   rates), keeping Delphi's own time floor, "no event" rate and tie masking.
3. Delphi's `train.py` trains on it and writes its checkpoint to `<train.out_dir>_delphi_code`.
4. `modlehr evaluate` and `modlehr delphi`, given that checkpoint and `--delphi-repo`, score it with
   the same code as the modLEHR model; `modlehr compare` sets the two side by side.

How the export maps one format to the other:

- Delphi's ids: 0 is "no event", 1 and 2 are female and male (its AUC code reads sex from them), 3
  to 11 are left unused because its training shifts their ages at random (its lifestyle tokens),
  then the inputs that are not targets, then the targets in modLEHR's order. Everything but the
  targets and "no event" is left out of Delphi's loss (`ignore_tokens`).
- Ages are rounded to whole days. Events after the end of follow-up are not written. When
  follow-up ends after the last event written (it ended at an event `prepare` dropped), a "no event"
  token closes the record there, so Delphi sees the same follow-up.
- Numbers and note words do not reach Delphi; a measurement enters as its token. For a like-for-like
  comparison, use binned values or none, and `--set model.text_encoding=none`.

How a Delphi checkpoint is scored: its log-rates per day for the modLEHR targets are turned into
rates per year, and every part of the evaluation reads them as it reads a modLEHR model's. At a
landmark, a "no event" token at the landmark age is appended to the history, which is how Delphi is
asked about a time with no event. Delphi's forward pass also returns every layer's attention
weights, so batches are split to keep those under about 1 GB.

On the toy data both models score at chance once the notes are left out (the toy diseases are drawn
independently of everything but the notes), which is what the toy data is for: checking that the
two pipelines run on the same data and are scored the same way.
