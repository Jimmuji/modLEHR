# Reading the prepared data

Back to the [README](../README.md).

`modlehr prepare` turns MEDS data into token arrays that any model can read. This page says what is in them and
how to load them from your own code. [examples/prepared_data.py](../examples/prepared_data.py) is a loader that
needs only numpy; copy it next to your code.

Every array, token and label column is listed in [docs/data_dictionary.md](data_dictionary.md). MIMIC-IV is credentialed data: report summary numbers only, never a patient's rows.

## What is in a prepared directory

```
vocab.json                    "tokens": token id -> name (id 0 is padding); "targets": target id -> name
prepare.json                  the data config, subjects and events per split, bin cut points
train/  tuning/  held_out/    one .npy array per name below; subjects never cross splits
tasks/<task>/<split>.npz      the samples of one binary task (written by modlehr finetune --samples-only)
```

Token names keep their MEDS codes: `DIAGNOSIS//ICD//10//E11`, `LAB//RESULT//50912//mg/dL//Q2` (a lab result in the
second of three bins), `MEDICATION//START//...`, `GENDER//F`, `MEDS_DEATH`.

## Arrays of a split

Subjects are stored one after another. The events of subject `i` are rows `offsets[i]:offsets[i + 1]` of every
event array, in time order.

| Array | Shape | Meaning |
|---|---|---|
| `subject_ids` | subjects | MEDS subject_id |
| `offsets` | subjects + 1 | where each subject's events start and end |
| `tokens` | events | token id (index into `vocab.json` "tokens") |
| `ages` | events | age in years at the event (sex is stored at age 0) |
| `values` | events | the measured number, NaN when there is none or values are binned into tokens |
| `target` | events | target id if this event is a prediction target (a diagnosis category or death), else -1 |
| `next_target` | events | the next target after this event, -1 if none follows before follow-up ends |
| `dt` | events | years until that next target, or until follow-up ends |
| `loss_mask` | events | true at the last event of each time point within follow-up |
| `follow_up_start`, `follow_up_end` | subjects | ages between which the record is followed |

For next-token training only `tokens` and `ages` are needed: predict `tokens[t + 1]` from events `0..t`.

## Loading it

```python
import numpy as np
from prepared_data import PreparedData

data = PreparedData("runs/prepared/mimic_iv_10pct")
train = data.split("train")
len(train), len(data.tokens)                     # subjects, vocabulary size
events = train.events(0)                         # dict of arrays for the first subject
batch = train.windows(batch_size=32, max_len=1024, rng=np.random.default_rng(0))
batch["tokens"].shape                            # (32, longest window), padded with 0; batch["mask"] marks real events
```

`windows` draws batches the way modLEHR does: random subjects, each cut to a random window of at most `max_len`
events. With the same random generator it returns the same tokens as modLEHR (tests/test_examples.py checks this).

## A prediction task

Each task sample is one patient at one prediction time with a yes/no label. In `tasks/<task>/<split>.npz`:
`index` is the subject's position in the split and `n` is how many of its events come at or before the prediction
time. The input is events `0:n` of subject `index`; the label is `label`.

```python
task = data.task("admission_icu", "train")
batch = data.task_inputs(task, "train", max_len=1024, rows=range(64))   # last 1,024 events of each, padded
batch["tokens"], batch["mask"], batch["label"]
```

Samples with `n == 0` have no events by their prediction time; modLEHR scores them with the training prevalence.

## Your own model on this data

[examples/train_own_model.py](../examples/train_own_model.py) trains a transformer written in plain PyTorch on the
next token, then fits a linear probe on one task. Replace its `Model` class with yours:

```bash
python examples/train_own_model.py runs/prepared/mimic_iv_10pct --task admission_icu --steps 2000
```

To use a modLEHR part instead (an attention, a time encoding), register it as described in
[docs/extending.md](extending.md) and change one config line.

## A smaller copy for quick runs

`modlehr subset` keeps a share of the subjects of every split and nothing else changes: same vocabulary, same bins,
same tasks (their samples are carried over for the kept subjects).

```bash
modlehr subset --config configs/data/mimic_iv.yaml --fraction 0.1   # writes runs/prepared/mimic_iv_10pct
```

The subjects kept are those that `train.data_fraction: 0.1` trains on. On Fir, `scripts/fir/quick.sh` runs a list
of ideas on the 10% copy for 2,000 steps each, one per 20G slice:

```bash
cat > ideas.txt <<'END'
baseline  mimic_iv_gpt_next_token
gqa       mimic_iv_gpt_next_token --set model.attention=gqa
END
bash scripts/fir/quick.sh ideas.txt
python3 scripts/fir/quick_results.py runs/quick/ideas_s1.list   # the ideas, lowest next-event loss first
```
