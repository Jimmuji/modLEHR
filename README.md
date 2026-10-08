# modLEHR

A flexible framework for modeling longitudinal electronic health records. Records from any
source are converted to [MEDS](https://github.com/Medical-Event-Data-Standard/meds), one code
base turns them into patient timelines and trains a model on them, and every part of the model
can be swapped by name in a config file.

The baseline is a plain GPT over events: each event is a code plus the patient's age at that
event, and the model predicts which outcome comes next and when.

> Private until reviewed. Do not make this repository public.

## Five stages

```
raw records ──convert──> MEDS ──prepare──> timelines ──train──> model ──evaluate──> AUC
                                                                     └──generate──> sampled futures
```

| Stage | Command | What it does |
|---|---|---|
| convert | `modlehr convert synthea --raw-dir RAW --out data/synthea_meds` | A data source → MEDS. MIMIC-IV, eICU and EHRSHOT already have MEDS converters |
| prepare | `modlehr prepare --config CONFIG` | Select events, compute ages, build vocabularies, scale or bin values, attach labels |
| train | `modlehr train --config CONFIG` | Fit the model; keeps the checkpoint with the lowest tuning loss |
| evaluate | `modlehr evaluate --config CONFIG` | AUC for each outcome at landmark ages and a horizon |
| generate | `modlehr generate --config CONFIG --subject 0 --start-age 50` | Sample future outcomes for one patient |

`modlehr validate --meds-dir DIR` checks a MEDS dataset against the official schemas.

## Quick start (Synthea)

```bash
uv venv --python 3.13 .venv && uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/modlehr convert synthea --raw-dir /path/to/synthea/output --out data/synthea_meds
.venv/bin/modlehr prepare  --config configs/experiment/synthea_gpt.yaml
.venv/bin/modlehr train    --config configs/experiment/synthea_gpt.yaml
.venv/bin/modlehr evaluate --config configs/experiment/synthea_gpt.yaml
```

On macOS with Python 3.13, if `import modlehr` fails after an editable install, clear the hidden flag on
the installed path file: `chflags nohidden .venv/lib/python3.13/site-packages/*.pth`.

Synthea must be run with `--exporter.csv.export true`. Run commands from the repository root;
paths in configs are relative to it. Training picks CUDA, then Apple MPS, then CPU.

## Layout

```
configs/
  data/          which MEDS dataset, which events, how to tokenize them
  model/         which registered parts make up the model
  experiment/    data + model + training + evaluation for one run
src/modlehr/
  data/          MEDS reader and validation; adapters/ convert sources to MEDS
  prepare/       event selection, target grouping, vocabularies, values, labels, batches
  model/         input head, swappable layers, output heads, the assembled model
  train/         training loop
  evaluate/      landmark AUC
  generate/      sampling futures
  registry.py    name → part, for everything that can be swapped
  config.py      experiment configuration
  cli.py         the commands above
tests/           unit tests and an end-to-end test on a tiny synthetic MEDS dataset
docs/            design notes
```

## Swapping a part of the model

Every swappable part is registered under a kind and a name:

| Kind | Registered now |
|---|---|
| `attention` | `mha` |
| `norm` | `layernorm`, `rmsnorm` |
| `ffn` | `gelu_mlp`, `swiglu` |
| `time_encoding` | `sinusoidal_age`, `none` |
| `value_encoding` | `none`, `mlp` |
| `head` | `competing_risk` |
| `target_grouping` | `identity`, `icd10_3char` |

To try a new attention, write a module with the same call signature, decorate it with
`@register("attention", "my_attention")`, import it in `modlehr/model/model.py`, and set
`model.attention: my_attention` in a config.

## Adding a data source

Write an adapter that produces MEDS (or use an existing MEDS ETL), then a data config with
event rules. Each rule selects codes by prefix, list or description, and gives them a role:
`target` (outcomes the model predicts), `static`, `measurement`, `categorical` or `context`.

## Tests

```bash
.venv/bin/python -m pytest
```

See `docs/design.md` for the modeling choices.
