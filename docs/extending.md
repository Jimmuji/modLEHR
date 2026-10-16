# Extending modLEHR

Two things are meant to be added often: **model parts** (to test an architecture) and **data
sources** (to run on new records). Neither needs changes outside the new code and a config.

## Model parts

Every swappable part is registered under a kind and a name in `modlehr.registry`. A config
names the part; `modlehr.model.model` builds it. What each kind is called with, and must
return:

| Kind | Built as | Called as | Returns |
|---|---|---|---|
| `attention` | `f(d_model, n_heads, dropout, **attention_options)` | `attn(x, mask, ages=ages)` | `[B, T, d_model]` |
| `norm` | `f(d_model)` | `norm(x)` | `[B, T, d_model]` |
| `ffn` | `f(d_model, dropout)` | `ffn(x)` | `[B, T, d_model]` |
| `time_encoding` | `f(d_model)` | `enc(ages)` | `[B, T, d_model]`, added to the token embedding |
| `value_encoding` | `f(d_model)` | `enc(values, has_value)` | `[B, T, d_model]`, added to the token embedding |
| `text_encoding` | `f(d_model, n_buckets)` | `enc(word_ids, starts)` | `[n_notes, d_model]`, added to each note's event, or `None` |
| `head` | `f(d_model, n_targets)` | `head(h)`, plus `loss`, `risk`, `sample` (below) | log-rates `[B, T, n_targets]` for `competing_risk` |
| `target_grouping` | (a plain function) | `group(code)` | the target name for a MEDS code |

Shapes: `x` is `[B, T, d_model]`; `ages` is `[B, T]` in years; `mask` is `[B, 1, T, T]`, `True`
where a query may attend (causal, and the key is not padding); `values` is `[B, T]`, already
scaled; `has_value` is `[B, T]` boolean.

The registering module has to be imported before the config is read.
`modlehr/model/model.py` imports `encodings`, `heads` and `layers`, and
`modlehr/prepare/pipeline.py` imports `grouping`; a new module goes in the same import line.

### Worked example: ALiBi over age

`alibi_age` was added this way, and is the pattern to copy. ALiBi (Press et al., 2022; used in
BLOOM and MPT) subtracts a per-head penalty proportional to the distance between query and key.
Here the distance is years, not positions.

1. **Write the part.** The shared `Attention` class in `modlehr/model/layers.py` already
   handles projections, grouped-query heads and rotary embeddings; the penalty is a few lines in
   its `forward`:

   ```python
   if self.age_bias:
       gap = (ages.unsqueeze(-1) - ages.unsqueeze(-2)).clamp(min=0.0).unsqueeze(1) * self.age_scale
       attn_mask = (-self.slopes * gap).to(q.dtype).masked_fill(~mask, float("-inf"))
   ```

2. **Register it under a name.**

   ```python
   @register("attention", "alibi_age")
   def alibi_age(d_model, n_heads, dropout=0.0, **options):
       return Attention(d_model, n_heads, dropout, age_bias=True, **options)
   ```

3. **Test it.** `tests/test_model.py::test_every_attention_is_causal` runs every registered
   attention, so a new one is checked for leaking future events without extra work. Add a test
   for what the part claims; for `alibi_age`, that shifting every age by the same amount
   changes nothing and changing the gap between events does.

4. **Point a config at it.** `configs/model/gpt_alibi_age.yaml` and
   `configs/experiment/synthea_gpt_alibi_age.yaml`, or no file at all:

   ```bash
   modlehr run --config configs/experiment/synthea_gpt.yaml --set model.attention=alibi_age \
     --set train.out_dir=runs/synthea_gpt_alibi_age --seed 1
   ```

5. **Compare it with the baseline** over the same seeds:

   ```bash
   modlehr compare runs/synthea_gpt_s1 runs/synthea_gpt_s2 runs/synthea_gpt_alibi_age_s1 runs/synthea_gpt_alibi_age_s2
   ```

### A new time encoding

```python
# modlehr/model/encodings.py
@register("time_encoding", "log_age")
class LogAge(nn.Module):
    """log(1 + age), then a small network: more resolution in childhood than at 70."""

    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, d_model), nn.GELU(), nn.Linear(d_model, d_model))

    def forward(self, ages: torch.Tensor) -> torch.Tensor:
        return self.net(torch.log1p(ages).unsqueeze(-1))
```

Then `--set model.time_encoding=log_age`.

### A new head

A head replaces the output layer and the loss. The training loop, evaluation and sampling call
four methods, so a new head has to provide all of them:

| Method | Used by | Must return |
|---|---|---|
| `forward(h)` | everything | the head's output for every event |
| `loss(out, next_target, dt, mask)` | training, `evaluate` likelihood | `(mean NLL over masked events, dict of named parts)` |
| `risk(out, horizon_years)` | `evaluate` | `[..., n_targets]` probability of each target within the horizon |
| `sample(out, generator)` | `generate` | `(target index, years until it)` |

`next_target` is `-1` for a censored stretch; `dt` is the years until the next target, or until
the end of the record. An optional `init_rates(counts, exposure_years)` is called before
training with the training-set count of each target and the total years of follow-up.

### A new target grouping

A grouping maps a MEDS code to the outcome the model predicts. Codes it does not recognize
should come back unchanged.

```python
# modlehr/prepare/grouping.py
import csv
from functools import cache
from pathlib import Path

@cache
def _phecode_map() -> dict[str, str]:
    with open(Path(__file__).parent / "maps" / "icd10cm_to_phecode.csv") as f:
        return {row["icd10cm"].replace(".", ""): row["phecode"] for row in csv.DictReader(f)}

@register("target_grouping", "phecode")
def phecode(code: str) -> str:
    vocab, sep, value = code.rpartition("//")
    target = _phecode_map().get(value.replace(".", "").upper())
    return f"PHECODE//{target}" if sep and "ICD" in vocab.upper() and target else code
```

The map file is not in the repository: phecode maps for ICD-9-CM and ICD-10-CM are public
(PheWAS catalog) and would be added under `modlehr/prepare/maps/`. Then
`target_grouping: phecode` in the data config. `modlehr prepare` prints, per rule, how many
codes were matched, how many targets they became, and how many survived
`min_subjects_per_token`, which is the first thing to check after changing a grouping.

## Data sources

The core reads only MEDS. A new source needs a MEDS dataset and a data config.

### If a MEDS converter already exists

MIMIC-IV, eICU and several others have converters written by the MEDS community
([MEDS_transforms](https://github.com/mmcdermott/MEDS_transforms),
[MIMIC_IV_MEDS](https://github.com/Medical-Event-Data-Standard/MIMIC_IV_MEDS)). Run the converter,
check the output with `modlehr validate --meds-dir DIR`, look at its `metadata/codes.parquet` to see
how codes are named, and write event rules against those names. `configs/data/mimic_iv_demo.yaml`
is an example: MIMIC diagnoses arrive as `DIAGNOSIS//ICD//10//E119` and labs as `LAB//...`.

### Writing an adapter

An adapter turns the source's files into MEDS and makes no modeling choices: no binning, no
deduplication, no filtering. Those belong to the data config, so that every source is prepared
by the same code. `modlehr/data/adapters/synthea.py` is the reference. The output must have:

| File | Contents |
|---|---|
| `data/<split>/<n>.parquet` | Events with `subject_id` (int64), `time` (datetime, null for static facts), `code` (string), `numeric_value` (float32, null when none), `text_value` (string, null when none). Sorted by subject, then time, nulls first |
| `metadata/codes.parquet` | One row per code: `code`, `description`, `parent_codes`. Descriptions are what `description_regex` matches |
| `metadata/subject_splits.parquet` | `subject_id`, `split` in `train`, `tuning`, `held_out`. Every subject in exactly one split |
| `metadata/dataset.json` | `dataset_name`, `etl_name`, `etl_version`, `meds_version` |

Rules that matter downstream:

- **Every subject needs a `MEDS_BIRTH` event.** Ages are computed from it; subjects without one
  are dropped in `prepare`. Use `MEDS_DEATH` for death.
- **Prefix codes with their vocabulary**, `VOCAB//code` (`SNOMEDCT//44054006`, `LOINC//4548-4`,
  `ICD10//E11.9`), so rules can select by prefix and groupings can tell vocabularies apart.
- **Keep the source's codes.** Map to another vocabulary in a target grouping, not in the adapter.
- **Put note text in `text_value`**, one event per note, so a `text` rule can pick it up.
- **Add a source to an existing MEDS dataset by merging into its shards**, never as new shards:
  each subject's events must stay in one shard. `modlehr/data/adapters/mimic_iv_note.py` does this.
- **Write splits by subject**, never by event, so no subject's history leaks into evaluation.
- Write through `meds.DataSchema.schema()` and the other `meds` schemas, and finish with
  `modlehr validate`.

Then register the adapter as a `modlehr convert` source in `modlehr/cli.py` (the `choices` of
`convert` and the branch that calls it), and add a data config.

## Tests

`pytest` runs in seconds on a 160-subject synthetic MEDS dataset written in `tests/conftest.py`.
A new part or source should come with a test of the property it is for; the end-to-end tests
(`tests/test_end_to_end.py`, `tests/test_tools.py`) catch most wiring mistakes.
