"""Experiment configuration: one YAML file per experiment, made of four sections.

    defaults: [../data/synthea.yaml, ../model/gpt_baseline.yaml]   # merged first, in order
    data: ...      # where the MEDS data is, which events to use, how to tokenize them
    model: ...     # which registered parts to assemble, and their sizes
    train: ...     # optimization
    evaluate: ...  # prediction tasks

Paths inside the config are relative to the directory the command is run from. Any value can
be overridden on the command line without a new file, as section.key=value with the value
written in YAML:

    modlehr train --config CONFIG --set model.attention=gqa --set "model.attention_options={n_kv_heads: 2}"
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class EventRule:
    """Selects MEDS codes and says what role their events play.

    role:
      target       an outcome the model predicts (diseases, death); tokenized as its target
                   group, and only the first occurrence per patient is kept by default
      static       a fact without a time (sex); placed at age 0
      measurement  a numeric value (labs, vitals); value kept, optionally thinned to one per window
      categorical  a text answer (smoking status); the token is code//answer
      context      any other event used as input only (medications, procedures)
      text         an event with text (clinical notes): the code is a token, and the words of its
                   text_value are kept for the model's text encoding
    """

    name: str
    role: str
    prefix: str | None = None
    codes: list[str] | None = None
    description_regex: str | None = None
    exclude_regex: str | None = None  # codes matching this are left out of the rule
    thin_years: float | None = None  # keep the first event of each code per window of this many years
    first_occurrence: bool | None = None
    # Do events of this rule extend follow-up? Set false for events recorded where the targets are not,
    # e.g. outpatient BMI when diagnoses come only from admissions: they stay inputs, but a subject is
    # not counted as followed (a control, or censored later) because of them.
    follow_up: bool = True


@dataclass
class DataConfig:
    meds_dir: str = "data/meds"
    prepared_dir: str = "runs/prepared"
    events: list[EventRule] = field(default_factory=list)
    target_grouping: str = "identity"
    target_grouping_options: dict = field(default_factory=dict)  # e.g. {gems_path: maps/2018_I9gem.txt}
    min_subjects_per_token: int = 25
    value_mode: str = "continuous"  # continuous | binned | none
    n_bins: int = 3
    text_buckets: int = 65536  # text: words are hashed into this many buckets
    text_max_words: int = 256  # text: words kept per note, from the start
    # Where a subject's follow-up starts. first_event: at the first event of any rule, so at age 0 when a
    # static fact (sex) is recorded. first_contact: at the first event of a follow_up rule other than a
    # static one, for sources whose records begin at a first contact (MIMIC-IV's first admission): before
    # it nothing could have been recorded, so earlier events carry no loss, and a landmark task needs the
    # subject followed by the landmark.
    follow_up_from: str = "first_event"
    # Delphi-2M's "no event" tokens: training inserts one at a random age every this many years on
    # average (Delphi-2M: 5), each labelled with the next target from that age, so the model learns
    # rates at times with no event; evaluation and sampling then ask at a "no event" token placed at
    # the landmark or start age. Off (null) by default.
    no_event_years: float | None = None
    # A washout: follow-up starts this many years after it otherwise would. Events in that stretch stay
    # inputs but carry no loss, and a target first recorded there counts as already had at a landmark,
    # not as a new case: in a record that begins at a first admission, the diagnoses of the first stays
    # are often diseases the patient already had (prevalent at entry), not new ones. 0 for none.
    washout_years: float = 0.0


@dataclass
class ModelConfig:
    # modlehr: the GPT assembled from the registered parts below. delphi2m: Delphi-2M's own model and training
    # (modlehr.model.delphi, modlehr.train.delphi) at this size; the parts below are then not used.
    architecture: str = "modlehr"
    d_model: int = 128
    n_layers: int = 6
    n_heads: int = 4
    dropout: float = 0.1
    max_len: int = 256
    time_encoding: str = "sinusoidal_age"
    value_encoding: str = "none"
    text_encoding: str = "none"  # none | hashed_bag (reads the words of text events)
    attention: str = "mha"
    attention_options: dict = field(default_factory=dict)  # passed to the attention, e.g. {n_kv_heads: 2}
    norm: str = "layernorm"
    ffn: str = "gelu_mlp"
    ffn_options: dict = field(default_factory=dict)  # passed to the feed-forward part, e.g. {n_experts: 8, top_k: 2}
    head: str = "competing_risk"


@dataclass
class TrainConfig:
    out_dir: str = "runs/experiment"
    batch_size: int = 64
    steps: int = 3000
    lr: float = 6e-4
    min_lr: float = 6e-5
    warmup: int = 200
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_every: int = 500
    eval_batches: int = 20
    seed: int = 1
    device: str = "auto"
    precision: str = "fp32"  # fp32 | bf16 (CUDA): forward pass in bfloat16
    grad_accum: int = 1  # batches whose gradients are added up per optimizer step
    compile: bool = False  # torch.compile the model (CUDA; first steps are slower)
    wandb_project: str | None = None  # Weights & Biases project; tracking is off when unset
    wandb_entity: str | None = None  # wandb team or user (default: the logged-in user's)
    # Early stopping: end the run after this many tuning checks in a row without a new lowest tuning loss
    # (null: run every step). The learning-rate schedule still follows train.steps, so a stopped run is the
    # first part of the full run, and best.pt is the same unless a lower tuning loss would have come later.
    # The tuning loss is an estimate from eval_batches batches: with the default 20 it moved by about 1 between
    # checks on MIMIC-IV, so an early low value can stop a run that would have gone on improving. A stopped
    # run resumed with patience: null continues to train.steps on the same path.
    patience: int | None = None
    # Train on this share of the training subjects, for a learning curve: the first share of one fixed
    # random order (seed 0, whatever train.seed is), so smaller shares are subsets of larger ones and every
    # seed sees the same subjects. The vocabulary, starting rates and evaluation stay those of the full data.
    # `modlehr baseline` fits on the same subjects.
    data_fraction: float = 1.0


@dataclass
class EvaluateConfig:
    landmarks: list[float] = field(default_factory=lambda: [40.0, 50.0, 60.0, 70.0])
    horizon_years: float = 5.0
    min_cases: int = 20
    controls: str = "at_risk"  # "at_risk" (censored dropped) or "never" (Delphi-2M convention)
    risk: str = "marginal"  # "marginal" P(k within H) = 1 - exp(-lambda_k H), or "next_event" (k is the next target)
    split: str = "held_out"
    batch_size: int = 256
    likelihood: bool = True  # also report the held-out negative log-likelihood (event and time parts)
    # Recalibrate before scoring: at each landmark, scale every rate by the one factor that makes the expected
    # number of new targets on the tuning split equal the observed one. Ranks within a target, so AUC, do not
    # change; calibration on the evaluation split then shows what is left once the overall level is right.
    recalibrate: bool = False


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    evaluate: EvaluateConfig = field(default_factory=EvaluateConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def _layer(base: dict, override: dict) -> dict:
    """_merge, except that a layer naming a target grouping brings its own options: the options of
    the grouping it replaces (mapping file paths, say) mean nothing to the new one."""
    if "target_grouping" in (override.get("data") or {}) and "target_grouping_options" in (base.get("data") or {}):
        base = {**base, "data": {k: v for k, v in base["data"].items() if k != "target_grouping_options"}}
    return _merge(base, override)


def _read(path: Path) -> dict:
    raw = yaml.safe_load(path.read_text()) or {}
    merged: dict = {}
    for parent in raw.pop("defaults", []) or []:
        merged = _layer(merged, _read((path.parent / parent).resolve()))
    return _layer(merged, raw)


def _section(cls: type, values: dict | None) -> Any:
    values = dict(values or {})
    known = {f.name for f in fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"unknown keys for {cls.__name__}: {sorted(unknown)}")
    if cls is DataConfig and "events" in values:
        values["events"] = [_rule(rule) for rule in values["events"] or []]
    return cls(**values)


def _rule(raw: Any) -> EventRule:
    if not isinstance(raw, dict):
        raise ValueError(f"each entry of data.events is a mapping with a name and a role, not {raw!r}")
    unknown, missing = set(raw) - {f.name for f in fields(EventRule)}, {"name", "role"} - set(raw)
    if unknown or missing:
        problem = f"unknown keys {sorted(unknown)}" if unknown else f"missing {sorted(missing)}"
        raise ValueError(f"event rule {raw.get('name', '(no name)')!r}: {problem}; "
                         f"keys are {[f.name for f in fields(EventRule)]}")
    return EventRule(**raw)


# Values checked when a config is loaded, so a typo fails before any work is done rather than at the
# stage that reads it (a misspelt evaluate.risk would otherwise fail after training).
CHOICES = {("model", "architecture"): ("modlehr", "delphi2m"),
           ("data", "value_mode"): ("continuous", "binned", "none"), ("data", "follow_up_from"): ("first_event", "first_contact"),
           ("train", "precision"): ("fp32", "bf16"),
           ("evaluate", "controls"): ("at_risk", "never"), ("evaluate", "risk"): ("marginal", "next_event")}


def check(cfg: Config) -> Config:
    for (section, key), allowed in CHOICES.items():
        value = getattr(getattr(cfg, section), key)
        if value not in allowed:
            raise ValueError(f"{section}.{key} must be one of {', '.join(allowed)}, not {value!r}")
    landmarks = cfg.evaluate.landmarks
    if not isinstance(landmarks, list) or not all(isinstance(a, (int, float)) for a in landmarks):
        raise ValueError(f"evaluate.landmarks must be a list of ages, e.g. [40, 50, 60], not {landmarks!r}")
    if cfg.data.no_event_years is not None and not cfg.data.no_event_years > 0:
        raise ValueError(f"data.no_event_years must be positive, or null for none, not {cfg.data.no_event_years!r}")
    if not cfg.data.washout_years >= 0:
        raise ValueError(f"data.washout_years must be 0 or more, not {cfg.data.washout_years!r}")
    if cfg.train.patience is not None and not (isinstance(cfg.train.patience, int) and cfg.train.patience >= 1):
        raise ValueError(f"train.patience must be a whole number of tuning checks (1 or more), or null, not {cfg.train.patience!r}")
    if not 0 < cfg.train.data_fraction <= 1:
        raise ValueError(f"train.data_fraction must be in (0, 1], not {cfg.train.data_fraction!r}")
    if not cfg.evaluate.horizon_years > 0:
        raise ValueError(f"evaluate.horizon_years must be positive, not {cfg.evaluate.horizon_years!r}")
    return cfg


SECTIONS = ("data", "model", "train", "evaluate")


def load_config(path: str | Path) -> Config:
    raw = _read(Path(path).resolve())
    unknown = set(raw) - set(SECTIONS)
    if unknown:
        raise ValueError(f"{path}: unknown sections {sorted(unknown)}; expected {', '.join(SECTIONS)}")
    return check(Config(
        data=_section(DataConfig, raw.get("data")),
        model=_section(ModelConfig, raw.get("model")),
        train=_section(TrainConfig, raw.get("train")),
        evaluate=_section(EvaluateConfig, raw.get("evaluate")),
    ))


def apply_overrides(cfg: Config, items: list[str]) -> Config:
    """Apply section.key=value overrides in place; the value is parsed as YAML."""
    for item in items or []:
        key, sep, raw = item.partition("=")
        section, _, name = key.strip().partition(".")
        part = getattr(cfg, section, None) if section in SECTIONS else None
        if not sep or part is None or name not in {f.name for f in fields(part)}:
            raise ValueError(f"cannot apply '{item}': expected section.key=value with section one of {SECTIONS}")
        if name == "events":
            raise ValueError("data.events cannot be overridden on the command line; edit the data config")
        setattr(part, name, yaml.safe_load(raw))
    return check(cfg)
