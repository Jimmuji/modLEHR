"""Experiment configuration: one YAML file per experiment, made of four sections.

    defaults: [../data/synthea.yaml, ../model/gpt_baseline.yaml]   # merged first, in order
    data: ...      # where the MEDS data is, which events to use, how to tokenize them
    model: ...     # which registered parts to assemble, and their sizes
    train: ...     # optimization
    evaluate: ...  # prediction tasks

Paths inside the config are relative to the directory the command is run from.
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
      measurement  a numeric value (labs, vitals); value kept, thinned to one per window
      categorical  a text answer (smoking status); the token is code//answer
      context      any other event used as input only (medications, procedures)
    """

    name: str
    role: str
    prefix: str | None = None
    codes: list[str] | None = None
    description_regex: str | None = None
    thin_years: float | None = None
    first_occurrence: bool | None = None


@dataclass
class DataConfig:
    meds_dir: str = "data/meds"
    prepared_dir: str = "runs/prepared"
    events: list[EventRule] = field(default_factory=list)
    target_grouping: str = "identity"
    min_subjects_per_token: int = 25
    value_mode: str = "continuous"  # continuous | binned | none
    n_bins: int = 3


@dataclass
class ModelConfig:
    d_model: int = 128
    n_layers: int = 6
    n_heads: int = 4
    dropout: float = 0.1
    max_len: int = 256
    time_encoding: str = "sinusoidal_age"
    value_encoding: str = "none"
    attention: str = "mha"
    norm: str = "layernorm"
    ffn: str = "gelu_mlp"
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


@dataclass
class EvaluateConfig:
    landmarks: list[float] = field(default_factory=lambda: [40.0, 50.0, 60.0, 70.0])
    horizon_years: float = 5.0
    min_cases: int = 20
    split: str = "held_out"
    batch_size: int = 256


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


def _read(path: Path) -> dict:
    raw = yaml.safe_load(path.read_text()) or {}
    merged: dict = {}
    for parent in raw.pop("defaults", []) or []:
        merged = _merge(merged, _read((path.parent / parent).resolve()))
    return _merge(merged, raw)


def _section(cls: type, values: dict | None) -> Any:
    values = dict(values or {})
    known = {f.name for f in fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"unknown keys for {cls.__name__}: {sorted(unknown)}")
    if cls is DataConfig and "events" in values:
        values["events"] = [EventRule(**rule) for rule in values["events"]]
    return cls(**values)


def load_config(path: str | Path) -> Config:
    raw = _read(Path(path).resolve())
    return Config(
        data=_section(DataConfig, raw.get("data")),
        model=_section(ModelConfig, raw.get("model")),
        train=_section(TrainConfig, raw.get("train")),
        evaluate=_section(EvaluateConfig, raw.get("evaluate")),
    )
