"""Fixtures: the toy MEDS dataset (modlehr.data.toy) and a small config that reads it."""

from __future__ import annotations

from pathlib import Path

import pytest

from modlehr.config import Config, DataConfig, EvaluateConfig, EventRule, ModelConfig, TrainConfig
from modlehr.data.toy import DISORDERS, FINDING, HBA1C, NOTE, make_meds, note_text  # noqa: F401


@pytest.fixture
def tiny_meds(tmp_path: Path) -> Path:
    return make_meds(tmp_path / "meds")


@pytest.fixture
def tiny_config(tmp_path: Path, tiny_meds: Path) -> Config:
    data = DataConfig(
        meds_dir=str(tiny_meds), prepared_dir=str(tmp_path / "prepared"), min_subjects_per_token=2, value_mode="continuous",
        events=[EventRule(name="sex", role="static", prefix="GENDER//"),
                EventRule(name="death", role="target", codes=["MEDS_DEATH"]),
                EventRule(name="disorders", role="target", prefix="SNOMEDCT//", description_regex=r"\(disorder\)"),
                EventRule(name="hba1c", role="measurement", codes=[HBA1C], thin_years=5)])
    model = ModelConfig(d_model=32, n_layers=2, n_heads=2, dropout=0.0, max_len=64, value_encoding="mlp")
    train = TrainConfig(out_dir=str(tmp_path / "run"), batch_size=16, steps=30, warmup=5, eval_every=10, eval_batches=2, device="cpu")
    evaluate = EvaluateConfig(landmarks=[50.0, 60.0], horizon_years=10.0, min_cases=2)
    return Config(data=data, model=model, train=train, evaluate=evaluate)
