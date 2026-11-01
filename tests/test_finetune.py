from __future__ import annotations

import copy
import datetime as dt
import json
from pathlib import Path

import meds
import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from conftest import write_meds
from modlehr.config import DataConfig, EventRule, load_config
from modlehr.prepare.pipeline import run_prepare
from modlehr.prepare.select import SECONDS_PER_YEAR
from modlehr.train.finetune import finetune, task_samples
from modlehr.train.loop import train

QUIET = lambda *_: None  # noqa: E731


def write_labels(meds_dir: Path, out: Path, age: float, extra: list[tuple] = ()) -> Path:
    """Labels in the MEDS label format for every subject of the toy data: predicted at `age`, the label is
    whether disease A was recorded by then, which the model can read from its input. extra: (split, subject_id,
    time, label) rows added as they are."""
    for split_dir in sorted((meds_dir / "data").iterdir()):
        ev = pd.concat(pd.read_parquet(f) for f in split_dir.glob("*.parquet"))
        birth = ev.loc[ev["code"] == meds.birth_code].groupby("subject_id")["time"].min()
        at = birth + pd.Timedelta(days=365.25 * age)
        a = ev.loc[ev["code"] == "SNOMEDCT//1001"]
        had = set(a.loc[a["time"] <= a["subject_id"].map(at), "subject_id"])
        rows = pd.DataFrame({"subject_id": birth.index.astype("int64"), "prediction_time": at.to_numpy(),
                             "boolean_value": [s in had for s in birth.index]})
        more = pd.DataFrame([r[1:] for r in extra if r[0] == split_dir.name],
                            columns=["subject_id", "prediction_time", "boolean_value"])
        rows = pd.concat([rows, more], ignore_index=True) if len(more) else rows
        rows["prediction_time"] = rows["prediction_time"].astype("datetime64[us]")
        (out / split_dir.name).mkdir(parents=True, exist_ok=True)
        rows.to_parquet(out / split_dir.name / "0.parquet", index=False)
    return out


@pytest.fixture
def pretrained(tiny_config):
    run_prepare(tiny_config.data, log=QUIET)
    return tiny_config, train(tiny_config, log=QUIET)


@pytest.mark.parametrize("start", ["pretrained", "frozen", "scratch"])
def test_finetune_scores_every_sample(pretrained, tmp_path, start):
    base, checkpoint = pretrained
    cfg = copy.deepcopy(base)
    cfg.finetune.labels = str(write_labels(Path(cfg.data.meds_dir), tmp_path / "labels" / "disease_a_by_60", 60.0,
                                           extra=[("held_out", 10**6, dt.datetime(2000, 1, 1), True)]))
    cfg.finetune.init = None if start == "scratch" else str(checkpoint)
    cfg.finetune.freeze = start == "frozen"
    cfg.train.out_dir, cfg.train.steps, cfg.train.lr, cfg.train.eval_every = str(tmp_path / f"ft_{start}"), 600, 3e-3, 50
    result = finetune(cfg, log=QUIET)

    out = Path(cfg.train.out_dir)
    held_out = result["held_out"]
    labels = pd.read_parquet(Path(cfg.finetune.labels) / "held_out")
    predictions = pd.read_parquet(out / "predictions_held_out.parquet")
    assert held_out["samples"] == len(labels) == len(predictions)
    assert held_out["without_inputs"] == 1  # the subject who is not in the data gets the training prevalence
    assert predictions["predicted_boolean_probability"].between(0, 1).all()
    assert json.loads((out / "finetune.json").read_text())["held_out"] == held_out
    tuned = torch.load(out / "best.pt", weights_only=False)["classifier"]
    before = torch.load(checkpoint, weights_only=False)["model"]
    name = next(k for k in before if k.startswith("blocks.0.") and k.endswith("weight"))
    if start == "frozen":  # a linear probe of a model pretrained for 30 steps learns little; only its weights are checked
        assert torch.equal(tuned[f"model.{name}"], before[name])
    else:  # disease A is an input, so whole models learn it
        assert held_out["auroc"] > 0.9 and result["tune_auroc"] > 0.9
        if start == "pretrained":
            assert not torch.equal(tuned[f"model.{name}"], before[name])


def test_input_stops_before_a_later_event_with_the_same_float32_age(tmp_path):
    """An event 30 seconds after the prediction time gets its float32 age at 80, so that sample reads only the
    events before age 80; an event a day later leaves the events at the prediction time in."""
    later = 80 + 30 / SECONDS_PER_YEAR
    rows = [(1, None, "GENDER//F", None), (1, 79.0, "DX//A", None), (1, 80.0, "DX//B", None), (1, later, "ADMIT", None),
            (2, None, "GENDER//M", None), (2, 79.0, "DX//A", None), (2, 80.0, "DX//B", None), (2, 80 + 1 / 365.25, "ADMIT", None)]
    meds_dir = write_meds(tmp_path / "meds", rows)
    data = DataConfig(meds_dir=str(meds_dir), prepared_dir=str(tmp_path / "prepared"), min_subjects_per_token=1,
                      value_mode="none", events=[EventRule(name="sex", role="static", prefix="GENDER//"),
                                                 EventRule(name="dx", role="target", prefix="DX//"),
                                                 EventRule(name="admit", role="context", codes=["ADMIT"])])
    run_prepare(data, log=QUIET)
    assert np.float32((later * SECONDS_PER_YEAR) / SECONDS_PER_YEAR) == np.float32(80.0)
    labels = tmp_path / "labels" / "task" / meds.train_split
    labels.mkdir(parents=True)
    at = dt.datetime(1950, 1, 1) + dt.timedelta(days=365.25 * 80)
    pd.DataFrame({"subject_id": [1, 2], "prediction_time": pd.to_datetime([at, at]).astype("datetime64[us]"),
                  "boolean_value": [True, False]}).to_parquet(labels / "0.parquet", index=False)
    cfg = load_config("configs/experiment/toy_gpt.yaml")
    cfg.data, cfg.finetune.labels = data, str(labels.parent)

    samples = task_samples(cfg, meds.train_split, log=QUIET)
    assert samples["strict"].tolist() == [True, False]
    assert samples["n"].tolist() == [2, 3]  # sex and A; sex, A and B
    assert {k: v.tolist() for k, v in task_samples(cfg, meds.train_split, log=QUIET).items()} == \
        {k: v.tolist() for k, v in samples.items()}  # read back from the cache


def test_samples_only_writes_the_samples_into_the_prepared_data(tiny_config, tmp_path):
    from modlehr.cli import main
    run_prepare(tiny_config.data, log=QUIET)
    tiny_config.finetune.labels = str(write_labels(Path(tiny_config.data.meds_dir), tmp_path / "labels" / "a60", 60.0))
    path = tmp_path / "finetune.yaml"
    path.write_text(yaml.safe_dump(tiny_config.to_dict()))
    main(["finetune", "--config", str(path), "--samples-only"])
    assert sorted(p.name for p in (Path(tiny_config.data.prepared_dir) / "tasks" / "a60").iterdir()) == \
        ["held_out.npz", "train.npz", "tuning.npz"]
    assert not Path(tiny_config.train.out_dir).exists()  # nothing trained


def test_finetune_configs_load():
    for task in ("readmission_30d", "mortality_post_discharge_90d"):
        cfg = load_config(f"configs/experiment/mimic_iv_gpt_{task}.yaml")
        assert Path(cfg.finetune.labels).name == task and cfg.finetune.init and cfg.train.patience
