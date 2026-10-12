"""Overrides, seeds, the shared landmark task, baselines, comparison and the prepare report."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from modlehr.cli import build_parser, load
from modlehr.config import apply_overrides, load_config
from modlehr.evaluate.baselines import run_baselines
from modlehr.evaluate.compare import compare, group_runs
from modlehr.evaluate.run import evaluate
from modlehr.evaluate.task import landmark_task
from modlehr.prepare.dataset import PreparedSplit
from modlehr.prepare.pipeline import describe, is_current, run_prepare
from modlehr.prepare.vocabulary import Vocabulary
from modlehr.train.loop import train

QUIET = lambda *_: None  # noqa: E731
CONFIG = "configs/experiment/synthea_gpt.yaml"


def test_overrides_parse_yaml_values():
    cfg = apply_overrides(load_config(CONFIG), ["model.attention=gqa", "model.attention_options={n_kv_heads: 2}",
                                                "train.steps=10", "evaluate.landmarks=[50, 60]"])
    assert cfg.model.attention == "gqa" and cfg.model.attention_options == {"n_kv_heads": 2}
    assert cfg.train.steps == 10 and cfg.evaluate.landmarks == [50, 60]


@pytest.mark.parametrize("bad", ["model.not_a_key=1", "nosection.steps=1", "train.steps", "data.events=[]"])
def test_bad_overrides_fail(bad):
    with pytest.raises(ValueError):
        apply_overrides(load_config(CONFIG), [bad])


def test_seed_sets_seed_and_suffix():
    args = build_parser().parse_args(["train", "--config", CONFIG, "--seed", "3"])
    cfg = load(args)
    assert cfg.train.seed == 3 and cfg.train.out_dir.endswith("synthea_gpt_s3")


def test_group_runs_by_seed_suffix():
    groups = group_runs(["runs/a_s1", "runs/a_s2", "runs/a_rope_s1", "runs/a_logistic"])
    assert list(groups) == ["a", "a_rope", "a_logistic"] and len(groups["a"]) == 2


def test_landmark_task_labels(tiny_config):
    run_prepare(tiny_config.data, log=QUIET)
    prepared = Path(tiny_config.data.prepared_dir)
    vocab = Vocabulary.load(prepared / "vocab.json")
    split = PreparedSplit(prepared / "held_out.npz")
    task = landmark_task(split, vocab, 50.0, 10.0)
    for k in range(len(vocab.targets)):
        keep, case = task.labels(k)
        f = task.first[:, k]
        assert np.all(f[keep] > 50.0)                      # nobody who already had k
        assert np.all((f[case] > 50.0) & (f[case] <= 60.0))  # cases inside the window
        assert np.all(task.end[keep & ~case] >= 60.0)       # controls followed to the end of the window


def test_prepare_reports_rules_and_is_current(tiny_config):
    meta = run_prepare(tiny_config.data, log=QUIET)
    rules = {r["rule"]: r for r in meta["rules"]}
    assert set(rules) == {"sex", "death", "disorders", "hba1c"}
    assert rules["disorders"]["codes"] == 3 and all(r["kept"] <= r["tokens"] for r in rules.values())
    assert is_current(tiny_config.data)
    changed = copy.deepcopy(tiny_config.data)
    changed.min_subjects_per_token = 3
    assert not is_current(changed)
    assert any("prediction targets" in line for line in describe(tiny_config.data))


def test_evaluate_baselines_and_compare(tiny_config):
    run_prepare(tiny_config.data, log=QUIET)
    evaluate(tiny_config, train(tiny_config, log=QUIET), log=QUIET)
    run_dir = Path(tiny_config.train.out_dir)
    summary = json.loads((run_dir / "evaluate.json").read_text())
    assert 0 < summary["mean_auc"] < 1 and summary["mean_auc_ci95"][0] <= summary["mean_auc"]
    assert summary["likelihood"]["predictions"] > 0 and np.isfinite(summary["likelihood"]["nll"])

    written = run_baselines(tiny_config, models=("logistic", "gbm"), log=QUIET)
    for name, out in written.items():
        assert out == run_dir.with_name(f"{run_dir.name}_{name}")
        assert (out / "evaluate.csv").exists()

    table = compare([run_dir, *written.values()])
    assert list(table["run"]) == ["run", "run_logistic", "run_gbm"]
    assert np.isnan(table["diff"].iloc[0]) and table["shared_pairs"].iloc[1] > 0
