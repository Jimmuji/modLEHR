"""Overrides, seeds, the shared landmark task, baselines, comparison and the prepare report."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from modlehr.cli import build_parser, load
from modlehr.config import apply_overrides, load_config
from modlehr.evaluate.baselines import run_baselines
from modlehr.evaluate.compare import compare, group_runs
from modlehr.evaluate.run import evaluate
from modlehr.evaluate.task import landmark_task
from modlehr.generate.simulate import simulate
from modlehr.prepare.dataset import open_split
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
    split = open_split(prepared, "held_out")
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
    shard = next(Path(tiny_config.data.meds_dir).rglob("*.parquet"))  # the MEDS data rewritten since
    st = shard.stat()
    os.utime(shard, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    assert not is_current(tiny_config.data)
    os.utime(shard, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert is_current(tiny_config.data)
    assert any("prediction targets" in line for line in describe(tiny_config.data))


def test_evaluate_baselines_and_compare(tiny_config):
    run_prepare(tiny_config.data, log=QUIET)
    evaluate(tiny_config, train(tiny_config, log=QUIET), log=QUIET)
    run_dir = Path(tiny_config.train.out_dir)
    summary = json.loads((run_dir / "evaluate.json").read_text())
    assert 0 < summary["mean_auc"] < 1 and summary["mean_auc_ci95"][0] <= summary["mean_auc"]
    assert summary["likelihood"]["predictions"] > 0 and np.isfinite(summary["likelihood"]["nll"])
    assert summary["calibration"]["expected_over_observed"] > 0 and (run_dir / "evaluate_calibration.csv").exists()

    table = simulate(tiny_config, run_dir / "best.pt", start_age=50.0, n_subjects=20, log=QUIET)
    sim = json.loads((run_dir / "simulate.json").read_text())
    assert sim["subjects"] > 0 and sim["observed_person_years"] > 0
    assert (table["sampled_person_years"] <= table["observed_person_years"] + 1e-6).all()  # cut at the record's end

    written = run_baselines(tiny_config, models=("logistic", "gbm"), log=QUIET)
    for name, out in written.items():
        assert out == run_dir.with_name(f"{run_dir.name}_{name}")
        assert (out / "evaluate.csv").exists()

    table = compare([run_dir, *written.values()])
    assert list(table["run"]) == ["run", "run_logistic", "run_gbm"]
    assert np.isnan(table["diff"].iloc[0]) and table["shared_pairs"].iloc[1] > 0


def test_resume_continues_exactly(tiny_config, tmp_path):
    run_prepare(tiny_config.data, log=QUIET)
    straight = copy.deepcopy(tiny_config)
    straight.train.out_dir = str(tmp_path / "straight")
    train(straight, log=QUIET)

    split = copy.deepcopy(tiny_config)
    split.train.out_dir = str(tmp_path / "split")
    train(split, log=QUIET, stop_after=14)
    train(split, log=QUIET, resume=True)

    read = lambda d: [json.loads(x) for x in (Path(d) / "log.jsonl").read_text().splitlines()]  # noqa: E731
    a, b = read(straight.train.out_dir), read(split.train.out_dir)
    final_a = [r for r in a if r["step"] == 29][0]
    final_b = [r for r in b if r["step"] == 29][0]
    assert final_a["train_loss"] == pytest.approx(final_b["train_loss"], abs=1e-6)
    assert final_a["tune_loss"] == pytest.approx(final_b["tune_loss"], abs=1e-6)


def test_grad_accum_runs(tiny_config):
    run_prepare(tiny_config.data, log=QUIET)
    tiny_config.train.grad_accum = 2
    tiny_config.train.steps = 5
    assert train(tiny_config, log=QUIET).exists()
    if not torch.cuda.is_available():
        tiny_config.train.precision = "bf16"
        with pytest.raises(ValueError):  # bf16 is for CUDA only
            train(tiny_config, log=QUIET)


def test_convert_toy_from_the_command_line(tmp_path):
    from modlehr.cli import main
    from modlehr.data.meds_io import open_meds

    main(["convert", "toy", "--out", str(tmp_path / "toy"), "--subjects", "50"])
    ds = open_meds(tmp_path / "toy", validate=True)
    assert len(ds.splits) == 50 and "NOTE//DISCHARGE" in set(ds.codes["code"])


def test_a_layer_naming_a_grouping_replaces_its_options(tmp_path):
    """Grouping options are file paths for one grouping; layering another grouping drops them,
    while a layer that leaves the grouping alone still merges into them."""
    (tmp_path / "gems.yaml").write_text("data:\n  target_grouping: icd_gems\n"
                                        "  target_grouping_options: {gems_path: maps/gems.txt}\n")
    (tmp_path / "split.yaml").write_text("defaults: [gems.yaml]\ndata:\n  target_grouping: icd_3char\n")
    (tmp_path / "phecode.yaml").write_text("defaults: [gems.yaml]\ndata:\n  target_grouping: phecode\n"
                                           "  target_grouping_options: {icd9_map: maps/p.csv}\n")
    (tmp_path / "same.yaml").write_text("defaults: [gems.yaml]\ndata:\n  min_subjects_per_token: 7\n")
    assert load_config(tmp_path / "split.yaml").data.target_grouping_options == {}
    assert load_config(tmp_path / "phecode.yaml").data.target_grouping_options == {"icd9_map": "maps/p.csv"}
    assert load_config(tmp_path / "same.yaml").data.target_grouping_options == {"gems_path": "maps/gems.txt"}


def test_calibration_window_starts_at_the_landmark():
    """The last event before L can be years earlier; the observed window is (L, L + H] all the same."""
    from modlehr.evaluate.run import calibration

    class Split:  # three subjects, two events each: sex at age 0, then one more
        offsets = np.array([0, 2, 4, 6])
        arrays = {"ages": np.array([0, 51, 0, 55, 0, 51], np.float32)}
        next_target = np.array([0, -1, 0, -1, -1, -1])
        dt = np.array([51, 0, 55, 0, 51, 0], np.float32)  # from each event to the next target or record end

        def subject(self, i, start, n):
            a = self.offsets[i]
            return {"ages": self.arrays["ages"][a + start:a + n], "next_target": self.next_target[a + start:a + n],
                    "dt": self.dt[a + start:a + n]}

    # landmark 50, horizon 2: subject 0 has the target at 51 (a case), subject 1 at 55 (none within the
    # window), subject 2's record ends at 51 without one (censored, left out)
    summary, _ = calibration(Split(), np.arange(3), 50.0, 2.0, np.full((3, 1), 0.25))
    assert summary["subjects"] == 2 and summary["observed_any"] == 1
    assert summary["expected_any"] == 0.5
