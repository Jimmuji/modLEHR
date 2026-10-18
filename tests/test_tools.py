"""Overrides, seeds, the shared landmark task, baselines, comparison and the prepare report."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
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


@pytest.mark.parametrize("bad", ["model.not_a_key=1", "nosection.steps=1", "train.steps", "data.events=[]",
                                 "evaluate.risk=marginl", "evaluate.controls=nevr", "evaluate.landmarks=50",
                                 "evaluate.horizon_years=0", "data.value_mode=bins", "train.precision=fp16",
                                 "data.follow_up_from=birth"])
def test_bad_overrides_fail(bad):
    with pytest.raises(ValueError):
        apply_overrides(load_config(CONFIG), [bad])


def test_an_unknown_section_fails(tmp_path):
    """A misspelt section would otherwise leave every value in it at its default without a word."""
    path = tmp_path / "typo.yaml"
    path.write_text(f"defaults: [{Path(CONFIG).resolve()}]\nevalute:\n  controls: never\n")
    with pytest.raises(ValueError, match="evalute"):
        load_config(path)


@pytest.mark.parametrize("rule, message", [("{name: sex, role: static, folow_up: false}", "folow_up"),
                                           ("{name: sex, prefix: GENDER//}", "role")])
def test_a_bad_event_rule_names_the_rule(tmp_path, rule, message):
    path = tmp_path / "rule.yaml"
    path.write_text(f"defaults: [{Path(CONFIG).resolve()}]\ndata:\n  events:\n    - {rule}\n")
    with pytest.raises(ValueError, match=f"event rule 'sex'.*{message}"):
        load_config(path)


def test_seed_sets_seed_and_suffix():
    args = build_parser().parse_args(["train", "--config", CONFIG, "--seed", "3"])
    cfg = load(args)
    assert cfg.train.seed == 3 and cfg.train.out_dir.endswith("synthea_gpt_s3")


def test_group_runs_by_seed_suffix():
    groups = group_runs(["runs/a_s1", "runs/a_s2", "runs/a_rope_s1", "runs/a_logistic"])
    assert list(groups) == ["a", "a_rope", "a_logistic"] and len(groups["a"]) == 2


def _evaluated(path: Path, aucs, baseline: str | None = None) -> Path:
    import pandas as pd
    path.mkdir(parents=True)
    pd.DataFrame({"target": [f"T{i}" for i in range(len(aucs))], "landmark": 50, "cases": 30, "controls": 30,
                  "auc": aucs}).to_csv(path / "evaluate.csv", index=False)
    (path / "evaluate.json").write_text(json.dumps({"baseline": baseline} if baseline else {"checkpoint": "best.pt"}))
    return path


def test_runs_of_the_same_name_in_different_directories_are_not_seeds(tmp_path):
    a = _evaluated(tmp_path / "diagnoses" / "gpt_s1", [0.60, 0.62])
    b = _evaluated(tmp_path / "bmi" / "gpt_s1", [0.70, 0.72])
    table = compare([a, b])
    assert list(table["seeds"]) == [1, 1] and table["run"].iloc[1].endswith("bmi/gpt")
    assert table["diff"].iloc[1] == pytest.approx(0.10)


def test_seed_intervals(tmp_path):
    """Seed by seed over the seeds both groups have; one-sample only against a classical baseline;
    none against a single seeded model run, which is one draw rather than a fixed number."""
    ref = [_evaluated(tmp_path / f"gpt_s{s}", [0.60 + 0.01 * s, 0.62 + 0.01 * s]) for s in (1, 2, 3)]
    var = [_evaluated(tmp_path / f"rope_s{s}", [0.61 + 0.01 * s + 0.002 * s, 0.63 + 0.01 * s]) for s in (1, 2)]
    single = _evaluated(tmp_path / "moe_s1", [0.62, 0.64])
    logistic = _evaluated(tmp_path / "gpt_logistic", [0.55, 0.57], baseline="logistic")
    table = compare([*ref, *var, single, logistic]).set_index("run")
    # rope against gpt on seeds 1 and 2: differences 0.012 and 0.014
    assert table.loc["rope", "seed_n"] == 2 and table.loc["rope", "seed_ci_low"] < 0.012
    assert table.loc["rope", "seed_ci_high"] > 0.014
    assert pd.isna(table.loc["moe", "seed_n"])  # a single seed against three
    assert table.loc["gpt_logistic", "seed_n"] == 3  # the baseline is fixed: one-sample over gpt's seeds
    alone = compare([single, *var]).set_index("run")
    assert pd.isna(alone.loc["rope", "seed_n"])  # a single seeded reference is not fixed either


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
        assert json.loads((out / "evaluate.json").read_text())["controls"] == "at_risk"

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
    """Counts targets first occurring in (L, L + H], however long ago the last event was; a patient
    whose follow-up ends inside the window is censored (weight 0) and the fully observed ones stand in."""
    from modlehr.evaluate.run import calibration
    from modlehr.evaluate.task import LandmarkTask

    inf = float("inf")
    # landmark 50, horizon 2: patient 0 gets the target at 51, patient 1 at 55 (after the window),
    # patient 2's follow-up ends at 51 without it
    task = LandmarkTask(50.0, 2.0, np.arange(3), np.array([[51.0], [55.0], [inf]]), np.array([60.0, 60.0, 51.0]))
    summary, _ = calibration(task, np.full((3, 1), 0.25), death=None)
    assert summary["subjects"] == 3 and summary["fully_observed"] == 2
    assert summary["expected"] == 0.75 and abs(summary["observed"] - 1.5) < 1e-9  # 1 x 1 / G(2) = 1 / (2/3)


def test_calibration_counts_a_death_in_the_window_as_observed():
    from modlehr.evaluate.run import calibration
    from modlehr.evaluate.task import LandmarkTask

    inf = float("inf")
    # target 0 is death: patient 0 dies at 51, so follow-up ends there, but its window is settled
    task = LandmarkTask(50.0, 2.0, np.arange(2), np.array([[51.0, inf], [inf, inf]]), np.array([51.0, 60.0]))
    summary, _ = calibration(task, np.full((2, 2), 0.1), death=0)
    assert summary["fully_observed"] == 2 and summary["observed"] == 1.0


def test_calibration_is_unbiased_under_censoring():
    """Perfect predictions (P(target within the horizon) = 0.5) with heavy independent censoring:
    dropping censored patients would read 0.82, the censoring weights read 1."""
    from modlehr.evaluate.run import calibration
    from modlehr.evaluate.task import LandmarkTask

    rng = np.random.default_rng(0)
    n, landmark, horizon = 20000, 50.0, 2.0
    gap = rng.exponential(horizon / np.log(2), n)
    end = landmark + rng.uniform(0, 4, n)
    first = np.where(landmark + gap <= end, landmark + gap, np.inf)[:, None]
    task = LandmarkTask(landmark, horizon, np.arange(n), first, end)
    summary, _ = calibration(task, np.full((n, 1), 0.5), death=None)
    assert abs(summary["expected_over_observed"] - 1.0) < 0.03


def test_calibration_reads_one_for_the_true_hazards_when_death_competes():
    """Nothing is recorded after death, so the expected count uses the probability that a target comes
    before it. Each target on its own (the marginal risk) read 1.17 here."""
    from modlehr.evaluate.run import calibration
    from modlehr.evaluate.task import LandmarkTask
    from modlehr.model.heads import CompetingRiskHead

    rng = np.random.default_rng(0)
    n, landmark, horizon, death = 200_000, 80.0, 2.0, 2
    rates = np.array([0.3, 0.05, 0.4])  # two diseases, then death
    wait = rng.exponential(1 / rates, size=(n, 3))
    followed = np.minimum(wait[:, death], 10.0)  # until death, or well past the window
    end = landmark + followed
    first = np.where(wait <= followed[:, None], landmark + wait, np.inf)
    task = LandmarkTask(landmark, horizon, np.arange(n), first, end)
    log_rates = torch.log(torch.tensor(rates, dtype=torch.float64)).expand(n, 3)
    observed = CompetingRiskHead.observed_risk(log_rates, horizon, death).numpy()
    summary, _ = calibration(task, observed, death)
    assert abs(summary["expected_over_observed"] - 1.0) < 0.02
    marginal = CompetingRiskHead.risk(log_rates, horizon).numpy()
    assert calibration(task, marginal, death)[0]["expected_over_observed"] > 1.1
    assert np.allclose(CompetingRiskHead.observed_risk(log_rates[:1], horizon), marginal[:1])  # no death target


def test_calibration_does_not_depend_on_the_auc_score(tiny_config):
    """evaluate.risk picks the AUC score; calibration comes from the rates either way."""
    run_prepare(tiny_config.data, log=QUIET)
    best = train(tiny_config, log=QUIET)
    read = lambda: json.loads((Path(tiny_config.train.out_dir) / "evaluate.json").read_text())  # noqa: E731
    evaluate(tiny_config, best, log=QUIET)
    marginal = read()
    tiny_config.evaluate.risk = "next_event"
    evaluate(tiny_config, best, log=QUIET)
    next_event = read()
    assert next_event["risk"] == "next_event" and marginal["mean_auc"] != next_event["mean_auc"]
    assert next_event["calibration"] == marginal["calibration"]

def test_wandb_tracking_with_a_stand_in(tiny_config, monkeypatch):
    """With train.wandb_project set, training logs every check and evaluation adds its summary to the
    same run: one id, kept in <out_dir>/wandb_id."""
    import sys
    import types

    runs = []

    class Run:
        def __init__(self, **kwargs):
            self.kwargs, self.logged, self.summary, self.finished = kwargs, [], {}, False
            runs.append(self)

        def log(self, values, step):
            self.logged.append((step, values))

        def finish(self):
            self.finished = True

    monkeypatch.setitem(sys.modules, "wandb", types.SimpleNamespace(init=lambda **kw: Run(**kw)))
    cfg = copy.deepcopy(tiny_config)
    cfg.train.wandb_project = "modlehr-test"
    run_prepare(cfg.data, log=QUIET)
    evaluate(cfg, train(cfg, log=QUIET), log=QUIET)

    trained, evaluated = runs
    assert trained.kwargs["id"] == evaluated.kwargs["id"] == (Path(cfg.train.out_dir) / "wandb_id").read_text()
    assert trained.kwargs["project"] == "modlehr-test" and trained.kwargs["job_type"] == "train"
    steps = [step for step, _ in trained.logged]
    assert steps == [0, 10, 20, 29] and {"train_loss", "tune_loss", "lr", "steps_per_second"} <= set(trained.logged[-1][1])
    assert "best_tune_loss" in trained.summary and trained.finished
    assert 0 < evaluated.summary["held_out/at_risk/mean_auc"] < 1 and evaluated.finished
    assert "held_out/at_risk/likelihood/nll" in evaluated.summary


def test_tracking_is_off_by_default(tiny_config):
    from modlehr.track import Tracker

    tracker = Tracker(tiny_config, tiny_config.train.out_dir, "train")
    tracker.log({"a": 1.0}, 0)
    tracker.summary({"b": 2.0})
    tracker.finish()
    assert tracker.run is None and not (Path(tiny_config.train.out_dir) / "wandb_id").exists()


def test_delphi_prediction_points_and_bands():
    from modlehr.evaluate.delphi import band_of, prediction_index

    ages = np.array([0.0, 50.0, 50.0, 51.0, 60.0], np.float32)
    # strictly before (time - gap): a diagnosis at 51 with a 0.1-day gap is predicted from the events at 50
    assert list(prediction_index(ages, np.array([51.0, 50.0, 60.5]), 0.1 / 365.25)) == [2, 0, 4]
    assert list(prediction_index(ages, np.array([51.0]), 1.0)) == [0]  # one year back: only age 0 is before 50
    assert list(band_of(np.array([39.9, 40.0, 44.9, 79.9, 80.0]))) == [-1, 0, 0, 7, -1]


def test_delphi_auc_runs_end_to_end(tiny_config):
    from modlehr.evaluate.delphi import delphi_auc

    run_prepare(tiny_config.data, log=QUIET)
    best = train(tiny_config, log=QUIET)
    table = delphi_auc(tiny_config, best, gap_days=0.1, log=QUIET)
    summary = json.loads((Path(tiny_config.train.out_dir) / "evaluate_delphi_gap0.1d.json").read_text())
    assert len(table) and table["auc"].between(0, 1).all() and set(table["sex"]) <= {"female", "male", "all"}
    assert summary["gap_days"] == 0.1 and 0 < summary["mean_auc"] < 1


def test_delphi_band_aucs_match_a_plain_loop():
    """band_aucs reads each target's rows from sorted cases and controls; a plain loop over every
    (target, sex, band), masking all rows each time, gives the same table."""
    from sklearn.metrics import roc_auc_score

    from modlehr.evaluate.delphi import AGE_BANDS, MIN_PER_BAND, band_aucs

    rng = np.random.default_rng(0)
    n_subjects, n_targets, n_cases, n_controls = 300, 6, 900, 700
    targets = [f"T{t}" for t in range(n_targets)]
    first = np.where(rng.random((n_subjects, n_targets)) < 0.3, 50.0, np.inf)
    sex = np.where(rng.random(n_subjects) < 0.5, "female", "male").astype(object)
    k_subject, k_target = rng.integers(0, n_subjects, n_cases), rng.integers(0, n_targets, n_cases)
    k_band, k_score = rng.integers(0, len(AGE_BANDS), n_cases), rng.normal(size=n_cases)
    c_subject, c_band = rng.integers(0, n_subjects, n_controls), rng.integers(0, len(AGE_BANDS), n_controls)
    c_rates = rng.normal(size=(n_controls, n_targets))
    table = band_aucs(targets, first, sex, (k_subject, k_target, k_band, k_score), (c_subject, c_band, c_rates))

    rows = []
    for t, name in enumerate(targets):
        never = np.isinf(first[c_subject, t])
        for sx in np.unique(sex):
            for k in range(len(AGE_BANDS)):
                case = k_score[(k_target == t) & (k_band == k) & (sex[k_subject] == sx)]
                control = c_rates[never & (c_band == k) & (sex[c_subject] == sx), t]
                if len(case) >= MIN_PER_BAND and len(control) >= MIN_PER_BAND:
                    y = np.r_[np.ones(len(case)), np.zeros(len(control))]
                    rows.append((name, sx, int(AGE_BANDS[k]), len(case), len(control),
                                 roc_auc_score(y, np.r_[case, control])))
    assert len(rows) > 50 and [tuple(r) for r in table.itertuples(index=False)] == rows


def test_readme_lists_every_command():
    """The README's command table names every command the parser accepts."""
    import argparse

    commands = next(a for a in build_parser()._actions if isinstance(a, argparse._SubParsersAction)).choices
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text()
    table = readme[readme.index("| Command |"):readme.index("## Installation")]
    assert [c for c in commands if f"`modlehr {c}" not in table] == []


def test_scripts_parse():
    """Every shell script and job file in scripts/ is valid bash."""
    import subprocess

    scripts = sorted((Path(__file__).resolve().parents[1] / "scripts").rglob("*.sh")) + \
        sorted((Path(__file__).resolve().parents[1] / "scripts").rglob("*.sbatch"))
    assert scripts
    for path in scripts:
        result = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
        assert result.returncode == 0, f"{path.name}: {result.stderr}"
