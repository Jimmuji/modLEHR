from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from modlehr.config import EventRule
from modlehr.data.toy import NOTE, make_meds
from modlehr.prepare.dataset import open_split, subject_subset
from modlehr.prepare.pipeline import is_current, run_prepare
from modlehr.prepare.subset import SPLITS, default_out, make_subset
from modlehr.train.finetune import task_samples, write_samples
from modlehr.train.loop import train
from test_finetune import write_labels

QUIET = lambda *_: None  # noqa: E731


def same_subject(full, sub, i_full: int, i_sub: int) -> None:
    assert full.subject_ids[i_full] == sub.subject_ids[i_sub]
    a, b = full.subject(i_full), sub.subject(i_sub)
    for k in a:
        if k == "text":
            assert len(a[k]) == len(b[k])
            for x, y in zip(a[k], b[k]):
                assert (x is None and y is None) or np.array_equal(x, y)
        else:
            np.testing.assert_array_equal(a[k], b[k])
    assert full.follow_up_start[i_full] == sub.follow_up_start[i_sub]
    assert full.follow_up_end[i_full] == sub.follow_up_end[i_sub]


def test_subset_keeps_whole_subjects_and_the_training_fraction(tiny_config, tmp_path):
    run_prepare(tiny_config.data, log=QUIET)
    src = Path(tiny_config.data.prepared_dir)
    dst = make_subset(src, 0.5, log=QUIET)
    assert dst == default_out(src, 0.5) == Path(f"{src}_50pct")
    assert (dst / "vocab.json").read_bytes() == (src / "vocab.json").read_bytes()
    meta = json.loads((dst / "prepare.json").read_text())
    assert meta["subset"]["fraction"] == 0.5 and meta["data_config"]["prepared_dir"] == str(dst)
    for split in SPLITS:
        full, sub = open_split(src, split), open_split(dst, split)
        keep = subject_subset(len(full), 0.5)  # the subjects train.data_fraction 0.5 trains on
        assert len(sub) == len(keep) == meta["summary"][split]["subjects"]
        assert int(sub.offsets[-1]) == meta["summary"][split]["events"]
        for j, i in enumerate(keep):
            same_subject(full, sub, int(i), j)
    # the data config still matches, so `modlehr run` trains on it without preparing again
    sub_cfg = copy.deepcopy(tiny_config.data)
    sub_cfg.prepared_dir = str(dst)
    assert is_current(sub_cfg)
    with pytest.raises(ValueError, match="subset"):
        run_prepare(sub_cfg, log=QUIET)
    with pytest.raises(ValueError, match="itself a subset"):
        make_subset(dst, 0.5, out=tmp_path / "again", log=QUIET)
    # a smaller fraction keeps a subset of a larger one
    small = make_subset(src, 0.2, log=QUIET)
    assert set(open_split(small, "train").subject_ids) <= set(open_split(dst, "train").subject_ids)


def test_subset_carries_task_samples_and_trains(tiny_config, tmp_path):
    run_prepare(tiny_config.data, log=QUIET)
    tiny_config.finetune.labels = str(write_labels(Path(tiny_config.data.meds_dir), tmp_path / "labels" / "a60", 60.0))
    write_samples(tiny_config, log=QUIET)
    src = Path(tiny_config.data.prepared_dir)
    dst = make_subset(src, 0.5, log=QUIET)
    cfg = copy.deepcopy(tiny_config)
    cfg.data.prepared_dir = str(dst)
    logged = []
    for split in SPLITS:
        full, sub = task_samples(tiny_config, split, QUIET), task_samples(cfg, split, logged.append)
        assert not logged  # read from the carried cache, not rebuilt from MEDS
        kept = set(open_split(dst, split).subject_ids)
        rows = np.isin(full["subject_id"], list(kept))
        assert rows.sum() == len(sub["subject_id"]) > 0
        for k in ("subject_id", "prediction_time", "label", "n", "strict"):
            np.testing.assert_array_equal(full[k][rows], sub[k])
        np.testing.assert_array_equal(open_split(dst, split).subject_ids[sub["index"]], sub["subject_id"])
    cfg.train.out_dir = str(tmp_path / "run_subset")
    assert Path(train(cfg, log=QUIET)).exists()


def test_subset_keeps_each_events_notes(tiny_config, tmp_path):
    cfg = copy.deepcopy(tiny_config)
    cfg.data.meds_dir = str(make_meds(tmp_path / "meds_notes", notes=True))
    cfg.data.prepared_dir = str(tmp_path / "prepared_notes")
    cfg.data.events = [*cfg.data.events, EventRule(name="notes", role="text", codes=[NOTE])]
    run_prepare(cfg.data, log=QUIET)
    src = Path(cfg.data.prepared_dir)
    dst = make_subset(src, 0.3, log=QUIET)
    for split in SPLITS:
        full, sub = open_split(src, split), open_split(dst, split)
        assert sub.has_text
        for j, i in enumerate(subject_subset(len(full), 0.3)):
            same_subject(full, sub, int(i), j)


def test_subset_command(tiny_config, tmp_path):
    from modlehr.cli import main
    run_prepare(tiny_config.data, log=QUIET)
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(tiny_config.to_dict()))
    main(["subset", "--config", str(path), "--fraction", "0.25"])
    assert (Path(f"{tiny_config.data.prepared_dir}_25pct") / "train" / "tokens.npy").exists()


def test_task_samples_on_a_subset_made_before_them(tiny_config, tmp_path):
    """Samples asked for on a subset that did not carry them come from the full data, for its subjects only."""
    run_prepare(tiny_config.data, log=QUIET)
    src = Path(tiny_config.data.prepared_dir)
    dst = make_subset(src, 0.5, log=QUIET)
    tiny_config.finetune.labels = str(write_labels(Path(tiny_config.data.meds_dir), tmp_path / "labels" / "a60", 60.0))
    cfg = copy.deepcopy(tiny_config)
    cfg.data.prepared_dir = str(dst)
    for split in SPLITS:
        sub = task_samples(cfg, split, QUIET)
        full = task_samples(tiny_config, split, QUIET)
        rows = np.isin(full["subject_id"], open_split(dst, split).subject_ids)
        np.testing.assert_array_equal(full["n"][rows], sub["n"])
        np.testing.assert_array_equal(open_split(dst, split).subject_ids[sub["index"]], sub["subject_id"])
        assert (dst / "tasks" / "a60" / f"{split}.npz").exists()


def test_ensure_subset_remakes_a_stale_one(tiny_config):
    from modlehr.prepare.subset import ensure_subset, is_current_subset
    run_prepare(tiny_config.data, log=QUIET)
    src = Path(tiny_config.data.prepared_dir)
    dst = ensure_subset(src, 0.5, log=QUIET)
    assert is_current_subset(src, 0.5)
    made = (dst / "prepare.json").stat().st_mtime_ns
    assert ensure_subset(src, 0.5, log=QUIET) == dst and (dst / "prepare.json").stat().st_mtime_ns == made
    run_prepare(tiny_config.data, log=QUIET)  # prepared again from the same files: the same data
    assert is_current_subset(src, 0.5)
    tiny_config.data.min_subjects_per_token = 3  # prepared again with another config
    run_prepare(tiny_config.data, log=QUIET)
    assert not is_current_subset(src, 0.5)
    ensure_subset(src, 0.5, log=QUIET)
    assert is_current_subset(src, 0.5)
