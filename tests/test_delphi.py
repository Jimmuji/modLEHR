"""Delphi-2M in modLEHR: its export to Delphi's format, training it, scoring its checkpoints, and the
check that it trains the same model as Delphi's own code."""

from __future__ import annotations

import dataclasses
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from modlehr.evaluate.run import evaluate
from modlehr.model.delphi import FEMALE, LIFESTYLE, MALE, NO_EVENT, DelphiModel, delphi_ids
from modlehr.prepare.dataset import open_split
from modlehr.prepare.pipeline import run_prepare
from modlehr.prepare.vocabulary import Vocabulary
from modlehr.train.delphi import export
from modlehr.train.loop import train

QUIET = lambda *_: None  # noqa: E731


def _delphi_config(cfg):
    """The tiny config with Delphi-2M as the model (it reads tokens only)."""
    return dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, architecture="delphi2m", value_encoding="none"))


def _rows(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype=np.uint32).reshape(-1, 3).astype(np.int64)


def test_export_writes_delphi_format(tiny_config, tmp_path):
    run_prepare(tiny_config.data, log=QUIET)
    out = tmp_path / "delphi_data"
    counts = export(tiny_config, out, tmp_path / "delphi_run", log=QUIET)
    prepared = Path(tiny_config.data.prepared_dir)
    vocab = Vocabulary.load(prepared / "vocab.json")
    ids = delphi_ids(vocab)
    assert ids[vocab.token_index["GENDER//F"]] == FEMALE and ids[vocab.token_index["GENDER//M"]] == MALE
    assert not set(ids.tolist()) & set(LIFESTYLE)  # Delphi shifts the ages of 3 to 11 at random

    for name, split in (("train", "train"), ("val", "tuning"), ("test", "held_out")):
        rows, data = _rows(out / f"{name}.bin"), open_split(prepared, split)
        patients = rows[:, 0]
        assert np.all(np.diff(patients) >= 0) and len(np.unique(patients)) == len(data) == counts[name]["patients"]
        for p in np.unique(patients)[:20]:
            assert np.all(np.diff(rows[patients == p, 1]) >= 0)  # each record in age order

    lines = (out / "labels.csv").read_text().splitlines()
    assert lines[:2] == ["Padding", "No event"]
    for token in ("GENDER//F", "MEDS_DEATH", "SNOMEDCT//1001"):
        assert lines[ids[vocab.token_index[token]] + 1] == token  # line i + 1 names Delphi id i

    settings: dict = {}
    exec((out / "delphi_config.py").read_text(), settings)
    assert settings["vocab_size"] == len(lines) and settings["dataset"] == str(out.resolve())
    assert (settings["n_layer"], settings["n_embd"], settings["block_size"]) == (2, 32, 64)
    targets = {int(i) + 1 for i in ids[vocab.target_to_input]}
    assert set(settings["ignore_tokens"]) == set(range(len(lines))) - targets - {NO_EVENT + 1}


def test_export_ends_each_record_at_the_end_of_follow_up(tmp_path):
    """Delphi ends a record at its last token; a follow-up that ended at an event prepare dropped
    (here a rare code at 53) gets a "no event" token there."""
    from conftest import write_meds
    from modlehr.config import Config, DataConfig, EventRule

    rows = []
    for s in range(30):
        rows += [(s, None, "GENDER//F", None), (s, 40.0, "DX//A", None), (s, 50.0, "LAB//1", 1.0)]
    rows.append((0, 53.0, "LAB//RARE", 3.0))
    cfg = Config(data=DataConfig(meds_dir=str(write_meds(tmp_path / "meds", rows)), prepared_dir=str(tmp_path / "prepared"),
                                 min_subjects_per_token=5, value_mode="none",
                                 events=[EventRule(name="sex", role="static", prefix="GENDER//"),
                                         EventRule(name="dx", role="target", prefix="DX//"),
                                         EventRule(name="labs", role="measurement", prefix="LAB//")]))
    cfg.evaluate.split = "train"
    run_prepare(cfg.data, log=QUIET)
    export(cfg, tmp_path / "delphi_data", tmp_path / "delphi_run", log=QUIET)
    rows = _rows(tmp_path / "delphi_data" / "test.bin")
    first = rows[rows[:, 0] == 0]
    assert first[-1, 2] == NO_EVENT and abs(first[-1, 1] - 53.0 * 365.25) <= 1
    assert all(rows[rows[:, 0] == p][-1, 2] != NO_EVENT for p in range(1, 30))  # followed to their last event


def test_delphi2m_trains_and_is_scored_like_a_modlehr_model(tiny_config, tmp_path):
    """model.architecture: delphi2m trains Delphi-2M on the prepared data; its checkpoint is read as log-rates
    per year of the modLEHR targets, and each landmark is asked with a "no event" token at the landmark age."""
    from modlehr.evaluate.run import landmark_outputs, load_model
    from modlehr.evaluate.task import landmark_task

    cfg = _delphi_config(tiny_config)
    run_prepare(cfg.data, log=QUIET)
    checkpoint = train(cfg, log=QUIET)
    records = [json.loads(line) for line in (Path(cfg.train.out_dir) / "log.jsonl").read_text().splitlines()]
    assert [r["step"] for r in records] == [10, 20, 30] and all(math.isfinite(r["tune_loss"]) for r in records)

    prepared = Path(cfg.data.prepared_dir)
    vocab = Vocabulary.load(prepared / "vocab.json")
    model, _ = load_model(checkpoint, prepared, "cpu")
    assert isinstance(model, DelphiModel)
    ids = delphi_ids(vocab)
    batch = {"tokens": torch.tensor([[vocab.token_index["GENDER//F"]]]), "ages": torch.zeros(1, 1)}
    with torch.no_grad():
        logits = model.delphi(torch.tensor([[FEMALE + 1]]), torch.zeros(1, 1))[0]
        expected = logits[0, 0, ids[vocab.target_to_input] + 1] + math.log(365.25)  # per day → per year
        np.testing.assert_allclose(model(batch)[0, 0].numpy(), expected.numpy(), rtol=1e-6)

    seen = []
    forward = model.delphi.forward
    model.delphi.forward = lambda idx, age, *a, **k: seen.append((idx.clone(), age.clone())) or forward(idx, age, *a, **k)
    split = open_split(prepared, "held_out")
    landmark_outputs(model, split, landmark_task(split, vocab, 60.0, 10.0).subjects, 60.0, 256, "cpu")
    idx, age = seen[0]
    last = (idx > 0).sum(dim=1) - 1
    rows = torch.arange(len(idx))
    assert torch.all(idx[rows, last] == NO_EVENT + 1) and torch.allclose(age[rows, last], torch.tensor(60.0 * 365.25))
    assert math.isclose(float(age[idx == 0].max()), -10000.0)  # padding at Delphi's padding age

    evaluate(cfg, checkpoint, log=QUIET)
    summary = json.loads((Path(cfg.train.out_dir) / "evaluate.json").read_text())
    assert summary["pairs"] > 0 and "calibration" in summary and "likelihood" in summary

    # Delphi-2M's AUC with --query gap asks at a "no event" token after each point's history
    from modlehr.evaluate.delphi import delphi_auc, log_rates_with_query

    delphi_auc(cfg, checkpoint, 0.1, tmp_path / "gap.csv", log=QUIET, points_dir=tmp_path / "points", query="gap")
    assert json.loads((tmp_path / "gap.json").read_text())["query"] == "gap"
    seen.clear()
    log_rates_with_query(model, split, np.array([[0, 1, 55.0], [1, 2, 61.5]]), 8, "cpu")
    idx, age = seen[0]
    last = (idx > 0).sum(dim=1) - 1
    rows = torch.arange(len(idx))
    assert torch.all(idx[rows, last] == NO_EVENT + 1)
    assert torch.allclose(age[rows, last], torch.tensor([55.0, 61.5]) * 365.25)


DELPHI_REPO = Path(os.environ.get("DELPHI_REPO", Path(__file__).resolve().parents[1] / ".cache" / "delphi"))


@pytest.mark.skipif(not (DELPHI_REPO / "train.py").exists(),
                    reason="needs a clone of Delphi-2M's code (scripts/compare_delphi_code.sh makes one, or set DELPHI_REPO)")
def test_delphi2m_trains_the_same_model_as_delphis_own_code(tiny_config, tmp_path):
    """With the same data and seed, modLEHR's Delphi-2M and Delphi's own train.py end with the same weights."""
    cfg = _delphi_config(tiny_config)
    cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, dropout=0.1))  # dropout draws must match too
    run_prepare(cfg.data, log=QUIET)
    export(cfg, tmp_path / "export", tmp_path / "theirs", log=QUIET)
    subprocess.run([sys.executable, "train.py", str(tmp_path / "export" / "delphi_config.py"), "--device=cpu",
                    "--log_interval=1000"], cwd=DELPHI_REPO, check=True, capture_output=True)
    ours = torch.load(train(cfg, log=QUIET), weights_only=False)
    theirs = torch.load(tmp_path / "theirs" / "ckpt.pt", weights_only=False)
    assert ours["iter_num"] == theirs["iter_num"] and ours["model_args"] == theirs["model_args"]
    for key, value in theirs["model"].items():
        assert torch.equal(ours["model"][key], value), key
