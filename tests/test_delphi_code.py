"""Delphi-2M's own code on modLEHR's data: the export to its format, and scoring its checkpoints."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from modlehr.evaluate.run import evaluate
from modlehr.external.delphi import FEMALE, LIFESTYLE, MALE, NO_EVENT, delphi_ids, export
from modlehr.prepare.dataset import open_split
from modlehr.prepare.pipeline import run_prepare
from modlehr.prepare.vocabulary import Vocabulary

QUIET = lambda *_: None  # noqa: E731

# A stand-in for Delphi's model.py with the same interface: Delphi(DelphiConfig(**model_args)),
# forward(idx, age) -> (logits, loss, attention). Token v gets the rate 0.001 * v per year (logits are
# per day, as Delphi's are), and every input is recorded so the test can check what it was given.
STAND_IN = '''
import math
from dataclasses import dataclass, field
import torch
import torch.nn as nn

SEEN = []

@dataclass
class DelphiConfig:
    block_size: int = 64
    vocab_size: int = 10
    n_layer: int = 1
    n_head: int = 1
    n_embd: int = 4
    ignore_tokens: list = field(default_factory=lambda: [0])

class Delphi(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, idx, age, targets=None, targets_age=None):
        SEEN.append((idx.clone(), age.clone()))
        v = torch.arange(self.config.vocab_size, dtype=torch.float32).clamp(min=1)
        logits = (torch.log(0.001 * v) - math.log(365.25)) * self.scale
        return logits.expand(*idx.shape, -1), None, None
'''


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


def _stand_in_checkpoint(tmp_path: Path, vocab: Vocabulary) -> tuple[Path, Path]:
    repo = tmp_path / "delphi_repo"
    repo.mkdir()
    (repo / "model.py").write_text(STAND_IN)
    n = int(delphi_ids(vocab).max()) + 2
    state = {"model": {"scale": torch.ones(())}, "model_args": {"block_size": 64, "vocab_size": n}}
    torch.save(state, tmp_path / "ckpt.pt")
    return repo, tmp_path / "ckpt.pt"


def test_a_delphi_checkpoint_is_scored_like_a_modlehr_model(tiny_config, tmp_path):
    """Rates come back per year for the modLEHR targets, and each landmark is asked with a "no event"
    token at the landmark age, after the history."""
    from modlehr.evaluate.run import landmark_outputs, load_model
    from modlehr.evaluate.task import landmark_task

    run_prepare(tiny_config.data, log=QUIET)
    prepared = Path(tiny_config.data.prepared_dir)
    vocab = Vocabulary.load(prepared / "vocab.json")
    repo, checkpoint = _stand_in_checkpoint(tmp_path, vocab)
    with pytest.raises(ValueError, match="--delphi-repo"):
        load_model(checkpoint, prepared, "cpu")

    model, _ = load_model(checkpoint, prepared, "cpu", repo)
    ids = delphi_ids(vocab)
    rates = model({"tokens": torch.tensor([[vocab.token_index["GENDER//F"]]]), "ages": torch.zeros(1, 1)}).exp()[0, 0].detach()
    expected = 0.001 * (ids[vocab.target_to_input] + 1)  # the stand-in's rate for Delphi id i is 0.001 (i + 1) a year
    np.testing.assert_allclose(rates.numpy(), expected, rtol=1e-5)

    seen = type(model.delphi).forward.__globals__["SEEN"]
    seen.clear()
    split = open_split(prepared, "held_out")
    landmark_outputs(model, split, landmark_task(split, vocab, 60.0, 10.0).subjects, 60.0, 256, "cpu")
    idx, age = seen[0]
    last = (idx > 0).sum(dim=1) - 1
    rows = torch.arange(len(idx))
    assert torch.all(idx[rows, last] == NO_EVENT + 1) and torch.allclose(age[rows, last], torch.tensor(60.0 * 365.25))
    assert math.isclose(float(age[idx == 0].max()), -10000.0)  # padding at Delphi's padding age

    evaluate(tiny_config, checkpoint, log=QUIET, delphi_repo=repo)
    summary = json.loads((Path(tiny_config.train.out_dir) / "evaluate.json").read_text())
    assert summary["pairs"] > 0 and "calibration" in summary and "likelihood" in summary

    # Delphi-2M's AUC with --query gap asks at a "no event" token after each point's history
    from modlehr.evaluate.delphi import delphi_auc, log_rates_with_query

    delphi_auc(tiny_config, checkpoint, 0.1, tmp_path / "gap.csv", log=QUIET, delphi_repo=repo,
               points_dir=tmp_path / "points", query="gap")
    assert json.loads((tmp_path / "gap.json").read_text())["query"] == "gap"
    seen.clear()
    log_rates_with_query(model, split, np.array([[0, 1, 55.0], [1, 2, 61.5]]), 8, "cpu")
    idx, age = seen[0]
    last = (idx > 0).sum(dim=1) - 1
    rows = torch.arange(len(idx))
    assert torch.all(idx[rows, last] == NO_EVENT + 1)
    assert torch.allclose(age[rows, last], torch.tensor([55.0, 61.5]) * 365.25)
