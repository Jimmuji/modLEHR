"""Delphi-2M's own code on modLEHR's data: the export to its format."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from modlehr.external.delphi import FEMALE, LIFESTYLE, MALE, NO_EVENT, delphi_ids, export
from modlehr.prepare.dataset import open_split
from modlehr.prepare.pipeline import run_prepare
from modlehr.prepare.vocabulary import Vocabulary

QUIET = lambda *_: None  # noqa: E731


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
