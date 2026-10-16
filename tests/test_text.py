"""Clinical notes: stored with the right event, read by the model, and never seen by earlier events."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from conftest import NOTE, make_meds
from modlehr.config import EventRule
from modlehr.data.meds_io import open_meds
from modlehr.evaluate.run import evaluate, load_model
from modlehr.prepare.dataset import collate, open_split
from modlehr.prepare.pipeline import run_prepare
from modlehr.prepare.text import hash_words
from modlehr.prepare.vocabulary import Vocabulary
from modlehr.train.loop import train

QUIET = lambda *_: None  # noqa: E731


@pytest.fixture
def notes_config(tiny_config, tmp_path):
    cfg = copy.deepcopy(tiny_config)
    cfg.data.meds_dir = str(make_meds(tmp_path / "meds_notes", notes=True))
    cfg.data.prepared_dir = str(tmp_path / "prepared_notes")
    cfg.data.events = [*cfg.data.events, EventRule(name="notes", role="text", codes=[NOTE])]
    cfg.model.text_encoding = "hashed_bag"
    cfg.train.out_dir = str(tmp_path / "run_notes")
    return cfg


def test_notes_stay_with_their_event(notes_config):
    meta = run_prepare(notes_config.data, log=QUIET)
    assert meta["text_buckets"] == notes_config.data.text_buckets
    prepared = Path(notes_config.data.prepared_dir)
    vocab = Vocabulary.load(prepared / "vocab.json")
    events = open_meds(notes_config.data.meds_dir)
    truth = {}
    for shard in events.shards():
        df = events.read_shard(shard)
        notes = df[df["code"] == NOTE]
        truth.update(dict(zip(notes["subject_id"], notes["text_value"])))
    note_token = vocab.token_index[NOTE]
    for split in ("train", "held_out"):
        data = open_split(prepared, split)
        assert data.has_text
        for i in range(len(data)):
            row = data.subject(i)
            at = np.flatnonzero(row["tokens"] == note_token)
            assert len(at) == 1
            words = row["text"][at[0]]
            expected = hash_words(truth[int(data.subject_ids[i])], notes_config.data.text_buckets, 256)
            assert np.array_equal(words, expected)
            assert sum(w is not None for w in row["text"]) == 1


def test_model_reads_notes_causally(notes_config):
    run_prepare(notes_config.data, log=QUIET)
    checkpoint = train(notes_config, log=QUIET)
    prepared = Path(notes_config.data.prepared_dir)
    model, vocab = load_model(checkpoint, prepared, "cpu")
    data = open_split(prepared, "held_out")
    row = data.subject(0)
    at = int(np.flatnonzero(row["tokens"] == vocab.token_index[NOTE])[0])
    changed = copy.deepcopy(row)
    changed["text"][at] = hash_words("completely different words here", notes_config.data.text_buckets, 256)
    with torch.no_grad():
        a, b = model(collate([row])), model(collate([changed]))
    assert torch.allclose(a[0, :at], b[0, :at], atol=1e-6)  # earlier events cannot see the note
    assert not torch.allclose(a[0, at:], b[0, at:], atol=1e-6)  # the note's event and later ones do
    assert len(evaluate(notes_config, checkpoint, log=QUIET)) > 0


def test_text_encoding_needs_text(tiny_config):
    run_prepare(tiny_config.data, log=QUIET)
    cfg = copy.deepcopy(tiny_config)
    cfg.model.text_encoding = "hashed_bag"
    with pytest.raises(ValueError):
        train(cfg, log=QUIET)


def test_mimic_note_merge_keeps_meds_valid(tiny_meds, tmp_path, tiny_config):
    """Notes in MIMIC-IV-Note's column layout are merged into the shard that holds each subject."""
    import pandas as pd

    from modlehr.data.adapters.mimic_iv_note import convert

    raw = tmp_path / "note"
    raw.mkdir()
    rows = [{"note_id": f"{s}-DS-1", "subject_id": s, "hadm_id": 1, "note_type": "DS", "note_seq": 1,
             "charttime": "2000-01-01 10:00:00", "storetime": None, "text": f"note for subject {s}"}
            for s in range(0, 160, 3)] + [{"note_id": "x", "subject_id": 99999, "hadm_id": 1, "note_type": "DS",
                                           "note_seq": 1, "charttime": "2000-01-01", "storetime": None, "text": "orphan"}]
    pd.DataFrame(rows).to_csv(raw / "discharge.csv.gz", index=False)
    out = tmp_path / "merged"
    summary = convert(raw, tiny_meds, out, kinds=("discharge",))
    assert summary == {"notes_read": len(rows), "notes_added": len(rows) - 1, "shards": 3}
    ds = open_meds(out, validate=True)
    assert "NOTE//DISCHARGE" in set(ds.codes["code"])
    seen = {}
    for shard in ds.shards():
        df = ds.read_shard(shard)
        for s in df["subject_id"].unique():
            assert s not in seen  # each subject still in exactly one shard
            seen[s] = shard
        notes = df[df["code"] == "NOTE//DISCHARGE"]
        assert all(t == f"note for subject {s}" for s, t in zip(notes["subject_id"], notes["text_value"]))

    cfg = copy.deepcopy(tiny_config)
    cfg.data.meds_dir, cfg.data.prepared_dir = str(out), str(tmp_path / "prepared_merged")
    cfg.data.events = [*cfg.data.events, EventRule(name="notes", role="text", prefix="NOTE//")]
    cfg.data.min_subjects_per_token = 2
    run_prepare(cfg.data, log=QUIET)
    assert open_split(cfg.data.prepared_dir, "train").has_text
