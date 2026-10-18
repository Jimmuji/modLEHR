from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from modlehr.evaluate.run import evaluate, load_model
from modlehr.generate.sample import sample_future
from modlehr.prepare.dataset import open_split
from modlehr.prepare.pipeline import run_prepare
from modlehr.train.loop import train


def test_prepare_train_evaluate_generate(tiny_config):
    quiet = lambda *_: None  # noqa: E731
    run_prepare(tiny_config.data, log=quiet)
    checkpoint = train(tiny_config, log=quiet)
    assert checkpoint.exists()
    log = [json.loads(line) for line in (Path(tiny_config.train.out_dir) / "log.jsonl").read_text().splitlines()]
    assert log[-1]["tune_loss"] < log[0]["tune_loss"] * 1.5  # trains without blowing up

    result = evaluate(tiny_config, checkpoint, log=quiet)
    assert len(result) > 0 and result["auc"].between(0, 1).all()
    summary = json.loads((Path(tiny_config.train.out_dir) / "evaluate.json").read_text())
    held_out = open_split(tiny_config.data.prepared_dir, "held_out")  # every prediction scored once
    assert summary["likelihood"]["predictions"] == int(np.asarray(held_out.arrays["loss_mask"]).sum())

    prepared = Path(tiny_config.data.prepared_dir)
    model, vocab = load_model(checkpoint, prepared, "cpu")
    history = open_split(prepared, "held_out").subject(0)
    events = sample_future(model, vocab, history, start_age=50.0, horizon_years=30.0, seed=1)
    assert all(50.0 < age <= 80.0 and token in vocab.targets for age, token in events)
    already = {vocab.tokens[t] for t in history["tokens"][history["ages"] <= 50.0]}
    sampled = [token for _, token in events]
    assert len(sampled) == len(set(sampled)) and not already & set(sampled)  # first occurrences stay first
