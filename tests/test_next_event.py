"""`modlehr next-event`: a next_token model's cross-entropy, top-k accuracy and event groups, checked against a
direct pass over each held-out timeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from modlehr.evaluate.next_event import family, group_members, next_event, overlapping_windows
from modlehr.evaluate.run import load_model
from modlehr.prepare.dataset import collate, open_split
from modlehr.prepare.pipeline import run_prepare
from modlehr.train.loop import train

QUIET = lambda *_: None  # noqa: E731


def test_overlapping_windows_score_every_next_event_once(tiny_config):
    run_prepare(tiny_config.data, log=QUIET)
    split = open_split(tiny_config.data.prepared_dir, "held_out")
    subjects = np.arange(len(split))
    expected = int(np.clip(split.lengths() - 1, 0, None).sum())
    for length in (2, 5, 64):
        rows = overlapping_windows(split, subjects, length)
        assert sum(len(r["tokens"]) - 1 for r in rows) == expected
        assert max(len(r["tokens"]) for r in rows) <= length


def test_a_group_must_match_a_token():
    assert group_members(["<pad>", "ICU//A", "ICU//B", "LAB"], {"icu": "^ICU//"})["icu"].tolist() == [1, 2]
    with pytest.raises(ValueError, match="matches no token"):
        group_members(["<pad>", "LAB"], {"icu": "^ICU//"})


def test_next_event_scores_each_next_event_as_a_full_pass_would(tiny_config):
    cfg = tiny_config
    cfg.model.head, cfg.model.max_len = "next_token", 512
    cfg.evaluate.next_event_groups = {"death": "^MEDS_DEATH$", "disorders": "^SNOMEDCT//"}
    run_prepare(cfg.data, log=QUIET)
    split = open_split(cfg.data.prepared_dir, "held_out")
    assert split.lengths().max() <= cfg.model.max_len  # one window per timeline, so a full pass sees the same history
    best = train(cfg, log=QUIET)
    summary = next_event(cfg, best, log=QUIET)

    model, vocab = load_model(best, Path(cfg.data.prepared_dir), torch.device("cpu"))
    disorders = group_members(list(vocab.tokens), cfg.evaluate.next_event_groups)["disorders"]
    loss, top1, top10, p, y, fam = [], [], [], [], [], []
    with torch.no_grad():
        for i in range(len(split)):
            row = split.subject(i, labels=False)
            if len(row["tokens"]) < 2:
                continue
            logp = model(collate([row]))[0, :len(row["tokens"]) - 1].log_softmax(-1)
            nxt = torch.tensor(np.array(row["tokens"][1:]), dtype=torch.long)
            loss += (-logp.gather(1, nxt[:, None]).squeeze(1)).tolist()
            top1 += (logp.argmax(-1) == nxt).tolist()
            top10 += (logp.topk(min(10, logp.shape[-1]), dim=-1).indices == nxt[:, None]).any(1).tolist()
            p += logp[:, torch.from_numpy(disorders)].exp().sum(-1).tolist()
            y += np.isin(nxt.numpy(), disorders).tolist()
            fam += [family(vocab.tokens[int(k)]) for k in nxt]
    assert summary["predictions"] == len(loss) > 0
    assert summary["nll"] == pytest.approx(np.mean(loss), rel=1e-4)
    assert summary["perplexity"] == pytest.approx(np.exp(np.mean(loss)), rel=1e-4)
    assert summary["top1"] == pytest.approx(np.mean(top1)) and summary["top10"] == pytest.approx(np.mean(top10))
    assert summary["top1"] <= summary["top5"] <= summary["top10"] <= 1
    group = summary["groups"]["disorders"]
    assert group["prevalence"] == pytest.approx(np.mean(y)) and group["positives"] == sum(y)
    assert group["auroc"] == pytest.approx(roc_auc_score(y, p), abs=1e-5)
    assert group["auprc"] == pytest.approx(average_precision_score(y, p), abs=1e-5)
    assert group["mean_predicted"] == pytest.approx(np.mean(p), rel=1e-4)
    assert sum(f["share"] for f in summary["by_family"].values()) == pytest.approx(1.0)
    fam, loss, top1 = np.array(fam), np.array(loss), np.array(top1)
    assert set(summary["by_family"]) == set(fam) and len(set(fam)) > 1
    for name, f in summary["by_family"].items():  # each family's numbers are its own events'
        assert f["share"] == pytest.approx(np.mean(fam == name))
        assert f["nll"] == pytest.approx(loss[fam == name].mean(), rel=1e-4)
        assert f["top1"] == pytest.approx(top1[fam == name].mean())
    saved = json.loads((Path(cfg.train.out_dir) / "next_event.json").read_text())
    assert saved["predictions"] == summary["predictions"] and set(saved["groups"]) == {"death", "disorders"}


def test_next_event_refuses_a_model_without_the_next_token_head(tiny_config):
    run_prepare(tiny_config.data, log=QUIET)
    tiny_config.train.steps = 2
    best = train(tiny_config, log=QUIET)
    with pytest.raises(ValueError, match="next_token"):
        next_event(tiny_config, best, log=QUIET)
