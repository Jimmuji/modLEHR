"""Delphi-2M's own code on modLEHR's data: export a prepared dataset to its format, and score its
checkpoints with modLEHR's evaluation.

Delphi-2M (Shmatko et al., Nature 2025; github.com/gerstung-lab/Delphi) trains on a flat array of
(patient, age in days, token) rows of uint32. `export` writes a prepared dataset that way, so its
train.py runs unchanged on the same subjects, events and splits as a modLEHR model:

    train.bin val.bin test.bin   the train, tuning and evaluation splits
    labels.csv                   one name per token, in Delphi's layout ("Padding", "No event", then id 1, 2, ...)
    delphi_config.py             a config for Delphi's train.py: this dataset, its vocabulary, the
                                 tokens left out of the loss, and the size and schedule of the modLEHR run
    mapping.json                 token name → Delphi id

Delphi's ids follow its own conventions: 0 is "no event" (it inserts these tokens itself, one every
five years on average), 1 and 2 are female and male (its AUC code reads sex from them), and 3 to 11
are left unused, since its training shifts the ages of tokens 3 to 11 at random (its lifestyle
tokens). Inputs that are not targets (admissions, measurement bins) come next and are left out of
the loss, as Delphi-2M leaves out sex and lifestyle; then the targets, in modLEHR's order.

Each subject's events run to the end of follow-up. When the last of them comes earlier (follow-up
ended at an event prepare dropped), a "no event" token is written at the end of follow-up, so
Delphi, which ends a record at its last token, sees the same follow-up.

`DelphiModel` wraps a checkpoint of Delphi's train.py so that `modlehr evaluate` and `modlehr
delphi` score it like a modLEHR model: its log-rates per day for the modLEHR targets, per year. At
a landmark it is asked the way Delphi is meant to be asked about a time with no event: a "no event"
token at the landmark age is appended to the history.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import meds
import numpy as np
import torch
import torch.nn as nn

from modlehr.config import Config
from modlehr.evaluate.delphi import SEX_TOKENS
from modlehr.model.heads import CompetingRiskHead
from modlehr.prepare.dataset import open_split
from modlehr.prepare.vocabulary import Vocabulary

DAYS_PER_YEAR = 365.25
NO_EVENT = 0                 # Delphi's id for "no event"; the model sees every id shifted by one
FEMALE, MALE = 1, 2
LIFESTYLE = range(3, 12)     # ids whose ages Delphi's training shifts at random
MASK_TIME = -10000.0         # Delphi's age for padding


def delphi_ids(vocab: Vocabulary) -> np.ndarray:
    """Delphi id of every modLEHR input token (-1 for padding). The same vocabulary always gives the
    same ids, so a checkpoint trained on an export can be scored without the export."""
    ids = np.full(len(vocab.tokens), -1, dtype=np.int64)
    for token, delphi_id in ((SEX_TOKENS["female"], FEMALE), (SEX_TOKENS["male"], MALE)):
        if token in vocab.token_index:
            ids[vocab.token_index[token]] = delphi_id
    is_target = vocab.input_to_target >= 0
    others = [i for i in range(1, len(vocab.tokens)) if ids[i] < 0 and not is_target[i]]
    ids[others] = LIFESTYLE.stop + np.arange(len(others))
    ids[vocab.target_to_input] = LIFESTYLE.stop + len(others) + np.arange(len(vocab.targets))
    return ids


def _rows(split, ids: np.ndarray) -> np.ndarray:
    """[n, 3] uint32 (patient, age in days, Delphi id), patient = index in the split, events up to
    the end of follow-up, then a "no event" at the end of follow-up when no event is there."""
    offsets, end = split.offsets, split.follow_up_end
    patient = np.repeat(np.arange(len(split)), np.diff(offsets))
    ages = np.asarray(split.arrays["ages"], dtype=np.float64)
    days = np.rint(ages * DAYS_PER_YEAR).astype(np.int64)
    keep = ages <= end[patient]
    end_days = np.rint(end.astype(np.float64) * DAYS_PER_YEAR).astype(np.int64)
    last = np.full(len(split), -1, dtype=np.int64)
    np.maximum.at(last, patient[keep], days[keep])
    closing = np.flatnonzero(end_days > last)
    rows = np.concatenate([np.stack([patient[keep], days[keep], ids[np.asarray(split.arrays["tokens"])[keep]]], 1),
                           np.stack([closing, end_days[closing], np.full(len(closing), NO_EVENT)], 1)])
    order = np.lexsort((np.r_[np.zeros(keep.sum()), np.ones(len(closing))], rows[:, 1], rows[:, 0]))
    return rows[order].astype(np.uint32)


def export(cfg: Config, out_dir: str | Path, delphi_out_dir: str | Path, log=print) -> dict:
    prepared, out_dir = Path(cfg.data.prepared_dir), Path(out_dir).resolve()
    vocab = Vocabulary.load(prepared / "vocab.json")
    ids = delphi_ids(vocab)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, split in (("train", meds.train_split), ("val", meds.tuning_split), ("test", cfg.evaluate.split)):
        rows = _rows(open_split(prepared, split), ids)
        rows.tofile(out_dir / f"{name}.bin")
        counts[name] = {"patients": int(len(np.unique(rows[:, 0]))), "rows": int(len(rows))}
    names = {int(i): vocab.tokens[t] for t, i in enumerate(ids) if i >= 0}
    n_ids = int(ids.max()) + 1
    lines = ["Padding", "No event"] + [names.get(i, "Unused") for i in range(1, n_ids)]
    (out_dir / "labels.csv").write_text("\n".join(lines) + "\n")
    targets = set(ids[vocab.target_to_input].tolist())
    ignore = [0] + [i + 1 for i in range(1, n_ids) if i not in targets]  # padding, then every id but the targets
    m, t = cfg.model, cfg.train
    settings = {  # Delphi's own choices (time floor, no-event rate, tie masking), at the modLEHR run's size and schedule
        "dataset": str(out_dir), "out_dir": str(Path(delphi_out_dir).resolve()), "vocab_size": n_ids + 1,
        "ignore_tokens": ignore, "n_layer": m.n_layers, "n_head": m.n_heads, "n_embd": m.d_model,
        "block_size": m.max_len, "dropout": m.dropout, "batch_size": t.batch_size, "max_iters": t.steps,
        "lr_decay_iters": t.steps, "warmup_iters": t.warmup, "learning_rate": t.lr, "min_lr": t.min_lr,
        "weight_decay": t.weight_decay, "beta2": 0.99, "eval_interval": t.eval_every, "eval_iters": t.eval_batches,
        "seed": t.seed, "t_min": 0.1, "mask_ties": True, "no_event_token_rate": 5, "always_save_checkpoint": False,
    }
    (out_dir / "delphi_config.py").write_text(
        "# Written by `modlehr export-delphi` for Delphi-2M's train.py: python train.py THIS_FILE [--device=cuda]\n"
        + "".join(f"{k} = {v!r}\n" for k, v in settings.items()))
    (out_dir / "mapping.json").write_text(json.dumps({"vocab_size": n_ids + 1, "ignore_tokens": ignore,
                                                      "ids": {vocab.tokens[t]: int(i) for t, i in enumerate(ids) if i >= 0},
                                                      "splits": counts}, indent=1))
    log(f"[export-delphi] {counts['train']['patients']:,} training, {counts['val']['patients']:,} tuning and "
        f"{counts['test']['patients']:,} {cfg.evaluate.split} patients, {n_ids + 1:,} Delphi tokens → {out_dir}")
    return counts


def is_delphi_checkpoint(state: dict) -> bool:
    return "model_args" in state and "model_config" not in state


class DelphiModel(nn.Module):
    """A checkpoint of Delphi-2M's train.py, read like a modLEHR model: forward(batch) gives the
    log-rates per year of the modLEHR targets at every event."""

    def __init__(self, repo: str | Path, state: dict, vocab: Vocabulary):
        super().__init__()
        path = Path(repo).expanduser() / "model.py"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; --delphi-repo should be a clone of github.com/gerstung-lab/Delphi")
        spec = importlib.util.spec_from_file_location("delphi_model", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.delphi = module.Delphi(module.DelphiConfig(**state["model_args"]))
        weights = {k.removeprefix("_orig_mod."): v for k, v in state["model"].items()}
        self.delphi.load_state_dict(weights)
        ids = delphi_ids(vocab)
        if ids.max() + 2 > state["model_args"]["vocab_size"]:
            raise ValueError("the checkpoint's vocabulary is smaller than this prepared dataset's; "
                             "was it trained on an export of different prepared data?")
        self.query_token = len(vocab.tokens)  # an input id beyond the vocabulary: Delphi's "no event"
        lookup = np.r_[ids + 1, NO_EVENT + 1]
        lookup[0] = 0  # padding
        self.register_buffer("lookup", torch.as_tensor(lookup), persistent=False)
        self.register_buffer("columns", torch.as_tensor(ids[vocab.target_to_input] + 1), persistent=False)
        self.cfg = SimpleNamespace(max_len=int(state["model_args"]["block_size"]))
        self.head = CompetingRiskHead  # loss, risk, observed_risk and sample work on any log-rates per year

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        tokens = batch["tokens"]
        idx = self.lookup[tokens]
        age = torch.where(tokens != 0, batch["ages"] * DAYS_PER_YEAR, torch.full_like(batch["ages"], MASK_TIME))
        # Delphi's forward also returns every layer's attention weights; keep those under about 1 GB
        c = self.delphi.config
        rows = max(1, 2**28 // (c.n_layer * c.n_head * tokens.shape[1] ** 2))
        logits = torch.cat([self.delphi(idx[i:i + rows], age[i:i + rows])[0] for i in range(0, len(idx), rows)])
        return logits[..., self.columns].float() + math.log(DAYS_PER_YEAR)  # its log-rates are per day
