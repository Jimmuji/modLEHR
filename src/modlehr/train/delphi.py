"""Delphi-2M's data format, batches and training, on modLEHR's prepared data.

`model.architecture: delphi2m` makes `modlehr train` (and `run`) call `train_delphi`, a port of
Delphi's `train.py` at commit fb72166 (MIT License, Gerstung Lab): its batches (the first block of
each record, a "no event" token at a random age every five years on average, lifestyle ages shifted),
its two losses, its tuning loss and checkpoint rule, its learning-rate schedule and optimizer, in the
same order of random draws, so with the same seed it trains the same model as Delphi's own code on
the export below. Size and schedule come from the modLEHR config; Delphi's own choices are kept: a
time floor of 0.1 day, tie masking and the "no event" rate.

`export` writes the same data for Delphi's own code (`modlehr export-delphi`):

    train.bin val.bin test.bin   the train, tuning and evaluation splits, rows of (patient, age in days, id), uint32
    labels.csv                   one name per id, in Delphi's layout ("Padding", "No event", then id 1, 2, ...)
    delphi_config.py             a config for Delphi's train.py: this dataset, its vocabulary, the tokens left
                                 out of the loss, and the size and schedule of the modLEHR run
    mapping.json                 token name → Delphi id

Each record runs to the end of follow-up: when follow-up ends after the last event written (it ended
at an event prepare dropped), a "no event" token closes the record there, so Delphi, which ends a
record at its last token, sees the same follow-up. Inputs that are not targets (admissions,
measurement bins) are left out of the loss, as Delphi-2M leaves out sex and lifestyle.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import meds
import numpy as np
import torch
import yaml

from modlehr.config import Config
from modlehr.model.delphi import (DAYS_PER_YEAR, LIFESTYLE, MASK_TIME, NO_EVENT, Delphi, DelphiConfig,
                                  delphi_ids)
from modlehr.prepare.dataset import open_split
from modlehr.prepare.vocabulary import Vocabulary
from modlehr.track import Tracker

T_MIN, NO_EVENT_RATE, BETA2 = 0.1, 5, 0.99   # Delphi-2M's time floor (days), "no event" rate (years), Adam beta2


def delphi_rows(split, ids: np.ndarray) -> np.ndarray:
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


def model_args(cfg: Config, vocab: Vocabulary) -> dict:
    """Delphi's model arguments at the modLEHR run's size, with Delphi-2M's own time floor and tie masking.
    Every id but the targets and "no event" is left out of the loss."""
    ids = delphi_ids(vocab)
    n_ids = int(ids.max()) + 1
    targets = set(ids[vocab.target_to_input].tolist())
    ignore = [0] + [i + 1 for i in range(1, n_ids) if i not in targets]  # padding, then every id but the targets
    m = cfg.model
    return {"n_layer": m.n_layers, "n_head": m.n_heads, "n_embd": m.d_model, "block_size": m.max_len, "bias": False,
            "vocab_size": n_ids + 1, "dropout": m.dropout, "token_dropout": 0.0, "t_min": T_MIN, "mask_ties": True,
            "ignore_tokens": ignore}


def export(cfg: Config, out_dir: str | Path, delphi_out_dir: str | Path, log=print) -> dict:
    prepared, out_dir = Path(cfg.data.prepared_dir), Path(out_dir).resolve()
    vocab = Vocabulary.load(prepared / "vocab.json")
    ids = delphi_ids(vocab)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, split in (("train", meds.train_split), ("val", meds.tuning_split), ("test", cfg.evaluate.split)):
        rows = delphi_rows(open_split(prepared, split), ids)
        rows.tofile(out_dir / f"{name}.bin")
        counts[name] = {"patients": int(len(np.unique(rows[:, 0]))), "rows": int(len(rows))}
    names = {int(i): vocab.tokens[t] for t, i in enumerate(ids) if i >= 0}
    args = model_args(cfg, vocab)
    lines = ["Padding", "No event"] + [names.get(i, "Unused") for i in range(1, args["vocab_size"] - 1)]
    (out_dir / "labels.csv").write_text("\n".join(lines) + "\n")
    t = cfg.train
    settings = {"dataset": str(out_dir), "out_dir": str(Path(delphi_out_dir).resolve()),
                **{k: args[k] for k in ("vocab_size", "ignore_tokens", "n_layer", "n_head", "n_embd", "block_size", "dropout")},
                "batch_size": t.batch_size, "max_iters": t.steps, "lr_decay_iters": t.steps, "warmup_iters": t.warmup,
                "learning_rate": t.lr, "min_lr": t.min_lr, "weight_decay": t.weight_decay, "beta2": BETA2,
                "eval_interval": t.eval_every, "eval_iters": t.eval_batches, "seed": t.seed, "t_min": T_MIN,
                "mask_ties": True, "no_event_token_rate": NO_EVENT_RATE, "always_save_checkpoint": False}
    (out_dir / "delphi_config.py").write_text(
        "# Written by `modlehr export-delphi` for Delphi-2M's train.py: python train.py THIS_FILE [--device=cuda]\n"
        + "".join(f"{k} = {v!r}\n" for k, v in settings.items()))
    (out_dir / "mapping.json").write_text(json.dumps({"vocab_size": args["vocab_size"], "ignore_tokens": args["ignore_tokens"],
                                                      "ids": {vocab.tokens[t]: int(i) for t, i in enumerate(ids) if i >= 0},
                                                      "splits": counts}, indent=1))
    log(f"[export-delphi] {counts['train']['patients']:,} training, {counts['val']['patients']:,} tuning and "
        f"{counts['test']['patients']:,} {cfg.evaluate.split} patients, {args['vocab_size']:,} Delphi tokens → {out_dir}")
    return counts


def get_p2i(data: np.ndarray) -> np.ndarray:
    """[patients, 2]: each patient's first row and number of rows."""
    starts = np.r_[0, np.flatnonzero(np.diff(data[:, 0].astype(np.int64))) + 1]
    return np.stack([starts, np.diff(np.r_[starts, len(data)])], 1)


def get_batch(ix, data, p2i, select="left", padding="regular", block_size=48, device="cpu",
              lifestyle_augmentations=False, no_event_token_rate=NO_EVENT_RATE, cut_batch=False):
    """Delphi's batches: the first block of each record (select "left"), "no event" tokens on a regular
    grid (tuning) or at random ages (training) up to the last token, lifestyle ages shifted at random
    in training, every id shifted by one so 0 is padding. Returns inputs, input ages, targets, target ages."""
    x = torch.tensor(np.array([p2i[int(i)] for i in ix]))
    ix = ix.clone() if torch.is_tensor(ix) else torch.tensor(np.array(ix))
    gen = torch.Generator(device="cpu")
    gen.manual_seed(ix.sum().item())  # random, but the same for the same patients

    if select == "left":
        traj_start_idx = x[:, 0]
    elif select == "right":
        traj_start_idx = torch.clamp(x[:, 0] + x[:, 1] - block_size - 1, 0, data.shape[0])
    elif select == "random":
        traj_start_idx = x[:, 0] + (torch.randint(2**63 - 1, (len(ix),), generator=gen) % torch.clamp(x[:, 1] - block_size, 1))
        traj_start_idx = torch.clamp(traj_start_idx, 0, data.shape[0])
    else:
        raise NotImplementedError(select)
    traj_start_idx = torch.clamp(traj_start_idx, 0, data.shape[0] - block_size - 1).numpy()
    batch_idx = np.arange(block_size + 1)[None, :] + traj_start_idx[:, None]

    mask = torch.from_numpy(data[:, 0][batch_idx].astype(np.int64))
    mask = mask == torch.tensor(data[p2i[ix.numpy()][:, 0], 0][:, None].astype(np.int64)).to(mask.dtype)
    tokens = torch.from_numpy(data[:, 2][batch_idx].astype(np.int64)).masked_fill(~mask, -1)
    ages = torch.from_numpy(data[:, 1][batch_idx].astype(np.float32)).masked_fill(~mask, MASK_TIME)

    if padding is None or padding.lower() == "none" or not no_event_token_rate:
        pad = torch.ones(len(ix), 0)
    elif padding == "regular":
        pad = torch.arange(0, 36525, 365.25 * no_event_token_rate) * torch.ones(len(ix), 1) + 1
    elif padding == "random":
        pad = torch.randint(1, 36525, (len(ix), int(100 / no_event_token_rate)), generator=gen)
    else:
        raise NotImplementedError(padding)
    m = ages.max(1, keepdim=True).values
    if lifestyle_augmentations:  # after m, so shifted ages do not let more "no event" tokens through
        lifestyle_idx = (tokens >= LIFESTYLE.start) * (tokens < LIFESTYLE.stop)
        if lifestyle_idx.sum():
            ages[lifestyle_idx] += torch.randint(-20 * 365, 365 * 40, (int(lifestyle_idx.sum()),), generator=gen).float()
    tokens = torch.hstack([tokens, torch.zeros_like(pad, dtype=torch.int)])
    ages = torch.hstack([ages, pad])
    tokens = tokens.masked_fill(ages > m, -1)  # no "no event" after the last real token
    ages = ages.masked_fill(ages > m, MASK_TIME)
    s = torch.argsort(ages, 1)
    tokens = torch.gather(tokens, 1, s) + 1
    ages = torch.gather(ages, 1, s)

    if cut_batch:
        cut_margin = torch.min(torch.sum(tokens == 0, 1))
        tokens, ages = tokens[:, cut_margin:], ages[:, cut_margin:]
    if tokens.shape[1] > block_size + 1:
        cut_margin = tokens.shape[1] - block_size - 1
        tokens, ages = tokens[:, cut_margin:], ages[:, cut_margin:]
    x, a, y, b = tokens[:, :-1], ages[:, :-1], tokens[:, 1:], ages[:, 1:]
    x = x.masked_fill((x == 0) * (y == 1), 0)  # a leading "no event" is not a prediction point
    y = y.masked_fill(x == 0, 0)
    b = b.masked_fill(x == 0, MASK_TIME)
    if torch.device(device).type == "cuda":
        return [t.pin_memory().to(device, non_blocking=True) for t in (x, a, y, b)]
    return [t.to(device) for t in (x, a, y, b)]


# Delphi's train.py defaults; a config file of its own (config/train_delphi.py, or delphi_config.py from
# `modlehr export-delphi`) and --key=value arguments override them, as its configurator does.
DELPHI_DEFAULTS = {
    "out_dir": "out", "eval_interval": 2000, "log_interval": 1, "eval_iters": 200, "eval_only": False,
    "always_save_checkpoint": False, "init_from": "scratch", "seed": 42, "wandb_log": False, "wandb_project": "delphi",
    "wandb_run_name": "run", "dataset": "ukb_simulated_data", "gradient_accumulation_steps": 1, "batch_size": 128,
    "block_size": 24, "n_layer": 6, "n_head": 6, "n_embd": 96, "dropout": 0.2, "bias": False, "vocab_size": 256,
    "learning_rate": 6e-4, "max_iters": 10000, "weight_decay": 1e-1, "beta1": 0.9, "beta2": 0.95, "grad_clip": 1.0,
    "decay_lr": True, "warmup_iters": 2000, "lr_decay_iters": 10000, "min_lr": 6e-5, "device": "cpu",
    "dtype": "float32", "compile": False, "token_dropout": 0.0, "t_min": 0.0, "mask_ties": True,
    "ignore_tokens": [0], "data_fraction": 1.0, "no_event_token_rate": 5,
}
MODEL_ARGS = ("n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size", "dropout", "token_dropout", "t_min",
              "mask_ties", "ignore_tokens")


def lr_at(it: int, s: dict) -> float:
    """Delphi's schedule: linear warmup from zero, cosine decay to min_lr at lr_decay_iters."""
    if not s["decay_lr"]:
        return s["learning_rate"]
    if it < s["warmup_iters"]:
        return s["learning_rate"] * it / s["warmup_iters"]
    if it > s["lr_decay_iters"]:
        return s["min_lr"]
    coeff = 0.5 * (1.0 + math.cos(math.pi * (it - s["warmup_iters"]) / (s["lr_decay_iters"] - s["warmup_iters"])))
    return s["min_lr"] + coeff * (s["learning_rate"] - s["min_lr"])


def fit(train_data: np.ndarray, val_data: np.ndarray, s: dict, out_dir: Path, checkpoint: str, device: torch.device,
        tracker: Tracker, record: dict, log=print) -> tuple[Path, float]:
    """Delphi's train.py on rows of (patient, age in days, id), with its settings `s`: returns the checkpoint
    with the lowest tuning loss, saved as Delphi saves it, and that loss. Runs max_iters + 1 steps."""
    if (s["gradient_accumulation_steps"] != 1 or s["dtype"] != "float32" or s["compile"] or s["init_from"] != "scratch"
            or s["eval_only"]):
        raise ValueError("Delphi-2M trains here with gradient_accumulation_steps=1, dtype float32, no compile, "
                         "init_from scratch and eval_only False")
    out_dir.mkdir(parents=True, exist_ok=True)
    train_p2i, val_p2i = get_p2i(train_data), get_p2i(val_data)
    if s["data_fraction"] < 1.0:
        train_p2i = train_p2i[:int(s["data_fraction"] * len(train_p2i))]
    matmul_precision = torch.get_float32_matmul_precision()
    torch.manual_seed(s["seed"])
    torch.set_float32_matmul_precision("high")  # as Delphi's train.py: TF32 matrix products on a GPU
    args = {k: s[k] for k in MODEL_ARGS}
    model = Delphi(DelphiConfig(**args)).to(device)
    opt = model.configure_optimizers(s["weight_decay"], s["learning_rate"], (s["beta1"], s["beta2"]), device.type)
    log(f"[train] Delphi-2M, {model.get_num_params() / 1e6:.2f}M parameters, {args['vocab_size']:,} Delphi tokens, "
        f"{len(train_p2i):,} training records, device {device}")
    block, rate, batch_size = args["block_size"], s["no_event_token_rate"], s["batch_size"]

    @torch.no_grad()
    def estimate_loss() -> dict:
        out = {}
        model.eval()
        for split, data, p2i in (("train", train_data, train_p2i), ("val", val_data, val_p2i)):
            losses = torch.zeros(s["eval_iters"], 2)
            for k in range(s["eval_iters"]):
                ix = torch.randint(len(p2i), (batch_size,))
                X, A, Y, B = get_batch(ix, data, p2i, block_size=block, device=device, select="left",
                                       no_event_token_rate=rate, cut_batch=True)
                _, loss = model(X, A, Y, B, validation_loss_mode=True)
                losses[k] = torch.stack([loss["loss_ce"], loss["loss_dt"]])
            out[split] = losses.mean(0)
        model.train()
        return out

    def train_batch(cut_batch: bool = True):
        ix = torch.randint(len(train_p2i), (batch_size,))
        return get_batch(ix, train_data, train_p2i, block_size=block, device=device, padding="random",
                         lifestyle_augmentations=True, select="left", no_event_token_rate=rate, cut_batch=cut_batch)

    X, A, Y, B = train_batch(cut_batch=False)  # Delphi's first batch keeps its padding, so dropout draws match
    best, t0 = float("inf"), time.time()
    try:
        with open(out_dir / "log.jsonl", "w") as logf:
            for it in range(s["max_iters"] + 1):
                lr = lr_at(it, s)
                for group in opt.param_groups:
                    group["lr"] = lr
                if it % s["eval_interval"] == 0 and it > 0:
                    losses = estimate_loss()
                    tune = losses["val"].sum().item()
                    entry = {"step": it, "train_loss": losses["train"].sum().item(), "tune_loss": tune,
                             "loss_ce": losses["val"][0].item(), "loss_dt": losses["val"][1].item(),
                             "seconds": round(time.time() - t0, 1)}
                    logf.write(json.dumps(entry) + "\n")
                    logf.flush()
                    log(f"[train] step {it:6d}  train {entry['train_loss']:.4f}  tune {tune:.4f}  ({entry['seconds']}s)")
                    tracker.log({**{k: v for k, v in entry.items() if k not in ("step", "seconds")}, "lr": lr}, it)
                    if s["always_save_checkpoint"] or tune < best:
                        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "model_args": args,
                                    "iter_num": it, "best_val_loss": tune, "config": record},
                                   out_dir / checkpoint)
                    best = min(best, tune)
                _, loss = model(X, A, Y, B)
                X, A, Y, B = train_batch()
                loss = loss["loss_ce"] + loss["loss_dt"]
                loss.backward()
                if s["grad_clip"]:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), s["grad_clip"])
                opt.step()
                opt.zero_grad(set_to_none=True)
    finally:
        torch.set_float32_matmul_precision(matmul_precision)
    log(f"[train] best tuning loss {best:.4f} → {out_dir / checkpoint}")
    return out_dir / checkpoint, best


def train_delphi(cfg: Config, log=print, resume: bool = False, stop_after: int | None = None) -> Path:
    """Train Delphi-2M on the prepared data, at the modLEHR config's size and schedule, and return the path
    of the checkpoint with the lowest tuning loss (best.pt)."""
    from modlehr.train.loop import pick_device

    if resume or stop_after is not None:
        raise ValueError("model.architecture: delphi2m trains in one go; --resume and --stop-after are not supported")
    if cfg.data.no_event_years is not None:
        raise ValueError("Delphi-2M inserts its own \"no event\" tokens; set data.no_event_years: null")
    if cfg.train.grad_accum != 1 or cfg.train.precision != "fp32":
        raise ValueError("model.architecture: delphi2m trains with train.grad_accum: 1 and train.precision: fp32")
    prepared, out_dir, t = Path(cfg.data.prepared_dir), Path(cfg.train.out_dir), cfg.train
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
    vocab = Vocabulary.load(prepared / "vocab.json")
    ids = delphi_ids(vocab)
    train_data = delphi_rows(open_split(prepared, meds.train_split), ids)
    val_data = delphi_rows(open_split(prepared, meds.tuning_split), ids)
    s = {**DELPHI_DEFAULTS, **model_args(cfg, vocab), "batch_size": t.batch_size, "max_iters": t.steps,
         "lr_decay_iters": t.steps, "warmup_iters": t.warmup, "learning_rate": t.lr, "min_lr": t.min_lr,
         "weight_decay": t.weight_decay, "beta2": BETA2, "grad_clip": t.grad_clip, "eval_interval": t.eval_every,
         "eval_iters": t.eval_batches, "seed": t.seed, "data_fraction": t.data_fraction,
         "no_event_token_rate": NO_EVENT_RATE}
    tracker = Tracker(cfg, out_dir, "train")
    best_path, best = fit(train_data, val_data, s, out_dir, "best.pt", pick_device(t.device), tracker, cfg.to_dict(), log)
    tracker.summary({"best_tune_loss": best})
    tracker.finish()
    return best_path


def read_delphi_config(path: str | Path | None, overrides: list[str]) -> dict:
    """Settings as Delphi's train.py reads them: its defaults, then a config file, then --key=value arguments."""
    from ast import literal_eval

    s = dict(DELPHI_DEFAULTS)
    if path:
        scope: dict = {}
        exec(Path(path).read_text(), scope)  # a config file of Delphi's own: plain assignments
        s.update({k: v for k, v in scope.items() if k in DELPHI_DEFAULTS})
    for arg in overrides:
        key, _, value = arg.removeprefix("--").partition("=")
        if key not in DELPHI_DEFAULTS:
            raise ValueError(f"unknown Delphi setting {key!r}")
        try:
            s[key] = literal_eval(value)
        except (SyntaxError, ValueError):
            s[key] = value
    return s


def train_from_delphi_config(path: str | Path | None, overrides: list[str], log=print) -> Path:
    """`modlehr delphi-train`: Delphi-2M trained here on data in Delphi's own format (train.bin and val.bin in
    data/<dataset>, or dataset itself when it is a path), with a config of Delphi's own, the way its train.py
    would. The checkpoint goes to out_dir/ckpt.pt, as Delphi writes it, so Delphi's own evaluation reads it."""
    from types import SimpleNamespace

    s = read_delphi_config(path, overrides)
    data_dir = Path(s["dataset"]) if Path(s["dataset"]).is_absolute() else Path("data") / s["dataset"]
    train_data = np.memmap(data_dir / "train.bin", dtype=np.uint32, mode="r").reshape(-1, 3)
    val_data = np.memmap(data_dir / "val.bin", dtype=np.uint32, mode="r").reshape(-1, 3)
    tracking = SimpleNamespace(train=SimpleNamespace(wandb_project=s["wandb_project"] if s["wandb_log"] else None,
                                                     wandb_entity=None), to_dict=lambda: s)
    tracker = Tracker(tracking, s["out_dir"], "train")
    checkpoint, best = fit(train_data, val_data, s, Path(s["out_dir"]), "ckpt.pt", torch.device(s["device"]), tracker,
                           s, log)
    tracker.summary({"best_tune_loss": best})
    tracker.finish()
    return checkpoint
