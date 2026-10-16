"""`modlehr train`: fit a model on a prepared dataset.

Saves to train.out_dir: best.pt (lowest tuning loss), last.pt, config.yaml, log.jsonl.
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
from modlehr.model.model import ModLEHR
from modlehr.prepare.dataset import fixed_windows, open_split, random_windows
from modlehr.prepare.vocabulary import Vocabulary


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def lr_at(step: int, cfg) -> float:
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    progress = (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup)
    return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1 + math.cos(math.pi * min(progress, 1.0)))


def make_optimizer(model: torch.nn.Module, cfg) -> torch.optim.Optimizer:
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2 and "embed" not in n]
    rest = [p for n, p in model.named_parameters() if not (p.dim() >= 2 and "embed" not in n)]
    return torch.optim.AdamW([{"params": decay, "weight_decay": cfg.weight_decay},
                              {"params": rest, "weight_decay": 0.0}], lr=cfg.lr, betas=(0.9, 0.99))


def to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) for k, v in batch.items()}


@torch.no_grad()
def mean_loss(model: ModLEHR, batches: list[dict], device: torch.device) -> float:
    model.eval()
    losses = [float(model.loss(to_device(b, device))[0]) for b in batches]
    model.train()
    return float(np.mean(losses))


def build_model(cfg: Config, prepared_dir: Path) -> tuple[ModLEHR, Vocabulary, dict]:
    vocab = Vocabulary.load(prepared_dir / "vocab.json")
    meta = json.loads((prepared_dir / "prepare.json").read_text())
    model = ModLEHR(cfg.model, n_tokens=len(vocab.tokens), n_targets=len(vocab.targets))
    return model, vocab, meta


def train(cfg: Config, log=print) -> Path:
    prepared = Path(cfg.data.prepared_dir)
    out_dir = Path(cfg.train.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
    torch.manual_seed(cfg.train.seed)
    rng = np.random.default_rng(cfg.train.seed)
    device = pick_device(cfg.train.device)

    model, vocab, meta = build_model(cfg, prepared)
    if hasattr(model.head, "init_rates"):
        model.head.init_rates(meta["target_counts"], meta["exposure_years"])
    model.to(device)
    log(f"[train] {model.n_parameters() / 1e6:.2f}M parameters, {len(vocab.tokens):,} tokens, "
        f"{len(vocab.targets):,} targets, device {device}")

    train_split = open_split(prepared, meds.train_split)
    tune_split = open_split(prepared, meds.tuning_split)
    tune_batches = fixed_windows(tune_split, cfg.train.batch_size, cfg.model.max_len, cfg.train.eval_batches)
    opt = make_optimizer(model, cfg.train)

    best, t0 = float("inf"), time.time()
    with open(out_dir / "log.jsonl", "w") as logf:
        for step in range(cfg.train.steps):
            for group in opt.param_groups:
                group["lr"] = lr_at(step, cfg.train)
            batch = to_device(random_windows(train_split, cfg.train.batch_size, cfg.model.max_len, rng), device)
            loss, parts = model.loss(batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            opt.step()

            last = step == cfg.train.steps - 1
            if step % cfg.train.eval_every == 0 or last:
                tune = mean_loss(model, tune_batches, device)
                record = {"step": step, "train_loss": float(loss.detach()), "tune_loss": tune, **parts,
                          "seconds": round(time.time() - t0, 1)}
                logf.write(json.dumps(record) + "\n")
                logf.flush()
                log(f"[train] step {step:6d}  train {record['train_loss']:.4f}  tune {tune:.4f}  ({record['seconds']}s)")
                state = {"model": model.state_dict(), "model_config": cfg.model.__dict__, "step": step, "tune_loss": tune}
                if tune < best:
                    best = tune
                    torch.save(state, out_dir / "best.pt")
                if last:
                    torch.save(state, out_dir / "last.pt")
    log(f"[train] best tuning loss {best:.4f} → {out_dir / 'best.pt'}")
    return out_dir / "best.pt"
