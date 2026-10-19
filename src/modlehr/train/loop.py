"""`modlehr train`: fit a model on a prepared dataset.

Saves to train.out_dir: config.yaml, log.jsonl, best.pt (lowest tuning loss) and last.pt. last.pt
is rewritten at every tuning check with the optimizer and random-number state, so a run that is
stopped (a job's time limit, a crash) continues exactly where it was with `--resume`.

For large data on a GPU: train.precision: bf16 runs the forward pass in bfloat16 (CUDA), and
train.grad_accum: k adds up gradients over k batches per optimizer step, for a larger effective
batch than fits in memory.
"""

from __future__ import annotations

import contextlib
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
from modlehr.prepare.vocabulary import NO_EVENT, Vocabulary
from modlehr.track import Tracker


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
    model = ModLEHR(cfg.model, n_tokens=len(vocab.tokens), n_targets=len(vocab.targets),
                    n_text_buckets=meta.get("text_buckets", 0))
    return model, vocab, meta


def autocast(device: torch.device, precision: str):
    if precision == "fp32":
        return contextlib.nullcontext()
    if precision == "bf16":
        if device.type != "cuda":
            raise ValueError("train.precision: bf16 needs a CUDA device; use fp32 on MPS and CPU")
        return torch.autocast("cuda", dtype=torch.bfloat16)
    raise ValueError(f"train.precision must be fp32 or bf16, not '{precision}'")


def train(cfg: Config, log=print, resume: bool = False, stop_after: int | None = None) -> Path:
    """Train, and return the path of the best checkpoint.

    resume continues from <out_dir>/last.pt. stop_after ends the run after that step (last.pt is
    saved there), for runs split over several jobs.
    """
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
    if cfg.train.grad_accum < 1:
        raise ValueError("train.grad_accum must be at least 1")
    log(f"[train] {model.n_parameters() / 1e6:.2f}M parameters, {len(vocab.tokens):,} tokens, "
        f"{len(vocab.targets):,} targets, device {device}, {cfg.train.precision}"
        + (f", {cfg.train.grad_accum} batches per step" if cfg.train.grad_accum > 1 else ""))

    tracker = Tracker(cfg, out_dir, "train")
    train_split = open_split(prepared, meds.train_split)
    tune_split = open_split(prepared, meds.tuning_split)
    if (NO_EVENT in vocab.token_index) != (cfg.data.no_event_years is not None):
        raise ValueError(f"data.no_event_years is {cfg.data.no_event_years} but {prepared} was prepared "
                         f"{'with' if NO_EVENT in vocab.token_index else 'without'} it; run `modlehr prepare` again")
    no_event = (vocab.token_index[NO_EVENT], cfg.data.no_event_years) if cfg.data.no_event_years is not None else None
    tune_batches = fixed_windows(tune_split, cfg.train.batch_size, cfg.model.max_len, cfg.train.eval_batches,
                                 no_event=no_event)
    opt = make_optimizer(model, cfg.train)
    if cfg.train.compile:
        model.compile()  # in place; parameter names in checkpoints stay the same

    best, first_step = float("inf"), 0
    if resume:
        if not (out_dir / "last.pt").exists():
            raise FileNotFoundError(f"{out_dir / 'last.pt'} not found; nothing to resume")
        state = torch.load(out_dir / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        rng.bit_generator.state = state["numpy_rng"]
        torch.set_rng_state(state["torch_rng"].cpu())
        if device.type == "cuda" and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(state["cuda_rng"].cpu())
        if device.type == "mps" and state.get("mps_rng") is not None:  # dropout on an Apple GPU
            torch.mps.set_rng_state(state["mps_rng"].cpu())
        best, first_step = state["best_tune_loss"], state["step"] + 1
        log(f"[train] resuming after step {state['step']} (best tuning loss so far {best:.4f})")

    t0 = time.time()
    last_check = (first_step, t0)
    with open(out_dir / "log.jsonl", "a" if resume else "w") as logf:
        for step in range(first_step, cfg.train.steps):
            for group in opt.param_groups:
                group["lr"] = lr_at(step, cfg.train)
            opt.zero_grad(set_to_none=True)
            for _ in range(cfg.train.grad_accum):
                batch = to_device(random_windows(train_split, cfg.train.batch_size, cfg.model.max_len, rng, no_event),
                                  device)
                with autocast(device, cfg.train.precision):
                    loss, parts = model.loss(batch)
                (loss / cfg.train.grad_accum).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            opt.step()

            last = step == cfg.train.steps - 1
            stopping = stop_after is not None and step >= stop_after
            if step % cfg.train.eval_every == 0 or last or stopping:
                with autocast(device, cfg.train.precision):
                    tune = mean_loss(model, tune_batches, device)
                record = {"step": step, "train_loss": float(loss.detach()), "tune_loss": tune, **parts,
                          "seconds": round(time.time() - t0, 1)}
                logf.write(json.dumps(record) + "\n")
                logf.flush()
                log(f"[train] step {step:6d}  train {record['train_loss']:.4f}  tune {tune:.4f}  ({record['seconds']}s)")
                now = time.time()
                tracked = {k: v for k, v in record.items() if k not in ("step", "seconds")}
                tracked["lr"] = opt.param_groups[0]["lr"]
                tracked["steps_per_second"] = (step + 1 - last_check[0]) / max(now - last_check[1], 1e-9)
                if device.type == "cuda":
                    tracked["gpu_peak_memory_gb"] = torch.cuda.max_memory_allocated(device) / 2**30
                tracker.log(tracked, step)
                last_check = (step + 1, now)
                weights = {"model": model.state_dict(), "model_config": cfg.model.__dict__, "step": step, "tune_loss": tune}
                if tune < best:
                    best = tune
                    torch.save(weights, out_dir / "best.pt")
                torch.save({**weights, "optimizer": opt.state_dict(), "best_tune_loss": best,
                            "numpy_rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
                            "cuda_rng": torch.cuda.get_rng_state() if device.type == "cuda" else None,
                            "mps_rng": torch.mps.get_rng_state() if device.type == "mps" else None},
                           out_dir / "last.pt")
            if stopping and not last:
                log(f"[train] stopped after step {step}; continue with --resume")
                tracker.finish()
                return out_dir / "best.pt"
    log(f"[train] best tuning loss {best:.4f} → {out_dir / 'best.pt'}")
    tracker.summary({"best_tune_loss": best})
    tracker.finish()
    return out_dir / "best.pt"
