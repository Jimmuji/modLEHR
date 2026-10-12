"""`modlehr evaluate`: how well a trained model ranks patients, and how well it predicts when.

Two measures, on the evaluation split (held_out by default):

AUC at landmark ages (see modlehr.evaluate.task). For each landmark L and target k, the score
is the model's competing-risks probability of k within the horizon, read from its output at
the last event at or before L. One AUC per (target, landmark) pair with enough cases and
controls; the summary is the mean over pairs.

Likelihood. The mean negative log-likelihood per prediction of the training objective, split
into its event part (which target comes next) and its time part (how long until it). AUC is
computed within age groups and only ranks patients, so it cannot see whether the model gets
timing right; the time part can. Lower is better. Each timeline is cut into consecutive
windows of max_len events, so every prediction is scored once.

Writes <out>.csv (one row per pair) and <out>.json (summary).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from modlehr.config import Config, ModelConfig
from modlehr.evaluate.task import history_before, landmark_task, summarize
from modlehr.model.model import ModLEHR
from modlehr.prepare.dataset import PreparedSplit, collate
from modlehr.prepare.timeline import first_target_ages
from modlehr.prepare.vocabulary import Vocabulary
from modlehr.train.loop import pick_device, to_device


def load_model(checkpoint: str | Path, prepared_dir: Path, device: torch.device) -> tuple[ModLEHR, Vocabulary]:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    vocab = Vocabulary.load(prepared_dir / "vocab.json")
    model = ModLEHR(ModelConfig(**state["model_config"]), n_tokens=len(vocab.tokens), n_targets=len(vocab.targets))
    model.load_state_dict(state["model"])
    return model.to(device).eval(), vocab


@torch.no_grad()
def landmark_risks(model: ModLEHR, split: PreparedSplit, subjects: np.ndarray, landmark: float, horizon: float,
                   batch_size: int, device: torch.device) -> np.ndarray:
    """[len(subjects), n_targets] risk of each target within the horizon, from the history up to the landmark."""
    rows = [history_before(split, i, landmark, model.cfg.max_len) for i in subjects]
    risks = []
    for b in range(0, len(rows), batch_size):
        chunk = rows[b:b + batch_size]
        batch = to_device(collate(chunk), device)
        out = model(batch)
        last = torch.tensor([len(r["tokens"]) - 1 for r in chunk], device=device)
        risks.append(model.head.risk(out[torch.arange(len(chunk), device=device), last], horizon).float().cpu().numpy())
    return np.concatenate(risks) if risks else np.zeros((0, model.head.proj.out_features))


@torch.no_grad()
def likelihood(model: ModLEHR, split: PreparedSplit, batch_size: int, device: torch.device) -> dict:
    """Mean negative log-likelihood per prediction, with the head's parts (event, time)."""
    length = model.cfg.max_len
    windows = [split.subject(i, s, min(n, s + length))
               for i, n in enumerate(split.lengths()) for s in range(0, int(n), length)]
    windows.sort(key=lambda w: len(w["tokens"]))  # similar lengths per batch, less padding
    sums: dict[str, float] = {}
    count = 0
    for b in range(0, len(windows), batch_size):
        batch = to_device(collate(windows[b:b + batch_size]), device)
        n = int(batch["loss_mask"].sum())
        if not n:
            continue
        nll, parts = model.head.loss(model(batch), batch["next_target"], batch["dt"], batch["loss_mask"])
        for key, value in {"nll": float(nll), **parts}.items():
            sums[key] = sums.get(key, 0.0) + value * n
        count += n
    return {"predictions": count, **{k: v / max(count, 1) for k, v in sums.items()}}


def evaluate(cfg: Config, checkpoint: str | Path, out_csv: str | Path | None = None, log=print) -> pd.DataFrame:
    prepared = Path(cfg.data.prepared_dir)
    device = pick_device(cfg.train.device)
    model, vocab = load_model(checkpoint, prepared, device)
    split = PreparedSplit(prepared / f"{cfg.evaluate.split}.npz")
    first = first_target_ages(split.arrays["tokens"], split.arrays["ages"], split.offsets, vocab)
    H, min_cases = cfg.evaluate.horizon_years, cfg.evaluate.min_cases

    rows = []
    for L in cfg.evaluate.landmarks:
        task = landmark_task(split, vocab, L, H, first)
        risk = landmark_risks(model, split, task.subjects, L, H, cfg.evaluate.batch_size, device)
        for k, name in enumerate(vocab.targets):
            keep, case = task.labels(k)
            n_case, n_ctrl = int(case[keep].sum()), int((~case[keep]).sum())
            if n_case < min_cases or n_ctrl < min_cases:
                continue
            rows.append({"target": name, "landmark": L, "cases": n_case, "controls": n_ctrl,
                         "auc": roc_auc_score(case[keep], risk[keep, k])})
    result = pd.DataFrame(rows, columns=["target", "landmark", "cases", "controls", "auc"])

    summary = {**summarize(result), "horizon_years": H, "landmarks": cfg.evaluate.landmarks,
               "split": cfg.evaluate.split, "checkpoint": str(checkpoint)}
    if cfg.evaluate.likelihood:
        summary["likelihood"] = likelihood(model, split, cfg.evaluate.batch_size, device)

    out_csv = Path(out_csv) if out_csv else Path(cfg.train.out_dir) / "evaluate.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_csv, index=False)
    out_csv.with_suffix(".json").write_text(json.dumps(summary, indent=1))
    if len(result):
        lo, hi = summary["mean_auc_ci95"]
        log(f"[evaluate] mean AUC {summary['mean_auc']:.3f} (95% CI {lo:.3f} to {hi:.3f}) over "
            f"{summary['pairs']} target and landmark pairs")
    else:
        log("[evaluate] no target had enough cases and controls; lower evaluate.min_cases or add landmarks")
    if "likelihood" in summary:
        lk = summary["likelihood"]
        log(f"[evaluate] negative log-likelihood per prediction {lk['nll']:.3f} "
            f"(event {lk.get('nll_event', float('nan')):.3f}, time {lk.get('nll_time', float('nan')):.3f}) "
            f"over {lk['predictions']:,} predictions")
    log(f"[evaluate] → {out_csv}")
    return result
