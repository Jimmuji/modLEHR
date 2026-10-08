"""`modlehr evaluate`: AUC for each target at landmark ages.

Task, for each landmark age L and target k: from all events up to L, predict a first
occurrence of k in (L, L + horizon]. Patients who already had k by L are left out; so are
patients whose record ends before L + horizon without k (censored). The score is the
competing-risks probability from the model's output at the last event before L.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from modlehr.config import Config, ModelConfig
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
def landmark_scores(model: ModLEHR, split: PreparedSplit, landmark: float, horizon: float, batch_size: int,
                    device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """(subject indices in the record at the landmark, [n, n_targets] risks)."""
    ages, offsets = split.arrays["ages"], split.offsets
    start, end = ages[offsets[:-1]], ages[offsets[1:] - 1]
    eligible = np.flatnonzero((start <= landmark) & (end > landmark))
    rows = []
    for i in eligible:
        a = split.offsets[i]
        n_before = int(np.searchsorted(ages[a:split.offsets[i + 1]], landmark, side="right"))
        s = max(0, n_before - model.cfg.max_len)
        rows.append(split.subject(int(i), s, n_before))
    risks = []
    for b in range(0, len(rows), batch_size):
        chunk = rows[b:b + batch_size]
        batch = to_device(collate(chunk), device)
        out = model(batch)
        last = torch.tensor([len(r["tokens"]) - 1 for r in chunk], device=device)
        risks.append(model.head.risk(out[torch.arange(len(chunk), device=device), last], horizon).float().cpu().numpy())
    return eligible, (np.concatenate(risks) if risks else np.zeros((0, model.head.proj.out_features)))


def evaluate(cfg: Config, checkpoint: str | Path, out_csv: str | Path | None = None, log=print) -> pd.DataFrame:
    prepared = Path(cfg.data.prepared_dir)
    device = pick_device(cfg.train.device)
    model, vocab = load_model(checkpoint, prepared, device)
    split = PreparedSplit(prepared / f"{cfg.evaluate.split}.npz")
    first = first_target_ages(split.arrays["tokens"], split.arrays["ages"], split.offsets, vocab)
    end = split.arrays["ages"][split.offsets[1:] - 1]
    H = cfg.evaluate.horizon_years

    rows = []
    for L in cfg.evaluate.landmarks:
        idx, risk = landmark_scores(model, split, L, H, cfg.evaluate.batch_size, device)
        f, e = first[idx], end[idx]
        for k, name in enumerate(vocab.targets):
            before = f[:, k] <= L
            case = (f[:, k] > L) & (f[:, k] <= L + H)
            keep = ~before & (case | (e >= L + H))
            n_case, n_ctrl = int(case[keep].sum()), int((~case[keep]).sum())
            if n_case < cfg.evaluate.min_cases or n_ctrl < cfg.evaluate.min_cases:
                continue
            rows.append({"target": name, "landmark": L, "cases": n_case, "controls": n_ctrl,
                         "auc": roc_auc_score(case[keep], risk[keep, k])})
    result = pd.DataFrame(rows)
    out_csv = Path(out_csv) if out_csv else Path(cfg.train.out_dir) / "evaluate.csv"
    result.to_csv(out_csv, index=False)
    summary = {"pairs": len(result), "mean_auc": float(result["auc"].mean()) if len(result) else None,
               "horizon_years": H, "landmarks": cfg.evaluate.landmarks}
    out_csv.with_suffix(".json").write_text(json.dumps(summary, indent=1))
    log(f"[evaluate] mean AUC {summary['mean_auc']:.3f} over {summary['pairs']} target and landmark pairs → {out_csv}"
        if len(result) else "[evaluate] no target had enough cases")
    return result
