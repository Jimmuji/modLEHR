"""`modlehr evaluate`: how well a trained model ranks patients, and how well it predicts when.

Two measures, on the evaluation split (held_out by default):

AUC at landmark ages (see modlehr.evaluate.task). For each landmark L and target k, the score
is the model's competing-risks probability of k within the horizon, read from its output at
the last event at or before L. One AUC per (target, landmark) pair with enough cases and
controls; the summary is the mean over pairs.

Calibration. At each landmark the head's probability that the next target is k and comes in
(L, L + H] is compared with what the record shows. The window starts at L even when the last event
is years earlier: nothing happened in between, and the head's waiting time is exponential, so its
probability for a window of length H does not depend on where the window starts. Patients whose
follow-up ends before L + H without a target are not dropped, which would leave the observed rate
too high (only people still followed, plus everyone with an early target): each patient whose
status at L + H is known is weighted by 1 / P(still followed), estimated by Kaplan-Meier from the
censoring times (inverse probability of censoring weights), and the weighted outcomes are averaged
over everyone. Reported as expected over observed (1 is calibrated) for any target and for each
target pooled, and as a table of predicted against observed by risk bin.

Likelihood. The mean negative log-likelihood per prediction of the training objective, split
into its event part (which target comes next) and its time part (how long until it). AUC is
computed within age groups and only ranks patients, so it cannot see whether the model gets
timing right; the time part can. Lower is better. Each timeline is cut into consecutive
windows of max_len events, so every prediction is scored once.

Writes <out>.csv (one row per pair), <out>_calibration.csv and <out>.json (summary).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from modlehr.config import Config, ModelConfig
from modlehr.evaluate.task import history_before, landmark_task, split_first_ages, summarize
from modlehr.model.model import ModLEHR
from modlehr.prepare.dataset import PreparedSplit, collate, open_split
from modlehr.prepare.vocabulary import Vocabulary
from modlehr.track import Tracker
from modlehr.train.loop import pick_device, to_device


def load_model(checkpoint: str | Path, prepared_dir: Path, device: torch.device) -> tuple[ModLEHR, Vocabulary]:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    vocab = Vocabulary.load(prepared_dir / "vocab.json")
    meta = json.loads((Path(prepared_dir) / "prepare.json").read_text())
    model = ModLEHR(ModelConfig(**state["model_config"]), n_tokens=len(vocab.tokens), n_targets=len(vocab.targets),
                    n_text_buckets=meta.get("text_buckets", 0))
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
        nll, parts = model.head.loss(model(batch), batch.get("next_set", batch["next_target"]), batch["dt"], batch["loss_mask"])
        for key, value in {"nll": float(nll), **parts}.items():
            sums[key] = sums.get(key, 0.0) + value * n
        count += n
    return {"predictions": count, **{k: v / max(count, 1) for k, v in sums.items()}}


CALIBRATION_BINS = np.array([0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0])


def censoring_weights(time: np.ndarray, event: np.ndarray, horizon: float) -> np.ndarray:
    """Inverse probability of censoring weights for each subject's status at the horizon.

    time: years from the landmark to the next target, or to the end of follow-up; event: a target came
    then. G(t) is the Kaplan-Meier probability of still being followed just before t (a target and a
    censoring at the same time count the target first). A target by the horizon gets 1 / G(time); a
    subject followed to the horizon without one gets 1 / G(horizon); a subject censored earlier gets 0.
    """
    order = np.sort(time)
    cens_times, cens_counts = np.unique(time[~event], return_counts=True)
    at_risk = len(time) - np.searchsorted(order, cens_times, side="left")
    surv_after = np.cumprod(1.0 - cens_counts / np.maximum(at_risk, 1))

    def survival_before(t):
        j = np.searchsorted(cens_times, t, side="left") - 1  # censoring times strictly before t
        return np.where(j >= 0, surv_after[np.maximum(j, 0)], 1.0)

    hit = event & (time <= horizon)
    followed = ~hit & (time >= horizon)
    w = np.zeros(len(time))
    w[hit] = 1.0 / np.maximum(survival_before(time[hit]), 1e-12)
    w[followed] = 1.0 / np.maximum(survival_before(np.full(followed.sum(), horizon)), 1e-12)
    return w


def calibration(split: PreparedSplit, subjects: np.ndarray, landmark: float, horizon: float,
                risk: np.ndarray) -> tuple[dict, pd.DataFrame]:
    """Predicted against observed next targets in (L, L + H], weighted for censoring."""
    rows = [history_before(split, i, landmark) for i in subjects]
    n_targets = risk.shape[1]
    gap = landmark - np.array([r["ages"][-1] for r in rows])  # years from the last event to L
    time = np.array([r["dt"][-1] for r in rows]) - gap  # from L to the next target or end of follow-up
    sets = [r["next_set"][-1] if "next_set" in r else r["next_target"][-1:] for r in rows]
    event = np.array([s[0] >= 0 for s in sets])
    w = censoring_weights(time, event, horizon)
    hit = event & (time <= horizon)
    observed = np.zeros((len(rows), n_targets), dtype=bool)
    for j in np.flatnonzero(hit):
        observed[j, sets[j][sets[j] >= 0]] = True
    summary = {"landmark": landmark, "subjects": int(len(rows)), "followed_to_horizon": int((w > 0).sum()),
               "expected_any": float(risk.sum(axis=1).sum()), "observed_any": float((w * hit).sum())}
    summary["expected_over_observed"] = summary["expected_any"] / max(summary["observed_any"], 1e-9)
    bins = np.digitize(risk.ravel(), CALIBRATION_BINS[1:-1])
    weighted = (observed * w[:, None]).ravel()
    table = pd.DataFrame({"bin": bins, "predicted": risk.ravel(), "observed": weighted}).groupby("bin").agg(
        pairs=("predicted", "size"), mean_predicted=("predicted", "mean"), observed_rate=("observed", "mean")).reset_index()
    table.insert(0, "landmark", landmark)
    table["bin_low"] = CALIBRATION_BINS[table["bin"]]
    table["bin_high"] = CALIBRATION_BINS[table["bin"] + 1]
    return summary, table.drop(columns="bin")


def evaluate(cfg: Config, checkpoint: str | Path, out_csv: str | Path | None = None, log=print) -> pd.DataFrame:
    prepared = Path(cfg.data.prepared_dir)
    device = pick_device(cfg.train.device)
    model, vocab = load_model(checkpoint, prepared, device)
    split = open_split(prepared, cfg.evaluate.split)
    first = split_first_ages(split, vocab)
    H, min_cases = cfg.evaluate.horizon_years, cfg.evaluate.min_cases

    rows, calib, calib_tables = [], [], []
    for L in cfg.evaluate.landmarks:
        task = landmark_task(split, vocab, L, H, first, cfg.evaluate.controls)
        risk = landmark_risks(model, split, task.subjects, L, H, cfg.evaluate.batch_size, device)
        if len(task.subjects):
            c_summary, c_table = calibration(split, task.subjects, L, H, risk)
            calib.append(c_summary)
            calib_tables.append(c_table)
        for k, name in enumerate(vocab.targets):
            keep, case = task.labels(k)
            n_case, n_ctrl = int(case[keep].sum()), int((~case[keep]).sum())
            if n_case < min_cases or n_ctrl < min_cases:
                continue
            rows.append({"target": name, "landmark": L, "cases": n_case, "controls": n_ctrl,
                         "auc": roc_auc_score(case[keep], risk[keep, k])})
    result = pd.DataFrame(rows, columns=["target", "landmark", "cases", "controls", "auc"])

    summary = {**summarize(result), "horizon_years": H, "controls": cfg.evaluate.controls,
               "landmarks": cfg.evaluate.landmarks,
               "split": cfg.evaluate.split, "checkpoint": str(checkpoint)}
    if calib:
        expected, observed = sum(c["expected_any"] for c in calib), sum(c["observed_any"] for c in calib)
        summary["calibration"] = {"expected_over_observed": expected / max(observed, 1e-9), "window": "landmark",
                                  "observed": "censoring-weighted",
                                  "by_landmark": calib}
    if cfg.evaluate.likelihood:
        summary["likelihood"] = likelihood(model, split, cfg.evaluate.batch_size, device)

    out_csv = Path(out_csv) if out_csv else Path(cfg.train.out_dir) / "evaluate.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_csv, index=False)
    if calib_tables:
        pd.concat(calib_tables).to_csv(out_csv.with_name(out_csv.stem + "_calibration.csv"), index=False)
    out_csv.with_suffix(".json").write_text(json.dumps(summary, indent=1))
    tracker = Tracker(cfg, cfg.train.out_dir, "evaluate")
    tracker.summary({k: v for k, v in summary.items() if k not in ("checkpoint", "split")},
                    prefix=f"{cfg.evaluate.split}/{cfg.evaluate.controls}/")
    tracker.finish()
    if len(result):
        lo, hi = summary["mean_auc_ci95"]
        log(f"[evaluate] mean AUC {summary['mean_auc']:.3f} (95% CI {lo:.3f} to {hi:.3f}) over "
            f"{summary['pairs']} target and landmark pairs, controls={cfg.evaluate.controls}")
    else:
        log("[evaluate] no target had enough cases and controls; lower evaluate.min_cases or add landmarks")
    if "calibration" in summary:
        log(f"[evaluate] calibration: expected / observed next targets within {H:g} years of the landmark = "
            f"{summary['calibration']['expected_over_observed']:.2f} (1 is calibrated; "
            + ", ".join(f"age {c['landmark']:g}: {c['expected_over_observed']:.2f}" for c in calib) + ")")
    if "likelihood" in summary:
        lk = summary["likelihood"]
        log(f"[evaluate] negative log-likelihood per prediction {lk['nll']:.3f} "
            f"(event {lk.get('nll_event', float('nan')):.3f}, time {lk.get('nll_time', float('nan')):.3f}) "
            f"over {lk['predictions']:,} predictions")
    log(f"[evaluate] → {out_csv}")
    return result
