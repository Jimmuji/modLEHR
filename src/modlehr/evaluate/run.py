"""`modlehr evaluate`: how well a trained model ranks patients, and how well it predicts when.

Two measures, on the evaluation split (held_out by default):

AUC at landmark ages (see modlehr.evaluate.task). For each landmark L and target k, the score
is the model's competing-risks probability of k within the horizon, read from its output at
the last event at or before L. One AUC per (target, landmark) pair with enough cases and
controls; the summary is the mean over pairs.

Calibration. At each landmark, the predicted number of targets that first occur in (L, L + H] is
compared with the number observed. The prediction sums, over the targets a patient has not had,
the probability that each is recorded in the window: it has to come before death, since nothing is
recorded after it (the head's observed_risk). This comes from the head's rates whatever
evaluate.risk is; that setting picks the AUC score only. The window starts at L even when the last
event is years earlier. Patients whose follow-up ends before L + H are not dropped, which would
leave the observed numbers too high: each patient whose window is fully observed (followed to
L + H, or dead within it) is weighted by 1 / P(still followed), estimated by Kaplan-Meier from the
follow-up end times (inverse probability of censoring weights), and the weighted counts are
averaged over everyone. Reported as expected
over observed (1 is calibrated), overall and per landmark, and as a table of predicted against
observed per (patient, target) by risk bin.

With evaluate.recalibrate, every rate at a landmark is first scaled by one factor, fit on the tuning split so
that its expected number of new targets equals the observed one; AUCs do not change, and calibration on the
evaluation split shows what is left once the overall level is right.

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

import meds
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from modlehr.config import Config, ModelConfig
from modlehr.evaluate.task import history_before, landmark_task, split_first_ages, summarize
from modlehr.model.delphi import DelphiModel, is_delphi_checkpoint
from modlehr.model.model import ModLEHR
from modlehr.prepare.dataset import PreparedSplit, collate, open_split
from modlehr.prepare.vocabulary import NO_EVENT, Vocabulary
from modlehr.track import Tracker
from modlehr.train.loop import pick_device, to_device


def load_model(checkpoint: str | Path, prepared_dir: Path, device: torch.device) -> tuple[ModLEHR, Vocabulary]:
    """A modLEHR checkpoint, or a Delphi-2M one (model.architecture: delphi2m, or Delphi's own train.py)."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    vocab = Vocabulary.load(prepared_dir / "vocab.json")
    if is_delphi_checkpoint(state):
        return DelphiModel(state, vocab).to(device).eval(), vocab
    meta = json.loads((Path(prepared_dir) / "prepare.json").read_text())
    model = ModLEHR(ModelConfig(**state["model_config"]), n_tokens=len(vocab.tokens), n_targets=len(vocab.targets),
                    n_text_buckets=meta.get("text_buckets", 0))
    model.load_state_dict(state["model"])
    if NO_EVENT in vocab.token_index:  # trained with "no event" tokens: ask at one, placed at the landmark
        model.query_token = vocab.token_index[NO_EVENT]
    return model.to(device).eval(), vocab


@torch.no_grad()
def landmark_outputs(model: ModLEHR, split: PreparedSplit, subjects: np.ndarray, landmark: float, batch_size: int,
                     device: torch.device) -> torch.Tensor:
    """[len(subjects), n_targets] the head's output (log-rates) at each subject's last event at or before the
    landmark. A model with a query_token (Delphi-2M's "no event", for a Delphi checkpoint or a model trained with
    data.no_event_years) is read at that token, placed at the landmark."""
    query = getattr(model, "query_token", None)
    rows = [history_before(split, i, landmark, model.cfg.max_len - (query is not None)) for i in subjects]
    if query is not None:
        rows = [with_event(r, query, landmark) for r in rows]
    outputs = []
    for b in range(0, len(rows), batch_size):
        chunk = rows[b:b + batch_size]
        out = model(to_device(collate(chunk), device))
        last = torch.tensor([len(r["tokens"]) - 1 for r in chunk], device=device)
        outputs.append(out[torch.arange(len(chunk), device=device), last].float().cpu())
    return torch.cat(outputs) if outputs else torch.zeros((0, model.head.proj.out_features))


def with_event(row: dict, token: int, age: float) -> dict:
    """The row with one more event, without a value or a label, at its end."""
    extra = {"tokens": token, "ages": age, "values": np.nan, "target": -1, "next_target": -1, "dt": 0.0, "loss_mask": False}
    return {k: np.append(v, np.asarray(extra[k], dtype=v.dtype)) if k in extra else v for k, v in row.items()}


def observed_probabilities(model: ModLEHR, outputs: torch.Tensor, horizon: float, death: int | None) -> np.ndarray:
    """Probability that each target is recorded within the horizon, for calibration. Heads without
    observed_risk fall back to their marginal risk."""
    if hasattr(model.head, "observed_risk"):
        return model.head.observed_risk(outputs, horizon, death).numpy()
    return model.head.risk(outputs, horizon, "marginal").numpy()


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


def censoring_weights(time: np.ndarray, censored: np.ndarray, known: np.ndarray) -> np.ndarray:
    """Inverse probability of censoring weights.

    time: years from the landmark until the patient's window is settled (the horizon, death within
    it, or the end of follow-up); censored: follow-up ended before the horizon without death; known:
    the window is fully observed. G(t) is the Kaplan-Meier probability of still being followed just
    before t; a known patient gets 1 / G(time), a censored one 0.
    """
    order = np.sort(time)
    cens_times, cens_counts = np.unique(time[censored], return_counts=True)
    at_risk = len(time) - np.searchsorted(order, cens_times, side="left")
    surv_after = np.cumprod(1.0 - cens_counts / np.maximum(at_risk, 1))
    j = np.searchsorted(cens_times, time, side="left") - 1  # censoring times strictly before each time
    g = np.where(j >= 0, surv_after[np.maximum(j, 0)], 1.0) if len(cens_times) else np.ones(len(time))
    return np.where(known, 1.0 / np.maximum(g, 1e-12), 0.0)


def calibration(task, risk: np.ndarray, death: int | None) -> tuple[dict, pd.DataFrame]:
    """Predicted against observed numbers of targets first occurring in (L, L + H], weighted for censoring.
    risk: [n, n_targets] the probability that each target is recorded in the window (observed_probabilities)."""
    L, H = task.landmark, task.horizon
    open_ = task.first > L  # targets the patient has not had by L
    observed = open_ & (task.first <= L + H)
    died = (task.first[:, death] > L) & (task.first[:, death] <= L + H) if death is not None \
        else np.zeros(len(task.first), bool)
    settled = np.minimum(np.where(died, task.first[:, death] if death is not None else np.inf, task.end), L + H) - L
    known = died | (task.end >= L + H)
    w = censoring_weights(settled, ~known, known)
    p = np.where(open_, risk, 0.0)
    summary = {"landmark": L, "subjects": int(len(w)), "fully_observed": int(known.sum()),
               "expected": float(p.sum()), "observed": float((observed.sum(axis=1) * w).sum())}
    summary["expected_over_observed"] = summary["expected"] / max(summary["observed"], 1e-9)
    pairs = open_.ravel()
    bins = np.digitize(p.ravel()[pairs], CALIBRATION_BINS[1:-1])
    weighted = (observed * w[:, None]).ravel()[pairs]
    table = pd.DataFrame({"bin": bins, "predicted": p.ravel()[pairs], "observed": weighted}).groupby("bin").agg(
        pairs=("predicted", "size"), mean_predicted=("predicted", "mean"), observed_rate=("observed", "mean")).reset_index()
    table.insert(0, "landmark", L)
    table["bin_low"] = CALIBRATION_BINS[table["bin"]]
    table["bin_high"] = CALIBRATION_BINS[table["bin"] + 1]
    return summary, table.drop(columns="bin")


def recalibration_shift(model: ModLEHR, split: PreparedSplit, vocab: Vocabulary, first: np.ndarray, landmark: float,
                        horizon: float, controls: str, batch_size: int, device: torch.device, death: int | None) -> float:
    """The log of the factor that, added to every log-rate at this landmark, makes the expected number of new
    targets in the window equal the observed (censoring-weighted) number on `split` (the tuning split)."""
    task = landmark_task(split, vocab, landmark, horizon, first, controls)
    if not len(task.subjects):
        return 0.0
    outputs = landmark_outputs(model, split, task.subjects, landmark, batch_size, device)
    observed = calibration(task, np.zeros((len(task.subjects), len(vocab.targets))), death)[0]["observed"]
    open_ = task.first > landmark
    expected = lambda s: float(np.where(open_, observed_probabilities(model, outputs + s, horizon, death), 0.0).sum())  # noqa: E731
    if observed <= 0:
        return 0.0
    lo, hi = -10.0, 10.0  # expected rises with the shift: bisect for expected = observed
    for _ in range(50):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if expected(mid) < observed else (lo, mid)
    return (lo + hi) / 2


def evaluate(cfg: Config, checkpoint: str | Path, out_csv: str | Path | None = None, log=print) -> pd.DataFrame:
    prepared = Path(cfg.data.prepared_dir)
    device = pick_device(cfg.train.device)
    model, vocab = load_model(checkpoint, prepared, device)
    split = open_split(prepared, cfg.evaluate.split)
    first = split_first_ages(split, vocab)
    H, min_cases = cfg.evaluate.horizon_years, cfg.evaluate.min_cases
    death = vocab.targets.index(meds.death_code) if meds.death_code in vocab.targets else None

    rows, calib, calib_tables, shifts = [], [], [], {}
    if cfg.evaluate.recalibrate:
        tuning = open_split(prepared, meds.tuning_split)
        tuning_first = split_first_ages(tuning, vocab)
    for L in cfg.evaluate.landmarks:
        task = landmark_task(split, vocab, L, H, first, cfg.evaluate.controls)
        if not len(task.subjects):
            continue
        outputs = landmark_outputs(model, split, task.subjects, L, cfg.evaluate.batch_size, device)
        if cfg.evaluate.recalibrate:
            shifts[L] = recalibration_shift(model, tuning, vocab, tuning_first, L, H, cfg.evaluate.controls,
                                            cfg.evaluate.batch_size, device, death)
            outputs = outputs + shifts[L]
        risk = model.head.risk(outputs, H, cfg.evaluate.risk).numpy()
        c_summary, c_table = calibration(task, observed_probabilities(model, outputs, H, death), death)
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

    summary = {**summarize(result), "horizon_years": H, "controls": cfg.evaluate.controls, "risk": cfg.evaluate.risk,
               "landmarks": cfg.evaluate.landmarks,
               "split": cfg.evaluate.split, "checkpoint": str(checkpoint)}
    if calib:
        expected, observed = sum(c["expected"] for c in calib), sum(c["observed"] for c in calib)
        summary["calibration"] = {"expected_over_observed": expected / max(observed, 1e-9), "window": "landmark",
                                  "counts": "targets first occurring in the window", "observed": "censoring-weighted",
                                  "expected": "probability each target is recorded in the window, before death",
                                  "by_landmark": calib}
    if cfg.evaluate.recalibrate:
        summary["recalibration"] = {"split": meds.tuning_split, "rate_factor_by_landmark": {
            f"{L:g}": float(np.exp(v)) for L, v in shifts.items()}}
    if cfg.evaluate.likelihood:  # of the model as trained, not recalibrated
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
            f"{summary['pairs']} target and landmark pairs, controls={cfg.evaluate.controls}, risk={cfg.evaluate.risk}")
    else:
        log("[evaluate] no target had enough cases and controls; lower evaluate.min_cases or add landmarks")
    if "calibration" in summary:
        log(f"[evaluate] calibration: expected / observed new targets within {H:g} years of the landmark = "
            f"{summary['calibration']['expected_over_observed']:.2f} (1 is calibrated; "
            + ", ".join(f"age {c['landmark']:g}: {c['expected_over_observed']:.2f}" for c in calib) + ")")
    if shifts:
        log("[evaluate] recalibrated on the tuning split: rates times "
            + ", ".join(f"{np.exp(v):.2f} at {L:g}" for L, v in shifts.items()))
    if "likelihood" in summary:
        lk = summary["likelihood"]
        log(f"[evaluate] negative log-likelihood per prediction {lk['nll']:.3f} "
            f"(event {lk.get('nll_event', float('nan')):.3f}, time {lk.get('nll_time', float('nan')):.3f}) "
            f"over {lk['predictions']:,} predictions")
    log(f"[evaluate] → {out_csv}")
    return result
