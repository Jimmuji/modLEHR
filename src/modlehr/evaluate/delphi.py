"""`modlehr delphi`: Delphi-2M's own AUC, to set a model next to the numbers Delphi-2M published.

Delphi-2M (Shmatko et al., Nature 2025; evaluate_auc_pipeline in its code) scores each disease k so:

    cases      every patient whose first k is in the record, predicted from the last event strictly
               before (diagnosis time - gap)
    controls   patients who never have k, each predicted from the last event strictly before
               (one of their own event times - gap)
    score      the predicted rate of k at that event (here the log-rate, which ranks the same)
    strata     sex, and the age at the prediction in 5-year bands from 40 to 80; one prediction per
               patient per band; a band counts with at least 2 cases and 2 controls
    result     per (disease, sex), the mean AUC over the bands; Delphi-2M reports the mean over
               diseases per sex: 0.760 (female) and 0.754 (male) with a gap of 0.1 day, 0.701 and
               0.692 with a gap of one year (UK Biobank, its supplementary table)

This is not modLEHR's main task (`modlehr evaluate`, from a fixed age forward over a window). Here a
case's prediction point is chosen by when its diagnosis happened, and with a short gap that point is
usually the admission of the same stay on MIMIC-IV, so the score mixes forecasting with recognising
a patient already in hospital. Report it for comparison with Delphi-2M, next to the landmark AUC.

One difference from Delphi-2M's code, for speed: each control patient's prediction point per band
is drawn once and shared by all diseases (Delphi-2M draws again for each disease). Each draw is
uniform over the patient's events, so the expected AUC is the same.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from modlehr.config import Config
from modlehr.evaluate.task import split_first_ages
from modlehr.prepare.dataset import PreparedSplit, collate, open_split
from modlehr.train.loop import pick_device, to_device

AGE_BANDS = np.arange(40, 80, 5)  # Delphi-2M's evaluate_auc_pipeline: [40, 45), ..., [75, 80)
SEX_TOKENS = {"female": "GENDER//F", "male": "GENDER//M"}
MIN_PER_BAND = 2


def prediction_index(ages: np.ndarray, times: np.ndarray, gap_years: float) -> np.ndarray:
    """For each time, the index of the last event strictly before (time - gap); -1 if none."""
    return np.searchsorted(ages, times - gap_years, side="left") - 1


def band_of(age: np.ndarray) -> np.ndarray:
    """Index into AGE_BANDS, or -1 outside 40 to 80."""
    b = np.floor((age - AGE_BANDS[0]) / 5).astype(int)
    return np.where((age >= AGE_BANDS[0]) & (age < AGE_BANDS[-1] + 5), b, -1)


@torch.no_grad()
def log_rates_at(model, split: PreparedSplit, needed: dict[int, np.ndarray], batch_size: int,
                 device: torch.device) -> dict[int, np.ndarray]:
    """{subject: [len(events), n_targets] log-rates at those event indices}, from each event's history
    (its last max_len events)."""
    max_len = model.cfg.max_len
    jobs = []  # (subject, events): one window per subject when it fits, one per event otherwise
    for s, events in needed.items():
        n = int(split.offsets[s + 1] - split.offsets[s])
        if n <= max_len:
            jobs.append((s, 0, n, events))
        else:
            jobs += [(s, max(0, int(e) + 1 - max_len), int(e) + 1, np.array([e])) for e in events]
    out: dict[int, list] = {s: [] for s in needed}
    jobs.sort(key=lambda j: j[2] - j[1])
    for b in range(0, len(jobs), batch_size):
        chunk = jobs[b:b + batch_size]
        rows = [split.subject(s, a, z) for s, a, z, _ in chunk]
        log_rates = model(to_device(collate(rows), device)).float()
        for j, (s, a, _, events) in enumerate(chunk):
            out[s].append((events, log_rates[j, torch.as_tensor(events - a, device=device)].cpu().numpy()))
    result = {}
    for s, parts in out.items():
        order = np.concatenate([e for e, _ in parts])
        values = np.concatenate([v for _, v in parts])
        result[s] = values[np.argsort(order, kind="stable")]
    return result


def band_aucs(targets: list[str], first: np.ndarray, sex: np.ndarray, cases: tuple, controls: tuple) -> pd.DataFrame:
    """One AUC per (target, sex, age band) with at least MIN_PER_BAND cases and controls.

    first: [n_subjects, n_targets] age at each first target; sex: [n_subjects] name. cases: (subject,
    target, band, score) per case; controls: (subject, band, [n, n_targets] scores) per control
    prediction point. A control counts for the targets its subject never has. Cases and controls are
    sorted by sex and band once, so each target reads only its own rows.
    """
    k_subject, k_target, k_band, k_score = cases
    c_subject, c_band, c_rates = controls
    names, sex_code = np.unique(sex, return_inverse=True)
    n_groups = len(names) * len(AGE_BANDS)  # group = sex, then band: the order rows are written in
    c_group = sex_code[c_subject] * len(AGE_BANDS) + c_band
    c_order = np.argsort(c_group, kind="stable")
    c_bounds = np.searchsorted(c_group[c_order], np.arange(n_groups + 1))
    k_group = sex_code[k_subject] * len(AGE_BANDS) + k_band
    k_order = np.lexsort((k_group, k_target))
    t_bounds = np.searchsorted(k_target[k_order], np.arange(len(targets) + 1))
    rows = []
    for t, name in enumerate(targets):
        mine = k_order[t_bounds[t]:t_bounds[t + 1]]
        if len(mine) < MIN_PER_BAND:
            continue
        never = np.isinf(first[c_subject, t])
        g_bounds = np.searchsorted(k_group[mine], np.arange(n_groups + 1))
        for g in np.flatnonzero(np.diff(g_bounds) >= MIN_PER_BAND):
            case_scores = k_score[mine[g_bounds[g]:g_bounds[g + 1]]]
            pool = c_order[c_bounds[g]:c_bounds[g + 1]]
            control_scores = c_rates[pool[never[pool]], t]
            if len(control_scores) < MIN_PER_BAND:
                continue
            y = np.r_[np.ones(len(case_scores)), np.zeros(len(control_scores))]
            rows.append({"target": name, "sex": names[g // len(AGE_BANDS)], "age_band": int(AGE_BANDS[g % len(AGE_BANDS)]),
                         "cases": len(case_scores), "controls": len(control_scores),
                         "auc": roc_auc_score(y, np.r_[case_scores, control_scores])})
    return pd.DataFrame(rows, columns=["target", "sex", "age_band", "cases", "controls", "auc"])


def delphi_auc(cfg: Config, checkpoint: str | Path, gap_days: float, out_csv: str | Path | None = None,
               seed: int = 0, log=print) -> pd.DataFrame:
    from modlehr.evaluate.run import load_model

    prepared = Path(cfg.data.prepared_dir)
    device = pick_device(cfg.train.device)
    model, vocab = load_model(checkpoint, prepared, device)
    split = open_split(prepared, cfg.evaluate.split)
    first = split_first_ages(split, vocab)
    gap = gap_days / 365.25
    rng = np.random.default_rng(seed)
    sex_ids = {name: vocab.token_index[t] for name, t in SEX_TOKENS.items() if t in vocab.token_index}
    tokens, ages_all, offsets = np.asarray(split.arrays["tokens"]), np.asarray(split.arrays["ages"]), split.offsets

    sex = np.full(len(offsets) - 1, "all", dtype=object)
    control_event = np.full((len(offsets) - 1, len(AGE_BANDS)), -1)
    case_event: dict[int, list[tuple[int, int]]] = {}  # subject -> [(target, event index)]
    for s in range(len(offsets) - 1):
        a, b = offsets[s], offsets[s + 1]
        ages, toks = ages_all[a:b], tokens[a:b]
        for name, t in sex_ids.items():
            if (toks == t).any():
                sex[s] = name
        followed = ages <= split.follow_up_end[s]
        # controls: one event per band, drawn uniformly over the patient's events in follow-up
        p = prediction_index(ages, ages[followed], gap)
        p = p[p >= 0]
        bands = band_of(ages[p])
        for k in np.unique(bands[bands >= 0]):
            control_event[s, k] = rng.choice(p[bands == k])
        # cases: every target first recorded in follow-up
        targets = np.flatnonzero(np.isfinite(first[s]))
        if len(targets):
            pi = prediction_index(ages, first[s, targets], gap)
            ok = (pi >= 0) & (band_of(ages[np.maximum(pi, 0)]) >= 0)
            case_event[s] = list(zip(targets[ok].tolist(), pi[ok].tolist()))

    needed = {}
    for s in range(len(offsets) - 1):
        ev = set(control_event[s][control_event[s] >= 0].tolist()) | {e for _, e in case_event.get(s, [])}
        if ev:
            needed[s] = np.array(sorted(ev))
    log(f"[delphi] gap {gap_days:g} days: rates at {sum(len(v) for v in needed.values()):,} events of "
        f"{len(needed):,} {cfg.evaluate.split} patients")
    rates = log_rates_at(model, split, needed, cfg.evaluate.batch_size, device)

    def rate(s: int, event: int) -> np.ndarray:
        return rates[s][np.searchsorted(needed[s], event)]

    # controls as rows: patient, band, sex, log-rates for every target
    c_rows = [(s, k) for s in range(len(offsets) - 1) for k in np.flatnonzero(control_event[s] >= 0)]
    c_subject = np.array([s for s, _ in c_rows], dtype=int)
    c_band = np.array([k for _, k in c_rows], dtype=int)
    c_rates = np.stack([rate(s, control_event[s, k]) for s, k in c_rows]) if c_rows else np.zeros((0, len(vocab.targets)))
    cases = [(s, t, e) for s, items in case_event.items() for t, e in items]
    k_subject = np.array([s for s, _, _ in cases], dtype=int)
    k_target = np.array([t for _, t, _ in cases], dtype=int)
    k_band = band_of(ages_all[offsets[k_subject] + np.array([e for _, _, e in cases], dtype=int)]) if cases else np.array([], int)
    k_score = np.array([rate(s, e)[t] for s, t, e in cases]) if cases else np.array([])

    by_band = band_aucs(vocab.targets, first, sex, (k_subject, k_target, k_band, k_score), (c_subject, c_band, c_rates))
    per_disease = by_band.groupby(["target", "sex"]).agg(auc=("auc", "mean"), bands=("auc", "size"),
                                                         cases=("cases", "sum")).reset_index()
    summary = {"gap_days": gap_days, "split": cfg.evaluate.split, "checkpoint": str(checkpoint),
               "definition": "Delphi-2M evaluate_auc_pipeline: per (disease, sex) mean AUC over 5-year age bands",
               "by_sex": {sx: {"mean_auc": float(g["auc"].mean()), "diseases": int(len(g))}
                          for sx, g in per_disease.groupby("sex")},
               "mean_auc": float(per_disease["auc"].mean()) if len(per_disease) else None,
               "diseases_with_25_cases": {sx: float(g.loc[g["cases"] >= 25, "auc"].mean())
                                          for sx, g in per_disease.groupby("sex")}}
    out_csv = Path(out_csv) if out_csv else Path(cfg.train.out_dir) / f"evaluate_delphi_gap{gap_days:g}d.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    by_band.to_csv(out_csv, index=False)
    out_csv.with_suffix(".json").write_text(json.dumps(summary, indent=1))
    log(f"[delphi] gap {gap_days:g} days: " + ", ".join(
        f"{sx} {v['mean_auc']:.3f} over {v['diseases']} diseases" for sx, v in summary["by_sex"].items())
        + f" → {out_csv}")
    return by_band
