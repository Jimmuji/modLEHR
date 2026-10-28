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

To compare models, score them on one set of prediction points (`--points-dir`): drawn once, from the
data alone, and read by every later model, so each is scored on the same diseases, cases, controls and
times. A model with a "no event" token can also be read at one placed a gap before the diagnosis
(`--query gap`) rather than at the last event before it, on the same points.

One difference from Delphi-2M's code, for speed: each control patient's prediction point per band
is drawn once and shared by all diseases (Delphi-2M draws again for each disease). Each draw is
uniform over the patient's events, so the expected AUC is the same.
"""

from __future__ import annotations

import hashlib
import json
import os
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
        rows = [split.subject(s, a, z, labels=False) for s, a, z, _ in chunk]
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


def split_fingerprint(split: PreparedSplit, vocab) -> str:
    """A hash of what the prediction points depend on: subjects, event ages, follow-up and the age at each
    first target (by target name). Token ids are left out, so a vocabulary with one more token (the "no
    event" token) still matches."""
    first = split_first_ages(split, vocab)
    order = np.argsort(np.asarray(vocab.targets, dtype=object))
    h = hashlib.sha256()
    for part in (split.subject_ids, split.offsets, np.asarray(split.arrays["ages"]), split.follow_up_end,
                 first[:, order]):
        h.update(np.ascontiguousarray(part).tobytes())
    h.update("\n".join(np.asarray(vocab.targets, dtype=object)[order]).encode())
    return h.hexdigest()[:16]


def draw_points(split: PreparedSplit, vocab, gap_days: float, seed: int = 0) -> pd.DataFrame:
    """The prediction points of Delphi-2M's AUC, one row each: subject_id, sex, age_band (its start),
    role (case or control), target (a case's disease, empty for a control), age (of the event predicted
    from) and anchor (a case's diagnosis age, or the control event the point was drawn back from)."""
    first = split_first_ages(split, vocab)
    gap = gap_days / 365.25
    rng = np.random.default_rng(seed)
    sex_ids = {name: vocab.token_index[t] for name, t in SEX_TOKENS.items() if t in vocab.token_index}
    tokens, ages_all, offsets = np.asarray(split.arrays["tokens"]), np.asarray(split.arrays["ages"]), split.offsets
    rows = []
    for s in range(len(offsets) - 1):
        a, b = offsets[s], offsets[s + 1]
        ages, toks = ages_all[a:b], tokens[a:b]
        sex = "all"
        for name, t in sex_ids.items():
            if (toks == t).any():
                sex = name
        sid = int(split.subject_ids[s])
        # controls: one event per band, drawn uniformly over the patient's events in follow-up
        anchors = ages[ages <= split.follow_up_end[s]]
        p = prediction_index(ages, anchors, gap)
        anchors, p = anchors[p >= 0], p[p >= 0]
        bands = band_of(ages[p])
        for k in np.unique(bands[bands >= 0]):
            j = rng.choice(np.flatnonzero(bands == k))
            rows.append((sid, sex, int(AGE_BANDS[k]), "control", "", float(ages[p[j]]), float(anchors[j])))
        # cases: every target first recorded in follow-up
        targets = np.flatnonzero(np.isfinite(first[s]))
        if len(targets):
            pi = prediction_index(ages, first[s, targets], gap)
            ok = pi >= 0
            k = band_of(ages[np.maximum(pi, 0)])
            for t, e, kb in zip(targets[ok & (k >= 0)], pi[ok & (k >= 0)], k[ok & (k >= 0)]):
                rows.append((sid, sex, int(AGE_BANDS[kb]), "case", vocab.targets[t], float(ages[e]), float(first[s, t])))
    return pd.DataFrame(rows, columns=["subject_id", "sex", "age_band", "role", "target", "age", "anchor"])


def points_for(split: PreparedSplit, vocab, gap_days: float, seed: int, points_dir: str | Path | None,
               log=print) -> tuple[pd.DataFrame, dict]:
    """The prediction points: read from points_dir when it holds them for this gap (they must come from
    the same data), otherwise drawn from this split and, with points_dir, written there for other
    models to share. Shared points are drawn with seed 0 whatever the model's seed, so they depend on
    the data alone and jobs that draw them at once write the same file. They name patients and ages,
    so they stay with the data, never with results."""
    fingerprint = split_fingerprint(split, vocab)
    if points_dir:
        seed = 0
    meta = {"gap_days": gap_days, "seed": seed, "fingerprint": fingerprint}
    path = Path(points_dir) / f"points_gap{gap_days:g}d.parquet" if points_dir else None
    if path is not None and path.exists():
        stored = json.loads(path.with_suffix(".json").read_text())
        if stored["fingerprint"] != fingerprint:
            raise ValueError(f"{path} was drawn from other data (fingerprint {stored['fingerprint']}, this split "
                             f"{fingerprint}): subjects, ages, follow-up or first diagnoses differ")
        log(f"[delphi] prediction points from {path}")
        return pd.read_parquet(path), {**stored, "points": str(path)}
    points = draw_points(split, vocab, gap_days, seed)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}")  # written whole, then renamed: readers never see half
        tmp.write_text(json.dumps({**meta, "rows": len(points)}, indent=1))
        os.replace(tmp, path.with_suffix(".json"))
        points.to_parquet(tmp, index=False)
        os.replace(tmp, path)
        log(f"[delphi] prediction points drawn and written to {path}")
        meta["points"] = str(path)
    return points, meta


@torch.no_grad()
def log_rates_with_query(model, split: PreparedSplit, keys: np.ndarray, batch_size: int,
                         device: torch.device) -> np.ndarray:
    """[len(keys), n_targets] log-rates at a query token (the model's "no event") appended after event e
    at age q, for keys (subject index, event index e, q)."""
    from modlehr.evaluate.run import with_event

    max_len = model.cfg.max_len - 1
    order = np.argsort(keys[:, 1], kind="stable")
    out = None
    for b in range(0, len(order), batch_size):
        idx = order[b:b + batch_size]
        rows = []
        for s, e, q in keys[idx]:
            s, e = int(s), int(e)
            rows.append(with_event(split.subject(s, max(0, e + 1 - max_len), e + 1, labels=False), model.query_token, q))
        log_rates = model(to_device(collate(rows), device)).float()
        last = torch.tensor([len(r["tokens"]) - 1 for r in rows], device=device)
        out = np.zeros((len(keys), log_rates.shape[-1]), dtype=np.float32) if out is None else out
        out[idx] = log_rates[torch.arange(len(rows), device=device), last].cpu().numpy()
    return out if out is not None else np.zeros((0, 0), dtype=np.float32)


QUERIES = ("event", "gap")


def delphi_auc(cfg: Config, checkpoint: str | Path, gap_days: float, out_csv: str | Path | None = None,
               seed: int = 0, log=print,
               points_dir: str | Path | None = None, query: str = "event",
               targets_file: str | Path | None = None) -> pd.DataFrame:
    """Delphi-2M's AUC for one checkpoint and gap. points_dir shares one set of prediction points between
    models (points_for), so every model is scored on the same diseases, cases, controls and times.

    query: "event" reads the model at the event each point predicts from; "gap" appends the model's "no
    event" token at (anchor - gap), the latest time the point may use, so the model knows how long it
    has been since that event (a model trained with data.no_event_years, or Delphi-2M's own code).

    targets_file: a list of target names, one per line (# for comments), such as
    configs/targets/delphi2m.txt; the summary then also gives the means over the listed targets only.
    The table keeps every target."""
    from modlehr.evaluate.run import load_model

    if query not in QUERIES:
        raise ValueError(f"query must be one of {QUERIES}, not {query!r}")
    prepared = Path(cfg.data.prepared_dir)
    device = pick_device(cfg.train.device)
    model, vocab = load_model(checkpoint, prepared, device)
    if query == "gap" and getattr(model, "query_token", None) is None:
        raise ValueError("query=gap needs a model with a \"no event\" token: one trained with data.no_event_years, "
                         "or a checkpoint of Delphi-2M's own code")
    split = open_split(prepared, cfg.evaluate.split)
    first = split_first_ages(split, vocab)
    gap = gap_days / 365.25
    points, meta = points_for(split, vocab, gap_days, seed, points_dir, log)

    index = {int(sid): i for i, sid in enumerate(split.subject_ids)}
    subject = points["subject_id"].map(index).to_numpy()
    if np.isnan(subject.astype(float)).any():
        raise ValueError("the prediction points name subjects that are not in this split")
    subject = subject.astype(int)
    ages_all, offsets = np.asarray(split.arrays["ages"]), split.offsets
    p_age = points["age"].to_numpy(np.float32)
    # each point's event: the last one at its age, found in one search over (subject, age) keys
    owner = np.repeat(np.arange(len(offsets) - 1), np.diff(offsets))
    event = np.searchsorted(owner * 1000.0 + ages_all, subject * 1000.0 + p_age, side="right") - 1 - offsets[subject]
    if len(event) and not np.array_equal(ages_all[offsets[subject] + event], p_age):
        raise ValueError("the prediction points' ages are not event ages of this split")

    if query == "event":
        pair = offsets[subject] + event  # the event's index in the whole split
        unique, row = np.unique(pair, return_inverse=True)
        u_subject = owner[unique]
        needed = {int(s): unique[u_subject == s] - offsets[s] for s in np.unique(u_subject)}
        log(f"[delphi] gap {gap_days:g} days: rates at {len(unique):,} events of "
            f"{len(needed):,} {cfg.evaluate.split} patients")
        rates = log_rates_at(model, split, needed, cfg.evaluate.batch_size, device)
        table = np.concatenate([rates[s] for s in sorted(needed)]) if needed else np.zeros((0, len(vocab.targets)), np.float32)
    else:
        q_age = (points["anchor"].to_numpy() - gap).astype(np.float32)
        keys, row = np.unique(np.c_[subject, event, q_age], axis=0, return_inverse=True)
        log(f"[delphi] gap {gap_days:g} days: rates at {len(keys):,} \"no event\" queries of "
            f"{len(np.unique(subject)):,} {cfg.evaluate.split} patients")
        table = log_rates_with_query(model, split, keys, cfg.evaluate.batch_size, device)
    row = np.asarray(row).reshape(-1)  # each point's row of table

    band = np.searchsorted(AGE_BANDS, points["age_band"].to_numpy())
    sex = np.full(len(offsets) - 1, "all", dtype=object)
    sex[subject] = points["sex"].to_numpy()
    is_case = (points["role"] == "case").to_numpy()
    target_index = {t: i for i, t in enumerate(vocab.targets)}
    missing = set(points.loc[is_case, "target"]) - set(target_index)
    if missing:
        raise ValueError(f"the prediction points name targets this vocabulary lacks: {sorted(missing)[:5]}")
    k_target = points.loc[is_case, "target"].map(target_index).to_numpy(int)
    cases = (subject[is_case], k_target, band[is_case], table[row[is_case], k_target])
    controls = (subject[~is_case], band[~is_case], table[row[~is_case]])

    by_band = band_aucs(vocab.targets, first, sex, cases, controls)
    per_disease = by_band.groupby(["target", "sex"]).agg(auc=("auc", "mean"), bands=("auc", "size"),
                                                         cases=("cases", "sum")).reset_index()
    def by_sex(table: pd.DataFrame) -> tuple[dict, dict]:
        return ({sx: {"mean_auc": float(g["auc"].mean()), "diseases": int(len(g))} for sx, g in table.groupby("sex")},
                {sx: float(g.loc[g["cases"] >= 25, "auc"].mean()) for sx, g in table.groupby("sex")})

    summary = {"gap_days": gap_days, "split": cfg.evaluate.split, "checkpoint": str(checkpoint), "query": query,
               "points_fingerprint": meta["fingerprint"], "points": meta.get("points"),
               "definition": "Delphi-2M evaluate_auc_pipeline: per (disease, sex) mean AUC over 5-year age bands",
               "by_sex": by_sex(per_disease)[0],
               "mean_auc": float(per_disease["auc"].mean()) if len(per_disease) else None,
               "diseases_with_25_cases": by_sex(per_disease)[1]}
    if targets_file:
        listed = {line.strip() for line in Path(targets_file).read_text().splitlines()
                  if line.strip() and not line.startswith("#")}
        mine = per_disease[per_disease["target"].isin(listed)]
        summary["targets"] = {"file": str(targets_file), "listed": len(listed),
                              "in_vocabulary": len(listed & set(vocab.targets)),
                              "by_sex": by_sex(mine)[0], "diseases_with_25_cases": by_sex(mine)[1]}
    out_csv = Path(out_csv) if out_csv else Path(cfg.train.out_dir) / f"evaluate_delphi_gap{gap_days:g}d.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    by_band.to_csv(out_csv, index=False)
    out_csv.with_suffix(".json").write_text(json.dumps(summary, indent=1))
    log(f"[delphi] gap {gap_days:g} days, query at {query}: " + ", ".join(
        f"{sx} {v['mean_auc']:.3f} over {v['diseases']} diseases" for sx, v in summary["by_sex"].items())
        + f" → {out_csv}")
    if targets_file:
        log(f"[delphi]   targets in {targets_file}: " + ", ".join(
            f"{sx} {v['mean_auc']:.3f} over {v['diseases']} diseases" for sx, v in summary["targets"]["by_sex"].items()))
    return by_band
