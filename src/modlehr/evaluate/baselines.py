"""`modlehr baseline`: classical models on the same landmark task as `modlehr evaluate`.

Before reading anything into a transformer's AUC, check what a simple model gets from the same
inputs. If a logistic regression does as well, the transformer has not learned anything the
inputs did not already say; if gradient boosting does no better than the logistic regression,
the data carry little signal beyond single features, and a bigger model will not find more.

Features for a subject at landmark L, built from the prepared timeline:

    one column per input token: 1 if it occurred at or before L
    with value_mode: continuous, one more column per token: its latest scaled value before L
    (0, the training mean, when there is none)

Features are stored sparse, so vocabularies of tens of thousands of tokens fit in memory.
Gradient boosting needs a dense matrix and uses the max_dense_features columns that are
non-empty for the most training subjects (all of them when there are fewer).

Models:

    logistic   L2-regularized logistic regression
    gbm        histogram gradient boosting, which can use thresholds and interactions

Each (target, landmark) pair is fit on the training split and scored on the evaluation split.
Only pairs that the model evaluation also scores are fit, so the numbers line up pair by pair.
Each model's results go to <train.out_dir>_<model>/evaluate.csv and .json, in the same format
as `modlehr evaluate`, so `modlehr compare` treats a baseline like any other run.
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import meds
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from scipy import sparse
from sklearn.metrics import roc_auc_score

from modlehr.config import Config
from modlehr.evaluate.task import LandmarkTask, history_before, landmark_task, summarize
from modlehr.prepare.dataset import PreparedSplit, open_split
from modlehr.prepare.timeline import first_target_ages
from modlehr.prepare.vocabulary import Vocabulary

MODELS = ("logistic", "gbm")


def make_model(name: str, seed: int):
    if name == "logistic":
        return LogisticRegression(C=0.5, tol=1e-6, max_iter=10000)  # converged tightly: it is the yardstick
    if name == "gbm":
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, early_stopping=True, random_state=seed)
    raise ValueError(f"unknown baseline '{name}'; available: {MODELS}")


def landmark_features(split: PreparedSplit, task: LandmarkTask, n_tokens: int, with_values: bool) -> sparse.csr_matrix:
    """Sparse [len(task.subjects), n_tokens (x2 with values)]: token present by L, and its latest value."""
    rows, cols, vals = [], [], []
    for row, i in enumerate(task.subjects):
        h = history_before(split, i, task.landmark)
        present = np.unique(h["tokens"])
        present = present[present != 0]
        rows.append(np.full(len(present), row)), cols.append(present), vals.append(np.ones(len(present), np.float32))
        if with_values:
            has = ~np.isnan(h["values"])
            if has.any():
                tokens, values = h["tokens"][has][::-1], h["values"][has][::-1]  # latest first
                unique, first = np.unique(tokens, return_index=True)
                rows.append(np.full(len(unique), row)), cols.append(n_tokens + unique), vals.append(values[first])
    shape = (len(task.subjects), n_tokens * (2 if with_values else 1))
    if not rows:
        return sparse.csr_matrix(shape, dtype=np.float32)
    return sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=shape,
                             dtype=np.float32)


def dense_columns(X: sparse.csr_matrix, k: int) -> np.ndarray:
    """The k columns that are non-empty for the most training rows, in their original order."""
    if X.shape[1] <= k:
        return np.arange(X.shape[1])
    return np.sort(np.argsort(-X.getnnz(axis=0), kind="stable")[:k])


def run_baselines(cfg: Config, models: tuple[str, ...] = MODELS, max_dense_features: int = 2000,
                  log=print) -> dict[str, Path]:
    for name in models:
        make_model(name, 0)  # fail early on an unknown name
    prepared = Path(cfg.data.prepared_dir)
    vocab = Vocabulary.load(prepared / "vocab.json")
    meta = json.loads((prepared / "prepare.json").read_text())
    with_values = meta["data_config"]["value_mode"] == "continuous"
    train_split = open_split(prepared, meds.train_split)
    eval_split = open_split(prepared, cfg.evaluate.split)
    first_train = first_target_ages(train_split.arrays["tokens"], train_split.arrays["ages"], train_split.offsets, vocab)
    first_eval = first_target_ages(eval_split.arrays["tokens"], eval_split.arrays["ages"], eval_split.offsets, vocab)
    H, min_cases, n_tokens = cfg.evaluate.horizon_years, cfg.evaluate.min_cases, len(vocab.tokens)
    log(f"[baseline] {', '.join(models)} on {n_tokens:,} token features"
        f"{' plus their latest values' if with_values else ''}, {len(vocab.targets):,} targets")

    rows: dict[str, list[dict]] = {m: [] for m in models}
    t0 = time.time()
    for L in cfg.evaluate.landmarks:
        tr = landmark_task(train_split, vocab, L, H, first_train, cfg.evaluate.controls)
        ev = landmark_task(eval_split, vocab, L, H, first_eval, cfg.evaluate.controls)
        X_tr = landmark_features(train_split, tr, n_tokens, with_values)
        X_ev = landmark_features(eval_split, ev, n_tokens, with_values)
        if "gbm" in models:  # gradient boosting needs a dense matrix: keep the most common columns
            keep_cols = dense_columns(X_tr, max_dense_features)
            dense_tr, dense_ev = X_tr[:, keep_cols].toarray(), X_ev[:, keep_cols].toarray()
        fitted = 0
        for k, target in enumerate(vocab.targets):
            keep_e, case_e = ev.labels(k)
            keep_t, case_t = tr.labels(k)
            counts = (case_e[keep_e].sum(), (~case_e[keep_e]).sum(), case_t[keep_t].sum(), (~case_t[keep_t]).sum())
            if min(counts) < min_cases:
                continue
            for name in models:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    tr_x, ev_x = (dense_tr, dense_ev) if name == "gbm" else (X_tr, X_ev)
                    clf = make_model(name, cfg.train.seed).fit(tr_x[keep_t], case_t[keep_t])
                score = clf.predict_proba(ev_x[keep_e])[:, 1]
                rows[name].append({"target": target, "landmark": L, "cases": int(counts[0]), "controls": int(counts[1]),
                                   "auc": roc_auc_score(case_e[keep_e], score)})
            fitted += 1
        log(f"[baseline] landmark {L:g}: {fitted} targets fit on {len(tr.subjects):,} training subjects "
            f"({time.time() - t0:.0f}s)")

    run_dir = Path(cfg.train.out_dir)
    written = {}
    for name in models:
        result = pd.DataFrame(rows[name], columns=["target", "landmark", "cases", "controls", "auc"])
        out_dir = run_dir.with_name(f"{run_dir.name}_{name}")
        out_dir.mkdir(parents=True, exist_ok=True)
        result.to_csv(out_dir / "evaluate.csv", index=False)
        summary = {**summarize(result), "horizon_years": H, "controls": cfg.evaluate.controls,
                   "landmarks": cfg.evaluate.landmarks, "split": cfg.evaluate.split, "baseline": name,
                   "values": with_values}
        (out_dir / "evaluate.json").write_text(json.dumps(summary, indent=1))
        if len(result):
            lo, hi = summary["mean_auc_ci95"]
            log(f"[baseline] {name}: mean AUC {summary['mean_auc']:.3f} (95% CI {lo:.3f} to {hi:.3f}) "
                f"over {summary['pairs']} pairs → {out_dir / 'evaluate.csv'}")
        else:
            log(f"[baseline] {name}: no pair had enough cases and controls")
        written[name] = out_dir
    return written
