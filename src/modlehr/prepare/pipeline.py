"""`modlehr prepare`: MEDS → tokenized timelines with labels, one .npz per split.

Output directory:
    vocab.json      input and target vocabularies
    prepare.json    value statistics or bin cut points, target counts, exposure, the data config
    <split>.npz     flat arrays per split (see prepare.timeline)
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import meds
import numpy as np
import pandas as pd

from modlehr import registry
from modlehr.config import DataConfig
from modlehr.data.meds_io import open_meds
from modlehr.prepare import grouping  # noqa: F401  (registers the target groupings)
from modlehr.prepare.select import ROLES, load_events
from modlehr.prepare.timeline import build_split
from modlehr.prepare.values import apply_bins, fit_bins, fit_scaler, scale
from modlehr.prepare.vocabulary import Vocabulary

SPLITS = (meds.train_split, meds.tuning_split, meds.held_out_split)


def tokenize(events: pd.DataFrame, cfg: DataConfig) -> pd.DataFrame:
    rules = cfg.events
    role = pd.Series(np.array([r.role for r in rules])[events["rule"].to_numpy()], index=events.index)
    code = events["code"].astype(str)
    token = code.copy()

    is_target = role == "target"
    group = registry.get("target_grouping", cfg.target_grouping)
    unique = code[is_target].unique()
    token[is_target] = code[is_target].map(dict(zip(unique, map(group, unique))))

    is_cat = (role == "categorical") & events["text_value"].notna()
    token[is_cat] = code[is_cat] + "//" + events.loc[is_cat, "text_value"].astype(str).str.strip()

    out = pd.DataFrame({"subject_id": events["subject_id"], "code": code, "token": token, "role": role,
                        "rule": events["rule"], "age": events["age"],
                        "value": events["numeric_value"].astype("float64").where(role == "measurement")})
    out.loc[out["role"] == "static", "age"] = 0.0
    out = out.sort_values(["subject_id", "age"], kind="stable")

    for i, rule in enumerate(rules):
        if rule.role == "measurement" and rule.thin_years:
            rows = out["rule"] == i
            bucket = np.floor(out.loc[rows, "age"] / rule.thin_years)
            dup = out.loc[rows].assign(bucket=bucket).duplicated(["subject_id", "code", "bucket"])
            out = out.drop(index=dup[dup].index)
        first = rule.first_occurrence if rule.first_occurrence is not None else rule.role == "target"
        if first:
            rows = out["rule"] == i
            dup = out.loc[rows].duplicated(["subject_id", "token"])
            out = out.drop(index=dup[dup].index)
    return out


def run_prepare(cfg: DataConfig, log=print) -> dict:
    out_dir = Path(cfg.prepared_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = open_meds(cfg.meds_dir)
    log(f"[prepare] reading {cfg.meds_dir}")
    events = load_events(ds, cfg.events)
    log(f"[prepare] {len(events):,} matched events, {events['subject_id'].nunique():,} subjects")
    frame = tokenize(events, cfg)
    split_of = dict(zip(ds.splits["subject_id"], ds.splits["split"]))
    frame["split"] = frame["subject_id"].map(split_of).fillna(meds.train_split)
    train = frame["split"] == meds.train_split
    measurement = frame["role"] == "measurement"

    meta: dict = {"data_config": asdict(cfg)}
    if cfg.value_mode == "binned":
        cuts = fit_bins(frame.loc[train & measurement, "code"], frame.loc[train & measurement, "value"], cfg.n_bins)
        frame.loc[measurement, "token"] = apply_bins(frame.loc[measurement, "code"], frame.loc[measurement, "value"], cuts)
        frame["value"] = np.nan
        meta["bins"] = cuts
    elif cfg.value_mode == "none":
        frame["value"] = np.nan
    elif cfg.value_mode != "continuous":
        raise ValueError(f"value_mode must be continuous, binned or none, not '{cfg.value_mode}'")

    counts = frame.loc[frame["split"] == meds.train_split].groupby("token")["subject_id"].nunique()
    kept = set(counts[counts >= cfg.min_subjects_per_token].index)
    targets = sorted(set(frame.loc[frame["role"] == "target", "token"]) & kept)
    vocab = Vocabulary.build(sorted(kept), targets)
    frame = frame.loc[frame["token"].isin(kept)].copy()
    frame["token_id"] = frame["token"].map(vocab.token_index).astype(np.int32)

    if cfg.value_mode == "continuous":
        rows = (frame["split"] == meds.train_split) & (frame["role"] == "measurement")
        stats = fit_scaler(frame.loc[rows, "token"], frame.loc[rows, "value"])
        frame["value"] = scale(frame["token"], frame["value"], stats)
        meta["value_stats"] = stats

    order = {r: i for i, r in enumerate(ROLES)}
    frame["order"] = frame["role"].map(order)
    frame = frame.sort_values(["subject_id", "age", "order", "token_id"], kind="stable")

    summary = {}
    for split in SPLITS:
        part = frame.loc[frame["split"] == split]
        arrays = build_split(part, vocab)
        np.savez(out_dir / f"{split}.npz", **arrays)
        summary[split] = {"subjects": int(len(arrays["subject_ids"])), "events": int(len(arrays["tokens"]))}
        if split == meds.train_split:
            ages, offs = arrays["ages"], arrays["offsets"]
            exposure = float(np.sum(ages[offs[1:] - 1] - ages[offs[:-1]]))
            is_t = vocab.input_to_target[arrays["tokens"]] >= 0
            target_counts = np.bincount(vocab.input_to_target[arrays["tokens"][is_t]], minlength=len(targets))
            meta["exposure_years"] = exposure
            meta["target_counts"] = target_counts.tolist()
        log(f"[prepare] {split}: {summary[split]['subjects']:,} subjects, {summary[split]['events']:,} events")

    vocab.save(out_dir / "vocab.json")
    meta["summary"] = summary
    meta["vocab_size"] = len(vocab.tokens)
    meta["n_targets"] = len(vocab.targets)
    (out_dir / "prepare.json").write_text(json.dumps(meta, indent=1))
    log(f"[prepare] vocabulary {len(vocab.tokens):,} tokens, {len(vocab.targets):,} targets → {out_dir}")
    return meta
