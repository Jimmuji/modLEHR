"""`modlehr prepare`: MEDS → tokenized timelines with labels.

Works one MEDS shard at a time, so memory grows with the largest shard, not with the dataset.
Statistics come from the training split only:

    pass 1  training shards: collect measurement values (bin cut points or scaling statistics)
    pass 2  training shards: count how many training subjects have each token (binned tokens
            need the cut points from pass 1, so the two passes are separate)
    pass 3  every shard: tokenize, drop rare tokens, attach labels, append to its split

Pass 1 is skipped when values are dropped (value_mode: none).

Output directory:
    vocab.json      input and target vocabularies
    prepare.json    the data config, a fingerprint of the MEDS files read, per-rule counts (codes
                    matched, tokens, tokens kept), value statistics or bin cut points, target counts
                    and exposure in the training split
    <split>/        one .npy per array, subjects in subject_id order (see prepare.dataset)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import meds
import numpy as np
import pandas as pd

from modlehr.config import DataConfig
from modlehr.data.meds_io import open_meds
from modlehr.prepare.grouping import make_grouping
from modlehr.prepare.select import ROLES, classify_codes, read_events
from modlehr.prepare.dataset import SplitWriter
from modlehr.prepare.text import build_text
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
    group = make_grouping(cfg.target_grouping, cfg.target_grouping_options)
    unique = code[is_target].unique()
    token[is_target] = code[is_target].map(dict(zip(unique, map(group, unique))))

    is_cat = (role == "categorical") & events["text_value"].notna()
    token[is_cat] = code[is_cat] + "//" + events.loc[is_cat, "text_value"].astype(str).str.strip()

    follows = pd.Series(np.array([r.follow_up for r in rules])[events["rule"].to_numpy()], index=events.index)
    out = pd.DataFrame({"subject_id": events["subject_id"], "code": code, "token": token, "role": role,
                        "rule": events["rule"], "age": events["age"], "follow_up": follows,
                        "value": events["numeric_value"].astype("float64").where(role == "measurement")})
    if (role == "text").any():
        out["text"] = events["text_value"].where(role == "text")
    out.loc[out["role"] == "static", "age"] = 0.0
    # from every event the rules matched, before any is dropped below or as a rare token: a thinned
    # measurement or a repeated diagnosis still shows the subject was followed then
    out["follow_up_start"], out["follow_up_end"] = follow_up_window(out, cfg.follow_up_from)
    out = out.loc[out["token"].notna()]  # a grouping can leave a code out of the targets
    out = out.sort_values(["subject_id", "age"], kind="stable")

    for i, rule in enumerate(rules):
        if rule.thin_years and rule.role in ("measurement", "context", "categorical"):
            rows = out["rule"] == i
            bucket = np.floor(out.loc[rows, "age"] / rule.thin_years)
            key = "code" if rule.role == "measurement" else "token"
            dup = out.loc[rows].assign(bucket=bucket).duplicated(["subject_id", key, "bucket"])
            out = out.drop(index=dup[dup].index)
        first = rule.first_occurrence if rule.first_occurrence is not None else rule.role == "target"
        if first:
            rows = out["rule"] == i
            dup = out.loc[rows].duplicated(["subject_id", "token"])
            out = out.drop(index=dup[dup].index)
    return out


def follow_up_window(frame: pd.DataFrame, start_from: str = "first_event") -> tuple[pd.Series, pd.Series]:
    """Per event, its subject's start and end of follow-up. The end is the last event of a rule with
    follow_up: true (the last event at all when there is none), and no later than death. The start is
    the first event (start_from first_event) or the first event of a follow_up rule that is not static
    (first_contact; never, inf, for a subject without one)."""
    subject, age = frame["subject_id"], frame["age"]
    end = age.where(frame["follow_up"]).groupby(subject).transform("max")
    end = end.fillna(age.groupby(subject).transform("max"))
    death = age.where(frame["code"] == meds.death_code).groupby(subject).transform("min")
    if start_from == "first_contact":
        contact = frame["follow_up"] & (frame["role"] != "static")
        start = age.where(contact).groupby(subject).transform("min").fillna(np.inf).astype(age.dtype)
    else:
        start = age.groupby(subject).transform("min")
    return start, np.fmin(end, death)


def format_rules(rules: list[dict]) -> list[str]:
    lines = [f"{'rule':18} {'role':12} {'codes':>8} {'tokens':>8} {'kept':>8} {'events':>12}"]
    for r in rules:
        lines.append(f"{r['rule']:18} {r['role']:12} {r['codes']:8,} {r['tokens']:8,} {r['kept']:8,} {r['events']:12,}")
    return lines


def _shards(ds, cfg: DataConfig, code_rule: pd.Series, split_of: dict, train_only: bool):
    """Yield (events, tokenized frame with a split column) for every shard that has matched events."""
    for shard in ds.shards():
        events = read_events(ds, shard, code_rule)
        if events.empty:
            continue
        frame = tokenize(events, cfg)
        frame["split"] = frame["subject_id"].map(split_of).fillna(meds.train_split)
        if train_only:
            frame = frame.loc[frame["split"] == meds.train_split]
            events = events.loc[events["subject_id"].isin(frame["subject_id"].unique())]
        if len(frame):
            yield events, frame


def meds_fingerprint(meds_dir: str | Path) -> str:
    """Hash of every file's path, size and modification time: changes when the MEDS data is rewritten."""
    root, h = Path(meds_dir), hashlib.sha1()
    for f in sorted(p for p in root.rglob("*") if p.is_file()):
        st = f.stat()
        h.update(f"{f.relative_to(root)}\t{st.st_size}\t{st.st_mtime_ns}\n".encode())
    return h.hexdigest()


def run_prepare(cfg: DataConfig, log=print) -> dict:
    out_dir = Path(cfg.prepared_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if cfg.value_mode not in ("continuous", "binned", "none"):
        raise ValueError(f"value_mode must be continuous, binned or none, not '{cfg.value_mode}'")
    ds = open_meds(cfg.meds_dir)
    code_rule = classify_codes(ds.codes, cfg.events)
    split_of = dict(zip(ds.splits["subject_id"], ds.splits["split"]))
    log(f"[prepare] reading {cfg.meds_dir}: {len(ds.shards())} shards, {len(code_rule):,} codes matched by the rules")
    meta: dict = {"data_config": asdict(cfg), "meds_fingerprint": meds_fingerprint(cfg.meds_dir)}

    # pass 1: training values of every measurement code
    cuts, stats = None, None
    if cfg.value_mode != "none":
        codes, values = [], []
        for _, frame in _shards(ds, cfg, code_rule, split_of, train_only=True):
            rows = (frame["role"] == "measurement") & frame["value"].notna()
            codes.append(frame.loc[rows, "code"].to_numpy())
            values.append(frame.loc[rows, "value"].to_numpy(np.float32))
        codes = pd.Series(np.concatenate(codes) if codes else np.array([], dtype=object))
        values = pd.Series(np.concatenate(values).astype(np.float64) if values else np.array([], dtype=np.float64))
        if cfg.value_mode == "binned":
            cuts = fit_bins(codes, values, cfg.n_bins)
            meta["bins"] = cuts
        else:
            stats = fit_scaler(codes, values)
        log(f"[prepare] pass 1: {len(values):,} training values of {codes.nunique():,} measurements")
        del codes, values

    # pass 2: how many training subjects have each token
    subjects_per_token: dict[str, int] = {}
    train_targets: set[str] = set()
    for _, frame in _shards(ds, cfg, code_rule, split_of, train_only=True):
        if cuts is not None:
            frame = _apply_values(frame, cfg, cuts)
        for token, n in frame.groupby("token")["subject_id"].nunique().items():
            subjects_per_token[token] = subjects_per_token.get(token, 0) + int(n)
        train_targets.update(frame.loc[frame["role"] == "target", "token"].unique())
    kept = {t for t, n in subjects_per_token.items() if n >= cfg.min_subjects_per_token}
    targets = sorted(train_targets & kept)  # a kept token is in the training split, so this is every target
    vocab = Vocabulary.build(sorted(kept), targets)
    if stats is not None:
        stats = {t: v for t, v in stats.items() if t in kept}
        meta["value_stats"] = stats

    # pass 3: every shard, written to its split
    with_text = any(rule.role == "text" for rule in cfg.events)
    writers = {split: SplitWriter(out_dir / split, with_text=with_text) for split in SPLITS}
    if with_text:
        meta["text_buckets"] = cfg.text_buckets
    order = {r: i for i, r in enumerate(ROLES)}
    exposure, target_counts = 0.0, np.zeros(len(targets), dtype=np.int64)
    rule_codes = [set() for _ in cfg.events]
    rule_tokens = [set() for _ in cfg.events]
    rule_events = [0] * len(cfg.events)
    for events, frame in _shards(ds, cfg, code_rule, split_of, train_only=False):
        frame = _apply_values(frame, cfg, cuts)
        for i in range(len(cfg.events)):
            rule_codes[i].update(events.loc[events["rule"] == i, "code"].unique())
            rule_tokens[i].update(frame.loc[frame["rule"] == i, "token"].unique())
        frame = frame.loc[frame["token"].isin(kept)].copy()
        for i in range(len(cfg.events)):
            rule_events[i] += int((frame["rule"] == i).sum())
        frame["token_id"] = frame["token"].map(vocab.token_index).astype(np.int32)
        if stats is not None:
            frame["value"] = scale(frame["token"], frame["value"], stats)
        frame["order"] = frame["role"].map(order)
        frame = frame.sort_values(["subject_id", "age", "order", "token_id"], kind="stable")
        for split in SPLITS:
            part = frame.loc[frame["split"] == split]
            if not len(part):
                continue
            arrays = build_split(part, vocab)
            if with_text:
                arrays.update(build_text(part, cfg.text_buckets, cfg.text_max_words))
            writers[split].append(arrays)
            if split == meds.train_split:
                ages, offs = arrays["ages"], arrays["offsets"]
                exposure += float(np.sum(ages[offs[1:] - 1] - ages[offs[:-1]]))
                is_t = vocab.input_to_target[arrays["tokens"]] >= 0
                target_counts += np.bincount(vocab.input_to_target[arrays["tokens"][is_t]], minlength=len(targets))

    summary = {}
    for split in SPLITS:
        summary[split] = writers[split].close()
        log(f"[prepare] {split}: {summary[split]['subjects']:,} subjects, {summary[split]['events']:,} events")
    meta["exposure_years"] = exposure
    meta["target_counts"] = target_counts.tolist()
    meta["rules"] = [{"rule": rule.name, "role": rule.role, "codes": len(rule_codes[i]), "tokens": len(rule_tokens[i]),
                      "kept": len(rule_tokens[i] & kept), "events": rule_events[i]} for i, rule in enumerate(cfg.events)]
    vocab.save(out_dir / "vocab.json")
    for line in format_rules(meta["rules"]):
        log(f"[prepare] {line}")
    meta["summary"] = summary
    meta["vocab_size"] = len(vocab.tokens)
    meta["n_targets"] = len(vocab.targets)
    (out_dir / "prepare.json").write_text(json.dumps(meta, indent=1))
    log(f"[prepare] vocabulary {len(vocab.tokens):,} tokens, {len(vocab.targets):,} targets → {out_dir}")
    return meta


def _apply_values(frame: pd.DataFrame, cfg: DataConfig, cuts: dict | None) -> pd.DataFrame:
    """Binned: measurement tokens become code//Qn and the number is dropped. None: the number is dropped."""
    if cfg.value_mode == "binned":
        frame = frame.copy()
        rows = frame["role"] == "measurement"
        frame.loc[rows, "token"] = apply_bins(frame.loc[rows, "code"], frame.loc[rows, "value"], cuts or {})
        frame["value"] = np.nan
    elif cfg.value_mode == "none":
        frame = frame.assign(value=np.nan)
    return frame


def is_current(cfg: DataConfig) -> bool:
    """True if prepared_dir holds a complete prepare run made with this exact data config, from the
    MEDS files as they are now (a converter run since then rewrites them)."""
    out_dir = Path(cfg.prepared_dir)
    needed = [out_dir / "prepare.json", out_dir / "vocab.json"] + [out_dir / s / "offsets.npy" for s in SPLITS]
    if not all(p.exists() for p in needed):
        return False
    stored = json.loads((out_dir / "prepare.json").read_text())
    return (stored.get("data_config") == json.loads(json.dumps(asdict(cfg)))
            and stored.get("meds_fingerprint") == meds_fingerprint(cfg.meds_dir))


def describe(cfg: DataConfig, top: int = 10) -> list[str]:
    """A readable summary of a prepared dataset (`modlehr info`)."""
    out_dir = Path(cfg.prepared_dir)
    meta_path = out_dir / "prepare.json"
    if not meta_path.exists():
        return [f"{out_dir} has no prepare.json; run `modlehr prepare` first"]
    meta = json.loads(meta_path.read_text())
    vocab = Vocabulary.load(out_dir / "vocab.json")
    data = meta["data_config"]
    mode = data["value_mode"]
    values = {"binned": f"binned into {data['n_bins']} training-quantile bins (code//Q1..Q{data['n_bins']})",
              "continuous": "kept as numbers, scaled per measurement (training mean and SD, clipped to ±5)",
              "none": "dropped; measurements enter as codes only"}.get(mode, mode)
    lines = [f"prepared data   {out_dir}  (from {data['meds_dir']})",
             f"values          {values}",
             f"target grouping {data['target_grouping']}, tokens kept if seen in >= {data['min_subjects_per_token']} "
             f"training subjects"]
    for split, s in meta["summary"].items():
        lines.append(f"{split:15} {s['subjects']:>9,} subjects {s['events']:>12,} events")
    lines.append(f"vocabulary      {len(vocab.tokens):,} input tokens (with padding), {len(vocab.targets):,} prediction targets; "
                 f"{meta['exposure_years']:,.0f} person-years in training")
    if "rules" in meta:
        lines += ["", *format_rules(meta["rules"])]
    descriptions: dict[str, str] = {}
    if Path(data["meds_dir"]).exists():
        codes = open_meds(data["meds_dir"]).codes
        descriptions = dict(zip(codes["code"].astype(str), codes["description"].fillna("").astype(str)))
    order = np.argsort(meta["target_counts"])[::-1][:top]
    lines += ["", f"most frequent targets in training (events):"]
    for k in order:
        name = vocab.targets[k]
        lines.append(f"  {meta['target_counts'][k]:>8,}  {name:28} {descriptions.get(name, '')}")
    return lines
