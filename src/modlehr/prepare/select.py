"""Select events from MEDS by rules, and compute each event's age.

MEDS stores datetimes; the model works in age. Age is event time minus the subject's
MEDS_BIRTH, in years. Static rows (null time) sit at age 0. Subjects without a birth event
are dropped.
"""

from __future__ import annotations

import meds
import numpy as np
import pandas as pd

from modlehr.config import EventRule
from modlehr.data.meds_io import MedsDataset

ROLES = ("static", "measurement", "categorical", "context", "target")
SECONDS_PER_YEAR = 365.25 * 86400


def classify_codes(codes: pd.DataFrame, rules: list[EventRule]) -> pd.Series:
    """MEDS code → index of the first rule that matches it. Unmatched codes are left out."""
    for rule in rules:
        if rule.role not in ROLES:
            raise ValueError(f"rule '{rule.name}': unknown role '{rule.role}', expected one of {ROLES}")
    code = codes["code"].astype(str)
    desc = codes["description"].fillna("").astype(str)
    assigned = np.full(len(codes), -1)
    for i, rule in enumerate(rules):
        match = np.ones(len(codes), dtype=bool)
        if rule.prefix:
            match &= code.str.startswith(rule.prefix).to_numpy()
        if rule.codes is not None:
            match &= code.isin(rule.codes).to_numpy()
        if rule.description_regex:
            match &= desc.str.contains(rule.description_regex, regex=True).to_numpy()
        if rule.exclude_regex:
            match &= ~code.str.contains(rule.exclude_regex, regex=True).to_numpy()
        assigned[match & (assigned == -1)] = i
    keep = assigned >= 0
    return pd.Series(assigned[keep], index=code[keep].to_numpy())


def load_events(ds: MedsDataset, rules: list[EventRule]) -> pd.DataFrame:
    """Matched events from every shard, with columns subject_id, code, age, numeric_value,
    text_value and rule (index into rules)."""
    code_rule = classify_codes(ds.codes, rules)
    wanted = set(code_rule.index) | {meds.birth_code}
    frames = [ds.read_shard(shard, codes=wanted) for shard in ds.shards()]
    events = pd.concat(frames, ignore_index=True)

    birth = events.loc[events["code"] == meds.birth_code].groupby("subject_id")["time"].min()
    events = events.loc[(events["code"] != meds.birth_code) & events["subject_id"].isin(birth.index)].copy()
    age = (events["time"] - events["subject_id"].map(birth)).dt.total_seconds() / SECONDS_PER_YEAR
    events["age"] = age.fillna(0.0).clip(lower=0.0).astype("float32")
    events["rule"] = events["code"].map(code_rule).astype(int)
    return events.drop(columns="time")
