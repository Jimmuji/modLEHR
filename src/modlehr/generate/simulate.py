"""`modlehr simulate`: sample one future per patient and compare it with what the records show.

For held-out patients whose record covers a start age A, the model samples a future from A, one
outcome at a time. Each sampled future is cut where that patient's real record ends: a real record
stops when follow-up stops, a sampled one would run on to death or the age limit, and comparing
the two without matching the window makes the model look several times too active.

Both sides are then counted the same way, as outcome events per 1,000 person-years in 5-year age
bands. Person-years run from A to the end of the record, or to death when death comes first (in
the record, or in the sample). Targets are first occurrences, so a target already in the history
at A is neither sampled nor counted.

Writes simulate.csv (by age band), simulate_targets.csv (events per target) and simulate.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from modlehr.config import Config
from modlehr.generate.sample import sample_future
from modlehr.prepare.dataset import open_split
from modlehr.train.loop import pick_device

BAND = 5.0


def _person_years(start: float, stop: float, bands: np.ndarray) -> np.ndarray:
    """Years spent in each band [b, b + BAND) between start and stop."""
    return np.clip(np.minimum(stop, bands + BAND) - np.maximum(start, bands), 0.0, None)


def simulate(cfg: Config, checkpoint: str | Path, start_age: float, n_subjects: int = 1000,
             out_dir: str | Path | None = None, seed: int = 0, log=print) -> pd.DataFrame:
    from modlehr.evaluate.run import load_model

    prepared = Path(cfg.data.prepared_dir)
    device = pick_device(cfg.train.device)
    model, vocab = load_model(checkpoint, prepared, device)
    split = open_split(prepared, cfg.evaluate.split)
    first_age, last_age = split.follow_up_start, split.follow_up_end  # follow-up, not the record
    eligible = np.flatnonzero((first_age <= start_age) & (last_age > start_age))
    if not len(eligible):
        raise ValueError(f"no {cfg.evaluate.split} patient is followed at age {start_age:g}")
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(eligible, min(n_subjects, len(eligible)), replace=False))
    death = vocab.token_index.get("MEDS_DEATH")

    bands = np.arange(np.floor(start_age / BAND) * BAND, 120.0, BAND)
    tally = {key: np.zeros(len(bands)) for key in ("obs_events", "obs_py", "sim_events", "sim_py")}
    obs_by_target, sim_by_target = np.zeros(len(vocab.targets), int), np.zeros(len(vocab.targets), int)
    band_of = lambda age: min(int((age - bands[0]) // BAND), len(bands) - 1)  # noqa: E731

    log(f"[simulate] {len(chosen):,} of {len(eligible):,} {cfg.evaluate.split} patients followed at age {start_age:g}")
    for j, i in enumerate(chosen):
        h = split.subject(int(i))
        end = float(last_age[i])
        target = vocab.input_to_target[h["tokens"]]
        later = (target >= 0) & (h["ages"] > start_age) & (h["ages"] <= end)
        for age, k in zip(h["ages"][later], target[later]):
            tally["obs_events"][band_of(age)] += 1
            obs_by_target[k] += 1
        tally["obs_py"] += _person_years(start_age, end, bands)

        sampled = sample_future(model, vocab, h, start_age, end - start_age, seed=seed + j, device=device)
        stop = end
        for age, token in sampled:
            k = int(vocab.input_to_target[vocab.token_index[token]])
            tally["sim_events"][band_of(age)] += 1
            sim_by_target[k] += 1
            if vocab.token_index[token] == death:
                stop = age
        tally["sim_py"] += _person_years(start_age, stop, bands)
        if (j + 1) % 250 == 0:
            log(f"[simulate] {j + 1:,} patients sampled")

    used = tally["obs_py"] > 0
    table = pd.DataFrame({
        "age_band": [f"{b:g}-{b + BAND:g}" for b in bands[used]],
        "observed_events": tally["obs_events"][used].astype(int), "observed_person_years": tally["obs_py"][used],
        "sampled_events": tally["sim_events"][used].astype(int), "sampled_person_years": tally["sim_py"][used],
    })
    table["observed_per_1000"] = 1000 * table["observed_events"] / table["observed_person_years"]
    table["sampled_per_1000"] = 1000 * table["sampled_events"] / table["sampled_person_years"].clip(lower=1e-9)
    targets = pd.DataFrame({"target": vocab.targets, "observed": obs_by_target, "sampled": sim_by_target})
    targets = targets[(targets["observed"] + targets["sampled"]) > 0].sort_values("observed", ascending=False)

    out = Path(out_dir) if out_dir else Path(cfg.train.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(out / "simulate.csv", index=False)
    targets.to_csv(out / "simulate_targets.csv", index=False)
    obs_rate = tally["obs_events"].sum() / max(tally["obs_py"].sum(), 1e-9)
    sim_rate = tally["sim_events"].sum() / max(tally["sim_py"].sum(), 1e-9)
    summary = {"start_age": start_age, "subjects": int(len(chosen)), "observed_events": int(tally["obs_events"].sum()),
               "sampled_events": int(tally["sim_events"].sum()), "observed_person_years": float(tally["obs_py"].sum()),
               "sampled_person_years": float(tally["sim_py"].sum()), "observed_per_1000": 1000 * obs_rate,
               "sampled_per_1000": 1000 * sim_rate, "sampled_over_observed": sim_rate / max(obs_rate, 1e-12)}
    (out / "simulate.json").write_text(json.dumps(summary, indent=1))
    log(f"[simulate] outcomes per 1,000 person-years: observed {summary['observed_per_1000']:.1f}, "
        f"sampled {summary['sampled_per_1000']:.1f} (ratio {summary['sampled_over_observed']:.2f}) → {out / 'simulate.csv'}")
    return table
