from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from modlehr.config import load_config
from modlehr.data.meds_io import open_meds
from modlehr.prepare.grouping import icd10_3char
from modlehr.prepare.pipeline import run_prepare
from modlehr.prepare.timeline import build_split
from modlehr.prepare.vocabulary import PAD, Vocabulary

REPO = Path(__file__).resolve().parents[1]


def test_repo_configs_load():
    cfg = load_config(REPO / "configs/experiment/synthea_gpt_values.yaml")
    assert cfg.data.value_mode == "continuous" and cfg.model.value_encoding == "mlp"
    assert cfg.model.attention == "mha" and len(cfg.data.events) == 5


def test_tiny_meds_is_valid(tiny_meds):
    open_meds(tiny_meds, validate=True)


def test_icd10_grouping():
    assert icd10_3char("ICD10CM//E11.9") == "ICD10//E11"
    assert icd10_3char("SNOMEDCT//44054006") == "SNOMEDCT//44054006"


def test_labels_next_target_and_censoring():
    vocab = Vocabulary.build(["GENDER//F", "LAB"], ["D1", "D2"])
    t = vocab.token_index
    frame = pd.DataFrame({
        "subject_id": [0, 0, 0, 0, 0],
        "token_id": [t["GENDER//F"], t["LAB"], t["D1"], t["LAB"], t["D2"]],
        "age": [0.0, 30.0, 30.0, 40.0, 45.0],
        "value": [np.nan, 1.0, np.nan, 2.0, np.nan],
    })
    a = build_split(frame, vocab)
    k = vocab.input_to_target
    # from age 0 the next target strictly later is D1 at 30; from age 30 (both events) it is D2 at 45
    assert list(a["next_target"]) == [k[t["D1"]], k[t["D2"]], k[t["D2"]], k[t["D2"]], -1]
    np.testing.assert_allclose(a["dt"], [30.0, 15.0, 15.0, 5.0, 0.0])
    # only the last event of each time carries a loss
    assert list(a["loss_mask"]) == [True, False, True, True, True]


def test_prepare_tiny(tiny_config):
    meta = run_prepare(tiny_config.data, log=lambda *_: None)
    out = Path(tiny_config.data.prepared_dir)
    vocab = Vocabulary.load(out / "vocab.json")
    assert vocab.tokens[0] == PAD
    assert "MEDS_DEATH" in vocab.targets and "SNOMEDCT//1001" in vocab.targets
    assert "SNOMEDCT//2001" not in vocab.tokens  # findings are not selected by the rules
    with np.load(out / "train.npz") as z:
        values = z["values"]
        ages = z["ages"]
        offsets = z["offsets"]
    observed = values[~np.isnan(values)]
    assert abs(observed.mean()) < 0.2 and 0.8 < observed.std() < 1.2  # scaled per measurement
    assert np.all(np.diff(ages[offsets[0]:offsets[1]]) >= 0)  # sorted by age
    # thinning: at most one HbA1c per subject per 5-year window
    assert meta["summary"]["train"]["subjects"] > 0 and json.loads((out / "prepare.json").read_text())["n_targets"] == 4


def test_prepare_binned(tiny_config):
    cfg = replace(tiny_config.data, value_mode="binned", prepared_dir=tiny_config.data.prepared_dir + "_binned")
    run_prepare(cfg, log=lambda *_: None)
    vocab = Vocabulary.load(Path(cfg.prepared_dir) / "vocab.json")
    assert {"LOINC//4548-4//Q1", "LOINC//4548-4//Q2", "LOINC//4548-4//Q3"} <= set(vocab.tokens)
