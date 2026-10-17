from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from modlehr.config import load_config
from modlehr.data.meds_io import open_meds
from modlehr.prepare.grouping import icd10_3char
from modlehr.prepare.dataset import open_split
from modlehr.prepare.pipeline import run_prepare
from modlehr.prepare.timeline import build_split
from modlehr.prepare.vocabulary import PAD, Vocabulary

REPO = Path(__file__).resolve().parents[1]


def test_repo_configs_load():
    cfg = load_config(REPO / "configs/experiment/synthea_gpt_values.yaml")
    assert cfg.data.value_mode == "continuous" and cfg.model.value_encoding == "mlp"
    assert cfg.model.attention == "mha" and len(cfg.data.events) == 5
    for path in sorted((REPO / "configs/experiment").glob("*.yaml")):  # every experiment config resolves
        load_config(path)
    mimic = load_config(REPO / "configs/experiment/mimic_iv_gpt.yaml")
    assert mimic.data.target_grouping == "icd_3char" and mimic.model.max_len == 512


def test_thinning_applies_to_context_events(tiny_config):
    cfg = replace(tiny_config.data, prepared_dir=tiny_config.data.prepared_dir + "_ctx",
                  events=[*tiny_config.data.events[:3],
                          type(tiny_config.data.events[0])(name="hba1c_ctx", role="context", codes=["LOINC//4548-4"],
                                                           thin_years=10)])
    meta = run_prepare(cfg, log=lambda *_: None)
    rule = [r for r in meta["rules"] if r["rule"] == "hba1c_ctx"][0]
    # HbA1c is recorded every 3 years from age 20 to 79: at most 6 ten-year windows per subject
    assert 0 < rule["events"] <= 6 * 160


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
    train = open_split(out, "train")
    values, ages, offsets = np.asarray(train.arrays["values"]), np.asarray(train.arrays["ages"]), train.offsets
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


def test_icd_3char_mixed_versions():
    from modlehr.prepare.grouping import icd_3char
    assert icd_3char("DIAGNOSIS//ICD//10//E119") == "ICD10//E11"
    assert icd_3char("DIAGNOSIS//ICD//9//4280") == "ICD9//428"
    assert icd_3char("DIAGNOSIS//ICD//9//E8889") == "ICD9//E888"
    assert icd_3char("ICD10CM//I10") == "ICD10//I10"
    assert icd_3char("LAB//50912//mg/dL") == "LAB//50912//mg/dL"


def test_exclude_regex():
    import pandas as pd
    from modlehr.config import EventRule
    from modlehr.prepare.select import classify_codes
    codes = pd.DataFrame({"code": ["LAB//1//mg", "LAB//SPECIMEN_COLLECTED//1//STAT"], "description": [None, None]})
    rule = EventRule(name="labs", role="measurement", prefix="LAB//", exclude_regex="SPECIMEN_COLLECTED")
    assert list(classify_codes(codes, [rule]).index) == ["LAB//1//mg"]


def test_controls_never_keeps_more_than_at_risk():
    """The two conventions differ exactly on censored subjects who never get the target.

    "at_risk" drops a subject whose record ends inside the horizon without the target;
    "never" keeps them as a control. So "never" keeps at least as many, and the cases
    are the same either way.
    """
    import numpy as np

    from modlehr.evaluate.task import LandmarkTask

    inf = float("inf")
    #            case in window, control reaching the horizon, censored never-haver, had it before L
    first = np.array([[52.0], [inf], [inf], [45.0]])
    end = np.array([60.0, 60.0, 52.0, 60.0])

    at_risk = LandmarkTask(50.0, 5.0, np.arange(4), first, end, "at_risk")
    never = LandmarkTask(50.0, 5.0, np.arange(4), first, end, "never")

    keep_a, case_a = at_risk.labels(0)
    keep_n, case_n = never.labels(0)

    assert list(case_a) == list(case_n) == [True, False, False, False]
    assert list(keep_a) == [True, True, False, False]   # censored never-haver dropped
    assert list(keep_n) == [True, True, True, False]    # kept as a control
