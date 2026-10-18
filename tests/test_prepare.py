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
    assert mimic.data.target_grouping == "icd_gems" and mimic.model.max_len == 512
    for path in sorted((REPO / "configs/experiment").glob("mimic_iv_core_gpt_*.yaml")):
        variant = load_config(path)  # a model file layered on top must not reset the size
        assert (variant.model.d_model, variant.model.n_heads, variant.model.max_len) == (256, 8, 512), path.name


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



def test_tied_targets_all_label_their_time_and_follow_up_ends_at_death():
    """Diagnoses recorded at one discharge are all labels of that time (not only the first), death
    counts even when tied with diagnoses, follow-up ends at death, and events of a follow_up: false rule
    (outpatient BMI) neither extend follow-up nor carry a loss after it."""
    from modlehr.prepare.timeline import next_target_sets

    vocab = Vocabulary.build(["GENDER//F", "BMI", "ADM"], ["D1", "D2", "D3", "MEDS_DEATH"])
    t, k = vocab.token_index, vocab.input_to_target
    rows = [("GENDER//F", 0.0, True), ("BMI", 40.0, False), ("D1", 50.0, True), ("D2", 50.0, True),
            ("ADM", 50.0, True), ("D3", 60.0, True), ("MEDS_DEATH", 60.0, True), ("ADM", 61.0, True),
            ("BMI", 62.0, False)]
    frame = pd.DataFrame({"subject_id": 0, "token_id": [t[r[0]] for r in rows], "age": [r[1] for r in rows],
                          "value": np.nan, "follow_up": [r[2] for r in rows]})
    a = build_split(frame, vocab)
    assert a["follow_up_end"][0] == 60.0  # the admission at 61 comes after death
    sets = next_target_sets(a["ages"], a["target"], 60.0, 0, len(rows))
    as_set = lambda r: set(r[r >= 0].tolist())  # noqa: E731
    assert as_set(sets[0]) == {k[t["D1"]], k[t["D2"]]}  # from age 0: both diagnoses at 50
    assert as_set(sets[4]) == {k[t["D3"]], k[t["MEDS_DEATH"]]}  # from 50: the diagnosis and death at 60
    assert as_set(sets[6]) == set() and as_set(sets[8]) == set()  # after the last target: censored
    assert list(a["loss_mask"]) == [True, True, False, False, True, False, True, False, False]


def test_outpatient_events_do_not_turn_censored_subjects_into_controls(tmp_path):
    """At landmark 50 with a 2-year horizon, a subject last admitted at 51 is censored, also when an
    outpatient BMI at 58 follows: BMI is an input, not follow-up."""
    from modlehr.evaluate.task import landmark_task
    from modlehr.prepare.dataset import SplitWriter

    vocab = Vocabulary.build(["GENDER//F", "BMI", "ADM"], ["D1"])
    t = vocab.token_index
    for follows, expect_control in ((False, False), (True, True)):
        frame = pd.DataFrame({"subject_id": 0, "token_id": [t["GENDER//F"], t["ADM"], t["BMI"]],
                              "age": [0.0, 51.0, 58.0], "value": np.nan, "follow_up": [True, True, follows]})
        writer = SplitWriter(tmp_path / f"split_{follows}")
        writer.append(build_split(frame, vocab))
        writer.close()
        task = landmark_task(open_split(tmp_path, f"split_{follows}"), vocab, 50.0, 2.0)
        keep, case = task.labels(0)
        assert bool(keep[0]) == expect_control and not case[0]


def test_loss_counts_every_tied_target():
    import torch

    from modlehr.model.heads import CompetingRiskHead

    log_rates = torch.log(torch.tensor([[[0.1, 0.2, 0.3]]]))
    dt = torch.tensor([[2.0]])
    mask = torch.tensor([[True]])
    nll, _ = CompetingRiskHead.loss(log_rates, torch.tensor([[[0, 2, -1]]]), dt, mask)
    expected = -(np.log(0.1) + np.log(0.3)) + 0.6 * 2.0
    assert abs(float(nll) - expected) < 1e-5
    single, _ = CompetingRiskHead.loss(log_rates, torch.tensor([[1]]), dt, mask)  # one target, as before
    assert abs(float(single) - (-np.log(0.2) + 1.2)) < 1e-5


def test_follow_up_counts_events_that_prepare_drops(tmp_path):
    """A contact shows the subject was followed then, even when prepare thins its measurement away,
    drops its code as rare, or drops a diagnosis already recorded earlier (first occurrence)."""
    from conftest import write_meds
    from modlehr.config import DataConfig, EventRule

    rows = []
    for s in range(30):
        rows += [(s, None, "GENDER//F", None), (s, 40.0, "DX//A", None), (s, 50.5, "LAB//1", 1.0),
                 (s, 51.5, "LAB//1", 2.0)]  # one lab per 5 years is kept: 51.5 is thinned away
    rows += [(0, 53.0, "LAB//RARE", 3.0),  # a contact whose only code is too rare to keep
             (1, 54.0, "DX//A", None),     # a repeated diagnosis
             (2, 55.0, "MEDS_DEATH", None), (2, 56.0, "LAB//RARE", 1.0)]  # nothing counts after death
    cfg = DataConfig(meds_dir=str(write_meds(tmp_path / "meds", rows)), prepared_dir=str(tmp_path / "prepared"),
                     min_subjects_per_token=5, value_mode="none",
                     events=[EventRule(name="sex", role="static", prefix="GENDER//"),
                             EventRule(name="death", role="target", codes=["MEDS_DEATH"]),
                             EventRule(name="dx", role="target", prefix="DX//"),
                             EventRule(name="labs", role="measurement", prefix="LAB//", thin_years=5)])
    run_prepare(cfg, log=lambda *_: None)
    end = open_split(cfg.prepared_dir, "train").follow_up_end
    np.testing.assert_allclose(end[:4], [53.0, 54.0, 55.0, 51.5], atol=1e-4)


def test_shards_without_birth_events_are_skipped(tmp_path):
    """Subjects without MEDS_BIRTH are dropped, also when a whole shard has none, or no rows at all."""
    import meds
    import pyarrow as pa
    import pyarrow.parquet as pq
    from conftest import write_meds
    from modlehr.config import DataConfig, EventRule

    root = write_meds(tmp_path / "meds", [(s, 50.0, "DX//A", None) for s in range(5)])
    no_birth = pd.DataFrame({"subject_id": [99], "time": [pd.Timestamp("2000-01-01")], "code": ["DX//A"],
                             "numeric_value": [None], "text_value": [None]})
    for name, df in (("1", no_birth), ("2", no_birth.iloc[:0])):
        df = df.astype({"time": "datetime64[us]"})
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False).cast(meds.DataSchema.schema()),
                       root / "data" / "train" / f"{name}.parquet")
    cfg = DataConfig(meds_dir=str(root), prepared_dir=str(tmp_path / "prepared"), min_subjects_per_token=1,
                     value_mode="none", events=[EventRule(name="dx", role="target", prefix="DX//")])
    meta = run_prepare(cfg, log=lambda *_: None)
    assert meta["summary"]["train"]["subjects"] == 5


def test_inputs_without_label_sets_are_the_same_inputs(tiny_config):
    """Evaluation reads histories with labels=False, which skips building the tied-target sets."""
    from modlehr.evaluate.task import history_before

    run_prepare(tiny_config.data, log=lambda *_: None)
    split = open_split(tiny_config.data.prepared_dir, "held_out")
    for i in range(len(split)):
        full, inputs = split.subject(i, 2, 9), split.subject(i, 2, 9, labels=False)
        assert "next_set" in full and "next_set" not in inputs
        assert all(np.array_equal(full[k], inputs[k], equal_nan=True) for k in inputs)
    assert "next_set" not in history_before(split, 0, 60.0)
