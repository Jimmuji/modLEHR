from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
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
        if path.stem.rsplit("_", 1)[-1] in ("xs", "m", "l"):  # the model-size curve sets its own width
            assert variant.model.max_len == 512 and variant.model.d_model != 256, path.name
            continue
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


def test_first_target_ages_within_follow_up():
    """The earliest age of each target per subject, counting only targets up to that subject's end of
    follow-up; inf for targets never seen, and for subjects without events."""
    from modlehr.prepare.timeline import first_target_ages

    vocab = Vocabulary.build(["GENDER//F"], ["D1", "D2"])
    t = vocab.token_index
    tokens = np.array([t["GENDER//F"], t["D2"], t["D1"], t["D2"], t["D1"], t["D2"]], dtype=np.int32)
    ages = np.array([0.0, 30.0, 40.0, 20.0, 50.0, 70.0], dtype=np.float32)
    offsets = np.array([0, 3, 3, 6])  # subject 1 has no events
    inf = np.inf
    np.testing.assert_array_equal(first_target_ages(tokens, ages, offsets, vocab), [[40, 30], [inf, inf], [50, 20]])
    end = np.array([35.0, 0.0, 60.0], dtype=np.float32)
    np.testing.assert_array_equal(first_target_ages(tokens, ages, offsets, vocab, end), [[inf, 30], [inf, inf], [50, 20]])


def test_split_arrays_do_not_depend_on_sharding(tmp_path):
    """SplitWriter joins shards in subject_id order, so the same events give the same arrays however
    the MEDS data is sharded."""
    from conftest import write_meds
    from modlehr.config import DataConfig, EventRule

    rng = np.random.default_rng(0)
    rows = []
    for s in rng.permutation(40):
        rows += [(int(s), None, "GENDER//" + "FM"[s % 2], None), (int(s), 30.0 + s % 7, "DX//A", None),
                 (int(s), 41.0 + s % 5, "DX//B", None)]
        rows += [(int(s), float(a), "LAB//1", float(rng.normal())) for a in range(35, 60, 4)]
    rules = [EventRule(name="sex", role="static", prefix="GENDER//"), EventRule(name="dx", role="target", prefix="DX//"),
             EventRule(name="labs", role="measurement", prefix="LAB//")]
    splits = []
    for shards in (1, 3):
        cfg = DataConfig(meds_dir=str(write_meds(tmp_path / f"meds{shards}", rows, shards=shards)), events=rules,
                         prepared_dir=str(tmp_path / f"prepared{shards}"), min_subjects_per_token=5, value_mode="binned")
        run_prepare(cfg, log=lambda *_: None)
        splits.append(open_split(cfg.prepared_dir, "train"))
    one, three = splits
    assert len(one) == 40 and list(one.subject_ids) == sorted(one.subject_ids)
    for k in ("subject_ids", "offsets", "tokens", "ages", "target", "next_target", "dt", "loss_mask", "follow_up_end"):
        assert np.array_equal(np.asarray(one.arrays[k]), np.asarray(three.arrays[k])), k


def test_splits_written_before_tied_labels_still_load(tiny_config):
    """Without target and follow_up_end, a split gets one label per time and follow-up to the last event."""
    from modlehr.prepare.dataset import collate

    run_prepare(tiny_config.data, log=lambda *_: None)
    path = Path(tiny_config.data.prepared_dir) / "held_out"
    (path / "target.npy").unlink()
    (path / "follow_up_end.npy").unlink()
    split = open_split(tiny_config.data.prepared_dir, "held_out")
    ages = np.asarray(split.arrays["ages"])
    np.testing.assert_array_equal(split.follow_up_end, ages[split.offsets[1:] - 1])
    row = split.subject(0)
    assert "next_set" not in row and "next_set" not in collate([row])


@pytest.mark.parametrize("start_from, start, loss_before", [("first_event", 0.0, True), ("first_contact", 50.0, False)])
def test_follow_up_from_the_first_contact(tmp_path, start_from, start, loss_before):
    """With follow_up_from: first_contact, follow-up starts at the first admission, not at the sex
    recorded at age 0 nor at an outpatient reading (follow_up: false) before it: those events carry no
    loss, and a landmark before the first admission leaves the subject out."""
    from conftest import write_meds
    from modlehr.config import DataConfig, EventRule
    from modlehr.evaluate.task import landmark_task

    rows = []
    for s in range(30):
        rows += [(s, None, "GENDER//F", None), (s, 30.0, "BMI", 25.0), (s, 50.0, "ADM", None),
                 (s, 50.5, "DX//A", None), (s, 58.0, "ADM", None), (s, 58.5, "DX//B", None)]
    cfg = DataConfig(meds_dir=str(write_meds(tmp_path / "meds", rows)), prepared_dir=str(tmp_path / "prepared"),
                     min_subjects_per_token=5, value_mode="none", follow_up_from=start_from,
                     events=[EventRule(name="sex", role="static", prefix="GENDER//"),
                             EventRule(name="dx", role="target", prefix="DX//"),
                             EventRule(name="adm", role="context", codes=["ADM"]),
                             EventRule(name="bmi", role="measurement", codes=["BMI"], follow_up=False)])
    run_prepare(cfg, log=lambda *_: None)
    split = open_split(cfg.prepared_dir, "train")
    vocab = Vocabulary.load(Path(cfg.prepared_dir) / "vocab.json")
    np.testing.assert_allclose(split.follow_up_start, start, atol=1e-4)
    row = split.subject(0)
    assert list(row["loss_mask"][:2]) == [loss_before, loss_before] and row["loss_mask"][2]  # sex, BMI, then ADM
    assert (len(landmark_task(split, vocab, 45.0, 2.0).subjects) == 30) == (start_from == "first_event")
    assert len(landmark_task(split, vocab, 52.0, 2.0).subjects) == 30


def test_washout_moves_the_start_of_follow_up(tmp_path):
    """With data.washout_years, follow-up starts that many years after the first contact: a diagnosis of the
    first stay carries no loss and, at a later landmark, counts as already had rather than as a new case."""
    from conftest import write_meds
    from modlehr.config import DataConfig, EventRule
    from modlehr.evaluate.task import landmark_task, split_first_ages

    rows = []
    for s in range(30):
        rows += [(s, None, "GENDER//F", None), (s, 50.0, "ADM", None), (s, 50.5, "DX//A", None),
                 (s, 58.0, "ADM", None), (s, 58.5, "DX//B", None)]
    cfg = DataConfig(meds_dir=str(write_meds(tmp_path / "meds", rows)), prepared_dir=str(tmp_path / "prepared"),
                     min_subjects_per_token=5, value_mode="none", follow_up_from="first_contact", washout_years=1.0,
                     events=[EventRule(name="sex", role="static", prefix="GENDER//"),
                             EventRule(name="dx", role="target", prefix="DX//"),
                             EventRule(name="adm", role="context", codes=["ADM"])])
    run_prepare(cfg, log=lambda *_: None)
    split = open_split(cfg.prepared_dir, "train")
    vocab = Vocabulary.load(Path(cfg.prepared_dir) / "vocab.json")
    np.testing.assert_allclose(split.follow_up_start, 51.0, atol=1e-4)
    row = split.subject(0)
    assert not row["loss_mask"][:3].any() and row["loss_mask"][3]  # sex, ADM, DX//A; then the second ADM
    assert len(landmark_task(split, vocab, 50.8, 2.0).subjects) == 0  # still in the washout
    task = landmark_task(split, vocab, 52.0, 10.0, split_first_ages(split, vocab))
    keep, case = task.labels(vocab.targets.index("DX//A"))
    assert len(task.subjects) == 30 and not keep.any()  # had A before the landmark
    keep, case = task.labels(vocab.targets.index("DX//B"))
    assert keep.all() and case.all()


def test_data_prepared_before_an_option_existed_is_still_current(tiny_config):
    """prepare.json written before data.washout_years existed matches a config that leaves it at 0."""
    from modlehr.prepare.pipeline import is_current
    run_prepare(tiny_config.data, log=lambda *_: None)
    meta_path = Path(tiny_config.data.prepared_dir) / "prepare.json"
    meta = json.loads(meta_path.read_text())
    del meta["data_config"]["washout_years"]
    meta_path.write_text(json.dumps(meta))
    assert is_current(tiny_config.data)
    assert not is_current(replace(tiny_config.data, washout_years=1.0))


def test_no_event_tokens_are_labelled_from_their_own_age(tmp_path):
    """A "no event" token at age a gets every target of the next target time after a and the years
    until it (or until the end of follow-up), carries a loss, and leaves the real events as they were."""
    from modlehr.prepare.dataset import SplitWriter, with_no_events
    from modlehr.prepare.timeline import next_target_sets

    vocab = Vocabulary.build(["GENDER//F", "ADM", "NO_EVENT"], ["D1", "D2", "D3"])
    t = vocab.token_index
    events = [("GENDER//F", 0.0), ("ADM", 40.0), ("D1", 40.5), ("D2", 40.5), ("ADM", 47.0), ("D3", 47.2), ("ADM", 60.0)]
    frame = pd.DataFrame({"subject_id": 0, "token_id": [t[e] for e, _ in events], "age": [a for _, a in events],
                          "value": np.nan})
    writer = SplitWriter(tmp_path / "split")
    writer.append(build_split(frame, vocab))
    writer.close()
    split = open_split(tmp_path, "split")
    row = split.subject(0)
    out = with_no_events(row, t["NO_EVENT"], 2.0, np.random.default_rng(0), float(split.follow_up_start[0]),
                         float(split.follow_up_end[0]), max_len=64)
    added = out["tokens"] == t["NO_EVENT"]
    assert added.sum() > 10 and np.all(np.diff(out["ages"]) >= 0)
    assert np.array_equal(out["tokens"][~added], row["tokens"]) and np.array_equal(out["dt"][~added], row["dt"])
    a = out["ages"][added]
    assert np.all((a >= 0.0) & (a <= 60.0)) and out["loss_mask"][added].all()
    full = np.asarray(split.arrays["ages"]), np.asarray(split.arrays["target"])
    for age, labels, dt in zip(a, out["next_set"][added], out["dt"][added]):
        expected = next_target_sets(np.r_[full[0], age], np.r_[full[1], -1], 60.0, len(full[0]), len(full[0]) + 1)[0]
        assert set(labels[labels >= 0].tolist()) == set(expected[expected >= 0].tolist())
        later = full[0][(full[1] >= 0) & (full[0] > age)]
        assert dt == pytest.approx((later.min() if len(later) else 60.0) - age, abs=1e-4)
    short = with_no_events(row, t["NO_EVENT"], 0.5, np.random.default_rng(1), 0.0, 60.0, max_len=10)
    assert len(short["tokens"]) == 10 and short["ages"][-1] == 60.0  # the last max_len events are kept


def test_training_with_no_event_tokens(tiny_config):
    """With data.no_event_years the vocabulary holds the token, training inserts it, and evaluation and
    sampling ask at one placed at the landmark or start age."""
    import json

    from modlehr.evaluate.run import evaluate, load_model
    from modlehr.generate.sample import sample_future
    from modlehr.prepare.vocabulary import NO_EVENT
    from modlehr.train.loop import train

    run_prepare(tiny_config.data, log=lambda *_: None)
    tiny_config.data.no_event_years = 5.0
    with pytest.raises(ValueError, match="prepare"):  # prepared without the token
        train(tiny_config, log=lambda *_: None)
    run_prepare(tiny_config.data, log=lambda *_: None)
    prepared = Path(tiny_config.data.prepared_dir)
    vocab = Vocabulary.load(prepared / "vocab.json")
    assert NO_EVENT in vocab.tokens and NO_EVENT not in vocab.targets
    best = train(tiny_config, log=lambda *_: None)
    model, _ = load_model(best, prepared, "cpu")
    assert model.query_token == vocab.token_index[NO_EVENT]
    evaluate(tiny_config, best, log=lambda *_: None)
    assert json.loads((Path(tiny_config.train.out_dir) / "evaluate.json").read_text())["pairs"] > 0
    history = open_split(prepared, "held_out").subject(0)
    events = sample_future(model, vocab, history, start_age=50.0, horizon_years=30.0, seed=1)
    assert all(50.0 < age <= 80.0 and token in vocab.targets for age, token in events)
