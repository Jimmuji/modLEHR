"""Groupings that read mapping tables: ICD-9 to ICD-10 through the GEMs, and phecodes."""

from __future__ import annotations

import copy

import pytest

from modlehr.prepare.grouping import ICDWithGEMs, Phecode, load_gems, make_grouping
from modlehr.prepare.pipeline import run_prepare
from modlehr.prepare.vocabulary import Vocabulary

# The CMS file layout: ICD-9 code, ICD-10 code, five flags (approximate, no map, combination,
# scenario, choice list).
GEMS = """4280  I509     10000
42731 I480     10000
42731 I4891    10000
42731 I4819    10000
25000 E119     10000
V5861 Z7901    10000
7999  NoDx     11000
"""


@pytest.fixture
def gems_path(tmp_path):
    path = tmp_path / "2018_I9gem.txt"
    path.write_text(GEMS)
    return str(path)


def test_gems_majority_and_no_map(gems_path):
    table = load_gems(gems_path)
    assert table["4280"] == "I50" and table["25000"] == "E11"
    assert table["42731"] == "I48"  # I480, I4891, I4819 all share I48
    assert "7999" not in table  # flagged no map


def test_icd_gems_puts_both_eras_in_one_set(gems_path):
    group = ICDWithGEMs(gems_path)
    assert group("DIAGNOSIS//ICD//9//4280") == group("DIAGNOSIS//ICD//10//I5022") == "ICD10//I50"
    assert group("DIAGNOSIS//ICD//9//7999") == "ICD9//799"  # unmapped ICD-9 keeps its category
    assert group("MEDS_DEATH") == "MEDS_DEATH"


def test_phecode_prefix_fallback_and_unmapped(tmp_path):
    icd9 = tmp_path / "phecode_icd9_rolled.csv"
    icd9.write_text("ICD9,ICD9 String,PheCode,Phenotype\n250.00,Diabetes,250.2,Type 2 diabetes\n428,Heart failure,428,Heart failure\n")
    icd10 = tmp_path / "Phecode_map_v1_2_icd10cm_beta.csv"
    icd10.write_text("icd10cm,icd10cm_str,phecode,phecode_str\nE11.9,T2D,250.2,Type 2 diabetes\nI50,HF,428,Heart failure\n")
    keep = Phecode(str(icd9), str(icd10))
    assert keep("DIAGNOSIS//ICD//9//25000") == keep("DIAGNOSIS//ICD//10//E119") == "PHECODE//250.2"
    assert keep("DIAGNOSIS//ICD//9//4280") == keep("DIAGNOSIS//ICD//10//I5022") == "PHECODE//428"  # by prefix
    assert keep("DIAGNOSIS//ICD//10//Z999") == "ICD10//Z99"
    assert Phecode(str(icd9), str(icd10), unmapped="drop")("DIAGNOSIS//ICD//10//Z999") is None


def test_grouping_options_are_checked(gems_path):
    with pytest.raises(ValueError):
        make_grouping("identity", {"gems_path": gems_path})
    with pytest.raises(FileNotFoundError):
        make_grouping("icd_gems", {"gems_path": "/no/such/file.txt"})


def test_prepare_drops_codes_a_grouping_leaves_out(tiny_config, tmp_path):
    cfg = copy.deepcopy(tiny_config)
    cfg.data.prepared_dir = str(tmp_path / "prepared_drop")
    # a grouping that drops Disease C: it must vanish from targets and from the inputs
    from modlehr.registry import register

    @register("target_grouping", "test_drop_1003")
    def drop_c(code: str):
        return None if code == "SNOMEDCT//1003" else code

    cfg.data.target_grouping = "test_drop_1003"
    run_prepare(cfg.data, log=lambda *_: None)
    vocab = Vocabulary.load(tmp_path / "prepared_drop" / "vocab.json")
    assert "SNOMEDCT//1003" not in vocab.tokens and "SNOMEDCT//1001" in vocab.targets
