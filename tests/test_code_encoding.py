from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from modlehr.config import ModelConfig, check
from modlehr.evaluate.run import load_model
from modlehr.model.code_encodings import groups_of, principal_components, token_ancestors
from modlehr.model.model import ModLEHR
from modlehr.prepare.pipeline import run_prepare
from modlehr.prepare.token_text import describe_tokens, read_vectors, token_texts, write_vectors
from modlehr.prepare.vocabulary import NO_EVENT, PAD, Vocabulary
from modlehr.train.loop import train

QUIET = lambda *a, **k: None  # noqa: E731


def test_each_token_gets_a_text():
    descriptions = {"DIAGNOSIS//ICD//10//E11": "Type 2 diabetes mellitus", "DIAGNOSIS//ICD//9//4280": "Congestive heart failure",
                    "LAB//50912": "Creatinine", "SMOKING": "Smoking status", "GENDER//F": "Sex: female", "X": None}
    tokens = [PAD, NO_EVENT, "GENDER//F", "ICD10//E11", "ICD9//428", "ICD9//999", "LAB//50912//Q2", "SMOKING//Never",
              "MEDICATION//START//ASPIRIN"]
    texts, sources = describe_tokens(tokens, descriptions, n_bins=3)
    assert texts == ["", "No event", "Sex: female", "Type 2 diabetes mellitus", "Congestive heart failure", "ICD9: 999",
                     "Creatinine, level 2 of 3", "Smoking status: Never", "Medication: START ASPIRIN"]
    assert sources == ["pad", "exact", "exact", "category", "member", "name", "value", "value", "name"]


def test_groups_join_related_codes():
    assert set(groups_of("ICD10//I21")) & set(groups_of("ICD9//410")) == {"CHAPTER//9"}  # circulatory, both versions
    assert "ICD10//I2*" in set(groups_of("ICD10//I21")) & set(groups_of("ICD10//I25"))
    assert "LAB//50912" in groups_of("LAB//50912//Q3")
    assert "PROCEDURE//ICD//10//0DT*" in groups_of("PROCEDURE//ICD//10//0DTJ4ZZ")
    assert "MEDICATION//START//INSULIN*" in groups_of("MEDICATION//START//INSULIN GLARGINE")
    tokens = [PAD, "ICD10//I21", "ICD10//I25", "ICD9//410", "GENDER//F"]
    ancestors = token_ancestors(tokens).numpy()
    assert (ancestors[0] == 0).all() and (ancestors[4] == 0).all()  # padding; GENDER has no group of two
    assert set(ancestors[1]) & set(ancestors[3]) - {0}  # a shared chapter


def test_principal_components_start_at_the_default_spread():
    vectors = torch.randn(50, 40)
    vectors[0] = 0
    start = principal_components(vectors, 16)
    assert start.shape == (50, 16) and (start[0] == 0).all() and abs(float(start[1:].std()) - 0.02) < 1e-6
    assert principal_components(vectors, 64)[:, 49:].abs().max() == 0  # more dims than components: zeros


def test_learned_is_the_plain_embedding():
    cfg = ModelConfig(d_model=16, n_layers=1, n_heads=2)
    model = ModLEHR(cfg, n_tokens=10, n_targets=3)
    assert isinstance(model.input_head.embed, torch.nn.Embedding)
    assert [k for k in model.state_dict() if k.startswith("input_head.embed")] == ["input_head.embed.weight"]


def test_text_encodings_need_vectors(tiny_config):
    run_prepare(tiny_config.data, log=QUIET)
    cfg = dataclasses.replace(tiny_config, model=dataclasses.replace(tiny_config.model, code_encoding="text_sum"))
    with pytest.raises(ValueError, match="code_vectors"):
        train(check(cfg), log=QUIET)


def _vectors(cfg, tmp_path: Path) -> Path:
    prepared = Path(cfg.data.prepared_dir)
    table = token_texts(prepared, cfg.data.meds_dir)
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(len(table), 24)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors[0] = 0
    return write_vectors(tmp_path / "vectors.npz", table, vectors, "random", "none")


@pytest.mark.parametrize("encoding", ["text_frozen", "text_init", "text_sum", "hierarchy"])
def test_code_encodings_train_and_reload(tiny_config, tmp_path, encoding):
    run_prepare(tiny_config.data, log=QUIET)
    path = _vectors(tiny_config, tmp_path)
    model_cfg = dataclasses.replace(tiny_config.model, code_encoding=encoding,
                                    code_vectors=str(path) if encoding.startswith("text") else None)
    cfg = check(dataclasses.replace(tiny_config, model=model_cfg))
    checkpoint = train(cfg, log=QUIET)
    state = torch.load(checkpoint, weights_only=False)
    assert state["model_config"]["code_encoding"] == encoding

    path.unlink()  # a checkpoint is read without the vector file
    model, vocab = load_model(checkpoint, Path(cfg.data.prepared_dir), "cpu")
    tokens = torch.arange(len(vocab.tokens)).unsqueeze(0)
    with torch.no_grad():
        encoded = model.input_head.embed(tokens)
    assert encoded.shape == (1, len(vocab.tokens), cfg.model.d_model) and torch.isfinite(encoded).all()
    for k, v in model.state_dict().items():
        assert torch.equal(v, state["model"][k]), k


def test_vectors_of_another_vocabulary_are_refused(tmp_path):
    table = pd.DataFrame({"token": [PAD, "A"], "text": ["", "a"], "source": ["pad", "name"]})
    path = write_vectors(tmp_path / "v.npz", table, np.zeros((2, 4), np.float32), "none", "none")
    assert read_vectors(path, [PAD, "A"]).shape == (2, 4)
    with pytest.raises(ValueError, match="another vocabulary"):
        read_vectors(path, [PAD, "B"])


def test_prepared_tokens_have_texts(tiny_config):
    run_prepare(tiny_config.data, log=QUIET)
    table = token_texts(tiny_config.data.prepared_dir, tiny_config.data.meds_dir)
    vocab = Vocabulary.load(Path(tiny_config.data.prepared_dir) / "vocab.json")
    assert table["token"].tolist() == vocab.tokens
    assert (table.loc[1:, "text"].str.len() > 0).all()
