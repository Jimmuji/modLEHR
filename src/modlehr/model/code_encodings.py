"""Code encodings: how a token becomes its input vector, E_code(c) in the input head (model.code_encoding).

    learned      one learned vector per token, started at random (the default; BEHRT, Delphi-2M)
    text_frozen  W t(c): t(c) is a text encoder's vector of the token's description (model.code_vectors, from
                 `modlehr embed-tokens`), fixed; only the projection W is learned, so codes with similar
                 descriptions start and stay close
    text_init    a learned table started from t(c): its top principal components, at the spread of the default
                 start, then trained like `learned`
    text_sum     W t(c) + R(c): the fixed text part plus a learned per-token part R that starts at zero, for
                 what the description does not say (type 1 and type 2 diabetes read alike)
    hierarchy    E(c) + the sum of E_g(g) over the groups c belongs to: its ICD chapter (ICD-9 and ICD-10 share
                 them) and first two characters, the leading characters of a procedure code, a measurement's
                 code (shared by its bins), the first word of a drug, and the leading parts of any code
                 written A//B//C (A, A//B); a group with one token is left out. No outside data needed

The text encodings keep t(c) in the checkpoint (as float16), so a checkpoint is read without the vector file.
"""

from __future__ import annotations

import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from modlehr.registry import register

START_STD = 0.02  # the spread of a default embedding at the start (ModLEHR._init)


@register("code_encoding", "learned")
def learned(n_tokens: int, d_model: int, vectors=None, ancestors=None) -> nn.Embedding:
    return nn.Embedding(n_tokens, d_model, padding_idx=0)


class TextCodes(nn.Module):
    def __init__(self, n_tokens: int, d_model: int, vectors: torch.Tensor, residual: bool):
        super().__init__()
        if vectors is None or vectors.shape[0] != n_tokens:
            raise ValueError(f"a text code encoding needs one vector per token ({n_tokens}); set model.code_vectors")
        self.register_buffer("vectors", vectors.to(torch.float16))
        self.proj = nn.Linear(vectors.shape[1], d_model, bias=False)
        self.residual = nn.Parameter(torch.zeros(n_tokens, d_model)) if residual else None

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        h = self.proj(self.vectors[tokens].to(self.proj.weight.dtype))
        return h if self.residual is None else h + F.embedding(tokens, self.residual, padding_idx=0)


@register("code_encoding", "text_frozen")
def text_frozen(n_tokens: int, d_model: int, vectors=None, ancestors=None) -> TextCodes:
    return TextCodes(n_tokens, d_model, vectors, residual=False)


@register("code_encoding", "text_sum")
def text_sum(n_tokens: int, d_model: int, vectors=None, ancestors=None) -> TextCodes:
    return TextCodes(n_tokens, d_model, vectors, residual=True)


def principal_components(vectors: torch.Tensor, d_model: int) -> torch.Tensor:
    """[n_tokens, d_model]: each token's coordinates on the top d_model principal components of the vectors
    (padding, row 0, left out and kept at zero), scaled to START_STD; missing components are zero."""
    x = vectors[1:].double()
    x = x - x.mean(0)
    _, _, vt = torch.linalg.svd(x, full_matrices=False)
    z = x @ vt[:d_model].T
    out = torch.zeros(vectors.shape[0], d_model, dtype=torch.float64)
    out[1:, :z.shape[1]] = z
    std = float(out[1:].std())
    return (out * (START_STD / std if std > 0 else 1.0)).float()


@register("code_encoding", "text_init")
class TextInitCodes(nn.Module):
    def __init__(self, n_tokens: int, d_model: int, vectors=None, ancestors=None):
        super().__init__()
        # without vectors (a checkpoint being read) the table is a placeholder its saved weights replace
        start = principal_components(vectors, d_model) if vectors is not None else torch.zeros(n_tokens, d_model)
        self.weight = nn.Parameter(start)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return F.embedding(tokens, self.weight, padding_idx=0)


@register("code_encoding", "hierarchy")
class HierarchyCodes(nn.Module):
    def __init__(self, n_tokens: int, d_model: int, vectors=None, ancestors: torch.Tensor | None = None):
        super().__init__()
        if ancestors is None or ancestors.shape[0] != n_tokens:
            raise ValueError("code_encoding hierarchy needs the groups of each token (token_ancestors)")
        self.embed = nn.Embedding(n_tokens, d_model, padding_idx=0)
        self.register_buffer("ancestors", ancestors.long())
        groups = torch.randn(int(ancestors.max()) + 1, d_model) * START_STD
        groups[0] = 0.0  # "no group"
        self.groups = nn.Parameter(groups)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.embed(tokens) + F.embedding(self.ancestors[tokens], self.groups, padding_idx=0).sum(-2)


# ICD-10(-CM) chapters by their first three characters, and the ICD-9-CM ranges that hold the same diseases
ICD10_CHAPTERS = [("A00", "B99"), ("C00", "D49"), ("D50", "D89"), ("E00", "E89"), ("F01", "F99"), ("G00", "G99"),
                  ("H00", "H59"), ("H60", "H95"), ("I00", "I99"), ("J00", "J99"), ("K00", "K95"), ("L00", "L99"),
                  ("M00", "M99"), ("N00", "N99"), ("O00", "O9A"), ("P00", "P96"), ("Q00", "Q99"), ("R00", "R99"),
                  ("S00", "T88"), ("V00", "Y99"), ("Z00", "Z99"), ("U00", "U85")]
ICD9_CHAPTERS = [(1, 139, 1), (140, 239, 2), (240, 279, 4), (280, 289, 3), (290, 319, 5), (320, 359, 6), (360, 379, 7),
                 (380, 389, 8), (390, 459, 9), (460, 519, 10), (520, 579, 11), (580, 629, 14), (630, 679, 15),
                 (680, 709, 12), (710, 739, 13), (740, 759, 17), (760, 779, 16), (780, 799, 18), (800, 999, 19)]
ICD9_LETTER_CHAPTER = {"V": 21, "E": 20}


def icd10_chapter(code: str) -> int | None:
    head = code[:3].upper()
    for i, (lo, hi) in enumerate(ICD10_CHAPTERS, start=1):
        if lo <= head <= hi:
            return i
    return None


def icd9_chapter(code: str) -> int | None:
    if code[:1].upper() in ICD9_LETTER_CHAPTER:
        return ICD9_LETTER_CHAPTER[code[:1].upper()]
    digits = re.match(r"\d{1,3}", code)
    if not digits:
        return None
    n = int(digits.group())
    return next((ch for lo, hi, ch in ICD9_CHAPTERS if lo <= n <= hi), None)


def groups_of(token: str) -> list[str]:
    """The groups a token belongs to, coarsest first (see the module docstring)."""
    parts = token.split("//")
    out = ["//".join(parts[:i]) for i in range(1, len(parts))]
    family, code = (parts[0], parts[-1]) if len(parts) == 2 else (None, None)
    if family == "ICD10":
        chapter = icd10_chapter(code)
        out += ([f"CHAPTER//{chapter}"] if chapter else []) + [f"ICD10//{code[:2]}*"]
    elif family == "ICD9":
        chapter = icd9_chapter(code)
        out += ([f"CHAPTER//{chapter}"] if chapter else []) + [f"ICD9//{code[:2]}*"]
    elif family == "PHECODE":
        out.append(f"PHECODE//{code.split('.')[0]}*")
    if len(parts) == 4 and parts[0] == "PROCEDURE" and parts[1] == "ICD":
        out += [f"{token[:-len(parts[3])]}{parts[3][:k]}*" for k in ((1, 2, 3) if parts[2] == "10" else (1, 2))]
    if len(parts) >= 3 and parts[0] == "MEDICATION":
        out.append(f"{'//'.join(parts[:-1])}//{parts[-1].split()[0] if parts[-1].split() else parts[-1]}*")
    m = re.match(r"^(.*)//Q\d+$", token)
    if m:
        out.append(m.group(1))
    return list(dict.fromkeys(out))


def token_ancestors(tokens: list[str]) -> torch.Tensor:
    """[n_tokens, depth] ids (from 1) of each token's groups that hold at least two tokens; 0 pads, and padding
    (token 0) has none."""
    per_token = [[] if i == 0 else groups_of(t) for i, t in enumerate(tokens)]
    size: dict[str, int] = {}
    for groups in per_token:
        for g in groups:
            size[g] = size.get(g, 0) + 1
    ids = {g: i + 1 for i, g in enumerate(sorted(g for g, n in size.items() if n >= 2))}
    kept = [[ids[g] for g in groups if g in ids] for groups in per_token]
    out = np.zeros((len(tokens), max([len(k) for k in kept] + [1])), dtype=np.int64)
    for i, k in enumerate(kept):
        out[i, :len(k)] = k
    return torch.as_tensor(out)


def code_encoding_inputs(model_cfg, tokens: list[str], state: dict | None = None) -> dict:
    """What the configured code encoding is built from: the vectors or groups of these tokens, from a
    checkpoint's state when one is being read, otherwise from model.code_vectors or the tokens."""
    name = model_cfg.code_encoding
    if name in ("text_frozen", "text_sum"):
        key = "input_head.embed.vectors"
        if state is not None and key in state:
            return {"vectors": state[key]}
        return {"vectors": _read_vectors(model_cfg, tokens)}
    if name == "text_init":
        return {} if state is not None else {"vectors": _read_vectors(model_cfg, tokens)}
    if name == "hierarchy":
        key = "input_head.embed.ancestors"
        return {"ancestors": state[key] if state is not None and key in state else token_ancestors(tokens)}
    return {}


def _read_vectors(model_cfg, tokens: list[str]) -> torch.Tensor:
    from modlehr.prepare.token_text import read_vectors

    if not model_cfg.code_vectors:
        raise ValueError(f"model.code_encoding {model_cfg.code_encoding} needs model.code_vectors, a file written by "
                         "modlehr embed-tokens")
    return torch.as_tensor(read_vectors(model_cfg.code_vectors, tokens))
