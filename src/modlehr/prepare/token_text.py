"""`modlehr embed-tokens`: a text for every token of a prepared vocabulary, and its vector from a text encoder.

The vectors are what model.code_encoding text_frozen, text_init and text_sum read (model.code_vectors), so a
token enters the model through what its code means rather than through an id alone. Each token's text:

    exact       the token's own description in the MEDS codes table
    category    a grouped target (ICD10//E11, ICD9//428): its category's title (MIMIC-IV's d_icd tables hold
                the three-character ICD-10 categories), else the title of its first member code ("member")
    value       a binned measurement (code//Q2): its code's description, "level 2 of 3"; a categorical answer
                (code//answer): its code's description, then the answer
    name        no description: the code itself, its parts joined (MEDICATION//START//ASPIRIN →
                "Medication: START ASPIRIN"); padding has no text and a zero vector

The encoder runs once, offline: a Hugging Face model (a path or a name), each text pooled to one vector (the
[CLS] token of an encoder such as SapBERT, the last token of a decoder such as Qwen3-Embedding) and scaled
to length 1. The output, <prepared_dir>/token_vectors/<encoder>.npz, holds the tokens it was made for; a model
checks them against its vocabulary. The texts are written beside it (.tsv) to read through.

A code table (--code-table) gives the vectors instead: vectors of codes made elsewhere, such as MedTok's
code2embeddings.json ({code: vector}, ICD codes written with their dot, E11.9, 428.0). A diagnosis token takes
its code's vector, or else the mean of the vectors of the codes under it (ICD10//E11: E11.0, E11.65, ...); a
token the table does not have (an admission type, sex) takes a fixed random vector of its own, so tokens stay
apart. The table has no families, so where ICD-9 and ICD-10 write the same code (V codes) the two are joined.
"""

from __future__ import annotations

import bisect
import json
import re
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from modlehr.prepare.vocabulary import NO_EVENT, PAD, Vocabulary

BIN = re.compile(r"^(.*)//Q(\d+)$")
# raw diagnosis codes as the adapters write them → (family, code without dots) of a grouped target
RAW_ICD = (re.compile(r"^DIAGNOSIS//ICD//(9|10)//(.+)$"), re.compile(r"^ICD(9|10)(?:CM)?//(.+)$"))


def icd_titles(descriptions: dict[str, str]) -> dict[str, dict[str, str]]:
    """{"ICD10": {code: title}, "ICD9": {...}} from the described diagnosis codes, dots removed."""
    out: dict[str, dict[str, str]] = {"ICD9": {}, "ICD10": {}}
    for code, text in descriptions.items():
        for pattern in RAW_ICD:
            m = pattern.match(code)
            if m:
                out[f"ICD{m.group(1)}"].setdefault(m.group(2).replace(".", "").upper(), text)
                break
    return out


def name_text(token: str) -> str:
    head, *rest = token.split("//")
    head = head.replace("_", " ")
    if head.isupper() and not any(c.isdigit() for c in head):  # MEDICATION → Medication; ICD9 stays
        head = head.capitalize()
    return head + (": " + " ".join(rest) if rest else "")


def describe_tokens(tokens: list[str], descriptions: dict[str, str], n_bins: int | None = None) -> tuple[list[str], list[str]]:
    """The text of each token and where it came from (exact, category, member, value, name, pad)."""
    descriptions = {c: d.strip() for c, d in descriptions.items() if isinstance(d, str) and d.strip()}
    titles = icd_titles(descriptions)
    members = {fam: sorted(t) for fam, t in titles.items()}
    texts, sources = [], []
    for token in tokens:
        text, source = describe_one(token, descriptions, titles, members, n_bins)
        texts.append(text)
        sources.append(source)
    return texts, sources


def describe_one(token, descriptions, titles, members, n_bins) -> tuple[str, str]:
    if token == PAD:
        return "", "pad"
    if token == NO_EVENT:
        return "No event", "exact"
    if token in descriptions:
        return descriptions[token], "exact"
    family, sep, code = token.partition("//")
    if sep and family in titles:
        code = code.replace(".", "").upper()
        if code in titles[family]:
            return titles[family][code], "category"
        found = members[family]
        i = int(np.searchsorted(found, code))
        if i < len(found) and found[i].startswith(code):
            return titles[family][found[i]], "member"
    m = BIN.match(token)
    if m and m.group(1) in descriptions:
        of = f" of {n_bins}" if n_bins else ""
        return f"{descriptions[m.group(1)]}, level {m.group(2)}{of}", "value"
    base, sep, answer = token.rpartition("//")
    if sep and base in descriptions:
        return f"{descriptions[base]}: {answer}", "value"
    return name_text(token), "name"


def token_texts(prepared_dir: str | Path, meds_dir: str | Path) -> pd.DataFrame:
    """token, text, source for every token of the prepared vocabulary, in vocabulary order."""
    from modlehr.data.meds_io import open_meds

    prepared_dir = Path(prepared_dir)
    vocab = Vocabulary.load(prepared_dir / "vocab.json")
    meta = json.loads((prepared_dir / "prepare.json").read_text())
    data = meta.get("data_config") or {}
    n_bins = data.get("n_bins") if data.get("value_mode") == "binned" else None
    codes = open_meds(meds_dir).codes
    descriptions = dict(zip(codes["code"].astype(str), codes["description"]))
    texts, sources = describe_tokens(vocab.tokens, descriptions, n_bins)
    return pd.DataFrame({"token": vocab.tokens, "text": texts, "source": sources})


def encode(texts: list[str], encoder: str, pooling: str = "auto", batch_size: int = 32, device: str = "cpu",
           max_length: int = 128) -> tuple[np.ndarray, str]:
    """[len(texts), dim] unit vectors, and the pooling used. Empty texts get zero vectors."""
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as e:
        raise SystemExit("modlehr embed-tokens needs the transformers package: pip install transformers") from e

    tokenizer = AutoTokenizer.from_pretrained(encoder)
    if pooling == "auto":  # an encoder (BERT, SapBERT) is read at its [CLS] token, a decoder (Qwen3-Embedding) at its last
        pooling = "cls" if tokenizer.cls_token else "last"
    if pooling not in ("last", "cls", "mean"):
        raise ValueError(f"pooling must be auto, last, cls or mean, not {pooling!r}")
    if pooling == "last":
        tokenizer.padding_side = "left"
    import transformers

    dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    key = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"  # renamed in transformers 5
    model = AutoModel.from_pretrained(encoder, **{key: dtype}).to(device).eval()
    keep = [i for i, t in enumerate(texts) if t]
    out = None
    with torch.no_grad():
        for b in range(0, len(keep), batch_size):
            idx = keep[b:b + batch_size]
            batch = tokenizer([texts[i] for i in idx], padding=True, truncation=True, max_length=max_length,
                              return_tensors="pt").to(device)
            hidden = model(**batch).last_hidden_state.float()
            if pooling == "last":
                vec = hidden[:, -1]
            elif pooling == "cls":
                vec = hidden[:, 0]
            else:
                mask = batch["attention_mask"].unsqueeze(-1).float()
                vec = (hidden * mask).sum(1) / mask.sum(1)
            vec = torch.nn.functional.normalize(vec, dim=-1).cpu().numpy()
            if out is None:
                out = np.zeros((len(texts), vec.shape[1]), dtype=np.float32)
            out[idx] = vec
    if out is None:
        raise ValueError("no token has a text to encode")
    return out, pooling


def write_vectors(path: str | Path, table: pd.DataFrame, vectors: np.ndarray, encoder: str, pooling: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, tokens=np.asarray(table["token"], dtype=str), vectors=vectors.astype(np.float16),
             encoder=np.asarray(encoder), pooling=np.asarray(pooling))
    table.to_csv(path.with_suffix(".tsv"), sep="\t", index=False)
    return path


def read_vectors(path: str | Path, tokens: list[str]) -> np.ndarray:
    """The [len(tokens), dim] vectors of a file written for exactly this vocabulary."""
    raw = np.load(path)
    stored = [str(t) for t in raw["tokens"]]
    if stored != list(tokens):
        raise ValueError(f"{path} was written for another vocabulary ({len(stored):,} tokens, this one {len(tokens):,}); "
                         "run modlehr embed-tokens on this prepared data")
    return raw["vectors"].astype(np.float32)


def read_code_table(path: str | Path) -> dict[str, np.ndarray]:
    """{code: vector} from a JSON file of {code: [numbers]} (MedTok's code2embeddings.json) or an .npz of codes and
    vectors."""
    path = Path(path)
    if path.suffix == ".npz":
        raw = np.load(path)
        return dict(zip((str(c) for c in raw["codes"]), raw["vectors"].astype(np.float32)))
    with open(path) as f:
        return {str(c): np.asarray(v, dtype=np.float32) for c, v in json.load(f).items()}


def icd_key(family: str, code: str) -> str:
    """A diagnosis code as a code table writes it: E119 → E11.9, 4280 → 428.0, ICD-9 E8000 → E800.0, E11 → E11."""
    code = code.replace(".", "").upper()
    cut = 4 if family == "ICD9" and code.startswith("E") else 3
    return code if len(code) <= cut else f"{code[:cut]}.{code[cut:]}"


def table_vectors(tokens: list[str], table: dict[str, np.ndarray]) -> tuple[np.ndarray, pd.DataFrame]:
    """Each token's unit vector from a code table, and a table of token, key, codes (how many were averaged) and
    source: exact (the token's code is in the table), member (the mean of the codes under it), random (not in
    the table: a random vector seeded by the token, the same every time) or pad (zero)."""
    keys = sorted(table)
    dim = len(next(iter(table.values())))
    out = np.zeros((len(tokens), dim), dtype=np.float32)
    rows = []
    for i, token in enumerate(tokens):
        key, n, source = "", 0, "pad"
        if token != PAD:
            m = next(filter(None, (p.match(token) for p in RAW_ICD)), None)
            key = icd_key(f"ICD{m.group(1)}", m.group(2)) if m else token
            if key in table:
                vec, n, source = table[key], 1, "exact"
            else:
                # codes under the key: E11 → E11.*, and E11.6 → E11.6* (E800.0 is not under E80)
                prefix = key if "." in key else key + "."
                lo = bisect.bisect_left(keys, prefix)
                hi = bisect.bisect_left(keys, prefix + "￿")
                if m and hi > lo:
                    vec, n, source = np.mean([table[k] for k in keys[lo:hi]], axis=0), hi - lo, "member"
                else:
                    vec, source = np.random.default_rng(zlib.crc32(token.encode())).standard_normal(dim), "random"
            out[i] = vec / (np.linalg.norm(vec) or 1.0)
        rows.append((token, key, n, source))
    return out, pd.DataFrame(rows, columns=["token", "key", "codes", "source"])


def embed_tokens(prepared_dir: str | Path, meds_dir: str | Path, encoder: str | None, out: str | Path | None = None,
                 pooling: str = "auto", batch_size: int = 32, device: str = "cpu", log=print,
                 code_table: str | Path | None = None) -> Path:
    """Write the token texts and, with an encoder, their vectors; or, with a code table, each token's vector from
    it. Returns the .npz path (or the .tsv with neither)."""
    if code_table is not None:
        tokens = Vocabulary.load(Path(prepared_dir) / "vocab.json").tokens
        vectors, found = table_vectors(tokens, read_code_table(code_table))
        log("[embed] " + ", ".join(f"{n:,} {s}" for s, n in found["source"].value_counts().items())
            + f" of {len(found):,} tokens")
        path = Path(out or Path(prepared_dir) / "token_vectors" / f"{Path(code_table).stem}.npz")
        write_vectors(path, found, vectors, str(code_table), "table")
        log(f"[embed] {vectors.shape[1]}-dimensional vectors from {code_table} → {path}")
        return path
    table = token_texts(prepared_dir, meds_dir)
    counts = table["source"].value_counts()
    log("[embed] " + ", ".join(f"{n:,} {s}" for s, n in counts.items()) + f" of {len(table):,} tokens")
    if encoder is None:
        path = Path(out or Path(prepared_dir) / "token_vectors" / "texts.tsv").with_suffix(".tsv")
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, sep="\t", index=False)
        log(f"[embed] texts → {path}")
        return path
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(encoder).name or encoder)
    path = Path(out or Path(prepared_dir) / "token_vectors" / f"{slug}.npz")
    vectors, pooling = encode(table["text"].tolist(), encoder, pooling, batch_size, device)
    write_vectors(path, table, vectors, encoder, pooling)
    log(f"[embed] {vectors.shape[1]}-dimensional vectors from {encoder} ({pooling} pooling) → {path}")
    return path
