"""The words of clinical notes as hashed ids, with no external tokenizer or model.

A note is lowercased and split into runs of letters and digits; each word is hashed with CRC-32,
which gives the same id on every machine and in every Python run, into one of `buckets` ids. The
first `max_words` words of each note are kept. At 65,536 buckets collisions are rare for a clinical
vocabulary and cost a bag-of-words encoder little. A text encoding that needs real word pieces (a
pretrained clinical language model) would register its own tokenization the same way.

For one split, the words are stored ragged: text_ref gives each event its note index (-1 if it has
no text), and note i's words are note_words[note_offsets[i]:note_offsets[i + 1]].
"""

from __future__ import annotations

import re
import zlib

import numpy as np
import pandas as pd

WORD = re.compile(r"[a-z0-9]+")
TEXT_FILES = ("text_ref", "note_offsets", "note_words")


def hash_words(text: str, buckets: int, max_words: int) -> np.ndarray:
    words = WORD.findall(str(text).lower())[:max_words]
    return np.fromiter((zlib.crc32(w.encode()) % buckets for w in words), dtype=np.int32, count=len(words))


def build_text(frame: pd.DataFrame, buckets: int, max_words: int) -> dict[str, np.ndarray]:
    """text_ref, note_offsets, note_words for a frame already in its final event order."""
    has = frame["text"].notna().to_numpy() if "text" in frame else np.zeros(len(frame), dtype=bool)
    text_ref = np.full(len(frame), -1, dtype=np.int32)
    text_ref[has] = np.arange(int(has.sum()), dtype=np.int32)
    notes = [hash_words(t, buckets, max_words) for t in frame.loc[has, "text"]] if has.any() else []
    lengths = np.array([len(n) for n in notes], dtype=np.int64)
    return {"text_ref": text_ref, "note_offsets": np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64),
            "note_words": np.concatenate(notes).astype(np.int32) if notes else np.zeros(0, dtype=np.int32)}
