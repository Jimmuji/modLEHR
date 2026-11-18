"""`modlehr subset`: a smaller copy of prepared data, for quick runs.

Keeps a fixed fraction of the subjects of every split and nothing else changes: the vocabulary, bin cut points and
target counts are those of the full data, so a model trained on the subset reads the same tokens, and its numbers
compare with other runs on the same subset. The subjects kept are those of `train.data_fraction` at the same
fraction (subject_subset, seed 0), and a smaller fraction keeps a subset of a larger one.

Task samples already written for the full data (`modlehr finetune --samples-only`, <prepared_dir>/tasks) are
carried over for the kept subjects, and token vectors (<prepared_dir>/token_vectors) are copied, since both depend
on the vocabulary only. prepare.json records where the subset came from; `modlehr prepare` refuses to write over a
subset, which is remade from the full data instead.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from modlehr.prepare.dataset import DTYPES, open_split, subject_subset
from modlehr.prepare.text import TEXT_FILES
from modlehr.prepare.timeline import ARRAYS, SUBJECT_ARRAYS

SPLITS = ("train", "tuning", "held_out")


def _sha1(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def default_out(prepared_dir: str | Path, fraction: float) -> Path:
    """runs/prepared/mimic_iv and 0.1 give runs/prepared/mimic_iv_10pct."""
    pct = f"{fraction * 100:g}".replace(".", "p")
    return Path(f"{Path(prepared_dir)}_{pct}pct")


def _write_split(src: Path, dst: Path, keep: np.ndarray | None) -> dict[str, int]:
    split = open_split(src.parent, src.name)
    keep = np.arange(len(split)) if keep is None else keep
    a, b = split.offsets[keep], split.offsets[keep + 1]
    lengths = b - a
    rows = np.repeat(a - np.concatenate([[0], np.cumsum(lengths)[:-1]]), lengths) + np.arange(int(lengths.sum()))
    dst.mkdir(parents=True)
    np.save(dst / "subject_ids.npy", split.subject_ids[keep].astype(np.int64))
    np.save(dst / "offsets.npy", np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64))
    for k in SUBJECT_ARRAYS:
        if k in split.arrays:
            np.save(dst / f"{k}.npy", np.asarray(split.arrays[k])[keep].astype(DTYPES[k]))
    for k in ARRAYS:
        if k in split.arrays:
            np.save(dst / f"{k}.npy", np.asarray(split.arrays[k][rows]).astype(DTYPES[k]))
    if split.has_text:  # renumber the kept notes in order
        refs = np.asarray(split.arrays["text_ref"][rows])
        old = refs[refs >= 0]
        starts, stops = split.note_offsets[old], split.note_offsets[old + 1]
        words = split.arrays["note_words"]
        new_refs = np.full(len(refs), -1, dtype=np.int32)
        new_refs[refs >= 0] = np.arange(len(old), dtype=np.int32)
        np.save(dst / "text_ref.npy", new_refs)
        np.save(dst / "note_words.npy", np.concatenate([np.asarray(words[s:e]) for s, e in zip(starts, stops)]
                                                      or [np.zeros(0, np.int32)]).astype(np.int32))
        np.save(dst / "note_offsets.npy", np.concatenate([[0], np.cumsum(stops - starts)]).astype(np.int64))
        assert set(TEXT_FILES) <= {p.stem for p in dst.glob("*.npy")}
    return {"subjects": int(len(keep)), "events": int(lengths.sum())}


def subset_samples(samples: dict[str, np.ndarray], n_full: int, keep: np.ndarray) -> dict[str, np.ndarray]:
    """A task's samples of a full split (finetune.task_samples) restricted to its subjects `keep`, with `index`
    pointing into the subset split."""
    new_index = np.full(n_full, -1, dtype=np.int64)
    new_index[keep] = np.arange(len(keep))
    rows = samples["index"] >= 0
    rows[rows] = new_index[samples["index"][rows]] >= 0
    out = {k: v[rows] for k, v in samples.items()}
    out["index"] = new_index[samples["index"][rows]]
    return out


def _carry_tasks(src: Path, dst: Path, kept: dict[str, np.ndarray], prepare_hash: str, log) -> None:
    for cache in sorted((src / "tasks").glob("*/*.npz")):
        split = cache.stem
        if split not in kept:
            continue
        with np.load(cache) as z:
            arrays = {k: z[k] for k in z.files}
        key = json.loads(str(arrays.pop("key")))
        key["prepare"] = prepare_hash  # finetune.task_samples reuses a cache whose key matches
        out = subset_samples(arrays, len(open_split(src, split)), kept[split])
        (dst / "tasks" / cache.parent.name).mkdir(parents=True, exist_ok=True)
        np.savez(dst / "tasks" / cache.parent.name / cache.name, key=np.asarray(json.dumps(key)), **out)
        log(f"[subset] task {cache.parent.name}/{split}: {len(out['index']):,} of {len(arrays['index']):,} samples "
            f"({int(out['label'].sum()):,} positive)")


def make_subset(prepared_dir: str | Path, fraction: float, out: str | Path | None = None, log=print) -> Path:
    src = Path(prepared_dir)
    if not 0 < fraction < 1:
        raise ValueError(f"fraction must be in (0, 1), not {fraction!r}")
    if not (src / "prepare.json").exists():
        raise FileNotFoundError(f"{src} has no prepare.json; run `modlehr prepare` first")
    meta = json.loads((src / "prepare.json").read_text())
    if "subset" in meta:
        raise ValueError(f"{src} is itself a subset of {meta['subset']['of']}; make the subset from the full data")
    dst = Path(out) if out else default_out(src, fraction)
    if dst.resolve() == src.resolve():
        raise ValueError("the subset needs a directory of its own")
    if dst.exists():
        old = json.loads((dst / "prepare.json").read_text()) if (dst / "prepare.json").exists() else {}
        if "subset" not in old:
            raise FileExistsError(f"{dst} exists and is not a subset; not writing over it")
        shutil.rmtree(dst)
    tmp = dst.with_name(dst.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    kept, summary = {}, {}
    for split in SPLITS:
        n = len(open_split(src, split))
        kept[split] = subject_subset(n, fraction)
        summary[split] = _write_split(src / split, tmp / split, kept[split])
        full = meta["summary"][split]
        log(f"[subset] {split}: {summary[split]['subjects']:,} of {full['subjects']:,} subjects, "
            f"{summary[split]['events']:,} of {full['events']:,} events")
    shutil.copy2(src / "vocab.json", tmp / "vocab.json")
    if (src / "token_vectors").is_dir():
        shutil.copytree(src / "token_vectors", tmp / "token_vectors")
    meta["subset"] = {"of": str(src), "fraction": fraction, "seed": 0, "full_summary": meta["summary"],
                      "full_prepare_sha1": _sha1(src / "prepare.json")}
    meta["summary"] = summary
    meta["data_config"]["prepared_dir"] = str(dst)
    (tmp / "prepare.json").write_text(json.dumps(meta, indent=1))
    _carry_tasks(src, tmp, kept, _sha1(tmp / "prepare.json"), log)
    tmp.rename(dst)
    log(f"[subset] {fraction:g} of the subjects of {src} → {dst}")
    return dst


def is_current_subset(prepared_dir: str | Path, fraction: float, out: str | Path | None = None) -> bool:
    """True if the subset exists and was made from the full data as it is now."""
    dst = Path(out) if out else default_out(prepared_dir, fraction)
    if not (dst / "prepare.json").exists():
        return False
    sub = json.loads((dst / "prepare.json").read_text()).get("subset") or {}
    return sub.get("fraction") == fraction and sub.get("full_prepare_sha1") == _sha1(Path(prepared_dir) / "prepare.json")


def ensure_subset(prepared_dir: str | Path, fraction: float, out: str | Path | None = None, log=print) -> Path:
    """The subset, made (or remade, if the full data was prepared again since) unless it is current. Several jobs
    can call this at once: one makes it while the others wait."""
    import fcntl

    dst = Path(out) if out else default_out(prepared_dir, fraction)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst.with_name(dst.name + ".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if is_current_subset(prepared_dir, fraction, dst):
            log(f"[subset] {dst} is current")
            return dst
        return make_subset(prepared_dir, fraction, dst, log)
