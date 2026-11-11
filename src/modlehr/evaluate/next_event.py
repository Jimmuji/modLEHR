"""`modlehr next-event`: how well a model pretrained with model.head: next_token predicts the next event.

A next_token model gives no risk within a horizon, so `modlehr evaluate` does not apply. This scores what it
was trained for, on the evaluation split (held_out by default), in two parts.

Overall. For every event after a subject's first, the model's distribution over the next event, read at the
event before it: mean cross-entropy of the event that came (nats), its perplexity, and top-1, top-5 and top-10
accuracy. The same broken down by the family of the event that came (the code's first part: LAB, MEDICATION,
PROCEDURE, ICD10, ...), since families differ in how predictable they are.

Representative events. evaluate.next_event_groups names groups of tokens by a regular expression over token
names (every ICU admission, death, a lab result in its top third, a vasopressor). For each group the model's
probability that the next event is in the group (the sum over its tokens) is scored against whether it is:
AUROC, AUPRC, the share of next events in the group, the mean predicted probability (calibration in the large)
and how often the event that came was among the model's top 10 when it was in the group.

Each timeline is cut into windows of max_len events that overlap by one, so every next event is scored once,
with as much history as fits in a window. evaluate.next_event_max_subjects scores a random subset of subjects.

Writes <train.out_dir>/next_event.json and puts the numbers in the run's wandb summary.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from modlehr.config import Config
from modlehr.evaluate.run import load_model
from modlehr.prepare.dataset import PreparedSplit, collate, open_split
from modlehr.track import Tracker
from modlehr.train.loop import pick_device, to_device

TOP_K = (1, 5, 10)
CHUNK = 8192  # positions scored at once (each holds a row of n_tokens logits)


def family(token: str) -> str:
    """LAB//RESULT (hospital labs) and LAB (ICU charts) apart, else the code's first part."""
    parts = token.split("//")
    return "LAB//RESULT" if parts[:2] == ["LAB", "RESULT"] else parts[0]


def group_members(tokens: list[str], groups: dict[str, str]) -> dict[str, np.ndarray]:
    """Token indices of each group: the tokens whose name matches its regular expression (padding never does)."""
    out = {}
    for name, pattern in groups.items():
        rx = re.compile(pattern)
        idx = np.array([i for i, t in enumerate(tokens) if i and rx.search(t)], dtype=np.int64)
        if not len(idx):
            raise ValueError(f"evaluate.next_event_groups: {name} ({pattern!r}) matches no token")
        out[name] = idx
    return out


def overlapping_windows(split: PreparedSplit, subjects: np.ndarray, length: int) -> list[dict]:
    """Windows of up to `length` events that overlap by one, so each event after a subject's first is the next
    event of exactly one position."""
    step = max(length - 1, 1)
    lengths = split.lengths()
    rows = []
    for i in subjects:
        n = int(lengths[i])
        for s in range(0, max(n - 1, 1), step):
            rows.append(split.subject(int(i), s, min(n, s + length), labels=False))
    return rows


@torch.no_grad()
def next_event(cfg: Config, checkpoint: str | Path, out_json: str | Path | None = None, log=print) -> dict:
    prepared = Path(cfg.data.prepared_dir)
    device = pick_device(cfg.train.device)
    model, vocab = load_model(checkpoint, prepared, device)
    if getattr(model, "cfg", None) is None or model.cfg.head != "next_token":
        raise ValueError(f"{checkpoint} was not trained with model.head: next_token, so it gives no distribution over "
                         "every next event; score it with `modlehr evaluate` or `modlehr finetune`")
    tokens = list(vocab.tokens)
    n_tokens = len(tokens)
    families = sorted({family(t) for t in tokens[1:]})
    family_of = torch.tensor([0] + [families.index(family(t)) for t in tokens[1:]], device=device)
    groups = group_members(tokens, cfg.evaluate.next_event_groups)
    in_group = {g: torch.zeros(n_tokens, dtype=torch.bool, device=device).index_fill_(0, torch.from_numpy(idx).to(device), True)
                for g, idx in groups.items()}

    split = open_split(prepared, cfg.evaluate.split)
    subjects = np.arange(len(split))
    limit = cfg.evaluate.next_event_max_subjects
    if limit and limit < len(subjects):
        subjects = np.sort(np.random.default_rng(cfg.train.seed).choice(subjects, limit, replace=False))
    rows = overlapping_windows(split, subjects, model.cfg.max_len)
    rows.sort(key=lambda r: len(r["tokens"]))

    n = 0
    nll = 0.0
    hits = {k: 0 for k in TOP_K}
    fam_n = torch.zeros(len(families) + 1, dtype=torch.float64, device=device)
    fam_nll = torch.zeros_like(fam_n)
    fam_hit = {k: torch.zeros_like(fam_n) for k in (1, 10)}
    prob = {g: [] for g in groups}
    label = {g: [] for g in groups}
    group_hit10 = {g: 0 for g in groups}
    bs = max(1, cfg.evaluate.next_event_batch_size)
    for b in range(0, len(rows), bs):
        batch = to_device(collate(rows[b:b + bs]), device)
        logits = model(batch)
        tok = batch["tokens"]
        following = torch.cat([tok[:, 1:], torch.zeros_like(tok[:, :1])], dim=1)
        keep = following != 0
        flat, true = logits[keep], following[keep]
        for c in range(0, len(true), CHUNK):
            x, y = flat[c:c + CHUNK].float(), true[c:c + CHUNK]
            lse = torch.logsumexp(x, dim=-1)
            loss = lse - x.gather(1, y[:, None]).squeeze(1)
            top = x.topk(min(max(TOP_K), n_tokens), dim=-1).indices  # a vocabulary may be smaller than 10
            hit = {k: (top[:, :k] == y[:, None]).any(dim=1) for k in TOP_K}
            nll += float(loss.double().sum())
            for k in TOP_K:
                hits[k] += int(hit[k].sum())
            f = family_of[y]
            fam_n.index_add_(0, f, torch.ones_like(loss, dtype=torch.float64))
            fam_nll.index_add_(0, f, loss.double())
            for k in fam_hit:
                fam_hit[k].index_add_(0, f, hit[k].double())
            for g, idx in groups.items():
                lp = torch.logsumexp(x[:, torch.from_numpy(idx).to(device)], dim=-1) - lse
                yes = in_group[g][y]
                prob[g].append(lp.exp().cpu().numpy().astype(np.float32))
                label[g].append(yes.cpu().numpy())
                group_hit10[g] += int((hit[10] & yes).sum())
            n += len(y)
    if not n:
        raise ValueError(f"no next events to score in {cfg.evaluate.split}")

    summary = {"split": cfg.evaluate.split, "checkpoint": str(checkpoint), "subjects": int(len(subjects)),
               "predictions": n, "nll": nll / n, "perplexity": float(np.exp(nll / n)),
               **{f"top{k}": hits[k] / n for k in TOP_K}, "by_family": {}, "groups": {}}
    for j, name in enumerate(families, start=1):
        m = float(fam_n[j])
        if m:
            summary["by_family"][name] = {"share": m / n, "nll": float(fam_nll[j]) / m,
                                          "top1": float(fam_hit[1][j]) / m, "top10": float(fam_hit[10][j]) / m}
    for g in groups:
        p, y = np.concatenate(prob[g]), np.concatenate(label[g])
        pos = int(y.sum())
        both = 0 < pos < len(y)
        summary["groups"][g] = {
            "pattern": cfg.evaluate.next_event_groups[g], "tokens": int(len(groups[g])), "positives": pos,
            "prevalence": pos / n, "mean_predicted": float(p.mean()),
            "auroc": float(roc_auc_score(y, p)) if both else None,
            "auprc": float(average_precision_score(y, p)) if both else None,
            "top10_when_next": group_hit10[g] / pos if pos else None}

    out = Path(out_json) if out_json else Path(cfg.train.out_dir) / "next_event.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1))
    tracker = Tracker(cfg, cfg.train.out_dir, "evaluate")
    tracker.summary({k: v for k, v in summary.items() if k not in ("checkpoint", "split")},
                    prefix=f"{cfg.evaluate.split}/next_event/")
    tracker.finish()
    log(f"[next-event] {n:,} next events of {len(subjects):,} subjects: cross-entropy {summary['nll']:.3f} "
        f"(perplexity {summary['perplexity']:.1f}), top-1 {summary['top1']:.3f}, top-5 {summary['top5']:.3f}, "
        f"top-10 {summary['top10']:.3f}")
    for g, s in summary["groups"].items():
        auroc = "n/a" if s["auroc"] is None else f"{s['auroc']:.3f}"
        log(f"[next-event] {g}: {s['positives']:,} next events ({s['prevalence']:.4f}), AUROC {auroc}, "
            f"mean predicted {s['mean_predicted']:.4f}")
    log(f"[next-event] → {out}")
    return summary
