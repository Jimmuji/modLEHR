"""`modlehr finetune`: one binary prediction task on top of a model, and its AUROC.

The task comes as labels in the MEDS label format, such as an ACES task's output (configs/tasks):
<finetune.labels>/<split>/*.parquet, a row per sample with subject_id, prediction_time and boolean_value.
A sample's input is its subject's events at or before the prediction time in the prepared data (the last
max_len of them), and a linear classifier reads the model's final state at the last of them. The model starts
from the checkpoint finetune.init and is trained whole, or only the classifier with finetune.freeze; with
finetune.init: null the same model is trained from scratch, the supervised baseline.

Training draws random training samples. Every train.eval_every steps every tuning sample is scored; best.pt
keeps the classifier with the highest tuning AUROC, and train.patience checks in a row without a new highest
stop the run. The best one then scores every sample of evaluate.split. Saves to train.out_dir: config.yaml,
log.jsonl, best.pt, finetune.json (AUROC, AUPRC and counts per split) and predictions_<split>.parquet in the
MEDS prediction format (subject_id, prediction_time, boolean_value, predicted_boolean_value,
predicted_boolean_probability), which names patients and so stays with the data.

A prediction time becomes an age the way prepare computes event ages: float32 years since MEDS_BIRTH. float32
cannot tell apart times a few minutes from each other (about 4 minutes at age 80), so an event recorded just
after the prediction time can get the same age and would be read as before it. When one of the subject's input
events (a code the data rules select) comes after the prediction time at the same float32 age, the sample's
input stops before that age instead: it loses the events at the prediction time rather than seeing a later one.
The samples of a split, with how many events each reads, are kept in <prepared_dir>/tasks/<labels name>/.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import meds
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

from modlehr.config import Config
from modlehr.data.meds_io import open_meds
from modlehr.evaluate.run import load_model
from modlehr.model.model import ModLEHR
from modlehr.prepare.dataset import PreparedSplit, collate, open_split
from modlehr.prepare.pipeline import comparable_config, meds_fingerprint
from modlehr.prepare.select import SECONDS_PER_YEAR, classify_codes
from modlehr.track import Tracker
from modlehr.train.loop import autocast, build_model, lr_at, make_optimizer, pick_device, to_device

SAMPLE_ARRAYS = ("subject_id", "prediction_time", "label", "index", "n", "strict")


class Classifier(nn.Module):
    """A model's final state at a sample's last event → the log-odds of its label."""

    def __init__(self, model: ModLEHR, prevalence: float, frozen: bool = False):
        super().__init__()
        self.model, self.frozen = model, frozen
        if frozen:
            model.requires_grad_(False)
        self.out = nn.Linear(model.cfg.d_model, 1)
        nn.init.zeros_(self.out.weight)
        with torch.no_grad():  # start at the training prevalence
            self.out.bias.fill_(math.log(prevalence / (1 - prevalence)))

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        h = self.model.hidden(batch)
        last = (batch["tokens"] != 0).sum(1) - 1  # rows are right-padded with token 0
        return self.out(h[torch.arange(len(last), device=h.device), last]).squeeze(-1)

    def train(self, mode: bool = True) -> Classifier:
        super().train(mode)
        if self.frozen:  # no dropout in a model that does not learn
            self.model.eval()
        return self


def read_labels(path: Path) -> pd.DataFrame:
    files = sorted(path.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no labels in {path}: expected .parquet files with subject_id, prediction_time, "
                                "boolean_value (an ACES task's output for this split)")
    labels = pd.concat([pd.read_parquet(f, columns=["subject_id", "prediction_time", "boolean_value"]) for f in files],
                       ignore_index=True)
    labels["prediction_time"] = labels["prediction_time"].astype("datetime64[us]")
    return labels.sort_values(["subject_id", "prediction_time"], kind="stable").reset_index(drop=True)


def task_samples(cfg: Config, split: str, log=print) -> dict[str, np.ndarray]:
    """Per label row of one split: subject_id, prediction_time, label, the subject's index in the prepared split
    (-1 when it has no prepared events), n (how many of its events are the sample's input) and strict (whether
    the input stops before the prediction age, see the module docstring)."""
    labels_dir, prepared = Path(cfg.finetune.labels), Path(cfg.data.prepared_dir)
    cache = prepared / "tasks" / labels_dir.name / f"{split}.npz"
    key = json.dumps({"labels": meds_fingerprint(labels_dir / split),
                      "prepare": hashlib.sha1((prepared / "prepare.json").read_bytes()).hexdigest()})
    if cache.exists():
        with np.load(cache) as z:
            if str(z["key"]) == key:
                return {k: z[k] for k in SAMPLE_ARRAYS}

    labels = read_labels(labels_dir / split)
    ds = open_meds(cfg.data.meds_dir)
    inputs = set(classify_codes(ds.codes, cfg.data.events).index)
    wanted = set(labels["subject_id"])
    birth, after = {}, pd.Series(pd.NaT, index=labels.index, dtype="datetime64[us]")
    # MEDS names shard directories by split; a dataset that does not is read whole
    shards = [p for p in ds.shards() if p.parent.name == split] or ds.shards()
    for shard in shards:
        ev = ds.read_shard(shard, codes=inputs | {meds.birth_code})
        ev = ev.loc[ev["subject_id"].isin(wanted) & ev["time"].notna(), ["subject_id", "time", "code"]]
        if ev.empty:
            continue
        ev["time"] = ev["time"].astype("datetime64[us]")
        birth.update(ev.loc[ev["code"] == meds.birth_code].groupby("subject_id")["time"].min())
        later = ev.loc[ev["code"] != meds.birth_code, ["subject_id", "time"]].sort_values("time")
        rows = labels.loc[labels["subject_id"].isin(set(ev["subject_id"]))].reset_index().sort_values("prediction_time")
        nxt = pd.merge_asof(rows, later, left_on="prediction_time", right_on="time", by="subject_id",
                            direction="forward", allow_exact_matches=False)  # the first input event after it
        after.loc[nxt["index"].to_numpy()] = nxt["time"].to_numpy()

    born = labels["subject_id"].map(birth).astype("datetime64[us]")
    years = lambda t: ((t - born).dt.total_seconds() / SECONDS_PER_YEAR).astype("float32")  # noqa: E731  (as prepare)
    age, next_age = years(labels["prediction_time"]), years(after)
    strict = (next_age <= age).to_numpy()  # NaN (no birth, or no later event) compares False

    split_data = open_split(prepared, split)
    sid = labels["subject_id"].to_numpy(np.int64)
    if len(split_data):
        pos = np.minimum(np.searchsorted(split_data.subject_ids, sid), len(split_data) - 1)
        found = split_data.subject_ids[pos] == sid
    else:
        pos, found = np.zeros(len(sid), np.int64), np.zeros(len(sid), bool)
    index = np.where(found & born.notna().to_numpy(), pos, -1)
    n = np.zeros(len(labels), dtype=np.int64)
    ages = split_data.arrays["ages"]
    for j in np.flatnonzero(index >= 0):
        a, b = split_data.offsets[index[j]], split_data.offsets[index[j] + 1]
        n[j] = np.searchsorted(ages[a:b], age.iat[j], side="left" if strict[j] else "right")
    out = {"subject_id": sid, "prediction_time": labels["prediction_time"].to_numpy(),
           "label": labels["boolean_value"].to_numpy(bool), "index": index.astype(np.int64), "n": n, "strict": strict}
    log(f"[finetune] {split}: {len(sid):,} samples ({int(out['label'].sum()):,} positive) of "
        f"{len(np.unique(sid)):,} subjects; {int((n == 0).sum()):,} without inputs, {int(strict.sum()):,} read up to "
        "just before the prediction time (a later input event has the same float32 age)")
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, key=np.asarray(key), **out)
    return out


def write_samples(cfg: Config, log=print) -> None:
    """task_samples of the training, tuning and evaluation splits, kept in the prepared data for later runs."""
    if not cfg.finetune.labels:
        raise ValueError("finetune.labels is not set: the directory of a task's labels, one subdirectory per split")
    for split in dict.fromkeys((meds.train_split, meds.tuning_split, cfg.evaluate.split)):
        task_samples(cfg, split, log)


def history(split: PreparedSplit, i: int, n: int, max_len: int) -> dict[str, np.ndarray]:
    return split.subject(int(i), max(0, int(n) - max_len), int(n), labels=False)


@torch.no_grad()
def predict(clf: Classifier, split: PreparedSplit, samples: dict, batch_size: int, device: torch.device,
            precision: str, prior: float) -> np.ndarray:
    """Probability of the label for every sample; a sample without inputs gets the training prevalence."""
    max_len = clf.model.cfg.max_len
    p = np.full(len(samples["label"]), prior, dtype=np.float64)
    usable = np.flatnonzero(samples["n"] > 0)
    order = usable[np.argsort(np.minimum(samples["n"][usable], max_len), kind="stable")]  # similar lengths together
    was_training = clf.training
    clf.eval()
    for b in range(0, len(order), batch_size):
        chunk = order[b:b + batch_size]
        rows = [history(split, samples["index"][j], samples["n"][j], max_len) for j in chunk]
        with autocast(device, precision):
            logits = clf(to_device(collate(rows), device))
        p[chunk] = torch.sigmoid(logits.float()).cpu().numpy()
    clf.train(was_training)
    return p


def scores(labels: np.ndarray, p: np.ndarray) -> dict:
    both = 0 < labels.sum() < len(labels)
    return {"auroc": float(roc_auc_score(labels, p)) if both else None,
            "auprc": float(average_precision_score(labels, p)) if both else None,
            "samples": int(len(labels)), "positives": int(labels.sum())}


def start_model(cfg: Config, prepared: Path, log=print) -> ModLEHR:
    """The pretrained model of finetune.init (its own model config), or a new one of cfg.model."""
    if cfg.finetune.init is None:
        model = build_model(cfg, prepared)[0]
        log(f"[finetune] from scratch: {model.n_parameters() / 1e6:.2f}M parameters")
        return model
    init = Path(cfg.finetune.init)
    stored = init.parent / "config.yaml"
    if stored.exists():  # the pretrained model must have read the same prepared data
        theirs = {k: v for k, v in ((yaml.safe_load(stored.read_text()) or {}).get("data") or {}).items()
                  if k != "prepared_dir"}
        mine = {k: v for k, v in comparable_config(cfg.data, theirs).items() if k != "prepared_dir"}
        if mine != theirs:
            raise ValueError(f"{init} was trained on other data than this config's: data.{', data.'.join(sorted(
                k for k in set(mine) | set(theirs) if mine.get(k) != theirs.get(k)))} differ")
    model = load_model(init, prepared, torch.device("cpu"))[0]
    if not isinstance(model, ModLEHR):
        raise ValueError(f"{init} is a Delphi-2M checkpoint; fine-tuning takes a modLEHR model")
    log(f"[finetune] from {init}: {model.n_parameters() / 1e6:.2f}M parameters"
        + (", only the classifier trained" if cfg.finetune.freeze else ""))
    return model.train()


def finetune(cfg: Config, log=print) -> dict:
    ft, tc = cfg.finetune, cfg.train
    if not ft.labels:
        raise ValueError("finetune.labels is not set: the directory of a task's labels, one subdirectory per split")
    if ft.freeze and ft.init is None:
        raise ValueError("finetune.freeze trains only the classifier, so it needs a pretrained model in finetune.init")
    prepared, out_dir = Path(cfg.data.prepared_dir), Path(tc.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
    torch.manual_seed(tc.seed)
    rng = np.random.default_rng(tc.seed)
    device = pick_device(tc.device)

    model = start_model(cfg, prepared, log)
    names = tuple(dict.fromkeys((meds.train_split, meds.tuning_split, cfg.evaluate.split)))
    splits = {s: open_split(prepared, s) for s in names}
    samples = {s: task_samples(cfg, s, log) for s in names}
    if len(np.unique(samples[meds.tuning_split]["label"])) < 2:
        raise ValueError(f"the tuning labels in {ft.labels} are all one class, so there is no AUROC to pick the best "
                         "model by")
    train_s = samples[meds.train_split]
    usable = np.flatnonzero(train_s["n"] > 0)
    prevalence = float(train_s["label"][usable].mean())
    clf = Classifier(model, prevalence, frozen=ft.freeze).to(device).train()
    opt = make_optimizer(clf, tc, trainable=lambda n: n.startswith("out.") or not (ft.freeze or n.startswith("model.head.")))
    if tc.compile:
        clf.model.compile()
    tracker = Tracker(cfg, out_dir, "finetune")
    max_len = clf.model.cfg.max_len
    tune = (splits[meds.tuning_split], samples[meds.tuning_split])

    best, since_best, best_step, losses = -1.0, 0, -1, []
    t0 = time.time()
    with open(out_dir / "log.jsonl", "w") as logf:
        for step in range(tc.steps):
            for group in opt.param_groups:
                group["lr"] = lr_at(step, tc)
            opt.zero_grad(set_to_none=True)
            for _ in range(tc.grad_accum):
                pick = usable[rng.integers(0, len(usable), tc.batch_size)]
                rows = [history(splits[meds.train_split], train_s["index"][j], train_s["n"][j], max_len) for j in pick]
                y = torch.from_numpy(train_s["label"][pick].astype(np.float32)).to(device)
                with autocast(device, tc.precision):
                    logits = clf(to_device(collate(rows), device))
                loss = F.binary_cross_entropy_with_logits(logits.float(), y)
                (loss / tc.grad_accum).backward()
                losses.append(float(loss.detach()))
            torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], tc.grad_clip)
            opt.step()

            last = step == tc.steps - 1
            if step % tc.eval_every == 0 or last:
                auroc = scores(tune[1]["label"], predict(clf, *tune, cfg.evaluate.batch_size, device, tc.precision,
                                                         prevalence))["auroc"]
                record = {"step": step, "train_loss": float(np.mean(losses)), "tune_auroc": auroc,
                          "seconds": round(time.time() - t0, 1)}
                losses = []
                logf.write(json.dumps(record) + "\n")
                logf.flush()
                log(f"[finetune] step {step:6d}  train loss {record['train_loss']:.4f}  tuning AUROC "
                    f"{auroc if auroc is None else round(auroc, 4)}  ({record['seconds']}s)")
                tracker.log({k: v for k, v in [("train_loss", record["train_loss"]), ("tune_auroc", auroc),
                                                ("lr", opt.param_groups[0]["lr"])] if v is not None}, step)
                if auroc is not None and auroc > best:
                    best, since_best, best_step = auroc, 0, step
                    torch.save({"classifier": clf.state_dict(), "model_config": clf.model.cfg.__dict__, "step": step,
                                "tune_auroc": auroc, "prevalence": prevalence}, out_dir / "best.pt")
                else:
                    since_best += 1
                if tc.patience is not None and since_best >= tc.patience and not last:
                    log(f"[finetune] no new highest tuning AUROC in {since_best} checks: stopping after step {step}")
                    break

    if (out_dir / "best.pt").exists():
        clf.load_state_dict(torch.load(out_dir / "best.pt", map_location=device, weights_only=False)["classifier"])
    split = cfg.evaluate.split
    p = predict(clf, splits[split], samples[split], cfg.evaluate.batch_size, device, tc.precision, prevalence)
    s = samples[split]
    pd.DataFrame({"subject_id": s["subject_id"], "prediction_time": s["prediction_time"], "boolean_value": s["label"],
                  "predicted_boolean_value": p >= 0.5, "predicted_boolean_probability": p}) \
        .to_parquet(out_dir / f"predictions_{split}.parquet", index=False)
    result = {"labels": str(ft.labels), "init": ft.init, "freeze": ft.freeze, "best_step": best_step,
              "tune_auroc": best if best_step >= 0 else None, "train_prevalence": prevalence,
              split: {**scores(s["label"], p), "without_inputs": int((s["n"] == 0).sum()), "strict": int(s["strict"].sum())}}
    (out_dir / "finetune.json").write_text(json.dumps(result, indent=1))
    log(f"[finetune] {split} AUROC {result[split]['auroc']}, AUPRC {result[split]['auprc']} over "
        f"{result[split]['samples']:,} samples (best tuning AUROC {result['tune_auroc']} at step {best_step}) → {out_dir}")
    tracker.summary(result)
    tracker.finish()
    return result
