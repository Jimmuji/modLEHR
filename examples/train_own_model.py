"""A model written without modLEHR, trained on modLEHR's prepared data: a small causal transformer from plain
PyTorch, pretrained on the next token, then a linear probe of its last state on one binary task.

It shows the whole contract in one page: windows of tokens for pretraining, (index, n) samples for a task. Put your
own architecture in place of Model; the data and the task samples stay the same.

    python examples/train_own_model.py runs/prepared/mimic_iv_10pct --task admission_icu --steps 2000
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from prepared_data import PreparedData  # noqa: E402


class Model(nn.Module):
    def __init__(self, n_tokens: int, d: int = 128, layers: int = 4, heads: int = 4, max_len: int = 1024):
        super().__init__()
        self.embed = nn.Embedding(n_tokens, d, padding_idx=0)
        self.pos = nn.Embedding(max_len, d)
        block = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.body = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, n_tokens)

    def states(self, tokens: torch.Tensor) -> torch.Tensor:
        length = tokens.shape[1]
        causal = torch.triu(torch.full((length, length), float("-inf"), device=tokens.device), 1)
        x = self.embed(tokens) + self.pos(torch.arange(length, device=tokens.device))
        return self.norm(self.body(x, mask=causal, is_causal=True))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("prepared_dir")
    ap.add_argument("--task", help="a task under <prepared_dir>/tasks for the probe")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--probe-samples", type=int, default=4000, help="per split, at most")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    data = PreparedData(args.prepared_dir)
    model = Model(len(data.tokens), max_len=args.max_len).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    lr = lambda step: min(1.0, (step + 1) / 100) * 0.5 * (1 + math.cos(math.pi * step / args.steps))  # noqa: E731
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr)

    def loss_of(batch: dict) -> torch.Tensor:
        tokens = torch.from_numpy(batch["tokens"]).to(device)
        logits = model.out(model.states(tokens[:, :-1]))
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens[:, 1:].reshape(-1), ignore_index=0)

    train, tuning = data.split("train"), data.split("tuning")
    fixed = [tuning.windows(args.batch_size, args.max_len, np.random.default_rng(1)) for _ in range(10)]
    for step in range(args.steps):
        loss = loss_of(train.windows(args.batch_size, args.max_len, rng))
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % max(1, args.steps // 5) == 0 or step == args.steps - 1:
            with torch.no_grad():
                model.eval()
                tune = float(np.mean([loss_of(b).item() for b in fixed]))
                model.train()
            print(f"step {step:>6}  train loss {loss.item():.3f}  tuning loss {tune:.3f}")

    if not args.task:
        return
    model.eval()

    @torch.no_grad()
    def features(split: str) -> tuple[np.ndarray, np.ndarray]:
        task = data.task(args.task, split)
        rows = np.flatnonzero(task["n"] > 0)  # samples with events before the prediction time
        rows = rows[rng.permutation(len(rows))[:args.probe_samples]]
        out, labels = [], []
        for chunk in np.array_split(rows, max(1, len(rows) // 64)):
            batch = data.task_inputs(task, split, args.max_len, chunk)
            h = model.states(torch.from_numpy(batch["tokens"]).to(device))
            last = torch.from_numpy(batch["mask"].sum(1) - 1).to(device)
            out.append(h[torch.arange(len(chunk), device=device), last].float().cpu().numpy())
            labels.append(batch["label"])
        return np.concatenate(out), np.concatenate(labels)

    x_train, y_train = features("train")
    x_test, y_test = features("held_out")
    probe = LogisticRegression(max_iter=2000).fit(x_train, y_train)
    print(f"{args.task}: linear probe AUROC {roc_auc_score(y_test, probe.predict_proba(x_test)[:, 1]):.3f} on "
          f"{len(y_test):,} held-out samples ({int(y_test.sum()):,} positive)")


if __name__ == "__main__":
    main()
