"""`modlehr generate`: sample future target events for a patient, one step at a time.

From the history up to a start age, the head gives each target a rate; the next event and the
wait until it are drawn from that competing-risks distribution, the event is appended, and
the model runs again, until the horizon is passed or death is drawn.

Targets are first occurrences by default (data rule role: target), so a target already in the
history, or already sampled, is not drawn again: its rate is set to zero. Pass
exclude_seen=False for target rules that keep repeated events.
"""

from __future__ import annotations

import numpy as np
import torch

from modlehr.model.model import ModLEHR
from modlehr.prepare.dataset import collate
from modlehr.prepare.vocabulary import Vocabulary


@torch.no_grad()
def sample_future(model: ModLEHR, vocab: Vocabulary, history: dict[str, np.ndarray], start_age: float,
                  horizon_years: float, max_events: int = 50, seed: int = 0, device: str | torch.device = "cpu",
                  exclude_seen: bool = True) -> list[tuple[float, str]]:
    keep = history["ages"] <= start_age
    tokens = list(history["tokens"][keep])
    ages = list(history["ages"][keep])
    values = list(history["values"][keep])
    gen = torch.Generator(device="cpu").manual_seed(seed)
    death = vocab.token_index.get("MEDS_DEATH")
    seen = {int(t) for t in vocab.input_to_target[np.asarray(tokens, dtype=np.int64)] if t >= 0} if exclude_seen else set()
    age, out = start_age, []
    for _ in range(max_events):
        row = {k: np.asarray(v[-model.cfg.max_len:]) for k, v in
               (("tokens", tokens), ("ages", ages), ("values", values))}
        n = len(row["tokens"])
        row.update(next_target=np.full(n, -1), dt=np.zeros(n, np.float32), loss_mask=np.zeros(n, bool))
        batch = {k: v.to(device) for k, v in collate([row]).items()}
        log_rates = model(batch)[0, -1].float().cpu()
        if seen:
            log_rates[list(seen)] = float("-inf")
        k, wait = model.head.sample(log_rates.unsqueeze(0), generator=gen)
        age = age + float(wait)
        if age > start_age + horizon_years:
            break
        token = int(vocab.target_to_input[int(k)])
        out.append((round(age, 2), vocab.tokens[token]))
        if exclude_seen:
            seen.add(int(k))
        tokens.append(token)
        ages.append(np.float32(age))
        values.append(np.float32("nan"))
        if token == death:
            break
    return out
