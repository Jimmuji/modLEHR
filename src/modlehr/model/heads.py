"""Output heads.

competing_risk: one log-rate per target (per year). Between events each target k has a
constant hazard lambda_k; the next target event is k after dt years with likelihood
lambda_k * exp(-Lambda * dt), Lambda = sum_k lambda_k, and a censored stretch of dt years has
likelihood exp(-Lambda * dt). This is Delphi's next-token plus waiting-time loss written as one
competing-risks likelihood, with censoring at the end of follow-up handled directly. Targets recorded
at the same time (a set S, e.g. the diagnoses of one discharge) have likelihood
prod_{k in S} lambda_k * exp(-Lambda * dt), the same hazards observed at a recorded resolution;
with one target it is the formula above.

next_token: one logit per token, trained on which event comes next in the record, whatever its role
(a lab result, a drug, an ICU admission, a diagnosis), as ETHOS, PORTER and MEDS-EIC-AR pretrain. It
predicts no waiting time and gives no risk within a horizon, so a model trained with it is scored on a
task with `modlehr finetune`, not with `modlehr evaluate`.

A head is built with d_model, n_tokens and n_targets and sizes its output by the one it predicts.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from modlehr.registry import register


@register("head", "competing_risk")
class CompetingRiskHead(nn.Module):
    def __init__(self, d_model: int, n_targets: int, n_tokens: int | None = None):
        super().__init__()
        self.proj = nn.Linear(d_model, n_targets)

    def init_rates(self, counts: list[int], exposure_years: float) -> None:
        """Start each log-rate at its training-set rate, so the first steps are not spent on scale."""
        rates = torch.tensor([(c + 1.0) / max(exposure_years, 1.0) for c in counts])
        with torch.no_grad():
            self.proj.weight.zero_()
            self.proj.bias.copy_(rates.log())

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.proj(h)

    @staticmethod
    def loss(log_rates: torch.Tensor, next_target: torch.Tensor, dt: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Mean negative log-likelihood over masked positions. next_target is [B, T] (one target, -1 for
        censored) or [B, T, K] (every target of the next target time, -1 padded; all -1 for censored)."""
        log_total = torch.logsumexp(log_rates, dim=-1)
        targets = next_target if next_target.dim() == log_rates.dim() else next_target.unsqueeze(-1)
        present = (targets >= 0).to(log_rates.dtype)
        log_rate_k = log_rates.gather(-1, targets.clamp(min=0))
        nll_event = -(log_rate_k * present).sum(dim=-1)
        nll_time = log_total.exp() * dt
        m = mask.to(log_rates.dtype)
        n = m.sum().clamp(min=1.0)
        nll = ((nll_event + nll_time) * m).sum() / n
        return nll, {"nll_event": float((nll_event * m).sum().detach() / n), "nll_time": float((nll_time * m).sum().detach() / n)}

    @classmethod
    def batch_loss(cls, log_rates: torch.Tensor, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict]:
        """loss on a batch as the dataset makes it, with every target of the next target time when it has them."""
        return cls.loss(log_rates, batch.get("next_set", batch["next_target"]), batch["dt"], batch["loss_mask"])

    @staticmethod
    def risk(log_rates: torch.Tensor, horizon_years: float, kind: str = "marginal") -> torch.Tensor:
        """Probability of each target within the horizon, holding the hazards constant.

        marginal     P(k occurs within the horizon) = 1 - exp(-lambda_k * H): each target on its own
                     hazard, which is what the loss fits when targets share their time, and what the
                     landmark task asks (k first occurs in the window, whether or not something else
                     comes first). Ranks subjects by lambda_k.
        next_event   P(k is the next target and comes within the horizon)
                     = (lambda_k / Lambda)(1 - exp(-Lambda * H)): one target at a time. With tied
                     targets Lambda counts every target of a discharge, so this understates each k.
        """
        if kind == "marginal":
            return -torch.expm1(-log_rates.exp() * horizon_years)
        if kind != "next_event":
            raise ValueError(f"risk must be marginal or next_event, not {kind!r}")
        log_total = torch.logsumexp(log_rates, dim=-1, keepdim=True)
        share = (log_rates - log_total).exp()
        p_any = 1.0 - torch.exp(-log_total.exp() * horizon_years)
        return share * p_any

    @staticmethod
    def observed_risk(log_rates: torch.Tensor, horizon_years: float, death: int | None = None) -> torch.Tensor:
        """Probability that each target is recorded within the horizon, which calibration compares with
        the records. Nothing is recorded after death, so k must come first:
        (lambda_k / (lambda_k + lambda_d))(1 - exp(-(lambda_k + lambda_d) H)), with lambda_d death's rate;
        death itself has 1 - exp(-lambda_d H). Without a death target it is the marginal risk."""
        rates = log_rates.exp()
        if death is None:
            return -torch.expm1(-rates * horizon_years)
        with_death = rates + rates[..., death:death + 1]
        out = rates / with_death * -torch.expm1(-with_death * horizon_years)
        out[..., death] = -torch.expm1(-rates[..., death] * horizon_years)
        return out

    @staticmethod
    def sample(log_rates: torch.Tensor, generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw (target, years until it) from the competing-risks distribution."""
        log_total = torch.logsumexp(log_rates, dim=-1)
        u = torch.rand(log_total.shape, generator=generator, device=log_rates.device).clamp(min=1e-12)
        wait = -torch.log(u) / log_total.exp()
        k = torch.multinomial(torch.softmax(log_rates, dim=-1), 1, generator=generator).squeeze(-1)
        return k, wait


@register("head", "next_token")
class NextTokenHead(nn.Module):
    def __init__(self, d_model: int, n_tokens: int, n_targets: int | None = None):
        super().__init__()
        self.proj = nn.Linear(d_model, n_tokens)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.proj(h)

    @staticmethod
    def batch_loss(logits: torch.Tensor, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict]:
        """Mean cross-entropy of the token that follows each event in its row. The last event of a row has
        none in the window, and padding (token 0) is never a next token, so neither is counted."""
        tokens = batch["tokens"]
        following = torch.cat([tokens[:, 1:], tokens.new_zeros(tokens.shape[0], 1)], dim=1)  # 0: nothing follows
        n = (following != 0).sum().clamp(min=1)
        nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), following.reshape(-1), ignore_index=0,
                              reduction="sum") / n
        with torch.no_grad():
            right = ((logits.argmax(dim=-1) == following) & (following != 0)).sum()
        return nll, {"accuracy": float(right / n)}
