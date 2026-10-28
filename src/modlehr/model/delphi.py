"""Delphi-2M (Shmatko et al., Nature 2025), as a modLEHR model.

A port of `model.py` from github.com/gerstung-lab/Delphi at commit fb72166 (MIT License, Gerstung Lab):
the same modules, names and initialization, so a checkpoint of either loads into the other, and with
the same seed the two train the same model. `model.architecture: delphi2m` trains it with Delphi's own
batches and schedule (modlehr.train.delphi); `DelphiModel` then scores a checkpoint with modLEHR's
evaluation, whether this code or Delphi's own `train.py` wrote it.

Delphi works in its own ids and in days. Id 0 is "no event", 1 and 2 are female and male, 3 to 11 are
lifestyle tokens whose ages its training shifts at random, and the model sees every id shifted by one,
with 0 for padding. `delphi_ids` maps a modLEHR vocabulary onto these ids.

    tokens, ages in days → token embedding + age encoding → GPT-2 blocks → final norm → logits
    logits: a log-rate per day for every token; the next token by softmax, the wait by an exponential
"""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from modlehr.evaluate.delphi import SEX_TOKENS
from modlehr.model.heads import CompetingRiskHead
from modlehr.prepare.vocabulary import Vocabulary

DAYS_PER_YEAR = 365.25
NO_EVENT = 0                 # Delphi's id for "no event"
FEMALE, MALE = 1, 2
LIFESTYLE = range(3, 12)     # ids whose ages Delphi's training shifts at random
MASK_TIME = -10000.0         # Delphi's age for padding


def new_gelu(x):
    """GELU, tanh approximation (GPT-2)."""
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class LayerNorm(nn.Module):
    """LayerNorm with an optional bias."""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # a buffer of Delphi's, unused in its forward; kept so checkpoints match key for key
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))

    def forward(self, x, attn_mask):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(attn_mask == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(new_gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, attn_mask):
        x = x + self.attn(self.ln_1(x), attn_mask)
        return x + self.mlp(self.ln_2(x))


class AgeEncoding(nn.Module):
    """Sinusoidal encoding of age (in days, read in years), then a linear map."""

    def __init__(self, config):
        super().__init__()
        div_term = torch.exp(torch.arange(0, config.n_embd, 2) * (-math.log(10000.0) / config.n_embd))
        self.register_buffer("div_term", div_term)
        self.n_embd = config.n_embd
        self.linear = torch.nn.Linear(config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        y = torch.zeros(x.shape[0], x.shape[1], self.n_embd, device=x.device)
        y[..., 0::2] = torch.sin(x / 365.25 * self.div_term)
        y[..., 1::2] = torch.cos(x / 365.25 * self.div_term)
        return self.linear(y)


@dataclass
class DelphiConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    token_dropout: float = 0.0
    t_min: float = 1.0           # days added to every wait
    bias: bool = True
    mask_ties: bool = False      # tokens at the same age do not attend to each other
    ignore_tokens: list = field(default_factory=lambda: [0])   # left out of the loss


class Delphi(nn.Module):
    def __init__(self, config: DelphiConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wae=AgeEncoding(config),
            token_drop=nn.Dropout(config.token_dropout),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():  # scaled init of the residual projections, as in GPT-2
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, age, targets=None, targets_age=None, validation_loss_mode=False):
        """Logits (log-rates per day) at every token, and with targets the two losses: cross-entropy
        of the next token, and the exponential likelihood of the wait until it."""
        device = idx.device
        x = self.transformer.token_drop(self.transformer.wte(idx))
        x = self.transformer.drop(x + self.transformer.wae(age.unsqueeze(-1)))

        n, t = idx.shape
        causal = torch.tril(torch.ones(t, t, device=device))[None, None, :, :] > 0
        attn_mask = (idx > 0).view(n, 1, 1, t) * (idx > 0).view(n, 1, t, 1)  # no attention to padding
        attn_mask *= causal
        if targets is not None and self.config.mask_ties:
            attn_mask *= age.view(n, 1, 1, t) != targets_age.view(n, 1, t, 1)  # nor to tokens of the same age
            attn_mask += (attn_mask.sum(-1, keepdim=True) == 0) * torch.diag(torch.ones(t, device=device)) > 0
        attn_mask = attn_mask + (idx == 0).view(n, 1, 1, t) * torch.diag(torch.ones(t, device=device)) > 0
        attn_mask *= causal

        for block in self.transformer.h:
            x = block(x, attn_mask)
        logits = self.lm_head(self.transformer.ln_f(x))
        if targets is None:
            return logits, None

        ignored_tokens = self.config.ignore_tokens.copy()
        if validation_loss_mode:
            ignored_tokens += [1]  # "no event" is not scored on the tuning split
            logits[..., ignored_tokens] = -torch.inf
        targets = targets.reshape(-1)
        pass_tokens = targets != -1
        for k in ignored_tokens:
            pass_tokens *= targets != k
        loss_ce = F.cross_entropy(logits.reshape(-1, logits.size(-1))[pass_tokens], targets[pass_tokens], ignore_index=-1)

        lse = torch.logsumexp(logits, -1)
        lse = -torch.log(torch.exp(-lse) + self.config.t_min)
        dt = torch.clamp(targets_age - age, min=1.0)
        if self.config.mask_ties:  # the wait counts from the last token of an earlier age
            dt = torch.gather(dt, -1, (attn_mask * torch.arange(0, t, device=device, dtype=torch.float32)
                                       .view(1, 1, 1, -1)).max(-1).indices.squeeze((1, 2)))
        ldt = -torch.log(dt + self.config.t_min).view(-1)
        loss_dt = -(lse.reshape(-1) - torch.exp(lse.reshape(-1) - ldt.reshape(-1)))
        loss_dt = torch.mean(loss_dt[pass_tokens])
        return logits, {"loss_ce": loss_ce, "loss_dt": loss_dt}

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """AdamW with weight decay on the weights of linear layers only, as Delphi's."""
        decay, no_decay = set(), set()
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn
                if pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, torch.nn.Linear):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, (torch.nn.LayerNorm, LayerNorm, torch.nn.Embedding)):
                    no_decay.add(fpn)
        decay.remove("lm_head.weight")  # tied to the embedding, which is not decayed
        param_dict = dict(self.named_parameters())
        assert not decay & no_decay and not param_dict.keys() - (decay | no_decay)
        groups = [{"params": [param_dict[pn] for pn in sorted(decay)], "weight_decay": weight_decay},
                  {"params": [param_dict[pn] for pn in sorted(no_decay)], "weight_decay": 0.0}]
        fused = device_type == "cuda" and "fused" in inspect.signature(torch.optim.AdamW).parameters
        return torch.optim.AdamW(groups, lr=learning_rate, betas=betas, **({"fused": True} if fused else {}))


def delphi_ids(vocab: Vocabulary) -> np.ndarray:
    """Delphi id of every modLEHR input token (-1 for padding): sex, then the inputs that are not
    targets, then the targets in modLEHR's order. The same vocabulary always gives the same ids."""
    ids = np.full(len(vocab.tokens), -1, dtype=np.int64)
    for token, delphi_id in ((SEX_TOKENS["female"], FEMALE), (SEX_TOKENS["male"], MALE)):
        if token in vocab.token_index:
            ids[vocab.token_index[token]] = delphi_id
    is_target = vocab.input_to_target >= 0
    others = [i for i in range(1, len(vocab.tokens)) if ids[i] < 0 and not is_target[i]]
    ids[others] = LIFESTYLE.stop + np.arange(len(others))
    ids[vocab.target_to_input] = LIFESTYLE.stop + len(others) + np.arange(len(vocab.targets))
    return ids


def is_delphi_checkpoint(state: dict) -> bool:
    return "model_args" in state and "model_config" not in state


class DelphiModel(nn.Module):
    """A Delphi-2M checkpoint read like a modLEHR model: forward(batch) gives the log-rates per year
    of the modLEHR targets at every event. At a landmark it is asked with a "no event" token at the
    landmark age after the history, the way Delphi is asked about a time with no event."""

    def __init__(self, state: dict, vocab: Vocabulary):
        super().__init__()
        self.delphi = Delphi(DelphiConfig(**state["model_args"]))
        self.delphi.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in state["model"].items()})
        ids = delphi_ids(vocab)
        if ids.max() + 2 > state["model_args"]["vocab_size"]:
            raise ValueError("the checkpoint's vocabulary is smaller than this prepared dataset's; "
                             "was it trained on different prepared data?")
        self.query_token = len(vocab.tokens)  # an input id beyond the vocabulary: Delphi's "no event"
        lookup = np.r_[ids + 1, NO_EVENT + 1]
        lookup[0] = 0  # padding
        self.register_buffer("lookup", torch.as_tensor(lookup), persistent=False)
        self.register_buffer("columns", torch.as_tensor(ids[vocab.target_to_input] + 1), persistent=False)
        self.cfg = SimpleNamespace(max_len=int(state["model_args"]["block_size"]))
        self.head = CompetingRiskHead  # loss, risk, observed_risk and sample work on any log-rates per year

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        tokens = batch["tokens"]
        idx = self.lookup[tokens]
        age = torch.where(tokens != 0, batch["ages"] * DAYS_PER_YEAR, torch.full_like(batch["ages"], MASK_TIME))
        c = self.delphi.config  # one layer's attention weights at a time; keep them under about 1 GB
        rows = max(1, 2**28 // (c.n_head * tokens.shape[1] ** 2))
        logits = torch.cat([self.delphi(idx[i:i + rows], age[i:i + rows])[0] for i in range(0, len(idx), rows)])
        return logits[..., self.columns].float() + math.log(DAYS_PER_YEAR)  # its log-rates are per day
