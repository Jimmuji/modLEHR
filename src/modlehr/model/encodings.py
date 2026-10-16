"""Time and value encodings for the input head.

Time: the model works in age (years). The default is a sinusoidal encoding of age followed by
a linear layer, the way Delphi encodes age in place of token positions.

Value: how a measured number enters the model. "none" ignores it (values can still enter as
binned tokens, see data.value_mode); "mlp" maps the scaled value through a small network and
adds it to the token embedding.

Text: how the words of a text event (a clinical note) enter. "none" ignores them (the note still
enters as its code); "hashed_bag" averages a learned embedding of each hashed word and adds it to
that event's embedding. Called as enc(word_ids, starts) with every note's words in one flat tensor.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from modlehr.registry import register


@register("time_encoding", "sinusoidal_age")
class SinusoidalAge(nn.Module):
    def __init__(self, d_model: int, max_period: float = 10000.0):
        super().__init__()
        if d_model % 2:
            raise ValueError("sinusoidal_age needs an even d_model")
        freqs = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(max_period) / d_model))
        self.register_buffer("freqs", freqs, persistent=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, ages: torch.Tensor) -> torch.Tensor:
        x = ages.unsqueeze(-1) * self.freqs
        enc = torch.stack((torch.sin(x), torch.cos(x)), dim=-1).flatten(-2)
        return self.proj(enc)


@register("time_encoding", "none")
class NoTime(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(self, ages: torch.Tensor) -> torch.Tensor:
        return ages.new_zeros(*ages.shape, self.d_model)


@register("value_encoding", "none")
class NoValue(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(self, values: torch.Tensor, has_value: torch.Tensor) -> torch.Tensor:
        return values.new_zeros(*values.shape, self.d_model)


@register("value_encoding", "mlp")
class MLPValue(nn.Module):
    """m_i * F(z_i): F is a small network shared by all measurements, m_i is 1 when the event has a value."""

    def __init__(self, d_model: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, hidden), nn.GELU(), nn.Linear(hidden, d_model))

    def forward(self, values: torch.Tensor, has_value: torch.Tensor) -> torch.Tensor:
        return self.net(values.unsqueeze(-1)) * has_value.unsqueeze(-1).to(values.dtype)


@register("text_encoding", "none")
class NoText(nn.Module):
    def __init__(self, d_model: int, n_buckets: int = 0):
        super().__init__()

    def forward(self, word_ids: torch.Tensor, starts: torch.Tensor) -> None:
        return None


@register("text_encoding", "hashed_bag")
class HashedBag(nn.Module):
    """Mean of learned word embeddings over a note's hashed words (an EmbeddingBag)."""

    def __init__(self, d_model: int, n_buckets: int = 0):
        super().__init__()
        if n_buckets <= 0:
            raise ValueError("text_encoding hashed_bag needs prepared data with text events (data rule role: text)")
        self.bag = nn.EmbeddingBag(n_buckets, d_model, mode="mean")

    def forward(self, word_ids: torch.Tensor, starts: torch.Tensor) -> torch.Tensor:
        return self.bag(word_ids, starts)
