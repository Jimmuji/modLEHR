"""Swappable transformer parts: attention, normalization, feed-forward.

The baseline is a plain GPT block: pre-norm LayerNorm, multi-head causal attention, GELU MLP.
New parts from recent LLMs are added here (or in a new module) and registered by name.
Attention modules are called as attn(x, mask, ages=ages); modules that do not use ages ignore them.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from modlehr.registry import register


class Attention(nn.Module):
    """Causal self-attention with options that cover several recent designs.

    n_kv_heads < n_heads gives grouped-query attention (query heads share key/value heads).
    rope="index" rotates queries and keys by event position; rope="age" rotates them by the
    patient's age at each event, so attention depends on how many years apart two events are,
    not on how many events lie between them. age_bias=True adds ALiBi's linear penalty, but on
    years rather than positions: head h subtracts slope_h * (years between query and key) from
    its scores, so each head looks back over a different time span.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, n_kv_heads: int | None = None,
                 rope: str | None = None, rope_base: float = 10000.0, age_scale: float = 1.0,
                 age_bias: bool = False, max_slope_exponent: float = 8.0):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads, self.n_kv = n_heads, n_kv_heads or n_heads
        if n_heads % self.n_kv:
            raise ValueError("n_heads must be a multiple of n_kv_heads")
        self.head_dim = d_model // n_heads
        if rope not in (None, "index", "age"):
            raise ValueError("rope must be None, 'index' or 'age'")
        if rope and self.head_dim % 2:
            raise ValueError("rope needs an even head dimension")
        self.rope, self.age_scale = rope, age_scale
        self.q = nn.Linear(d_model, n_heads * self.head_dim)
        self.kv = nn.Linear(d_model, 2 * self.n_kv * self.head_dim)
        self.out = nn.Linear(n_heads * self.head_dim, d_model)
        self.dropout = dropout
        inv_freq = rope_base ** (-torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.age_bias = age_bias
        # ALiBi slopes: a geometric sequence from 2^(-8/n) down to 2^-8, one per head (Press et al., 2022)
        slopes = 2.0 ** (-max_slope_exponent * torch.arange(1, n_heads + 1, dtype=torch.float32) / n_heads)
        self.register_buffer("slopes", slopes.view(1, n_heads, 1, 1), persistent=False)

    def _rotate(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """x: [B, H, T, D]; positions: [B, T]. Rotates pairs of dimensions by position * frequency."""
        angle = positions.unsqueeze(1).unsqueeze(-1) * self.inv_freq  # [B, 1, T, D/2]
        cos, sin = angle.cos(), angle.sin()
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, ages: torch.Tensor | None = None) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.q(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k, v = self.kv(x).view(b, t, 2, self.n_kv, self.head_dim).permute(2, 0, 3, 1, 4)
        if self.rope:
            if self.rope == "age":
                if ages is None:
                    raise ValueError("rope='age' needs ages")
                positions = ages * self.age_scale
            else:
                positions = torch.arange(t, device=x.device, dtype=x.dtype).expand(b, t)
            q, k = self._rotate(q, positions), self._rotate(k, positions)
        if self.n_kv < self.n_heads:
            k = k.repeat_interleave(self.n_heads // self.n_kv, dim=1)
            v = v.repeat_interleave(self.n_heads // self.n_kv, dim=1)
        attn_mask = mask
        if self.age_bias:
            if ages is None:
                raise ValueError("age_bias needs ages")
            gap = (ages.unsqueeze(-1) - ages.unsqueeze(-2)).clamp(min=0.0).unsqueeze(1) * self.age_scale  # [B, 1, T, T]
            attn_mask = (-self.slopes * gap).to(q.dtype).masked_fill(~mask, float("-inf"))
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0)
        return self.out(y.transpose(1, 2).reshape(b, t, -1))


@register("attention", "mha")
def mha(d_model: int, n_heads: int, dropout: float = 0.0, **options) -> Attention:
    """Plain multi-head attention, as in GPT-2."""
    return Attention(d_model, n_heads, dropout, **options)


@register("attention", "gqa")
def gqa(d_model: int, n_heads: int, dropout: float = 0.0, n_kv_heads: int | None = None, **options) -> Attention:
    """Grouped-query attention (Llama 2/3, Qwen): n_kv_heads key/value heads, default half the query heads."""
    return Attention(d_model, n_heads, dropout, n_kv_heads=n_kv_heads or max(1, n_heads // 2), **options)


@register("attention", "rope")
def rope_index(d_model: int, n_heads: int, dropout: float = 0.0, **options) -> Attention:
    """Rotary position embedding over event order."""
    return Attention(d_model, n_heads, dropout, rope="index", **options)


@register("attention", "rope_age")
def rope_age(d_model: int, n_heads: int, dropout: float = 0.0, **options) -> Attention:
    """Rotary embedding over age: attention sees the years between events."""
    return Attention(d_model, n_heads, dropout, rope="age", **options)


@register("attention", "alibi_age")
def alibi_age(d_model: int, n_heads: int, dropout: float = 0.0, **options) -> Attention:
    """ALiBi over age: a per-head linear penalty on the years between events (BLOOM and MPT use ALiBi over positions)."""
    return Attention(d_model, n_heads, dropout, age_bias=True, **options)


@register("norm", "layernorm")
def layernorm(d_model: int) -> nn.Module:
    return nn.LayerNorm(d_model)


@register("norm", "rmsnorm")
def rmsnorm(d_model: int) -> nn.Module:
    return nn.RMSNorm(d_model)


@register("ffn", "gelu_mlp")
class GeluMLP(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0, mult: int = 4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, mult * d_model), nn.GELU(), nn.Linear(mult * d_model, d_model), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@register("ffn", "swiglu")
class SwiGLU(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0, mult: float = 8 / 3):
        super().__init__()
        hidden = int(mult * d_model)
        self.gate = nn.Linear(d_model, hidden, bias=False)
        self.up = nn.Linear(d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))
