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

    qk_norm=True normalizes queries and keys (RMSNorm over each head) before their dot product, as
    in OLMo 2, Gemma 3 and Qwen3. kv_rank=r computes keys and values from an r-dimensional latent,
    as multi-head latent attention does in DeepSeek-V2 and V3. window_years=w lets an event attend
    only to events in the w years before it, a sliding window over time rather than positions.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, n_kv_heads: int | None = None,
                 rope: str | None = None, rope_base: float = 10000.0, age_scale: float = 1.0,
                 age_bias: bool = False, max_slope_exponent: float = 8.0, qk_norm: bool = False,
                 kv_rank: int | None = None, window_years: float | None = None):
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
        if kv_rank:
            self.kv = nn.Sequential(nn.Linear(d_model, kv_rank, bias=False), nn.RMSNorm(kv_rank),
                                    nn.Linear(kv_rank, 2 * self.n_kv * self.head_dim))
        else:
            self.kv = nn.Linear(d_model, 2 * self.n_kv * self.head_dim)
        self.q_norm = nn.RMSNorm(self.head_dim) if qk_norm else None
        self.k_norm = nn.RMSNorm(self.head_dim) if qk_norm else None
        self.window_years = window_years
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
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        if self.window_years is not None:
            if ages is None:
                raise ValueError("window_years needs ages")
            near = (ages.unsqueeze(-1) - ages.unsqueeze(-2)) <= self.window_years  # [B, T, T]
            eye = torch.eye(t, dtype=torch.bool, device=x.device)
            mask = mask & (near | eye).unsqueeze(1)  # an event always sees itself
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


@register("attention", "mla")
def mla(d_model: int, n_heads: int, dropout: float = 0.0, kv_rank: int | None = None, **options) -> Attention:
    """Multi-head latent attention (DeepSeek-V2, V3): keys and values from a low-rank latent, default d_model / 4."""
    return Attention(d_model, n_heads, dropout, kv_rank=kv_rank or max(8, d_model // 4), **options)


@register("attention", "local_age")
def local_age(d_model: int, n_heads: int, dropout: float = 0.0, window_years: float = 5.0, **options) -> Attention:
    """Sliding-window attention over time (Mistral and Gemma slide over positions): only the last window_years."""
    return Attention(d_model, n_heads, dropout, window_years=window_years, **options)


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


@register("ffn", "moe")
class MixtureOfExperts(nn.Module):
    """Mixture of experts (Mixtral, DeepSeek, Qwen3-MoE): a router sends each event to its top_k of
    n_experts SwiGLU networks and mixes their outputs by the router's weights.

    Without pressure the router tends to use few experts, so it records a load-balancing loss
    (Switch Transformer: n_experts * sum over experts of the share of events routed there times the
    mean router probability), which the model adds to the training loss times aux_weight. Both are
    taken over real events: the model sets `valid` to its padding mask before each forward pass.

    dispatch "dense" (default) runs every expert on every event and weights each by the router,
    zero for the experts an event is not routed to: the same outputs and gradients as "sparse", at
    n_experts / top_k times the feed-forward compute, with tensor shapes that never change. "sparse"
    gives each expert only its own events, a different number every step; on an Apple GPU the memory
    cache kept a buffer for every new size, and a MIMIC-IV run grew to 47 GB on a 24 GB machine and
    stalled. Use "sparse" for many experts on CUDA.
    """

    def __init__(self, d_model: int, dropout: float = 0.0, n_experts: int = 4, top_k: int = 2,
                 mult: float = 4 / 3, aux_weight: float = 0.01, dispatch: str = "dense"):
        super().__init__()
        if not 1 <= top_k <= n_experts:
            raise ValueError("need 1 <= top_k <= n_experts")
        if dispatch not in ("dense", "sparse"):
            raise ValueError(f"dispatch must be dense or sparse, not {dispatch!r}")
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList(SwiGLU(d_model, dropout, mult=mult) for _ in range(n_experts))
        self.n_experts, self.top_k, self.aux_weight, self.dispatch = n_experts, top_k, aux_weight, dispatch
        self.aux_loss = torch.zeros(())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        probs = torch.softmax(self.router(flat), dim=-1)
        weight, chosen = probs.topk(self.top_k, dim=-1)
        weight = weight / weight.sum(dim=-1, keepdim=True)
        if self.dispatch == "dense":
            gates = torch.zeros_like(probs).scatter(1, chosen, weight)  # [events, n_experts]
            out = sum(expert(flat) * gates[:, e:e + 1] for e, expert in enumerate(self.experts))
        else:
            out = torch.zeros_like(flat)
            for e, expert in enumerate(self.experts):
                rows, slot = (chosen == e).nonzero(as_tuple=True)
                if len(rows):
                    out.index_add_(0, rows, expert(flat[rows]) * weight[rows, slot].unsqueeze(-1))
        # over real events only: the model sets self.valid to its padding mask
        valid = getattr(self, "valid", None)
        v = (valid.to(probs.dtype) if valid is not None and valid.numel() == len(flat)
             else torch.ones(len(flat), dtype=probs.dtype, device=x.device)).unsqueeze(-1)
        n = v.sum().clamp(min=1)
        share = (F.one_hot(chosen, self.n_experts).to(probs.dtype).sum(dim=1) * v).sum(dim=0) / (n * self.top_k)
        self.aux_loss = self.n_experts * (share * (probs * v).sum(dim=0) / n).sum()
        return out.reshape(shape)
