"""Swappable transformer parts: attention, normalization, feed-forward.

The baseline is a plain GPT block: pre-norm LayerNorm, multi-head causal attention, GELU MLP.
New parts from recent LLMs are added here (or in a new module) and registered by name.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from modlehr.registry import register


@register("attention", "mha")
class CausalSelfAttention(nn.Module):
    """Multi-head attention. The mask passed in is causal plus key padding."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(x).view(b, t, 3, self.n_heads, d // self.n_heads).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0)
        return self.out(y.transpose(1, 2).reshape(b, t, d))


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
