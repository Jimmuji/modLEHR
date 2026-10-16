"""The model, assembled from registered parts by the model config.

    tokens, ages, values → input head → N blocks → final norm → output head

Input head: h_i = E_code(c_i) + E_age(t_i) + m_i F(z_i) + T(words_i): the value encoding reads a
measured number, the text encoding the words of a note. An optional `extra` tensor of the same
shape is added to h, the hook for other modalities (images) later.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from modlehr import registry
from modlehr.config import ModelConfig
from modlehr.model import encodings, heads, layers  # noqa: F401  (register the parts)


class InputHead(nn.Module):
    def __init__(self, n_tokens: int, cfg: ModelConfig, n_text_buckets: int = 0):
        super().__init__()
        self.embed = nn.Embedding(n_tokens, cfg.d_model, padding_idx=0)
        self.time = registry.build("time_encoding", cfg.time_encoding, d_model=cfg.d_model)
        self.value = registry.build("value_encoding", cfg.value_encoding, d_model=cfg.d_model)
        self.text = registry.build("text_encoding", cfg.text_encoding, d_model=cfg.d_model, n_buckets=n_text_buckets)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, tokens, ages, values, has_value, extra: torch.Tensor | None = None,
                text: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None) -> torch.Tensor:
        h = self.embed(tokens) + self.time(ages) + self.value(values, has_value)
        if text is not None and text[2].numel():
            word_ids, starts, positions = text
            encoded = self.text(word_ids, starts)
            if encoded is not None:
                h = h.reshape(-1, h.shape[-1]).index_add(0, positions, encoded.to(h.dtype)).reshape(h.shape)
        if extra is not None:
            h = h + extra
        return self.drop(h)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = registry.build("norm", cfg.norm, d_model=cfg.d_model)
        self.attn = registry.build("attention", cfg.attention, d_model=cfg.d_model, n_heads=cfg.n_heads, dropout=cfg.dropout,
                                   **cfg.attention_options)
        self.norm2 = registry.build("norm", cfg.norm, d_model=cfg.d_model)
        self.ffn = registry.build("ffn", cfg.ffn, d_model=cfg.d_model, dropout=cfg.dropout, **cfg.ffn_options)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, ages: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.attn(self.norm1(x), mask, ages=ages))
        return x + self.ffn(self.norm2(x))


def attention_mask(tokens: torch.Tensor) -> torch.Tensor:
    """[B, 1, T, T] True where a query may attend: causal, and the key is not padding."""
    t = tokens.shape[1]
    causal = torch.ones(t, t, dtype=torch.bool, device=tokens.device).tril()
    return causal.view(1, 1, t, t) & (tokens != 0).view(-1, 1, 1, t)


class ModLEHR(nn.Module):
    def __init__(self, cfg: ModelConfig, n_tokens: int, n_targets: int, n_text_buckets: int = 0):
        super().__init__()
        self.cfg = cfg
        self.input_head = InputHead(n_tokens, cfg, n_text_buckets)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = registry.build("norm", cfg.norm, d_model=cfg.d_model)
        self.head = registry.build("head", cfg.head, d_model=cfg.d_model, n_targets=n_targets)
        self.apply(self._init)

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, batch: dict[str, torch.Tensor], extra: torch.Tensor | None = None) -> torch.Tensor:
        tokens = batch["tokens"]
        text = (batch["text_ids"], batch["text_offsets"], batch["text_pos"]) if "text_pos" in batch else None
        x = self.input_head(tokens, batch["ages"], batch["values"], batch["has_value"], extra, text)
        mask = attention_mask(tokens)
        for block in self.blocks:
            x = block(x, mask, batch["ages"])
        return self.head(self.norm(x))

    def loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict]:
        out = self(batch)
        nll, parts = self.head.loss(out, batch["next_target"], batch["dt"], batch["loss_mask"])
        aux = [mod.aux_weight * mod.aux_loss for mod in self.modules() if hasattr(mod, "aux_loss")]
        if aux and self.training:  # e.g. the load-balancing loss of a mixture of experts
            extra = torch.stack(aux).sum()
            parts["aux"] = float(extra.detach())
            return nll + extra, parts
        return nll, parts

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
