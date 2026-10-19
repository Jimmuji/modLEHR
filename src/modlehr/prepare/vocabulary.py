"""Input and target vocabularies.

The input vocabulary holds every token the model reads (diagnoses, measurements, answers,
sex). The target vocabulary holds the outcomes it predicts (grouped diseases, death). Target
tokens are also input tokens, so a predicted outcome can be fed back when sampling futures.
Index 0 of the input vocabulary is padding.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

PAD = "<pad>"
NO_EVENT = "MODLEHR//NO_EVENT"  # Delphi-2M's "no event" token, in the vocabulary when data.no_event_years is set


@dataclass
class Vocabulary:
    tokens: list[str]
    targets: list[str]
    token_index: dict[str, int] = field(init=False)
    input_to_target: np.ndarray = field(init=False)
    target_to_input: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        if not self.tokens or self.tokens[0] != PAD:
            raise ValueError("input vocabulary must start with the padding token")
        self.token_index = {t: i for i, t in enumerate(self.tokens)}
        self.input_to_target = np.full(len(self.tokens), -1, dtype=np.int32)
        self.target_to_input = np.zeros(len(self.targets), dtype=np.int32)
        for k, name in enumerate(self.targets):
            i = self.token_index[name]
            self.input_to_target[i] = k
            self.target_to_input[k] = i

    @classmethod
    def build(cls, input_tokens: list[str], target_tokens: list[str]) -> "Vocabulary":
        targets = sorted(set(target_tokens))
        others = sorted(set(input_tokens) - set(targets))
        return cls(tokens=[PAD] + others + targets, targets=targets)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"tokens": self.tokens, "targets": self.targets}, indent=1))

    @classmethod
    def load(cls, path: Path) -> "Vocabulary":
        raw = json.loads(path.read_text())
        return cls(tokens=raw["tokens"], targets=raw["targets"])
