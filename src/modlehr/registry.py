"""Name-based registries for the parts of the model that can be swapped.

Every swappable part (attention, time encoding, value encoding, normalization, feed-forward,
output head, target grouping) is registered under a kind and a name. A config picks parts by
name, so trying a new architecture means writing one class, registering it, and changing the
config; nothing else in the code base changes.

    @register("attention", "mha")
    class CausalSelfAttention(nn.Module): ...

    attn = build("attention", "mha", d_model=128, n_heads=4)
"""

from __future__ import annotations

from typing import Any, Callable

_REGISTRY: dict[str, dict[str, Any]] = {}


def register(kind: str, name: str) -> Callable[[Any], Any]:
    def decorator(obj: Any) -> Any:
        table = _REGISTRY.setdefault(kind, {})
        if name in table:
            raise ValueError(f"{kind} '{name}' is already registered")
        table[name] = obj
        return obj

    return decorator


def get(kind: str, name: str) -> Any:
    try:
        return _REGISTRY[kind][name]
    except KeyError:
        raise KeyError(f"unknown {kind} '{name}'; available: {available(kind)}") from None


def build(kind: str, name: str, **kwargs: Any) -> Any:
    return get(kind, name)(**kwargs)


def available(kind: str) -> list[str]:
    return sorted(_REGISTRY.get(kind, {}))
