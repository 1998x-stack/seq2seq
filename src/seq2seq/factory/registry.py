"""Typed, explicit registries: unknown names fail, never silently fall back."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any, Generic, TypeVar

from torch import nn

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, category: str) -> None:
        self.category = category
        self._items: dict[str, Callable[..., T]] = {}

    def register(self, name: str, constructor: Callable[..., T]) -> None:
        key = name.strip().lower()
        if not key or key in self._items:
            raise ValueError(f"Duplicate or invalid {self.category}: {name}")
        self._items[key] = constructor

    def create(self, name: str, *args: Any, **kwargs: Any) -> T:
        key = name.strip().lower()
        if key not in self._items:
            raise ValueError(
                f"Unknown {self.category} '{name}'. Choices: {', '.join(self.names())}"
            )
        return self._items[key](*args, **kwargs)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._items))


CELL_REGISTRY: Registry[nn.Module] = Registry("recurrent cell")
CELL_REGISTRY.register("rnn", nn.RNN)
CELL_REGISTRY.register("lstm", nn.LSTM)
