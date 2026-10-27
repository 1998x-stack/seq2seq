"""Factory package.

Keep imports lazy: model primitives use factory.registry and the high-level
constructors import model primitives. Eager imports here create a cycle.
"""

from seq2seq.factory.registry import CELL_REGISTRY, Registry

__all__ = ["CELL_REGISTRY", "Registry"]
