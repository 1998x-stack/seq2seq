"""Length-aware mini-batches; source/target lengths remain variable.

Each epoch shuffles large pools, sorts each pool by maximum sequence length,
and shuffles its mini-batches. This avoids global deterministic length ordering.
Unlike a fixed seed on every __iter__, set_epoch enables reproducible replays.
"""
from __future__ import annotations

import math
import random
from collections.abc import Iterator
from typing import Any

from torch.utils.data import Sampler


class LengthBucketBatchSampler(Sampler[list[int]]):
    def __init__(
        self, dataset: Any, batch_size: int, seed: int,
        pool_multiplier: int = 50, drop_last: bool = False
    ) -> None:
        if batch_size <= 0 or pool_multiplier < 1:
            raise ValueError("Positive batch_size and pool_multiplier required")
        self.lengths = [
            max(len(src), len(tgt)) for src, tgt in dataset.items
        ]
        self.batch_size = batch_size
        self.seed = seed
        self.pool_multiplier = pool_multiplier
        self.drop_last = drop_last
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch * 1_000_003)
        indices = list(range(len(self.lengths)))
        rng.shuffle(indices)
        pool_size = self.batch_size * self.pool_multiplier
        batches = []
        for start in range(0, len(indices), pool_size):
            pool = sorted(
                indices[start:start + pool_size],
                key=lambda i: self.lengths[i],
            )
            for index in range(0, len(pool), self.batch_size):
                batch = pool[index:index + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)
        rng.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        count = len(self.lengths)
        if self.drop_last:
            return count // self.batch_size
        return math.ceil(count / self.batch_size)


class DataLoaderFactory:
    @staticmethod
    def create(dataset: Any, cfg: dict, collate: Any, shuffle: bool) -> Any:
        from torch.utils.data import DataLoader

        training = cfg["training"]
        mode = training.get("batching", "random")
        size = int(training["batch_size"])
        workers = int(training.get("num_workers", 0))
        if mode == "length_bucket" and shuffle:
            sampler = LengthBucketBatchSampler(
                dataset, size, seed=int(cfg.get("seed", 42)),
                pool_multiplier=int(training.get("pool_multiplier", 50)),
            )
            return DataLoader(
                dataset, batch_sampler=sampler, collate_fn=collate,
                num_workers=workers,
            )
        if mode not in ("random", "length_bucket"):
            raise ValueError(f"Unknown batching strategy: {mode}")
        return DataLoader(
            dataset, batch_size=size, shuffle=shuffle, num_workers=workers,
            collate_fn=collate,
        )
