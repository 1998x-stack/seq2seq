from __future__ import annotations

import math

import torch
from torch.nn import functional as F


def masked_cross_entropy(
    logits: torch.Tensor, targets: torch.Tensor, pad_id: int = 0,
    label_smoothing: float = 0.0
) -> tuple[torch.Tensor, int]:
    if logits.ndim != 3 or targets.shape != logits.shape[:2]:
        raise ValueError("Expected logits [batch, time, vocab], targets [batch, time]")
    valid = int(targets.ne(pad_id).sum().item())
    if valid == 0:
        raise ValueError("A batch must contain at least one non-padding target")
    total = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
        ignore_index=pad_id, label_smoothing=label_smoothing, reduction="sum",
    )
    return total / valid, valid


def perplexity(mean_nll: float) -> float:
    return math.exp(min(100.0, mean_nll))
