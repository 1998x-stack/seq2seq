"""Factories for model, objective, optimizer, schedule and inference.

Separate 2014 paper-inspired choices from later extensions. These factories
are wired into the active train/evaluate/translate paths, not just examples.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from seq2seq.losses import masked_cross_entropy
from seq2seq.model import Seq2Seq


class ModelFactory:
    @staticmethod
    def create(cfg: dict, src_size: int, tgt_size: int) -> Seq2Seq:
        settings = cfg["model"]
        name = settings.get("architecture", "encoder_decoder")
        if name != "encoder_decoder":
            raise ValueError(f"Unknown model architecture: {name}")
        model = Seq2Seq(src_size, tgt_size, settings)
        init = settings.get("initialization", "pytorch_default")
        if init == "uniform":
            radius = float(settings.get("init_range", 0.08))
            if radius <= 0:
                raise ValueError("model.init_range must be positive")
            for parameter in model.parameters():
                nn.init.uniform_(parameter, -radius, radius)
            # Keep PAD embeddings zero; otherwise PAD may be a learnable signal.
            with torch.no_grad():
                model.encoder.embedding.weight[model.src_pad].zero_()
                model.decoder.embedding.weight[model.tgt_pad].zero_()
        elif init != "pytorch_default":
            raise ValueError(f"Unknown initialization: {init}")
        return model


class LossFactory:
    @staticmethod
    def create(cfg: dict, pad_id: int = 0) -> Callable[
        [torch.Tensor, torch.Tensor], tuple[torch.Tensor, int]
    ]:
        mode = cfg["training"].get("loss", "token_ce")
        smoothing = float(cfg["training"].get("label_smoothing", 0.0))
        if mode == "token_ce":
            return lambda logits, targets: masked_cross_entropy(
                logits, targets, pad_id=pad_id, label_smoothing=smoothing
            )
        if mode == "sequence_nll":
            # Original 2014 Eq. (3.2) maximizes the mean sentence log P(T|S).
            # Unlike token CE, long sentences contribute proportional to length.
            def sequence_loss(
                logits: torch.Tensor, targets: torch.Tensor
            ) -> tuple[torch.Tensor, int]:
                if logits.ndim != 3 or logits.shape[:2] != targets.shape:
                    raise ValueError("logits/targets shape mismatch")
                mask = targets.ne(pad_id)
                if not bool(mask.any()):
                    raise ValueError("Batch contains no non-padding labels")
                token_losses = F.cross_entropy(
                    logits.transpose(1, 2), targets, ignore_index=pad_id,
                    reduction="none", label_smoothing=smoothing,
                )
                return token_losses.sum(dim=1).mean(), int(mask.sum().item())
            return sequence_loss
        raise ValueError(f"Unknown loss: {mode}")


class OptimizerFactory:
    @staticmethod
    def create(params: Any, cfg: dict) -> torch.optim.Optimizer:
        section = cfg["training"]
        mode = str(section.get("optimizer", "adamw")).lower()
        lr = float(section["learning_rate"])
        weight_decay = float(section.get("weight_decay", 0.0))
        if mode == "sgd":
            return torch.optim.SGD(
                params, lr=lr, momentum=float(section.get("momentum", 0.0)),
                weight_decay=weight_decay,
            )
        if mode == "adamw":
            return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        raise ValueError(f"Unknown optimizer: {mode}")


class SchedulerFactory:
    """Supports original-paper half-epoch LR halving with stable stateless formula."""

    @staticmethod
    def learning_rate(cfg: dict, completed_fractional_epoch: float) -> float:
        base = float(cfg["training"]["learning_rate"])
        schedule = cfg["training"].get("scheduler", "constant")
        if isinstance(schedule, str):
            schedule = {"name": schedule}
        name = schedule.get("name", "constant")
        if name == "constant":
            return base
        if name == "paper_halving":
            begin = float(schedule.get("begin_epoch", 5.0))
            interval = float(schedule.get("interval_epochs", 0.5))
            gamma = float(schedule.get("gamma", 0.5))
            if interval <= 0 or not 0 < gamma <= 1:
                raise ValueError("Invalid paper_halving scheduler parameters")
            if completed_fractional_epoch <= begin:
                return base
            exponent = math.floor(
                (completed_fractional_epoch - begin + 1e-9) / interval
            )
            return base * gamma ** max(0, exponent)
        raise ValueError(f"Unknown scheduler: {name}")

    @classmethod
    def step(
        cls, optimizer: torch.optim.Optimizer, cfg: dict, epoch_progress: float
    ) -> float:
        lr = cls.learning_rate(cfg, epoch_progress)
        for group in optimizer.param_groups:
            group["lr"] = lr
        return lr


class DecoderFactory:
    @staticmethod
    def create(cfg: dict) -> Callable[..., torch.Tensor]:
        section = cfg.get("decoding", {})
        name = section.get("strategy", "greedy")
        if name == "greedy":
            return DecoderFactory.greedy
        if name == "beam":
            width = int(section.get("beam_size", 2))
            alpha = float(section.get("length_penalty", 0.0))
            if width < 1 or alpha < 0:
                raise ValueError("Beam size must be positive and length penalty nonnegative")

            def decode(model: Seq2Seq, src: torch.Tensor,
                       lengths: torch.Tensor, bos_id: int = 2,
                       eos_id: int = 3, max_new_tokens: int = 64) -> torch.Tensor:
                return beam_search(
                    model, src, lengths, bos_id, eos_id, max_new_tokens,
                    width, alpha,
                )
            return decode
        raise ValueError(f"Unknown decoding strategy: {name}")

    @staticmethod
    def greedy(model: Seq2Seq, src: torch.Tensor, lengths: torch.Tensor,
               bos_id: int = 2, eos_id: int = 3,
               max_new_tokens: int = 64) -> torch.Tensor:
        return model.generate(src, lengths, bos_id, eos_id, max_new_tokens)


def _state_for_sample(state: Any, index: int) -> Any:
    if isinstance(state, tuple):
        return tuple(s[:, index:index + 1].contiguous() for s in state)
    return state[:, index:index + 1].contiguous()


@torch.inference_mode()
def beam_search(
    model: Seq2Seq, src: torch.Tensor, lengths: torch.Tensor,
    bos_id: int, eos_id: int, max_new_tokens: int, width: int,
    alpha: float = 0.0,
) -> torch.Tensor:
    """Reference beam search: slow but correct, B>=1, supports RNN and LSTM.

    Rankings are by raw cumulative log probability with alpha=0 (paper-like).
    EOS hypotheses are separated into completed and never expanded again.
    """
    if max_new_tokens <= 0 or width <= 0:
        raise ValueError("Positive max_new_tokens and width required")
    if width == 1:
        return model.generate(src, lengths, bos_id, eos_id, max_new_tokens)

    memory, state = model.encoder(src, lengths)
    mask = src.ne(model.src_pad)
    sequences: list[list[int]] = []
    for i in range(src.size(0)):
        one_state = _state_for_sample(state, i)
        one_mem = memory[i:i + 1]
        one_mask = mask[i:i + 1]
        # Each hypothesis: (ids, log_probability, RNN_state, last_token).
        active = [([], 0.0, one_state, bos_id)]
        completed: list[tuple[list[int], float]] = []
        for _ in range(max_new_tokens):
            options = []
            for ids, score, h, last in active:
                token = torch.tensor([last], device=src.device, dtype=torch.long)
                logits, next_h = model.decoder.step(token, h, one_mem, one_mask)
                logits[:, model.tgt_pad] = -float("inf")
                logits[:, bos_id] = -float("inf")
                log_probs = F.log_softmax(logits, dim=-1).squeeze(0)
                values, indices = torch.topk(log_probs, min(width, log_probs.numel()))
                for value, index in zip(values.tolist(), indices.tolist()):
                    new_ids = [*ids, index]
                    new_score = score + float(value)
                    if index == eos_id:
                        completed.append((new_ids, new_score))
                    else:
                        options.append((new_ids, new_score, next_h, index))
            options.sort(key=lambda h: h[1], reverse=True)
            active = options[:width]
            if not active:
                break

        def ranking(ids: list[int], score: float) -> float:
            if alpha == 0:
                return score
            return score / ((5 + len(ids)) / 6) ** alpha

        # At max length allow unfinished, unless completed is the only answer.
        candidates = completed + [(ids, score) for ids, score, _, _ in active]
        if not candidates:
            raise RuntimeError("Beam search produced no candidates")
        winner = max(candidates, key=lambda h: ranking(h[0], h[1]))[0]
        sequences.append(winner)
    output = torch.full(
        (len(sequences), max(len(s) for s in sequences)),
        eos_id, device=src.device, dtype=torch.long,
    )
    for i, seq in enumerate(sequences):
        output[i, :len(seq)] = torch.tensor(seq, device=src.device)
    return output
