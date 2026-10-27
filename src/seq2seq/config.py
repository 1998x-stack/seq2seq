from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError("Config must be a YAML mapping")
    validate_config(config)
    return config


def validate_config(cfg: dict[str, Any]) -> None:
    for name in ("dataset", "tokenizer", "model", "training"):
        if not isinstance(cfg.get(name), dict):
            raise TypeError(f"Missing configuration section: {name}")
    dataset, model, training = cfg["dataset"], cfg["model"], cfg["training"]
    if model.get("cell") not in ("rnn", "lstm"):
        raise ValueError("model.cell must be 'rnn' or 'lstm'")
    for field in ("embedding_dim", "hidden_dim", "num_layers"):
        if int(model.get(field, 0)) <= 0:
            raise ValueError(f"model.{field} must be positive")
    if not 0 <= float(model.get("dropout", 0)) < 1:
        raise ValueError("model.dropout must be in [0,1)")
    if dataset.get("backend") not in ("huggingface", "toy", "jsonl"):
        raise ValueError("dataset.backend must be 'huggingface', 'jsonl' or 'toy'")
    val = float(dataset.get("validation_fraction", 0.05))
    test = float(dataset.get("test_fraction", 0.05))
    if val <= 0 or test <= 0 or val + test >= 1:
        raise ValueError("Validation/test fractions must be positive and sum below 1")
    if int(dataset.get("max_length", 0)) < 2:
        raise ValueError("dataset.max_length must be at least 2")
    if int(dataset.get("max_samples", 0)) < 0:
        raise ValueError("dataset.max_samples cannot be negative")
    if dataset.get("split_strategy", "random") not in ("random", "group"):
        raise ValueError("dataset.split_strategy must be random or group")
    if dataset.get("backend") == "jsonl" and not dataset.get("path"):
        raise ValueError("dataset.path is required for jsonl")
    if dataset.get("backend") in ("huggingface", "jsonl"):
        required_fields = ("source_lang", "target_lang")
        for field in required_fields:
            if not dataset.get(field):
                raise ValueError(f"dataset.{field} must be set")
    if dataset.get("backend") == "huggingface":
        for field in ("name", "subset", "source_lang", "target_lang"):
            if not dataset.get(field):
                raise ValueError(f"dataset.{field} must be set for Hugging Face")
    if int(training.get("batch_size", 0)) <= 0:
        raise ValueError("training.batch_size must be positive")
    if int(training.get("epochs", 0)) <= 0:
        raise ValueError("training.epochs must be positive")
    if float(training.get("learning_rate", 0)) <= 0:
        raise ValueError("training.learning_rate must be positive")
    if model.get("architecture", "encoder_decoder") != "encoder_decoder":
        raise ValueError("Unsupported model.architecture")
    if model.get("initialization", "pytorch_default") not in ("pytorch_default", "uniform"):
        raise ValueError("Unknown model.initialization")
    if float(model.get("init_range", 0.08)) <= 0:
        raise ValueError("model.init_range must be positive")
    if training.get("optimizer", "adamw") not in ("adamw", "sgd"):
        raise ValueError("Unknown training.optimizer")
    if training.get("loss", "token_ce") not in ("token_ce", "sequence_nll"):
        raise ValueError("Unknown training.loss")
    if training.get("batching", "random") not in ("random", "length_bucket"):
        raise ValueError("Unknown training.batching")
    if int(training.get("pool_multiplier", 50)) <= 0:
        raise ValueError("training.pool_multiplier must be positive")
    for field in ("src_vocab_size", "tgt_vocab_size"):
        if field in cfg["tokenizer"] and int(cfg["tokenizer"][field]) < 4:
            raise ValueError(f"tokenizer.{field} must be at least four")
    decoding = cfg.get("decoding", {})
    if not isinstance(decoding, dict):
        raise TypeError("decoding must be a mapping")
    if decoding.get("strategy", "greedy") not in ("greedy", "beam"):
        raise ValueError("Unknown decoding.strategy")
    if int(decoding.get("beam_size", 2)) < 1:
        raise ValueError("decoding.beam_size must be positive")
    if float(decoding.get("length_penalty", 0.0)) < 0:
        raise ValueError("decoding.length_penalty cannot be negative")
    scheduler = training.get("scheduler", "constant")
    if isinstance(scheduler, dict):
        if scheduler.get("name", "constant") not in ("constant", "paper_halving"):
            raise ValueError("Unknown training.scheduler")
        if float(scheduler.get("interval_epochs", 0.5)) <= 0:
            raise ValueError("training.scheduler.interval_epochs must be positive")
    elif scheduler not in ("constant", "paper_halving"):
        raise ValueError("Unknown training.scheduler")
    if not isinstance(cfg.get("logging", {}), dict):
        raise TypeError("logging must be a YAML mapping")
    if int(cfg.get("logging", {}).get("log_every_steps", 20)) < 0:
        raise ValueError("logging.log_every_steps must be nonnegative")
    for field in ("teacher_forcing", "label_smoothing"):
        value = float(training.get(field, 0))
        if not 0 <= value <= 1:
            raise ValueError(f"training.{field} must be in [0,1]")
    if int(training.get("max_new_tokens", 0)) <= 0:
        raise ValueError("training.max_new_tokens must be positive")
