"""Declarative paper-inspired experiment runner and immutable manifest."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml

from seq2seq.config import load_config, validate_config
from seq2seq.engine import evaluate, train
from seq2seq.experiments import summarize
from seq2seq.telemetry import RunLogger


def _set_nested(cfg: dict, path: str, value: Any) -> None:
    parts = path.split(".")
    if len(parts) < 2 or not all(p.isidentifier() for p in parts):
        raise ValueError(f"Use dotted section.key path: {path}")
    node = cfg
    for key in parts[:-1]:
        if key == "decoding" and key not in node:
            node[key] = {}
        if key not in node or not isinstance(node[key], dict):
            raise ValueError(f"Unknown configuration section: {path}")
        node = node[key]
    if parts[-1].startswith("_"):
        raise ValueError(f"Private configuration override: {path}")
    node[parts[-1]] = value


def factory_plan(spec_path: str | Path) -> list[dict[str, Any]]:
    path = Path(spec_path)
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(spec, dict) or "base_config" not in spec:
        raise ValueError("Factory spec requires base_config")
    base_file = Path(spec["base_config"])
    if not base_file.is_absolute():
        candidate = path.parent / base_file
        base_file = candidate if candidate.exists() else base_file
    base = load_config(base_file)
    seeds = spec.get("seeds", [42, 43, 44])
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique and nonempty")
    variants = spec.get("variants", [])
    if not variants:
        raise ValueError("Factory spec requires variants")
    ids = [item["id"] for item in variants]
    if len(ids) != len(set(ids)) or any(
        not ident or not all(c.isalnum() or c in "-_" for c in ident)
        for ident in ids
    ):
        raise ValueError("Variant ids must be unique alphanumeric slugs")
    plan = []
    for item in variants:
        for seed in seeds:
            cfg = copy.deepcopy(base)
            cfg["seed"] = int(seed)
            cfg["dataset"]["split_seed"] = int(spec.get("split_seed", 2026))
            for dotted, value in item.get("overrides", {}).items():
                _set_nested(cfg, dotted, value)
            validate_config(cfg)
            plan.append({
                "name": item["id"], "seed": int(seed),
                "split_seed": int(cfg["dataset"]["split_seed"]),
                "scope": item.get("scope", "research_extension"),
                "hypothesis": item.get("hypothesis", ""),
                "config": cfg,
            })
    return plan


def factory_run(
    spec_path: str | Path, output_dir: str | Path,
    max_examples: int = 200, limit: int | None = None,
    split: str = "validation",
) -> dict:
    if split not in ("train", "validation", "test"):
        raise ValueError("Unsupported evaluation split")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    plan = factory_plan(spec_path)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    target = root / "plan.json"
    if target.exists():
        if json.loads(target.read_text(encoding="utf-8")) != plan:
            raise ValueError("Experiment plan mismatch; use a clean output directory")
    else:
        target.write_text(
            json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    with RunLogger(root, name="paper-factory") as logger:
        logger.emit("factory_started", planned=len(plan), split=split)
        for item in plan[:limit] if limit is not None else plan:
            folder = root / item["name"] / f"seed-{item['seed']}"
            result_path = folder / "result.json"
            if result_path.exists():
                previous = json.loads(result_path.read_text(encoding="utf-8"))
                if previous["config"] != item["config"] or previous.get(
                    "evaluation_split"
                ) != split:
                    raise ValueError(f"Existing result is incompatible: {result_path}")
                logger.emit("factory_reused", variant=item["name"], seed=item["seed"])
                continue
            if (folder / "last.pt").exists():
                raise FileExistsError(
                    f"Found incomplete run: {folder}; inspect before restarting"
                )
            folder.mkdir(parents=True, exist_ok=True)
            logger.emit("factory_run_started", variant=item["name"], seed=item["seed"])
            try:
                train(item["config"], folder)
                metrics = evaluate(
                    folder / "best.pt", split=split,
                    max_examples=max_examples,
                    device_name=str(
                        item["config"]["training"].get("device", "auto")
                    ),
                    predictions_path=folder / "predictions.jsonl",
                )
                metadata = json.loads(
                    (folder / "metadata.json").read_text(encoding="utf-8")
                )
                result = {
                    "name": item["name"], "seed": item["seed"],
                    "config": item["config"],
                    "data_fingerprint": metadata["data_fingerprint"],
                    "evaluation_split": split, "hypothesis": item["hypothesis"],
                    "scope": item["scope"], "metrics": metrics,
                }
                tmp = result_path.with_suffix(".tmp")
                tmp.write_text(
                    json.dumps(result, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                tmp.replace(result_path)
                logger.emit(
                    "factory_run_completed", variant=item["name"],
                    seed=item["seed"], eval_bleu=metrics.get("bleu"),
                )
            except Exception as exc:
                logger.emit(
                    "factory_run_failed", variant=item["name"],
                    seed=item["seed"], error=repr(exc),
                )
                raise
        logger.emit("factory_finished")
    return summarize(root)
