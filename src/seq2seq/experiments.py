"""Controlled ablations: the corpus split stays fixed across all model seeds.

This module executes experiments locally when explicitly invoked. It does not
schedule cloud jobs, assume measured results, or silently overwrite runs.
"""
from __future__ import annotations

import copy
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from seq2seq.config import load_config
from seq2seq.engine import evaluate, train


BASELINES: tuple[tuple[str, bool, bool], ...] = (
    ("rnn", False, False),
    ("rnn", False, True),
    ("lstm", False, False),
    ("lstm", False, True),
    ("lstm", True, False),
    ("lstm", True, True),
)


def experiment_name(cell: str, attention: bool, reverse: bool) -> str:
    return f"{cell}-attention-{int(attention)}-reverse-{int(reverse)}"


def make_plan(
    base_config: dict, seeds: list[int], split_seed: int
) -> list[dict[str, Any]]:
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Provide unique training seeds")
    plan: list[dict[str, Any]] = []
    for cell, attention, reverse in BASELINES:
        for seed in seeds:
            cfg = copy.deepcopy(base_config)
            cfg["seed"] = int(seed)
            cfg["dataset"]["split_seed"] = int(split_seed)
            cfg["model"]["cell"] = cell
            cfg["model"]["attention"] = attention
            cfg["model"]["reverse_source"] = reverse
            plan.append({
                "name": experiment_name(cell, attention, reverse),
                "seed": int(seed), "split_seed": int(split_seed), "config": cfg,
            })
    return plan


def run_matrix(
    config_path: str | Path, output_dir: str | Path,
    seeds: list[int], split_seed: int, max_examples: int,
    limit: int | None = None
) -> list[dict[str, Any]]:
    plan = make_plan(load_config(config_path), seeds, split_seed)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "plan.json").write_text(
        json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    results = []
    for item in plan[:limit] if limit is not None else plan:
        dirname = root / item["name"] / f"seed-{item['seed']}"
        path = dirname / "result.json"
        if path.exists():
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("config") != item["config"]:
                raise ValueError(f"Saved experiment config changed: {path}")
            results.append(record)
            continue
        if (dirname / "last.pt").exists():
            raise FileExistsError(
                f"Incomplete run exists at {dirname}; inspect and resume explicitly"
            )
        dirname.mkdir(parents=True, exist_ok=True)
        (dirname / "experiment.yaml").write_text(
            yaml.safe_dump(item["config"], sort_keys=False), encoding="utf-8"
        )
        train(item["config"], dirname)
        outcome = evaluate(
            dirname / "best.pt", split="test", max_examples=max_examples,
            device_name=str(item["config"]["training"].get("device", "auto")),
            predictions_path=dirname / "predictions.jsonl",
        )
        metadata = json.loads((dirname / "metadata.json").read_text(encoding="utf-8"))
        record = {
            "name": item["name"], "seed": item["seed"], "split_seed": item["split_seed"],
            "config": item["config"], "data_fingerprint": metadata["data_fingerprint"],
            "metrics": outcome,
        }
        path.write_text(
            json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        results.append(record)
    return results


def summarize(root: str | Path) -> dict[str, Any]:
    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(Path(root).glob("*/seed-*/result.json"))
    ]
    if not records:
        raise FileNotFoundError("No completed experiments found")
    fingerprints = {item["data_fingerprint"] for item in records}
    if len(fingerprints) != 1:
        raise ValueError(
            "Comparability failure: corpus splits or preprocessing differ across experiments"
        )
    grouped: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        grouped[rec["name"]].append(rec)
    report: dict[str, Any] = {
        "completed_runs": len(records),
        "expected_runs": len(BASELINES) * len({r["seed"] for r in records}),
        "data_fingerprint": next(iter(fingerprints)),
        "models": {},
    }
    for name, group in sorted(grouped.items()):
        metrics = {}
        for key in ("nll", "perplexity", "token_accuracy", "bleu", "exact_match"):
            values = [float(r["metrics"][key]) for r in group
                      if r["metrics"].get(key) is not None]
            metrics[key] = {
                "mean": statistics.mean(values) if values else None,
                "std": statistics.stdev(values) if len(values) > 1 else None,
                "n": len(values),
            }
        report["models"][name] = {
            "seeds": sorted(r["seed"] for r in group), "metrics": metrics,
        }
    return report
