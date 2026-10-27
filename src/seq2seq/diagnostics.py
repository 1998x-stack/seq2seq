"""Evaluation diagnostics for generated translations.

BLEU is always computed against the original reference text after the
repository's deterministic normalization, NEVER against vocab-decoded labels:
the latter would silently replace OOV reference words with <unk>.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def length_bucket(length: int) -> str:
    if length <= 10:
        return "01-10"
    if length <= 20:
        return "11-20"
    if length <= 40:
        return "21-40"
    return "41+"


def _corpus_bleu(hypotheses: list[str], references: list[str]) -> tuple[float, str]:
    from sacrebleu.metrics import BLEU

    metric = BLEU()
    score = metric.corpus_score(hypotheses, [references])
    return float(score.score), str(metric.get_signature())


def summarize_predictions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "bleu": None, "bleu_signature": None, "exact_match": None,
            "generated_examples": 0, "length_buckets": {}, "oov_analysis": {},
            "examples": [],
        }
    hyps = [str(row["prediction"]) for row in rows]
    refs = [str(row["reference"]) for row in rows]
    bleu, signature = _corpus_bleu(hyps, refs)

    def aggregate(part: list[dict[str, Any]]) -> dict[str, float | int | None]:
        if not part:
            return {"count": 0, "exact_match": None, "bleu": None}
        score, _ = _corpus_bleu(
            [r["prediction"] for r in part], [r["reference"] for r in part]
        )
        return {
            "count": len(part),
            "exact_match": sum(bool(r["exact_match"]) for r in part) / len(part),
            "bleu": score,
        }

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[length_bucket(int(row["source_length"]))].append(row)
    buckets = {key: aggregate(grouped.get(key, []))
               for key in ("01-10", "11-20", "21-40", "41+")}
    zero = [r for r in rows if int(r["source_oov_count"]) == 0]
    has_oov = [r for r in rows if int(r["source_oov_count"]) > 0]
    return {
        "bleu": bleu,
        "bleu_signature": signature,
        "exact_match": sum(bool(r["exact_match"]) for r in rows) / len(rows),
        "generated_examples": len(rows),
        "length_buckets": buckets,
        "oov_analysis": {
            "no_source_oov": aggregate(zero),
            "has_source_oov": aggregate(has_oov),
            "source_oov_token_rate": sum(r["source_oov_count"] for r in rows) /
                max(1, sum(r["source_length"] for r in rows)),
            "target_oov_token_rate": sum(r["target_oov_count"] for r in rows) /
                max(1, sum(r["target_length"] for r in rows)),
        },
        "examples": [
            {"source": r["source"], "reference": r["reference"],
             "prediction": r["prediction"]} for r in rows[:5]
        ],
    }


def export_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)
