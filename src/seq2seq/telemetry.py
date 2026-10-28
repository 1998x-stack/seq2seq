"""Structured, append-only experimental telemetry.

No global logging handlers: multiple tests/runs cannot steal one another's logs.
The JSONL stream is machine readable; metrics.csv is a portable epoch summary.
TensorBoard is optional and imported only when explicitly enabled.
"""
from __future__ import annotations

import csv
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self


def utc_timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _safe_number(value: Any) -> Any:
    # Torch/NumPy scalars expose item(), but bool, str and native numbers do not.
    if hasattr(value, "item") and callable(value.item):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


class RunLogger:
    """Write structured run events, with optional TensorBoard scalars."""

    def __init__(
        self,
        root: str | Path,
        name: str,
        tensorboard: bool = False,
        console: bool = True,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.name = name
        self.console = console
        self.file = (self.root / "events.jsonl").open("a", encoding="utf-8")
        self.writer = None
        if tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError as exc:
                self.file.close()
                raise RuntimeError(
                    "TensorBoard support is optional: pip install -e '.[viz]'"
                ) from exc
            self.writer = SummaryWriter(log_dir=str(self.root / "tensorboard"))

    def emit(self, event: str, **values: Any) -> dict[str, Any]:
        record = {
            "time_utc": utc_timestamp(),
            "event": event,
            "run": self.name,
            **{key: _safe_number(value) for key, value in values.items()},
        }
        line = json.dumps(record, ensure_ascii=False, allow_nan=False)
        self.file.write(line + "\n")
        self.file.flush()
        if self.console:
            print(line, file=sys.stderr, flush=True)
        return record

    def scalar(self, tag: str, value: float, step: int) -> None:
        if self.writer is not None:
            self.writer.add_scalar(tag, float(value), step)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None
        if not self.file.closed:
            self.file.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


COLUMNS = (
    "epoch",
    "train_nll", "validation_nll",
    "train_perplexity", "validation_perplexity",
    "train_token_accuracy", "validation_token_accuracy",
    "train_tokens", "validation_tokens",
    "train_elapsed_seconds", "validation_elapsed_seconds",
    "train_tokens_per_second", "validation_tokens_per_second",
    "learning_rate", "best_validation_nll",
)


def export_metrics_csv(root: str | Path) -> Path:
    """Flatten existing, factual metrics.jsonl into a stable CSV schema."""
    root = Path(root)
    source = root / "metrics.jsonl"
    if not source.is_file():
        raise FileNotFoundError(source)
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    for lineno, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        entry = json.loads(line)
        epoch = int(entry["epoch"])
        if epoch in seen:
            raise ValueError(f"Duplicate epoch {epoch} at line {lineno}: {source}")
        seen.add(epoch)
        row: dict[str, Any] = {"epoch": epoch}
        for split in ("train", "validation"):
            for name in (
                "nll", "perplexity", "token_accuracy", "tokens",
                "elapsed_seconds", "tokens_per_second",
            ):
                row[f"{split}_{name}"] = entry.get(split, {}).get(name)
        row["learning_rate"] = entry.get("learning_rate")
        row["best_validation_nll"] = entry.get("best_validation_nll")
        rows.append(row)
    output = root / "metrics.csv"
    temp = output.with_suffix(".csv.tmp")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(output)
    return output
