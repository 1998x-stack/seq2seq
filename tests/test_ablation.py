from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from seq2seq.config import load_config, validate_config
from seq2seq.experiments import run_matrix, summarize
from seq2seq.telemetry import RunLogger, export_metrics_csv
from seq2seq.visualize import collect, render_dashboard


def test_jsonl_event_logger_and_csv(tmp_path: Path):
    with RunLogger(tmp_path, name="unit", console=False) as logger:
        event = logger.emit("train_step", global_step=3, batch_loss=1.25)
        assert event["run"] == "unit"
        assert event["time_utc"].endswith("+00:00")
        logger.scalar("train/nll", 2.0, 1)  # no-op unless TB enabled
    assert logger.file.closed
    actual = [json.loads(s) for s in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert actual[0]["batch_loss"] == 1.25
    source = tmp_path / "metrics.jsonl"
    source.write_text(
        json.dumps({
            "epoch": 1, "train": {"nll": 1.7, "tokens": 8, "tokens_per_second": 40},
            "validation": {"nll": 2.0, "tokens": 2}, "learning_rate": 0.001,
            "best_validation_nll": 2.0,
        }) + "\n",
        encoding="utf-8",
    )
    path = export_metrics_csv(tmp_path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["train_tokens_per_second"] == "40"
    assert rows[0]["validation_nll"] == "2.0"
    source.write_text(source.read_text() * 2, encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate epoch"):
        export_metrics_csv(tmp_path)


def test_logging_config_validation():
    cfg = load_config("configs/smoke.yaml")
    cfg["logging"]["log_every_steps"] = -1
    with pytest.raises(ValueError, match="logging.log_every_steps"):
        validate_config(cfg)


def test_smoke_matrix_generates_factual_offline_dashboard(tmp_path: Path):
    root = tmp_path / "ablation"
    records = run_matrix(
        "configs/smoke.yaml", root, seeds=[42],
        split_seed=2026, max_examples=4, limit=1,
    )
    assert len(records) == 1
    run_dir = root / "rnn-attention-0-reverse-0" / "seed-42"
    for path in ("metrics.jsonl", "metrics.csv", "events.jsonl", "result.json"):
        assert (run_dir / path).is_file(), path
    train_events = [
        json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()
    ]
    assert any(e["event"] == "epoch_completed" for e in train_events)
    assert any(e["event"] == "train_step" for e in train_events)
    matrix_events = [
        json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()
    ]
    assert any(e["event"] == "run_completed" for e in matrix_events)

    result = summarize(root)
    assert result["completed_runs"] == 1
    assert result["expected_runs"] == 6
    assert result["generation_sample_count"] == 4

    summary = collect(root)
    assert summary["completed"] == 1
    assert summary["planned"] == 6
    assert len(summary["missing"]) == 5
    output = render_dashboard(root)
    source = output.read_text(encoding="utf-8")
    assert len(source) > 100_000  # plotly.js is fully embedded for offline viewing
    assert "INCOMPLETE" in source
    assert "Seq2Seq" in source
    assert "Plotly.newPlot" in source
    assert result["data_fingerprint"] in source

    # An idempotent re-run skips training without mutating the frozen plan.
    again = run_matrix(
        "configs/smoke.yaml", root, seeds=[42],
        split_seed=2026, max_examples=4, limit=1,
    )
    assert len(again) == 1
    with pytest.raises(ValueError, match="Experiment plan differs"):
        run_matrix(
            "configs/smoke.yaml", root, seeds=[43],
            split_seed=2026, max_examples=4, limit=1,
        )


def test_dashboard_requires_completed_data(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        collect(tmp_path / "missing")
