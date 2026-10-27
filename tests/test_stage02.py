from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from seq2seq.config import load_config
from seq2seq.data import prepare_data
from seq2seq.diagnostics import export_jsonl, length_bucket, summarize_predictions
from seq2seq.engine import evaluate, train
from seq2seq.experiments import make_plan, run_matrix, summarize


def smoke():
    return load_config("configs/smoke.yaml")


def test_fixed_split_seed_across_model_seeds():
    c1 = smoke()
    c1["dataset"]["split_seed"] = 2026
    c1["seed"] = 10
    c2 = copy.deepcopy(c1)
    c2["seed"] = 999
    a, b = prepare_data(c1), prepare_data(c2)
    assert a.fingerprint == b.fingerprint
    assert a.train == b.train
    assert a.src_vocab.itos == b.src_vocab.itos
    assert a.split_info["split_seed"] == 2026


def test_group_disjoint_jsonl(tmp_path: Path):
    path = tmp_path / "parallel.jsonl"
    with path.open("w", encoding="utf-8") as out:
        for group in range(10):
            for idx in range(4):
                row = {
                    "translation": {
                        "en": f"source {group} example {idx}",
                        "fr": f"cible {group} exemple {idx}",
                    },
                    "book_id": f"book-{group}",
                }
                out.write(json.dumps(row) + "\n")
    c = smoke()
    c["dataset"].update({
        "backend": "jsonl", "path": str(path),
        "source_lang": "en", "target_lang": "fr",
        "split_strategy": "group", "group_field": "book_id",
        "max_samples": 0, "max_length": 25,
        "validation_fraction": 0.2, "test_fraction": 0.2,
    })
    b = prepare_data(c)
    def groups(pairs):
        return {s.split()[1] for s, _ in pairs}
    assert not (groups(b.train) & groups(b.test))
    assert not (groups(b.train) & groups(b.validation))
    assert not (groups(b.test) & groups(b.validation))
    assert b.split_info["groups"] == {"train": 6, "validation": 2, "test": 2}
    c["dataset"]["group_field"] = "missing"
    with pytest.raises(ValueError, match="nonempty group_field"):
        prepare_data(c)


def test_prediction_diagnostics_and_export(tmp_path: Path):
    rows = [
        {
            "index": 0, "source": "rare red", "reference": "rouge rare",
            "prediction": "rouge rare", "exact_match": True,
            "source_length": 2, "target_length": 2,
            "source_oov_count": 1, "target_oov_count": 1,
        },
        {
            "index": 1, "source": "long blue", "reference": "bleu",
            "prediction": "rouge", "exact_match": False,
            "source_length": 18, "target_length": 1,
            "source_oov_count": 0, "target_oov_count": 0,
        },
    ]
    report = summarize_predictions(rows)
    assert report["generated_examples"] == 2
    assert report["exact_match"] == 0.5
    assert report["oov_analysis"]["source_oov_token_rate"] == pytest.approx(1 / 20)
    assert report["length_buckets"]["01-10"]["count"] == 1
    assert report["length_buckets"]["11-20"]["count"] == 1
    assert "tok:" in report["bleu_signature"]
    assert length_bucket(41) == "41+"
    out = tmp_path / "exports" / "preds.jsonl"
    export_jsonl(out, rows)
    assert [json.loads(line)["source"] for line in out.read_text().splitlines()] == [
        "rare red", "long blue"
    ]


def test_matrix_plan_and_offline_subset(tmp_path: Path):
    c = smoke()
    plan = make_plan(c, [5, 6], 2026)
    assert len(plan) == 12
    assert len({(p["name"], p["seed"]) for p in plan}) == 12
    assert {p["config"]["dataset"]["split_seed"] for p in plan} == {2026}
    assert all(p["config"]["seed"] == p["seed"] for p in plan)
    root = tmp_path / "matrix"
    results = run_matrix("configs/smoke.yaml", root, [5], 2026, 2, limit=1)
    assert len(results) == 1
    report = summarize(root)
    assert report["completed_runs"] == 1
    assert report["models"]["rnn-attention-0-reverse-0"]["metrics"]["nll"]["n"] == 1
    assert len(list(root.glob("*/seed-*/predictions.jsonl"))) == 1


def test_eval_original_reference_not_vocabulary_unknown(tmp_path: Path):
    config = smoke()
    run = tmp_path / "run"
    train(config, run)
    export_path = tmp_path / "outputs.jsonl"
    result = evaluate(run / "best.pt", max_examples=-1, device_name="cpu",
                      predictions_path=export_path)
    rows = [json.loads(line) for line in export_path.read_text().splitlines()]
    assert len(rows) == len(prepare_data(config).test)
    assert result["generated_examples"] == len(rows)
    assert all("<unk>" not in r["reference"] for r in rows)
    assert result["bleu_signature"] is not None


def test_summary_rejects_different_splits(tmp_path: Path):
    for name, fingerprint in [("a", "x"), ("b", "y")]:
        folder = tmp_path / name / "seed-1"
        folder.mkdir(parents=True)
        (folder / "result.json").write_text(json.dumps({
            "name": name, "seed": 1, "data_fingerprint": fingerprint,
            "metrics": {"nll": 1.0},
        }))
    with pytest.raises(ValueError, match="Comparability failure"):
        summarize(tmp_path)
