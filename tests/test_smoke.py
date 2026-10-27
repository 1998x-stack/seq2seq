from pathlib import Path

import pytest

from seq2seq.config import load_config
from seq2seq.engine import evaluate, train, translate


def test_end_to_end_offline(tmp_path: Path):
    cfg = load_config("configs/smoke.yaml")
    out = tmp_path / "smoke"
    result = train(cfg, out)
    assert (out / "best.pt").exists()
    assert (out / "last.pt").exists()
    assert (out / "metrics.jsonl").exists()
    assert result["epoch"] == 1
    evaluated = evaluate(out / "best.pt", max_examples=4, device_name="cpu")
    assert 0 <= evaluated["token_accuracy"] <= 1
    assert evaluated["generated_examples"] == 4
    translated = translate(out / "best.pt", ["red green circle", "blue square"], "cpu")
    assert len(translated) == 2


def test_resume_raises_when_config_changes(tmp_path: Path):
    cfg = load_config("configs/smoke.yaml")
    train(cfg, tmp_path / "baseline")
    cfg["training"]["epochs"] = 2
    # For exact-resume research experiments, config changes are intentionally blocked.
    with pytest.raises(ValueError, match="Resume config"):
        train(cfg, tmp_path / "baseline", str(tmp_path / "baseline" / "last.pt"))
