"""Factory contracts, paper-derived factors and offline integration."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from seq2seq.config import load_config
from seq2seq.data import make_loader, prepare_data
from seq2seq.factory.batching import LengthBucketBatchSampler
from seq2seq.factory.components import (
    DecoderFactory, LossFactory, ModelFactory, OptimizerFactory, SchedulerFactory,
)
from seq2seq.factory.experiments import factory_plan, factory_run
from seq2seq.factory.registry import Registry
from seq2seq.model import build_model
from seq2seq.experiments import summarize


def config():
    return load_config("configs/smoke.yaml")


def test_registry_strict_and_factory_model_initialization():
    reg = Registry("test")
    reg.register("x", lambda k: k + 1)
    assert reg.create("x", k=4) == 5
    with pytest.raises(ValueError, match="Unknown test"):
        reg.create("typo")
    with pytest.raises(ValueError, match="Duplicate"):
        reg.register("x", lambda: None)

    cfg = config()
    cfg["model"]["cell"] = "rnn"
    cfg["model"]["initialization"] = "uniform"
    cfg["model"]["init_range"] = 0.08
    model = build_model(cfg, 16, 17)
    assert isinstance(model, torch.nn.Module)
    assert isinstance(model.encoder.rnn, torch.nn.RNN)
    assert isinstance(model.decoder.rnn, torch.nn.RNN)
    assert max(abs(float(p.detach().max())) for p in model.parameters()) <= 0.08001
    assert min(float(p.detach().min()) for p in model.parameters()) >= -0.08001
    assert torch.count_nonzero(model.encoder.embedding.weight[0]) == 0

    cfg["model"]["cell"] = "lstm"
    lstm = ModelFactory.create(cfg, 16, 17)
    assert isinstance(lstm.encoder.rnn, torch.nn.LSTM)


def test_sequence_nll_vs_token_ce_weighting():
    # Zero logits -> every gold token has NLL=log(V).
    logits = torch.zeros(2, 3, 5, requires_grad=True)
    targets = torch.tensor([[2, 0, 0], [2, 3, 4]])
    cfg = config()
    ce, valid = LossFactory.create(cfg)(logits, targets)
    assert valid == 4
    assert ce.item() == pytest.approx(torch.log(torch.tensor(5.)).item())
    cfg["training"]["loss"] = "sequence_nll"
    seq, count = LossFactory.create(cfg)(logits, targets)
    assert count == 4
    assert seq.item() == pytest.approx(2 * ce.item())
    seq.backward()
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad[0, 1]) == 0


def test_optimizer_and_half_epoch_schedule():
    cfg = config()
    cfg["training"].update({
        "optimizer": "sgd", "momentum": 0.0,
        "learning_rate": 0.7, "weight_decay": 0.0,
        "scheduler": {
            "name": "paper_halving", "begin_epoch": 5.0,
            "interval_epochs": 0.5, "gamma": 0.5,
        },
    })
    model = build_model(cfg, 15, 15)
    opt = OptimizerFactory.create(model.parameters(), cfg)
    assert isinstance(opt, torch.optim.SGD)
    assert SchedulerFactory.step(opt, cfg, 5.0) == pytest.approx(0.7)
    assert SchedulerFactory.step(opt, cfg, 5.5) == pytest.approx(0.35)
    assert SchedulerFactory.step(opt, cfg, 6.0) == pytest.approx(0.175)
    assert opt.param_groups[0]["lr"] == pytest.approx(0.175)


def test_bucket_sampler_epoch_and_integrity():
    cfg = config()
    cfg["training"]["batching"] = "length_bucket"
    cfg["training"]["batch_size"] = 4
    bundle = prepare_data(cfg)
    loader = make_loader(bundle, "train", cfg, shuffle=True)
    sampler = loader.batch_sampler
    assert isinstance(sampler, LengthBucketBatchSampler)
    sampler.set_epoch(0)
    batches_0 = list(iter(sampler))
    sampler.set_epoch(0)
    assert batches_0 == list(iter(sampler))
    sampler.set_epoch(1)
    assert batches_0 != list(iter(sampler))
    indices = [index for batch in batches_0 for index in batch]
    assert len(indices) == len(set(indices)) == len(bundle.train)


@pytest.mark.parametrize("cell", ["rnn", "lstm"])
def test_decode_factory_and_beam1_equivalence(cell):
    cfg = config()
    cfg["model"]["cell"] = cell
    model = build_model(cfg, 21, 23).eval()
    source = torch.tensor([[5, 3, 0], [8, 9, 3]])
    lengths = torch.tensor([2, 3])
    cfg["decoding"] = {"strategy": "beam", "beam_size": 1}
    greedy = DecoderFactory.create(cfg)(
        model, source, lengths, max_new_tokens=6,
    )
    cfg["decoding"]["beam_size"] = 3
    beam = DecoderFactory.create(cfg)(
        model, source, lengths, max_new_tokens=6,
    )
    assert greedy.shape[0] == beam.shape[0] == 2
    assert 1 <= beam.shape[1] <= 6
    assert not torch.any(beam == 0)
    cfg["decoding"] = {"strategy": "greedy"}
    same = DecoderFactory.create(cfg)(
        model, source, lengths, max_new_tokens=6,
    )
    assert torch.equal(greedy, same)


def test_manifest_paper_provenance_and_factor_plan():
    plan = factory_plan("configs/factory/smoke_factory.yaml")
    assert [item["name"] for item in plan] == ["baseline", "reversed"]
    assert plan[0]["config"]["training"]["loss"] == "sequence_nll"
    assert plan[0]["config"]["decoding"]["strategy"] == "beam"
    assert plan[1]["config"]["model"]["reverse_source"]
    paper = load_config("configs/factory/paper_reference_architecture.yaml")
    assert paper["model"]["num_layers"] == 4
    assert paper["model"]["attention"] is False
    assert paper["model"]["hidden_dim"] == 1000
    assert paper["tokenizer"]["src_vocab_size"] == 160000
    assert paper["tokenizer"]["tgt_vocab_size"] == 80000


def test_factory_end_to_end_offline_and_manifest_lock(tmp_path: Path):
    root = tmp_path / "study"
    summary = factory_run(
        "configs/factory/smoke_factory.yaml", root,
        max_examples=4, limit=1, split="validation",
    )
    assert summary["completed_runs"] == 1
    assert summary["expected_runs"] == 2
    first = root / "baseline" / "seed-7"
    assert (first / "best.pt").is_file()
    assert (first / "metrics.csv").is_file()
    assert (first / "predictions.jsonl").is_file()
    result = json.loads((first / "result.json").read_text(encoding="utf-8"))
    assert result["evaluation_split"] == "validation"
    assert result["metrics"]["generated_examples"] == 4
    assert len((root / "events.jsonl").read_text().splitlines()) >= 3
    again = factory_run(
        "configs/factory/smoke_factory.yaml", root,
        max_examples=4, limit=1, split="validation",
    )
    assert again["completed_runs"] == 1
    assert summarize(root)["data_fingerprint"] == summary["data_fingerprint"]
    with pytest.raises(ValueError, match="Existing result"):
        factory_run(
            "configs/factory/smoke_factory.yaml", root,
            max_examples=4, limit=1, split="test",
        )
