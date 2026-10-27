from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.nn import functional as F

from seq2seq.config import validate_config
from seq2seq.data import DataBundle, make_loader, prepare_data
from seq2seq.losses import masked_cross_entropy, perplexity
from seq2seq.model import Seq2Seq, build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected = torch.device(device)
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return selected


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def run_epoch(
    model: Seq2Seq, loader: Any, device: torch.device, cfg: dict,
    optimizer: torch.optim.Optimizer | None = None
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    sum_nll, correct, tokens = 0.0, 0, 0
    for batch in loader:
        batch = to_device(batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(
                batch["src"], batch["src_lengths"], batch["tgt_in"],
                teacher_forcing=float(cfg["training"].get("teacher_forcing", 1.0)) if training else 1.0,
            )
            loss, count = masked_cross_entropy(
                logits, batch["tgt_out"], pad_id=model.tgt_pad,
                label_smoothing=float(cfg["training"].get("label_smoothing", 0.0)) if training else 0.0,
            )
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(cfg["training"].get("grad_clip", 1.0))
                )
                optimizer.step()
        with torch.no_grad():
            nll = F.cross_entropy(
                logits.detach().reshape(-1, logits.size(-1)),
                batch["tgt_out"].reshape(-1),
                ignore_index=model.tgt_pad, reduction="sum",
            )
            mask = batch["tgt_out"].ne(model.tgt_pad)
            correct += int(((logits.argmax(-1) == batch["tgt_out"]) & mask).sum().item())
            sum_nll += float(nll.item())
            tokens += count
    if tokens == 0:
        raise ValueError("Empty dataloader")
    mean_nll = sum_nll / tokens
    return {
        "nll": mean_nll,
        "perplexity": perplexity(mean_nll),
        "token_accuracy": correct / tokens,
        "tokens": tokens,
    }


@torch.inference_mode()
def generation_metrics(
    model: Seq2Seq, bundle: DataBundle, cfg: dict, split: str,
    device: torch.device, max_examples: int = 200
) -> dict[str, Any]:
    import sacrebleu

    if split not in ("validation", "test", "train"):
        raise ValueError(f"Unknown data split: {split}")
    if max_examples < 0:
        raise ValueError("max_examples must be nonnegative")
    if max_examples == 0:
        return {"bleu": None, "exact_match": None, "generated_examples": 0, "examples": []}
    loader = make_loader(bundle, split, cfg, shuffle=False)
    model.eval()
    hypotheses, references, samples = [], [], []
    for batch in loader:
        data = to_device(batch, device)
        output = model.generate(
            data["src"], data["src_lengths"],
            bos_id=bundle.tgt_vocab.bos_id, eos_id=bundle.tgt_vocab.eos_id,
            max_new_tokens=int(cfg["training"]["max_new_tokens"]),
        )
        for prediction, gold in zip(output.tolist(), batch["tgt_out"].tolist()):
            hyp = bundle.tokenizer.detokenize(bundle.tgt_vocab.decode(prediction))
            ref = bundle.tokenizer.detokenize(bundle.tgt_vocab.decode(gold))
            hypotheses.append(hyp)
            references.append(ref)
            if len(samples) < 5:
                samples.append({"reference": ref, "prediction": hyp})
            if len(hypotheses) >= max_examples:
                break
        if len(hypotheses) >= max_examples:
            break
    bleu = sacrebleu.corpus_bleu(hypotheses, [references]).score
    exact = sum(hyp == ref for hyp, ref in zip(hypotheses, references)) / len(hypotheses)
    return {
        "bleu": bleu, "exact_match": exact,
        "generated_examples": len(hypotheses), "examples": samples,
    }


def _vocab_payload(bundle: DataBundle) -> dict[str, list[str]]:
    return {"src": bundle.src_vocab.itos, "tgt": bundle.tgt_vocab.itos}


def save_checkpoint(
    path: Path, model: Seq2Seq, optimizer: torch.optim.Optimizer, cfg: dict,
    bundle: DataBundle, epoch: int, best_nll: float
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg,
            "vocab": _vocab_payload(bundle),
            "fingerprint": bundle.fingerprint,
            "epoch": epoch,
            "best_nll": best_nll,
            "torch_rng": torch.get_rng_state(),
        },
        temporary,
    )
    os.replace(temporary, path)


def read_checkpoint(path: str | Path, device: torch.device) -> dict:
    # This is a pickle-based artifact. Load only checkpoints from trusted sources.
    return torch.load(path, map_location=device, weights_only=False)


def restore_model(path: str | Path, device: torch.device) -> tuple[Seq2Seq, dict]:
    checkpoint = read_checkpoint(path, device)
    cfg = checkpoint["config"]
    validate_config(cfg)
    model = build_model(
        cfg, len(checkpoint["vocab"]["src"]), len(checkpoint["vocab"]["tgt"])
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model, checkpoint


def train(
    cfg: dict, output_dir: str | Path, resume: str | None = None
) -> dict[str, Any]:
    validate_config(cfg)
    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    device = choose_device(str(cfg["training"].get("device", "auto")))
    prior = read_checkpoint(resume, device) if resume else None
    if prior is not None and cfg != prior["config"]:
        raise ValueError("Resume config must exactly match the saved configuration")

    bundle = prepare_data(cfg, vocab_data=prior["vocab"] if prior else None)
    if prior is not None and bundle.fingerprint != prior["fingerprint"]:
        raise ValueError("Dataset fingerprint changed: refusing unsafe resume")
    train_loader = make_loader(bundle, "train", cfg, shuffle=True)
    val_loader = make_loader(bundle, "validation", cfg)
    model = build_model(cfg, len(bundle.src_vocab), len(bundle.tgt_vocab)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"].get("weight_decay", 0.0)),
    )
    start_epoch, best_nll = 1, float("inf")
    if prior is not None:
        model.load_state_dict(prior["model"])
        optimizer.load_state_dict(prior["optimizer"])
        start_epoch = int(prior["epoch"]) + 1
        best_nll = float(prior["best_nll"])
        torch.set_rng_state(prior["torch_rng"].cpu())

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.yaml").write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    metadata = {
        "seed": seed, "device": str(device), "data_fingerprint": bundle.fingerprint,
        "train_pairs": len(bundle.train), "validation_pairs": len(bundle.validation),
        "test_pairs": len(bundle.test), "src_vocab": len(bundle.src_vocab),
        "tgt_vocab": len(bundle.tgt_vocab),
        "parameters": sum(p.numel() for p in model.parameters()),
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False), flush=True)

    latest: dict[str, Any] = {}
    for epoch in range(start_epoch, int(cfg["training"]["epochs"]) + 1):
        train_metrics = run_epoch(model, train_loader, device, cfg, optimizer)
        val_metrics = run_epoch(model, val_loader, device, cfg)
        row = {"epoch": epoch, "train": train_metrics, "validation": val_metrics}
        is_best = val_metrics["nll"] < best_nll
        if is_best:
            best_nll = val_metrics["nll"]
        save_checkpoint(output / "last.pt", model, optimizer, cfg, bundle, epoch, best_nll)
        if is_best:
            save_checkpoint(output / "best.pt", model, optimizer, cfg, bundle, epoch, best_nll)
        with (output / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps(row, ensure_ascii=False), flush=True)
        latest = row
    if latest:
        latest["output_dir"] = str(output)
    return latest


def evaluate(
    checkpoint_path: str | Path, split: str = "test", max_examples: int = 200,
    device_name: str = "auto"
) -> dict[str, Any]:
    device = choose_device(device_name)
    model, checkpoint = restore_model(checkpoint_path, device)
    cfg = checkpoint["config"]
    bundle = prepare_data(cfg, vocab_data=checkpoint["vocab"])
    if bundle.fingerprint != checkpoint["fingerprint"]:
        raise ValueError("Dataset fingerprint mismatch: evaluation aborted")
    loader = make_loader(bundle, split, cfg)
    result = run_epoch(model, loader, device, cfg)
    result.update(generation_metrics(model, bundle, cfg, split, device, max_examples))
    result["split"] = split
    result["checkpoint"] = str(checkpoint_path)
    return result


@torch.inference_mode()
def translate(
    checkpoint_path: str | Path, texts: list[str], device_name: str = "auto"
) -> list[str]:
    device = choose_device(device_name)
    model, checkpoint = restore_model(checkpoint_path, device)
    cfg = checkpoint["config"]
    from seq2seq.text import Tokenizer, Vocabulary

    tokenizer = Tokenizer(lowercase=bool(cfg["tokenizer"].get("lowercase", True)))
    src_vocab = Vocabulary(checkpoint["vocab"]["src"])
    tgt_vocab = Vocabulary(checkpoint["vocab"]["tgt"])
    if not texts:
        return []
    encoded = []
    for sentence in texts:
        words = tokenizer.tokenize(sentence)[: int(cfg["dataset"]["max_length"])]
        if cfg["model"].get("reverse_source", False):
            words.reverse()
        encoded.append(torch.tensor(src_vocab.encode(words) + [src_vocab.eos_id]))
    from torch.nn.utils.rnn import pad_sequence

    lengths = torch.tensor([len(t) for t in encoded])
    src = pad_sequence(encoded, batch_first=True, padding_value=src_vocab.pad_id).to(device)
    output = model.generate(
        src, lengths, tgt_vocab.bos_id, tgt_vocab.eos_id,
        int(cfg["training"]["max_new_tokens"]),
    )
    return [
        tokenizer.detokenize(tgt_vocab.decode(row))
        for row in output.tolist()
    ]
