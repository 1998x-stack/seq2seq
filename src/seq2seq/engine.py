from __future__ import annotations

import json
import os
import platform
import random
import time
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.nn import functional as F

from seq2seq.config import validate_config
from seq2seq.data import DataBundle, make_loader, prepare_data
from seq2seq.diagnostics import export_jsonl, summarize_predictions
from seq2seq.losses import masked_cross_entropy, perplexity
from seq2seq.model import Seq2Seq, build_model
from seq2seq.telemetry import RunLogger, export_metrics_csv


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
    optimizer: torch.optim.Optimizer | None = None,
    on_step: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    sum_nll, correct, tokens = 0.0, 0, 0
    started = time.perf_counter()
    for step, batch in enumerate(loader, 1):
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
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(cfg["training"].get("grad_clip", 1.0))
                )
                optimizer.step()
                if on_step is not None:
                    on_step({
                        "step": step, "batch_tokens": count,
                        "batch_loss": float(loss.detach().item()),
                        "grad_norm": float(grad_norm.item()),
                        "elapsed_seconds": time.perf_counter() - started,
                    })
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
    elapsed = time.perf_counter() - started
    return {
        "elapsed_seconds": elapsed,
        "tokens_per_second": tokens / max(elapsed, 1e-12),
        "nll": mean_nll,
        "perplexity": perplexity(mean_nll),
        "token_accuracy": correct / tokens,
        "tokens": tokens,
    }


@torch.inference_mode()
def generation_metrics(
    model: Seq2Seq, bundle: DataBundle, cfg: dict, split: str,
    device: torch.device, max_examples: int = 200,
    predictions_path: str | Path | None = None,
) -> dict[str, Any]:
    if split not in ("validation", "test", "train"):
        raise ValueError(f"Unknown data split: {split}")
    if max_examples < -1:
        raise ValueError("max_examples must be -1 (all), 0 (skip), or positive")
    if max_examples == 0:
        return summarize_predictions([])

    references = getattr(bundle, split)
    loader = make_loader(bundle, split, cfg, shuffle=False)
    tokenizer = bundle.tokenizer
    model.eval()
    rows: list[dict[str, Any]] = []
    for batch in loader:
        data = to_device(batch, device)
        output = model.generate(
            data["src"], data["src_lengths"],
            bos_id=bundle.tgt_vocab.bos_id, eos_id=bundle.tgt_vocab.eos_id,
            max_new_tokens=int(cfg["training"]["max_new_tokens"]),
        )
        for prediction in output.tolist():
            index = len(rows)
            src_raw, ref_raw = references[index]
            src_tokens = tokenizer.tokenize(src_raw)
            tgt_tokens = tokenizer.tokenize(ref_raw)
            hyp = tokenizer.detokenize(bundle.tgt_vocab.decode(prediction))
            # Do not decode reference ids: unknown gold words would become <unk>.
            ref = tokenizer.detokenize(tgt_tokens)
            src_unk = sum(token not in bundle.src_vocab.stoi for token in src_tokens)
            tgt_unk = sum(token not in bundle.tgt_vocab.stoi for token in tgt_tokens)
            rows.append({
                "index": index, "source": src_raw, "reference": ref,
                "prediction": hyp, "exact_match": hyp == ref,
                "source_length": len(src_tokens), "target_length": len(tgt_tokens),
                "source_oov_count": src_unk, "target_oov_count": tgt_unk,
            })
            if max_examples > 0 and len(rows) >= max_examples:
                break
        if max_examples > 0 and len(rows) >= max_examples:
            break
    if predictions_path is not None:
        export_jsonl(predictions_path, rows)
    return summarize_predictions(rows)


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
            "python_rng": random.getstate(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
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
        if "python_rng" in prior:
            random.setstate(prior["python_rng"])
        if prior.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(prior["cuda_rng"])

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.yaml").write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    def pkgver(package: str) -> str | None:
        try:
            return version(package)
        except PackageNotFoundError:
            return None

    metadata = {
        "seed": seed,
        "split_info": bundle.split_info,
        "dataset": cfg["dataset"],
        "environment": {
            "python": platform.python_version(), "platform": platform.platform(),
            "torch": torch.__version__, "datasets": pkgver("datasets"),
            "sacrebleu": pkgver("sacrebleu"),
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        }, "device": str(device), "data_fingerprint": bundle.fingerprint,
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
    logging_cfg = cfg.get("logging", {})
    log_every = int(logging_cfg.get("log_every_steps", 20))
    enabled = bool(logging_cfg.get("enabled", True))
    with RunLogger(
        output, name=output.name,
        tensorboard=bool(logging_cfg.get("tensorboard", False)) if enabled else False,
        console=enabled,
    ) as logger:
        if enabled:
            logger.emit(
                "training_started", seed=seed, epochs=int(cfg["training"]["epochs"]),
                starting_epoch=start_epoch, device=str(device),
                data_fingerprint=bundle.fingerprint, parameters=metadata["parameters"],
            )
        try:
            for epoch in range(start_epoch, int(cfg["training"]["epochs"]) + 1):
                if enabled:
                    logger.emit("epoch_started", epoch=epoch)
                def step_event(values: dict[str, Any], current_epoch: int = epoch) -> None:
                    if not enabled or log_every <= 0:
                        return
                    if values["step"] % log_every != 0:
                        return
                    global_step = (current_epoch - 1) * len(train_loader) + values["step"]
                    logger.emit("train_step", epoch=current_epoch, global_step=global_step, **values)
                    logger.scalar("train/batch_loss", values["batch_loss"], global_step)
                    logger.scalar("train/grad_norm", values["grad_norm"], global_step)

                train_metrics = run_epoch(
                    model, train_loader, device, cfg, optimizer, on_step=step_event
                )
                val_metrics = run_epoch(model, val_loader, device, cfg)
                is_best = val_metrics["nll"] < best_nll
                if is_best:
                    best_nll = val_metrics["nll"]
                lr = float(optimizer.param_groups[0]["lr"])
                row = {
                    "epoch": epoch, "train": train_metrics, "validation": val_metrics,
                    "learning_rate": lr, "best_validation_nll": best_nll,
                }
                save_checkpoint(
                    output / "last.pt", model, optimizer, cfg, bundle, epoch, best_nll
                )
                if is_best:
                    save_checkpoint(
                        output / "best.pt", model, optimizer, cfg, bundle, epoch, best_nll
                    )
                # Append one epoch only. Detect duplicate records on resume.
                existing_epochs = set()
                metrics_path = output / "metrics.jsonl"
                if metrics_path.exists():
                    existing_epochs = {
                        int(json.loads(line)["epoch"])
                        for line in metrics_path.read_text(encoding="utf-8").splitlines()
                        if line.strip()
                    }
                if epoch in existing_epochs:
                    raise ValueError(f"metrics.jsonl already contains epoch {epoch}")
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                export_metrics_csv(output)
                if enabled:
                    logger.emit("epoch_completed", epoch=epoch, best=is_best,
                                train=train_metrics, validation=val_metrics,
                                learning_rate=lr, best_validation_nll=best_nll)
                    for split, metrics in (("train", train_metrics), ("validation", val_metrics)):
                        for name in ("nll", "perplexity", "token_accuracy", "tokens_per_second"):
                            logger.scalar(f"{split}/{name}", metrics[name], epoch)
                print(json.dumps(row, ensure_ascii=False), flush=True)
                latest = row
            if enabled:
                logger.emit("training_completed", final_epoch=(
                    latest.get("epoch", start_epoch - 1)
                ))
        except Exception as exc:
            if enabled:
                logger.emit("training_failed", error=repr(exc))
            raise
    if latest:
        latest["output_dir"] = str(output)
    return latest


def evaluate(
    checkpoint_path: str | Path, split: str = "test", max_examples: int = 200,
    device_name: str = "auto", predictions_path: str | Path | None = None
) -> dict[str, Any]:
    device = choose_device(device_name)
    model, checkpoint = restore_model(checkpoint_path, device)
    cfg = checkpoint["config"]
    bundle = prepare_data(cfg, vocab_data=checkpoint["vocab"])
    if bundle.fingerprint != checkpoint["fingerprint"]:
        raise ValueError("Dataset fingerprint mismatch: evaluation aborted")
    loader = make_loader(bundle, split, cfg)
    result = run_epoch(model, loader, device, cfg)
    result.update(generation_metrics(
        model, bundle, cfg, split, device, max_examples, predictions_path
    ))
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
