from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from seq2seq.text import Tokenizer, Vocabulary


@dataclass
class DataBundle:
    train: list[tuple[str, str]]
    validation: list[tuple[str, str]]
    test: list[tuple[str, str]]
    src_vocab: Vocabulary
    tgt_vocab: Vocabulary
    tokenizer: Tokenizer
    fingerprint: str


def _toy_pairs(seed: int, count: int) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    alphabet = ["red", "blue", "green", "circle", "square", "north", "south", "one", "two", "three"]
    return [
        (sequence := " ".join(rng.choices(alphabet, k=rng.randint(3, 8))), sequence)
        for _ in range(count)
    ]


def prepare_data(cfg: dict, vocab_data: dict | None = None) -> DataBundle:
    dc, tc = cfg["dataset"], cfg["tokenizer"]
    seed = int(cfg.get("seed", 42))
    tokenizer = Tokenizer(lowercase=bool(tc.get("lowercase", True)))
    if dc["backend"] == "toy":
        pairs = _toy_pairs(seed, int(dc.get("toy_samples", 1200)))
    else:
        # Import lazily so the offline toy and unit tests do not need network access.
        from datasets import load_dataset

        kw = {"revision": dc["revision"]} if dc.get("revision") else {}
        rows = load_dataset(dc["name"], dc["subset"], split=dc.get("split", "train"), **kw)
        source, target = dc["source_lang"], dc["target_lang"]
        pairs = [(row["translation"][source], row["translation"][target]) for row in rows]

    min_len, max_len = int(dc.get("min_length", 1)), int(dc["max_length"])
    clean = []
    seen = set()
    for src, tgt in pairs:
        if not isinstance(src, str) or not isinstance(tgt, str):
            continue
        a, b = tokenizer.tokenize(src), tokenizer.tokenize(tgt)
        if not (min_len <= len(a) <= max_len and min_len <= len(b) <= max_len):
            continue
        s, t = " ".join(a), " ".join(b)
        if (s, t) in seen:
            continue
        seen.add((s, t))
        clean.append((src.strip(), tgt.strip()))

    rng = random.Random(seed)
    rng.shuffle(clean)
    if int(dc.get("max_samples", 0)):
        clean = clean[: int(dc["max_samples"])]
    if len(clean) < 10:
        raise ValueError("At least 10 valid unique sequence pairs are required")
    nv = max(1, round(len(clean) * float(dc.get("validation_fraction", 0.05))))
    nt = max(1, round(len(clean) * float(dc.get("test_fraction", 0.05))))
    if nv + nt >= len(clean):
        raise ValueError("No training examples remain after splitting")
    test, validation, train = clean[:nt], clean[nt : nt + nv], clean[nt + nv :]

    if vocab_data is None:
        src_vocab = Vocabulary.build(
            [tokenizer.tokenize(src) for src, _ in train],
            int(tc.get("vocab_size", 12000)),
            int(tc.get("min_freq", 1)),
        )
        tgt_vocab = Vocabulary.build(
            [tokenizer.tokenize(tgt) for _, tgt in train],
            int(tc.get("vocab_size", 12000)),
            int(tc.get("min_freq", 1)),
        )
    else:
        src_vocab = Vocabulary(vocab_data["src"])
        tgt_vocab = Vocabulary(vocab_data["tgt"])
    fingerprint = hashlib.sha256(
        json.dumps(clean, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return DataBundle(train, validation, test, src_vocab, tgt_vocab, tokenizer, fingerprint)


class PairDataset(Dataset):
    def __init__(
        self, pairs: list[tuple[str, str]], tokenizer: Tokenizer,
        src_vocab: Vocabulary, tgt_vocab: Vocabulary, reverse_source: bool = False
    ) -> None:
        self.items: list[tuple[list[int], list[int]]] = []
        for src, tgt in pairs:
            src_tokens, tgt_tokens = tokenizer.tokenize(src), tokenizer.tokenize(tgt)
            if reverse_source:
                src_tokens.reverse()
            a = src_vocab.encode(src_tokens) + [src_vocab.eos_id]
            b = [tgt_vocab.bos_id] + tgt_vocab.encode(tgt_tokens) + [tgt_vocab.eos_id]
            self.items.append((a, b))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[list[int], list[int]]:
        return self.items[index]


def collate_pairs(
    batch: list[tuple[list[int], list[int]]], src_pad: int, tgt_pad: int
) -> dict[str, torch.Tensor]:
    from torch.nn.utils.rnn import pad_sequence

    src = [torch.tensor(a, dtype=torch.long) for a, _ in batch]
    targets = [torch.tensor(b, dtype=torch.long) for _, b in batch]
    return {
        "src": pad_sequence(src, batch_first=True, padding_value=src_pad),
        "src_lengths": torch.tensor([len(a) for a, _ in batch], dtype=torch.long),
        "tgt_in": pad_sequence([b[:-1] for b in targets], batch_first=True, padding_value=tgt_pad),
        "tgt_out": pad_sequence([b[1:] for b in targets], batch_first=True, padding_value=tgt_pad),
    }


def make_loader(bundle: DataBundle, split: str, cfg: dict, shuffle: bool = False) -> DataLoader:
    from functools import partial

    ds = PairDataset(
        getattr(bundle, split), bundle.tokenizer, bundle.src_vocab, bundle.tgt_vocab,
        reverse_source=bool(cfg["model"].get("reverse_source", False)),
    )
    return DataLoader(
        ds,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=shuffle,
        num_workers=int(cfg["training"].get("num_workers", 0)),
        collate_fn=partial(
            collate_pairs, src_pad=bundle.src_vocab.pad_id, tgt_pad=bundle.tgt_vocab.pad_id
        ),
    )
