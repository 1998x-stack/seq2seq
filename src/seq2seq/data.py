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
    split_info: dict


def _toy_pairs(seed: int, count: int) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    alphabet = ["red", "blue", "green", "circle", "square", "north", "south", "one", "two", "three"]
    return [
        (sequence := " ".join(rng.choices(alphabet, k=rng.randint(3, 8))), sequence)
        for _ in range(count)
    ]


def prepare_data(cfg: dict, vocab_data: dict | None = None) -> DataBundle:
    dc, tc = cfg["dataset"], cfg["tokenizer"]
    seed = int(dc.get("split_seed", cfg.get("seed", 42)))
    tokenizer = Tokenizer(lowercase=bool(tc.get("lowercase", True)))
    if dc["backend"] == "toy":
        pairs = _toy_pairs(seed, int(dc.get("toy_samples", 1200)))
        group_ids = [None] * len(pairs)
    elif dc["backend"] == "jsonl":
        source, target = dc["source_lang"], dc["target_lang"]
        pairs, group_ids = [], []
        with Path(dc["path"]).open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                translation = row.get("translation", row)
                pairs.append((translation[source], translation[target]))
                group_ids.append(row.get(dc.get("group_field", "group_id")))
    else:
        # Import lazily so the offline toy and unit tests do not need network access.
        from datasets import load_dataset

        kw = {"revision": dc["revision"]} if dc.get("revision") else {}
        rows = load_dataset(dc["name"], dc["subset"], split=dc.get("split", "train"), **kw)
        source, target = dc["source_lang"], dc["target_lang"]
        pairs = [(row["translation"][source], row["translation"][target]) for row in rows]
        group_ids = [row.get(dc.get("group_field", "group_id")) for row in rows]

    min_len, max_len = int(dc.get("min_length", 1)), int(dc["max_length"])
    clean = []
    clean_groups = []
    seen = set()
    for (src, tgt), group in zip(pairs, group_ids):
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
        clean_groups.append(group)

    rng = random.Random(seed)
    joined = list(zip(clean, clean_groups))
    rng.shuffle(joined)
    if int(dc.get("max_samples", 0)):
        joined = joined[: int(dc["max_samples"])]
    if len(joined) < 10:
        raise ValueError("At least 10 valid unique sequence pairs are required")
    strategy = dc.get("split_strategy", "random")
    if strategy == "group":
        if any(group is None or str(group).strip() == "" for _, group in joined):
            raise ValueError("group split requires nonempty group_field for every row; "
                             "OPUS Books has no book_id, and its row id is not a document ID")
        grouped = {}
        for pair, group in joined:
            grouped.setdefault(str(group), []).append(pair)
        keys = list(grouped)
        rng.shuffle(keys)
        if len(keys) < 3:
            raise ValueError("group split requires at least three distinct groups")
        nt = max(1, round(len(keys) * float(dc.get("test_fraction", 0.05))))
        nv = max(1, round(len(keys) * float(dc.get("validation_fraction", 0.05))))
        if nt + nv >= len(keys):
            raise ValueError("No train groups remain")
        test_keys = keys[:nt]
        val_keys = keys[nt:nt+nv]
        train_keys = keys[nt+nv:]
        test = [pair for k in test_keys for pair in grouped[k]]
        validation = [pair for k in val_keys for pair in grouped[k]]
        train = [pair for k in train_keys for pair in grouped[k]]
        split_info = {
            "strategy": "group", "split_seed": seed,
            "groups": {"train": len(train_keys), "validation": len(val_keys), "test": len(test_keys)},
            "group_overlap": 0,
        }
    else:
        clean = [pair for pair, _ in joined]
        nv = max(1, round(len(clean) * float(dc.get("validation_fraction", 0.05))))
        nt = max(1, round(len(clean) * float(dc.get("test_fraction", 0.05))))
        if nv + nt >= len(clean):
            raise ValueError("No training examples remain after splitting")
        test, validation, train = clean[:nt], clean[nt:nt+nv], clean[nt+nv:]
        split_info = {"strategy": "random", "split_seed": seed, "group_overlap": None}

    if vocab_data is None:
        src_vocab = Vocabulary.build(
            [tokenizer.tokenize(src) for src, _ in train],
            int(tc.get("src_vocab_size", tc.get("vocab_size", 12000))),
            int(tc.get("min_freq", 1)),
        )
        tgt_vocab = Vocabulary.build(
            [tokenizer.tokenize(tgt) for _, tgt in train],
            int(tc.get("tgt_vocab_size", tc.get("vocab_size", 12000))),
            int(tc.get("min_freq", 1)),
        )
    else:
        src_vocab = Vocabulary(vocab_data["src"])
        tgt_vocab = Vocabulary(vocab_data["tgt"])
    fingerprint = hashlib.sha256(
        json.dumps(
            {"train": train, "validation": validation, "test": test, "split_info": split_info},
            ensure_ascii=False, separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return DataBundle(train, validation, test, src_vocab, tgt_vocab, tokenizer, fingerprint, split_info)


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
    from seq2seq.factory.batching import DataLoaderFactory

    return DataLoaderFactory.create(
        ds, cfg,
        partial(
            collate_pairs, src_pad=bundle.src_vocab.pad_id,
            tgt_pad=bundle.tgt_vocab.pad_id,
        ),
        shuffle=shuffle,
    )
