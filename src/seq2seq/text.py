from __future__ import annotations

import re
from collections import Counter

PAD, UNK, BOS, EOS = "<pad>", "<unk>", "<bos>", "<eos>"
SPECIALS = (PAD, UNK, BOS, EOS)


class Tokenizer:
    def __init__(self, lowercase: bool = True) -> None:
        self.lowercase = lowercase

    def tokenize(self, text: str) -> list[str]:
        if self.lowercase:
            text = text.lower()
        return re.findall(r"\w+|[^\w\s]", text.strip(), flags=re.UNICODE)

    def detokenize(self, tokens: list[str]) -> str:
        result = " ".join(token for token in tokens if token not in SPECIALS)
        result = re.sub(r"\s+([.,!?;:%)\]}])", r"\1", result)
        result = re.sub(r"([(\[{])\s+", r"\1", result)
        return result.strip()


class Vocabulary:
    def __init__(self, tokens: list[str]) -> None:
        if tokens[:4] != list(SPECIALS) or len(tokens) != len(set(tokens)):
            raise ValueError("Vocabulary must start with four unique special tokens")
        self.itos = tokens
        self.stoi = {token: i for i, token in enumerate(tokens)}

    @classmethod
    def build(
        cls, sentences: list[list[str]], max_size: int, min_freq: int = 1
    ) -> Vocabulary:
        if max_size < len(SPECIALS):
            raise ValueError("max_size must be at least four")
        count = Counter(token for sentence in sentences for token in sentence if token not in SPECIALS)
        ranked = sorted(
            ((token, freq) for token, freq in count.items() if freq >= min_freq),
            key=lambda item: (-item[1], item[0]),
        )
        return cls(list(SPECIALS) + [token for token, _ in ranked[: max_size - 4]])

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.stoi.get(token, self.unk_id) for token in tokens]

    def decode(self, ids: list[int]) -> list[str]:
        result = []
        for index in ids:
            if index == self.eos_id:
                break
            if index not in (self.pad_id, self.bos_id):
                result.append(self.itos[index] if 0 <= index < len(self.itos) else UNK)
        return result

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def unk_id(self) -> int:
        return 1

    @property
    def bos_id(self) -> int:
        return 2

    @property
    def eos_id(self) -> int:
        return 3

    def __len__(self) -> int:
        return len(self.itos)
