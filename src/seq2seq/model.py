from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

State = torch.Tensor | tuple[torch.Tensor, torch.Tensor]


class Encoder(nn.Module):
    def __init__(
        self, vocab_size: int, embedding_dim: int, hidden_dim: int,
        layers: int, dropout: float, cell: str, pad_id: int
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        recurrent = nn.LSTM if cell == "lstm" else nn.RNN
        self.rnn = recurrent(
            embedding_dim, hidden_dim, num_layers=layers, batch_first=True,
            dropout=dropout if layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, State]:
        embedded = self.dropout(self.embedding(src))
        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        outputs, state = self.rnn(packed)
        outputs, _ = pad_packed_sequence(
            outputs, batch_first=True, total_length=src.size(1)
        )
        return outputs, state


class AdditiveAttention(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(2 * hidden_dim, hidden_dim)
        self.vector = nn.Linear(hidden_dim, 1, bias=False)

    def forward(
        self, query: torch.Tensor, memory: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, length, _ = memory.shape
        combined = torch.cat([query.unsqueeze(1).expand(-1, length, -1), memory], dim=-1)
        energies = self.vector(torch.tanh(self.score(combined))).squeeze(-1)
        weights = torch.softmax(energies.masked_fill(~mask, -1e9), dim=-1)
        context = torch.bmm(weights.unsqueeze(1), memory).squeeze(1)
        return context, weights


class Decoder(nn.Module):
    def __init__(
        self, vocab_size: int, embedding_dim: int, hidden_dim: int,
        layers: int, dropout: float, cell: str, pad_id: int, attention: bool
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.attention = AdditiveAttention(hidden_dim) if attention else None
        recurrent = nn.LSTM if cell == "lstm" else nn.RNN
        self.rnn = recurrent(
            embedding_dim + (hidden_dim if attention else 0),
            hidden_dim, num_layers=layers, batch_first=True,
            dropout=dropout if layers > 1 else 0,
        )
        self.classifier = nn.Linear(hidden_dim * (2 if attention else 1), vocab_size)
        self.dropout = nn.Dropout(dropout)

    def step(
        self, token: torch.Tensor, state: State, memory: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, State]:
        embedded = self.dropout(self.embedding(token)).unsqueeze(1)
        context = None
        if self.attention is not None:
            hidden = state[0][-1] if isinstance(state, tuple) else state[-1]
            context, _ = self.attention(hidden, memory, mask)
            embedded = torch.cat([embedded, context.unsqueeze(1)], dim=-1)
        output, next_state = self.rnn(embedded, state)
        features = output.squeeze(1)
        if context is not None:
            features = torch.cat([features, context], dim=-1)
        return self.classifier(features), next_state


class Seq2Seq(nn.Module):
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, cfg: dict,
                 src_pad: int = 0, tgt_pad: int = 0) -> None:
        super().__init__()
        hidden = int(cfg["hidden_dim"])
        common = dict(
            embedding_dim=int(cfg["embedding_dim"]), hidden_dim=hidden,
            layers=int(cfg["num_layers"]), dropout=float(cfg.get("dropout", 0)),
            cell=cfg["cell"],
        )
        self.encoder = Encoder(src_vocab_size, pad_id=src_pad, **common)
        self.decoder = Decoder(
            tgt_vocab_size, pad_id=tgt_pad, attention=bool(cfg.get("attention", False)),
            **common
        )
        self.src_pad, self.tgt_pad = src_pad, tgt_pad

    def forward(
        self, src: torch.Tensor, lengths: torch.Tensor, tgt_in: torch.Tensor,
        teacher_forcing: float = 1.0
    ) -> torch.Tensor:
        memory, state = self.encoder(src, lengths)
        mask = src.ne(self.src_pad)
        token = tgt_in[:, 0]
        logits = []
        for time in range(tgt_in.size(1)):
            scores, state = self.decoder.step(token, state, memory, mask)
            logits.append(scores)
            if time + 1 < tgt_in.size(1):
                if teacher_forcing >= 1.0:
                    token = tgt_in[:, time + 1]
                elif teacher_forcing <= 0.0:
                    token = scores.argmax(dim=-1)
                else:
                    select_truth = torch.rand(src.size(0), device=src.device) < teacher_forcing
                    token = torch.where(select_truth, tgt_in[:, time + 1], scores.argmax(-1))
        return torch.stack(logits, dim=1)

    @torch.inference_mode()
    def generate(
        self, src: torch.Tensor, lengths: torch.Tensor, bos_id: int = 2,
        eos_id: int = 3, max_new_tokens: int = 64
    ) -> torch.Tensor:
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        memory, state = self.encoder(src, lengths)
        mask = src.ne(self.src_pad)
        token = torch.full((src.size(0),), bos_id, device=src.device, dtype=torch.long)
        done = torch.zeros_like(token, dtype=torch.bool)
        generated = []
        for _ in range(max_new_tokens):
            scores, state = self.decoder.step(token, state, memory, mask)
            scores[:, self.tgt_pad] = -float("inf")
            scores[:, bos_id] = -float("inf")
            token = scores.argmax(dim=-1)
            token = torch.where(done, eos_id, token)
            generated.append(token)
            done |= token.eq(eos_id)
            if bool(done.all()):
                break
        return torch.stack(generated, dim=1)


def build_model(cfg: dict, src_size: int, tgt_size: int) -> Seq2Seq:
    return Seq2Seq(src_size, tgt_size, cfg["model"])
