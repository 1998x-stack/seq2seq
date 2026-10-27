import copy

import pytest
import torch

from seq2seq.config import load_config, validate_config
from seq2seq.data import collate_pairs, make_loader, prepare_data
from seq2seq.losses import masked_cross_entropy
from seq2seq.model import Seq2Seq
from seq2seq.text import Tokenizer, Vocabulary


def cfg():
    return load_config("configs/smoke.yaml")


def test_vocab_and_unicode_tokenizer():
    tok = Tokenizer()
    tokens = tok.tokenize("Hello, École!")
    assert tokens == ["hello", ",", "école", "!"]
    vocab = Vocabulary.build([tokens], max_size=20)
    assert vocab.encode(["missing"]) == [vocab.unk_id]
    assert vocab.decode(vocab.encode(tokens) + [vocab.eos_id, 0]) == tokens


def test_masked_loss_ignores_padding_and_backward():
    logits = torch.zeros(2, 3, 7, requires_grad=True)
    target = torch.tensor([[2, 3, 0], [4, 0, 0]])
    loss, count = masked_cross_entropy(logits, target, 0)
    assert count == 3
    assert loss.item() == pytest.approx(torch.log(torch.tensor(7.0)).item())
    loss.backward()
    assert logits.grad is not None
    assert logits.grad[0, 2].abs().sum() == 0


def test_invalid_config_rejected():
    config = cfg()
    config["model"]["cell"] = "gru"
    with pytest.raises(ValueError, match="model.cell"):
        validate_config(config)


def test_collator_shift_and_shapes():
    items = [([4, 5, 3], [2, 6, 7, 3]), ([7, 3], [2, 8, 3])]
    batch = collate_pairs(items, 0, 0)
    assert batch["src"].shape == (2, 3)
    assert batch["src_lengths"].tolist() == [3, 2]
    assert batch["tgt_in"].tolist() == [[2, 6, 7], [2, 8, 0]]
    assert batch["tgt_out"].tolist() == [[6, 7, 3], [8, 3, 0]]


@pytest.mark.parametrize("cell", ["rnn", "lstm"])
@pytest.mark.parametrize("attention", [False, True])
def test_model_forward_backward_generate(cell, attention):
    model_cfg = copy.deepcopy(cfg()["model"])
    model_cfg["cell"] = cell
    model_cfg["attention"] = attention
    model = Seq2Seq(20, 21, model_cfg)
    batch = collate_pairs(
        [([4, 5, 3], [2, 7, 8, 3]), ([9, 3], [2, 5, 3])], 0, 0
    )
    logits = model(batch["src"], batch["src_lengths"], batch["tgt_in"])
    assert logits.shape == (2, 3, 21)
    loss, _ = masked_cross_entropy(logits, batch["tgt_out"])
    loss.backward()
    assert model.encoder.embedding.weight.grad is not None
    with torch.no_grad():
        generated = model.generate(batch["src"], batch["src_lengths"], max_new_tokens=5)
    assert generated.shape[0] == 2
    assert 1 <= generated.shape[1] <= 5
    assert not (generated == 0).any()


def test_dataset_reproducibility_and_train_only_vocab():
    config = cfg()
    a = prepare_data(config)
    b = prepare_data(config)
    assert a.fingerprint == b.fingerprint
    assert a.train == b.train
    assert len(a.train) + len(a.test) + len(a.validation) == 64
    assert not (set(a.train) & set(a.validation))
    assert not (set(a.train) & set(a.test))
    assert a.src_vocab.itos == b.src_vocab.itos
    loader = make_loader(a, "train", config)
    batch = next(iter(loader))
    assert batch["src"].shape[0] <= config["training"]["batch_size"]


def test_rnn_config_integration():
    c = cfg()
    c["model"]["cell"] = "rnn"
    c["model"]["attention"] = False
    c["model"]["reverse_source"] = True
    bundle = prepare_data(c)
    batch = next(iter(make_loader(bundle, "train", c)))
    model = Seq2Seq(len(bundle.src_vocab), len(bundle.tgt_vocab), c["model"])
    output = model(batch["src"], batch["src_lengths"], batch["tgt_in"], teacher_forcing=0.5)
    assert output.shape[:2] == batch["tgt_out"].shape
