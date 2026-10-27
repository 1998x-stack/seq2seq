# seq2seq-from-scratch

A **reproducible, readable, PyTorch-first Sequence-to-Sequence research baseline**.
Switch between vanilla RNN and LSTM through YAML; optionally enable Bahdanau-style additive attention and the Sutskever source-reversal experiment.

### Capabilities

- Independent Encoder / Decoder, with RNN or LSTM chosen by config.
- Variable-length sequences with packing and masking.
- Optional additive cross-attention; optional reversed source tokens.
- Teacher forcing, masked token cross-entropy, label smoothing, gradient clipping, AdamW.
- Hugging Face OPUS Books English-to-French data; separate source/target vocabularies.
- Deterministic random train/validation/test split, duplicate-pair removal, data fingerprint.
- Token accuracy, token NLL, perplexity, corpus BLEU, exact-match.
- Trusted local checkpoints (model + optimizer + vocabulary + configuration).
- Offline toy copy task, pytest, Ruff and GitHub Actions.

## Quick start

Python 3.11+ recommended. Install PyTorch for your device as directed by pytorch.org and then install this package.

    git clone https://github.com/1998x-stack/seq2seq.git
    cd seq2seq
    python -m venv .venv
    source .venv/bin/activate
    pip install -e ".[dev]"

Run an **offline sanity check** with synthetic copy data:

    seq2seq train -c configs/smoke.yaml --output runs/smoke
    seq2seq evaluate --checkpoint runs/smoke/best.pt --max-examples 20
    seq2seq translate --checkpoint runs/smoke/best.pt --text "red green circle"

A one-epoch smoke run checks execution, NOT translation quality.

Run **real English-to-French LSTM + attention**:

    seq2seq inspect-data -c configs/opus_books_lstm.yaml
    seq2seq train -c configs/opus_books_lstm.yaml --output runs/lstm-en-fr
    seq2seq evaluate --checkpoint runs/lstm-en-fr/best.pt --split test
    seq2seq translate --checkpoint runs/lstm-en-fr/best.pt --text "I love machine learning."

Run the **RNN fixed-state, reversed-source baseline**:

    seq2seq train -c configs/opus_books_rnn.yaml --output runs/rnn-en-fr

Resume a paused run ONLY with the same configuration and matching data:

    seq2seq train -c configs/opus_books_lstm.yaml --output runs/lstm-en-fr --resume runs/lstm-en-fr/last.pt

To switch models, edit model.cell to "rnn" or "lstm"; keep all other hyperparameters identical for a controlled comparison. model.attention and model.reverse_source can be changed independently.

## Input and output contract

Text pair: translation.en and translation.fr from Helsinki-NLP/opus_books, subset en-fr.

Encoder input: [word ids..., EOS], shape [batch, src_time].
Decoder input: [BOS, target word ids...], shape [batch, tgt_time].
Decoder labels: [target word ids..., EOS], shape [batch, tgt_time].
Pad id=0, UNK=1, BOS=2, EOS=3; mask padding in loss and attention.
Model forward output: raw logits [batch, tgt_time, target_vocab].
Inference output: generated token ids ending at EOS or max_new_tokens.
For reproducibility, tokenizer and vocabularies are saved inside the checkpoint.

## Data and licensing

Default dataset: [Helsinki-NLP/opus_books](https://huggingface.co/datasets/Helsinki-NLP/opus_books), en-fr (127,085 raw pairs, train split). The dataset's card describes educational/research use and restrictions on commercial resale / redistribution. **The MIT license here applies only to this repository's code, not the corpus.** Do not redistribute the dataset as though it were MIT-licensed.

We randomly split *pairs*, not books. Thus this baseline is suitable for engineering validation and initial ablations, but random splits may share book/style context. Do not claim out-of-domain generalization from these metrics. For research comparisons use a separately curated held-out corpus and pin the Hugging Face dataset revision.

## Experiments

Change one factor at a time; keep seed, split, tokenization, data budget, model width, optimization budget, and evaluation procedure consistent. Baseline matrix:

| Run | cell | attention | reverse_source |
| --- | --- | --- | --- |
| A | rnn | false | false |
| B | rnn | false | true |
| C | lstm | false | false |
| D | lstm | false | true |
| E | lstm | true | false |
| F | lstm | true | true |

Report train/validation token NLL and test BLEU (including decoding and tokenizer settings). Do NOT report scores from different dataset budgets as directly comparable.

## Layout

    configs/             HF translation and offline smoke experiments
    src/seq2seq/config.py    YAML and validation
    src/seq2seq/text.py      Tokenizer and vocabulary
    src/seq2seq/data.py      HF loading, splits, collator
    src/seq2seq/model.py     Encoder, Decoder, Attention
    src/seq2seq/losses.py    Masked CE and perplexity
    src/seq2seq/engine.py    Training, evaluation, checkpointing
    src/seq2seq/cli.py       Command-line entry point
    tests/               Unit and offline integration tests
    docs/                Architecture, evaluation and research roadmap

## Development

    ruff check src tests
    pytest -q

A best checkpoint minimizes validation token NLL. BLEU/Exact Match require greedy generation and are evaluated on a configurable subset for cost control. Generated translations from a fresh untrained model are not expected to be useful.

**Security note:** PyTorch checkpoint files are pickle-based. Load only artifacts you trust. No untrusted checkpoints should be supplied to the CLI.

## Sources

- Sutskever, Vinyals & Le, 2014: [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215).
- Cho et al., 2014: [Learning Phrase Representations using RNN Encoder–Decoder](https://arxiv.org/abs/1406.1078).
- Bahdanau, Cho & Bengio, 2014/2015: [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473).
- Tiedemann, 2012: [Parallel Data, Tools and Interfaces in OPUS](https://aclanthology.org/L12-1246/).

See docs/ROADMAP.md for milestones and explicit non-goals.
