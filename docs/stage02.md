# Stage 02 — Controlled and reproducible Seq2Seq experiments

## Research questions

1. With parallel corpus, vocabulary, optimizer and evaluation data fixed, how does vanilla RNN compare with LSTM?
2. Does reversed encoder source order reduce long-distance dependency costs for a fixed-state encoder-decoder?
3. Does additive cross-attention improve long-input translation?
4. How much do outcomes vary across independent training seeds, and which errors co-occur with long inputs or unseen source words?

These are hypotheses, not confirmed discoveries. An offline toy-copy smoke test is not evidence for real-world translation.

## Data provenance

Hugging Face dataset: Helsinki-NLP/opus_books, en-fr configuration, upstream revision 1f9f619 (abbreviated commit identifier from March 29, 2024). The dataset card lists 127,085 raw training pairs. Filtered pair counts, training vocabulary sizes and data fingerprint are saved with each run.

dataset.split_seed is independent from top-level seed. Across ablations, keep the same split_seed and compare data_fingerprint strictly. Training seeds control model initialization, dropout and data-loader order.

WARNING: OPUS Books includes per-row id but no verified per-book group id. Calling random-pair splits book-disjoint is incorrect. To test on unseen books, curate JSONL with verified book_id and set backend: jsonl, group_field: book_id, split_strategy: group. The group splitter holds entire group IDs apart; sample-count ratios may differ from requested group-count ratios.

## Six architecture variants

| Condition | Cell | Attention | Reverse source |
| --- | --- | --- | --- |
| A | RNN | No | No |
| B | RNN | No | Yes |
| C | LSTM | No | No |
| D | LSTM | No | Yes |
| E | LSTM | Yes | No |
| F | LSTM | Yes | Yes |

With seeds 42,43,44, the full experiment consists of 18 independent runs. Parameter count and compute cost differ between RNN and LSTM and between attention variants, so it is NOT a compute-matched comparison.

## Commands

    seq2seq experiments plan -c configs/opus_books_lstm.yaml --seeds 42 43 44 --split-seed 2026
    seq2seq experiments run -c configs/opus_books_lstm.yaml --output runs/stage02 --seeds 42 43 44 --split-seed 2026 --max-examples -1
    seq2seq experiments summarize --output runs/stage02

--limit 1 runs just the first planned variant for an integration test. An already completed variant is reused only when the complete experiment config matches. Incomplete runs are NOT silently overwritten.

For a single checkpoint:

    seq2seq evaluate --checkpoint runs/lstm-en-fr/best.pt --split test --max-examples -1 --predictions runs/lstm-en-fr/predictions.jsonl

--max-examples -1 means all examples, 0 skips free-running generation, positive N means the first N split examples.

## Recorded artifacts

Per run: config.yaml, metadata.json, metrics.jsonl, best.pt, last.pt.
Experiment runner also writes experiment.yaml, predictions.jsonl, result.json, and root plan.json.

metadata.json records Python, PyTorch, datasets and SacreBLEU versions, operating system, device/GPU, effective dataset configuration, sample counts, parameter count and split fingerprint.

The free-running generation metrics are corpus BLEU (with SacreBLEU signature) and exact-string match. Gold references come from original reference sentences after shared deterministic tokenization/normalization, not from vocabulary-decoded training labels. OOV diagnostics separately measure source and target unknown word rates. Source-length buckets are 1–10, 11–20, 21–40, and 41+ words.

Token accuracy, negative log-likelihood and perplexity are teacher-forced, non-PAD metrics and should not be confused with free-running output quality. BLEU in small buckets can be unstable; always disclose the sample count.

Summary computes mean and sample standard deviation for completed seeds per condition; does not imply a confidence interval. Preserve the fixed research protocol before inspecting test outcomes.

## Scientific limitations

- Exact parallel sentence deduplication does not eliminate near duplicates or document-level leakage.
- Frequency-based word tokenization produces target vocabulary OOVs. Scoring original text (not gold UNKs) prevents silent metric inflation.
- Default random-pair test split reflects interpolation within the same corpus, not cross-book generalization.
- A fixed number of epochs does not equal matched compute, parameter count or throughput.
- Accelerator nondeterminism and system package differences remain possible; full dependency lock is future work.
- No real OPUS Books metrics are precomputed by this repository, and the six-condition matrix requires training resources.
- The current checkpoint uses Python-serialized state: only load checkpoints from trusted sources.
