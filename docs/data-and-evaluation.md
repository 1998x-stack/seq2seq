# Data, evaluation and experimental validity

## Corpus selection

Hugging Face repo: Helsinki-NLP/opus_books, subset en-fr, single train split (~127,085 original pairs). Each row includes a translation mapping with keys en and fr. Loading uses datasets.load_dataset in data.py. For stable published experiments, set dataset.revision to a full HF commit SHA.

## Preprocessing

- Unicode-preserving regex tokens (word or standalone punctuation), optional lowercase.
- Reject pairs with missing strings or source/target token lengths outside configured bounds.
- Remove exact normalized parallel-pair duplicates.
- Deterministically shuffle with Python random.Random(seed), then optionally cap samples.
- Divide the remaining pairs by configured fractions into test, validation and train.
- Fit two frequency-ranked vocabularies ONLY using train sentences, with lexical tie-breaking and four reserved special tokens.
- Encode source with EOS, target with BOS and EOS. Padded batches supply lengths.

Caveats: exact-pair deduplication does not remove near-duplicates or cross-book overlap. This does not certify independence of train and test. Punctuation/space tokenization is basic and not linguistically optimal, especially for French. Text reconstruction is lossy. For stronger results, evaluate using a well-defined detokenization system and a genuinely held-out corpus.

## Metric definitions

- Token NLL: sum of unsmoothed token negative log probabilities divided by non-PAD target count.
- Perplexity: exp(token NLL); depends on vocabulary/tokenization and is not directly comparable across tokenizer changes.
- Token accuracy: fraction of non-PAD target positions matching argmax logits under teacher forcing.
- Corpus BLEU: sacrebleu.corpus_bleu, generated greedily against a configurable sample of reference strings.
- Exact match: fraction of complete generated strings exactly matching reconstructed references (after shared detokenization).

BLEU and exact match use autoregressive generation, not teacher-forced outputs. Use --max-examples 0 to skip generation metrics. BLEU on a small sample has high variance; the code defaults to 200 examples to limit CPU test duration. For publishable test metrics, use all test examples by setting a cap above their count.

## Baseline experiments

Compare RNN vs LSTM, attention on/off, source reversal on/off, multiple seeds and target-length buckets. Control dataset revision, vocab, maximum sequence length, learning rate, training updates and decode budget. Capture GPU model, PyTorch version and throughput separately when publishing benchmarks.

## Research integrity

Never tune hyperparameters against test BLEU. A single random split does not justify assertions about generalization to other books or domains. Preserve the checkpoint's fingerprint and experiment config with published results.
