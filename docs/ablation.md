# Ablation: structured logs and offline visual analytics

This branch adds reproducible tracking and visual investigation without pretending toy-data results represent translation quality.

## Run the offline end-to-end smoke experiment

    git switch ablation
    pip install -e ".[dev]"
    pytest -q
    seq2seq experiments run -c configs/smoke.yaml --output runs/ablation-smoke --seeds 42 --split-seed 2026 --limit 1 --max-examples 4 --dashboard
    seq2seq visualize --root runs/ablation-smoke --output runs/ablation-smoke/report.html

The last command builds a portable HTML file containing plotly.js inline. Open the file in a browser offline. The smoke task uses synthetic copy pairs; its scores must never be interpreted as OPUS translation BLEU. Partial plans are marked INCOMPLETE.

## Run real translation ablations

    seq2seq experiments plan -c configs/opus_books_lstm.yaml --seeds 42 43 44 --split-seed 2026
    seq2seq experiments run -c configs/opus_books_lstm.yaml --output runs/ablation-opus --seeds 42 43 44 --split-seed 2026 --max-examples -1 --dashboard
    seq2seq experiments summarize --output runs/ablation-opus

Matrix: RNN/LSTM x reversed/not reversed, plus two LSTM attention variants, for six model conditions. With three seeds, the fixed plan has 18 runs.

## File layout

Each run folder:
- config.yaml: effective model, dataset and logging configuration.
- metadata.json: data fingerprint, hyperparameters, hardware, model parameters.
- metrics.jsonl: one row per epoch with train/validation token metrics.
- metrics.csv: flattened token NLL, accuracy, perplexity, throughput and elapsed time.
- events.jsonl: UTC run_start/train_step/epoch_completed/training_completed or failure events.
- best.pt and last.pt: model checkpoint, using minimum validation NLL for best.
- tensorboard/: optional local TensorBoard event files.
- predictions.jsonl and result.json: generated predictions and test metrics.

At matrix root:
- plan.json: frozen six-condition experiment plan.
- events.jsonl: run_started, run_completed, run_failed, run_reused.
- dashboard.html: generated interactive report when requested.

## Logging configuration

    logging:
      enabled: true
      log_every_steps: 20
      tensorboard: false

log_every_steps=0 disables batch-level events while retaining epoch-level events.
To enable TensorBoard, install optional dependencies and set tensorboard=true:

    pip install -e ".[viz]"
    tensorboard --logdir runs/ablation-opus --port 6006

The local logger sends no records to external tracking services. Do not upload dataset text or checkpoints to GitHub.

## Dashboard evidence contracts

The dashboard reads existing measured artifacts; it NEVER starts training or creates fake values. It contains:
1. Train vs validation token NLL and accuracy curves, grouped by architecture and random seed.
2. BLEU distributions using actual per-seed measurements (one seed means one point, not strong uncertainty evidence).
3. Length-bucket corpus BLEU, including sample counts.
4. An exact-value run table, experiment completion status and dataset fingerprint.

The comparison layer rejects mismatched data fingerprints, BLEU signatures or generation example counts. A fixed split fingerprint is not evidence of document-disjoint or book-disjoint evaluation. Word tokenization introduces unknown tokens; test BLEU scores preserve original normalized target text.

Note that the architecture variants have different parameter counts/FLOPs. These are controlled hyperparameter experiments, not yet equal-compute comparisons. For scientifically defensible outcomes, finish the actual 18 OPUS runs and add external document-disjoint evaluation, multiple-seed intervals, and compute-matched baselines.

## Checks

    ruff check src tests
    pytest -q
    seq2seq experiments run -c configs/smoke.yaml --output runs/local-ablation --limit 1 --seeds 42 --split-seed 2026 --max-examples 4 --dashboard
    seq2seq export-csv --run runs/local-ablation/rnn-attention-0-reverse-0/seed-42

GitHub Actions stores a seven-day toy dashboard HTML artifact as a visualization smoke test.
