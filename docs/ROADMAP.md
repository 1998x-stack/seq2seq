# Research and engineering roadmap

## Stage 00 — reproducible foundation (implemented)

- [x] Python src/ package, pyproject, CLI, YAML configs, pytest, Ruff, GitHub Actions.
- [x] Offline toy-copy task for testing without external downloads.
- [x] RNN / LSTM switch, validation, deterministic seed.
- [x] Vocabulary, padding, teacher-forcing I/O contract.

Acceptance: install package; offline tests and smoke train/evaluate/translate execute successfully.

## Stage 01 — classical Seq2Seq baselines (implemented)

- [x] Packed unidirectional RNN and LSTM encoders.
- [x] Recurrent decoders initialized from encoder state.
- [x] Optional additive cross-attention and source reversal.
- [x] Masked cross entropy, optimizer, clip norm, checkpoint.
- [x] Greedy decode with EOS, token accuracy, NLL, BLEU.

Acceptance: all 4 combinations of RNN/LSTM × attention pass forward/backward/generation tests.

## Stage 02 — scientific evaluation (in progress)

- [x] Pin OPUS Books dataset revision (abbreviated upstream commit) and record runtime metadata; full dependency lockfile pending.
- [x] Implement six-condition matrix runner on a fixed data split; real en-fr training and measured ablations pending.
- [x] Implement seed-controlled repeated runs and mean/sample-standard-deviation reports; full measured runs pending.
- [x] Implement per-example prediction exports, length buckets and OOV analysis; scientific interpretation pending.
- [x] Support genuinely group-disjoint split when JSONL contains book_id; curated external/book-disjoint corpus pending.
- [x] Store SacreBLEU signature and raw-normalized reference predictions in JSONL.
- [ ] Verify exact multi-epoch resume and introduce multi-worker deterministic shuffling safeguards.

Acceptance: offline tests/CI pass, then execute all 18 real corpus runs, compare data fingerprints, and preserve a source-backed evaluation report. Do not call Stage 02 scientifically complete until genuine results exist.

## Stage 03 — advanced decoding and training

- [ ] Beam Search with length normalization and early stopping.
- [ ] Scheduled Sampling and exposure-bias diagnostics.
- [ ] Beam/greedy oracle examples and optimality counterexamples.
- [x] JSONL/CSV logging, optional TensorBoard, offline interactive dashboard, basic token throughput timing (GPU profiling pending).
- [ ] AMP and gradient accumulation for constrained GPUs.
- [ ] Subword tokenizer with tokenizer-version tracking.

## Stage 04 — rigorous architecture comparisons

- [ ] GRU and bidirectional encoders with explicit state bridge.
- [ ] Multiplicative attention vs additive attention.
- [ ] Transformer encoder-decoder baseline under matched parameters/compute.
- [ ] Reproduce canonical source-reversal analysis with sequence-length stratification.
- [ ] Formal benchmark report with limitations and reproducible commands.

## Stage 05 — artifact release

- [ ] Tested dataset pinning and reproducibility manifests.
- [ ] Comprehensive model/dataset cards and benchmark table.
- [ ] Versioned educational walkthroughs, notebooks, release tags.

Do not check off a stage solely because scaffolding exists; record actual experiment results.
