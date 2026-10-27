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

## Stage 02 — scientific evaluation (next)

- [ ] Pin corpus revision; archive dependency locks and runtime metadata.
- [ ] Run six controlled ablations on identical en-fr split and fixed training budget.
- [ ] Compare at least three random seeds and report mean and dispersion.
- [ ] Length-bucket analysis, OOV breakdown, representative errors.
- [ ] Document- or book-disjoint test set and an external translation test set.
- [ ] Add BLEU signature and hypothesis/reference predictions as research artifacts.
- [ ] Resume tests (multi-epoch exact continuation where supported).

## Stage 03 — advanced decoding and training

- [ ] Beam Search with length normalization and early stopping.
- [ ] Scheduled Sampling and exposure-bias diagnostics.
- [ ] Beam/greedy oracle examples and optimality counterexamples.
- [ ] TensorBoard / CSV charts, throughput/token timing, GPU profiling.
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
