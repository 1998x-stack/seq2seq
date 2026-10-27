# Architecture and tensor contracts

## Scope

Independent, one-directional Encoder and Decoder with torch.nn.RNN or torch.nn.LSTM. Both use the same cell type, number of layers and hidden size; attention is optional. There are no pretrained Hugging Face model weights: only the *dataset* is taken from HF.

## Flow

1. A source sentence is tokenized, optionally reversed, encoded with source vocab, and terminated by EOS.
2. Encoder embeds the sequence, packs it using true lengths, and passes it through the configured recurrent layer.
3. Final encoder hidden state initializes the decoder; for LSTM both h and c states transfer.
4. Decoder consumes BOS and predicts the first target token. Training uses shifted target inputs, optionally mixed with its predictions (teacher_forcing probability).
5. With attention, Decoder computes additive alignment against all non-padding Encoder outputs, concatenates the context to its embedding, and uses its outputs and context for classification.
6. The classifier produces target-vocabulary logits. During generation, greedy token selection continues until EOS or the generation cap.

## Shapes

Notation: B=batch, S=source length, T=target length, E=embedding, H=hidden, L=layers, Vs/Vt=vocab sizes.

| Tensor | Shape | Type |
| --- | --- | --- |
| source ids | [B,S] | int64 |
| source lengths | [B] | int64 |
| source embeddings | [B,S,E] | float |
| encoder memory | [B,S,H] | float |
| encoder hidden | [L,B,H] | float |
| LSTM cell | [L,B,H] | float |
| source mask | [B,S] | bool |
| decoder input ids | [B,T] | int64 |
| target labels | [B,T] | int64 |
| attention weights | [B,S] | float |
| logits | [B,T,Vt] | float |

Source padding is ignored by pack_padded_sequence. Attention masks source padding positions. Loss ignores target PAD tokens. LSTM cell state and hidden state have the same layout. The model is unidirectional, so no bidirectional state bridge is required in v0.1.

## Loss

The training objective is negative conditional log likelihood with teacher forcing:

  L = -sum_{b,t:y[b,t]!=PAD} log p(y[b,t] | X[b], y[b,<t]) / N_valid_tokens

label_smoothing in training uses PyTorch cross_entropy; token NLL reported for train and eval is UNSMOOTHED for comparable perplexity exp(NLL). The best model is chosen by validation NLL, not by test performance.

Teacher forcing < 1 substitutes model predictions for some previous target inputs and changes the conditioning distribution; use 1.0 for the clean maximum-likelihood baseline. Gradient norm clipping is applied before AdamW optimizer step.

## Non-goals for v0.1

- Reproducing the exact 2014 380M-parameter, four-layer, eight-GPU setup.
- BPE/subword tokenizers, beam search, mixed precision, distributed training.
- General-purpose streaming datasets or large-scale BLEU benchmark claims.
- Exact bitwise reproducibility of arbitrary accelerator kernels.

All are future work tracked in ROADMAP.md.
