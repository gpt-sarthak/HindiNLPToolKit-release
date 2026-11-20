# training/lstm — LSTM language model (Wikipedia, 50.8 M tokens)

Builds `scoring/models/base_model_wiki50m.pt`, the model behind the
`adaptive_lstm_wiki50m` scorer. Needs PyTorch (installed with the toolkit);
an NVIDIA GPU is strongly recommended.

```bash
python training/lstm/train_lstm.py                       # defaults = how the shipped model was trained
python training/lstm/train_lstm.py --minutes 60          # longer budget
python training/lstm/train_lstm.py --max-sentences 2000 --minutes 1 \
       --out training/work/lstm_smoke.pt                 # quick plumbing test
python training/lstm/eval_perplexity.py --checkpoint scoring/models/base_model_wiki50m.pt
```

Inputs: `training/work/wiki_sentences.txt` and `scoring/models/vocab.pkl`
(both from `training/wiki/`).

## What it does

- Every sentence enters one token stream as `<SOS> w1 … wn <EOS>` (words outside
  the 30k vocabulary → `<UNK>`); the last 1% is the validation split.
- Non-overlapping 50-token windows, shuffled, batch 128.
- Adam, learning rate 1e-3, gradient clipping 1.0, seed 42.
- After each epoch: validation loss; the best checkpoint is saved. Stops after
  2 epochs without improvement, after 20 epochs, or when `--minutes` (default
  **25**) runs out.
- Architecture (`lm.py`, must match `scoring/_lstm_common.py`): 256-dim
  embeddings → 2-layer LSTM, 256 hidden, dropout 0.3 → linear output over the
  vocabulary (not tied).

Also written: `base_model_wiki50m.report.json` (loss per epoch, perplexity,
settings).

## The shipped checkpoint

Trained with the defaults above on one NVIDIA Tesla T4: 2,711,172 sentences,
~50.8 M tokens, stopped by the 25-minute budget after 2 epochs (still
improving), best validation perplexity **79.65**, 1,504 s.

The budget is wall-clock, so a different GPU completes a different number of
steps, and GPU arithmetic is not bit-reproducible — a retrain gives a model of
the same design and similar quality, not identical weights.

## How the scorer uses it

`adaptive_lstm_wiki50m` cleans each sentence like the training text, adds
`<SOS>`/`<EOS>`, and sums the per-word surprisal (nats). Before scoring a
reference and its variants it takes one SGD step (lr 0.01, dropout off) on the
preceding sentence of the document, starting from the pristine weights each
time.
