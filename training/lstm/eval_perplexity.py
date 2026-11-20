"""
eval_perplexity.py — held-out perplexity of an LSTM checkpoint, on the same
1% validation split train_lstm.py holds out (same corpus + vocabulary ->
identical split, so checkpoints are directly comparable).

    python training/lstm/eval_perplexity.py \
        --checkpoint scoring/models/base_model_wiki50m.pt

The shipped checkpoint scores 79.65.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import time
from pathlib import Path

import torch
import torch.nn as nn

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lm import LSTMLanguageModel                   # noqa: E402
from train_lstm import ROOT, build_stream, evaluate  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path,
                    default=ROOT / "training" / "work" / "wiki_sentences.txt")
    ap.add_argument("--vocab", type=Path,
                    default=ROOT / "scoring" / "models" / "vocab.pkl")
    ap.add_argument("--checkpoint", type=Path, action="append", required=True,
                    help="repeatable; every checkpoint is scored on the same split")
    ap.add_argument("--seq-len", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--val-fraction", type=float, default=0.01)
    ap.add_argument("--max-sentences", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(args.vocab, "rb") as fh:
        raw = pickle.load(fh)
    w2i = raw["word2idx"] if "word2idx" in raw else raw

    t0 = time.time()
    stream, kept = build_stream(args.corpus, w2i, args.max_sentences)
    cut = int(len(stream) * (1 - args.val_fraction))
    val_s = stream[cut:]
    print(f"corpus: {kept:,} sentences, {len(stream):,} tokens "
          f"-> held-out {len(val_s):,} tokens  ({time.time()-t0:.0f}s)")
    print(f"device: {device}\n")

    loss_fn = nn.CrossEntropyLoss()
    results = {}
    for ckpt in args.checkpoint:
        model = LSTMLanguageModel(len(w2i))
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.to(device)
        t1 = time.time()
        loss = evaluate(model, val_s, args.seq_len, args.batch_size, device, loss_fn)
        ppl = math.exp(loss)
        results[str(ckpt)] = {"val_loss": round(loss, 4),
                              "perplexity": round(ppl, 2),
                              "seconds": round(time.time() - t1, 1)}
        print(f"  {ckpt.name:28s} val_loss {loss:7.4f}   perplexity {ppl:8.2f}"
              f"   ({time.time()-t1:.0f}s)")
        del model

    payload = {
        "corpus": str(args.corpus),
        "vocab": str(args.vocab),
        "held_out_tokens": int(len(val_s)),
        "val_fraction": args.val_fraction,
        "seq_len": args.seq_len,
        "device": str(device),
        "checkpoints": results,
    }
    dest = args.out or (ROOT / "training" / "work" / "perplexity.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
