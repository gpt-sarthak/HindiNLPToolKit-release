"""
train_lstm.py — train the LSTM language model behind ``adaptive_lstm_wiki50m``.

    python training/lstm/train_lstm.py              # GPU recommended
    python training/lstm/train_lstm.py --max-sentences 2000 --minutes 1 \
        --out training/work/lstm_smoke.pt           # quick plumbing test

Input  : training/work/wiki_sentences.txt   (training/wiki/sentences.py)
         scoring/models/vocab.pkl           (training/wiki/build_vocab.py)
Output : scoring/models/base_model_wiki50m.pt
         scoring/models/base_model_wiki50m.report.json  (losses, perplexity, settings)

What it does
------------
Every sentence enters one long token stream as <SOS> tokens <EOS>; the last 1%
of the stream is held out for validation. The stream is cut into
non-overlapping 50-token windows and trained with Adam (lr 1e-3, batch 128,
gradient clipping 1.0, seed 42). After each epoch the validation loss is
checked; the best checkpoint is kept, training stops after 2 epochs without
improvement or when the wall-clock budget (--minutes) runs out.

Architecture (training/lstm/lm.py, must match the scorer): 256-dim embeddings,
2-layer LSTM with 256 hidden units and dropout 0.3, untied output layer, 30k
vocabulary.

The shipped checkpoint came from exactly these defaults on one NVIDIA Tesla T4:
2,711,172 sentences / ~50.8 M tokens, stopped by the 25-minute budget after
2 epochs (still improving), best validation perplexity 79.65, 1,504 s total.
Because the budget is wall-clock, a faster or slower GPU sees a different
number of steps; pass --minutes to trade time for quality, or --epochs with a
large --minutes to train to convergence.
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
from lm import EOS, SOS, UNK, LSTMLanguageModel  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def build_stream(corpus: Path, w2i, max_sentences: int = 0):
    """Flatten the corpus into one id stream: <SOS> tokens <EOS> per sentence."""
    unk, sos, eos = w2i.get(UNK, 1), w2i[SOS], w2i[EOS]
    stream = []
    kept = 0
    with open(corpus, "r", encoding="utf-8") as fh:
        for line in fh:
            toks = line.split()
            if len(toks) < 2:
                continue
            stream.append(sos)
            stream.extend(w2i.get(t, unk) for t in toks)
            stream.append(eos)
            kept += 1
            if max_sentences and kept >= max_sentences:
                break
    return torch.tensor(stream, dtype=torch.long), kept


def batches(stream, seq_len, batch_size, device):
    """Contiguous non-overlapping windows, reshaped into batches.

    Non-overlapping windows cover the corpus exactly once per epoch.
    """
    n = (len(stream) - 1) // seq_len
    if n == 0:
        return
    x = stream[:n * seq_len].view(n, seq_len)
    y = stream[1:n * seq_len + 1].view(n, seq_len)
    perm = torch.randperm(n)
    # Step to n, not n-batch_size+1: a trailing partial batch is still real
    # training data, and on a small corpus the strict form yields nothing at all.
    for i in range(0, n, batch_size):
        idx = perm[i:i + batch_size]
        yield x[idx].to(device), y[idx].to(device)


@torch.no_grad()
def evaluate(model, stream, seq_len, batch_size, device, loss_fn):
    model.eval()
    n = (len(stream) - 1) // seq_len
    if n == 0:
        # Guarded rather than silently returning 0.0: a validation split smaller
        # than one window would otherwise report perplexity 1.0 and make early
        # stopping fire on a number that means nothing.
        raise SystemExit(
            f"validation split holds {len(stream)} tokens, fewer than one "
            f"{seq_len}-token window - raise --val-fraction or the corpus size"
        )
    x = stream[:n * seq_len].view(n, seq_len)
    y = stream[1:n * seq_len + 1].view(n, seq_len)
    total, weight = 0.0, 0
    for i in range(0, n, batch_size):
        xb, yb = x[i:i + batch_size].to(device), y[i:i + batch_size].to(device)
        logits, _ = model(xb)
        # Weight by token count so a trailing partial batch is not over-counted.
        ntok = yb.numel()
        total += loss_fn(logits.view(-1, logits.size(-1)), yb.reshape(-1)).item() * ntok
        weight += ntok
    return total / weight


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path,
                    default=ROOT / "training" / "work" / "wiki_sentences.txt")
    ap.add_argument("--vocab", type=Path,
                    default=ROOT / "scoring" / "models" / "vocab.pkl")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "scoring" / "models" / "base_model_wiki50m.pt")
    ap.add_argument("--report", type=Path, default=None)
    ap.add_argument("--seq-len", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--val-fraction", type=float, default=0.01)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--minutes", type=float, default=25.0,
                    help="wall-clock budget; training stops when exceeded")
    ap.add_argument("--max-sentences", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(args.vocab, "rb") as fh:
        raw = pickle.load(fh)
    w2i = raw["word2idx"] if "word2idx" in raw else raw

    t_load = time.time()
    stream, kept = build_stream(args.corpus, w2i, args.max_sentences)
    cut = int(len(stream) * (1 - args.val_fraction))
    train_s, val_s = stream[:cut], stream[cut:]
    print(f"corpus: {kept:,} sentences, {len(stream):,} tokens "
          f"({len(train_s):,} train / {len(val_s):,} val) in {time.time()-t_load:.0f}s",
          flush=True)

    model = LSTMLanguageModel(len(w2i)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + args.minutes * 60
    best = float("inf")
    bad = 0
    history = []
    t0 = time.time()
    stopped = "epochs"

    for epoch in range(1, args.epochs + 1):
        model.train()
        run, steps = 0.0, 0
        for xb, yb in batches(train_s, args.seq_len, args.batch_size, device):
            opt.zero_grad()
            logits, _ = model(xb)
            loss = loss_fn(logits.view(-1, logits.size(-1)), yb.reshape(-1))
            loss.backward()
            if args.clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            run += loss.item()
            steps += 1
            if time.time() > deadline:
                stopped = "time budget"
                break
        val = evaluate(model, val_s, args.seq_len, args.batch_size, device, loss_fn)
        rec = {"epoch": epoch, "train_loss": round(run / max(steps, 1), 4),
               "val_loss": round(val, 4), "val_ppl": round(math.exp(val), 2),
               "steps": steps, "elapsed_s": round(time.time() - t0, 1)}
        history.append(rec)
        print(f"  epoch {epoch}: train {rec['train_loss']} "
              f"val {rec['val_loss']} ppl {rec['val_ppl']} "
              f"({rec['elapsed_s']}s)", flush=True)

        if val < best - 1e-4:
            best, bad = val, 0
            torch.save(model.state_dict(), args.out)
        else:
            bad += 1
            if bad >= args.patience:
                stopped = "early stopping"
                break
        if time.time() > deadline:
            stopped = "time budget"
            break

    if not args.out.exists():        # never improved; still emit something
        torch.save(model.state_dict(), args.out)

    report = {
        "checkpoint": str(args.out),
        "stopped_because": stopped,
        "best_val_loss": round(best, 4),
        "best_val_perplexity": round(math.exp(best), 2) if best < 50 else None,
        "sentences": kept, "tokens": int(len(stream)),
        "vocab_size": len(w2i),
        "seq_len": args.seq_len, "batch_size": args.batch_size,
        "lr": args.lr, "clip": args.clip, "seed": args.seed,
        "device": str(device),
        "total_seconds": round(time.time() - t0, 1),
        "history": history,
        "architecture": "Embedding(V,256) -> LSTM(256,256,2,dropout=0.3) -> Linear(256,V), untied",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    dest = args.report or args.out.with_suffix(".report.json")
    Path(dest).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"saved {args.out}  ({stopped}; best val ppl "
          f"{report['best_val_perplexity']})\nreport -> {dest}")


if __name__ == "__main__":
    main()
