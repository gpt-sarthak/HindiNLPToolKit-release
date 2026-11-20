"""
build_vocab.py — the 30,000-word vocabulary of the LSTM.

    python training/wiki/build_vocab.py

Input  : training/work/wiki_sentences.txt   (from sentences.py)
Output : scoring/models/vocab.pkl
         {"word2idx": {word: id}, "idx2word": {id: word}}

The four special tokens take ids 0-3: <PAD>, <UNK>, <SOS>, <EOS>. The remaining
29,996 slots are the most frequent words of the corpus (``Counter.most_common``,
so ties keep first-seen order and the result is deterministic for a given
input file). Words outside it become <UNK> at scoring time.

The shipped vocab.pkl has MD5 795fbd374f67e0363b10158bce2d19e0. Runtime ~2-5 min.
(The trigram model builds its own vocabulary — see training/trigram.)
"""

from __future__ import annotations

import argparse
import pickle
from collections import Counter
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "training" / "work"

VOCAB_SIZE = 30000
SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<SOS>", "<EOS>"]


def build_vocab(src: Path, dst: Path) -> int:
    counter = Counter()
    with open(src, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Counting tokens"):
            counter.update(line.strip().split())

    most_common = counter.most_common(VOCAB_SIZE - len(SPECIAL_TOKENS))
    vocab_words = SPECIAL_TOKENS + [word for word, _ in most_common]
    word2idx = {word: idx for idx, word in enumerate(vocab_words)}
    idx2word = {idx: word for word, idx in word2idx.items()}

    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "wb") as f:
        pickle.dump({"word2idx": word2idx, "idx2word": idx2word}, f)
    return len(word2idx)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sentences", type=Path, default=WORK / "wiki_sentences.txt")
    ap.add_argument("--out", type=Path, default=ROOT / "scoring" / "models" / "vocab.pkl")
    args = ap.parse_args()
    n = build_vocab(args.sentences, args.out)
    print(f"vocabulary size {n:,} -> {args.out}")


if __name__ == "__main__":
    main()
