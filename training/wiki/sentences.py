"""
sentences.py — split Wikipedia text into clean, length-filtered sentences.

    python training/wiki/sentences.py

Input  : training/work/wiki_plain.txt       (from extract.py)
Output : training/work/wiki_sentences.txt   (~627 MB, ~2.7 M sentences, ~50.8 M tokens)

Per line:
    1. split on Hindi sentence boundaries  । ! ?  (the delimiter is dropped)
    2. replace every character outside Devanagari U+0900-U+097F (and space,
       comma, hyphen) with a space — Latin, digits, brackets, URLs all go
    3. collapse whitespace
    4. keep sentences of 5-50 tokens

This one file trains BOTH language models (training/trigram, training/lstm)
and is the input to build_vocab.py. The same cleaning is applied when the
scorers read a sentence (scoring/_katz_trigram.py and scoring/_lstm_wiki50m.py
``normalize``), so training and scoring agree. With the 2026-02-19 dump the
output had MD5 615419998ef3a2c3be0e3a85b0bff045.

Runtime ~5-10 min. Next: build_vocab.py.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from tqdm import tqdm

WORK = Path(__file__).resolve().parents[1] / "work"

MIN_TOKENS = 5
MAX_TOKENS = 50
SENTENCE_SPLIT_REGEX = r"[।!?]"
DEVANAGARI_REGEX = re.compile(r"[^ऀ-ॿ\s,\-]")


def clean_line(line: str) -> str:
    """Strip non-Devanagari characters and normalise whitespace."""
    line = line.strip()
    line = DEVANAGARI_REGEX.sub(" ", line)
    line = re.sub(r"\s+", " ", line)
    return line.strip()


def tokenize_and_filter(sentence: str):
    tokens = sentence.split()
    if MIN_TOKENS <= len(tokens) <= MAX_TOKENS:
        return " ".join(tokens)
    return None


def process(src: Path, dst: Path) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(src, "r", encoding="utf-8") as infile, \
         open(dst, "w", encoding="utf-8", newline="\n") as outfile:  # same bytes on every OS
        for line in tqdm(infile, desc="Tokenising"):
            # split BEFORE cleaning, while । ! ? are still there as delimiters
            for sentence in re.split(SENTENCE_SPLIT_REGEX, line):
                sentence = clean_line(sentence)
                if not sentence:
                    continue
                processed = tokenize_and_filter(sentence)
                if processed:
                    outfile.write(processed + "\n")
                    count += 1
    return count


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plain", type=Path, default=WORK / "wiki_plain.txt")
    ap.add_argument("--out", type=Path, default=WORK / "wiki_sentences.txt")
    args = ap.parse_args()
    n = process(args.plain, args.out)
    print(f"sentences written: {n:,} -> {args.out}")


if __name__ == "__main__":
    main()
