"""
extract.py — strip wiki markup from the Hindi Wikipedia dump.

    python training/wiki/extract.py

Input  : training/work/hiwiki-pages-articles.xml.bz2   (from download.py)
Output : training/work/wiki_plain.txt                  (~850 MB)
         one article per line, wikitext removed

For every page the first revision in the dump (the only one in a
pages-articles dump) is parsed with mwparserfromhell and ``strip_code()`` is
written out. Runtime ~20-40 min on one CPU core. Next: sentences.py.
"""

from __future__ import annotations

import argparse
import bz2
from pathlib import Path

import mwparserfromhell
import mwxml
from tqdm import tqdm

WORK = Path(__file__).resolve().parents[1] / "work"


def clean_text(wikitext: str) -> str:
    """Strip wikitext markup and return plain unicode text."""
    return mwparserfromhell.parse(wikitext).strip_code()


def extract(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with bz2.open(src, mode="rt", encoding="utf-8", errors="ignore") as dump_file:
        dump = mwxml.Dump.from_file(dump_file)
        # newline="\n": identical output (and MD5) on Windows and Linux
        with open(dst, "w", encoding="utf-8", newline="\n") as out:
            for page in tqdm(dump.pages, desc="Extracting pages"):
                for revision in page:
                    if revision.text:
                        clean = clean_text(revision.text)
                        if clean.strip():
                            out.write(clean + "\n")
                    break  # only the first revision per page


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", type=Path, default=WORK / "hiwiki-pages-articles.xml.bz2")
    ap.add_argument("--out", type=Path, default=WORK / "wiki_plain.txt")
    args = ap.parse_args()
    extract(args.dump, args.out)
    print(f"wrote {args.out} ({args.out.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
