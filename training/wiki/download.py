"""
download.py — fetch the Hindi Wikipedia article dump.

The language models (trigram + LSTM) are trained on Hindi Wikipedia text. This
script downloads the compressed XML dump from Wikimedia.

    python training/wiki/download.py                    # latest dump
    python training/wiki/download.py --date 20260201    # a fixed dump

Output: training/work/hiwiki-pages-articles.xml.bz2   (~230 MB)

The shipped models were built from the dump fetched on 2026-02-19 from
``latest/`` (230,902,423 bytes, MD5 354ee8df18480b1e30ef5b3c40a67261), which is
most likely the 20260201 dump. Wikimedia keeps only the last few months of
dumps, so an older date may be gone; https://archive.org mirrors many of them
(search "hiwiki 20260201"). A newer dump gives slightly different models — see
training/README.md, "How exact is a retrain?".
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.request
from pathlib import Path

BASE = "https://dumps.wikimedia.org/hiwiki"
WORK = Path(__file__).resolve().parents[1] / "work"


def dump_url(date: str) -> str:
    return f"{BASE}/{date}/hiwiki-{date}-pages-articles.xml.bz2"


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", default="latest",
                    help="dump date like 20260201, or 'latest' (default)")
    ap.add_argument("--url", default=None, help="override the full download URL")
    ap.add_argument("--out", type=Path, default=WORK / "hiwiki-pages-articles.xml.bz2")
    ap.add_argument("--force", action="store_true", help="download even if the file exists")
    args = ap.parse_args()

    url = args.url or dump_url(args.date)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists() and not args.force:
        print(f"present: {args.out} ({args.out.stat().st_size:,} bytes) — use --force to refetch")
    else:
        print(f"downloading {url}")
        tmp = args.out.with_suffix(args.out.suffix + ".part")
        req = urllib.request.Request(url, headers={"User-Agent": "hindi-nlp-toolkit-training"})
        with urllib.request.urlopen(req) as resp, open(tmp, "wb") as fh:
            shutil.copyfileobj(resp, fh, length=1 << 20)
        tmp.replace(args.out)
        print(f"saved {args.out} ({args.out.stat().st_size:,} bytes)")
    print(f"md5 {md5_of(args.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
