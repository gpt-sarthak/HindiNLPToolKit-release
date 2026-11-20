# training/wiki — Hindi Wikipedia text and the LSTM vocabulary

Both language models (trigram and LSTM) are trained on the same file,
`training/work/wiki_sentences.txt`, built here in three steps. A fourth step
builds the LSTM's vocabulary.

```bash
pip install -r training/requirements-train.txt
python training/wiki/download.py            # [--date 20260201]  -> work/hiwiki-pages-articles.xml.bz2  (~230 MB)
python training/wiki/extract.py             #                    -> work/wiki_plain.txt                (~850 MB, 20-40 min)
python training/wiki/sentences.py           #                    -> work/wiki_sentences.txt            (~627 MB, 5-10 min)
python training/wiki/build_vocab.py         #                    -> scoring/models/vocab.pkl           (2-5 min)
```

| Script | What it does |
|---|---|
| `download.py` | Downloads `hiwiki-<date>-pages-articles.xml.bz2` from https://dumps.wikimedia.org/hiwiki/ and prints its MD5. |
| `extract.py` | Streams the dump with `mwxml`; for every page, strips wiki markup from its text with `mwparserfromhell.strip_code()`; one article per line. |
| `sentences.py` | Splits on `।`, `!`, `?`; replaces everything outside Devanagari (U+0900–U+097F) except space, comma and hyphen with a space; keeps sentences of 5–50 tokens. |
| `build_vocab.py` | `<PAD>`, `<UNK>`, `<SOS>`, `<EOS>` (ids 0–3) + the 29,996 most frequent words. |

The scorers clean an input sentence with exactly the same rules
(`normalize` in `scoring/_katz_trigram.py` and `scoring/_lstm_wiki50m.py`), so
what a model sees at scoring time matches what it was trained on.

## Which dump the shipped models used

Downloaded on **2026-02-19** from `latest/` — most likely the **20260201** dump:

| File | Size | MD5 |
|---|---|---|
| `hiwiki-…-pages-articles.xml.bz2` | 230,902,423 B | `354ee8df18480b1e30ef5b3c40a67261` |
| `wiki_plain.txt` | ~850 MB | `052c81223858762c2f638eb94da9bc39` |
| `wiki_sentences.txt` | ~627 MB (2,711,172 sentences) | `615419998ef3a2c3be0e3a85b0bff045` |
| `vocab.pkl` | — | `795fbd374f67e0363b10158bce2d19e0` |

The text-file checksums are of the LF-line-ending files these scripts write on
every OS (the originals were written with CRLF on Windows; the content is the
same).

Wikimedia keeps only the last few months of dumps. To rebuild the shipped
models exactly, find that dump on a mirror (e.g. search archive.org for
"hiwiki 20260201") and check the MD5. A newer dump works the same way and gives
models of the same quality, with slightly different numbers. With the pinned
`mwxml` / `mwparserfromhell` versions, `extract.py` reproduces the original
`wiki_plain.txt` line for line (checked on the first 3,191 lines).
