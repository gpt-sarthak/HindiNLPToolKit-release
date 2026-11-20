# Third-party data and software

This repository bundles material that is not original to it. Each item keeps
its own licence; using or redistributing this repository means following them.

## Data (`data/`)

| File | What | Licence / source |
|---|---|---|
| `hi_hdtb-ud-train.conllu` | Universal Dependencies Hindi-HDTB, training split (13,306 sentences) | **CC BY-NC-SA 4.0** — non-commercial use only. https://github.com/UniversalDependencies/UD_Hindi-HDTB |
| `fact_media_news_ISCNLP.conllu` | 17,405 Hindi news sentences, Paninian (`k1`/`k2`) annotation, used by the test suite | Supplied by the project team; no upstream licence is recorded. Check with the original provider before reusing it outside this project. |

Models built by `training/` are derived from these files and from Hindi
Wikipedia text (https://dumps.wikimedia.org/hiwiki/, CC BY-SA 4.0); the
Wikipedia text itself is not shipped.

## Software

| Path | What | Licence |
|---|---|---|
| `taru/external_resources/berkeleyparser/` | Berkeley Parser 1.7 (Petrov et al.) — used by the `berkeley_pcfg` scorer, the grammar trainer and the Taru tool | GPL v2 (`LICENSE` in that folder) |
| `taru/resource-*/` | ModelBlocks incremental parser sources (`synproc` etc.) behind the Taru tool | GPL v3 or later (file headers) |
| `taru/workspace/genmodel/` | Taru's bundled HDTB grammar (`hdtb_fresh.gr`) and synproc model, shipped as-is; no training script is included for them | as distributed with Taru |
| `training/pcfg/ds_ps_ud/` | DS-PS (dependency → phrase structure) conversion tool for UD Hindi, lightly modified | as distributed by its authors |

The web app's dependency-tree drawing (`webapp/static/index.html`,
`buildUDTreeSVG`) is a port of `rug-compling/conllu-viewer`; its BSD-2 notice
is kept in the source.
