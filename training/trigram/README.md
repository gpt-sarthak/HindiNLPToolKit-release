# training/trigram — Katz-backoff trigram language model

Builds `scoring/models/trigram_katz/`, the model behind the `trigram` scorer.
Pure NumPy, CPU only, ~12 minutes and ~2-3 GB RAM.

```bash
python training/trigram/train_trigram.py            # full build
python training/trigram/train_trigram.py --smoke    # ~1 min plumbing test (writes to training/work/trigram_smoke/)
python training/trigram/train_trigram.py --from 4   # redo from stage 4 (stages cache their output)
python training/trigram/train_trigram.py --force    # ignore every cache
```

Input: `training/work/wiki_sentences.txt` (from `training/wiki/`), and
`data/hi_hdtb-ud-train.conllu` only to report the treebank OOV rate.

## Stages

| # | Stage | What happens |
|---|---|---|
| 2 | corpus | seeded (42) sample: 1,000,000 training + 20,000 held-out sentences |
| 3 | vocabulary | the 30,000 most frequent training words (incl. `<UNK>`, `<s>`, `</s>`); the rest map to `<UNK>` |
| 4 | counting | every 1-, 2-, 3-gram of each sentence padded as `<s> <s> … </s>`, spilling to disk past 40 M keys |
| 5 | model | Good-Turing discounts for counts 1-7 (SRILM's default) + Katz backoff weights; written as memory-mapped `.npy` arrays |
| 6 | test | aborts unless P(next word) sums to 1.0 over the vocabulary; reports held-out perplexity and OOV rates to `training/work/trigram/lm_report.json` |

(Stage 1 of the original pipeline only copied input files and is not needed here.)

Output: 10 files — `meta.json`, `vocab.json`, `uni_logp.npy`, `bi_key.npy`,
`bi_logp.npy`, `bi_alpha_log.npy`, `tri_key.npy`, `tri_logp.npy`,
`tri_ctx_key.npy`, `tri_alpha_log.npy` (~148 MB). The scorer memory-maps them,
so loading takes ~0.05 s. Surprisal is reported in **bits**.

Shipped model: held-out perplexity **128.52**. Deterministic for a given
`wiki_sentences.txt`.

Why padding matters: an unpadded trigram scorer never scores the first two
words or the sentence end — exactly where preverbal reordering moves words.
Scoring the padded way is worth about +12 points of ranking accuracy.
