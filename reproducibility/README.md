# reproducibility/ — the paper comparison, rebuilt from scratch

Reproduces the evaluation of Ranjan & van Schijndel (2024), *Does Dependency
Locality Predict Non-canonical Word Order in Hindi?*
([arXiv:2405.07730](https://arxiv.org/abs/2405.07730)), with this toolkit's
own pipeline and models, on the UD Hindi-HDTB training treebank.

```bash
python reproducibility/build_features.py   # ~65 min -> reproducibility/data/ (git-ignored, ~130 MB)
python reproducibility/evaluate.py         # ~1 min  -> reproducibility/results/*.csv
```

Needs the trained models in `scoring/models/` (`training/README.md`) and Java
for the PCFG.

## 1. `build_features.py` — the feature table

Runs exactly what the web app's File mode does, on
`data/hi_hdtb-ud-train.conllu`:

1. **load** the 13,306 trees;
2. **filter** with the seven filters (questions, negation, empty nodes,
   non-projective trees, non-verbal roots, punctuation constituents, and at
   least 2 preverbal constituents including a subject and an object) — the
   paper's criteria;
3. **generate variants**: every permutation of the preverbal constituents whose
   adjacent dependency-label pairs occur in the treebank, at most 99 per
   reference (seed 42);
4. **label** each side SOV / DOSV / IOSV;
5. **score** with all five scorers, the PCFG in a parallel thread;
6. add the recommended order.

Output in `reproducibility/data/`: `features.csv` (one row per reference /
variant pair: sentences, word-order labels, every `*_Reference`, `*_Variant`
and `Delta_*` column), `passed_sentences.csv`, `rejected_sentences.csv`,
`summary.json`, and `scores/<scorer>.pkl` caches so an interrupted run resumes
where it stopped (delete `scores/` to rescore).

The pairwise coding is the paper's: `ML_Label` alternates 1 / 0, and each
`Delta_*` is reference − variant on 1-rows and variant − reference on 0-rows.

## 2. `evaluate.py` — the logistic regressions

Reads `features.csv` and, after checking the `ML_Label` / delta coding:

- splits the pairs into the paper's categories — full dataset, and references
  with a fronted object (OSV, split into DOSV / IOSV) against SOV variants;
- **univariate accuracy**: for each predictor alone, 10-fold cross-validated
  logistic regression (shuffled, seed 42, z-scored inside each fold), pooled
  out-of-fold accuracy (`results/univariate_accuracy.csv`);
- **coefficients**: one logistic regression on all five predictors, z-scored,
  `C = 1e6` (`results/regression_coefficients.csv`); negative = the human
  order minimises that measure.

The paper's published numbers are printed beside ours. The same estimator is
the library's `evaluation.evaluate_pairs`.

## Results with the shipped models

85,648 pairs from 2,294 references (paper's category sizes in brackets):
OSV 3,036 (2,834), DOSV 1,765 (1,663), IOSV 1,271 (1,353).

| Accuracy (%) | Full | OSV ours / paper | DOSV ours / paper | IOSV ours / paper |
|---|---:|---:|---:|---:|
| Dependency length | 61.86 | 63.31 / 65.24 | 60.96 / 68.49 | 65.15 / 58.91 |
| IS score | 52.96 | 59.72 / 52.19 | 60.28 / 53.88 | 59.09 / 50.92 |
| PCFG surprisal (DSPS) | 88.53 | 74.47 / 67.22 | 66.57 / 59.05 | 85.29 / 75.91 |
| Trigram surprisal (Katz) | 89.31 | 80.76 / 83.16 | 77.51 / 78.95 | 85.13 / 87.29 |
| Adaptive LSTM (Wiki-50M) | 92.45 | 79.91 / 83.84 | 74.22 / 79.98 | 87.73 / 88.32 |

## Read these with care

- **Not like-for-like with the paper.** The paper's trigram is SRILM on the
  EMILLE corpus, its LSTM a different model, its PCFG the Modelblocks parser;
  ours are Katz on Hindi Wikipedia, a Wiki-50M LSTM and a Berkeley DS-PS
  grammar. Its treebank is HUTB; ours is UD Hindi-HDTB. The pair sets differ
  too, so category comparisons are indicative.
- **The PCFG has seen these sentences.** Its grammar is trained on all HDTB
  training trees (`training/pcfg/`), so its accuracy here is inflated; the paper
  used 5-fold cross-validation.
- **The LSTM retrain is not bit-exact** (`training/README.md`), so with a freshly
  trained LSTM its row moves slightly; every other row reproduces exactly with
  models rebuilt from the same Wikipedia dump.
