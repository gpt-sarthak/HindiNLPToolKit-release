# Recommender weights (`scoring/models/recommender_weights.json`)

Sentence mode ranks a sentence's candidate orders by a weighted mean of each
scorer's z-scored, direction-oriented score. The weights are the absolute
standardized coefficients of a one-predictor logistic regression per scorer
(Joachims pairwise transform, `C = 1e6`), fitted offline. The sign always comes
from the scorer's declared `higher_is_more_natural`, never from the fit.

The file is committed (it is small), so nothing needs training. Its weights:

| Delta | Weight | Accuracy | Fitted on |
|---|---:|---:|---|
| `Delta_AdaptiveWiki50M` | 6.0666 | 92.6% | 92,299 HDTB pairs |
| `Delta_Trigram` | 4.9497 | 89.5% | 92,299 HDTB pairs |
| `Delta_PCFG` | 0.8956 | 63.4% | 88,436 HDTB pairs (rows with a PCFG score) |
| `Delta_DL` | 0.6377 | 61.7% | 193,474 HDTB pairs |
| `Delta_IS` | 0.3058 | 53.1% | 92,299 HDTB pairs |

The 92,299-pair table is the earlier replication's feature table (not shipped);
`Delta_DL` came from a full-corpus run made before the verb-only root filter.

## Refitting them yourself

`evaluation/fit_weights.py` runs the whole pipeline on a corpus and writes the
file. It **overwrites** it with only the scorers you name, so name all five:

```bash
python -m evaluation.fit_weights \
    --corpus data/hi_hdtb-ud-train.conllu --scheme ud \
    --scorers dependency_length,information_status,trigram,adaptive_lstm_wiki50m,berkeley_pcfg \
    --out scoring/models/recommender_weights.json
```

That takes about as long as `reproducibility/build_features.py` (~1 hour; the
PCFG dominates — set `HNT_PCFG_TIMEOUT=10800`). Expect weights close to, but not
identical with, the table above: the pair set differs (this toolkit's filters
give 85,648 HDTB pairs), and the PCFG has seen the HDTB trees it is scored on,
so its refit weight comes out inflated. Keeping the shipped `Delta_PCFG` weight
is recommended for that reason.
