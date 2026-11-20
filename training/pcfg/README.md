# training/pcfg — DS-PS phrase-structure grammar for `berkeley_pcfg`

Builds `scoring/models/hdtb_dsps_grammar`, a Berkeley latent-annotation PCFG.
Needs Python and Java 11+; CPU only.

```bash
python training/pcfg/generate_dsps_mrg.py     # HDTB dependency trees -> training/work/hdtb_train_dsps.mrg (~6 min)
bash   training/pcfg/train_grammar.sh         # .mrg -> scoring/models/hdtb_dsps_grammar           (~7 min)
```

## Step 1 — dependency trees to phrase-structure trees

`generate_dsps_mrg.py` runs the bundled DS-PS conversion tool
(`ds_ps_ud/`, adapted for UD Hindi) over `data/hi_hdtb-ud-train.conllu`:

1. **preprocess** — drop comments, multi-word tokens and empty nodes; move
   `ChunkId` from MISC into FEATS, the tool's input format;
2. **CoNLL2SSF** — group tokens into chunks and build the chunk dependency graph (SSF);
3. **SSF2PS** — apply phrase-structure rules (explicit VP, NP-SUBJ / NP-OBJ-1 / NP-OBJ-2);
4. **TC** — fix brackets, remove null and unary nodes → PTB-style `.mrg`;
5. drop trees that crash the Berkeley parser.

Result: 13,282 of the 13,306 sentences, one tree per line. Deterministic — the
output is byte-identical to the file the shipped grammar was trained on
(MD5 `5b558660e21e3349f2b1fc6938cdda66`). `--keep-work` keeps the intermediate
SSF/PS files for inspection.

## Step 2 — train the grammar

`train_grammar.sh` runs Berkeley Parser 1.7's trainer (the jar bundled in
`taru/external_resources/berkeleyparser/`):

```
java -Xmx3g -cp berkeleyParser.jar edu.berkeley.nlp.PCFGLA.GrammarTrainer \
     -treebank SINGLEFILE -path hdtb_train_dsps.mrg -out hdtb_dsps_grammar \
     -SMcycles 4 -maxL 10000
```

Four split-merge cycles, all trees, no length limit — the settings of the
shipped grammar. Override with `SM_CYCLES=…`, `JAVA_HEAP=…`, `MRG=…`, `OUT=…`.

## A caveat for evaluation

The grammar is trained on **all** HDTB training trees, and the reproducibility
table scores sentences from the same treebank — the PCFG has seen the test
sentences, which inflates its accuracy there (the paper used 5-fold
cross-validation). Treat its row in `reproducibility/results/` as an upper bound.
