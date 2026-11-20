#!/usr/bin/env bash
# train_grammar.sh — train the Berkeley latent-annotation PCFG used by the
# berkeley_pcfg scorer.
#
#   bash training/pcfg/train_grammar.sh
#
# Input  : training/work/hdtb_train_dsps.mrg   (training/pcfg/generate_dsps_mrg.py)
# Output : scoring/models/hdtb_dsps_grammar     (~1.7 MB, gzipped Java object)
#
# Four split-merge cycles (-SMcycles 4) over all 13,282 DS-PS trees, sentence
# length limit 10,000 (i.e. none). Needs Java 11+ and ~3 GB heap; about
# 7 minutes on 4 vCPUs. The jar is the Berkeley Parser 1.7 already bundled with
# the Taru tool (taru/external_resources/berkeleyparser/, GPL v2).
#
# Override with environment variables: SM_CYCLES, JAVA_HEAP, MRG, OUT.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
JAR="$ROOT/taru/external_resources/berkeleyparser/berkeleyParser.jar"
MRG="${MRG:-$ROOT/training/work/hdtb_train_dsps.mrg}"
OUT="${OUT:-$ROOT/scoring/models/hdtb_dsps_grammar}"
SM_CYCLES="${SM_CYCLES:-4}"
JAVA_HEAP="${JAVA_HEAP:-3g}"

[ -f "$MRG" ] || { echo "missing $MRG — run training/pcfg/generate_dsps_mrg.py first" >&2; exit 1; }
command -v java >/dev/null || { echo "java not found on PATH" >&2; exit 1; }

mkdir -p "$(dirname "$OUT")"
echo "training grammar: $(wc -l < "$MRG") trees, SMcycles=$SM_CYCLES -> $OUT"
java -Xmx"$JAVA_HEAP" -Dfile.encoding=UTF-8 -cp "$JAR" \
    edu.berkeley.nlp.PCFGLA.GrammarTrainer \
    -treebank SINGLEFILE \
    -path "$MRG" \
    -out "$OUT" \
    -SMcycles "$SM_CYCLES" \
    -maxL 10000

[ -s "$OUT" ] || { echo "GrammarTrainer finished but $OUT is missing" >&2; exit 1; }
echo "done: $OUT ($(du -h "$OUT" | cut -f1))"
