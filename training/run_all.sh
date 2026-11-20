#!/usr/bin/env bash
# run_all.sh — build every trained model the toolkit's scorers need.
#
#   bash training/run_all.sh                 # everything, in order
#   STEPS="pcfg" bash training/run_all.sh    # just some steps
#
# Steps (each writes into scoring/models/ unless noted):
#   wiki     download + extract + split Hindi Wikipedia -> training/work/wiki_sentences.txt
#   vocab    30k LSTM vocabulary                        -> vocab.pkl
#   trigram  Katz-backoff trigram                       -> trigram_katz/
#   lstm     2-layer LSTM (GPU recommended)             -> base_model_wiki50m.pt
#   pcfg     DS-PS conversion + Berkeley grammar        -> hdtb_dsps_grammar
#
# Runs on any Linux/macOS machine with Python 3.11 and Java (also in WSL).
# On AWS use training/train_models.sh, which runs this script on a GPU box.
# Environment:
#   PYTHON      python interpreter (default: python3)
#   WIKI_DATE   dump date for download.py (default: latest)
#   LSTM_ARGS   extra arguments for train_lstm.py, e.g. "--minutes 60"
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python3}"
STEPS="${STEPS:-wiki vocab trigram lstm pcfg}"
WIKI_DATE="${WIKI_DATE:-latest}"
LSTM_ARGS="${LSTM_ARGS:-}"

banner() { echo; echo "=============== $* — $(date -u +%H:%M:%S) ==============="; }
has() { case " $STEPS " in *" $1 "*) return 0;; *) return 1;; esac; }

mkdir -p training/work scoring/models

if has wiki; then
    banner "wiki: download"
    "$PYTHON" training/wiki/download.py --date "$WIKI_DATE"
    banner "wiki: extract"
    "$PYTHON" training/wiki/extract.py
    banner "wiki: sentences"
    "$PYTHON" training/wiki/sentences.py
fi
if has vocab; then
    banner "vocab"
    "$PYTHON" training/wiki/build_vocab.py
fi
if has trigram; then
    banner "trigram"
    "$PYTHON" training/trigram/train_trigram.py
fi
if has lstm; then
    banner "lstm"
    # shellcheck disable=SC2086
    "$PYTHON" training/lstm/train_lstm.py $LSTM_ARGS
fi
if has pcfg; then
    banner "pcfg: DS-PS conversion"
    "$PYTHON" training/pcfg/generate_dsps_mrg.py
    banner "pcfg: grammar"
    bash training/pcfg/train_grammar.sh
fi

banner "done"
ls -la scoring/models
