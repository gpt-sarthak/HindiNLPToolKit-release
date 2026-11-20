#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_trigram.py — build the Katz-backoff trigram model used by the
``trigram`` scorer.

    python training/trigram/train_trigram.py            # full build (~12 min, ~2-3 GB RAM)
    python training/trigram/train_trigram.py --smoke    # 1-minute plumbing test
    python training/trigram/train_trigram.py --from 4   # redo from stage 4
    python training/trigram/train_trigram.py --force    # ignore every cache

Input  : training/work/wiki_sentences.txt   (from training/wiki/sentences.py)
         data/hi_hdtb-ud-train.conllu       (only to report the treebank OOV rate)
Output : scoring/models/trigram_katz/       (10 memory-mapped .npy/.json files, ~148 MB)
         training/work/trigram/lm_report.json  (perplexity, OOV, sum-to-1 error)

STAGES (each caches its output in training/work/trigram/)
    2. corpus ....... seeded split of the sentences: 1,000,000 train + 20,000 held-out
    3. vocabulary ... the 30,000 most frequent training words; the rest -> <UNK>
    4. counting ..... every 1-, 2- and 3-gram of the <s> <s> ... </s>-padded sentences
    5. model ........ Good-Turing discounts (counts 1-7) + Katz backoff weights
    6. test ......... held-out perplexity, OOV rates, and a hard gate that
                      P(next word) sums to 1.0 over the vocabulary

The scorer (scoring/_katz_trigram.py) carries the same ``normalize`` and
``TrigramLM`` code, so a sentence is cleaned and scored exactly as the
training text was. The shipped model reports held-out perplexity 128.52.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter
from itertools import permutations
from pathlib import Path

import numpy as np

# ==============================================================================
#  CONFIG  - every tunable number lives here, nothing magic hides in the code
# ==============================================================================

ROOT = Path(__file__).resolve().parents[2]

CONFIG = {
    # ---- inputs
    "sentences": ROOT / "training" / "work" / "wiki_sentences.txt",
    "treebank":  ROOT / "data" / "hi_hdtb-ud-train.conllu",

    # ---- outputs
    "work":  ROOT / "training" / "work" / "trigram",
    "model": ROOT / "scoring" / "models" / "trigram_katz",
    "out":   ROOT / "training" / "work" / "trigram",

    # ---- STAGE 2: corpus construction
    # Cleaning scheme. "research_repo" = Devanagari only; the danda and ASCII
    # digits are deleted. The same normalize() runs at scoring time.
    "cleaning":        "research_repo",
    "min_tokens":      5,
    "max_tokens":      50,
    "n_train":         1_000_000,   # the paper used 1M EMILLE sentences
    "n_heldout":       20_000,

    # ---- STAGE 3: vocabulary
    "vocab_size":      30_000,      # including the 3 special tokens

    # ---- STAGE 4: counting
    "order":           3,
    "spill_threshold": 40_000_000,  # keys held in RAM before spilling to disk

    # ---- STAGE 5: smoothing (Katz backoff + Good-Turing, as in the paper)
    "gt_max":          7,           # SRILM's gtmax default: discount counts 1..7
    "alpha_floor":     1e-10,       # backoff weights never collapse to zero

    "seed":            42,
}

UNK, BOS, EOS = "<UNK>", "<s>", "</s>"


# ==============================================================================
#  Small helpers used throughout
# ==============================================================================

class Stage:
    """Prints a banner and the elapsed time for one pipeline stage."""

    def __init__(self, number, title):
        self.number, self.title = number, title

    def __enter__(self):
        self.t0 = time.time()
        print()
        print("=" * 78)
        print(f" STAGE {self.number}  -  {self.title}")
        print("=" * 78)
        return self

    def __exit__(self, *exc):
        if exc[0] is None:
            print(f"   [stage {self.number} done in {time.time() - self.t0:.1f}s]")
        return False


def log(msg=""):
    print(f"   {msg}")


def progress(i, every=500_000, what="lines"):
    if i and i % every == 0:
        print(f"   ... {i:,} {what}", flush=True)


def human(n_bytes):
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f}{unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f}TB"


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


# ==============================================================================
#  TEXT NORMALIZATION
#  One function, used for BOTH training text and sentences we score, so the
#  model is never asked about tokens it could not have been trained on.
# ==============================================================================

# Everything outside the Devanagari block (plus space, comma, hyphen) is removed.
# NOTE: the danda U+0964 IS inside this block, but sentence splitting consumes
# it as a delimiter, so it does not survive into training text.
_DEVANAGARI_ONLY = re.compile(r"[^\u0900-\u097F\s,\-]")

# The "treebank_match" alternative: also keep the danda and digits.
_KEEP_DANDA_DIGITS = re.compile(r"[^\u0900-\u097F0-9\s,\-\u0964।]")

_SENTENCE_SPLIT = re.compile(r"[।!?]")
_WHITESPACE = re.compile(r"\s+")


def normalize(text, scheme=None):
    """Clean a piece of Hindi text down to the token conventions the LM uses.

    Applied to Wikipedia text at training time AND to treebank sentences at
    scoring time, so the model is never asked about tokens it could not have
    been trained on.  Returns a whitespace-joined string (possibly empty).

    Note the danda: training text never contains one, because sentence
    splitting consumes it as a delimiter.  So under the "research_repo" scheme
    we must delete it here too - otherwise every treebank sentence (91% of them
    end in a danda) would score its last token as out-of-vocabulary.  That
    mismatch is exactly the bug in the existing implementation.
    """
    scheme = scheme or CONFIG["cleaning"]
    if scheme == "research_repo":
        text = _SENTENCE_SPLIT.sub(" ", text)
        pattern = _DEVANAGARI_ONLY
    else:
        pattern = _KEEP_DANDA_DIGITS
    text = pattern.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()


def split_sentences(line):
    """Split a paragraph into sentences on danda / ! / ? (the delimiter is
    consumed), then normalize each piece."""
    for piece in _SENTENCE_SPLIT.split(line):
        cleaned = normalize(piece)
        if cleaned:
            yield cleaned


# ==============================================================================
#  STAGE 2  -  CORPUS: clean sentences -> seeded train / held-out split
# ==============================================================================

def stage2_build_corpus(force=False, smoke=False):
    with Stage(2, "CORPUS - Wikipedia sentences into a train / held-out split"):
        work = CONFIG["work"]
        work.mkdir(parents=True, exist_ok=True)
        all_sents = CONFIG["sentences"]
        train_f = work / "train.txt"
        heldout_f = work / "heldout.txt"

        n_train = 2_000 if smoke else CONFIG["n_train"]
        n_heldout = 500 if smoke else CONFIG["n_heldout"]

        if not all_sents.exists():
            raise FileNotFoundError(
                f"{all_sents} not found - run training/wiki/sentences.py first")
        if smoke:
            small = work / "wiki_sentences_smoke.txt"
            with open(all_sents, "r", encoding="utf-8") as fin, \
                 open(small, "w", encoding="utf-8") as fout:
                for i, line in enumerate(fin):
                    if i >= 20_000:
                        break
                    fout.write(line)
            all_sents = small
        log(f"sentences  {all_sents} ({human(all_sents.stat().st_size)})")

        if train_f.exists() and heldout_f.exists() and not force:
            log(f"present  train.txt + heldout.txt")
        else:
            total = sum(1 for _ in open(all_sents, "r", encoding="utf-8"))
            need = n_train + n_heldout
            if total < need:
                raise RuntimeError(
                    f"only {total:,} sentences available, need {need:,}. "
                    f"Lower CONFIG['n_train'].")
            log(f"{total:,} sentences available; sampling {need:,} (seed {CONFIG['seed']})")

            rng = random.Random(CONFIG["seed"])
            picked = rng.sample(range(total), need)
            rng.shuffle(picked)
            heldout_idx = set(picked[:n_heldout])
            train_idx = set(picked[n_heldout:])

            with open(all_sents, "r", encoding="utf-8") as fin, \
                 open(train_f, "w", encoding="utf-8") as ftr, \
                 open(heldout_f, "w", encoding="utf-8") as fho:
                for i, line in enumerate(fin):
                    if i in train_idx:
                        ftr.write(line)
                    elif i in heldout_idx:
                        fho.write(line)

        n_tr = sum(1 for _ in open(train_f, "r", encoding="utf-8"))
        n_ho = sum(1 for _ in open(heldout_f, "r", encoding="utf-8"))
        log(f"train    {n_tr:,} sentences")
        log(f"heldout  {n_ho:,} sentences (the model never sees these)")
        return {"n_train": n_tr, "n_heldout": n_ho}


# ==============================================================================
#  STAGE 3  -  VOCABULARY: the N most frequent words; everything else is <UNK>
# ==============================================================================

def stage3_build_vocab(force=False, smoke=False):
    with Stage(3, "VOCABULARY - the most frequent words become the model's world"):
        vocab_f = CONFIG["work"] / "vocab.json"
        size = 2_000 if smoke else CONFIG["vocab_size"]

        if vocab_f.exists() and not force:
            vocab = json.loads(vocab_f.read_text(encoding="utf-8"))
            log(f"present  {len(vocab['word2id']):,} types")
            return vocab

        counts = Counter()
        with open(CONFIG["work"] / "train.txt", "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                progress(i, 200_000, "sentences counted")
                counts.update(line.split())
        log(f"{len(counts):,} distinct word types in the training text")

        specials = [UNK, BOS, EOS]
        top = [w for w, _ in counts.most_common(size - len(specials))]
        words = specials + top
        word2id = {w: i for i, w in enumerate(words)}

        covered = sum(counts[w] for w in top)
        total = sum(counts.values())
        log(f"vocabulary capped at {len(words):,} types "
            f"(covers {100.0 * covered / total:.2f}% of all word occurrences)")
        log(f"the other {len(counts) - len(top):,} types collapse to <UNK>")

        vocab = {"word2id": word2id, "size": len(words),
                 "token_coverage": covered / total}
        write_json(vocab_f, vocab)
        return vocab


# ==============================================================================
#  STAGE 4  -  COUNT N-GRAMS
#
#  The memory-critical stage.  Storing counts in a Python dict would need 5-6 GB
#  and crash this laptop (that is exactly how the existing 226 MB pickle came
#  about).  Instead every n-gram is packed into a single int64:
#
#         key = w1*V*V + w2*V + w3
#
#  so 19 million trigrams live in one 152 MB numpy array.  Sorting that array
#  also groups every n-gram sharing a context together, because w1 and w2 are
#  the high-order digits - which is what makes STAGE 5 fast.
# ==============================================================================

def _encode_sentences(vocab, path, order):
    """Yield padded integer-ID sequences, one per line of *path*."""
    word2id = vocab["word2id"]
    unk, bos, eos = word2id[UNK], word2id[BOS], word2id[EOS]
    pad = [bos] * (order - 1)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            ids = [word2id.get(w, unk) for w in line.split()]
            if ids:
                yield pad + ids + [eos]


class _KeyCollector:
    """Collects int64 n-gram keys, spilling sorted chunks to disk if huge."""

    def __init__(self, name, work, threshold):
        self.name, self.work, self.threshold = name, work, threshold
        self.buf, self.buf_n, self.chunks = [], 0, []

    def add(self, arr):
        self.buf.append(arr)
        self.buf_n += arr.size
        # Coalesce periodically: one tiny array per sentence would mean a
        # million numpy objects and ~250 MB of pure per-object overhead.
        if len(self.buf) >= 20_000:
            self.buf = [np.concatenate(self.buf)]
        if self.buf_n >= self.threshold:
            self._spill()

    def _spill(self):
        if not self.buf:
            return
        chunk = np.concatenate(self.buf)
        chunk.sort()
        path = self.work / f"_chunk_{self.name}_{len(self.chunks)}.npy"
        np.save(path, chunk)
        self.chunks.append(path)
        self.buf, self.buf_n = [], 0

    def finish(self):
        """Return (unique_keys, counts), then clean up any spill files."""
        if self.chunks:
            self._spill()
            parts = [np.load(p) for p in self.chunks]
            allk = np.concatenate(parts)
            del parts
        else:
            allk = np.concatenate(self.buf) if self.buf else np.empty(0, np.int64)
        self.buf = []
        keys, counts = np.unique(allk, return_counts=True)
        del allk
        for p in self.chunks:
            p.unlink(missing_ok=True)
        self.chunks = []
        return keys, counts.astype(np.int64)


def stage4_count_ngrams(vocab, force=False):
    with Stage(4, "COUNT N-GRAMS - how often does each 1-, 2-, 3-word sequence occur"):
        work, V = CONFIG["work"], vocab["size"]
        out = {n: (work / f"count{n}_key.npy", work / f"count{n}_val.npy")
               for n in (1, 2, 3)}

        if all(p.exists() for pair in out.values() for p in pair) and not force:
            log("present  count arrays for orders 1, 2, 3")
            return {n: (np.load(k), np.load(v)) for n, (k, v) in out.items()}

        log(f"vocabulary size V = {V:,}   (trigram keys fit in int64: "
            f"max {V ** 3:,} < {np.iinfo(np.int64).max:,})")

        thr = CONFIG["spill_threshold"]
        c1 = _KeyCollector("1", work, thr)
        c2 = _KeyCollector("2", work, thr)
        c3 = _KeyCollector("3", work, thr)

        n_sent = n_tok = 0
        for i, ids in enumerate(_encode_sentences(vocab, work / "train.txt", 3), 1):
            progress(i, 200_000, "sentences counted")
            n_sent += 1
            a = np.asarray(ids, dtype=np.int64)
            n_tok += a.size - 2                      # exclude the two <s> pads

            # Unigrams: only positions that can be PREDICTED (skip the <s> pads),
            # because <s> is never an outcome, only ever a context.
            c1.add(a[2:])
            # Bigrams over the padded sequence: (a[i-1], a[i])
            c2.add(a[:-1] * V + a[1:])
            # Trigrams over the padded sequence: (a[i-2], a[i-1], a[i])
            c3.add(a[:-2] * V * V + a[1:-1] * V + a[2:])

        log(f"{n_sent:,} sentences, {n_tok:,} predicted tokens")

        result = {}
        for n, coll in ((1, c1), (2, c2), (3, c3)):
            keys, counts = coll.finish()
            np.save(out[n][0], keys)
            np.save(out[n][1], counts)
            log(f"{n}-grams: {len(keys):>12,} distinct   "
                f"{counts.sum():>13,} total occurrences")
            result[n] = (keys, counts)
        return result


# ==============================================================================
#  STAGE 5  -  BUILD THE MODEL: Good-Turing discounting + Katz backoff
#
#  Raw counting says anything unseen is IMPOSSIBLE (probability zero), which is
#  both wrong and fatal - surprisal is a logarithm and log(0) is infinity.  So:
#
#    Good-Turing  decides HOW MUCH probability to shave off what we did see,
#                 by looking at how many things we saw exactly once.
#    Katz backoff decides WHERE that shaved-off probability goes: to the
#                 2-word estimate, and failing that to the 1-word estimate.
#
#  This is the paper's method (it used SRILM, whose default is exactly this).
# ==============================================================================

def _simple_good_turing(counts, k):
    """Simple Good-Turing (Gale & Sampson 1995) -> adjusted counts r* for r=1..k+1.

    A plain Good-Turing estimator divides by N_r and breaks on real data as soon
    as some frequency r is unobserved.  SGT fixes that by fitting a line through
    log(N_r) and switching to the fitted values once the raw ones get noisy.
    """
    nr = np.bincount(counts)                    # nr[r] = how many n-grams occur r times
    r_obs = np.flatnonzero(nr)[1:]              # observed frequencies (drop r=0)
    if r_obs.size < 2:
        return {r: float(r) for r in range(1, k + 2)}, nr

    # Averaging transform: spread each N_r over the gap to its neighbours.
    z = np.empty(r_obs.size, float)
    for i, r in enumerate(r_obs):
        prev_r = r_obs[i - 1] if i > 0 else 0
        next_r = r_obs[i + 1] if i + 1 < r_obs.size else 2 * r - prev_r
        z[i] = nr[r] / (0.5 * (next_r - prev_r))

    # Fit log(Z_r) = a + b*log(r).
    b, a = np.polyfit(np.log(r_obs), np.log(z), 1)

    def lgt(r):                                  # smoothed estimate
        return r * (1.0 + 1.0 / r) ** (b + 1.0)

    r_star, use_lgt = {}, (b >= -1.0)            # b >= -1 means the fit is unusable
    for r in range(1, k + 2):
        if not use_lgt and r + 1 < nr.size and nr[r] > 0 and nr[r + 1] > 0:
            turing = (r + 1) * nr[r + 1] / nr[r]
            smooth = lgt(r)
            # Switch to the smoothed curve once the two agree within noise.
            sd = math.sqrt((r + 1) ** 2 * nr[r + 1] / nr[r] ** 2
                           * (1.0 + nr[r + 1] / nr[r]))
            if abs(turing - smooth) <= 1.96 * sd:
                use_lgt = True
                r_star[r] = smooth
            else:
                r_star[r] = turing
        else:
            use_lgt = True
            r_star[r] = lgt(r)
    return r_star, nr


def _katz_discounts(counts, k):
    """Discount factor d_r for each count r (1..k); counts above k are untouched."""
    r_star, nr = _simple_good_turing(counts, k)
    n1 = nr[1] if nr.size > 1 else 0
    nk1 = nr[k + 1] if nr.size > k + 1 else 0

    d = np.ones(k + 2, dtype=np.float64)
    if n1 == 0:
        return d, nr                             # nothing to discount
    common = (k + 1) * nk1 / n1
    for r in range(1, k + 1):
        raw = r_star[r] / r
        if 0.0 < common < 1.0:
            val = (raw - common) / (1.0 - common)
        else:
            val = raw
        d[r] = min(1.0, max(1e-3, val))          # keep it a sane fraction
    return d, nr


def _apply_discount(counts, d, k):
    """Vectorised: discounted count for every n-gram."""
    idx = np.minimum(counts, k + 1)
    factors = np.where(counts <= k, d[idx], 1.0)
    return counts * factors


def _context_blocks(ctx_ids):
    """Given a non-decreasing array of context ids, return (start_indices,
    block_lengths, unique_context_ids).  Works because sorting the packed keys
    already grouped every shared context together."""
    if ctx_ids.size == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.int64)
    starts = np.concatenate(([0], np.flatnonzero(np.diff(ctx_ids)) + 1))
    lengths = np.diff(np.concatenate((starts, [ctx_ids.size])))
    return starts, lengths, ctx_ids[starts]


def stage5_build_model(vocab, counts, force=False):
    with Stage(5, "BUILD MODEL - Good-Turing discounting and Katz backoff weights"):
        mdl, V, k = CONFIG["model"], vocab["size"], CONFIG["gt_max"]
        mdl.mkdir(parents=True, exist_ok=True)
        floor = CONFIG["alpha_floor"]

        if (mdl / "meta.json").exists() and not force:
            log("present  trained model in model/")
            return load_model()

        k1, v1 = counts[1]
        k2, v2 = counts[2]
        k3, v3 = counts[3]

        # ---------------------------------------------------------------- 1-gram
        log("1-gram: discounting and spreading the leftover over the vocabulary")
        d1, nr1 = _katz_discounts(v1, k)
        log(f"        singletons N_1={nr1[1] if nr1.size > 1 else 0:,}  "
            f"discounts d_1..d_3 = {d1[1]:.4f}, {d1[2]:.4f}, {d1[3]:.4f}")
        disc1 = _apply_discount(v1, d1, k)
        total1 = float(v1.sum())

        uni_p = np.zeros(V, dtype=np.float64)
        uni_p[k1] = disc1 / total1
        leftover = 1.0 - uni_p.sum()
        # <s> is never predicted, so it must never receive probability mass.
        predictable = np.ones(V, dtype=bool)
        predictable[vocab["word2id"][BOS]] = False
        unseen = predictable & (uni_p <= 0.0)
        if unseen.any() and leftover > 0:
            uni_p[unseen] = leftover / unseen.sum()
        else:
            uni_p[predictable] /= uni_p[predictable].sum()
        uni_p[~predictable] = 0.0
        log(f"        reserved {leftover:.4%} of probability for unseen words")

        # ---------------------------------------------------------------- 2-gram
        log("2-gram: P(w2|w1) with backoff weight per context word")
        d2, nr2 = _katz_discounts(v2, k)
        log(f"        singletons N_1={nr2[1] if nr2.size > 1 else 0:,}  "
            f"discounts d_1..d_3 = {d2[1]:.4f}, {d2[2]:.4f}, {d2[3]:.4f}")
        disc2 = _apply_discount(v2, d2, k)

        w1 = k2 // V                              # context word of each bigram
        w2 = k2 % V
        starts2, lens2, ctx2 = _context_blocks(w1)
        ctx_total2 = np.add.reduceat(v2.astype(np.float64), starts2)
        denom2 = np.repeat(ctx_total2, lens2)
        bi_p = disc2 / denom2

        # alpha(w1) = leftover mass / mass the lower-order model puts on unseen
        seen_p = np.add.reduceat(bi_p, starts2)
        seen_backoff = np.add.reduceat(uni_p[w2], starts2)
        bi_alpha = np.zeros(V, dtype=np.float64)
        num = 1.0 - seen_p
        den = 1.0 - seen_backoff
        safe = den > 1e-12
        bi_alpha[ctx2[safe]] = np.maximum(num[safe], 0.0) / den[safe]
        # Contexts we never saw at all fall straight through to the unigram model.
        never = np.ones(V, dtype=bool)
        never[ctx2] = False
        bi_alpha[never] = 1.0
        np.maximum(bi_alpha, floor, out=bi_alpha)

        # ---------------------------------------------------------------- 3-gram
        log("3-gram: P(w3|w1,w2) with backoff weight per context pair")
        d3, nr3 = _katz_discounts(v3, k)
        log(f"        singletons N_1={nr3[1] if nr3.size > 1 else 0:,}  "
            f"discounts d_1..d_3 = {d3[1]:.4f}, {d3[2]:.4f}, {d3[3]:.4f}")
        disc3 = _apply_discount(v3, d3, k)

        ctx3_ids = k3 // V                        # packs (w1,w2) into one number
        t_w2 = ctx3_ids % V
        t_w3 = k3 % V
        starts3, lens3, ctx3 = _context_blocks(ctx3_ids)
        ctx_total3 = np.add.reduceat(v3.astype(np.float64), starts3)
        denom3 = np.repeat(ctx_total3, lens3)
        tri_p = disc3 / denom3

        # For the backoff weight we need P(w3|w2) for every trigram we saw.
        # That bigram always exists: if w1 w2 w3 occurred, then so did w2 w3.
        q = t_w2 * V + t_w3
        pos = np.searchsorted(k2, q)
        if not (np.all(pos < k2.size) and np.all(k2[pos] == q)):
            raise RuntimeError("internal error: a trigram's backoff bigram is missing")
        backoff_p = bi_p[pos]

        seen_p3 = np.add.reduceat(tri_p, starts3)
        seen_backoff3 = np.add.reduceat(backoff_p, starts3)
        num3 = 1.0 - seen_p3
        den3 = 1.0 - seen_backoff3
        tri_alpha = np.where(den3 > 1e-12,
                             np.maximum(num3, 0.0) / np.maximum(den3, 1e-12),
                             floor)
        np.maximum(tri_alpha, floor, out=tri_alpha)

        # ---- save as log2 probabilities: scoring then becomes plain addition ---
        def log2(x):
            return np.log2(np.maximum(x, 1e-300)).astype(np.float32)

        np.save(mdl / "uni_logp.npy", log2(uni_p))
        np.save(mdl / "bi_key.npy", k2)
        np.save(mdl / "bi_logp.npy", log2(bi_p))
        np.save(mdl / "bi_alpha_log.npy", log2(bi_alpha))
        np.save(mdl / "tri_key.npy", k3)
        np.save(mdl / "tri_logp.npy", log2(tri_p))
        np.save(mdl / "tri_ctx_key.npy", ctx3)
        np.save(mdl / "tri_alpha_log.npy", log2(tri_alpha))
        write_json(mdl / "vocab.json", vocab)
        write_json(mdl / "meta.json", {
            "V": V, "order": 3, "gt_max": k,
            "cleaning": CONFIG["cleaning"],
            "n_unigram": int(k1.size), "n_bigram": int(k2.size),
            "n_trigram": int(k3.size), "n_tri_contexts": int(ctx3.size),
            "discounts": {"uni": d1[1:k + 1].tolist(),
                          "bi": d2[1:k + 1].tolist(),
                          "tri": d3[1:k + 1].tolist()},
        })

        size = sum(p.stat().st_size for p in mdl.glob("*.npy"))
        log(f"model written to model/  ({human(size)}, memory-mapped on load)")
        return load_model()


# ==============================================================================
#  THE MODEL: loading and scoring
# ==============================================================================

class TrigramLM:
    """Katz-backoff trigram model.  Loads by memory-mapping, so opening it is
    instant and costs almost no RAM (the existing 226 MB pickle needs minutes
    and several GB)."""

    def __init__(self, mdl):
        self.meta = json.loads((mdl / "meta.json").read_text(encoding="utf-8"))
        vocab = json.loads((mdl / "vocab.json").read_text(encoding="utf-8"))
        self.word2id = vocab["word2id"]
        self.V = self.meta["V"]
        self.unk = self.word2id[UNK]
        self.bos = self.word2id[BOS]
        self.eos = self.word2id[EOS]

        m = lambda name: np.load(mdl / name, mmap_mode="r")
        self.uni_logp = m("uni_logp.npy")
        self.bi_key = m("bi_key.npy")
        self.bi_logp = m("bi_logp.npy")
        self.bi_alpha_log = m("bi_alpha_log.npy")
        self.tri_key = m("tri_key.npy")
        self.tri_logp = m("tri_logp.npy")
        self.tri_ctx_key = m("tri_ctx_key.npy")
        self.tri_alpha_log = m("tri_alpha_log.npy")

    # -- lookups -------------------------------------------------------------
    @staticmethod
    def _find(keys, q):
        i = np.searchsorted(keys, q)
        return int(i) if i < keys.size and keys[i] == q else -1

    def logp_level(self, w1, w2, w3):
        """log2 P(w3 | w1, w2) and which level answered ('3gram'/'2gram'/'1gram')."""
        i = self._find(self.tri_key, w1 * self.V * self.V + w2 * self.V + w3)
        if i >= 0:
            return float(self.tri_logp[i]), "3gram"

        # back off to the bigram, paying this context's backoff weight
        c = self._find(self.tri_ctx_key, w1 * self.V + w2)
        a3 = float(self.tri_alpha_log[c]) if c >= 0 else 0.0

        j = self._find(self.bi_key, w2 * self.V + w3)
        if j >= 0:
            return a3 + float(self.bi_logp[j]), "2gram"

        a2 = float(self.bi_alpha_log[w2])
        return a3 + a2 + float(self.uni_logp[w3]), "1gram"

    def logp(self, w1, w2, w3):
        """log2 P(w3 | w1, w2) as a plain number."""
        return self.logp_level(w1, w2, w3)[0]

    def encode(self, text, already_clean=False):
        toks = (text if already_clean else normalize(text)).split()
        return [self.word2id.get(t, self.unk) for t in toks], toks

    # -- sentence scoring ----------------------------------------------------
    def surprisal(self, text, already_clean=False):
        """Total surprisal in bits.  Every word is scored, including the first
        two, via <s> <s> padding - an unpadded scorer skips those, which biases
        every reference-vs-variant comparison."""
        ids, _ = self.encode(text, already_clean)
        if not ids:
            return 0.0
        seq = [self.bos, self.bos] + ids + [self.eos]
        return -sum(self.logp(seq[i - 2], seq[i - 1], seq[i])
                    for i in range(2, len(seq)))

    def per_word(self, text, already_clean=False):
        """Per-word breakdown: [(word, context, level, log2p, surprisal), ...]"""
        ids, toks = self.encode(text, already_clean)
        if not ids:
            return []
        id2w = {i: w for w, i in self.word2id.items()}
        seq = [self.bos, self.bos] + ids + [self.eos]
        shown = [BOS, BOS] + toks + [EOS]
        rows = []
        for i in range(2, len(seq)):
            lp, level = self.logp_level(seq[i - 2], seq[i - 1], seq[i])
            surface = shown[i]
            if i - 2 < len(toks) and seq[i] == self.unk and toks[i - 2] != UNK:
                surface = f"{toks[i - 2]}[UNK]"
            rows.append((surface, f"{shown[i - 2]} {shown[i - 1]}",
                         level, lp, -lp))
        return rows

    def oov_rate(self, sentences):
        n = oov = 0
        for s in sentences:
            ids, _ = self.encode(s)
            n += len(ids)
            oov += sum(1 for i in ids if i == self.unk)
        return (oov / n) if n else 0.0


def load_model():
    return TrigramLM(CONFIG["model"])


# ==============================================================================
#  STAGE 6  -  TEST THE LANGUAGE MODEL
#  Perplexity on sentences it has never seen, plus a hard gate: the
#  probabilities of all possible next words must sum to 1.0.  The existing
#  MLE-plus-fallback model fails that gate, which is the whole point of
#  implementing real smoothing.
# ==============================================================================

def stage6_test_lm(lm, force=False):
    with Stage(6, "TEST THE MODEL - perplexity, OOV rate, and a sum-to-1.0 gate"):
        rng = np.random.default_rng(CONFIG["seed"])

        # ---- gate: is this a real probability distribution? ------------------
        log("checking that P(next word) sums to 1.0 over the whole vocabulary...")
        pick = rng.integers(0, lm.tri_ctx_key.size, size=8)
        worst = 0.0
        for c in pick:
            ctx = int(lm.tri_ctx_key[c])
            w1, w2 = ctx // lm.V, ctx % lm.V
            total = 0.0
            for w3 in range(lm.V):
                if w3 == lm.bos:            # <s> is never predicted
                    continue
                total += 2.0 ** lm.logp(w1, w2, w3)
            worst = max(worst, abs(total - 1.0))
        log(f"largest deviation from 1.0 across 8 sampled contexts: {worst:.2e}")
        if worst > 1e-4:
            raise RuntimeError(
                f"the model is not a valid probability distribution "
                f"(off by {worst:.2e}) - the backoff weights are wrong")
        log("PASS - the smoothing produces genuine probabilities")

        # ---- held-out perplexity ---------------------------------------------
        heldout = [l.strip() for l in
                   open(CONFIG["work"] / "heldout.txt", "r", encoding="utf-8")]
        log(f"scoring {len(heldout):,} held-out sentences the model has never seen")
        bits = words = 0.0
        for i, s in enumerate(heldout, 1):
            progress(i, 5_000, "sentences scored")
            ids, _ = lm.encode(s, already_clean=True)
            if not ids:
                continue
            bits += lm.surprisal(s, already_clean=True)
            words += len(ids) + 1                 # +1 for </s>
        ppl = 2.0 ** (bits / words) if words else float("inf")
        oov_wiki = lm.oov_rate(heldout)

        if not math.isfinite(ppl):
            raise RuntimeError("perplexity is infinite - a backoff weight collapsed")

        # ---- OOV on the text we will actually score --------------------------
        tb_sents = []
        with open(CONFIG["treebank"], "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("# text = "):
                    tb_sents.append(line[9:].strip())
                if len(tb_sents) >= 2000:
                    break
        oov_tb = lm.oov_rate(tb_sents)

        log("")
        log(f"held-out perplexity      {ppl:10.2f}")
        log(f"OOV rate (Wikipedia)     {oov_wiki:10.2%}")
        log(f"OOV rate (treebank)      {oov_tb:10.2%}   <- the text we score")
        log(f"vocabulary               {lm.V:10,}")

        report = {"perplexity": ppl, "oov_wikipedia": oov_wiki,
                  "oov_treebank": oov_tb, "vocab_size": lm.V,
                  "sum_to_one_max_error": worst,
                  "n_trigram": lm.meta["n_trigram"],
                  "n_bigram": lm.meta["n_bigram"],
                  "cleaning": CONFIG["cleaning"]}
        CONFIG["out"].mkdir(parents=True, exist_ok=True)
        write_json(CONFIG["out"] / "lm_report.json", report)
        return report


# ==============================================================================
#  MAIN
# ==============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="start", type=int, default=2,
                    help="re-run starting at this stage (2-6, default 2)")
    ap.add_argument("--force", action="store_true", help="ignore all caches")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny fast run for testing the plumbing (writes to *_smoke folders)")
    ap.add_argument("--sentences", type=Path, default=None,
                    help="override the input sentence file")
    ap.add_argument("--model-dir", type=Path, default=None,
                    help="override the output model folder")
    args = ap.parse_args()

    if args.sentences:
        CONFIG["sentences"] = args.sentences
    if args.model_dir:
        CONFIG["model"] = args.model_dir
    if args.smoke:
        # Separate folders so a smoke run can never leave a toy model where the
        # scorer (or a later real run's cache) would pick it up.
        smoke_root = ROOT / "training" / "work" / "trigram_smoke"
        CONFIG["work"] = smoke_root
        CONFIG["out"] = smoke_root
        CONFIG["model"] = smoke_root / "model"
        CONFIG["spill_threshold"] = 200_000     # exercise the disk-spill path
        print("*** SMOKE MODE: tiny corpus, tiny vocabulary, trigram_smoke/ folder ***")

    for d in ("work", "model", "out"):
        CONFIG[d].mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    f, s = args.force, args.start
    stage2_build_corpus(force=(f or s <= 2) and s <= 2, smoke=args.smoke)
    vocab = stage3_build_vocab(force=(f or s <= 3) and s <= 3, smoke=args.smoke)
    counts = stage4_count_ngrams(vocab, force=(f or s <= 4) and s <= 4)
    lm = stage5_build_model(vocab, counts, force=(f or s <= 5) and s <= 5)
    del counts
    report = stage6_test_lm(lm)

    print()
    print("=" * 78)
    print(f" TRIGRAM MODEL BUILT in {time.time() - t0:.1f}s -> {CONFIG['model']}")
    print(f" held-out perplexity {report['perplexity']:.2f}")
    print("=" * 78)


if __name__ == "__main__":
    main()
