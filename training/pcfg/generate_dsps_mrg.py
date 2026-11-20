#!/usr/bin/env python3
"""
generate_dsps_mrg.py — convert the HDTB dependency treebank into phrase-structure
trees for training the ``berkeley_pcfg`` grammar.

    python training/pcfg/generate_dsps_mrg.py

Input  : data/hi_hdtb-ud-train.conllu          (13,306 sentences)
Output : training/work/hdtb_train_dsps.mrg      (13,282 trees, one per line,
                                                 PTB bracket format)

The DS-PS (dependency structure -> phrase structure) conversion runs the
bundled tool in training/pcfg/ds_ps_ud/ in four stages:
  1. preprocess  — inject ChunkId from MISC into FEATS, the DS-PS input format
  2. CoNLL2SSF   — group tokens into chunks, build the inter-chunk dependency graph -> SSF
  3. SSF2PS      — apply phrase-structure rules -> bracketed trees
  4. TC          — fix brackets, remove null/unary nodes -> .mrg
It produces trees with explicit VP nodes and argument-labelled NPs (NP-SUBJ,
NP-OBJ-1, NP-OBJ-2). Trees that crash the Berkeley parser are then dropped
(sanitize_trees), which is why 24 sentences are lost.

Next: training/pcfg/train_grammar.sh
"""

import os
import re
import sys
import shutil
import subprocess
import argparse
import tempfile

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT_DIR     = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HDTB_CONLLU  = os.path.join(ROOT_DIR, 'data', 'hi_hdtb-ud-train.conllu')
OUTPUT_MRG   = os.path.join(ROOT_DIR, 'training', 'work', 'hdtb_train_dsps.mrg')
DS_PS_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ds_ps_ud')


# ── Stage 1: Preprocess HDTB CoNLL-U → DS-PS input format ────────────────────

def preprocess_hdtb(conllu_path, only_sentences=None):
    """
    Convert HDTB CoNLL-U to DS-PS tool input format.

    Changes made:
      - Strip comment lines (# sent_id = ..., # text = ..., etc.)
      - Skip multi-word tokens (ID contains '-') and empty nodes (ID contains '.')
      - Move ChunkId from MISC (col 9, 'ChunkId=NP2') into FEATS (col 5)
        as 'chunkId-NP2', which is the format conll_api.py expects.

    Args:
        conllu_path:   path to HDTB CoNLL-U file
        only_sentences: set of reference strings to limit conversion (or None for all)

    Returns:
        list of preprocessed sentence strings (blank-line-separated token blocks)
    """
    sentences = []
    current   = []

    with open(conllu_path, encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')

            if line.startswith('#'):
                continue           # strip comment lines

            if not line.strip():   # sentence boundary
                if current:
                    sentences.append('\n'.join(current))
                    current = []
                continue

            parts = line.split('\t')
            if len(parts) < 10:
                continue

            tok_id = parts[0]
            if '-' in tok_id or '.' in tok_id:
                continue           # skip multi-word tokens and empty nodes

            # Extract ChunkId from MISC (col 9): 'ChunkId=NP2|ChunkType=head|...'
            misc     = parts[9]
            chunk_id = 'NONE'
            for kv in misc.split('|'):
                if kv.startswith('ChunkId='):
                    chunk_id = kv.split('=', 1)[1]
                    break

            # Inject chunkId into FEATS (col 5) as 'chunkId-NP2'
            feats = parts[5]
            if feats == '_':
                feats = f'chunkId-{chunk_id}'
            else:
                feats = f'{feats}|chunkId-{chunk_id}'

            parts[5] = feats
            current.append('\t'.join(parts))

    if current:
        sentences.append('\n'.join(current))

    return sentences


# ── Stage runner ──────────────────────────────────────────────────────────────

def run_python_stage(script_name, cwd, stage_label):
    """
    Run a DS-PS pipeline stage by calling its Python script as a subprocess.
    Raises RuntimeError if the stage fails.
    """
    print(f"  [{stage_label}] running {script_name} in {cwd} ...")
    result = subprocess.run(
        [sys.executable, script_name],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
    )
    if result.stdout.strip():
        for ln in result.stdout.strip().splitlines():
            print(f"    {ln}")
    if result.returncode != 0:
        print(f"  ERROR: {stage_label} exited {result.returncode}")
        if result.stderr.strip():
            print(f"  STDERR:\n{result.stderr[:2000]}")
        raise RuntimeError(f"{stage_label} failed")
    print(f"  [{stage_label}] done")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(sentences, output_mrg, work_dir):
    """
    Run the full CoNLL→SSF→PS→TC pipeline in work_dir.

    Directory layout inside work_dir:
        CoNLL2SSF/
            data/DS/       ← input (our preprocessed CoNLL)
            data/SSF/      ← CoNLL2SSF output
        SSF2PS/
            data/SSF/      ← copied from CoNLL2SSF output
            data/PS/       ← SSF2PS output
        TC/
            data/PS/       ← copied from SSF2PS output
            data/PS-Braces/
            data/TC/       ← final tree output
    """

    # ── set up CoNLL2SSF ──────────────────────────────────────────────────────
    conll2ssf_dir = os.path.join(work_dir, 'CoNLL2SSF')
    ds_dir        = os.path.join(conll2ssf_dir, 'data', 'DS')
    ssf_dir_out   = os.path.join(conll2ssf_dir, 'data', 'SSF')
    os.makedirs(ds_dir,      exist_ok=True)
    os.makedirs(ssf_dir_out, exist_ok=True)

    # Copy our patched CoNLL2SSF scripts
    for fname in ('conll_api.py', 'ssf_api.py', 'main.py'):
        shutil.copy(
            os.path.join(DS_PS_DIR, 'CoNLL2SSF', fname),
            os.path.join(conll2ssf_dir, fname),
        )

    # Write preprocessed sentences to one input file
    input_file = os.path.join(ds_dir, 'hdtb_train.txt')
    with open(input_file, 'w', encoding='utf-8') as f:
        for sent in sentences:
            f.write(sent + '\n\n')
    print(f"  Written {len(sentences)} sentences to CoNLL2SSF/data/DS/hdtb_train.txt")

    # ── Stage 1: CoNLL → SSF ─────────────────────────────────────────────────
    run_python_stage('main.py', conll2ssf_dir, 'CoNLL2SSF')

    ssf_files = os.listdir(ssf_dir_out)
    print(f"  CoNLL2SSF produced {len(ssf_files)} SSF file(s): {ssf_files[:3]}")

    # ── set up SSF2PS ─────────────────────────────────────────────────────────
    ssf2ps_dir     = os.path.join(work_dir, 'SSF2PS')
    ssf_dir_in     = os.path.join(ssf2ps_dir, 'data', 'SSF')
    ps_dir_out     = os.path.join(ssf2ps_dir, 'data', 'PS')
    os.makedirs(ssf_dir_in, exist_ok=True)
    os.makedirs(ps_dir_out, exist_ok=True)

    # Copy SSF output from CoNLL2SSF stage
    for fname in ssf_files:
        shutil.copy(
            os.path.join(ssf_dir_out, fname),
            os.path.join(ssf_dir_in, fname),
        )

    # Copy our patched SSF2PS scripts
    for fname in ('ssf_api.py', 'Word_order.py', 'Phrases.py',
                  'Clause_joining.py', 'Rule_Selection.py', 'conversion.py'):
        shutil.copy(
            os.path.join(DS_PS_DIR, 'SSF2PS', fname),
            os.path.join(ssf2ps_dir, fname),
        )

    # ── Stage 2: SSF → PS ────────────────────────────────────────────────────
    run_python_stage('conversion.py', ssf2ps_dir, 'SSF2PS')

    ps_files = os.listdir(ps_dir_out)
    print(f"  SSF2PS produced {len(ps_files)} PS file(s): {ps_files[:3]}")

    # ── set up TC ─────────────────────────────────────────────────────────────
    tc_dir       = os.path.join(work_dir, 'TC')
    ps_dir_in    = os.path.join(tc_dir, 'data', 'PS')
    braces_dir   = os.path.join(tc_dir, 'data', 'PS-Braces')
    tc_dir_out   = os.path.join(tc_dir, 'data', 'TC')
    os.makedirs(ps_dir_in,  exist_ok=True)
    os.makedirs(braces_dir, exist_ok=True)
    os.makedirs(tc_dir_out, exist_ok=True)

    # Copy PS output from SSF2PS stage
    for fname in ps_files:
        shutil.copy(
            os.path.join(ps_dir_out, fname),
            os.path.join(ps_dir_in, fname),
        )

    # Copy our patched TC script
    shutil.copy(
        os.path.join(DS_PS_DIR, 'TC', 'tree_from_ssf.py'),
        os.path.join(tc_dir, 'tree_from_ssf.py'),
    )

    # ── Stage 3: TC (tree fixing) ────────────────────────────────────────────
    run_python_stage('tree_from_ssf.py', tc_dir, 'TC')

    # ── Collect output trees ─────────────────────────────────────────────────
    trees = []
    for fname in sorted(os.listdir(tc_dir_out)):
        fpath = os.path.join(tc_dir_out, fname)
        with open(fpath, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    trees.append(line)

    print(f"  Collected {len(trees)} trees from TC output")
    return trees


# ── Verification ──────────────────────────────────────────────────────────────

def sanitize_trees(trees):
    """
    Drop trees that would crash Berkeley Parser's SophisticatedLexicon.

    Berkeley Parser crashes with ArrayIndexOutOfBoundsError whenever it
    encounters a leaf that has no preterminal parent.  Two DS-PS bugs produce
    such leaves:

    1. Unresolved 'flag N' position markers (Clause_joining.py): bare integer
       tokens appear as direct children of a phrase node with no POS wrapper.
       Regex: ) DIGITS ( or ) DIGITS )

    2. Multi-leaf preterminals (bracket-protection artefact): the replace_allbraces
       step turns literal ( / ) into LB / RB, sometimes leaving several raw words
       as sibling string children of one SYM or other tag node.  Berkeley Parser's
       binarizer creates ghost nodes for the extra leaves; the ghost nodes have no
       preterminal, causing the crash.

    Both are detected structurally via NLTK:
      - mixed children: a phrase node has both Tree children and bare-string children
      - multi-leaf preterminal: a node has more than one bare-string child
    """
    from nltk.tree import Tree as NLTKTree

    def _is_bad(tree_str):
        try:
            t = NLTKTree.fromstring(tree_str)
        except Exception:
            return True
        for sub in t.subtrees():
            kids = list(sub)
            str_kids  = [k for k in kids if isinstance(k, str)]
            tree_kids = [k for k in kids if isinstance(k, NLTKTree)]
            if str_kids and tree_kids:   # mixed: bare string(s) alongside sub-trees
                return True
            if len(str_kids) > 1:        # more than one leaf under one POS tag
                return True
        return False

    good, dropped = [], 0
    for t in trees:
        if _is_bad(t):
            dropped += 1
        else:
            good.append(t)
    if dropped:
        print(f"  Dropped {dropped} trees with invalid PTB structure (bare numeric or multi-leaf nodes)")
    return good


def verify_trees(trees):
    """Basic sanity checks on the collected trees."""
    issues = []
    n_s_root  = sum(1 for t in trees if t.startswith('(S ') or t.startswith('(S('))
    n_balanced = 0
    for t in trees:
        depth = 0
        ok = True
        for ch in t:
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
            if depth < 0:
                ok = False
                break
        if ok and depth == 0:
            n_balanced += 1

    print(f"  Trees with (S root  : {n_s_root}/{len(trees)} ({100*n_s_root/max(1,len(trees)):.1f}%)")
    print(f"  Balanced parens     : {n_balanced}/{len(trees)} ({100*n_balanced/max(1,len(trees)):.1f}%)")

    # Sample trees
    import random
    rng = random.Random(42)
    sample = rng.sample(trees, min(3, len(trees)))
    print("\n  Sample trees:")
    for t in sample:
        print(f"    {t[:120]}{'...' if len(t) > 120 else ''}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    global HDTB_CONLLU, OUTPUT_MRG
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', default=HDTB_CONLLU, help='CoNLL-U treebank')
    parser.add_argument('--out', default=OUTPUT_MRG, help='output .mrg file')
    parser.add_argument('--keep-work', action='store_true',
                        help='keep the temporary DS-PS work folder for debugging')
    args = parser.parse_args()
    HDTB_CONLLU, OUTPUT_MRG = args.input, args.out

    print("=" * 60)
    print("DS-PS .mrg generation pipeline")
    print(f"  Input  : {HDTB_CONLLU}")
    print(f"  Output : {OUTPUT_MRG}")
    print("=" * 60)

    print("\nStep 1: Preprocessing HDTB CoNLL-U ...")
    sentences = preprocess_hdtb(HDTB_CONLLU, None)
    print(f"  Preprocessed {len(sentences)} sentences")

    work_dir = tempfile.mkdtemp(prefix='dsps_mrg_')
    print(f"\nStep 2-4: Running DS-PS pipeline in {work_dir}")
    trees = run_pipeline(sentences, OUTPUT_MRG, work_dir)
    if not trees:
        print(f"\nERROR: No trees collected. Work folder kept at {work_dir}")
        sys.exit(1)

    trees = sanitize_trees(trees)

    os.makedirs(os.path.dirname(OUTPUT_MRG), exist_ok=True)
    with open(OUTPUT_MRG, 'w', encoding='utf-8', newline='\n') as f:  # same bytes on every OS
        for tree in trees:
            f.write(tree + '\n')

    print("\nStep 5: Verification")
    verify_trees(trees)

    if args.keep_work:
        print(f"Work folder kept at {work_dir}")
    else:
        shutil.rmtree(work_dir, ignore_errors=True)

    size_kb = os.path.getsize(OUTPUT_MRG) / 1024
    print(f"\n{'=' * 60}")
    print(f"Done. Written {len(trees)} trees to {OUTPUT_MRG} ({size_kb:.0f} KB)")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
