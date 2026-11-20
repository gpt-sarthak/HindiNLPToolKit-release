"""
Integration test suite for the HindiNLPToolKit library and web app.

Script-style (not pytest). Run from anywhere:

    python tests/test_all.py

Needs the bundled corpora in data/ and the trained models in scoring/models/
(see training/README.md). Writes its outputs to tests/output/ (gitignored).
"""

import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from stanza_parser import load_input
from filtering import (
    filter_sentences, summarize,
    filter_questions, filter_negatives, filter_ghost_ids,
    filter_non_projective, filter_bad_root, filter_punct_constituents,
    filter_min_phrases,
)
from variants import generate_variants

TEST_DIR = REPO / "tests" / "output"
TEST_DIR.mkdir(exist_ok=True)

CONLLU_FILE  = str(REPO / "data" / "fact_media_news_ISCNLP.conllu")
TXT_FILE     = str(REPO / "tests" / "data" / "test_sentences.txt")
RAW_SENTENCE = "राम ने बाज़ार से ताज़ी सब्जियाँ और फल ख़रीदे।"

PASS = "✓ PASS"
FAIL = "✗ FAIL"

results = []

def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    msg = f"  [{status}] {name}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    results.append({"test": name, "status": "PASS" if condition else "FAIL", "detail": detail})

def section(title):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print("=" * 65)


# ═══════════════════════════════════════════════════════════════════
# 1. load_input
# ═══════════════════════════════════════════════════════════════════
section("1. load_input — CoNLL-U file")

try:
    conllu_sents = load_input(CONLLU_FILE)
    check("Returns a non-empty list",        len(conllu_sents) > 0)
    check("Returns TokenList objects",       hasattr(conllu_sents[0], "metadata"))
    check("Correct sentence count (17405)",  len(conllu_sents) == 17405,
          f"got {len(conllu_sents)}")
    check("Sentences have sent_id metadata", conllu_sents[0].metadata.get("sent_id") is not None)
    check("Tokens have head/deprel fields",
          all(k in conllu_sents[0][0] for k in ("id", "form", "upos", "head", "deprel")))
    print(f"    Sample: {conllu_sents[0].metadata.get('sent_id')} | "
          f"{conllu_sents[0].metadata.get('text', '')[:60]}")
except Exception as e:
    check("load_input(.conllu) raised no exception", False, str(e))
    traceback.print_exc()

section("1b. load_input — TXT file")

try:
    txt_sents = load_input(TXT_FILE)
    check("Returns a non-empty list",       len(txt_sents) > 0)
    check("Returns TokenList objects",      hasattr(txt_sents[0], "metadata"))
    check("Each sentence has sent_id",      txt_sents[0].metadata.get("sent_id") is not None)
    check("Tokens have deprel fields",
          all("deprel" in tok for tok in txt_sents[0]))
    print(f"    Parsed {len(txt_sents)} sentence(s) from {TXT_FILE}")
    for s in txt_sents:
        print(f"      {s.metadata.get('sent_id')} | {s.metadata.get('text','')[:60]}")
except Exception as e:
    check("load_input(.txt) raised no exception", False, str(e))
    traceback.print_exc()
    txt_sents = []

section("1c. load_input — raw Hindi string")

try:
    raw_sents = load_input(RAW_SENTENCE)
    check("Returns a non-empty list",      len(raw_sents) > 0)
    check("Returns TokenList objects",     hasattr(raw_sents[0], "metadata"))
    check("Has at least 1 token",          len(raw_sents[0]) > 0)
    print(f"    Text : {RAW_SENTENCE}")
    print(f"    Tokens: {[tok['form'] for tok in raw_sents[0]]}")
    print(f"    Deprels: {[tok['deprel'] for tok in raw_sents[0]]}")
except Exception as e:
    check("load_input(raw string) raised no exception", False, str(e))
    traceback.print_exc()
    raw_sents = []

section("1d. load_input — error handling")

try:
    load_input("nonexistent_file.conllu")
    check("FileNotFoundError for missing .conllu", False, "no exception raised")
except FileNotFoundError:
    check("FileNotFoundError for missing .conllu", True)
except Exception as e:
    check("FileNotFoundError for missing .conllu", False, type(e).__name__)

try:
    load_input("nonexistent_file.txt")
    check("FileNotFoundError for missing .txt", False, "no exception raised")
except FileNotFoundError:
    check("FileNotFoundError for missing .txt", True)
except Exception as e:
    check("FileNotFoundError for missing .txt", False, type(e).__name__)

try:
    load_input("   ")
    check("ValueError for blank string", False, "no exception raised")
except ValueError:
    check("ValueError for blank string", True)
except Exception as e:
    check("ValueError for blank string", False, type(e).__name__)


# ═══════════════════════════════════════════════════════════════════
# 2. filter_sentences — CoNLL-U corpus
# ═══════════════════════════════════════════════════════════════════
section("2. filter_sentences — CoNLL-U corpus (17405 sentences)")

try:
    passed, rejected_df, passed_df = filter_sentences(
        conllu_sents, output_dir=str(TEST_DIR)
    )

    total = len(passed) + len(rejected_df)
    check("Returns 3-tuple",                       True)
    check("passed_list is a list of dicts",        isinstance(passed, list) and isinstance(passed[0], dict))
    check("Every passed item has required keys",
          all({"sentence","root_id","constituents"} <= set(item.keys()) for item in passed))
    check("rejected_df has correct columns",
          list(rejected_df.columns) == ["Sent_ID","Ref_Index","Sentence","Reason"])
    check("passed_df has correct columns",
          {"Sent_ID","Sentence","Root_ID","Constituent_Count","Sentence_Length"}.issubset(set(passed_df.columns)))
    check("passed + rejected == total input",      total == 17405, f"{len(passed)} + {len(rejected_df)} = {total}")
    check("At least some sentences passed",        len(passed) > 0)
    check("At least some sentences rejected",      len(rejected_df) > 0)
    check("All passed items have >= 2 preverbal constituents",
          all(len(item["constituents"]) >= 2 for item in passed))
    check("passed_sentences.csv saved",            (TEST_DIR / "passed_sentences.csv").exists())
    check("rejected_sentences.csv saved",          (TEST_DIR / "rejected_sentences.csv").exists())
    check("passed_df row count matches passed_list", len(passed_df) == len(passed))
    check("rejected_df row count matches rejected rows", len(rejected_df) == total - len(passed))

    rejection_reasons = rejected_df["Reason"].value_counts().to_dict()
    print(f"\n    Filter breakdown:")
    print(f"      Total input   : {total}")
    print(f"      Passed        : {len(passed)}")
    print(f"      Rejected      : {len(rejected_df)}")
    for reason, count in sorted(rejection_reasons.items(), key=lambda x: -x[1]):
        print(f"        {reason:<45}: {count}")

    print(f"\n    passed_df sample:")
    print(passed_df[["Sent_ID","Sentence","Constituent_Count","Sentence_Length"]].head(3).to_string(index=False))

except Exception as e:
    check("filter_sentences(conllu_sents) raised no exception", False, str(e))
    traceback.print_exc()
    passed, rejected_df, passed_df = [], pd.DataFrame(), pd.DataFrame()

section("2b. filter_sentences — TXT / raw sentences")

for label, sents in [("TXT sentences", txt_sents), ("Raw string sentence", raw_sents)]:
    if not sents:
        print(f"  Skipped {label} (load_input failed)")
        continue
    try:
        p, r_df, p_df = filter_sentences(sents)
        check(f"{label}: returns 3-tuple without error", True)
        check(f"{label}: rejected_df has Reason column", "Reason" in r_df.columns)
        print(f"    {label}: {len(p)} passed, {len(r_df)} rejected")
        if not r_df.empty:
            print(f"    Rejection reasons: {r_df['Reason'].value_counts().to_dict()}")
    except Exception as e:
        check(f"{label}: no exception", False, str(e))
        traceback.print_exc()

section("2c. filter_sentences — error handling")

try:
    filter_sentences([])
    check("ValueError for empty list", False, "no exception raised")
except ValueError:
    check("ValueError for empty list", True)
except Exception as e:
    check("ValueError for empty list", False, type(e).__name__)


# ═══════════════════════════════════════════════════════════════════
# 2d. Individual filter functions
# ═══════════════════════════════════════════════════════════════════
section("2d. filter_questions — individual")

try:
    q_pass, q_rej = filter_questions(conllu_sents[:1000])
    check("Returns (list, DataFrame)",
          isinstance(q_pass, list) and isinstance(q_rej, pd.DataFrame))
    check("passed + rejected == input",
          len(q_pass) + len(q_rej) == 1000)
    check("rejected Reason is 'Interrogative'",
          q_rej.empty or (q_rej["Reason"] == "Interrogative").all())
    # All signals disabled — nothing is filtered
    q_all, _ = filter_questions(conllu_sents[:1000],
                                check_punct=False, check_pos=False, check_features=False)
    check("All signals off passes all sentences", len(q_all) == 1000)
    print(f"    1000 sentences → {len(q_pass)} passed, {len(q_rej)} rejected as questions")
except Exception as e:
    check("filter_questions raised no exception", False, str(e))
    traceback.print_exc()

section("2d. filter_negatives — individual")

try:
    n_pass, n_rej = filter_negatives(conllu_sents[:1000])
    check("Returns (list, DataFrame)",
          isinstance(n_pass, list) and isinstance(n_rej, pd.DataFrame))
    check("passed + rejected == input",
          len(n_pass) + len(n_rej) == 1000)
    check("rejected Reason is 'Negation'",
          n_rej.empty or (n_rej["Reason"] == "Negation").all())
    n_all, _ = filter_negatives(conllu_sents[:1000], check_pos=False, check_features=False)
    check("All signals off passes all sentences", len(n_all) == 1000)
    print(f"    1000 sentences → {len(n_pass)} passed, {len(n_rej)} rejected as negatives")
except Exception as e:
    check("filter_negatives raised no exception", False, str(e))
    traceback.print_exc()

section("2d. filter_ghost_ids — individual")

try:
    g_pass, g_rej = filter_ghost_ids(conllu_sents[:1000])
    check("Returns (list, DataFrame)",
          isinstance(g_pass, list) and isinstance(g_rej, pd.DataFrame))
    check("passed + rejected == input",
          len(g_pass) + len(g_rej) == 1000)
    check("rejected Reason is 'Empty nodes'",
          g_rej.empty or (g_rej["Reason"] == "Empty nodes").all())
    print(f"    1000 sentences → {len(g_pass)} passed, {len(g_rej)} have ghost IDs")
except Exception as e:
    check("filter_ghost_ids raised no exception", False, str(e))
    traceback.print_exc()

section("2d. filter_non_projective — individual")

try:
    np_pass, np_rej = filter_non_projective(conllu_sents[:1000])
    check("Returns (list, DataFrame)",
          isinstance(np_pass, list) and isinstance(np_rej, pd.DataFrame))
    check("passed + rejected == input",
          len(np_pass) + len(np_rej) == 1000)
    check("rejected Reason is 'Non-projective dependencies'",
          np_rej.empty or (np_rej["Reason"] == "Non-projective dependencies").all())
    print(f"    1000 sentences → {len(np_pass)} passed, {len(np_rej)} non-projective")
except Exception as e:
    check("filter_non_projective raised no exception", False, str(e))
    traceback.print_exc()

section("2d. filter_bad_root — individual")

br_pass_1000 = []
try:
    br_pass_1000, br_rej = filter_bad_root(conllu_sents[:1000])
    check("Returns (list of dicts, DataFrame)",
          isinstance(br_pass_1000, list) and isinstance(br_rej, pd.DataFrame))
    check("Passed items have required keys",
          br_pass_1000 == [] or all(
              {"sentence", "root_id", "constituents"} <= set(i.keys())
              for i in br_pass_1000))
    check("passed + rejected == input",
          len(br_pass_1000) + len(br_rej) == 1000)
    check("rejected Reason starts with 'Non-verbal root'",
          br_rej.empty or br_rej["Reason"].str.startswith("Non-verbal root").all())
    # Narrower allowed_pos → same or fewer passing
    br_verb_only, _ = filter_bad_root(conllu_sents[:1000], allowed_pos=["VERB"])
    check("Custom allowed_pos=['VERB'] more restrictive",
          len(br_verb_only) <= len(br_pass_1000))
    print(f"    1000 sentences → {len(br_pass_1000)} passed, {len(br_rej)} bad root")
    print(f"    VERB-only → {len(br_verb_only)} passed")
except Exception as e:
    check("filter_bad_root raised no exception", False, str(e))
    traceback.print_exc()

section("2d. filter_punct_constituents — individual")

try:
    # Run on output of filter_bad_root (Dict input)
    pc_pass, pc_rej = filter_punct_constituents(br_pass_1000)
    check("Returns (list of dicts, DataFrame)",
          isinstance(pc_pass, list) and isinstance(pc_rej, pd.DataFrame))
    check("Passed items have required keys",
          pc_pass == [] or all(
              {"sentence", "root_id", "constituents"} <= set(i.keys())
              for i in pc_pass))
    check("passed + rejected == input",
          len(pc_pass) + len(pc_rej) == len(br_pass_1000))
    check("rejected Reason is 'Punctuation-attached constituent'",
          pc_rej.empty or
          (pc_rej["Reason"] == "Punctuation-attached constituent").all())
    # No passed sentence may have a punct-attached constituent
    def _has_punct_const(item):
        root_id = item["root_id"]
        return any(
            tok["deprel"] == "punct"
            for const in item["constituents"]
            for tok in const
            if tok["head"] == root_id
        )
    check("No passed item has a punct-attached constituent",
          not any(_has_punct_const(i) for i in pc_pass))
    # TokenList input also accepted
    pc_pass_tl, pc_rej_tl = filter_punct_constituents(conllu_sents[:1000])
    check("TokenList input accepted",
          isinstance(pc_pass_tl, list) and isinstance(pc_rej_tl, pd.DataFrame))
    check("TokenList: passed + rejected == input",
          len(pc_pass_tl) + len(pc_rej_tl) == 1000)
    print(f"    Dict input  : {len(pc_pass)} passed, {len(pc_rej)} rejected")
    print(f"    Token input : {len(pc_pass_tl)} passed, {len(pc_rej_tl)} rejected")
except Exception as e:
    check("filter_punct_constituents raised no exception", False, str(e))
    traceback.print_exc()

section("2d. filter_min_phrases — individual (TokenList input)")

mp_from_tokens = []
try:
    mp_from_tokens, mp_rej_t = filter_min_phrases(conllu_sents[:1000])
    check("Returns (list of dicts, DataFrame)",
          isinstance(mp_from_tokens, list) and isinstance(mp_rej_t, pd.DataFrame))
    check("Passed items have required keys",
          mp_from_tokens == [] or all(
              {"sentence", "root_id", "constituents"} <= set(i.keys())
              for i in mp_from_tokens))
    check("All passed items have >= 2 preverbal constituents",
          all(len(i["constituents"]) >= 2 for i in mp_from_tokens))
    check("passed + rejected == input",
          len(mp_from_tokens) + len(mp_rej_t) == 1000)
    print(f"    TokenList input: {len(mp_from_tokens)} passed, {len(mp_rej_t)} rejected")
except Exception as e:
    check("filter_min_phrases (TokenList) raised no exception", False, str(e))
    traceback.print_exc()

section("2d. filter_min_phrases — individual (Dict input from filter_bad_root)")

try:
    if br_pass_1000:
        mp_from_dicts, mp_rej_d = filter_min_phrases(br_pass_1000)
        check("Returns (list of dicts, DataFrame)",
              isinstance(mp_from_dicts, list) and isinstance(mp_rej_d, pd.DataFrame))
        check("passed + rejected == Dict input length",
              len(mp_from_dicts) + len(mp_rej_d) == len(br_pass_1000))
        check("All passed items have >= 2 preverbal constituents",
              all(len(i["constituents"]) >= 2 for i in mp_from_dicts))
        # min_phrases=3 is strictly more restrictive
        mp3, _ = filter_min_phrases(br_pass_1000, min_phrases=3)
        check("min_phrases=3 passes subset of min_phrases=2",
              len(mp3) <= len(mp_from_dicts))
        check("All min_phrases=3 items have >= 3 phrases",
              all(len(i["constituents"]) >= 3 for i in mp3))
        print(f"    Dict input: min=2 → {len(mp_from_dicts)} passed; min=3 → {len(mp3)} passed")
    else:
        print("  Skipped (no br_pass_1000 items)")
except Exception as e:
    check("filter_min_phrases (Dict) raised no exception", False, str(e))
    traceback.print_exc()

section("2d-core. filter_min_phrases — preverbal subject + object requirement")

# Criterion (a) of Ranjan & van Schijndel (2024): trees must contain both
# well-defined subjects and objects.  Enforced by filter_min_phrases alongside
# the constituent count, so a subject + three obliques no longer passes.
SUBJ = {"nsubj", "nsubj:pass", "k1"}
OBJ = {"obj", "iobj", "k2", "k4"}

def _const_deprels(item):
    """Deprels of the constituents' attachments to the root."""
    out = []
    for c in item["constituents"]:
        out.append(next((t["deprel"] for t in c if t["head"] == item["root_id"]), "UNKNOWN"))
    return out

try:
    if br_pass_1000:
        core_on, core_rej = filter_min_phrases(br_pass_1000)
        core_off, _ = filter_min_phrases(br_pass_1000, require_core_args=False)

        check("require_core_args=True passes a subset of False",
              len(core_on) <= len(core_off))
        check("Every passed item has a preverbal subject",
              all(SUBJ & set(_const_deprels(i)) for i in core_on))
        check("Every passed item has a preverbal object",
              all(OBJ & set(_const_deprels(i)) for i in core_on))
        check("Rejections carry a 'Missing preverbal ...' reason",
              core_rej.empty or any(
                  str(r).startswith("Missing preverbal") for r in core_rej["Reason"]))
        # Narrowing the object set can only shrink the passed set.
        direct_only, _ = filter_min_phrases(br_pass_1000, object_deprels=["obj", "k2"])
        check("object_deprels=['obj','k2'] passes a subset of the default",
              len(direct_only) <= len(core_on))
        # A sentence whose only non-subject constituents are obliques must fail.
        no_obj = [i for i in core_off if not (OBJ & set(_const_deprels(i)))]
        check("Objectless items are all rejected by the default",
              all(i not in core_on for i in no_obj))
        print(f"    core_args on → {len(core_on)} passed; off → {len(core_off)}; "
              f"objectless dropped → {len(no_obj)}")
    else:
        print("  Skipped (no br_pass_1000 items)")
except Exception as e:
    check("filter_min_phrases (core args) raised no exception", False, str(e))
    traceback.print_exc()

section("2d. Individual filters — error handling")

for fn_name, fn in [
    ("filter_questions",          filter_questions),
    ("filter_negatives",          filter_negatives),
    ("filter_ghost_ids",          filter_ghost_ids),
    ("filter_non_projective",     filter_non_projective),
    ("filter_bad_root",           filter_bad_root),
    ("filter_punct_constituents", filter_punct_constituents),
    ("filter_min_phrases",        filter_min_phrases),
]:
    try:
        fn([])
        check(f"{fn_name}: ValueError on empty list", False, "no exception raised")
    except ValueError:
        check(f"{fn_name}: ValueError on empty list", True)
    except Exception as e:
        check(f"{fn_name}: ValueError on empty list", False, type(e).__name__)

try:
    filter_min_phrases(conllu_sents[:5], min_phrases=0)
    check("filter_min_phrases: min_phrases=0 raises ValueError", False, "no exception raised")
except ValueError:
    check("filter_min_phrases: min_phrases=0 raises ValueError", True)
except Exception as e:
    check("filter_min_phrases: min_phrases=0 raises ValueError", False, type(e).__name__)


# ═══════════════════════════════════════════════════════════════════
# 3. generate_variants
# ═══════════════════════════════════════════════════════════════════
section("3. generate_variants — full corpus passed sentences")

VARIANT_COLUMNS = ["Sent_ID", "Variant_ID", "Ref_Index", "Variant_No", "ML_Label",
                   "Reference_Sentence", "Variant_Sentence"]

if passed:
    try:
        pairs_df = generate_variants(passed, output_dir=str(TEST_DIR))

        check("generate_variants returns a DataFrame", isinstance(pairs_df, pd.DataFrame))
        check("pairs_df has surface-only column order",
              list(pairs_df.columns) == VARIANT_COLUMNS, f"cols={list(pairs_df.columns)}")
        check("No feature columns leak from the generator",
              not ({"DL_Reference","DL_Variant","IS_Reference","IS_Variant"} & set(pairs_df.columns)))
        labels = set(pairs_df["ML_Label"].tolist())
        check("ML_Label values are only 0 and 1", labels.issubset({0, 1}))
        check("ML_Label roughly balanced (the flip signal)",
              0.35 < pairs_df["ML_Label"].mean() < 0.65, f"mean={pairs_df['ML_Label'].mean():.3f}")
        check("Variant_ID is {Sent_ID}_v{n}",
              all(vid.startswith(f"{sid}_v")
                  for sid, vid in zip(pairs_df["Sent_ID"], pairs_df["Variant_ID"])))
        check("reference_variant_pairs.csv saved", (TEST_DIR / "reference_variant_pairs.csv").exists())
        check("At least some variants generated", len(pairs_df) > 0)
        identical = (pairs_df["Reference_Sentence"] == pairs_df["Variant_Sentence"]).sum()
        identical_pct = identical / len(pairs_df) * 100
        check("Identical-text pairs < 0.1% (repeated-word constituents edge case)",
              identical_pct < 0.1,
              f"{identical} identical pairs ({identical_pct:.4f}%)")

        print(f"\n    Total pairs generated : {len(pairs_df)}")
        print(f"    Label distribution    : 0={int((pairs_df['ML_Label']==0).sum())}  1={int((pairs_df['ML_Label']==1).sum())}")
        print(f"\n    Sample pair:")
        row = pairs_df.iloc[0]
        print(f"      Ref : {row['Reference_Sentence'][:70]}")
        print(f"      Var : {row['Variant_Sentence'][:70]}")
        print(f"      Label: {row['ML_Label']}")

    except Exception as e:
        check("generate_variants raised no exception", False, str(e))
        traceback.print_exc()
        pairs_df = pd.DataFrame(columns=VARIANT_COLUMNS)
else:
    print("  Skipped (no passed sentences)")
    pairs_df = pd.DataFrame(columns=VARIANT_COLUMNS)

section("3b. generate_variants — custom valid_deprel_pairs")

if passed:
    try:
        custom_pairs = {("k1","k2"), ("k2","k1"), ("k1","vmod"), ("vmod","k1"),
                        ("k2","vmod"), ("vmod","k2"), ("k1","k1"), ("k2","k2")}
        df2 = generate_variants(passed[:100], valid_deprel_pairs=custom_pairs, max_variants=5)
        check("Custom valid_deprel_pairs accepted",  True)
        check("max_variants=5 respected",
              len(df2) <= 100 * 5, f"got {len(df2)} pairs for 100 sentences")
        print(f"    Generated {len(df2)} pairs from 100 sentences with max_variants=5")
    except Exception as e:
        check("generate_variants with custom pairs", False, str(e))
        traceback.print_exc()

section("3c. generate_variants — single parsed sentence (raw string)")

if raw_sents:
    try:
        p_raw, r_raw, _ = filter_sentences(raw_sents)
        if p_raw:
            dfr = generate_variants(p_raw)
            check("Variants generated for single sentence", True)
            print(f"    Sentence  : {RAW_SENTENCE}")
            print(f"    Passed filter: yes  |  Variants: {len(dfr)}")
            if not dfr.empty:
                print(f"    Variant   : {dfr.iloc[0]['Variant_Sentence']}")
        else:
            print(f"    Single sentence did not pass filters — skipping variant generation")
            r_reasons = r_raw["Reason"].tolist()
            print(f"    Reason(s): {r_reasons}")
            check("generate_variants handles no-pass case gracefully", True)
    except Exception as e:
        check("generate_variants for raw sentence", False, str(e))
        traceback.print_exc()

section("3d. generate_variants — empty input")

try:
    dfe = generate_variants([])
    check("Empty input returns empty df",      len(dfe) == 0)
    check("Empty df has correct columns",      list(dfe.columns) == VARIANT_COLUMNS)
except Exception as e:
    check("generate_variants([]) raised no exception", False, str(e))
    traceback.print_exc()

section("3e. scorers — dependency_length + information_status")

if passed and not pairs_df.empty:
    try:
        from scoring import apply_scorers, build_corpus_context, get_scorers

        avail = set(get_scorers())
        check("dependency_length scorer discovered", "dependency_length" in avail)
        check("information_status scorer discovered", "information_status" in avail)

        sample = pairs_df.head(3000).copy()
        ctx = {"corpus": build_corpus_context(conllu_sents),
               "passed": passed, "scheme": "paninian"}
        scored = apply_scorers(sample, ["dependency_length", "information_status"], context=ctx)

        check("Row count unchanged by scoring", len(scored) == len(sample))
        check("DL_Reference / DL_Variant added",
              {"DL_Reference","DL_Variant"}.issubset(scored.columns))
        check("DL_Reference is a scalar total_DL",
              isinstance(scored.iloc[0]["DL_Reference"], (int, np.integer)))
        check("Delta_DL added by central diff step", "Delta_DL" in scored.columns)
        check("IS_Reference / IS_Variant added",
              {"IS_Reference","IS_Variant"}.issubset(scored.columns))
        check("Delta_IS added by central diff step", "Delta_IS" in scored.columns)

        # Delta_DL must equal (first - second) total_DL per ML_Label on every row.
        ok = True
        for _, r in scored.iterrows():
            ref0, var0, lbl = r["DL_Reference"], r["DL_Variant"], int(r["ML_Label"])
            expected = (ref0 - var0) if lbl == 1 else (var0 - ref0)
            if int(r["Delta_DL"]) != int(expected):
                ok = False
                break
        check("Delta_DL oriented by ML_Label (first - second)", ok)

        print(f"    scored {len(scored)} rows | "
              f"nonzero Delta_DL={int((scored['Delta_DL']!=0).sum())} | "
              f"nonzero Delta_IS={int((scored['Delta_IS']!=0).sum())}")
    except Exception as e:
        check("scorers ran without exception", False, str(e))
        traceback.print_exc()
else:
    print("  Skipped (no variants to score)")

section("3f. evaluation — pairwise ranking accuracy")

if passed and not pairs_df.empty and "scored" in globals():
    try:
        from evaluation import evaluate_pairs

        ev = evaluate_pairs(scored)
        check("Returns a dict with expected keys",
              {"n_pairs", "folds", "seed", "predictors", "combined"} <= set(ev.keys()))
        check("n_pairs matches input rows", ev["n_pairs"] == len(scored))
        by_delta = {p["delta"]: p for p in ev["predictors"]}
        check("Delta_DL and Delta_IS both evaluated",
              {"Delta_DL", "Delta_IS"} <= set(by_delta))
        check("Accuracies are 0-1 fractions",
              all(0.0 <= p["accuracy"] <= 1.0 for p in ev["predictors"]))
        # ci_low/ci_high are gated by evaluation.ranking.INCLUDE_CI: absent when
        # off, and bracketing the accuracy when on.
        from evaluation.ranking import INCLUDE_CI

        check("CI keys match INCLUDE_CI (%s)" % INCLUDE_CI,
              all((p["ci_low"] <= p["accuracy"] <= p["ci_high"])
                  if INCLUDE_CI else
                  ("ci_low" not in p and "ci_high" not in p)
                  for p in ev["predictors"]))
        check("Predictors sorted by accuracy descending",
              all(a["accuracy"] >= b["accuracy"]
                  for a, b in zip(ev["predictors"], ev["predictors"][1:])))
        check("Direction text matches coefficient sign",
              all(("lower" in p["direction"]) == (p["coefficient"] < 0)
                  for p in ev["predictors"]))
        check("Combined model present with >= 2 deltas",
              ev["combined"] is not None
              and set(ev["combined"]["coefficients"]) == {"Delta_DL", "Delta_IS"})
        check("Delta_DL beats chance on corpus pairs",
              by_delta["Delta_DL"]["accuracy"] > 0.5,
              f"acc={by_delta['Delta_DL']['accuracy']}")

        ev2 = evaluate_pairs(scored)
        check("Deterministic for a fixed seed",
              [p["accuracy"] for p in ev2["predictors"]]
              == [p["accuracy"] for p in ev["predictors"]])

        ev_solo = evaluate_pairs(scored, deltas=["Delta_DL"])
        check("Single predictor -> combined is None", ev_solo["combined"] is None)

        try:
            evaluate_pairs(scored.head(5))
            check("Tiny input raises ValueError", False, "no exception raised")
        except ValueError:
            check("Tiny input raises ValueError", True)

        print(f"    evaluated {ev['n_pairs']} pairs | "
              + " | ".join(f"{p['delta']}={p['accuracy']:.3f}" for p in ev["predictors"])
              + f" | combined={ev['combined']['accuracy']:.3f}")
    except Exception as e:
        check("evaluate_pairs ran without exception", False, str(e))
        traceback.print_exc()
else:
    print("  Skipped (no scored pairs from 3e)")

section("3g. recommend_order — most natural word order")

if passed and not pairs_df.empty and "scored" in globals():
    try:
        from scoring import recommend_order

        rec = recommend_order(scored, ["dependency_length", "information_status"])

        by_delta = {f["delta"]: f for f in rec["features"]}
        check("Features declared for both scorers",
              {"Delta_DL", "Delta_IS"} <= set(by_delta))
        check("Delta_DL direction is lower-is-better",
              by_delta["Delta_DL"]["higher_is_more_natural"] is False)
        check("Delta_IS direction is higher-is-better",
              by_delta["Delta_IS"]["higher_is_more_natural"] is True)
        check("Feature weights are positive",
              all(f["weight"] > 0 for f in rec["features"]))
        check("One entry per scored sentence",
              len(rec["sentences"]) == scored["Sent_ID"].nunique())

        ok_shape = ok_ref_first = ok_winner = True
        for s in rec["sentences"]:
            n_rows = int((scored["Sent_ID"] == s["sent_id"]).sum())
            if len(s["candidates"]) != n_rows + 1:
                ok_shape = False
            first = s["candidates"][0]
            if not first["is_reference"] or first["id"] != "reference":
                ok_ref_first = False
            finite = [c["score"] for c in s["candidates"] if c["score"] is not None]
            if s["recommended"] is not None and finite \
                    and s["recommended"]["score"] != max(finite):
                ok_winner = False
        check("Candidates = variants + 1 (reference)", ok_shape)
        check("Reference is listed first", ok_ref_first)
        check("Recommended candidate has the max score", ok_winner)

        n_ref_wins = sum(1 for s in rec["sentences"]
                         if s["recommended"] and s["recommended"]["is_reference"])
        print(f"    {len(rec['sentences'])} sentences | reference recommended in "
              f"{n_ref_wins} ({n_ref_wins / len(rec['sentences']):.0%})")

        # Synthetic single-sentence tables: only the columns the DL scorer's
        # deltas() extractors read (DL_Reference / DL_Variant) are needed.
        def _mini_pairs(ref_dl, var_dls):
            return pd.DataFrame([{
                "Sent_ID": "synth_s1",
                "Variant_ID": f"synth_s1_v{i}",
                "ML_Label": i % 2,
                "Reference_Sentence": "क ख ग",
                "Variant_Sentence": f"ख ग क{i}",
                "DL_Reference": ref_dl,
                "DL_Variant": v,
            } for i, v in enumerate(var_dls, 1)])

        low = recommend_order(_mini_pairs(10, [4, 12]), ["dependency_length"])
        check("Lower-DL variant beats the reference",
              low["sentences"][0]["recommended"]["id"] == "synth_s1_v1")

        tie = recommend_order(_mini_pairs(7, [7, 7]), ["dependency_length"])
        check("Exact tie goes to the reference",
              tie["sentences"][0]["recommended"]["is_reference"])

        nan_rec = recommend_order(
            _mini_pairs(float("nan"), [float("nan"), float("nan")]),
            ["dependency_length"],
        )
        check("All-NaN feature -> no recommendation",
              nan_rec["sentences"][0]["recommended"] is None
              and all(c["score"] is None
                      for c in nan_rec["sentences"][0]["candidates"]))

        empty_rec = recommend_order(
            pd.DataFrame(columns=scored.columns), ["dependency_length"]
        )
        check("Empty input -> no sentences", empty_rec["sentences"] == [])

        try:
            recommend_order(scored, ["not_a_scorer"])
            check("Unknown scorer raises KeyError", False, "no exception raised")
        except KeyError:
            check("Unknown scorer raises KeyError", True)
    except Exception as e:
        check("recommend_order ran without exception", False, str(e))
        traceback.print_exc()
else:
    print("  Skipped (no scored pairs from 3e)")

section("3h. recommend_column — recommendation as variants.csv columns")

if passed and not pairs_df.empty and "scored" in globals():
    try:
        from scoring import recommend_column, recommend_order

        names = ["dependency_length", "information_status"]
        col_df = recommend_column(scored, names)
        cols = ["Recommended_Sentence", "Recommended_Is_Reference",
                "Recommended_Score"]

        check("Returns a DataFrame with the three columns",
              col_df is not None and list(col_df.columns) == cols)
        check("Aligned to the pairs table (index + length)",
              col_df.index.equals(scored.index) and len(col_df) == len(scored))

        merged = scored.join(col_df)
        # One distinct recommendation per source sentence.
        per_sent = merged.groupby("Sent_ID")[cols].nunique()
        check("Constant within each Sent_ID",
              bool((per_sent <= 1).all().all()))

        # Matches recommend_order's per-sentence winner exactly.
        rec = recommend_order(scored, names)
        want = {s["sent_id"]: s["recommended"] for s in rec["sentences"]}
        ok_match = True
        for sid, grp in merged.groupby("Sent_ID"):
            r = grp.iloc[0]
            w = want[str(sid)]
            if w is None:
                if not (pd.isna(r["Recommended_Sentence"])
                        or r["Recommended_Sentence"] is None):
                    ok_match = False
            elif (r["Recommended_Sentence"] != w["sentence"]
                  or bool(r["Recommended_Is_Reference"]) != w["is_reference"]):
                ok_match = False
        check("Column winner matches recommend_order's pick", ok_match)

        # No rankable scorer -> no columns at all.
        check("Empty scorer list -> None (columns omitted)",
              recommend_column(scored, []) is None)

        # Synthetic: lower-DL variant wins, broadcast to every row.
        mini = _mini_pairs(10, [4, 12])
        mini_cols = recommend_column(mini, ["dependency_length"])
        check("Synthetic: winner broadcast to all rows",
              mini_cols["Recommended_Sentence"].nunique() == 1
              and mini_cols["Recommended_Is_Reference"].iloc[0] == False)

        n_ref = int(merged.groupby("Sent_ID")["Recommended_Is_Reference"]
                    .first().sum())
        print(f"    {merged['Sent_ID'].nunique()} sentences | reference order "
              f"recommended in {n_ref}")
    except Exception as e:
        check("recommend_column ran without exception", False, str(e))
        traceback.print_exc()
else:
    print("  Skipped (no scored pairs from 3e)")

section("3i. cancellation — heartbeat, killer, child processes")

try:
    import subprocess as _sp
    import sys as _sys
    import time as _time

    from helpers import (CancelToken, JobCancelled, bind, current, heartbeat,
                         register_process, unbind)
    from scoring import apply_scorers

    # Outside a job nothing is bound: the library must be unaffected.
    unbind()
    heartbeat()
    check("Heartbeat is a no-op when unbound", current() is None)

    tok = CancelToken()
    bind(tok)
    try:
        heartbeat()
        check("Heartbeat passes while the token is live", True)
    except JobCancelled:
        check("Heartbeat passes while the token is live", False)

    beat_before = tok.last_beat
    _time.sleep(0.01)
    heartbeat()
    check("Heartbeat records liveness", tok.last_beat > beat_before)

    tok.cancel()                                   # the killer
    check("cancel() flips the token", tok.cancelled)
    try:
        heartbeat()
        check("Heartbeat raises once cancelled", False, "no exception raised")
    except JobCancelled:
        check("Heartbeat raises once cancelled", True)
    tok.cancel()                                   # idempotent
    check("cancel() is idempotent", tok.cancelled)

    unbind()
    heartbeat()
    check("Heartbeat is a no-op again after unbind", current() is None)

    # The killer must terminate registered child processes — the part a
    # cooperative flag alone cannot do (e.g. the Berkeley JVM).
    tok2 = CancelToken()
    bind(tok2)
    proc = _sp.Popen([_sys.executable, "-c", "import time; time.sleep(60)"])
    with register_process(proc):
        tok2.cancel()
        try:
            proc.wait(timeout=15)
            killed = True
        except Exception:
            killed = False
            proc.kill()
    check("Killer terminates a registered child process", killed,
          f"returncode={proc.returncode}")
    unbind()

    # Registering after a cancel must not leave an orphan behind.
    tok3 = CancelToken()
    tok3.cancel()
    bind(tok3)
    late = _sp.Popen([_sys.executable, "-c", "import time; time.sleep(60)"])
    with register_process(late):
        pass
    try:
        late.wait(timeout=15)
        late_killed = True
    except Exception:
        late_killed = False
        late.kill()
    check("Child registered after a cancel is killed too", late_killed)
    unbind()

    # A cancelled scorer run unwinds out of apply_scorers.
    if "pairs_df" in globals() and not pairs_df.empty:
        tok4 = CancelToken()
        tok4.cancel()
        bind(tok4)
        try:
            apply_scorers(pairs_df.head(10), ["dependency_length"])
            check("apply_scorers unwinds on a cancelled token", False,
                  "no exception raised")
        except JobCancelled:
            check("apply_scorers unwinds on a cancelled token", True)
        finally:
            unbind()
except Exception as e:
    check("cancellation helpers ran without exception", False, str(e))
    traceback.print_exc()
    unbind()


# ═══════════════════════════════════════════════════════════════════
# 3j. Trigram scorer (Katz backoff)
# ═══════════════════════════════════════════════════════════════════
section("3j. trigram — Katz-backoff trigram scorer")

try:
    from scoring import apply_scorers, get_scorers

    registry = get_scorers()
    check("trigram is discovered", "trigram" in registry)
    katz = registry["trigram"]
    check("trigram declares Delta_Trigram",
          [d[0] for d in katz.deltas()] == ["Delta_Trigram"])
    check("lower surprisal is more natural",
          katz.higher_is_more_natural.get("Delta_Trigram") is False)
    # The NLTK MLE scorer was retired on 2026-08-29; the Katz model took
    # over the `trigram` name, so the old one must be gone, not renamed.
    check("the retired MLE scorer is gone", "trigram_katz" not in registry)
    check("trigram is the Katz-backoff model", "Katz" in katz.built_with)

    from scoring._katz_trigram import normalize

    # The danda is a delimiter in the training text, so it must not survive
    # into a scored token — this is the alignment the model depends on.
    check("normalize drops the sentence-final danda",
          "।" not in normalize("राम ने किताब पढ़ी ।"))
    check("normalize collapses to empty for punctuation-only input",
          normalize("।") == "")

    katz_pairs = pd.DataFrame({
        "Sent_ID": ["s1", "s1"],
        "Variant_ID": ["s1_v1", "s1_v2"],
        "ML_Label": [1, 0],
        "Reference_Sentence": ["राम ने किताब पढ़ी ।"] * 2,
        "Variant_Sentence": ["किताब राम ने पढ़ी ।", "ने राम किताब पढ़ी ।"],
    })
    scored_katz = apply_scorers(katz_pairs, ["trigram"])

    check("emits both raw columns",
          {"Trigram_Reference", "Trigram_Variant"} <= set(scored_katz.columns))
    check("emits the delta column", "Delta_Trigram" in scored_katz.columns)
    check("row count and order preserved",
          list(scored_katz["Variant_ID"]) == list(katz_pairs["Variant_ID"]))
    values = scored_katz[["Trigram_Reference", "Trigram_Variant"]].to_numpy(float)
    check("no NaN or inf in the raw columns", bool(np.isfinite(values).all()))
    check("surprisal is positive", bool((values > 0).all()))

    # Central diff step: ref-var when ML_Label==1, var-ref when 0.
    r0 = scored_katz.iloc[0]
    r1 = scored_katz.iloc[1]
    check("delta is ref-var on ML_Label==1",
          abs(r0["Delta_Trigram"]
              - (r0["Trigram_Reference"] - r0["Trigram_Variant"])) < 1e-9)
    check("delta is var-ref on ML_Label==0",
          abs(r1["Delta_Trigram"]
              - (r1["Trigram_Variant"] - r1["Trigram_Reference"])) < 1e-9)

    empty_katz = apply_scorers(katz_pairs.iloc[0:0].copy(), ["trigram"])
    check("empty pairs frame is handled",
          "Trigram_Reference" in empty_katz.columns and len(empty_katz) == 0)

except Exception as e:
    check("trigram section raised no exception", False, f"{type(e).__name__}: {e}")
    traceback.print_exc()


# ═══════════════════════════════════════════════════════════════════
# 3k. Wikipedia-50M adaptive LSTM scorer
# ═══════════════════════════════════════════════════════════════════
section("3k. adaptive_lstm_wiki50m — Wikipedia-50M LSTM")

try:
    from scoring import apply_scorers, get_scorers

    registry = get_scorers()
    check("adaptive_lstm_wiki50m is discovered", "adaptive_lstm_wiki50m" in registry)
    aw = registry["adaptive_lstm_wiki50m"]
    check("adaptive_lstm_wiki50m declares Delta_AdaptiveWiki50M",
          [d[0] for d in aw.deltas()] == ["Delta_AdaptiveWiki50M"])
    check("lower surprisal is more natural",
          aw.higher_is_more_natural.get("Delta_AdaptiveWiki50M") is False)
    check("it asks for a context sentence", aw.needs_previous_sentence is True)

    from scoring._lstm_wiki50m import normalize as w_normalize

    check("normalize drops the sentence-final danda",
          "।" not in w_normalize("राम ने किताब पढ़ी ।"))
    check("normalize collapses to empty for punctuation-only input",
          w_normalize("।") == "")

    w_pairs = pd.DataFrame({
        "Sent_ID": ["s1", "s1"],
        "Variant_ID": ["s1_v1", "s1_v2"],
        "ML_Label": [1, 0],
        "Reference_Sentence": ["राम ने किताब पढ़ी ।"] * 2,
        "Variant_Sentence": ["किताब राम ने पढ़ी ।", "ने राम किताब पढ़ी ।"],
    })
    # No corpus context: the adaptation step is skipped, scores stay finite.
    scored_w = apply_scorers(w_pairs, ["adaptive_lstm_wiki50m"], context={})
    cols_w = ["AdaptiveWiki50M_Reference", "AdaptiveWiki50M_Variant"]
    check("emits both raw columns", set(cols_w) <= set(scored_w.columns))
    check("emits the delta column", "Delta_AdaptiveWiki50M" in scored_w.columns)
    check("row count and order preserved",
          list(scored_w["Variant_ID"]) == list(w_pairs["Variant_ID"]))
    w_values = scored_w[cols_w].to_numpy(float)
    check("no NaN or inf in the raw columns", bool(np.isfinite(w_values).all()))
    check("surprisal is positive", bool((w_values > 0).all()))

    r0, r1 = scored_w.iloc[0], scored_w.iloc[1]
    check("delta is ref-var on ML_Label==1",
          abs(r0["Delta_AdaptiveWiki50M"]
              - (r0["AdaptiveWiki50M_Reference"] - r0["AdaptiveWiki50M_Variant"])) < 1e-9)
    check("delta is var-ref on ML_Label==0",
          abs(r1["Delta_AdaptiveWiki50M"]
              - (r1["AdaptiveWiki50M_Variant"] - r1["AdaptiveWiki50M_Reference"])) < 1e-9)

    empty_w = apply_scorers(w_pairs.iloc[0:0].copy(), ["adaptive_lstm_wiki50m"], context={})
    check("empty pairs frame is handled",
          "AdaptiveWiki50M_Reference" in empty_w.columns and len(empty_w) == 0)

    # Batching must not change the answer (speed only, float noise at most).
    from scoring._lstm_wiki50m import batch_surprisal, get_lstm_wiki50m
    _m, _v, _d = get_lstm_wiki50m()
    _sents = list(w_pairs["Reference_Sentence"]) + list(w_pairs["Variant_Sentence"])
    _b1 = batch_surprisal(_sents, _m, _v, _d, batch_size=1)
    _b8 = batch_surprisal(_sents, _m, _v, _d, batch_size=8)
    check("batch size does not change the score",
          max(abs(a - b) for a, b in zip(_b1, _b8)) < 1e-3)
    check("no context -> scores equal the unadapted model (<= 1e-4)",
          max(abs(scored_w["AdaptiveWiki50M_Variant"].iat[i] - _b1[2 + i])
              for i in range(2)) <= 1e-4)

except Exception as e:
    check("adaptive_lstm_wiki50m section raised no exception", False, f"{type(e).__name__}: {e}")
    traceback.print_exc()


# ═══════════════════════════════════════════════════════════════════
# 3l. word order — SOV / DOSV / IOSV construction labels
# ═══════════════════════════════════════════════════════════════════
section("3l. word order — SOV / DOSV / IOSV labels")

# Ranjan & van Schijndel (2024), Table 1: canonical SOV vs object-fronted OSV,
# split into DOSV (direct object before the subject) and IOSV (indirect object
# before the subject).  OSV is never stored — it is DOSV or IOSV — and DOSV
# wins when both objects are fronted.
try:
    from helpers import classify_word_order, is_osv, recover_permutation
    from variants import label_word_orders, WORD_ORDER_COLUMNS
    from evaluation import evaluate_by_construction, construction_mask, CONSTRUCTIONS

    # Pure rule, both schemes.
    cases = [
        (["nsubj", "obj", "obl"], "SOV"),
        (["obj", "nsubj", "obl"], "DOSV"),
        (["iobj", "nsubj", "obj"], "IOSV"),
        (["iobj", "obj", "nsubj"], "DOSV"),       # both fronted -> DOSV wins
        (["obl", "obj"], "DOSV"),                 # object but no preverbal subject
        (["nsubj", "obl"], "SOV"),                # no object at all
        (["k1", "k2"], "SOV"),
        (["k2", "k1"], "DOSV"),
        (["k4", "k1", "k2"], "IOSV"),
        (["nsubj:pass", "obj"], "SOV"),
    ]
    check("classify_word_order: all synthetic cases",
          all(classify_word_order(d) == exp for d, exp in cases),
          str([(d, classify_word_order(d), exp) for d, exp in cases if classify_word_order(d) != exp]))
    check("is_osv: DOSV and IOSV only",
          is_osv("DOSV") and is_osv("IOSV") and not is_osv("SOV"))

    if passed and not pairs_df.empty:
        wo = label_word_orders(pairs_df, passed)
        check("label_word_orders: index-aligned, two columns",
              list(wo.columns) == WORD_ORDER_COLUMNS and wo.index.equals(pairs_df.index))
        joined = pairs_df.join(wo)
        check("Row count unchanged by the join", len(joined) == len(pairs_df))
        allowed = {"SOV", "DOSV", "IOSV"}
        check("Labels drawn from {SOV, DOSV, IOSV} on both sides",
              set(joined["Reference_Word_Order"]) <= allowed
              and set(joined["Variant_Word_Order"]) <= allowed,
              f"ref={set(joined['Reference_Word_Order'])} var={set(joined['Variant_Word_Order'])}")
        check("Reference_Word_Order constant within each Sent_ID",
              (joined.groupby("Sent_ID")["Reference_Word_Order"].nunique() == 1).all())

        # Reference label == the rule applied to the parse's own deprels.
        by_id = {it["sentence"].metadata.get("sent_id", "Unknown_ID"): it for it in passed}
        firsts = joined.drop_duplicates("Sent_ID")
        agree = all(
            classify_word_order(_const_deprels(by_id[r.Sent_ID])) == r.Reference_Word_Order
            for r in firsts.itertuples() if r.Sent_ID in by_id
        )
        check("Reference label matches classify_word_order on the parse", agree)

        # Distribution over the Paninian passed set (measured 2026-09-12 on
        # the 1,726 passed sentences: SOV 1,480 / DOSV 178 / IOSV 68).
        dist = firsts["Reference_Word_Order"].value_counts().to_dict()
        check("DOSV and IOSV references both occur",
              dist.get("DOSV", 0) > 0 and dist.get("IOSV", 0) > 0, str(dist))
        check("SOV is the majority construction",
              dist.get("SOV", 0) > dist.get("DOSV", 0) + dist.get("IOSV", 0), str(dist))
        full_dist = {"SOV": 0, "DOSV": 0, "IOSV": 0}
        for it in passed:
            full_dist[classify_word_order(_const_deprels(it))] += 1
        check("Passed-set distribution == SOV 1480 / DOSV 178 / IOSV 68",
              full_dist == {"SOV": 1480, "DOSV": 178, "IOSV": 68}, str(full_dist))

        # Permutations do front objects, and every variant label agrees with
        # the permutation recovered from its surface.
        sov_rows = joined[joined["Reference_Word_Order"] == "SOV"]
        check("Some variants of SOV references are object-fronted",
              (sov_rows["Variant_Word_Order"] != "SOV").any())
        spot_ok = True
        for r in joined.head(500).itertuples():
            item = by_id.get(r.Sent_ID)
            if item is None:
                continue
            deps = _const_deprels(item)
            perm = recover_permutation(r.Variant_Sentence.split(" "), item["constituents"])
            if classify_word_order([deps[i] for i in perm]) != r.Variant_Word_Order:
                spot_ok = False
                break
        check("Variant label matches the recovered permutation (first 500 rows)", spot_ok)

        # Unknown reference -> blank, never an exception: by Ref_Index, and by
        # Sent_ID on an older table that has no Ref_Index column.
        odd = pairs_df.head(2).copy()
        odd["Ref_Index"] = -999
        blank = label_word_orders(odd, passed)
        check("Unknown Ref_Index gives empty labels",
              (blank["Reference_Word_Order"] == "").all() and (blank["Variant_Word_Order"] == "").all())
        odd = pairs_df.head(2).drop(columns=["Ref_Index", "Variant_No"])
        odd["Sent_ID"] = "no_such_sentence"
        blank = label_word_orders(odd, passed)
        check("Unknown Sent_ID (no Ref_Index column) gives empty labels",
              (blank["Reference_Word_Order"] == "").all() and (blank["Variant_Word_Order"] == "").all())
        empty = label_word_orders(pairs_df.iloc[0:0], passed)
        check("Empty pairs table gives an empty, two-column frame",
              len(empty) == 0 and list(empty.columns) == WORD_ORDER_COLUMNS)

        # Per-construction evaluation on the 3e scored sample.
        if "scored" in globals() and not scored.empty:
            sc = scored.join(label_word_orders(scored, passed))
            per = evaluate_by_construction(sc, min_pairs=1)
            check("evaluate_by_construction: one entry per construction, in order",
                  [e["key"] for e in per] == [c[0] for c in CONSTRUCTIONS])
            check("Every entry has a valid status",
                  all(e["status"] in ("ok", "insufficient_pairs", "not_evaluable") for e in per),
                  str([(e["key"], e["status"]) for e in per]))
            n = {e["key"]: e["n_pairs"] for e in per}
            check("OSV subset == DOSV + IOSV subsets", n["OSV"] == n["DOSV"] + n["IOSV"], str(n))
            check("Subsets pair a non-canonical reference with an SOV variant",
                  (sc[construction_mask(sc, {"DOSV", "IOSV"})]["Variant_Word_Order"] == "SOV").all())
            check("SOV subset == every SOV-reference row, any variant",
                  n["SOV"] == int((sc["Reference_Word_Order"] == "SOV").sum()), str(n))
            check("SOV subset holds no object-fronted reference",
                  not sc[construction_mask(sc, {"SOV"}, None)]["Reference_Word_Order"]
                  .isin({"DOSV", "IOSV"}).any())
            ok_entries = [e for e in per if e["status"] == "ok"]
            check("ok entries carry evaluate_pairs fields",
                  all({"predictors", "n_pairs", "folds"} <= set(e) for e in ok_entries))
            print(f"      per-construction: {[(e['key'], e['n_pairs'], e['status']) for e in per]}")
            for e in ok_entries:
                for p_ in e["predictors"]:
                    print(f"        {e['key']:4s} {p_['label']:22s} acc={p_['accuracy']:.4f} n={p_['n']}")
        try:
            evaluate_by_construction(pairs_df)
            check("evaluate_by_construction rejects a table without labels", False)
        except ValueError:
            check("evaluate_by_construction rejects a table without labels", True)
    else:
        print("  Skipped corpus checks (no variants)")
except Exception as e:
    check("word order section raised no exception", False, f"{type(e).__name__}: {e}")
    traceback.print_exc()


# ═══════════════════════════════════════════════════════════════════
# 3m. Reference grouping keys + batched LSTM engine (added 2026-10-04)
# ═══════════════════════════════════════════════════════════════════
section("3m. Ref_Index grouping + batched/prefix-cached LSTM engine")

try:
    import copy as _copy
    import os as _os3m

    from helpers import (
        build_corpus_context as _bcc,
        passed_ref_indices,
        reference_groups,
        reference_lookup,
        stamp_ref_index,
    )
    from scoring import apply_scorers, get_scorers

    # --- keys ---------------------------------------------------------------
    check("load_input stamps hnt_index 1..n in file order",
          [s_.metadata.get("hnt_index") for s_ in conllu_sents[:3]] == ["1", "2", "3"]
          and conllu_sents[-1].metadata.get("hnt_index") == str(len(conllu_sents)))
    check("passed sentences keep their file position (gaps where rejected)",
          passed_ref_indices(passed) == [int(it["sentence"].metadata["hnt_index"]) for it in passed]
          and passed_ref_indices(passed) == sorted(passed_ref_indices(passed)))
    if not pairs_df.empty:
        keyed = pairs_df.sort_values(["Ref_Index", "Variant_No"], kind="stable")
        check("sorting by (Ref_Index, Variant_No) reproduces the emitted order",
              keyed.index.equals(pairs_df.index))
        check("Variant_No is 1..n within every reference",
              all(list(g) == list(range(1, len(g) + 1))
                  for _, g in pairs_df.groupby("Ref_Index")["Variant_No"]))
        check("exactly one Sent_ID per Ref_Index",
              int(pairs_df.groupby("Ref_Index")["Sent_ID"].nunique().max()) == 1)
        check("reference_groups: one group per Ref_Index, positions cover every row",
              len(reference_groups(pairs_df)) == pairs_df["Ref_Index"].nunique()
              and sorted(p for _, ps in reference_groups(pairs_df) for p in ps)
              == list(range(len(pairs_df))))

    # Duplicated sent_id: two different references must stay two groups, each
    # with its own parse and its own predecessor.
    dup_items = [_copy.deepcopy(passed[i]) for i in (0, 1, 2)]
    dup_sents = [it["sentence"] for it in dup_items]
    for s_, sid in zip(dup_sents, ["dup", "middle", "dup"]):
        s_.metadata["sent_id"] = sid
    stamp_ref_index(dup_sents)
    dup_passed = [dup_items[0], dup_items[2]]
    # Every adjacency licit: with two sentences the corpus-derived grammar
    # filter can leave one of them without a single variant.
    dup_labels = {t["deprel"] for it in dup_passed for c in it["constituents"] for t in c}
    dup_pairs = generate_variants(
        dup_passed, valid_deprel_pairs={(a, b) for a in dup_labels for b in dup_labels})
    dup_ctx = _bcc(dup_sents)
    check("duplicated sent_id -> two reference groups (Ref_Index 1 and 3)",
          sorted(dup_pairs["Ref_Index"].unique().tolist()) == [1, 3]
          and dup_pairs["Sent_ID"].unique().tolist() == ["dup"],
          f"Ref_Index={sorted(dup_pairs['Ref_Index'].unique().tolist())}")
    looked = reference_lookup(dup_pairs, dup_passed)
    check("reference_lookup maps each row to its own parse",
          all(it is dup_passed[0 if r == 1 else 1]
              for it, r in zip(looked, dup_pairs["Ref_Index"])))
    check("previous(): duplicated id resolved by ref_index",
          dup_ctx.previous("dup", 1, ref_index=3) == [dup_sents[1]]
          and dup_ctx.previous("dup", 1, ref_index=1) == []
          and dup_ctx.previous("dup", 1) == [])          # sent_id alone: first occurrence

    # --- context: rejected predecessors count, documents don't leak ----------
    full_ctx = _bcc(conllu_sents)
    passed_idx = set(passed_ref_indices(passed))
    after_rejected = next((it for it in passed
                           if int(it["sentence"].metadata["hnt_index"]) > 1
                           and int(it["sentence"].metadata["hnt_index"]) - 1 not in passed_idx),
                          None)
    if after_rejected is not None:
        k_ = int(after_rejected["sentence"].metadata["hnt_index"])
        got_ = full_ctx.previous(after_rejected["sentence"].metadata["sent_id"], 1, ref_index=k_)
        check("context: a filtered-out predecessor is still the preceding sentence",
              len(got_) == 1 and got_[0] is conllu_sents[k_ - 2], f"ref_index {k_}")
    else:
        check("context: corpus has a passed sentence right after a rejected one", False)
    check("context: the first sentence of a file has no predecessor",
          full_ctx.previous(conllu_sents[0].metadata["sent_id"], 3, ref_index=1) == [])

    from conllu import parse as _parse_conllu
    _row = "1\tx\tx\tX\t_\t_\t0\troot\t_\t_\n\n"
    for marker in ("# newdoc id = d2", "# newdoc"):
        nd_sents = _parse_conllu("".join(
            (f"{marker}\n" if i == 3 else "") + f"# sent_id = s{i}\n" + _row
            for i in range(1, 5)))
        stamp_ref_index(nd_sents)
        nd_ctx = _bcc(nd_sents)
        s1_, s2_, s3_, s4_ = nd_sents
        check(f"context: '{marker}' starts a document (no predecessor across it)",
              nd_ctx.previous("s3", 1, ref_index=3) == []
              and nd_ctx.previous("s4", 1, ref_index=4) == [s3_]
              and nd_ctx.previous("s4", 3, ref_index=4) == [s3_]
              and nd_ctx.previous("s2", 1, ref_index=2) == [s1_]
              and nd_ctx.previous_text("s3", 1, ref_index=3) == "")

    # --- engine -------------------------------------------------------------
    from scoring._lstm_engine import sequence_surprisals
    from scoring._lstm_wiki50m import _ids as w_ids
    from scoring._lstm_wiki50m import batch_surprisal, get_lstm_wiki50m

    m_, w2i_, dev_ = get_lstm_wiki50m()
    eng_sample = pairs_df[pairs_df["Ref_Index"].isin(
        pairs_df["Ref_Index"].unique()[:4])].copy()
    texts = [eng_sample["Reference_Sentence"].iat[0]] + eng_sample["Variant_Sentence"].tolist()
    one_by_one = batch_surprisal(texts, m_, w2i_, dev_, batch_size=1)
    ids_ = [w_ids(t, w2i_) for t in texts]
    worst = 0.0
    for pc in (True, False):
        for bs in (1, 7, 16):
            got = sequence_surprisals(ids_, m_, dev_, batch_size=bs, prefix_cache=pc)
            worst = max(worst, max(abs(a - b) for a, b in zip(one_by_one, got)))
    check("engine == one-at-a-time scoring (any batch size, cache on/off), <= 1e-4 nats",
          worst <= 1e-4, f"max diff {worst:.2e}")
    check("engine: sequences shorter than two tokens score 0.0",
          sequence_surprisals([[5], []], m_, dev_) == [0.0, 0.0])

    lstm_names = ["adaptive_lstm_wiki50m"]
    eng_ctx = {"corpus": _bcc(conllu_sents), "passed": passed,
               "scheme": "paninian"}

    def _run(env, frame=None, ctx=eng_ctx):
        saved = {k: _os3m.environ.get(k) for k in env}
        _os3m.environ.update(env)
        try:
            return apply_scorers((eng_sample if frame is None else frame).copy(),
                                 lstm_names, context=ctx)
        finally:
            for k, v in saved.items():
                if v is None:
                    _os3m.environ.pop(k, None)
                else:
                    _os3m.environ[k] = v

    def _sd_sum(model):
        return sum(float(t.double().sum()) for t in model.state_dict().values())

    base_sum = _sd_sum(m_)
    on = _run({"HNT_LSTM_PREFIX_CACHE": "1"})
    off = _run({"HNT_LSTM_PREFIX_CACHE": "0", "HNT_LSTM_BATCH": "3"})
    cols = [c for c in on.columns if c.endswith(("_Reference", "_Variant"))]
    diff = max(float(np.nanmax(np.abs(on[c].to_numpy(float) - off[c].to_numpy(float))))
               for c in cols)
    check("prefix cache on/off and batch size give the same scores (<= 1e-4)",
          diff <= 1e-4, f"max diff {diff:.2e}")
    check("adaptive LSTM scores finite", bool(np.isfinite(on[cols].to_numpy(float)).all()))
    again = _run({"HNT_LSTM_PREFIX_CACHE": "1"})
    check("adaptive LSTM is deterministic (two runs bit-identical)",
          on["AdaptiveWiki50M_Reference"].equals(again["AdaptiveWiki50M_Reference"])
          and on["AdaptiveWiki50M_Variant"].equals(again["AdaptiveWiki50M_Variant"]))
    check("shared base LSTM untouched by adaptation", _sd_sum(m_) == base_sum)

    # Unadapted score of every variant, for the no-context / newdoc checks.
    plain = dict(zip(eng_sample.index,
                     batch_surprisal(eng_sample["Variant_Sentence"].tolist(), m_, w2i_, dev_)))

    def _gap_to_plain(frame, k=None):
        rows = frame if k is None else frame[frame["Ref_Index"] == k]
        return float(max(abs(rows.at[i, "AdaptiveWiki50M_Variant"] - plain[i])
                         for i in rows.index))

    no_ctx = _run({}, ctx={})
    gap = _gap_to_plain(no_ctx)
    check("no preceding sentence -> adaptive == unadapted LSTM (<= 1e-4)", gap <= 1e-4,
          f"max diff {gap:.2e}")

    # A '# newdoc' on a reference cuts its context: no adaptation for that
    # reference only, while a reference with a predecessor still adapts.
    nd_refs = [int(k) for k in eng_sample["Ref_Index"].unique() if int(k) > 1]
    nd_k, other_k = nd_refs[0], nd_refs[1]
    nd_corpus = list(conllu_sents)
    nd_corpus[nd_k - 1] = _copy.deepcopy(conllu_sents[nd_k - 1])
    nd_corpus[nd_k - 1].metadata["newdoc id"] = "test_doc"
    nd_scored = _run({}, ctx={"corpus": _bcc(nd_corpus), "passed": passed,
                              "scheme": "paninian"})
    check("newdoc reference -> no adaptation (== unadapted LSTM, <= 1e-4)",
          _gap_to_plain(nd_scored, nd_k) <= 1e-4,
          f"max diff {_gap_to_plain(nd_scored, nd_k):.2e}")
    check("reference with a predecessor in its document still adapts",
          _gap_to_plain(nd_scored, other_k) > 1e-4,
          f"max diff {_gap_to_plain(nd_scored, other_k):.2e}")

    # Row order must not matter: score a shuffled copy, then undo the shuffle.
    shuffled = eng_sample.sample(frac=1.0, random_state=7)
    sh = _run({"HNT_LSTM_PREFIX_CACHE": "1"}, frame=shuffled)
    check("scorers never reorder rows (index preserved)", sh.index.equals(shuffled.index))
    back = sh.loc[on.index]
    sdiff = max(float(np.abs(back[c].to_numpy(float) - on[c].to_numpy(float)).max())
                for c in cols)
    check("scores are independent of row order (<= 1e-4)", sdiff <= 1e-4,
          f"max diff {sdiff:.2e}")
    print(f"    engine parity {worst:.1e} | cache on/off {diff:.1e} | "
          f"no-context gap {gap:.1e} | shuffle {sdiff:.1e}")
except Exception as e:
    check("section 3m raised no exception", False, f"{type(e).__name__}: {e}")
    traceback.print_exc()


# ═══════════════════════════════════════════════════════════════════
# 4. summarize
# ═══════════════════════════════════════════════════════════════════
section("4. summarize — full corpus results")

if passed:
    try:
        summary = summarize(passed, rejected_df)

        check("Returns a dict",                         isinstance(summary, dict))
        check("total_input == 17405",                   summary["total_input"] == 17405,
              f"got {summary['total_input']}")
        check("total_passed + total_rejected == total_input",
              summary["total_passed"] + summary["total_rejected"] == summary["total_input"])
        wo_dist = summary.get("word_order_distribution", {})
        check("word_order_distribution has SOV / DOSV / IOSV keys",
              set(wo_dist) == {"SOV", "DOSV", "IOSV"}, str(wo_dist))
        check("word_order_distribution sums to total_passed",
              sum(wo_dist.values()) == summary["total_passed"], str(wo_dist))
        check("pass_rate in (0, 1)",                    0 < summary["pass_rate"] < 1,
              f"got {summary['pass_rate']}")
        check("rejection_counts is a dict",             isinstance(summary["rejection_counts"], dict))
        check("filter_order has 7 entries",             len(summary["filter_order"]) == 7)
        check("avg_sentence_length > 0",                summary["avg_sentence_length"] > 0)
        check("avg_constituent_count >= 2",                  summary["avg_constituent_count"] >= 2,
              f"got {summary['avg_constituent_count']}")
        check("constituent_count_distribution is a dict",    isinstance(summary["constituent_count_distribution"], dict))
        check("min constituent bucket is >= 2",
              min(summary["constituent_count_distribution"].keys()) >= 2)

        # Save summary to JSON
        summary_path = TEST_DIR / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        check("summary.json saved",                     summary_path.exists())

        print(f"\n    Summary report:")
        print(f"      total_input          : {summary['total_input']}")
        print(f"      total_passed         : {summary['total_passed']}")
        print(f"      total_rejected       : {summary['total_rejected']}")
        print(f"      pass_rate            : {summary['pass_rate']:.2%}")
        print(f"      avg_sentence_length  : {summary['avg_sentence_length']} tokens")
        print(f"      avg_constituent_count     : {summary['avg_constituent_count']}")
        print(f"      constituent_count_distribution: {summary['constituent_count_distribution']}")
        print(f"\n      Per-filter rejection counts:")
        for f_entry in summary["filter_order"]:
            print(f"        {f_entry['filter']:<45}: {f_entry['rejected']}")

    except Exception as e:
        check("summarize raised no exception", False, str(e))
        traceback.print_exc()
else:
    print("  Skipped (no passed sentences)")


# ═══════════════════════════════════════════════════════════════════
# 5. File output validation
# ═══════════════════════════════════════════════════════════════════
section("5. Output files validation")

expected_files = {
    "passed_sentences.csv": ["Sent_ID","Sentence","Root_ID","Constituent_Count"],
    "rejected_sentences.csv": ["Sent_ID","Sentence","Reason"],
    "reference_variant_pairs.csv": ["Reference_Sentence","Variant_Sentence","ML_Label"],
    "summary.json": None,
}

for fname, expected_cols in expected_files.items():
    fpath = TEST_DIR / fname
    check(f"{fname} exists", fpath.exists())
    if fpath.exists() and expected_cols:
        try:
            df = pd.read_csv(fpath)
            check(f"{fname} has expected columns",
                  all(c in df.columns for c in expected_cols),
                  f"cols={list(df.columns)}")
            check(f"{fname} is non-empty", len(df) > 0, f"rows={len(df)}")
        except Exception as e:
            check(f"{fname} readable as CSV", False, str(e))


# ═══════════════════════════════════════════════════════════════════
# 6. Rate limiting & upload guard (webapp)
# ═══════════════════════════════════════════════════════════════════
section("6. Rate limiting and upload guard")

# The first HTTP-level section in this suite: everything above drives the
# library directly.  TestClient needs no server — it speaks ASGI to the app
# object.  Guarded routes are exercised with deliberately invalid payloads so
# the limiter is measured without ever starting a real pipeline; the rate
# check runs as a dependency, which FastAPI resolves before the handler.
import os as _os

# Importing the web app pulls in the vendored Taru backend, which runs
# os.chdir(HERE) at import time (taru/workspace/taru_backend.py:17).  Every
# path in this suite is relative, so the cwd is restored in the finally below —
# without it, the final report cannot be written.
_CWD_BEFORE_WEBAPP = _os.getcwd()

try:
    import os
    import time

    os.environ.setdefault("HNT_NO_PREWARM", "1")
    from fastapi.testclient import TestClient

    from webapp import jobs as web_jobs
    from webapp import ratelimit
    from webapp.app import MAX_TEXT_CHARS, app

    HINDI_SENTENCE = "राम ने किताब पढ़ी ।"

    # --- the poller and the other cheap reads are never limited -----
    ratelimit.reset()
    client = TestClient(app)
    poll_codes = {client.get("/api/jobs/does-not-exist").status_code for _ in range(60)}
    check("status poller survives 60 rapid calls", 429 not in poll_codes, f"codes={poll_codes}")
    plugin_codes = {client.get("/api/plugins").status_code for _ in range(30)}
    check("/api/plugins is never rate-limited", 429 not in plugin_codes, f"codes={plugin_codes}")

    # --- session cookie -------------------------------------------
    ratelimit.reset()
    client = TestClient(app)
    first = client.post("/api/jobs", data={})
    sid = client.cookies.get(ratelimit.SESSION_COOKIE)
    check("session cookie is minted", bool(sid))
    check("session cookie is HttpOnly",
          "httponly" in first.headers.get("set-cookie", "").lower(),
          first.headers.get("set-cookie"))
    # The cookie must survive an error reply: FastAPI discards the injected
    # Response when a handler raises, so this is set by SessionMiddleware.
    check("cookie is set even on an error response", first.status_code == 400 and bool(sid),
          f"status={first.status_code}")
    client.post("/api/jobs", data={})
    check("session cookie is reused, not re-minted",
          client.cookies.get(ratelimit.SESSION_COOKIE) == sid)

    # --- session rate tier ----------------------------------------
    ratelimit.reset()
    client = TestClient(app)
    codes = [client.post("/api/jobs", data={}).status_code
             for _ in range(ratelimit.JOBS_SESSION_PER_MIN + 1)]
    check("job POSTs under the session limit are not throttled",
          codes[:ratelimit.JOBS_SESSION_PER_MIN] == [400] * ratelimit.JOBS_SESSION_PER_MIN,
          f"codes={codes}")
    check("job POST over the session limit is 429", codes[-1] == 429, f"codes={codes}")
    throttled = client.post("/api/jobs", data={})
    check("429 carries a numeric Retry-After",
          throttled.headers.get("retry-after", "").isdigit(),
          throttled.headers.get("retry-after"))
    check("429 uses the rate message, not the busy message",
          "faster than" in throttled.json().get("detail", ""), throttled.json())

    # --- IP tier still binds when the cookie is cleared ------------
    ratelimit.reset()
    codes = []
    for _ in range(ratelimit.JOBS_IP_PER_MIN + 1):
        fresh_client = TestClient(app)          # new cookie jar = new session
        codes.append(fresh_client.post("/api/jobs", data={}).status_code)
    check("a fresh cookie never trips the session tier",
          codes[:ratelimit.JOBS_IP_PER_MIN] == [400] * ratelimit.JOBS_IP_PER_MIN,
          f"codes={codes}")
    check("IP tier still binds despite cookie resets", codes[-1] == 429, f"codes={codes}")

    # --- buckets are per route ------------------------------------
    ratelimit.reset()
    client = TestClient(app)
    for _ in range(ratelimit.JOBS_SESSION_PER_MIN):
        client.post("/api/jobs", data={})
    many_lines = "\n".join([HINDI_SENTENCE] * 50)
    spill = client.post("/taru/parse", json={"sentence": many_lines, "surprisal": False})
    check("the jobs bucket does not drain the taru bucket", spill.status_code == 400,
          spill.status_code)

    # --- /taru/parse line cap (400 fires before any JVM starts) ----
    check("too many lines in one /taru/parse is rejected", spill.status_code == 400,
          spill.status_code)
    check("line-cap message names the limit",
          "Too many lines" in spill.json().get("detail", ""), spill.json())

    # --- typed-text length cap ------------------------------------
    ratelimit.reset()
    client = TestClient(app)
    long_text = client.post("/api/jobs", data={"text": "क" * (MAX_TEXT_CHARS + 1)})
    check("over-long typed text is 400", long_text.status_code == 400, long_text.status_code)
    check("over-long text says so", "too long" in long_text.json().get("detail", ""),
          long_text.json())

    # --- upload cap -----------------------------------------------
    import webapp.app as webapp_app

    ratelimit.reset()
    real_cap = webapp_app.MAX_UPLOAD_BYTES
    started_at = time.time()
    webapp_app.MAX_UPLOAD_BYTES = 1024          # shrink so the test stays fast
    try:
        client = TestClient(app)
        oversized = b"x" * 5000
        big = client.post("/api/jobs",
                          files={"file": ("corpus.conllu", oversized, "text/plain")})
        check("oversized upload is 413", big.status_code == 413, big.status_code)
        check("413 names the limit", "upload limit" in big.json().get("detail", ""), big.json())
        stragglers = [p for p in web_jobs.OUTPUT_ROOT.glob("*/input.conllu")
                      if p.stat().st_mtime > started_at]
        check("no partial upload is left on disk", not stragglers, stragglers)
        # A lying Content-Length must not get past the streaming counter — the
        # middleware's header check is only a fast path, not the enforcement.
        spoofed = client.post("/api/jobs",
                              files={"file": ("corpus.conllu", oversized, "text/plain")},
                              headers={"Content-Length": "10"})
        check("spoofed Content-Length is still rejected",
              spoofed.status_code in (400, 413), spoofed.status_code)
    finally:
        webapp_app.MAX_UPLOAD_BYTES = real_cap

    ratelimit.reset()
    client = TestClient(app)
    blank = client.post("/api/jobs", files={"file": ("corpus.conllu", b"  \n\n ", "text/plain")})
    check("whitespace-only upload is still 'empty'",
          blank.status_code == 400 and "empty" in blank.json().get("detail", ""),
          blank.json())

    # --- active scorers (Scorer.active) ----------------------------
    from scoring import get_active_scorers, get_scorers as _all_scorers

    ACTIVE = {"dependency_length", "information_status", "trigram",
              "adaptive_lstm_wiki50m", "berkeley_pcfg"}
    check("library registry holds the five scorers",
          set(_all_scorers()) == ACTIVE, sorted(_all_scorers()))
    check("get_active_scorers() is all five",
          set(get_active_scorers()) == ACTIVE, sorted(get_active_scorers()))

    ratelimit.reset()
    client = TestClient(app)
    listed = [p["name"] for p in client.get("/api/plugins").json()]
    check("/api/plugins lists exactly the five scorers",
          set(listed) == ACTIVE and len(listed) == 5, listed)
    ratelimit.reset()
    bogus = client.post("/api/jobs", data={"text": HINDI_SENTENCE, "scorers": "no_such"})
    check("an unknown scorer is still 400 'Unknown scorer'",
          bogus.status_code == 400 and "Unknown scorer" in bogus.json().get("detail", ""),
          bogus.json())

    # --- in-flight concurrency cap (a real job) -------------------
    ratelimit.reset()
    client = TestClient(app)
    job_a = client.post("/api/jobs", data={"text": HINDI_SENTENCE, "scorers": ""})
    check("a first job is accepted", job_a.status_code == 200,
          f"status={job_a.status_code}")
    if job_a.status_code == 200:
        job_a_id = job_a.json()["job_id"]
        job_b = client.post("/api/jobs", data={"text": HINDI_SENTENCE, "scorers": ""})
        check("a second concurrent job is 429", job_b.status_code == 429, job_b.status_code)
        check("concurrency uses the busy message",
              "run in progress" in job_b.json().get("detail", ""), job_b.json())
        # Clearing the rate buckets proves the refusal is the in-flight cap and
        # not leftover rate budget from the two POSTs above.
        ratelimit.reset()
        still_busy = client.post("/api/jobs", data={"text": HINDI_SENTENCE, "scorers": ""})
        check("still refused after the rate window is cleared",
              still_busy.status_code == 429, still_busy.status_code)
        client.post(f"/api/jobs/{job_a_id}/cancel")
        ratelimit.reset()
        job_c = client.post("/api/jobs", data={"text": HINDI_SENTENCE, "scorers": ""})
        check("cancelling frees the slot", job_c.status_code == 200, job_c.status_code)
        if job_c.status_code == 200:
            client.post(f"/api/jobs/{job_c.json()['job_id']}/cancel")
        live_codes = {client.get(f"/api/jobs/{job_a_id}").status_code for _ in range(40)}
        check("poller stays unlimited while a job is in flight", live_codes == {200},
              f"codes={live_codes}")

    # --- opt-out ---------------------------------------------------
    ratelimit.reset()
    os.environ["HNT_NO_RATELIMIT"] = "1"
    try:
        client = TestClient(app)
        codes = [client.post("/api/jobs", data={}).status_code
                 for _ in range(ratelimit.JOBS_IP_PER_MIN + 2)]
        check("HNT_NO_RATELIMIT=1 disables limiting", 429 not in codes, f"codes={codes}")
        check("no cookie is minted while disabled",
              not client.cookies.get(ratelimit.SESSION_COOKIE))
    finally:
        os.environ.pop("HNT_NO_RATELIMIT", None)
    ratelimit.reset()

except Exception as e:
    check("section 6 raised no exception", False, f"{type(e).__name__}: {e}")
finally:
    _os.chdir(_CWD_BEFORE_WEBAPP)
    check("working directory restored after importing the web app",
          Path.cwd() == Path(_CWD_BEFORE_WEBAPP))


# ═══════════════════════════════════════════════════════════════════
# Final report
# ═══════════════════════════════════════════════════════════════════
section("FINAL TEST REPORT")

total_tests = len(results)
passed_tests = sum(1 for r in results if r["status"] == "PASS")
failed_tests = total_tests - passed_tests

print(f"\n  Total  : {total_tests}")
print(f"  Passed : {passed_tests}")
print(f"  Failed : {failed_tests}")

if failed_tests:
    print("\n  Failed tests:")
    for r in results:
        if r["status"] == "FAIL":
            detail = f"  — {r['detail']}" if r["detail"] else ""
            print(f"    ✗ {r['test']}{detail}")

report_path = TEST_DIR / "test_report.json"
with open(report_path, "w", encoding="utf-8") as f:
    json.dump({
        "total": total_tests,
        "passed": passed_tests,
        "failed": failed_tests,
        "results": results,
    }, f, ensure_ascii=False, indent=2)
print(f"\n  Full report saved to {report_path}")

sys.exit(0 if failed_tests == 0 else 1)
