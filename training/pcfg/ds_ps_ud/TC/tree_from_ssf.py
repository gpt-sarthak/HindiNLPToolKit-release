#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Adapted from DS-PS-Tool-TACL/DS-PS-Conversion-Tool_UD/TC/tree_from_ssf.py
# Changes vs original:
#   - Removed GUI imports (CanvasFrame, TreeWidget) — headless use only
#   - Removed deprecated 'bracket_parse', 'treetransforms' imports
#   - Removed sys.stdout.close() at end
#   - All file opens use explicit encoding='utf-8'

from nltk.tree import Tree
from copy import deepcopy
import re
import sys
import os

LR_BRACES = ['(', ')']


def remove_unary(tree):
    if isinstance(tree, Tree) and len(tree) == 1:
        nodeList = [tree[0]]
    else:
        nodeList = [tree]
    while nodeList:
        node = nodeList.pop()
        if isinstance(node, Tree):
            if len(node) == 1 and isinstance(node[0], Tree):
                node.set_label(node[0].label())
                node[0:] = [child for child in node[0]]
                nodeList.append(node)
            else:
                for child in node:
                    nodeList.append(child)


def remove_null(tree):
    for ind, leaf in reversed(list(enumerate(tree.leaves()))):
        if leaf.endswith("**") or leaf.endswith("NULL"):
            postn = tree.leaf_treeposition(ind)
            parentpos = postn[:-1]
            while parentpos and len(tree[parentpos]) == 1:
                postn = parentpos
                parentpos = postn[:-1]
            del tree[postn]


def add_extra_S(treestr):
    """Wrap in (.S ...) if root is not already .S."""
    tree = Tree.fromstring(treestr)
    root = tree.label()
    if root != '.S':
        return "(.S " + treestr + ")"
    return treestr


def replace_allbraces(filename, out_filename):
    """
    Convert DS-PS bracket notation to Penn Treebank format.
    PS output uses [.LABEL ...] notation; PTB uses (LABEL ...).
    Steps:
      1. Protect any existing ( ) by replacing with LB / RB
      2. Replace [ with ( and ] with )
    """
    with open(filename, 'r', encoding="utf-8") as infile:
        contents = infile.read()
        if any(x in contents for x in LR_BRACES):
            contents = contents.replace('(', 'LB')
            contents = contents.replace(')', 'RB')
        contents = contents.replace('[', '(')
        contents = contents.replace(']', ')')
    with open(out_filename, 'w', encoding="utf-8") as outfile:
        outfile.write(contents)


def strip_dot_prefix(tree_str):
    """
    Remove leading '.' from all nonterminal labels produced by DS-PS.
    Converts '.S' → 'S', '.NP-SUBJ' → 'NP-SUBJ', '.VP' → 'VP', etc.
    Leaf tokens (words) are not affected.
    """
    return re.sub(r'\(\.([A-Z])', r'(\1', tree_str)


input_dir           = "data/PS"
intermediate_fname  = "data/PS-Braces/{}.txt"
output_fname        = "data/TC/tree_{}.txt"

for filepath, dirs, files in os.walk(input_dir):
    for file in files:
        in_fname  = os.path.splitext(file)[0]
        in_fpath  = os.path.join(filepath, file)
        mid_fname = intermediate_fname.format(in_fname)
        replace_allbraces(in_fpath, mid_fname)

print("Replaced all braces")

print("Starting tree processing")
for filepath, dirs, files in os.walk('data/PS-Braces'):
    for file in files:
        fname    = os.path.splitext(file)[0]
        in_fpath = os.path.join(filepath, file)
        p = open(in_fpath, 'r', encoding="utf-8")
        q = open(output_fname.format(fname), "w+", encoding="utf-8")
        for line in p:
            if line == "\n":
                continue
            newtree = add_extra_S(line.strip())
            try:
                tree = Tree.fromstring(newtree)
                remove_null(tree)
                remove_unary(tree)
                parse_string = ' '.join(str(tree).split())
                # Strip leading dots from nonterminal labels (.S → S, .NP-SUBJ → NP-SUBJ)
                parse_string = strip_dot_prefix(parse_string)
                q.write(parse_string + '\n')
            except Exception as e:
                sys.stderr.write(f"WARNING: tree parse failed: {e}\n")
                continue
        p.close()
        q.close()
