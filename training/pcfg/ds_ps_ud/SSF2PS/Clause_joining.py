# -*- coding: UTF-8 -*-
# Adapted from DS-PS-Tool-TACL/DS-PS-Conversion-Tool_UD/SSF2PS/Clause_joining.py
# Changes vs original: string.replace(s, old, new) → s.replace(old, new) (Python 3)

from ssf_api import *
from Word_order import *
from Rule_Selection import *


class Join_clauses(object):
    def __init__(self, tree):
        self.tree  = tree
        self.check = Word_order(self.tree)

    def join_clauses(self, i_ChunkName, i_clause):
        clause    = i_clause
        ChunkName = i_ChunkName

        for chunk in self.tree.getChildren(ChunkName):
            ChunkName = chunk.getName()
            if chunk.getPos() in ('VGF', 'NULL__VGF', 'VGNN', 'VGNF', 'NULL__VGNN', 'NULL__VGNF'):
                draw_PS    = Rule_selection(self.tree, chunk)
                emb_clause = draw_PS.rule_selection()
                n_flag = emb_clause.split().count('flag')
                if n_flag != 0:
                    emb_clause = self.join_clauses(ChunkName, emb_clause)
                clause = clause.replace(' flag ' + str(self.check.order()[ChunkName]), emb_clause)
            else:
                clause = self.join_clauses(ChunkName, clause)

        return clause

    def ps_tree(self):
        PS_tree = ''
        for chunk in self.tree:
            if not chunk.getDrel():
                if not chunk.getDMrel():
                    ChunkName = chunk.getName()
                    PS_clause = Rule_selection(self.tree, chunk)
                    clause    = PS_clause.rule_selection()
                    PS_tree   = self.join_clauses(ChunkName, clause)
        return PS_tree
