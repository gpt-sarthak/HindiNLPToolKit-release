# -*- coding: UTF-8 -*-
# Adapted from DS-PS-Tool-TACL/DS-PS-Conversion-Tool_UD/SSF2PS/Rule_Selection.py
# Changes vs original:
#   - string.replace(s, old, new) → s.replace(old, new) (Python 3)
#   - dict.items() + dict.items() → explicit dict.update() (the original code
#     already had the correct Python 3 version on the next two lines; the
#     Python 2 line is simply removed)

from ssf_api import *
from Phrases import *
from operator import itemgetter


class Rule_selection(object):
    def __init__(self, tree, pred):
        self.tree = tree
        self.pred = pred

    def rule_selection(self):
        get       = Fragments(self.tree, self.pred)
        predicate = {get.predicate()[0]: get.predicate()[2]}
        Pred_P    = get.predicate()[1]
        Pred_Pos  = get.predicate()[3]

        dict_frag     = get.fragments()
        all_dict_frag = dict(dict_frag)
        all_dict_frag.update(predicate)

        SYM = ''
        for chunk in self.tree.getChildren(self.pred.getName()):
            if chunk.isChild('punct', self.pred.getName()):
                sym_term = ''
                for node in chunk:
                    if not node.getLemma() == '((':
                        sym_term = sym_term + ' ' + node.getName()
                SYM = '[.SYM' + sym_term + ' ]'

        S    = ''
        Arg0 = ''
        all_frag         = ''
        ordered_fragments = sorted(all_dict_frag.items(), key=itemgetter(1))

        if Pred_Pos in ('VGF', 'NULL__VGF'):
            if not (ordered_fragments[0][0].split()[0] in ('[.VGF', '[.NULL__VGF')):
                for frag in ordered_fragments:
                    if frag[0] == ordered_fragments[0][0]:
                        Arg0 = frag[0]
                    else:
                        all_frag = all_frag + frag[0]
                Arg0     = Arg0.replace(' Label ', '')
                all_frag = all_frag.replace(' Label ', '')
                if SYM:
                    S = ' [.S' + Arg0 + ' [.VP' + all_frag + ' ]' + ' ' + SYM + ' ]'
                else:
                    S = ' [.S' + Arg0 + ' [.VP' + all_frag + ' ]' + ' ]'
            else:
                for frag in ordered_fragments:
                    all_frag = all_frag + frag[0]
                if SYM:
                    S = ' [.S' + ' [.VP' + all_frag + ' ]' + ' ' + SYM + ' ]'
                else:
                    S = ' [.S' + ' [.VP' + all_frag + ' ]' + ' ]'

        elif Pred_Pos in ('VGNN', 'VGNF'):
            if not (ordered_fragments[0][0].split()[0] in ('[.VGNN', '[.VGNF')):
                for frag in ordered_fragments:
                    if frag[0] == ordered_fragments[0][0]:
                        Arg0 = frag[0]
                    else:
                        all_frag = all_frag + frag[0]
                Arg0     = Arg0.replace(' Label ', '')
                all_frag = all_frag.replace(' Label ', '')
                suffix   = Pred_Pos[2:4]
                if SYM:
                    S = ' [.S-' + suffix + Arg0 + ' [.VP' + all_frag + ' ]' + ' ' + SYM + ' ]'
                else:
                    S = ' [.S-' + suffix + Arg0 + ' [.VP' + all_frag + ' ]' + ' ]'
            else:
                for frag in ordered_fragments:
                    all_frag = all_frag + frag[0]
                suffix = Pred_Pos[2:4]
                if SYM:
                    S = ' [.S-' + suffix + ' [.VP' + all_frag + ' ]' + ' ' + SYM + ' ]'
                else:
                    S = ' [.S-' + suffix + ' [.VP' + all_frag + ' ]' + ' ]'

        elif Pred_Pos in ('NULL__VGNF', 'NULL__VGNN'):
            if not (ordered_fragments[0][0].split()[0] in ('[.NULL__VGNN', '[.NULL__VGNF')):
                for frag in ordered_fragments:
                    if frag[0] == ordered_fragments[0][0]:
                        Arg0 = frag[0]
                    else:
                        all_frag = all_frag + frag[0]
                Arg0     = Arg0.replace(' Label ', '')
                all_frag = all_frag.replace(' Label ', '')
                suffix   = Pred_Pos[8:10]
                if SYM:
                    S = ' [.S-' + suffix + Arg0 + ' [.VP' + all_frag + ' ]' + ' ' + SYM + ' ]'
                else:
                    S = ' [.S-' + suffix + Arg0 + ' [.VP' + all_frag + ' ]' + ' ]'
            else:
                for frag in ordered_fragments:
                    all_frag = all_frag + frag[0]
                suffix = Pred_Pos[8:10]
                if SYM:
                    S = ' [.S-' + suffix + ' [.VP' + all_frag + ' ]' + ' ' + SYM + ' ]'
                else:
                    S = ' [.S-' + suffix + ' [.VP' + all_frag + ' ]' + ' ]'

        else:
            pred_terminal_nv = {get.predicate()[4]: get.predicate()[2]}
            all_dict_frag_nv = dict(dict_frag)
            all_dict_frag_nv.update(pred_terminal_nv)
            ordered_frag_nv = sorted(all_dict_frag_nv.items(), key=itemgetter(1))
            for frag in ordered_frag_nv:
                all_frag = all_frag + frag[0]
            all_frag = all_frag.replace(' Label ', '')
            if SYM:
                S = ' [.' + Pred_Pos + all_frag + ' ' + SYM + ' ]'
            else:
                S = ' [.' + Pred_Pos + all_frag + ' ]'

        return S
