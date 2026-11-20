# -*- coding: UTF-8 -*-
# Adapted from DS-PS-Tool-TACL/DS-PS-Conversion-Tool_UD/SSF2PS/ssf_api.py
# Changes vs original:
#   - open(filename) → open(filename, encoding='utf-8') throughout

import re

DELIM_NODE      = '\t'
DELIM_FS        = ' '
DELIM_KEY       = '='
DELIM_DREL      = ':'
DELIM_AF        = ','

KEY_DREL        = 'drel'
KEY_DMREL       = 'dmrel'
KEY_PBREL       = 'pbrel'
KEY_PBMREL      = 'pbmrel'
KEY_PBROLE      = 'pbrole'
KEY_AF          = 'af'
KEY_VOICETYPE   = 'voicetype'

REG_FS          = re.compile('([<>\s])')

POS_VERB        = 'VG'
POS_FINITE_VERB = 'VGF'
POS_MAIN_VERB   = 'VM'
POS_CONJUNCT    = 'CCP'
POS_NP          = 'NP'
POS_NOUN        = 'NN'

BEGIN_DOCUMENT  = '<document'
BEGIN_HEAD      = '<head'
BEGIN_BODY      = '<body'
BEGIN_TABLE     = '<tb'
BEGIN_TEXT      = '<text'
BEGIN_SENTENCE  = '<Sentence'
BEGIN_SSF_CHUNK = '0\t((\tSSF'
BEGIN_FS        = '<fs'

END_DOCUMENT    = '</document>'
END_HEAD        = '</head>'
END_BODY        = '</body>'
END_TABLE       = '</tb>'
END_TEXT        = '</text>'
END_SENTENCE    = '</Sentence>'
END_CHUNK       = '\t))'
END_FS          = '>'

ATTR_NAME       = 'name'


class SSF:
    def __init__(self, filename):
        fin = open(filename, encoding='utf-8')
        self.str_document = self._getText(fin, BEGIN_DOCUMENT)
        self.str_head     = self._getText(fin, BEGIN_HEAD, END_HEAD)
        self.str_body     = self._getText(fin, BEGIN_BODY)
        self.ls_table     = list()

        tbText = self._getText(fin, BEGIN_TABLE, END_TABLE)
        while tbText:
            table = Table(tbText)
            self.ls_table.append(table)
            tbText = self._getText(fin, BEGIN_TABLE, END_TABLE)

        fin.close()

    def _getText(self, fin, start, end=None):
        for text in fin:
            if text.strip(): break

        if not text.startswith(start): return None
        if not end                   : return text.strip()

        for line in fin:
            text += line
            if line.startswith(end): return text.strip()

        return None

    def printFile(self, filename):
        fout = open(filename, 'w', encoding='utf-8')
        fout.write(self.str_document + '\n')
        fout.write(self.str_head     + '\n')
        fout.write(self.str_body     + '\n')
        for table in self.ls_table:
            fout.write(str(table) + '\n')
        fout.write(END_BODY     + '\n')
        fout.write(END_DOCUMENT + '\n')
        fout.close()

    def getTrees(self):
        ls = list()
        for table in self.ls_table:
            ls.extend(table.getTrees())
        return ls


class Table:
    def __init__(self, tbText):
        lsText = tbText.split('\n')
        self.str_prefix = '\n'.join(self._getList(lsText, BEGIN_TABLE, BEGIN_TEXT))
        self.ls_tree    = list()
        lsTree = self._getList(lsText, BEGIN_SENTENCE, END_SENTENCE)
        while lsTree:
            self.ls_tree.append(Tree(lsTree))
            lsTree = self._getList(lsText, BEGIN_SENTENCE, END_SENTENCE)

    def _getList(self, lsText, start, end):
        if not lsText[0]: del lsText[0]
        if not lsText[0].startswith(start): return None
        ls = list()
        for n, text in enumerate(lsText):
            ls.append(text)
            if text.startswith(end):
                del lsText[:n+1]
                return ls
        return None

    def __str__(self):
        ls = list()
        ls.append(self.str_prefix)
        for tree in self.ls_tree:
            ls.append(str(tree) + '\n')
        ls.append(END_TEXT)
        ls.append(END_TABLE)
        return '\n'.join(ls)

    def getTrees(self):
        return self.ls_tree


class Tree(list):
    def __init__(self, lsTree):
        self.str_id = lsTree[0]
        del lsTree[0], lsTree[-1]

        if lsTree[0].startswith(BEGIN_SSF_CHUNK):
            self.is_ssfChunk = True
            del lsTree[0], lsTree[-1]
        else:
            self.is_ssfChunk = False

        chunk = self._getChunk(lsTree)
        while chunk:
            self.append(chunk)
            chunk = self._getChunk(lsTree)

    def _getChunk(self, lsTree):
        ls = list()
        for n, line in enumerate(lsTree):
            ls.append(line)
            if line.startswith(END_CHUNK):
                del lsTree[:n+1]
                return Chunk(ls)
        return None

    def __str__(self):
        ls = list()
        ls.append(self.str_id)
        if self.is_ssfChunk: ls.append(BEGIN_SSF_CHUNK)
        i = 1
        for chunk in self:
            if not chunk.isNull():
                ls.append(chunk.toStr(i))
                i += 1
        if self.is_ssfChunk: ls.append(END_CHUNK)
        ls.append(END_SENTENCE)
        return '\n'.join(ls)

    def getChunk(self, name):
        for chunk in self:
            if chunk.nameEquals(name): return chunk
        return None

    def existChild(self, drel, headId):
        for chunk in self:
            if chunk.isChild(drel, headId): return True
        return False

    def getChild(self, drel, headId):
        for chunk in self:
            if chunk.isChild(drel, headId): return chunk
        return None

    def getChildren(self, headId):
        ls = list()
        for chunk in self:
            if chunk.isChild(None, headId):
                ls.append(chunk)
        return ls

    def getDepChunks(self, headId):
        ls = list()
        for chunk in self:
            drel = chunk[0].getFs(KEY_DREL)
            if drel and drel[1] == headId:
                ls.append(chunk)
        return ls

    def getFirstDescendent(self, headId):
        for i, subchunk in enumerate(self):
            if subchunk.isChild(None, headId):
                return self.getFirstDescendent(subchunk.getName())
            elif subchunk.getName() == headId:
                return (i, subchunk)
        return (-1, None)

    def insertFirstChild(self, headId, node):
        for i, chunk in enumerate(self):
            if chunk.isChild(None, headId) or chunk.getName() == headId:
                self.insert(i, node)
                break

    def insertLastChild(self, headId, node):
        for i, chunk in enumerate(self):
            if chunk.getName() == headId:
                self.insert(i, node)
                break

    def isCycle(self, name):
        chunk = self.getChunk(name)
        rel   = chunk.getRel()
        while rel and len(rel) >= 2:
            chunk = self.getChunk(rel[1])
            if chunk.getName() == name: return True
            rel = chunk.getRel()
        return False

    def generateTokenIDs(self):
        tokId = 0
        for chunk in self:
            for i, node in enumerate(chunk[1:]):
                node.tokenId = tokId + i
            tokId += len(chunk) - 1


class Chunk(list):
    def __init__(self, lsChunk):
        self.isNone = False
        for line in lsChunk[:-1]:
            ls   = line.split(DELIM_NODE)
            word = ls[1]
            pos  = ls[2]
            fs   = ls[3] if len(ls) > 3 else None
            self.append(Node(word, pos, fs))

    def toStr(self, chunkId):
        ls = list()
        ls.append(str(chunkId) + DELIM_NODE + str(self[0]))
        for i in range(1, len(self)):
            ls.append(str(chunkId) + '.' + str(i) + DELIM_NODE + str(self[i]))
        ls.append(END_CHUNK)
        return '\n'.join(ls)

    def isVerb(self):       return self[0].posStarts(POS_VERB)
    def isFiniteVerb(self): return self[0].posEquals(POS_FINITE_VERB)
    def isConjunct(self):   return self[0].posEquals(POS_CONJUNCT)
    def getDrel(self):      return self.getFs(KEY_DREL)
    def getDMrel(self):
        if not self[0].dic_fs: return None
        return self[0].dic_fs.get(KEY_DMREL)

    def getRel(self):
        rel = self.getDrel()
        if rel: return rel
        return self.getDMrel()

    def getFs(self, fs):
        if not self[0].dic_fs: return None
        return self[0].dic_fs.get(fs)

    def isDrel(self, label):
        drel = self.getFs(KEY_DREL)
        return drel and drel[0] == label

    def isChild(self, drel, headId):
        minfo = self.getDMrel()
        if minfo:
            if len(minfo) < 2: return False
            if not drel: return minfo[1] == headId
            return minfo[0] == drel and minfo[1] == headId
        dinfo = self.getDrel()
        if not dinfo or len(dinfo) < 2: return False
        if not drel: return dinfo[1] == headId
        return dinfo[0] == drel and dinfo[1] == headId

    def getName(self): return self[0].getName()
    def getPos(self):  return self[0].getPos()
    def nameEquals(self, name): return self[0].nameEquals(name)
    def nullify(self): self.isNone = True
    def isNull(self):  return self.isNone

    def getMainVerb(self):
        for node in self[1:]:
            if node.posEquals(POS_MAIN_VERB) or node.posEquals('VMC'):
                return node
        return None


class Node:
    def __init__(self, word, pos, fs):
        self.str_word = word
        self.str_pos  = pos
        self.dic_fs   = self._getFs(fs)
        self.tokenId  = -1

    def _getFs(self, fs):
        if not fs: return None
        lsFs = REG_FS.split(fs)
        dic  = dict()
        for item in lsFs:
            if DELIM_KEY not in item: continue
            kv  = item.split(DELIM_KEY)
            key = kv[0]
            val = kv[1]
            if val and val[0] == '\'' and val[-1] == '\'':
                val = val[1:-1]
            if   key == KEY_DREL  : dic[key] = val.split(DELIM_DREL)
            elif key == KEY_DMREL : dic[key] = val.split(DELIM_DREL)
            elif key == KEY_PBREL : dic[key] = val.split(DELIM_DREL)
            elif key == KEY_PBMREL: dic[key] = val.split(DELIM_DREL)
            elif key == KEY_AF    : dic[key] = val.split(DELIM_AF)
            else                  : dic[key] = val
        return dic

    def __str__(self):
        ls = list()
        ls.append(self.str_word)
        ls.append(self.str_pos)
        if self.dic_fs: ls.append(self._strFs())
        return DELIM_NODE.join(ls)

    def _strFs(self):
        ls = list()
        ls.append(BEGIN_FS)
        if KEY_AF in self.dic_fs:
            ls.append(KEY_AF + DELIM_KEY + '\'' + DELIM_AF.join(self.dic_fs[KEY_AF]) + '\'')
        for key in self.dic_fs:
            if key == KEY_AF: continue
            val = self.dic_fs[key]
            if   key == KEY_DREL  : val = DELIM_DREL.join(val)
            elif key == KEY_DMREL : val = DELIM_DREL.join(val)
            elif key == KEY_PBREL : val = DELIM_DREL.join(val)
            elif key == KEY_PBMREL: val = DELIM_DREL.join(val)
            elif key == KEY_AF    : val = DELIM_AF  .join(val)
            ls.append(key + DELIM_KEY + '\'' + val + '\'')
        return DELIM_FS.join(ls) + END_FS

    def wordEnds(self, word):   return self.str_word.endswith(word)
    def wordEquals(self, word): return self.str_word == word
    def posStarts(self, pos):   return self.str_pos.startswith(pos)
    def posEquals(self, pos):   return self.str_pos == pos

    def getName(self):
        if not self.dic_fs: return None
        return self.dic_fs.get(ATTR_NAME)

    def getPos(self):   return self.str_pos

    def nameEquals(self, name):
        curr = self.getName()
        if curr: return curr == name
        return False

    def getFs(self, key):
        if not self.dic_fs       : return None
        return self.dic_fs.get(key)

    def getAF(self):
        if not self.dic_fs: return None
        return self.dic_fs.get(KEY_AF)

    def getVoiceType(self):
        if not self.dic_fs: return None
        return self.dic_fs.get(KEY_VOICETYPE)

    def getWord(self):  return self.str_word

    def getLemma(self):
        af = self.getAF()
        if af and af[0]: return af[0]
        return self.str_word

    def addFs(self, key, value):
        self.dic_fs[key] = value

    def getPBrole(self):
        return self.getFs(KEY_PBROLE)
