import re


class CoNLL:
    """
    Adapted from DS-PS-Tool-TACL/DS-PS-Conversion-Tool_UD/CoNLL2SSF/conll_api.py
    for HDTB UD CoNLL-U format.

    Changes vs original:
      - idRelationMapping skips comment lines (#), blank lines, multi-word tokens
        (ID contains '-') and empty nodes (ID contains '.').
      - getChunkId reads from word_list[9] (MISC column) instead of word_list[5]
        (FEATS), because HDTB stores ChunkId in MISC as 'ChunkId=NP2'.
      - getVoicetype still reads from word_list[5] (FEATS) — Voice=Act IS there.
    """

    def __init__(self, filename):
        fin = open(filename, 'r', encoding='UTF-8')
        f = fin.read()
        self.sentence_array = f.split("\n\n")
        fin.close()

    def idRelationMapping(self, sentence):
        index_chunk_dict = {}
        chunkid_relid_dict = {}
        root = ''
        voice = ''
        root_Id = 0
        for block in sentence.splitlines():
            word_list = block.split()
            # Skip comment lines, blank lines
            if not word_list or word_list[0].startswith('#'):
                continue
            # Skip multi-word tokens (3-4) and empty nodes (1.1)
            if '-' in word_list[0] or '.' in word_list[0]:
                continue
            # Need at least 10 columns (CoNLL-U standard)
            if len(word_list) < 10:
                continue

            id = word_list[0]
            chunkId = self.getChunkId(word_list[9])   # MISC column
            digitInChunk = re.sub("\D", "", chunkId)

            if word_list[7] == 'root':
                root = chunkId
                root_Id = word_list[0]
                voice = self.getVoicetype(word_list[5])  # FEATS column — Voice=Act
            elif ':' in word_list[7]:
                word_list[7] = word_list[7].replace(':', '_')

            if id not in index_chunk_dict:
                index_chunk_dict[id] = chunkId
            if chunkId not in chunkid_relid_dict:
                chunkid_relid_dict[chunkId] = {}
                chunkid_relid_dict[chunkId][word_list[6]] = word_list[7]
            else:
                if word_list[6] not in chunkid_relid_dict[chunkId]:
                    chunkid_relid_dict[chunkId][word_list[6]] = word_list[7]

        return index_chunk_dict, chunkid_relid_dict, root, voice, root_Id

    def getRelations(self, index_chunk_dict, chunkid_relid_dict):
        relations = {}
        for chunk, relidList in chunkid_relid_dict.items():
            if '0' not in relidList:
                for id, rel in relidList.items():
                    if id in index_chunk_dict and index_chunk_dict[id] != chunk:
                        if chunk not in relations.keys():
                            relations[chunk] = rel + ":" + index_chunk_dict[id]
        return relations

    def getChunkId(self, misc):
        """
        Extract ChunkId from MISC column (format: ChunkId=NP2|ChunkType=head|...).
        Returns the raw chunk label (e.g. 'NP', 'NP2', 'VGF', 'BLK').
        Returns 'none' if ChunkId is not present.
        """
        chunkId = 'none'
        for kv in misc.split('|'):
            if kv.startswith('ChunkId='):
                chunkId = kv.split('=', 1)[1]
                break
        return chunkId

    def getVoicetype(self, features):
        """Read Voice=Act/Pass from FEATS column (col 5)."""
        voice = ''
        for feature in features.split('|'):
            if 'Voice' in feature and '=' in feature:
                voice = feature.split('=')[1]
        return voice
