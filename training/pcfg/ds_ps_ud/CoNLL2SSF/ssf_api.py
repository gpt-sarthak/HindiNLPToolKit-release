import re

# Adapted from DS-PS-Tool-TACL/DS-PS-Conversion-Tool_UD/CoNLL2SSF/ssf_api.py
# Changes vs original:
#   - processSentence skips comment lines, blank lines, multi-word/empty tokens
#   - getChunkId reads from word_list[9] (MISC) instead of word_list[5] (FEATS)
#   - getFeatureList still reads from word_list[5] (FEATS) — no change needed there


class SSF:
    def __init__(self, sentence, relations, root, voice):
        self.sentence = sentence
        self.relations = relations
        self.rootChunk = root
        self.voice = voice

    def processSentence(self, sent_counter):
        id = 0
        internalId = 1
        prevChunk = ''
        currChunk = ''
        ssf_sentence = "<Sentence id='" + str(sent_counter) + "'>\n"
        for block in self.sentence.splitlines():
            word_list = block.split()
            # Skip comment lines, blank lines
            if not word_list or word_list[0].startswith('#'):
                continue
            # Skip multi-word tokens (3-4) and empty nodes (1.1)
            if '-' in word_list[0] or '.' in word_list[0]:
                continue
            if len(word_list) < 10:
                continue

            currChunk = self.getChunkId(word_list[9])   # MISC column
            digitInChunk = re.sub("\D", "", currChunk)
            featurelist = self.getFeatureList(word_list[5])   # FEATS column
            currChunkPOS = re.sub(r'\d+', '', currChunk)

            if prevChunk != currChunk:
                internalId = 1
                id += 1
                if id != 1:
                    ssf_sentence += "\t))\n"
                if currChunk != self.rootChunk:
                    drel = self.getDrel(currChunk)
                    ssf_sentence += (str(id) + "\t((\t" + currChunkPOS
                                     + "\t<fs name='" + currChunk + "' " + drel)
                else:
                    voicetype = self.getVoicetype()
                    ssf_sentence += (str(id) + "\t((\t" + currChunkPOS
                                     + "\t<fs name='" + currChunk + "' " + voicetype)
                ssf_sentence += (str(id) + "." + str(internalId) + "\t"
                                 + word_list[1] + "\t" + word_list[3]
                                 + "\t<fs af='" + word_list[1] + featurelist
                                 + "' name='" + word_list[1]
                                 + "' posn='" + word_list[0] + "0'>\n")
            else:
                internalId += 1
                ssf_sentence += (str(id) + "." + str(internalId) + "\t"
                                 + word_list[1] + "\t" + word_list[3]
                                 + "\t<fs af='" + word_list[1] + featurelist
                                 + "' name='" + word_list[1]
                                 + "' posn='" + word_list[0] + "0'>\n")
            prevChunk = currChunk
        return ssf_sentence

    def getChunkId(self, misc):
        """Read ChunkId from MISC column (ChunkId=NP2|ChunkType=head|...)."""
        for kv in misc.split('|'):
            if kv.startswith('ChunkId='):
                return kv.split('=', 1)[1]
        return 'none'

    def getDrel(self, chunkId):
        if chunkId in self.relations:
            return "drel='" + self.relations[chunkId] + "'>\n"
        return ">\n"

    def getVoicetype(self):
        return "voicetype='" + self.voice + "'>\n"

    def getFeatureList(self, features):
        """Build comma-separated feature values from FEATS column, excluding chunkId."""
        featurelist = ''
        for feature in features.split('|'):
            if feature == '_':
                continue
            if 'chunkId' in feature.lower():
                continue
            if '=' not in feature:
                continue
            featurelist += ',' + feature.split('=')[1]
        return featurelist
