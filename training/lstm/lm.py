"""
lm.py — the LSTM language-model architecture and its special tokens.

Identical to the network the scorer loads (scoring/_lstm_common.py
``_build_model_class``): Embedding(V, 256) -> 2-layer LSTM(256, dropout 0.3)
-> Linear(256, V), embeddings NOT tied. Changing any size here makes the
checkpoint unloadable by the scorer.
"""

from __future__ import annotations

import torch.nn as nn

PAD, UNK, SOS, EOS = "<PAD>", "<UNK>", "<SOS>", "<EOS>"


class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int = 256,
                 hidden_size: int = 256, num_layers: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=num_layers,
                            dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        return self.fc(out), hidden
