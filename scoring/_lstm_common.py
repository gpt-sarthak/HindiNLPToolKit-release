"""
scoring._lstm_common
====================
The LSTM language-model architecture shared by the Wiki-50M scorer
(``scoring/_lstm_wiki50m.py``) and the training script
(``training/lstm/lm.py`` defines the identical network).

Private support module: it defines no :class:`~scoring.base.Scorer` subclass,
so plugin discovery imports it harmlessly.  ``torch`` is imported only when the
class is built, so ``get_scorers()`` / ``/api/plugins`` stay cheap.
"""

from __future__ import annotations


def _build_model_class():
    """Define the LSTM architecture lazily (so importing this module is cheap).

    Embedding(vocab, 256) -> 2-layer LSTM(256, dropout 0.3) -> Linear(256, vocab).
    Must match the architecture the checkpoint was trained with, or
    ``load_state_dict`` will fail.
    """
    import torch.nn as nn

    class LSTMLanguageModel(nn.Module):
        def __init__(self, vocab_size, embed_size=256, hidden_size=256,
                     num_layers=2, dropout=0.3):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_size)
            self.lstm = nn.LSTM(
                embed_size, hidden_size, num_layers=num_layers,
                dropout=dropout, batch_first=True,
            )
            self.fc = nn.Linear(hidden_size, vocab_size)

        def forward(self, x, hidden=None):
            x = self.embedding(x)
            out, hidden = self.lstm(x, hidden)
            return self.fc(out), hidden

    return LSTMLanguageModel
