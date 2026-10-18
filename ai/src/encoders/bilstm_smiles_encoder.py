"""Bidirectional LSTM encoder for molecular SMILES strings"""

import re
from collections import Counter
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence

PAD_TOKEN = "<PAD>"  # padding
UNK_TOKEN = (
    "<UNK>"  # unknown; compare to Ian's system ? (organic, metals, other i think?)
)

# match bracket atoms, two character elements, ring identifiers, and individual SMILES symbols
TOKEN_PATTERN = re.compile(r"\[[^\[\]]+\]|Br|Cl|%\d{2}|\*|.")


def tokenize_smiles(smiles: str) -> list[str]:
    # split SMILES string into tokens
    if not isinstance(smiles, str):
        raise TypeError("SMILES must be a string")
    if not smiles or any(char.isspace() for char in smiles):
        raise ValueError("SMILES must be non empty and contain no whitespace")

    return TOKEN_PATTERN.findall(smiles)


def build_vocabulary(smiles_list: Sequence[str]) -> dict[str, int]:
    # build a token vocab from training SMILES
    if len(smiles_list) == 0:
        raise ValueError("Cannot build a vocabulary from an empty dataset.")

    counts = Counter(
        token for smiles in smiles_list for token in tokenize_smiles(smiles)
    )

    vocabulary = {PAD_TOKEN: 0, UNK_TOKEN: 1}

    for token in sorted(counts):
        vocabulary[token] = len(vocabulary)

    return vocabulary


class biLSTMEncoder(nn.Module):
    """
    biLSTM encoder for SMILES strings

    Accepts a single SMILES string or a batch of strings and returns one fixed size molecular embedding
    per input molecule.

    Vocabulary is built from training SMILES only.

    """

    def __init__(
        self,
        vocab: dict[str, int],
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        if (
            len(vocab) != len(set(vocab.values()))
            or set(vocab.values()) != set(range(len(vocab)))
            or vocab.get(PAD_TOKEN) != 0
            or vocab.get(UNK_TOKEN) != 1
        ):
            raise ValueError(
                "Vocab must contain contiguous IDs with PAD = 0 and UNK = 1"
            )

        if embedding_dim < 1 or hidden_dim < 1 or num_layers < 1:
            raise ValueError("Embedding, hidden, and layer dimensions must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("Dropout must be between 0 (inclusive) and 1 (exclusive)")

        self.vocab = dict(vocab)
        self.out_dim = hidden_dim * 2
        self.embedding = nn.Embedding(
            num_embeddings=len(vocab),
            embedding_dim=embedding_dim,
            padding_idx=0,
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, smiles: str | Sequence[str]) -> torch.Tensor:
        # encode SMILES into a batch of molecular embeddings
        if isinstance(smiles, str):
            smiles = [smiles]
        elif not isinstance(smiles, Sequence):
            raise TypeError("Expected a SMILES string or a sequence of strings")

        if len(smiles) == 0:
            raise ValueError("SMILES batch cannot be empty")

        device = self.embedding.weight.device

        sequences = []
        for smi in smiles:
            tokens = tokenize_smiles(smi)
            token_ids = [self.vocab.get(token, 1) for token in tokens]
            sequences.append(torch.tensor(token_ids, dtype=torch.long, device=device))

        lengths = [len(sequence) for sequence in sequences]
        padded = pad_sequence(
            sequences,
            batch_first=True,
            padding_value=0,
        )

        embedded = self.dropout(self.embedding(padded))

        # to ensure model doesn't treat artificial padding as part of the molecule:
        packed = pack_padded_sequence(
            embedded,
            lengths,
            batch_first=True,
            enforce_sorted=False,
        )

        _, (hidden, _) = self.lstm(packed)

        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]

        combined = torch.cat([forward_hidden, backward_hidden], dim=1)

        return self.dropout(combined)
