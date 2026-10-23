import pandas as pd
import pytest
import torch
from torch import nn

from encoders.fingerprint_encoder import FingerprintEncoder
from encoders.gnn_encoder import GNNEncoder
from mufu.contract import ACTIVE_COLUMNS, PIC50_COLUMNS, TOX_COLUMNS
from mufu.dataset import MuFuDataset
from mufu.model import MuFu

SMILES = ["CCO", "CCN", "CCC", "c1ccccc1", "CC(=O)O", "C", "CO", "CN"]
DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
    ),
]


class SmallSmilesEncoder(nn.Module):
    """Small character encoder so model behavior tests do not need downloads."""

    def __init__(self):
        super().__init__()
        self.out_dim = 8
        self.embedding = nn.Embedding(128, self.out_dim)

    def forward(self, smiles):
        embeddings = []
        for text in smiles:
            tokens = torch.tensor(
                [ord(char) for char in text], device=self.embedding.weight.device
            )
            embeddings.append(self.embedding(tokens).mean(dim=0))
        return torch.stack(embeddings)


def unlabeled_frame():
    frame = pd.DataFrame({"smiles": SMILES})
    for column in (*PIC50_COLUMNS, *ACTIVE_COLUMNS, *TOX_COLUMNS):
        frame[column] = float("nan")
    return frame


@pytest.fixture
def dataset():
    return MuFuDataset(unlabeled_frame())


@pytest.fixture
def labeled_dataset():
    frame = unlabeled_frame()
    bio_rows, tox_rows = frame.index[:4], frame.index[4:]
    for j, column in enumerate(PIC50_COLUMNS):
        frame.loc[bio_rows, column] = [5.0 + j + 0.5 * i for i in range(4)]
    for j, column in enumerate(ACTIVE_COLUMNS):
        frame.loc[bio_rows, column] = [float((i + j) % 2) for i in range(4)]
    for j, column in enumerate(TOX_COLUMNS):
        frame.loc[tox_rows, column] = [float((i + j) % 2) for i in range(4)]
    frame.loc[1, [PIC50_COLUMNS[-1], ACTIVE_COLUMNS[-1]]] = float("nan")
    return MuFuDataset(frame)


@pytest.fixture
def model():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        return MuFu(
            smiles_encoder=SmallSmilesEncoder(),
            fingerprint_encoder=FingerprintEncoder(hidden_dims=(16,), dropout=0.0),
            graph_encoder=GNNEncoder(hidden_dims=(12,), dropout=0.0),
        )
