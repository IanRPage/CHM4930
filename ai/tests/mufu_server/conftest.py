import pandas as pd
import pytest
import torch
from torch import nn

from encoders.fingerprint_encoder import FingerprintEncoder
from encoders.gnn_encoder import GNNEncoder
from mufu.contract import ACTIVE_COLUMNS, PIC50_COLUMNS, TOX_COLUMNS
from mufu.data.dataset import MuFuDataset
from mufu.mc_dropout import fit_mc_dropout
from mufu.model import MuFu
from mufu_server.infer import Predictor

# a finite 95% conformal quantile needs at least 19 labels
SMILES = [
    *("C" * n + "O" for n in range(1, 13)),
    *("C" * n + "N" for n in range(1, 13)),
]


# tests/mufu/conftest.py's SmallSmilesEncoder can't be imported from here
class TinySmilesEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.out_dim = 8
        self.embedding = nn.Embedding(128, self.out_dim)

    def forward(self, smiles):
        return torch.stack(
            [
                self.embedding(torch.tensor([ord(c) for c in text])).mean(dim=0)
                for text in smiles
            ]
        )


@pytest.fixture
def labeled_dataset():
    frame = pd.DataFrame({"smiles": SMILES})
    for column in (*PIC50_COLUMNS, *ACTIVE_COLUMNS, *TOX_COLUMNS):
        frame[column] = float("nan")
    for j, column in enumerate(PIC50_COLUMNS):
        frame[column] = [5.0 + j + 0.5 * i for i in range(len(SMILES))]
    return MuFuDataset(frame)


@pytest.fixture
def predictor(labeled_dataset, tmp_path):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = MuFu(
            smiles_encoder=TinySmilesEncoder(),
            fingerprint_encoder=FingerprintEncoder(hidden_dims=(16,), dropout=0.3),
            graph_encoder=GNNEncoder(hidden_dims=(12,), dropout=0.3),
        ).eval()
        ckpt = tmp_path / "mufu_seed_0.pt"
        torch.save(model.state_dict(), ckpt)
        conformal = fit_mc_dropout(model, labeled_dataset, ckpt, n_samples=4)
    return Predictor(
        model=model,
        family_states=None,
        calibration=None,
        conformal=conformal,
        model_version="mufu_seed_0@test",
    )
