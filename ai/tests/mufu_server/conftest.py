import shutil
from pathlib import Path

import pandas as pd
import pytest
import torch
from torch import nn

from mufu import ensemble
from mufu.contract import ACTIVE_COLUMNS, PIC50_COLUMNS, TOX_COLUMNS
from mufu.data.dataset import MuFuDataset
from mufu.experiment import make_fit_predict
from mufu.train import TrainConfig
from mufu_server.config import Settings
from mufu_server.infer import load_predictor

# a finite 95% conformal quantile needs at least 19 labels
SMILES = [
    *("C" * n + "O" for n in range(1, 13)),
    *("C" * n + "N" for n in range(1, 13)),
]
SEEDS = (0, 1, 2)
CFG = TrainConfig(
    batch_size=4,
    epochs=1,
    patience=1,
    gnn_hidden_dims=(32, 16),
    calibration="platt",
    device="cpu",
)


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


@pytest.fixture(scope="module")
def labeled_dataset():
    frame = pd.DataFrame({"smiles": SMILES})
    n = len(SMILES)
    for j, column in enumerate(PIC50_COLUMNS):
        frame[column] = [5.0 + j + 0.5 * i for i in range(n)]
    for j, column in enumerate((*ACTIVE_COLUMNS, *TOX_COLUMNS)):
        frame[column] = [float((i + j) % 2) for i in range(n)]
    return MuFuDataset(frame)


@pytest.fixture(scope="module")
def encoder():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        return TinySmilesEncoder().requires_grad_(False)


# trained once per module; tests that change files use `ensemble_copy`
@pytest.fixture(scope="module")
def ensemble_dir(labeled_dataset, encoder, tmp_path_factory):
    """
    Three tiny trained seeds and their ensemble file, as drivers/fit_ensemble.py saves
    them.
    """
    tmp_path = tmp_path_factory.mktemp("ensemble")
    data_path = tmp_path / "combined.csv"
    data_path.write_text("smiles\nCCO\n")
    fit_predict = make_fit_predict(
        {split: labeled_dataset for split in ("train", "val", "test")},
        encoder,
        CFG,
        run_dir=tmp_path,
        model_name="mufu",
        data_path=data_path,
    )
    ckpt_dir = tmp_path / "checkpoints"
    with torch.random.fork_rng(devices=[]):
        for seed in SEEDS:
            torch.manual_seed(seed)
            fit_predict(None, seed)
        paths = ensemble.member_paths(ckpt_dir, "mufu", SEEDS)
        params = ensemble.fit_ensemble_conformal(
            ensemble.load_members(paths, encoder),
            labeled_dataset,
            n_samples=4,
            coverages=(0.9, 0.95),
            batch_size=4,
        )
    ensemble.save_ensemble(params, ckpt_dir, "mufu")
    return ckpt_dir


def settings(checkpoint_dir: Path, **overrides) -> Settings:
    return Settings(checkpoint_dir=checkpoint_dir, api_key="test-key", **overrides)


@pytest.fixture
def ensemble_copy(ensemble_dir, tmp_path):
    return Path(shutil.copytree(ensemble_dir, tmp_path / "checkpoints"))


@pytest.fixture(scope="module")
def predictor(ensemble_dir, encoder):
    return load_predictor(settings(ensemble_dir), smiles_encoder=encoder, seeds=SEEDS)
