import pandas as pd
import pytest
import torch
from torch import nn

from encoders.fingerprint_encoder import FingerprintEncoder
from encoders.gnn_encoder import GNNEncoder
from mufu.contract import ACTIVE_COLUMNS, MODALITIES, PIC50_COLUMNS, TOX_COLUMNS
from mufu.dataset import MuFuDataset, collate
from mufu.modality_dropout import sample_modality_mask
from mufu.model import MuFu


class SmallSmilesEncoder(nn.Module):
    """Small character encoder so model behavior tests do not need downloads."""

    def __init__(self):
        super().__init__()
        self.out_dim = 8
        self.embedding = nn.Embedding(128, self.out_dim)

    def forward(self, smiles):
        embeddings = []
        for text in smiles:
            tokens = torch.tensor([ord(char) for char in text])
            embeddings.append(self.embedding(tokens).mean(dim=0))
        return torch.stack(embeddings)


@pytest.fixture
def dataset():
    frame = pd.DataFrame(
        {"smiles": ["CCO", "CCN", "CCC", "c1ccccc1", "CC(=O)O", "C", "CO", "CN"]}
    )
    for column in (*PIC50_COLUMNS, *ACTIVE_COLUMNS, *TOX_COLUMNS):
        frame[column] = float("nan")
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


def test_eval_keeps_all_modalities_without_sampling(model, dataset):
    batch = collate([dataset[i] for i in range(len(dataset))])
    model.eval()
    rng_before = torch.get_rng_state()
    with torch.no_grad():
        predictions = model(batch)
        repeated = model(batch)

    assert torch.equal(torch.get_rng_state(), rng_before)
    for name in predictions:
        assert torch.equal(predictions[name], repeated[name])


def test_training_masks_fingerprint_gradients_per_molecule(model, dataset, monkeypatch):
    # Keep this test's partial masks independent of the production default.
    def sample_partial_mask(batch_size, *, device):
        return sample_modality_mask(batch_size, p=1.0, device=device)

    monkeypatch.setattr("mufu.model.sample_modality_mask", sample_partial_mask)
    batch = collate([dataset[i] for i in range(len(dataset))])
    batch["fingerprint"].requires_grad_()
    model.train()

    # The test encoders have no dropout, so sampling is the first RNG operation.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        expected_mask = sample_modality_mask(len(dataset), p=1.0)
        torch.manual_seed(17)
        predictions = model(batch)
    predictions["tox_logits"].square().mean().backward()

    retained = expected_mask[:, MODALITIES.index("fingerprint")]
    gradient_norms = batch["fingerprint"].grad.norm(dim=1)
    assert retained.any() and (~retained).any()
    assert torch.equal(
        gradient_norms[~retained], torch.zeros_like(gradient_norms[~retained])
    )
    assert (gradient_norms[retained] > 0).all()


def test_single_molecule_eval_matches_batch_prediction(model, dataset):
    batch = collate([dataset[i] for i in range(len(dataset))])
    single = collate([dataset[2]])
    model.eval()
    with torch.no_grad():
        batch_predictions = model(batch)
        single_predictions = model(single)

    for name in batch_predictions:
        torch.testing.assert_close(
            single_predictions[name], batch_predictions[name][2:3], rtol=1e-5, atol=1e-6
        )
