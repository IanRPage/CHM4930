import pytest
import torch

from smiles_encoder import SmilesEncoder


@pytest.fixture(scope="module")
def encoder():
    model = SmilesEncoder(freeze=True)
    model.eval()
    return model


def test_output_shape(encoder):
    smiles = [
        "CCO",
        "CC(=O)O",
        "c1ccccc1",
    ]

    with torch.no_grad():
        output = encoder(smiles)

    assert output.shape == (3, encoder.out_dim)


def test_output_is_finite(encoder):
    with torch.no_grad():
        output = encoder(["CCO", "CCN"])

    assert torch.isfinite(output).all()


def test_empty_batch_rejected(encoder):
    with pytest.raises(ValueError):
        encoder([])


def test_single_string_rejected(encoder):
    with pytest.raises(TypeError):
        encoder("CCO")


def test_frozen_by_default(encoder):
    assert all(
        not parameter.requires_grad
        for parameter in encoder.model.parameters()
    )