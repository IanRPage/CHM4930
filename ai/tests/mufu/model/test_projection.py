import pytest
import torch

from mufu.contract import D_MODEL, MODALITIES
from mufu.model.projection import ModalityProjection

# distinct widths so any mis-routed or shared layer fails on shape
IN_DIMS = {"smiles": 768, "fingerprint": 512, "graph": 128}


def make_embeddings(batch_size=4):
    return {m: torch.randn(batch_size, IN_DIMS[m]) for m in MODALITIES}


def test_output_shapes():
    batch_size = 4
    outputs = ModalityProjection(IN_DIMS)(make_embeddings(batch_size))

    assert tuple(outputs) == MODALITIES
    for m in MODALITIES:
        assert outputs[m].shape == (batch_size, D_MODEL)


def test_rejects_bad_in_dims_keys():
    with pytest.raises(ValueError, match="in_dims: missing keys"):
        ModalityProjection({"smiles": 768, "fingerprint": 512})


def test_rejects_bad_embedding_keys():
    projection = ModalityProjection(IN_DIMS)
    embeddings = make_embeddings()
    del embeddings["graph"]

    with pytest.raises(ValueError, match="embeddings: missing keys"):
        projection(embeddings)


def test_rejects_non_tensor_embedding():
    projection = ModalityProjection(IN_DIMS)
    embeddings = {**make_embeddings(), "smiles": ["CCO"] * 4}

    with pytest.raises(TypeError, match="smiles: expected a tensor"):
        projection(embeddings)


def test_rejects_wrong_width():
    projection = ModalityProjection(IN_DIMS)
    embeddings = {**make_embeddings(), "fingerprint": torch.randn(4, 256)}

    with pytest.raises(ValueError, match="fingerprint: expected shape"):
        projection(embeddings)


def test_rejects_batch_size_mismatch():
    projection = ModalityProjection(IN_DIMS)
    embeddings = {**make_embeddings(), "graph": torch.randn(3, IN_DIMS["graph"])}

    with pytest.raises(ValueError, match="graph: expected shape"):
        projection(embeddings)
