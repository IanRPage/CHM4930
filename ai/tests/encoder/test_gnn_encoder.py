import pytest
import torch
from rdkit import Chem
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.nn import global_add_pool

from encoders.gnn_encoder import GNNEncoder
from featurize.graph import EDGE_FEATURE_DIM, NODE_FEATURE_DIM, mol_to_graph


def graph_batch(smiles=None):
    """Create a small batch of molecular graphs for testing."""
    if smiles is None:
        smiles = ["CCO", "CC(=O)O", "c1ccccc1", "CCN"]

    graphs = []

    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        graphs.append(mol_to_graph(mol))

    return Batch.from_data_list(graphs)


@pytest.mark.parametrize(
    "hidden_dims",
    [(128,), (256, 128), (256, 256), (512, 256), (256, 128, 64)],
)
def test_output_shape(hidden_dims):
    batch = graph_batch()

    enc = GNNEncoder(hidden_dims=hidden_dims)
    out = enc(batch)

    assert enc.out_dim == hidden_dims[-1]
    assert out.shape == (4, hidden_dims[-1])


def test_node_and_edge_feature_shapes():
    batch = graph_batch()

    assert batch.x.ndim == 2
    assert batch.x.shape[1] == NODE_FEATURE_DIM

    assert batch.edge_attr.ndim == 2
    assert batch.edge_attr.shape[1] == EDGE_FEATURE_DIM

    assert batch.edge_index.ndim == 2
    assert batch.edge_index.shape[0] == 2


def test_returns_float32():
    batch = graph_batch()

    enc = GNNEncoder()
    out = enc(batch)

    assert out.dtype == torch.float32


def test_eval_is_deterministic():
    enc = GNNEncoder(dropout=0.5).eval()
    batch = graph_batch()

    out1 = enc(batch)
    out2 = enc(batch)

    assert torch.equal(out1, out2)


def test_gradients_reach_every_parameter():
    enc = GNNEncoder()
    batch = graph_batch()

    enc(batch).sum().backward()

    for name, param in enc.named_parameters():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name


def test_output_is_finite():
    enc = GNNEncoder().eval()
    batch = graph_batch()

    out = enc(batch)

    assert torch.isfinite(out).all()


def test_handles_molecules_with_different_numbers_of_atoms():
    batch = graph_batch(["C", "CC", "CCO", "c1ccccc1", "CCCCCCCCCC"])

    enc = GNNEncoder(hidden_dims=(128, 64)).eval()
    out = enc(batch)

    assert out.shape == (5, 64)


def test_handles_molecule_with_no_bonds():
    mol = Chem.MolFromSmiles("C")
    graph = mol_to_graph(mol)

    assert graph.edge_index.shape == (2, 0)
    assert graph.edge_attr.shape == (0, EDGE_FEATURE_DIM)

    batch = Batch.from_data_list([graph])

    enc = GNNEncoder(hidden_dims=(64, 32)).eval()

    out = enc(batch)

    assert out.shape == (1, 32)
    assert torch.isfinite(out).all()


def test_uses_batch_norm():
    enc = GNNEncoder(hidden_dims=(128, 64))

    batch_norms = [
        module for module in enc.modules() if isinstance(module, nn.BatchNorm1d)
    ]

    assert len(batch_norms) == 3


def test_graph_batch_tracks_molecule_membership():
    batch = graph_batch(["CCO", "CC", "c1ccccc1"])

    assert batch.batch.shape[0] == batch.x.shape[0]

    assert set(batch.batch.tolist()) == {0, 1, 2}


def test_global_add_pool_sums_nodes_per_molecule():
    # 5 atoms total, each with a 2-dimensional embedding
    x = torch.tensor(
        [
            [1.0, 2.0],  # molecule 0
            [3.0, 4.0],  # molecule 0
            [5.0, 6.0],  # molecule 1
            [7.0, 8.0],  # molecule 1
            [9.0, 10.0],  # molecule 1
        ]
    )

    batch = torch.tensor([0, 0, 1, 1, 1])

    pooled = global_add_pool(x, batch)

    expected = torch.tensor(
        [
            [4.0, 6.0],  # [1+3, 2+4]
            [21.0, 24.0],  # [5+7+9, 6+8+10]
        ]
    )

    assert torch.equal(pooled, expected)
