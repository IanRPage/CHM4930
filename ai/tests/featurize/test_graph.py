import pytest
import torch
from rdkit import Chem

from featurize.graph import (
    ATOM_TYPES,
    BOND_TYPES,
    CHIRAL_TAGS,
    EDGE_FEATURE_DIM,
    FORMAL_CHARGES,
    HYBRIDIZATIONS,
    NODE_FEATURE_DIM,
    STEREO_TYPES,
    VALENCES,
    mol_to_graph,
)


def graph_of(smiles: str):
    return mol_to_graph(Chem.MolFromSmiles(smiles))


def test_ethanol_shapes():
    data = graph_of("CCO")
    assert data.x.shape == (3, NODE_FEATURE_DIM)
    assert data.edge_index.shape == (2, 4)
    assert data.edge_attr.shape == (4, EDGE_FEATURE_DIM)


def test_benzene_all_aromatic():
    data = graph_of("c1ccccc1")
    assert data.x.shape == (6, NODE_FEATURE_DIM)
    assert data.edge_index.shape == (2, 12)
    assert data.edge_attr.shape == (12, EDGE_FEATURE_DIM)

    aromatic_flag = data.x[:, -1]
    assert torch.all(aromatic_flag == 1)

    bond_type_block = data.edge_attr[:, : len(BOND_TYPES) + 1]
    aromatic_col = len(BOND_TYPES) - 1  # AROMATIC is last entry in BOND_TYPES
    assert torch.all(bond_type_block[:, aromatic_col] == 1)


def test_edges_are_symmetric():
    data = graph_of("CC(=O)Oc1ccccc1C(=O)O")
    edges = data.edge_index.t().tolist()
    attrs = data.edge_attr.tolist()
    lookup = {(i, j): tuple(a) for (i, j), a in zip(edges, attrs)}
    for (i, j), attr in lookup.items():
        assert (j, i) in lookup
        assert lookup[(j, i)] == attr


def _assert_one_hot_blocks(vec, block_sizes):
    idx = 0
    for size in block_sizes:
        block = vec[idx : idx + size]
        assert sum(block) == 1
        idx += size


def test_node_feature_rows_are_valid_one_hots():
    data = graph_of("CC(=O)Oc1ccccc1C(=O)O")
    block_sizes = [
        len(ATOM_TYPES) + 1,
        len(FORMAL_CHARGES) + 1,
        len(HYBRIDIZATIONS) + 1,
        len(VALENCES) + 1,
        len(CHIRAL_TAGS) + 1,
    ]
    for row in data.x.tolist():
        _assert_one_hot_blocks(row, block_sizes)


def test_edge_feature_rows_are_valid_one_hots():
    data = graph_of("CC(=O)Oc1ccccc1C(=O)O")
    block_sizes = [len(BOND_TYPES) + 1, len(STEREO_TYPES) + 1]
    for row in data.edge_attr.tolist():
        _assert_one_hot_blocks(row, block_sizes)


def test_chirality_distinguishes_enantiomers():
    a = graph_of("C[C@H](N)C(=O)O")
    b = graph_of("C[C@@H](N)C(=O)O")
    assert not torch.equal(a.x, b.x)


def test_ez_distinguishes_stereo_bonds():
    a = graph_of(r"F/C=C/F")
    b = graph_of(r"F/C=C\F")
    assert not torch.equal(a.edge_attr, b.edge_attr)


def test_single_atom_no_bonds():
    data = graph_of("[Na+]")
    assert data.x.shape == (1, NODE_FEATURE_DIM)
    assert data.edge_index.shape == (2, 0)
    assert data.edge_attr.shape == (0, EDGE_FEATURE_DIM)


def test_unknown_element_falls_in_other_bucket():
    data = graph_of("[Se]")
    atom_type_block = data.x[0, : len(ATOM_TYPES) + 1]
    assert atom_type_block[-1] == 1


def test_none_raises():
    with pytest.raises(ValueError):
        mol_to_graph(None)


def test_zero_atom_mol_raises():
    mol = Chem.RWMol()
    with pytest.raises(ValueError):
        mol_to_graph(mol)


def test_feature_dims_match_constants():
    data = graph_of("CCO")
    assert data.x.shape[1] == NODE_FEATURE_DIM
    assert data.edge_attr.shape[1] == EDGE_FEATURE_DIM
