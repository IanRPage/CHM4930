import numpy as np
import pandas as pd
import pytest
import torch
from rdkit import Chem, DataStructs
from torch_geometric.data import Batch

from featurize.graph import NODE_FEATURE_DIM
from pipeline.preprocess import (
    featurize,
    featurize_many,
    standardize_smiles,
    standardize_smiles_column,
    summarize_labels,
    to_mol,
)

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"


def test_str_and_mol_inputs_match():
    a = featurize(ASPIRIN)
    b = featurize(Chem.MolFromSmiles(ASPIRIN))
    assert a.smiles == b.smiles
    assert torch.equal(a.fp, b.fp)
    assert torch.equal(a.x, b.x)
    assert torch.equal(a.edge_index, b.edge_index)


def test_strips_salt_and_keeps_charge():
    assert featurize("CC(=O)[O-].[Na+]").smiles == "CC(=O)[O-]"


def test_stereo_survives():
    a = featurize("C[C@H](N)C(=O)O")
    b = featurize("C[C@@H](N)C(=O)O")
    assert a.smiles != b.smiles
    assert not torch.equal(a.x, b.x)


@pytest.mark.parametrize(
    "fingerprint",
    [
        np.zeros(2048, dtype=np.uint8),
        torch.zeros(2048),
        DataStructs.ExplicitBitVect(2048),
    ],
)
def test_fingerprint_input_rejected(fingerprint):
    with pytest.raises(TypeError, match="lossy"):
        featurize(fingerprint)


@pytest.mark.parametrize("bad", ["not_a_smiles", "", "   ", Chem.RWMol()])
def test_unusable_structure_raises(bad):
    with pytest.raises(ValueError):
        featurize(bad)


def test_unsupported_n_bits_raises():
    with pytest.raises(ValueError):
        featurize("CCO", n_bits=512)


@pytest.mark.parametrize("n_bits", [1024, 2048])
def test_shapes(n_bits):
    data = featurize(ASPIRIN, n_bits=n_bits)
    assert data.fp.shape == (1, n_bits)
    assert data.fp.dtype == torch.float
    assert data.x.shape == (13, NODE_FEATURE_DIM)


def test_batches_all_modalities():
    batch = Batch.from_data_list([featurize("CCO"), featurize("c1ccccc1")])
    assert batch.num_graphs == 2
    assert batch.fp.shape == (2, 2048)
    assert batch.smiles == ["CCO", "c1ccccc1"]
    assert batch.x.shape == (9, NODE_FEATURE_DIM)


def test_featurize_many_reports_failures():
    featurized, failed = featurize_many(["CCO", "not_a_smiles", "c1ccccc1"])
    assert [d.smiles for d in featurized] == ["CCO", "c1ccccc1"]
    assert failed == [1]


def test_featurize_many_propagates_fingerprint_input():
    with pytest.raises(TypeError):
        featurize_many(["CCO", np.zeros(2048, dtype=np.uint8)])


def test_input_mol_not_mutated():
    mol = Chem.MolFromSmiles("CC(=O)[O-].[Na+]", sanitize=False)
    before = Chem.MolToSmiles(mol)
    featurize(mol)
    assert Chem.MolToSmiles(mol) == before
    assert mol.GetNumAtoms() == 5


def test_smiles_is_idempotent():
    once = featurize("[Na+].[Cl-].c1ccccc1CC(=O)O").smiles
    assert featurize(once).smiles == once


def test_to_mol_keeps_largest_fragment():
    assert to_mol("Cl.CCN").GetNumAtoms() == 3


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("CC(=O)[O-].[Na+]", "CC(=O)[O-]"),  # strips counterion, keeps the charge
        ("Cl.CCN", "CCN"),  # strips HCl salt
        ("[Na+].[Cl-].c1ccccc1CC(=O)O", "O=C(O)Cc1ccccc1"),  # keeps the big fragment
        ("C[C@H](N)C(=O)O", "C[C@H](N)C(=O)O"),  # stereo preserved
    ],
)
def test_standardize_smiles(raw, expected):
    assert standardize_smiles(raw) == expected


def test_standardize_smiles_none_when_unparseable():
    assert standardize_smiles("not_a_smiles") is None


def test_standardize_smiles_column_drops_unparseable(caplog):
    df = pd.DataFrame({"raw": ["Cl.CCN", "not_a_smiles"], "label": [1, 0]})
    out = standardize_smiles_column(df, column="raw")
    assert out["smiles"].tolist() == ["CCN"]
    assert out["label"].tolist() == [1]
    assert "dropping 1 rows" in caplog.text


def test_summarize_labels_counts_missing_and_positives():
    df = pd.DataFrame({"T1": [1, 0, 1], "T2": [np.nan] * 3})
    summary = summarize_labels(df, ["T1", "T2"])
    assert summary.loc["T1"].tolist() == pytest.approx([3, 0, 2 / 3])
    assert summary.loc["T2", "labeled"] == 0
    assert summary.loc["T2", "missing"] == 3
    assert np.isnan(summary.loc["T2", "positive_rate"])
