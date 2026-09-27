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


@pytest.mark.parametrize(
    "mol",
    [Chem.MolFromSmiles(ASPIRIN), Chem.AddHs(Chem.MolFromSmiles(ASPIRIN))],
    ids=["mol", "mol_with_hs"],
)
def test_mol_input_matches_smiles(mol):
    a = featurize(ASPIRIN)
    b = featurize(mol)
    assert a.smiles == b.smiles
    assert torch.equal(a.fp, b.fp)
    assert torch.equal(a.x, b.x)
    assert torch.equal(a.edge_index, b.edge_index)


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


@pytest.mark.parametrize("bad", ["", "   ", Chem.RWMol()])
def test_unusable_structure_raises(bad):
    with pytest.raises(ValueError):
        featurize(bad)


def test_sanitize_error_names_input():
    with pytest.raises(ValueError, match=r"'c1cccc1'.*kekulize"):
        to_mol("c1cccc1")


def test_rdkit_errors_not_printed(capfd):
    featurize_many(["c1cccc1", "C1CC", "CC(=O)[O-].[Na+]"])
    assert capfd.readouterr().err == ""


def test_unsupported_n_bits_raises():
    with pytest.raises(ValueError):
        featurize("CCO", n_bits=512)


@pytest.mark.parametrize("n_bits", [1024, 2048])
def test_batches_all_modalities(n_bits):
    batch = Batch.from_data_list(
        [featurize("CCO", n_bits=n_bits), featurize("c1ccccc1", n_bits=n_bits)]
    )
    assert batch.num_graphs == 2
    assert batch.fp.shape == (2, n_bits)
    assert batch.fp.dtype == torch.uint8
    assert batch.smiles == ["CCO", "c1ccccc1"]
    assert batch.x.shape == (9, NODE_FEATURE_DIM)


@pytest.mark.parametrize(
    ("structures", "expected_failed"),
    [
        (["CCO", "not_a_smiles", "c1ccccc1"], [1]),
        (pd.Series(["CCO", "not_a_smiles", "c1ccccc1"], index=[0, 4, 7]), [4]),
    ],
)
def test_featurize_many_reports_failures(structures, expected_failed):
    featurized, failed = featurize_many(structures)
    assert [d.smiles for d in featurized] == ["CCO", "c1ccccc1"]
    assert failed == expected_failed


def test_input_mol_not_mutated():
    mol = Chem.MolFromSmiles("CC(=O)[O-].[Na+]", sanitize=False)
    before = Chem.MolToSmiles(mol)
    featurize(mol)
    assert Chem.MolToSmiles(mol) == before
    assert mol.GetNumAtoms() == 5


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("CC(=O)[O-].[Na+]", "CC(=O)[O-]"),  # strips counterion, keeps the charge
        ("Cl.CCN", "CCN"),  # strips HCl salt
        ("[Na+].[Cl-].c1ccccc1CC(=O)O", "O=C(O)Cc1ccccc1"),  # keeps the big fragment
        ("C[C@H](N)C(=O)O", "C[C@H](N)C(=O)O"),  # stereo preserved
        ("CC#N.F[P-](F)(F)(F)(F)F", "CC#N"),  # prefers organic over bigger PF6-
        ("not_a_smiles", None),
    ],
)
def test_standardize_smiles(raw, expected):
    assert standardize_smiles(raw) == expected


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
