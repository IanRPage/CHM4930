import numpy as np
import pytest
from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors

from featurize.fingerprint import mol_to_ecfp4


# helper for getting the fingerprint of a molecule
def fp_of(smiles: str, **kwargs) -> np.ndarray:
    return mol_to_ecfp4(Chem.MolFromSmiles(smiles), **kwargs)


def test_default_shape_and_dtype():
    fp = fp_of("CCO")
    assert fp.shape == (2048,)
    assert fp.dtype == np.uint8


def test_custom_n_bits():
    assert fp_of("CCO", n_bits=1024).shape == (1024,)


def test_binary_and_non_empty():
    fp = fp_of("CCO")
    assert set(np.unique(fp)) <= {0, 1}
    assert fp.sum() > 0


def test_deterministic():
    assert np.array_equal(
        fp_of("CC(=O)Oc1ccccc1C(=O)O"), fp_of("CC(=O)Oc1ccccc1C(=O)O")
    )


@pytest.mark.parametrize("spelling", ["OCC", "C(O)C"])
def test_spellings_give_same_fingerprint(spelling):
    assert np.array_equal(fp_of(spelling), fp_of("CCO"))


def test_different_molecules_differ():
    assert not np.array_equal(fp_of("CCO"), fp_of("c1ccccc1"))


def test_none_raises():
    with pytest.raises(ValueError):
        mol_to_ecfp4(None)


@pytest.mark.parametrize("n_bits", [0, -1])
def test_non_positive_n_bits_raises(n_bits):
    with pytest.raises(ValueError):
        fp_of("CCO", n_bits=n_bits)


def test_n_bits_not_shared_across_cached_generators():
    assert fp_of("CCO", n_bits=2048).shape == (2048,)
    assert fp_of("CCO", n_bits=1024).shape == (1024,)
    assert fp_of("CCO", n_bits=2048).shape == (2048,)


def test_matches_radius_2_morgan_reference():
    mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
    reference = np.zeros(2048, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(
        rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048), reference
    )
    assert np.array_equal(mol_to_ecfp4(mol), reference)
