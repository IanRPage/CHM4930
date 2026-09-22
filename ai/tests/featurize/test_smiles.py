import pytest
from rdkit import Chem

from featurize.smiles import mol_to_smiles


def smiles_of(smiles: str) -> str:
    return mol_to_smiles(Chem.MolFromSmiles(smiles))


@pytest.mark.parametrize("spelling", ["OCC", "C(O)C", "CCO"])
def test_spellings_collapse_to_one_string(spelling):
    assert smiles_of(spelling) == "CCO"


def test_aromatic_form_kept():
    assert smiles_of("C1=CC=CC=C1") == "c1ccccc1"


def test_idempotent():
    once = smiles_of("CC(=O)Oc1ccccc1C(=O)O")
    assert smiles_of(once) == once


def test_tetrahedral_stereo_preserved():
    assert smiles_of("C[C@H](N)C(=O)O") != smiles_of("C[C@@H](N)C(=O)O")


def test_double_bond_stereo_preserved():
    assert smiles_of("F/C=C/F") != smiles_of("F/C=C\\F")


def test_fragments_dot_separated():
    assert "." in smiles_of("CCO.Cl")


def test_none_raises():
    with pytest.raises(ValueError):
        mol_to_smiles(None)
