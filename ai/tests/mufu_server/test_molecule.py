import json
import math
from dataclasses import asdict
from types import SimpleNamespace

import pytest
from rdkit import Chem
from rdkit.Chem import QED, Crippen, Descriptors, Lipinski, rdMolDescriptors

from mufu_server import molecule
from mufu_server.molecule import InvalidMoleculeSmiles, molecule_info


@pytest.mark.parametrize(
    ("smiles", "formula"),
    [
        ("OCC", "C2H6O"),
        ("CC(=O)Oc1ccccc1C(=O)O", "C9H8O4"),
        ("Cn1c(=O)c2c(ncn2C)n(C)c1=O", "C8H10N4O2"),
        ("c1ccccc1", "C6H6"),
        ("[NH4+]", "H4N+"),
        ("[13CH3]CO", "C2H6O"),
        ("[13CH4]", "CH4"),
        ("[Na+].[Cl-]", "ClNa"),
        ("C" * 40, "C40H82"),
    ],
)
def test_identifiers_and_descriptors_match_rdkit(smiles, formula):
    result = molecule_info(smiles)
    mol = Chem.MolFromSmiles(smiles)
    expected = {
        "smiles": Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True),
        "formula": formula,
        "inchi_key": Chem.MolToInchiKey(mol),
        "name": None,
        "molecular_weight": Descriptors.MolWt(mol),
        "exact_mass": rdMolDescriptors.CalcExactMolWt(mol),
        "logp": Crippen.MolLogP(mol),
        "tpsa": rdMolDescriptors.CalcTPSA(mol),
        "molar_refractivity": Crippen.MolMR(mol),
        "h_bond_donors": Lipinski.NumHDonors(mol),
        "h_bond_acceptors": Lipinski.NumHAcceptors(mol),
        "rotatable_bonds": Lipinski.NumRotatableBonds(mol),
        "heavy_atom_count": Lipinski.HeavyAtomCount(mol),
        "aromatic_ring_count": Lipinski.NumAromaticRings(mol),
        "fraction_sp3": Lipinski.FractionCSP3(mol),
        "qed": QED.qed(mol),
    }
    expected["lipinski_violations"] = sum(
        (
            expected["molecular_weight"] > 500,
            expected["logp"] > 5,
            expected["h_bond_donors"] > 5,
            expected["h_bond_acceptors"] > 10,
        )
    )
    actual = asdict(result)
    actual.pop("molblock")
    assert actual == expected  # no rounding or synthetic values
    for field in (
        "h_bond_donors",
        "h_bond_acceptors",
        "rotatable_bonds",
        "heavy_atom_count",
        "aromatic_ring_count",
        "lipinski_violations",
    ):
        assert type(actual[field]) is int
    json.dumps(asdict(result), allow_nan=False)


def test_known_ethanol_values_and_3d_round_trip():
    result = molecule_info("OCC")
    assert result.smiles == "CCO"
    assert result.inchi_key == "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"
    assert result.molecular_weight == pytest.approx(46.069)
    assert result.exact_mass == pytest.approx(46.041864812)
    assert result.tpsa == pytest.approx(20.23)
    structure = Chem.MolFromMolBlock(result.molblock, removeHs=False)
    assert structure is not None
    assert structure.GetNumAtoms() == 9
    assert Chem.MolToSmiles(Chem.RemoveHs(structure)) == "CCO"
    assert structure.GetConformer().Is3D()
    positions = structure.GetConformer().GetPositions()
    assert all(math.isfinite(c) for position in positions for c in position)
    assert any(abs(c) > 0 for position in positions for c in position)


def test_stereochemistry_is_preserved():
    left = molecule_info("C[C@H](O)C(=O)O")
    right = molecule_info("C[C@@H](O)C(=O)O")
    assert left.smiles != right.smiles
    assert left.inchi_key != right.inchi_key
    structure = Chem.MolFromMolBlock(left.molblock)
    assert Chem.MolToSmiles(structure) == left.smiles


def test_embedding_is_reproducible():
    assert molecule_info("CCO").molblock == molecule_info("CCO").molblock


@pytest.mark.parametrize(
    "smiles",
    [
        "",
        " ",
        "CC O",
        "CCO\n",
        "CCO ethanol",
        "C1CC",
        "[Xx]",
        "C(C)(C)(C)(C)C",
        "not-smiles",
        ".",
        "*",
        "*C",
        "[999C]",
        None,
        5,
    ],
)
def test_invalid_smiles_raise(smiles):
    with pytest.raises(InvalidMoleculeSmiles):
        molecule_info(smiles)


@pytest.mark.parametrize("failure", ["status", "exception"])
def test_embedding_failure_keeps_descriptors(monkeypatch, failure):
    def fail(*args):
        if failure == "exception":
            raise RuntimeError("embedding failed")
        return -1

    monkeypatch.setattr(molecule.rdDistGeom, "EmbedMolecule", fail)
    result = molecule_info("CCO")
    assert result.molblock is None
    assert result.formula == "C2H6O"
    assert result.molecular_weight == pytest.approx(46.069)


def test_embedding_uses_bounded_parameters(monkeypatch):
    calls = []

    def embed(mol, params):
        calls.append(params)
        assert mol.GetNumAtoms() == 9
        assert params.randomSeed == 0
        assert params.numThreads == 1
        assert params.maxIterations == 20
        assert params.timeout == 2
        return -1

    monkeypatch.setattr(molecule.rdDistGeom, "EmbedMolecule", embed)
    assert molecule_info("CCO").molblock is None
    assert len(calls) == 1


@pytest.mark.parametrize("smiles", ["C" * 257, "C" * 90, "CCO.[Na+]"])
def test_large_or_disconnected_structures_skip_embedding(monkeypatch, smiles):
    def unexpected(*args):
        pytest.fail("embedding should be skipped")

    monkeypatch.setattr(molecule.rdDistGeom, "EmbedMolecule", unexpected)
    result = molecule_info(smiles)
    assert result.molblock is None
    assert result.molecular_weight == Descriptors.MolWt(Chem.MolFromSmiles(smiles))


def test_missing_native_timeout_skips_embedding(monkeypatch):
    monkeypatch.setattr(molecule.rdDistGeom, "ETKDGv3", SimpleNamespace)
    assert molecule_info("CCO").molblock is None


@pytest.mark.parametrize("invalid", ["2d", "nonfinite"])
def test_invalid_coordinates_are_not_returned(monkeypatch, invalid):
    def embed(mol, params):
        conformer = Chem.Conformer(mol.GetNumAtoms())
        conformer.Set3D(invalid != "2d")
        if invalid == "nonfinite":
            conformer.SetAtomPosition(0, (float("nan"), 0, 0))
        return mol.AddConformer(conformer, assignId=True)

    monkeypatch.setattr(molecule.rdDistGeom, "EmbedMolecule", embed)
    assert molecule_info("CCO").molblock is None


def test_missing_inchi_support_returns_null(monkeypatch):
    monkeypatch.delattr(Chem, "MolToInchiKey")
    result = molecule_info("CCO")
    assert result.inchi_key is None
    assert result.formula == "C2H6O"


@pytest.mark.parametrize("value", ["", None])
def test_empty_inchi_returns_null(monkeypatch, value):
    monkeypatch.setattr(Chem, "MolToInchiKey", lambda mol: value)
    assert molecule_info("CCO").inchi_key is None


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_unavailable_descriptor_and_dependent_rule_are_null(monkeypatch, value):
    monkeypatch.setattr(Crippen, "MolLogP", lambda mol: value)
    result = molecule_info("CCO")
    assert result.logp is None
    assert result.lipinski_violations is None
    assert result.formula == "C2H6O"
    json.dumps(asdict(result), allow_nan=False)


def test_descriptor_exception_does_not_fail_other_fields(monkeypatch):
    def fail(mol):
        raise RuntimeError("descriptor unavailable")

    monkeypatch.setattr(QED, "qed", fail)
    result = molecule_info("CCO")
    assert result.qed is None
    assert result.lipinski_violations == 0
    assert result.formula == "C2H6O"


def test_descriptors_do_not_attempt_embedding_when_disabled(monkeypatch):
    def unexpected(*args):
        pytest.fail("descriptor phase must not attempt embedding")

    monkeypatch.setattr(molecule.rdDistGeom, "EmbedMolecule", unexpected)
    result = molecule_info("[13CH4]", include_3d=False)
    assert result.exact_mass == pytest.approx(17.034654968)
    assert result.molblock is None


@pytest.mark.parametrize("increment", [0, 1])
def test_lipinski_thresholds_are_strict(monkeypatch, increment):
    monkeypatch.setattr(Descriptors, "MolWt", lambda mol: 500.0 + increment)
    monkeypatch.setattr(Crippen, "MolLogP", lambda mol: 5.0 + increment)
    monkeypatch.setattr(Lipinski, "NumHDonors", lambda mol: 5 + increment)
    monkeypatch.setattr(Lipinski, "NumHAcceptors", lambda mol: 10 + increment)
    assert molecule_info("CCO").lipinski_violations == 4 * increment
