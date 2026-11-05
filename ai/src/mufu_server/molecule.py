"""RDKit identifiers, descriptors, and optional 3D coordinates; no ML inference."""

import logging
import math
from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem import QED, Crippen, Descriptors, Lipinski, rdDistGeom, rdMolDescriptors

log = logging.getLogger(__name__)

MAX_EMBED_ATOMS = 256  # including explicit hydrogens
EMBED_MAX_ITERATIONS = 20
EMBED_TIMEOUT_S = 2
EMBED_SEED = 0


class InvalidMoleculeSmiles(ValueError):
    pass


@dataclass(frozen=True)
class MoleculeInfo:
    """All fields are present; unavailable values are None (JSON null)."""

    smiles: str
    formula: str | None = None
    inchi_key: str | None = None
    name: str | None = None
    molecular_weight: float | None = None
    exact_mass: float | None = None
    logp: float | None = None
    tpsa: float | None = None
    molar_refractivity: float | None = None
    h_bond_donors: int | None = None
    h_bond_acceptors: int | None = None
    rotatable_bonds: int | None = None
    heavy_atom_count: int | None = None
    aromatic_ring_count: int | None = None
    fraction_sp3: float | None = None
    lipinski_violations: int | None = None
    qed: float | None = None
    molblock: str | None = None


def _optional(calculate, mol):
    """A missing RDKit capability or non-finite result affects only its field."""
    try:
        value = calculate(mol)
        if value == "" or (isinstance(value, float) and not math.isfinite(value)):
            return None
        return value
    except Exception:
        log.warning("molecule field unavailable: %s", calculate.__name__, exc_info=True)
        return None


def _molblock(mol: Chem.Mol) -> str | None:
    """Bound one embedding attempt; never substitute 2D or fabricated coordinates."""
    try:
        if mol.GetNumAtoms() > MAX_EMBED_ATOMS or len(Chem.GetMolFrags(mol)) != 1:
            return None
        structure = Chem.AddHs(mol)  # a copy; descriptors use the original molecule
        if structure.GetNumAtoms() > MAX_EMBED_ATOMS:
            return None
        params = rdDistGeom.ETKDGv3()
        # Older RDKit builds must not silently run without the native timeout.
        if not hasattr(params, "timeout"):
            return None
        params.randomSeed = EMBED_SEED
        params.numThreads = 1
        params.maxIterations = EMBED_MAX_ITERATIONS
        params.timeout = EMBED_TIMEOUT_S
        conformer_id = rdDistGeom.EmbedMolecule(structure, params)
        if conformer_id < 0:
            return None
        conformer = structure.GetConformer(conformer_id)
        if not conformer.Is3D() or not all(
            math.isfinite(coordinate)
            for position in conformer.GetPositions()
            for coordinate in position
        ):
            return None
        return Chem.MolToMolBlock(structure, confId=conformer_id)
    except Exception:
        log.warning("molecule 3D coordinates unavailable", exc_info=True)
        return None


def _inchi_key(mol: Chem.Mol) -> str:
    # Resolve inside _optional so builds without InChI support also return null.
    return Chem.MolToInchiKey(mol)


def _parse_smiles(smiles: str) -> Chem.Mol:
    """Require concrete atoms and explicit isotopes with a known RDKit mass."""
    if not isinstance(smiles, str) or not smiles or any(c.isspace() for c in smiles):
        raise InvalidMoleculeSmiles
    params = Chem.SmilesParserParams()
    params.parseName = False
    params.allowCXSMILES = False
    try:
        mol = Chem.MolFromSmiles(smiles, params)
    except Exception as exc:
        raise InvalidMoleculeSmiles from exc
    if mol is None or mol.GetNumAtoms() == 0:
        raise InvalidMoleculeSmiles
    table = Chem.GetPeriodicTable()
    for atom in mol.GetAtoms():
        number, isotope = atom.GetAtomicNum(), atom.GetIsotope()
        if number == 0 or (isotope and table.GetMassForIsotope(number, isotope) <= 0):
            raise InvalidMoleculeSmiles
    return mol


def molecule_molblock(smiles: str) -> str | None:
    """Worker-only coordinate calculation; the parent enforces its deadline."""
    return _molblock(_parse_smiles(smiles))


def molecule_info(smiles: str, *, include_3d: bool = True) -> MoleculeInfo:
    """Synchronous worker calculation; never call on the HTTP process's threads."""
    mol = _parse_smiles(smiles)

    molecular_weight = _optional(Descriptors.MolWt, mol)
    logp = _optional(Crippen.MolLogP, mol)
    donors = _optional(Lipinski.NumHDonors, mol)
    acceptors = _optional(Lipinski.NumHAcceptors, mol)
    violations = None
    if all(value is not None for value in (molecular_weight, logp, donors, acceptors)):
        violations = sum((molecular_weight > 500, logp > 5, donors > 5, acceptors > 10))

    return MoleculeInfo(
        smiles=Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True),
        formula=_optional(rdMolDescriptors.CalcMolFormula, mol),
        inchi_key=_optional(_inchi_key, mol),
        molecular_weight=molecular_weight,
        exact_mass=_optional(rdMolDescriptors.CalcExactMolWt, mol),
        logp=logp,
        tpsa=_optional(rdMolDescriptors.CalcTPSA, mol),
        molar_refractivity=_optional(Crippen.MolMR, mol),
        h_bond_donors=donors,
        h_bond_acceptors=acceptors,
        rotatable_bonds=_optional(Lipinski.NumRotatableBonds, mol),
        heavy_atom_count=_optional(Lipinski.HeavyAtomCount, mol),
        aromatic_ring_count=_optional(Lipinski.NumAromaticRings, mol),
        fraction_sp3=_optional(Lipinski.FractionCSP3, mol),
        lipinski_violations=violations,
        qed=_optional(QED.qed, mol),
        molblock=_molblock(mol) if include_3d else None,
    )
