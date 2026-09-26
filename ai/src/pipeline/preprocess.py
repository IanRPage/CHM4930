"""
Turn a standard structure into MuFu's three model inputs.

A structure (SMILES string or RDKit `Mol`) is cleaned. Then RDKit derives its
SMILES, ECFP4 fingerprint, and molecular graph. This path runs at train and
inference time, that way modalities are never missing."""

import logging
from collections.abc import Iterable

import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from torch_geometric.data import Data

from featurize.fingerprint import mol_to_ecfp4
from featurize.graph import mol_to_graph
from featurize.smiles import mol_to_smiles

log = logging.getLogger(__name__)

ALLOWED_N_BITS = (1024, 2048)

_LARGEST_FRAGMENT = rdMolStandardize.LargestFragmentChooser()


def to_mol(structure: str | Chem.Mol) -> Chem.Mol:
    if isinstance(structure, str):
        if not structure.strip():
            raise ValueError("SMILES string is empty")
        mol = Chem.MolFromSmiles(structure)
        if mol is None:
            raise ValueError(f"RDKit couldn't parse SMILES {structure!r}")
    elif isinstance(structure, Chem.Mol):
        # copy so caller's Mol isn't mutated by sanitization
        mol = Chem.Mol(structure)
        try:
            Chem.SanitizeMol(mol)
        except Exception as e:
            raise ValueError(f"RDKit couldn't sanitize Mol: {e}") from e
    else:
        raise TypeError(
            f"expected a SMILES string or RDKit Mol, got {type(structure).__name__}; "
            "fingerprints can't be used as input since ECFP is a lossy hash"
        )

    if mol.GetNumAtoms() == 0:
        raise ValueError("mol has no atoms")
    return _LARGEST_FRAGMENT.choose(mol)


# SMILES of cleaned structure, None if parsing fails
def standardize_smiles(smiles: str) -> str | None:
    try:
        return mol_to_smiles(to_mol(smiles))
    except ValueError:
        return None


def featurize(structure: str | Chem.Mol, n_bits: int = 2048) -> Data:
    if n_bits not in ALLOWED_N_BITS:
        raise ValueError(f"n_bits must be one of {ALLOWED_N_BITS}, got {n_bits}")

    mol = to_mol(structure)
    data = mol_to_graph(mol)
    data.fp = torch.from_numpy(mol_to_ecfp4(mol, n_bits)).float().unsqueeze(0)
    data.smiles = mol_to_smiles(mol)
    return data


def featurize_many(
    structures: Iterable[str | Chem.Mol], n_bits: int = 2048
) -> tuple[list[Data], list[int]]:
    featurized, failed = [], []
    for i, structure in enumerate(structures):
        try:
            featurized.append(featurize(structure, n_bits))
        except ValueError as e:
            log.debug("skipping structure %d: %s", i, e)
            failed.append(i)

    if failed:
        log.warning("dropped %d structures RDKit couldn't featurize", len(failed))
    return featurized, failed


def summarize_labels(df: pd.DataFrame, tasks: list[str]) -> pd.DataFrame:
    labels = df[tasks]
    return pd.DataFrame(
        {
            "labeled": labels.notna().sum(),
            "missing": labels.isna().sum(),
            "positive_rate": labels.mean(),
        }
    ).rename_axis("task")
