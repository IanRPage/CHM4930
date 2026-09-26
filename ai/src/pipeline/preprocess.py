"""
Turn a standard structure into MuFu's three model inputs.

A structure (SMILES string or RDKit `Mol`) is cleaned. Then RDKit derives its
SMILES, ECFP4 fingerprint, and molecular graph. This path runs at train and
inference time, that way modalities are never missing.
"""

import logging
from collections.abc import Iterable

import pandas as pd
import torch
from rdkit import Chem, rdBase
from rdkit.Chem.MolStandardize import rdMolStandardize
from torch_geometric.data import Data

from featurize.fingerprint import mol_to_ecfp4
from featurize.graph import mol_to_graph
from featurize.smiles import mol_to_smiles

log = logging.getLogger(__name__)

ALLOWED_N_BITS = (1024, 2048)

_LARGEST_FRAGMENT = rdMolStandardize.LargestFragmentChooser(preferOrganic=True)


def to_mol(structure: str | Chem.Mol) -> Chem.Mol:
    if isinstance(structure, str):
        if not structure.strip():
            raise ValueError("SMILES string is empty")
        label = repr(structure)
    elif isinstance(structure, Chem.Mol):
        label = "Mol"
    else:
        raise TypeError(
            f"expected a SMILES string or RDKit Mol, got {type(structure).__name__}; "
            "fingerprints can't be used as input since ECFP is a lossy hash"
        )

    with rdBase.BlockLogs():
        if isinstance(structure, str):
            mol = Chem.MolFromSmiles(structure, sanitize=False)
            if mol is None:
                raise ValueError(f"RDKit couldn't parse SMILES {label}")
        else:
            mol = Chem.Mol(structure)
        try:
            Chem.SanitizeMol(mol)
            mol = Chem.RemoveHs(mol)
        except Exception as e:
            raise ValueError(f"RDKit couldn't sanitize {label}: {e}") from e

        if mol.GetNumAtoms() == 0:
            raise ValueError("mol has no atoms")
        return _LARGEST_FRAGMENT.choose(mol)


def standardize_smiles(smiles: str) -> str | None:
    try:
        return mol_to_smiles(to_mol(smiles))
    except ValueError as e:
        log.debug("couldn't standardize: %s", e)
        return None


def standardize_smiles_column(df: pd.DataFrame, column: str = "smiles") -> pd.DataFrame:
    df = df.assign(smiles=df[column].map(standardize_smiles))
    n_unparsed = int(df["smiles"].isna().sum())
    if n_unparsed:
        log.warning("dropping %d rows whose SMILES RDKit couldn't parse", n_unparsed)
        df = df.dropna(subset=["smiles"])
    return df


def featurize(structure: str | Chem.Mol, n_bits: int = 2048) -> Data:
    if n_bits not in ALLOWED_N_BITS:
        raise ValueError(f"n_bits must be one of {ALLOWED_N_BITS}, got {n_bits}")

    mol = to_mol(structure)
    data = mol_to_graph(mol)
    data.fp = torch.from_numpy(mol_to_ecfp4(mol, n_bits)).unsqueeze(0)
    data.smiles = mol_to_smiles(mol)
    return data


def featurize_many(
    structures: pd.Series | Iterable[str | Chem.Mol], n_bits: int = 2048
) -> tuple[list[Data], list]:
    items = (
        structures.items()
        if isinstance(structures, pd.Series)
        else enumerate(structures)
    )
    featurized, failed = [], []
    for i, structure in items:
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
