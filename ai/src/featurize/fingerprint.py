"""
Generate Morgan / ECFP4 circular fingerprint bit vectors from RDKit molecules.
"""

from functools import cache

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

FINGERPRINT_RADIUS = 2
FINGERPRINT_BITS = 2048
_POPCOUNT = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(
    axis=1, dtype=np.uint8
)


@cache
def _generator(n_bits: int):
    return rdFingerprintGenerator.GetMorganGenerator(
        radius=FINGERPRINT_RADIUS, fpSize=n_bits
    )


# ecfp4 bit vector with shape (n_bits, ), all uint8 dtype
def mol_to_ecfp4(mol: Chem.Mol, n_bits: int = FINGERPRINT_BITS) -> np.ndarray:
    if mol is None:
        raise ValueError("mol is None")
    if n_bits <= 0:
        raise ValueError("n_bits must be positive")
    return _generator(n_bits).GetFingerprintAsNumPy(mol)


def pack_fingerprints(fps, n_bits: int = FINGERPRINT_BITS) -> np.ndarray:
    arr = np.asarray(fps, dtype=np.uint8)
    if arr.ndim != 2 or arr.shape[0] < 1 or arr.shape[1] != n_bits:
        raise ValueError(
            f"expected non-empty (N, {n_bits}) fingerprint matrix, got {arr.shape}"
        )
    return np.packbits(arr, axis=1)


def max_tanimoto(query_fp, packed_train_fps: np.ndarray) -> float:
    packed_q = pack_fingerprints(np.asarray(query_fp, dtype=np.uint8)[None, :])[0]
    inter = _POPCOUNT[packed_train_fps & packed_q].sum(axis=1, dtype=np.int64)
    union = (
        _POPCOUNT[packed_train_fps].sum(axis=1, dtype=np.int64)
        + int(_POPCOUNT[packed_q].sum())
        - inter
    )
    sim = np.divide(
        inter, union, out=np.zeros(len(packed_train_fps), dtype=float), where=union > 0
    )
    return float(sim.max())
