from functools import cache

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator


@cache
def _generator(n_bits: int):
    return rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)


# ecfp4 bit vector with shape (n_bits, ), all uint8 dtype
def mol_to_ecfp4(mol: Chem.Mol, n_bits: int = 2048) -> np.ndarray:
    if mol is None:
        raise ValueError("mol is None")
    if n_bits <= 0:
        raise ValueError("n_bits must be positive")
    return _generator(n_bits).GetFingerprintAsNumPy(mol)
