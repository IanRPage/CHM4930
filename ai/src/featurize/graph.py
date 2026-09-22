import torch
from rdkit import Chem
from torch_geometric.data import Data

# standard orgo-chem subsets
ATOM_TYPES = ["C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]
FORMAL_CHARGES = [-2, -1, 0, 1, 2]
HYBRIDIZATIONS = [
    Chem.HybridizationType.SP,
    Chem.HybridizationType.SP2,
    Chem.HybridizationType.SP3,
    Chem.HybridizationType.SP3D,
    Chem.HybridizationType.SP3D2,
]
VALENCES = [0, 1, 2, 3, 4, 5, 6]
CHIRAL_TAGS = [
    Chem.ChiralType.CHI_UNSPECIFIED,
    Chem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
]
BOND_TYPES = [
    Chem.BondType.SINGLE,
    Chem.BondType.DOUBLE,
    Chem.BondType.TRIPLE,
    Chem.BondType.AROMATIC,
]
STEREO_TYPES = [
    Chem.BondStereo.STEREONONE,
    Chem.BondStereo.STEREOE,
    Chem.BondStereo.STEREOZ,
    Chem.BondStereo.STEREOCIS,
    Chem.BondStereo.STEREOTRANS,
]

# corresps to number of atom features (one-hot encoded)
NODE_FEATURE_DIM = (
    len(ATOM_TYPES)
    + 1
    + len(FORMAL_CHARGES)
    + 1
    + len(HYBRIDIZATIONS)
    + 1
    + len(VALENCES)
    + 1
    + len(CHIRAL_TAGS)
    + 1
    + 1  # aromaticity bool
)
# corresps to number of bond features (one-hot encoded)
EDGE_FEATURE_DIM = len(BOND_TYPES) + 1 + len(STEREO_TYPES) + 1


# one-hot encoding helper
def _one_hot(value, choices: list) -> list[float]:
    # floats b/c encodings feed into matrix mult later
    vec = [0.0] * (len(choices) + 1)
    vec[choices.index(value) if value in choices else len(choices)] = 1.0
    return vec


# one-hot encodes an atom's features
def _atom_features(atom: Chem.Atom) -> list[float]:
    return (
        _one_hot(atom.GetSymbol(), ATOM_TYPES)
        + _one_hot(atom.GetFormalCharge(), FORMAL_CHARGES)
        + _one_hot(atom.GetHybridization(), HYBRIDIZATIONS)
        + _one_hot(atom.GetTotalValence(), VALENCES)
        + _one_hot(atom.GetChiralTag(), CHIRAL_TAGS)
        + [float(atom.GetIsAromatic())]
    )


# one-hot encodes a bond's features
def _bond_features(bond: Chem.Bond) -> list[float]:
    return _one_hot(bond.GetBondType(), BOND_TYPES) + _one_hot(
        bond.GetStereo(), STEREO_TYPES
    )


# x is [N, NODE_FEATURE_DIM], edge_index is [2, E], and edge_attr is
# [E, EDGE_FEATURE_DIM]
def mol_to_graph(mol: Chem.Mol) -> Data:
    if mol is None:
        raise ValueError("mol is None")
    if mol.GetNumAtoms() == 0:
        raise ValueError("mol has no atoms")

    x = torch.tensor(
        [_atom_features(atom) for atom in mol.GetAtoms()], dtype=torch.float
    )

    if mol.GetNumBonds() == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, EDGE_FEATURE_DIM), dtype=torch.float)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    sources, targets, edge_features = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        features = _bond_features(bond)
        sources += [i, j]
        targets += [j, i]
        edge_features += [features, features]

    edge_index = torch.tensor([sources, targets], dtype=torch.long)
    edge_attr = torch.tensor(edge_features, dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
