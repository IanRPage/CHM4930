from rdkit import Chem


# literally a wrapper around rdkit's MolToSmiles
def mol_to_smiles(mol: Chem.Mol) -> str:
    if mol is None:
        raise ValueError("mol is None")
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
