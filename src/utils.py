import pandas as pd
from rdkit import Chem


def smi2mol(smiles: str) -> Chem.Mol:
    return  Chem.MolFromSmiles(smiles) if pd.notna(smiles) else None