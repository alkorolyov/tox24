import numpy as np

from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import rdReducedGraphs, rdFingerprintGenerator
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit.DataStructs import ExplicitBitVect

MORGAN_FP_SIZE = 1024
MORGAN_RADIUS = 2
AVALON_FP_SIZE = 1024
ERG_FP_SIZE = 315  # Constant value, cannot be changed
FPS_SIZE = MORGAN_FP_SIZE + MORGAN_FP_SIZE + ERG_FP_SIZE


def fp2numpy(x: ExplicitBitVect):
    return np.frombuffer(x.ToBitString().encode(), 'u1') - ord('0')


def get_morgan_fp(mol, radius=MORGAN_RADIUS, n_bits=MORGAN_FP_SIZE) -> np.ndarray:
    bitvect = GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return fp2numpy(bitvect)


def get_avalon_fp(mol, n_bits=AVALON_FP_SIZE) -> np.ndarray:
    bitvect = pyAvalonTools.GetAvalonFP(mol, nBits=n_bits)
    return fp2numpy(bitvect)


def get_erg_fp(mol) -> np.ndarray:
    return rdReducedGraphs.GetErGFingerprint(mol)


def get_fingerprints(mol) -> np.ndarray:
    fps = [
        get_morgan_fp(mol),
        get_avalon_fp(mol),
        get_erg_fp(mol)
    ]
    return np.concatenate(fps)