"""Molecular featurisation — open-source analogue of Syntelly's representations.

The Syntelly models combine *fingerprint-based CatBoost* and *fragment-based XGBoost*
(Sosnin et al., Molecules 2024, 29, 1826 — the platform paper referenced by [36]).
We mirror that exactly:

* CatBoost features  = ECFP4 Morgan bits (2048) ++ a panel of RDKit physicochemical
  descriptors  -> "fingerprint + descriptor" representation.
* XGBoost  features  = RDKit fragment counts (the ``rdkit.Chem.Fragments`` fr_* keys)
  -> "fragment-based" representation.

All functions are deterministic and operate on RDKit ``Mol`` objects.
"""
from __future__ import annotations

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, Fragments, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

# A compact, standard physicochemical descriptor panel (mirrors the properties
# Syntelly/QSAR models routinely use; superset of the paper's Fig.-2 properties).
DESCRIPTORS = {
    "MW": Descriptors.MolWt,
    "logP": Descriptors.MolLogP,
    "TPSA": Descriptors.TPSA,
    "HBA": Descriptors.NumHAcceptors,
    "HBD": Descriptors.NumHDonors,
    "RB": Descriptors.NumRotatableBonds,
    "AromaticRings": Descriptors.NumAromaticRings,
    "Rings": rdMolDescriptors.CalcNumRings,
    "FractionCSP3": Descriptors.FractionCSP3,
    "HeavyAtoms": Descriptors.HeavyAtomCount,
    "MolMR": Descriptors.MolMR,
    "NumHeteroatoms": Descriptors.NumHeteroatoms,
    "QED": Descriptors.qed,
}

# All fragment descriptors RDKit exposes (fr_*) — the "fragment-based" representation.
_FRAGMENT_FNS = [
    (name, getattr(Fragments, name))
    for name in sorted(dir(Fragments))
    if name.startswith("fr_")
]


def mol_from_smiles(smiles: str):
    return Chem.MolFromSmiles(smiles)


def canonical(smiles: str) -> str | None:
    m = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(m) if m is not None else None


def morgan_fp(mol, radius: int = 2, nbits: int = 2048) -> np.ndarray:
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    arr = np.zeros((nbits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def morgan_bitvect(mol, radius: int = 2, nbits: int = 2048):
    """Return the RDKit ExplicitBitVect (for Tanimoto similarity)."""
    return rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)


def descriptor_vector(mol) -> np.ndarray:
    return np.array([fn(mol) for fn in DESCRIPTORS.values()], dtype=np.float32)


def fragment_vector(mol) -> np.ndarray:
    return np.array([fn(mol) for _, fn in _FRAGMENT_FNS], dtype=np.float32)


def featurize_catboost(mols, radius: int = 2, nbits: int = 2048) -> np.ndarray:
    """ECFP4 bits ++ physicochemical descriptors (fingerprint-based CatBoost input)."""
    rows = [np.concatenate([morgan_fp(m, radius, nbits), descriptor_vector(m)]) for m in mols]
    return np.vstack(rows).astype(np.float32)


def featurize_xgboost(mols) -> np.ndarray:
    """RDKit fragment counts (fragment-based XGBoost input)."""
    return np.vstack([fragment_vector(m) for m in mols]).astype(np.float32)


def featurize(mols, backend: str, radius: int = 2, nbits: int = 2048) -> np.ndarray:
    if backend == "xgboost":
        return featurize_xgboost(mols)
    return featurize_catboost(mols, radius, nbits)


def mw(mol) -> float:
    return float(Descriptors.MolWt(mol))


def pld50_to_mgkg(pld50: float, molecular_weight: float) -> float:
    """Convert -log10(LD50 [mol/kg]) to LD50 in mg/kg given the molecular weight.

    LD50[mol/kg] = 10**(-pLD50);  LD50[mg/kg] = LD50[mol/kg] * MW[g/mol] * 1000.
    """
    return float(10.0 ** (-pld50) * molecular_weight * 1000.0)


def mgkg_to_pld50(mgkg: float, molecular_weight: float) -> float:
    molkg = (mgkg / 1000.0) / molecular_weight
    return float(-np.log10(molkg))
