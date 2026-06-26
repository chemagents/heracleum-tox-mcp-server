"""Load the reconstructed Heracleum sosnowskyi metabolite dataset."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

from .config import get_settings

RDLogger.DisableLog("rdApp.*")
logger = logging.getLogger(__name__)

# Paper cluster letter -> human-readable chemical family (Section 3.2 / Fig. 1).
CLUSTER_FAMILIES = {
    "A": "terpenoids (isoprenoids)",
    "B": "polyphenolic glycosides, flavonoids, cyclic polyols",
    "C": "fatty acids",
    "D": "aromatic compounds",
    "E": "furanocoumarins and coumarins",
}

# Dominant chemical class -> paper cluster letter (used to label computed clusters).
CLASS_TO_CLUSTER = {
    "terpenoid": "A",
    "polyphenol_flavonoid": "B",
    "fatty_acid": "C",
    "aromatic": "D",
    "aromatic_acid": "E",
    "furanocoumarin": "E",
    "coumarin": "E",
}


@dataclass
class Dataset:
    df: pd.DataFrame
    names: list[str]
    smiles: list[str]
    mols: list                       # RDKit Mol per row (all parse)
    classes: list[str]
    paper_cluster: list[str]
    _canon_index: dict[str, int] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.df)

    def index_for_smiles(self, smiles: str) -> int | None:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return self._canon_index.get(Chem.MolToSmiles(mol))

    def index_for_name(self, name: str) -> int | None:
        key = name.strip().lower()
        for i, nm in enumerate(self.names):
            if nm.strip().lower() == key:
                return i
        return None


@lru_cache(maxsize=1)
def load_dataset() -> Dataset:
    settings = get_settings()
    path = Path(settings.dataset_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run `uv run python build_dataset.py` to "
            "reconstruct it from the compound names (resolves SMILES via PubChem)."
        )
    df = pd.read_csv(path)
    smiles = df["smiles"].tolist()
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    # Drop any rows that failed to parse (should be none after build_dataset.py).
    keep = [i for i, m in enumerate(mols) if m is not None]
    if len(keep) != len(mols):
        df = df.iloc[keep].reset_index(drop=True)
        smiles = [smiles[i] for i in keep]
        mols = [mols[i] for i in keep]
    canon_index: dict[str, int] = {}
    for i, m in enumerate(mols):
        canon_index.setdefault(Chem.MolToSmiles(m), i)
    return Dataset(
        df=df,
        names=df["name"].tolist(),
        smiles=smiles,
        mols=mols,
        classes=df["class"].tolist(),
        paper_cluster=df["paper_cluster"].fillna("").astype(str).tolist(),
        _canon_index=canon_index,
    )


def cluster_e_mask(ds: Dataset) -> np.ndarray:
    """Boolean mask of the paper's cluster E (furanocoumarins/coumarins, Table 2)."""
    return np.array([c == "E" for c in ds.paper_cluster], dtype=bool)


def reference_value(ds: Dataset, row: int, column: str):
    """Return a published reference value (NaN -> None)."""
    if column not in ds.df.columns:
        return None
    val = ds.df.iloc[row][column]
    if pd.isna(val):
        return None
    return val
