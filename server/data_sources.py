"""Open training-data sources — the analogue of Syntelly's proprietary training sets.

The paper (Section 2.4) states the Syntelly models were trained on data aggregated from
**TOXRIC, ChemIDplus, NIH, NCATS-Flux, PyTDC and PubMed**. Of these, the cleanest openly
scriptable source is the Therapeutics Data Commons (PyTDC), which itself aggregates
ChemIDplus / ToxRefDB acute-toxicity and the standard tox-endpoint benchmarks. We train
our open CatBoost/XGBoost models on these public sets.

Endpoint -> open dataset mapping:
  ld50            (regression)     TDC `LD50_Zhu`           acute toxicity, -log10(mol/kg)
  hepatotoxicity  (classification) TDC `DILI`              drug-induced liver injury
  dili            (classification) TDC `DILI`              (same open source; see note)
  cardiotoxicity  (classification) TDC `hERG`              hERG blockade  -> cardiotoxicity
  carcinogenicity (classification) TDC `Carcinogens_Lagunin`

Exact route reproduction: the paper predicts LD50 for six mouse routes (oral, IV, IP, SC,
skin, IM) using TOXRIC's per-route sets, which are not openly scriptable. If the user drops
per-route CSVs (columns ``smiles,y`` with y = -log10(mol/kg)) into ``HERACLEUM_LD50_DATA_DIR``
as ``ld50_<route>.csv``, route-specific models train automatically; otherwise every route
falls back to the shared open acute-LD50 model and is flagged ``data_source="tdc_acute"``.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

LD50_ROUTES = ["oral", "iv", "ip", "sc", "skin", "im"]

# The paper's own reported model quality (Supplementary Table S6): RMSE in Log10(mg/kg)
# for the six LD50 routes, ROC-AUC for the four classification endpoints.
PAPER_METRICS: dict[str, tuple[str, float]] = {
    "ld50_iv": ("RMSE", 0.41), "ld50_oral": ("RMSE", 0.45), "ld50_ip": ("RMSE", 0.49),
    "ld50_sc": ("RMSE", 0.65), "ld50_skin": ("RMSE", 0.81), "ld50_im": ("RMSE", 0.87),
    "ld50": ("RMSE", 0.45),    # acute base model (compare to the oral RMSE)
    "carcinogenicity": ("ROC-AUC", 0.79), "hepatotoxicity": ("ROC-AUC", 0.81),
    "dili": ("ROC-AUC", 0.90), "cardiotoxicity": ("ROC-AUC", 0.93),
}

ENDPOINTS: dict[str, dict] = {
    "ld50": dict(task="regression", tdc="LD50_Zhu", group="Tox",
                 unit="neg_log10_mol_per_kg", section="2.4 / 3.3",
                 desc="acute toxicity LD50 (mouse), -log10(mol/kg)"),
    "hepatotoxicity": dict(task="classification", tdc="DILI", group="Tox",
                           section="3.5 / Table 2", desc="hepatotoxicity (toxic/non-toxic)"),
    "dili": dict(task="classification", tdc="DILI", group="Tox",
                 section="3.5 / Table 2", desc="drug-induced liver injury",
                 note="Same open source (TDC DILI) as hepatotoxicity; Syntelly used "
                      "separate proprietary sets."),
    "cardiotoxicity": dict(task="classification", tdc="hERG", group="Tox",
                           section="3.5 / Table 2", desc="cardiotoxicity (hERG blockade)"),
    "carcinogenicity": dict(task="classification", tdc="Carcinogens_Lagunin", group="Tox",
                            section="3.5 / Table 2", desc="carcinogenicity"),
}


def _tdc_cache_dir() -> str:
    return os.environ.get("HERACLEUM_TDC_DIR", "tdc_data")


def _local_csv(name: str) -> Path | None:
    data_dir = os.environ.get("HERACLEUM_LD50_DATA_DIR")
    if not data_dir:
        return None
    p = Path(data_dir) / f"{name}.csv"
    return p if p.exists() else None


def _load_tdc(tdc_name: str, group: str) -> pd.DataFrame:
    """Download (cached) a TDC single-prediction dataset -> DataFrame[Drug, Y]."""
    try:
        from tdc.single_pred import ADME, Tox  # noqa: F401  (lazy, heavy import)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "PyTDC is required for live model training but is not importable "
            f"({exc}). Install it with `uv pip install --no-deps PyTDC==0.4.1` "
            "(its data API needs only pandas/requests, already present)."
        ) from exc
    from tdc.single_pred import Tox

    data = Tox(name=tdc_name, path=_tdc_cache_dir())
    df = data.get_data()
    return df[["Drug", "Y"]].rename(columns={"Drug": "smiles", "Y": "y"})


def load_training_data(endpoint: str, route: str | None = None) -> tuple[list[str], np.ndarray, dict]:
    """Return (smiles, y, meta) for an endpoint, preferring a local CSV override.

    For ``endpoint="ld50"`` an optional ``route`` selects a per-route local CSV
    (``ld50_<route>.csv``); if absent it falls back to the shared acute-LD50 set.
    """
    if endpoint not in ENDPOINTS:
        raise KeyError(f"Unknown endpoint {endpoint!r}; known: {list(ENDPOINTS)}")
    spec = ENDPOINTS[endpoint]

    # Local CSV override (exact-route reproduction or custom training data).
    csv_name = f"ld50_{route}" if (endpoint == "ld50" and route) else endpoint
    local = _local_csv(csv_name)
    if local is not None:
        df = pd.read_csv(local)
        meta = dict(data_source=f"local:{local.name}", task=spec["task"], n=len(df))
        return df["smiles"].tolist(), df["y"].to_numpy(dtype=float), meta

    df = _load_tdc(spec["tdc"], spec["group"])
    source = "tdc_acute" if endpoint == "ld50" else f"tdc:{spec['tdc']}"
    meta = dict(data_source=source, tdc=spec["tdc"], task=spec["task"], n=len(df),
                unit=spec.get("unit"), note=spec.get("note"))
    return df["smiles"].tolist(), df["y"].to_numpy(dtype=float), meta
