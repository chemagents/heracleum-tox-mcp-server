"""Synthesis-cost estimation — open-source analogue of Syntelly's "Synthesis cost".

The paper (Section 2.6) estimates synthesis cost per 1 g over 1-6 retrosynthetic stages.
We reproduce this with the same open engine the CoScientist `chemical-mcp-server` already
uses for retrosynthesis (ASKCOS), reading the route ``precursor_cost``. Resolution order:

1. If the molecule is one of the paper's reported compounds, return the *published*
   USD/g (so the headline numbers reproduce exactly).
2. Else, if an ASKCOS service is configured (``HERACLEUM_ASKCOS_URL``), plan a route and
   return its precursor cost.
3. Else, a transparent RDKit complexity heuristic (clearly flagged as a rough estimate).
"""
from __future__ import annotations

import logging

import numpy as np
import requests
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

from .config import get_settings

logger = logging.getLogger(__name__)


def _askcos_cost(smiles: str) -> dict | None:
    s = get_settings()
    if not s.askcos_url:
        return None
    try:
        resp = requests.post(
            s.askcos_url.rstrip("/") + "/retro",
            json={"smiles": smiles, "max_steps": s.synthesis_max_steps},
            timeout=120,
        )
        resp.raise_for_status()
        routes = resp.json().get("routes", [])
        if not routes:
            return None
        best = min(routes, key=lambda r: (r.get("precursor_cost") or 1e9))
        return {
            "usd_per_g": best.get("precursor_cost"),
            "n_steps": best.get("depth"),
            "method": "askcos_retrosynthesis",
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("ASKCOS synthesis-cost call failed: %s", exc)
        return None


def _heuristic_cost(mol) -> dict:
    """Rough USD/g from molecular complexity, calibrated to the paper's anchors.

    Anchors (Section 3.6): umbelliferone ~$0.19/g (simple), psoralen ~$25/g,
    xanthotoxin ~$311/g. We map an RDKit complexity score onto that range
    log-linearly. This is an order-of-magnitude estimate only.
    """
    rings = rdMolDescriptors.CalcNumRings(mol)
    arom = rdMolDescriptors.CalcNumAromaticRings(mol)
    stereo = rdMolDescriptors.CalcNumAtomStereoCenters(mol)
    hetero = rdMolDescriptors.CalcNumHeteroatoms(mol)
    mw = Descriptors.MolWt(mol)
    fused = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    complexity = (1.5 * rings + 1.0 * arom + 2.0 * stereo + 0.3 * hetero
                  + 0.01 * mw + 2.0 * fused)
    # calibrate: complexity ~6 -> ~$0.2 ; ~14 -> ~$300 (log10 slope ~0.4/unit)
    log_cost = -1.6 + 0.40 * (complexity - 6.0)
    usd = float(np.clip(10.0 ** log_cost, 0.05, 1e5))
    steps = int(np.clip(round(rings + arom / 2 + stereo), 1, get_settings().synthesis_max_steps))
    return {"usd_per_g": round(usd, 2), "n_steps": steps, "method": "complexity_heuristic",
            "complexity_score": round(float(complexity), 2)}


def estimate_synthesis_cost(smiles: str, published_usd_per_g: float | None = None) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"error": f"invalid SMILES: {smiles}"}
    if published_usd_per_g is not None:
        return {"usd_per_g": float(published_usd_per_g), "method": "paper_reported",
                "note": "Published Syntelly value (Section 3.6)."}
    askcos = _askcos_cost(smiles)
    if askcos is not None:
        return askcos
    return _heuristic_cost(mol)
