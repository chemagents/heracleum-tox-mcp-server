"""High-level reproductions tying the open models to the paper's result blocks.

Kept free of MCP/FastMCP imports so the test-suite can call these directly.
"""
from __future__ import annotations

import numpy as np

from . import chemistry, models
from .data_sources import LD50_ROUTES, _local_csv
from .dataset import CLASS_TO_CLUSTER, Dataset

TOX_ENDPOINTS = ["hepatotoxicity", "dili", "cardiotoxicity", "carcinogenicity"]


def cluster_letter(ds: Dataset, i: int) -> str:
    """Paper cluster if known, else inferred from chemical class."""
    pc = ds.paper_cluster[i]
    if pc in ("A", "B", "C", "D", "E"):
        return pc
    if pc == "none":
        return "none"
    return CLASS_TO_CLUSTER.get(ds.classes[i], "?")


def cluster_e_indices(ds: Dataset) -> list[int]:
    """Canonical paper cluster E (the 22 furanocoumarins/coumarins of Table 2)."""
    return [i for i in range(ds.n) if ds.paper_cluster[i] == "E"]


# --------------------------------------------------------------------------- #
# LD50 (Section 3.3 / Fig. 2)
# --------------------------------------------------------------------------- #
def acute_ld50_predictions(ds: Dataset) -> list[dict]:
    model = models.get_model("ld50")
    pld50 = model.predict(ds.mols)
    ad = model.ad_percent(ds.mols)
    out = []
    for i in range(ds.n):
        mw = chemistry.mw(ds.mols[i])
        out.append({
            "name": ds.names[i],
            "cluster": cluster_letter(ds, i),
            "pLD50": round(float(pld50[i]), 3),
            "ld50_mgkg": round(chemistry.pld50_to_mgkg(float(pld50[i]), mw), 1),
            "ad_percent": round(float(ad[i]), 1),
        })
    return out


def route_ld50_table(ds: Dataset) -> dict:
    """Median LD50 (mg/kg) per cluster x route.

    A route uses its own model only if a local per-route training CSV is present
    (``HERACLEUM_LD50_DATA_DIR/ld50_<route>.csv``); otherwise it shares the open
    acute-LD50 model (flagged), since TOXRIC's per-route sets are not openly scriptable.
    """
    table: dict[str, dict[str, float]] = {}
    sources: dict[str, str] = {}
    base = models.get_model("ld50")
    for route in LD50_ROUTES:
        model = models.get_model("ld50", route) if _local_csv(f"ld50_{route}") else base
        sources[route] = model.meta.get("data_source", "tdc_acute")
        pld50 = model.predict(ds.mols)
        per_cluster: dict[str, list[float]] = {}
        for i in range(ds.n):
            mgkg = chemistry.pld50_to_mgkg(float(pld50[i]), chemistry.mw(ds.mols[i]))
            per_cluster.setdefault(cluster_letter(ds, i), []).append(mgkg)
        table[route] = {c: round(float(np.median(v)), 1) for c, v in per_cluster.items()}
    return {"table": table, "route_data_source": sources}


def cluster_ld50_ranking(ds: Dataset) -> list[dict]:
    preds = acute_ld50_predictions(ds)
    by_cluster: dict[str, list[float]] = {}
    for p in preds:
        by_cluster.setdefault(p["cluster"], []).append(p["ld50_mgkg"])
    rows = [{"cluster": c, "median_ld50_mgkg": round(float(np.median(v)), 1), "n": len(v)}
            for c, v in by_cluster.items() if c not in ("none", "?")]
    rows.sort(key=lambda r: r["median_ld50_mgkg"])  # most toxic (lowest) first
    return rows


# --------------------------------------------------------------------------- #
# General toxicity classification (Section 3.5 / Table 2)
# --------------------------------------------------------------------------- #
def general_tox_predictions(ds: Dataset, indices: list[int]) -> list[dict]:
    fitted = {ep: models.get_model(ep) for ep in TOX_ENDPOINTS}
    mols = [ds.mols[i] for i in indices]
    proba = {ep: m.predict(mols) for ep, m in fitted.items()}
    ad = {ep: m.ad_percent(mols) for ep, m in fitted.items()}
    out = []
    for k, i in enumerate(indices):
        row = {"name": ds.names[i], "cluster": cluster_letter(ds, i)}
        for ep in TOX_ENDPOINTS:
            p = float(proba[ep][k])
            row[ep] = {
                "effect": "Toxic" if p >= 0.5 else "Nontoxic",
                "probability": round(p, 3),
                "ad_percent": round(float(ad[ep][k]), 1),
            }
        out.append(row)
    return out


def model_metrics() -> dict:
    """Trained-model quality (RMSE / ROC-AUC) vs the paper's Supplementary Table S6."""
    from .data_sources import PAPER_METRICS

    out = {}
    for ep in ["ld50", *TOX_ENDPOINTS]:
        m = models.get_model(ep)
        paper = PAPER_METRICS.get(ep)
        out[ep] = {"backend": m.backend, "metric": m.metric_name,
                   "value": round(m.metric_value, 3), "n_train": m.n_train,
                   "data_source": m.meta.get("data_source"),
                   "paper_value": paper[1] if paper else None}
    return out
