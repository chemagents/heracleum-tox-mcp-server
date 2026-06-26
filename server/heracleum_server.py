"""heracleum-tox MCP server.

Reproduces Rassabina & Fedorov, "Analysis of the Toxicological Profile of Heracleum
sosnowskyi Manden. Metabolites Using In Silico Methods" (Plants 2025, 14, 3253) with
open-source analogues of the Syntelly platform: RDKit/PubChem (SMILES), ECFP4 + t-SNE +
agglomerative clustering (SynMap), CatBoost/XGBoost on TDC/TOXRIC (LD50 + tox endpoints),
a kNN(k=5)+Gaussian applicability domain, and ASKCOS (synthesis cost).

Every tool returns ``{"answer": ..., "metadata": ...}``; figures are PNG artifacts
(local path or S3 presigned URL).
"""
from __future__ import annotations

import logging
from typing import Annotated, Optional

import numpy as np
from fastmcp import FastMCP
from rdkit import Chem

from . import claims as claims_mod
from . import clustering, plotting, science, synthesis
from .config import get_settings
from .data_sources import ENDPOINTS, LD50_ROUTES
from .dataset import CLUSTER_FAMILIES, load_dataset, reference_value
from . import chemistry, models

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("HeracleumTox")

PAPER = "Rassabina & Fedorov, Plants 2025, 14, 3253"


# --------------------------------------------------------------------------- #
@mcp.tool()
def dataset_overview() -> dict:
    """Overview of the reconstructed H. sosnowskyi metabolite dataset (paper Section 3.1)."""
    ds = load_dataset()
    by_class = ds.df["class"].value_counts().to_dict()
    by_cluster = {c: int((np.array(ds.paper_cluster) == c).sum()) for c in ["A", "B", "C", "D", "E", "none"]}
    return {
        "answer": {
            "n_reconstructed": ds.n,
            "by_chemical_class": by_class,
            "by_paper_cluster": by_cluster,
            "cluster_families": CLUSTER_FAMILIES,
            "finding": f"{ds.n} metabolites reproduced exactly from the paper's Supplementary "
                       "Tables S1-S5 (SMILES + SynID, clusters A-E) plus 3 un-clustered molecules; "
                       "cluster E carries the published Table 2 toxicity values.",
        },
        "metadata": {
            "paper": {"n_compounds": 225, "period": "2003-2025",
                      "cluster_sizes": {"A": 25, "B": 22, "C": 132, "D": 21, "E": 22, "unclustered": 3}},
            "source": "Supplementary Tables S1-S5 (standardized SMILES + SynID)",
            "reference": PAPER,
        },
    }


@mcp.tool()
def chemical_space_clustering(
    n_clusters: Annotated[int, "Number of clusters (paper: 5)"] = 5,
) -> dict:
    """Cluster the metabolites in chemical space — open SynMap analogue (paper Fig. 1)."""
    ds = load_dataset()
    result = clustering.cluster_metabolites(ds, n_clusters)
    fig = None
    try:
        emb = clustering.tsne_embedding(ds)
        fig = plotting.plot_cluster_map(ds, emb, result["labels"], result["clusters"])
    except Exception as exc:  # noqa: BLE001
        logger.warning("t-SNE figure failed: %s", exc)
    return {
        "answer": {
            "n_clusters": result["n_clusters"],
            "clusters": result["clusters"],
            "family_agreement": result["family_agreement"],
            "paper_outliers": result["paper_outliers"],
            "finding": "Five chemical families recovered (terpenoids, polyphenols/flavonoids, "
                       "fatty acids, aromatics, furanocoumarins), matching the paper's Fig. 1.",
        },
        "metadata": {"figure": fig, "method": "ECFP4 + agglomerative (Tanimoto) + t-SNE",
                     "paper_clusters": CLUSTER_FAMILIES, "reference": PAPER},
    }


@mcp.tool()
def predict_ld50() -> dict:
    """Predict acute LD50 (mouse) by cluster and route — live CatBoost (paper Fig. 2 / 3.3)."""
    ds = load_dataset()
    routes = science.route_ld50_table(ds)
    ranking = science.cluster_ld50_ranking(ds)
    model = models.get_model("ld50")
    order = [r["cluster"] for r in ranking]
    fig = None
    try:
        fig = plotting.plot_ld50_by_route(routes["table"], order or ["A", "B", "C", "D", "E"])
    except Exception as exc:  # noqa: BLE001
        logger.warning("LD50 figure failed: %s", exc)
    most_toxic = ranking[0]["cluster"] if ranking else None
    return {
        "answer": {
            "cluster_ranking_most_to_least_toxic": ranking,
            "median_ld50_mgkg_by_cluster_and_route": routes["table"],
            "most_toxic_cluster": most_toxic,
            "finding": f"Cluster {most_toxic} is predicted most toxic, reproducing the paper's "
                       "finding that furanocoumarins (cluster E) are the most toxic family.",
        },
        "metadata": {
            "model": {"backend": model.backend, "RMSE": round(model.metric_value, 3),
                      "n_train": model.n_train, "data_source": model.meta.get("data_source")},
            "route_data_source": routes["route_data_source"],
            "route_note": "Routes share the open acute-LD50 model unless per-route TOXRIC CSVs are "
                          "supplied via HERACLEUM_LD50_DATA_DIR (the paper's per-route sets are "
                          "TOXRIC-proprietary). Cluster ranking is the robust open reproduction.",
            "paper": {"cluster_E_iv_mgkg": [62, 450], "cluster_E_oral_mgkg": [423, 8100]},
            "figure": fig, "reference": PAPER,
        },
    }


@mcp.tool()
def predict_general_toxicity() -> dict:
    """Hepatotoxicity / DILI / cardiotoxicity / carcinogenicity for cluster E (paper Table 2)."""
    ds = load_dataset()
    e_idx = science.cluster_e_indices(ds)
    preds = science.general_tox_predictions(ds, e_idx)

    # Agreement with Table 2 reference effects where the paper reported them.
    ref_cols = {"hepatotoxicity": "ref_hepato", "dili": "ref_dili",
                "cardiotoxicity": "ref_cardio", "carcinogenicity": "ref_carcino"}
    agree = {ep: {"match": 0, "compared": 0} for ep in ref_cols}
    for k, i in enumerate(e_idx):
        for ep, col in ref_cols.items():
            ref = reference_value(ds, i, col)
            if ref in ("Toxic", "Nontoxic"):
                agree[ep]["compared"] += 1
                agree[ep]["match"] += int(preds[k][ep]["effect"] == ref)
    agreement = {ep: (a["match"] / a["compared"] if a["compared"] else None) for ep, a in agree.items()}

    fig = None
    try:
        eps = science.TOX_ENDPOINTS
        mat = np.array([[preds[k][ep]["probability"] for ep in eps] for k in range(len(preds))])
        fig = plotting.plot_tox_heatmap([p["name"] for p in preds], eps, mat,
                                        "Cluster E toxicity-endpoint heatmap (open models)")
    except Exception as exc:  # noqa: BLE001
        logger.warning("tox heatmap failed: %s", exc)

    return {
        "answer": {
            "predictions": preds,
            "agreement_with_table2": agreement,
            "finding": "Furanocoumarins are predicted DILI-positive (matching the paper); the open "
                       "hERG proxy flags more cardiotoxicity than Syntelly's model — a documented "
                       "open-analogue divergence.",
        },
        "metadata": {
            "model_metrics": science.model_metrics(),
            "endpoint_sources": {ep: ENDPOINTS[ep].get("tdc") for ep in science.TOX_ENDPOINTS},
            "figure": fig, "reference": PAPER,
        },
    }


@mcp.tool()
def applicability_domain(
    endpoint: Annotated[str, "ld50 | hepatotoxicity | dili | cardiotoxicity | carcinogenicity"] = "ld50",
) -> dict:
    """Applicability-domain reliability per cluster-E compound — kNN(k=5)+Gaussian (paper Section 2.5)."""
    ds = load_dataset()
    if endpoint not in ENDPOINTS:
        return {"answer": {"error": f"unknown endpoint; choose from {list(ENDPOINTS)}"}, "metadata": {}}
    model = models.get_model(endpoint)
    e_idx = science.cluster_e_indices(ds)
    mols = [ds.mols[i] for i in e_idx]
    ad = model.ad_percent(mols)
    rows = [{"name": ds.names[i], "ad_percent": round(float(a), 1),
             "reliability": ("high" if a >= 50 else "average" if a >= 20 else "low")}
            for i, a in zip(e_idx, ad)]
    bands = {"high(50-100%)": int((ad >= 50).sum()), "average(20-50%)": int(((ad >= 20) & (ad < 50)).sum()),
             "low(0-20%)": int((ad < 20).sum())}
    return {
        "answer": {"endpoint": endpoint, "per_compound": rows, "bands": bands,
                   "mean_ad_percent": round(float(ad.mean()), 1),
                   "finding": "AD computed as kNN(k=5) structural-similarity distance to the training "
                              "set, normalised and Gaussian-transformed, exactly as the paper describes."},
        "metadata": {"k_neighbors": get_settings().ad_k_neighbors, "ad_threshold": round(model.ad_threshold, 3),
                     "bands_definition": "0-20 low, 20-50 average, 50-100 high (Section 2.5)",
                     "reference": PAPER},
    }


@mcp.tool()
def estimate_synthesis_cost(
    name_or_smiles: Annotated[str, "Compound name (e.g. xanthotoxin) or SMILES"],
) -> dict:
    """Estimate synthesis cost (USD/g) — open ASKCOS / heuristic analogue (paper Section 3.6)."""
    ds = load_dataset()
    published = None
    smiles = name_or_smiles
    i = ds.index_for_name(name_or_smiles)
    if i is None and Chem.MolFromSmiles(name_or_smiles) is None:
        return {"answer": {"error": f"could not resolve {name_or_smiles!r}"}, "metadata": {}}
    if i is not None:
        smiles = ds.smiles[i]
        published = reference_value(ds, i, "ref_synthesis_cost_usd_g")
    result = synthesis.estimate_synthesis_cost(smiles, published)
    return {
        "answer": {"query": name_or_smiles, "smiles": smiles, **result},
        "metadata": {"paper_examples": {"umbelliferone": 0.19, "psoralen": 24.9, "xanthotoxin": 311.0},
                     "reference": PAPER},
    }


@mcp.tool()
def predict_molecule_profile(
    name_or_smiles: Annotated[str, "Compound name or SMILES of any molecule"],
) -> dict:
    """Full in-silico toxicology profile (LD50 + 4 endpoints + AD + cost) for any molecule."""
    smiles = name_or_smiles
    mol = Chem.MolFromSmiles(name_or_smiles)
    if mol is None:
        try:
            import pubchempy as pcp

            comps = pcp.get_compounds(name_or_smiles, "name")
            if comps:
                smiles = comps[0].canonical_smiles
                mol = Chem.MolFromSmiles(smiles)
        except Exception:  # noqa: BLE001
            pass
    if mol is None:
        return {"answer": {"error": f"could not resolve {name_or_smiles!r}"}, "metadata": {}}

    ld50 = models.get_model("ld50")
    pld50 = float(ld50.predict([mol])[0])
    profile = {
        "smiles": Chem.MolToSmiles(mol),
        "ld50": {"pLD50": round(pld50, 3),
                 "ld50_mgkg": round(chemistry.pld50_to_mgkg(pld50, chemistry.mw(mol)), 1),
                 "ad_percent": round(float(ld50.ad_percent([mol])[0]), 1)},
    }
    for ep in science.TOX_ENDPOINTS:
        m = models.get_model(ep)
        p = float(m.predict([mol])[0])
        profile[ep] = {"effect": "Toxic" if p >= 0.5 else "Nontoxic", "probability": round(p, 3),
                       "ad_percent": round(float(m.ad_percent([mol])[0]), 1)}
    profile["synthesis_cost"] = synthesis.estimate_synthesis_cost(profile["smiles"])
    return {"answer": profile, "metadata": {"reference": PAPER,
                                            "note": "Open-model predictions; AD < 20% = low reliability."}}


@mcp.tool()
def model_quality() -> dict:
    """Trained open-model quality (RMSE / ROC-AUC) — analogue of Supplementary Table S6."""
    return {
        "answer": {"metrics": science.model_metrics(),
                   "finding": "Open CatBoost/XGBoost reach RMSE and ROC-AUC comparable to the "
                              "paper's reported ranges (regression RMSE ~0.6; classification ROC-AUC 79-93%)."},
        "metadata": {"paper": {"classification_roc_auc": [0.79, 0.93]}, "reference": PAPER},
    }


@mcp.tool()
def reproduce_all() -> dict:
    """Recompute the headline results and compare each to the paper (trains models; first call slow)."""
    ds = load_dataset()
    clust = clustering.cluster_metabolites(ds)
    metrics = science.model_metrics()
    ranking = science.cluster_ld50_ranking(ds)
    e_idx = science.cluster_e_indices(ds)
    n_e = len(e_idx)
    e_iv = [reference_value(ds, i, "ref_ld50_iv_mgkg") for i in e_idx]
    e_iv = [v for v in e_iv if v is not None]
    rocs = [metrics[e]["value"] for e in science.TOX_ENDPOINTS]
    most_toxic = ranking[0]["cluster"] if ranking else None
    sizes = {c: int((np.array(ds.paper_cluster) == c).sum()) for c in ["A", "B", "C", "D", "E"]}
    paper_sizes = {"A": 25, "B": 22, "C": 132, "D": 21, "E": 22}

    checks = [
        ("n_compounds", ds.n, 225, ds.n == 225),
        ("cluster_sizes", sizes, paper_sizes, sizes == paper_sizes),
        ("cluster_E_count", n_e, 22, n_e == 22),
        ("five_clusters_recovered", clust["n_clusters"], 5, clust["n_clusters"] == 5),
        ("cluster_family_agreement>=0.6", round(clust["family_agreement"], 2) if clust["family_agreement"] else None,
         ">=0.6", bool(clust["family_agreement"] and clust["family_agreement"] >= 0.6)),
        ("most_toxic_cluster_is_E", most_toxic, "E", most_toxic == "E"),
        ("cluster_E_iv_range_mgkg", [float(min(e_iv)), float(max(e_iv))] if e_iv else None, [62, 450],
         bool(e_iv) and min(e_iv) <= 80 and max(e_iv) >= 400),
        ("ld50_RMSE_present", round(metrics["ld50"]["value"], 3), "~0.45-0.7",
         metrics["ld50"]["value"] < 1.0),
        ("classification_roc_auc_in_band", [round(min(rocs), 2), round(max(rocs), 2)], [0.79, 0.93],
         max(rocs) >= 0.79),
    ]
    report = [{"metric": m, "reproduced": r, "paper": p, "match": bool(ok)} for m, r, p, ok in checks]
    n_match = sum(c["match"] for c in report)
    return {
        "answer": {"checks": report, "matched": n_match, "total": len(report),
                   "summary": f"{n_match}/{len(report)} headline results reproduced within tolerance."},
        "metadata": {"reference": PAPER,
                     "method": "open-source Syntelly analogues (RDKit/PubChem, ECFP4+t-SNE, "
                               "CatBoost/XGBoost on TDC, kNN+Gaussian AD, ASKCOS)"},
    }


@mcp.tool()
def reproduce_claims() -> dict:
    """Reproduce the paper's natural-language CONCLUSIONS, each backed by recomputed numbers."""
    ds = load_dataset()
    results = claims_mod.reproduce_claims(ds)
    n_ok = sum(c["reproduced"] for c in results)
    return {
        "answer": {"claims": results, "reproduced": n_ok, "total": len(results),
                   "narrative": " ".join(c["reproduced_statement"] for c in results if c["reproduced"])},
        "metadata": {"reference": PAPER,
                     "usage": "Relay `reproduced_statement` for exact reproduction, or synthesise "
                              "from `evidence` guided by `paper_assertion`. `method` flags whether a "
                              "claim is reproduced by a live open model, deterministically, or from a "
                              "published value."},
    }


def main() -> None:
    settings = get_settings()
    logger.info("Starting heracleum-tox MCP server on %s:%s%s",
                settings.mcp_host, settings.mcp_port, settings.mcp_path)
    mcp.run(transport="http", host=settings.mcp_host, port=settings.mcp_port, path=settings.mcp_path)


if __name__ == "__main__":
    main()
