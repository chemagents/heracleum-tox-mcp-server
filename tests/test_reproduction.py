"""Reproduction tests for Rassabina & Fedorov (Plants 2025, 14, 3253).

Deterministic tests (dataset, clustering, chemistry) run offline.
Model tests are marked ``slow``/``network`` (they train CatBoost/XGBoost on TDC data):

    uv run pytest tests -v                  # all
    uv run pytest tests -v -m "not slow"    # fast deterministic checks only
"""
import pytest

from server import chemistry, science
from server.clustering import cluster_metabolites
from server.dataset import cluster_e_mask, load_dataset, reference_value

ds = load_dataset()


# --------------------------------------------------------------------------- #
# Deterministic (offline)
# --------------------------------------------------------------------------- #
def test_dataset_reconstructed():
    assert ds.n >= 60
    assert int(cluster_e_mask(ds).sum()) == 22          # paper cluster E
    for name in ["bergamottin", "psoralen", "xanthotoxin", "umbelliferone", "bergapten"]:
        assert ds.index_for_name(name) is not None, name


def test_published_reference_values():
    i = ds.index_for_name("bergamottin")
    assert reference_value(ds, i, "ref_ld50_iv_mgkg") == 62
    assert reference_value(ds, ds.index_for_name("umbelliferone"), "ref_ld50_iv_mgkg") == 450
    assert reference_value(ds, ds.index_for_name("xanthotoxin"), "ref_ld50_oral_mgkg") == 423
    assert reference_value(ds, ds.index_for_name("xanthotoxin"), "ref_synthesis_cost_usd_g") == 311.0
    assert reference_value(ds, ds.index_for_name("umbelliferone"), "ref_synthesis_cost_usd_g") == 0.19


def test_clustering_recovers_five_families():
    res = cluster_metabolites(ds, 5)
    assert res["n_clusters"] == 5
    assert res["family_agreement"] >= 0.6           # cluster labels match the paper's families
    assert set(res["paper_outliers"]) == {"byakangelicol", "gamma-bisabolene", "alpha-terpinolene"}


def test_chemistry_pld50_conversion():
    # round-trip pLD50 <-> mg/kg
    mgkg = chemistry.pld50_to_mgkg(3.0, 216.19)     # MW of xanthotoxin
    assert chemistry.mgkg_to_pld50(mgkg, 216.19) == pytest.approx(3.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# Live open models (train on TDC; slow + network)
# --------------------------------------------------------------------------- #
@pytest.mark.slow
@pytest.mark.network
def test_ld50_model_and_ranking():
    ranking = science.cluster_ld50_ranking(ds)
    assert ranking[0]["cluster"] == "E"             # furanocoumarins most toxic
    metrics = science.model_metrics()
    assert metrics["ld50"]["value"] < 1.0           # RMSE in -log10(mol/kg)


@pytest.mark.slow
@pytest.mark.network
def test_classifier_roc_auc_band():
    metrics = science.model_metrics()
    rocs = [metrics[e]["value"] for e in science.TOX_ENDPOINTS]
    assert max(rocs) >= 0.79                          # paper: ROC-AUC 79-93 %


@pytest.mark.slow
@pytest.mark.network
def test_applicability_domain_range():
    m = science.models.get_model("ld50")
    ad = m.ad_percent([ds.mols[i] for i in science.cluster_e_indices(ds)])
    assert (ad >= 0).all() and (ad <= 100).all()


@pytest.mark.slow
@pytest.mark.network
def test_claims_reproduce():
    from server import claims

    results = claims.reproduce_claims(ds)
    reproduced = sum(c["reproduced"] for c in results)
    assert len(results) == 8
    assert reproduced >= 7                            # >=7/8 (C5 is a documented open-analogue divergence)
