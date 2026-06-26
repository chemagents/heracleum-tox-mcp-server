"""Open-source CatBoost / XGBoost toxicity models + applicability domain.

This is the analogue of Syntelly's proprietary predictive models. We reproduce the
*method* described by the platform paper (Molecules 2024, 29, 1826) and used by the
Heracleum study: fingerprint-based CatBoost (regression: RMSE) and fragment-based
XGBoost (classification: ROC-AUC), with a kNN(k=5) + Gaussian applicability domain.

Models are trained lazily on the open datasets (``data_sources``) and cached to disk
(``HERACLEUM_MODEL_CACHE_DIR``) so the server is fast after the first call.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from rdkit import Chem
from sklearn.metrics import roc_auc_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

from . import chemistry, data_sources
from .config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class TrainedModel:
    endpoint: str
    task: str                       # "regression" | "classification"
    backend: str                    # "catboost" | "xgboost"
    estimator: object
    train_fp: np.ndarray            # (n_train, nbits) uint8 ECFP4 — for the AD kNN
    ad_threshold: float             # learnt threshold distance (paper Section 2.5)
    metric_name: str                # "RMSE" | "ROC-AUC"
    metric_value: float
    n_train: int
    radius: int
    nbits: int
    meta: dict

    # --- applicability domain (Section 2.5: kNN k=5 + Gaussian) ----------------
    def ad_percent(self, mols, k: int = 5) -> np.ndarray:
        """Applicability-domain reliability (0-100 %) for each query molecule."""
        q = _fp_matrix(mols, self.radius, self.nbits)           # (m, nbits)
        d = _knn_distance(q, self.train_fp, k)                  # mean dist to k NN
        ad = np.exp(-((d / max(self.ad_threshold, 1e-9)) ** 2))  # Gaussian -> 0..1
        return np.clip(ad, 0.0, 1.0) * 100.0

    def predict(self, mols) -> np.ndarray:
        X = chemistry.featurize(mols, self.backend, self.radius, self.nbits)
        if self.task == "classification":
            proba = _predict_proba(self.estimator, X)
            return proba
        return self.estimator.predict(X)


# --------------------------------------------------------------------------- #
# Fingerprint helpers for the applicability domain (Tanimoto kNN).
# --------------------------------------------------------------------------- #
def _fp_matrix(mols, radius: int, nbits: int) -> np.ndarray:
    return np.vstack([chemistry.morgan_fp(m, radius, nbits) for m in mols]).astype(np.uint8)


def _knn_distance(query: np.ndarray, train: np.ndarray, k: int) -> np.ndarray:
    """Mean Tanimoto distance from each query fp to its k nearest training fps."""
    q_cnt = query.sum(axis=1)                       # (m,)
    t_cnt = train.sum(axis=1)                        # (n,)
    inter = query @ train.T                          # (m, n) shared bits
    union = q_cnt[:, None] + t_cnt[None, :] - inter
    sim = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
    dist = 1.0 - sim
    k = min(k, dist.shape[1])
    part = np.partition(dist, k - 1, axis=1)[:, :k]  # k smallest distances
    return part.mean(axis=1)


def _learn_ad_threshold(train: np.ndarray, k: int, sigma: float, sample: int = 500) -> float:
    """Predefined per-model threshold = mean + sigma*std of in-training kNN distances."""
    rng = np.random.default_rng(0)
    idx = rng.choice(train.shape[0], size=min(sample, train.shape[0]), replace=False)
    # distance to the (k+1)-th neighbour to exclude self (distance 0).
    sub = train[idx]
    q_cnt = sub.sum(axis=1)
    t_cnt = train.sum(axis=1)
    inter = sub @ train.T
    union = q_cnt[:, None] + t_cnt[None, :] - inter
    sim = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
    dist = 1.0 - sim
    part = np.partition(dist, k, axis=1)[:, 1:k + 1]   # exclude self (col 0)
    nn = part.mean(axis=1)
    return float(nn.mean() + sigma * nn.std())


def _predict_proba(estimator, X) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    return estimator.predict(X)


# --------------------------------------------------------------------------- #
# Estimators (open analogues of the Syntelly CatBoost / XGBoost models).
# --------------------------------------------------------------------------- #
def _make_estimator(task: str, backend: str):
    if backend == "xgboost":
        import xgboost as xgb

        common = dict(n_estimators=400, max_depth=6, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                      random_state=get_settings().random_state)
        if task == "classification":
            return xgb.XGBClassifier(eval_metric="logloss", **common)
        return xgb.XGBRegressor(eval_metric="rmse", **common)

    from catboost import CatBoostClassifier, CatBoostRegressor

    common = dict(iterations=500, depth=6, learning_rate=0.05, random_seed=get_settings().random_state,
                  verbose=False, allow_writing_files=False)
    if task == "classification":
        return CatBoostClassifier(loss_function="Logloss", **common)
    return CatBoostRegressor(loss_function="RMSE", **common)


def _parse(smiles: list[str], y: np.ndarray):
    mols, yy = [], []
    for s, v in zip(smiles, y):
        m = Chem.MolFromSmiles(s)
        if m is not None and np.isfinite(v):
            mols.append(m)
            yy.append(float(v))
    return mols, np.array(yy, dtype=float)


def _cache_path(cache_key: str) -> Path:
    d = Path(get_settings().model_cache_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{cache_key}.joblib"


def train_model(endpoint: str, route: str | None = None) -> TrainedModel:
    settings = get_settings()
    backend = settings.model_backend
    spec = data_sources.ENDPOINTS[endpoint]
    task = spec["task"]
    # classification uses the fragment-based XGBoost per the paper; regression the
    # fingerprint-based CatBoost. Honour an explicit backend override otherwise.
    if settings.model_backend == "catboost" and task == "classification":
        backend = "xgboost"

    smiles, y, dmeta = data_sources.load_training_data(endpoint, route)
    mols, y = _parse(smiles, y)
    X = chemistry.featurize(mols, backend, settings.morgan_radius, settings.morgan_nbits)

    stratify = y if task == "classification" else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=settings.random_state, stratify=stratify
    )
    est = _make_estimator(task, backend)
    est.fit(X_tr, y_tr)
    if task == "classification":
        proba = _predict_proba(est, X_te)
        metric_name, metric_value = "ROC-AUC", float(roc_auc_score(y_te, proba))
    else:
        metric_name, metric_value = "RMSE", float(root_mean_squared_error(y_te, est.predict(X_te)))

    # Refit on the full set for deployment.
    est = _make_estimator(task, backend)
    est.fit(X, y)

    train_fp = _fp_matrix(mols, settings.morgan_radius, settings.morgan_nbits)
    ad_threshold = _learn_ad_threshold(train_fp, settings.ad_k_neighbors, settings.ad_threshold_sigma)

    model = TrainedModel(
        endpoint=endpoint, task=task, backend=backend, estimator=est,
        train_fp=train_fp, ad_threshold=ad_threshold,
        metric_name=metric_name, metric_value=metric_value, n_train=len(mols),
        radius=settings.morgan_radius, nbits=settings.morgan_nbits,
        meta={**dmeta, "route": route, "section": spec.get("section"), "desc": spec.get("desc")},
    )
    return model


def get_model(endpoint: str, route: str | None = None) -> TrainedModel:
    settings = get_settings()
    key = endpoint if not (endpoint == "ld50" and route and data_sources._local_csv(f"ld50_{route}")) else f"ld50_{route}"
    cache_key = f"{key}__{('xgboost' if (settings.model_backend=='catboost' and data_sources.ENDPOINTS[endpoint]['task']=='classification') else settings.model_backend)}"
    path = _cache_path(cache_key)
    if path.exists() and not settings.retrain:
        try:
            return joblib.load(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load cached model %s (%s); retraining.", path, exc)
    model = train_model(endpoint, route)
    joblib.dump(model, path)
    logger.info("Trained & cached %s (%s=%.3f, n=%d)", cache_key, model.metric_name,
                model.metric_value, model.n_train)
    return model
