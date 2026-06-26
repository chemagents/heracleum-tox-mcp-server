"""Chemical-space clustering — open-source analogue of Syntelly's SynMap module.

The paper (Section 2.3) clusters the metabolites with SynMap: *parametric multiscale
t-SNE on differential fingerprints*, yielding five families (Fig. 1). The open analogue:

    ECFP4 (Morgan r=2) fingerprints
        -> Tanimoto distance matrix
        -> AgglomerativeClustering(n_clusters=5, linkage="average")   (the 5 clusters)
        -> t-SNE(2D) embedding for the Fig.-1 scatter map

Each computed cluster is labelled A-E by its dominant chemical family, reproducing the
paper's assignment (A terpenoids, B polyphenols/flavonoids, C fatty acids, D aromatics,
E furanocoumarins/coumarins).
"""
from __future__ import annotations

from collections import Counter

import numpy as np
from rdkit import DataStructs
from sklearn.cluster import AgglomerativeClustering

from . import chemistry
from .config import get_settings
from .dataset import CLASS_TO_CLUSTER, CLUSTER_FAMILIES, Dataset


def _bitvects(ds: Dataset):
    s = get_settings()
    return [chemistry.morgan_bitvect(m, s.morgan_radius, s.morgan_nbits) for m in ds.mols]


def tanimoto_distance_matrix(ds: Dataset) -> np.ndarray:
    fps = _bitvects(ds)
    n = len(fps)
    D = np.zeros((n, n), dtype=np.float64)
    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        d = 1.0 - np.asarray(sims)
        D[i, :i] = d
        D[:i, i] = d
    return D


def tsne_embedding(ds: Dataset, perplexity: float | None = None) -> np.ndarray:
    from sklearn.manifold import TSNE

    s = get_settings()
    X = np.vstack([chemistry.morgan_fp(m, s.morgan_radius, s.morgan_nbits) for m in ds.mols])
    perp = perplexity if perplexity is not None else min(s.tsne_perplexity, max(5, (ds.n - 1) / 3))
    emb = TSNE(n_components=2, init="pca", random_state=s.random_state,
               perplexity=perp).fit_transform(X)
    return emb


def cluster_metabolites(ds: Dataset, n_clusters: int | None = None) -> dict:
    """Agglomerative clustering on Tanimoto distance; label clusters A-E by family."""
    s = get_settings()
    k = n_clusters or s.n_clusters
    D = tanimoto_distance_matrix(ds)
    labels = AgglomerativeClustering(
        n_clusters=k, metric="precomputed", linkage="average"
    ).fit_predict(D)

    clusters = []
    used_letters: set[str] = set()
    for cid in range(k):
        idx = np.where(labels == cid)[0]
        classes = [ds.classes[i] for i in idx]
        dominant = Counter(classes).most_common(1)[0][0]
        letter = CLASS_TO_CLUSTER.get(dominant, "?")
        clusters.append({
            "cluster_id": int(cid),
            "letter": letter,
            "dominant_class": dominant,
            "family": CLUSTER_FAMILIES.get(letter, dominant),
            "size": int(len(idx)),
            "members": [ds.names[i] for i in idx],
            "indices": idx.tolist(),
        })

    # Resolve duplicate letters (keep the larger cluster's letter, suffix the rest).
    by_letter: dict[str, list] = {}
    for c in clusters:
        by_letter.setdefault(c["letter"], []).append(c)
    for letter, group in by_letter.items():
        if len(group) > 1:
            group.sort(key=lambda c: c["size"], reverse=True)
            for j, c in enumerate(group[1:], start=2):
                c["letter"] = f"{letter}{j}"

    clusters.sort(key=lambda c: c["size"], reverse=True)

    # Agreement with the paper's stated cluster for the compounds we know (esp. E).
    known = [(i, ds.paper_cluster[i]) for i in range(ds.n) if ds.paper_cluster[i] in CLUSTER_FAMILIES]
    label_of = {i: c["letter"][0] for c in clusters for i in c["indices"]}
    agree = sum(1 for i, pc in known if label_of.get(i) == pc)
    purity = agree / len(known) if known else None

    outliers = [ds.names[i] for i in range(ds.n) if ds.paper_cluster[i] == "none"]
    return {
        "n_clusters": k,
        "clusters": [{kk: vv for kk, vv in c.items() if kk != "indices"} for c in clusters],
        "labels": labels.tolist(),
        "paper_outliers": outliers,
        "family_agreement": purity,
        "n_known_labeled": len(known),
    }
