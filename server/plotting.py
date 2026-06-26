"""Matplotlib figures reproducing the paper's Figures 1-3."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import artifacts  # noqa: E402

_CLUSTER_COLORS = {
    "A": "#2ca02c", "B": "#9467bd", "C": "#1f77b4",
    "D": "#ff7f0e", "E": "#d62728",
}
_ROUTE_LABELS = {"oral": "Oral", "iv": "IV", "ip": "IP", "sc": "SC", "skin": "Skin", "im": "IM"}


def plot_cluster_map(ds, embedding: np.ndarray, labels: list[int], clusters: list[dict]) -> str:
    """Fig. 1 — 2D t-SNE chemical-space map coloured by cluster (A-E)."""
    letter_of = {}
    for c in clusters:
        for nm in c["members"]:
            letter_of[nm] = c["letter"][0]
    fig, ax = plt.subplots(figsize=(8, 6))
    for letter in sorted(set(letter_of.values())):
        idx = [i for i in range(ds.n) if letter_of.get(ds.names[i]) == letter]
        ax.scatter(embedding[idx, 0], embedding[idx, 1],
                   c=_CLUSTER_COLORS.get(letter, "#777777"), label=f"Cluster {letter}",
                   s=45, alpha=0.8, edgecolors="white", linewidths=0.5)
    outliers = [i for i in range(ds.n) if ds.paper_cluster[i] == "none"]
    if outliers:
        ax.scatter(embedding[outliers, 0], embedding[outliers, 1], marker="x",
                   c="black", s=70, label="unclustered")
    ax.set_title("Chemical-space map of H. sosnowskyi metabolites (open SynMap analogue)")
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.legend(loc="best", fontsize=8)
    return artifacts.save_figure(fig, "fig1_cluster_map")


def plot_ld50_by_route(table: dict[str, dict[str, float]], cluster_order: list[str]) -> str:
    """Fig. 2 — median LD50 (mg/kg) per cluster for each route of administration."""
    routes = [r for r in ["oral", "iv", "ip", "sc", "skin", "im"] if r in table]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(cluster_order))
    width = 0.8 / max(len(routes), 1)
    for j, route in enumerate(routes):
        vals = [table[route].get(c, np.nan) for c in cluster_order]
        ax.bar(x + j * width, vals, width, label=_ROUTE_LABELS.get(route, route))
    ax.set_xticks(x + width * (len(routes) - 1) / 2)
    ax.set_xticklabels([f"Cluster {c}" for c in cluster_order])
    ax.set_ylabel("Median predicted LD50 (mouse), mg/kg")
    ax.set_title("Median toxicity by cluster and route of administration")
    ax.legend(title="Route", fontsize=8)
    return artifacts.save_figure(fig, "fig2_ld50_routes")


def plot_tox_heatmap(names: list[str], endpoints: list[str], matrix: np.ndarray, title: str) -> str:
    """Fig. 3 / Table 2 — toxicity-endpoint heatmap for cluster-E compounds."""
    fig, ax = plt.subplots(figsize=(7, max(4, 0.35 * len(names))))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(endpoints)))
    ax.set_xticklabels(endpoints, rotation=30, ha="right")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="toxic probability / scaled value")
    return artifacts.save_figure(fig, "fig3_tox_heatmap")
