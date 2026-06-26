#!/usr/bin/env python
"""Pre-train and cache all open toxicity models (LD50 + 4 endpoints).

Run once so the server responds instantly on first request. Downloads the open TDC
datasets and trains CatBoost/XGBoost, caching to ``HERACLEUM_MODEL_CACHE_DIR``.

    uv run python prepare_models.py
"""
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

from server import models                       # noqa: E402
from server.data_sources import ENDPOINTS       # noqa: E402


def main() -> None:
    for endpoint in ENDPOINTS:
        m = models.get_model(endpoint)
        print(f"  {endpoint:16s} {m.backend:9s} {m.metric_name}={m.metric_value:.3f} "
              f"n_train={m.n_train} -> cached")
    print("All models trained and cached.")


if __name__ == "__main__":
    main()
