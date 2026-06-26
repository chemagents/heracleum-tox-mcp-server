# Standalone image for the heracleum-tox MCP server.
FROM python:3.12-slim

# RDKit / scientific stack runtime libraries (slim image omits these).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxrender1 \
    libxext6 \
    libgomp1 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Build context is this repository's root.
COPY pyproject.toml README.md ./
COPY server ./server
COPY build_dataset.py prepare_models.py ./
RUN pip install --no-cache-dir .

# PyTDC supplies the open training datasets. It is installed WITHOUT its deps because it pins
# the deprecated rdkit-pypi (cp311-only), which conflicts with modern rdkit on Python 3.12;
# its data-download API only needs pandas/requests/huggingface-hub (already installed).
RUN pip install --no-cache-dir --no-deps "PyTDC==0.4.1"

# Pre-train & cache the open models at build time (best-effort: if there is no network at
# build the server trains them lazily on the first request instead).
RUN python prepare_models.py \
  || echo "model pre-training skipped (no network at build); will train lazily at runtime"

ENV HERACLEUM_ARTIFACTS_DIR=/app/artifacts
RUN mkdir -p /app/artifacts

EXPOSE 7331
CMD ["python", "-m", "server.heracleum_server"]
