# heracleum-tox-mcp-server

An MCP server that **reproduces the results** of:

> Rassabina, A.E.; Fedorov, M.V.
> *Analysis of the Toxicological Profile of Heracleum sosnowskyi Manden. Metabolites Using In Silico Methods.*
> **Plants 2025, 14, 3253.** https://doi.org/10.3390/plants14213253

The paper runs entirely on the proprietary **Syntelly** platform. Since Syntelly is not
openly accessible, this server reproduces the same pipeline with **open-source analogues
of every Syntelly module** — including the very models Syntelly itself uses
(fingerprint-based CatBoost + fragment-based XGBoost; Sosnin et al., *Molecules* **2024**, 29, 1826,
the platform paper cited as ref. [36]), trained on the same open datasets the paper names
(TOXRIC / ChemIDplus / PyTDC).

## Syntelly → open-source analogue mapping

| Syntelly module (in the paper) | What it does | Open-source analogue used here |
|---|---|---|
| Canonical SMILES search | name → SMILES, standardisation | **RDKit + PubChemPy** |
| **SynMap** (clustering, §2.3) | parametric multiscale t-SNE + differential fingerprints | **ECFP4 (Morgan) + agglomerative (Tanimoto) + t-SNE** |
| **LD50 (mouse) prediction** (§2.4) | fingerprint-CatBoost regression, RMSE | **CatBoost** on ECFP4+descriptors, trained on **TDC `LD50_Zhu`** (TOXRIC hook for exact routes) |
| **General toxicity** (§3.5) | CatBoost/XGBoost classification, ROC-AUC | **XGBoost** on fragment descriptors, trained on **TDC `DILI` / `hERG` / `Carcinogens_Lagunin`** |
| **Applicability Domain** (§2.5) | kNN(k=5) distance → normalise → Gaussian → % | **Tanimoto kNN(k=5) + Gaussian**, identical formula |
| **Synthesis cost** (§2.6) | USD/g over 1–6 stages | **ASKCOS** retrosynthesis (same engine as `chemical-mcp-server`) + heuristic fallback |

## Tools

| Tool | Reproduces | What it returns |
|------|-----------|-----------------|
| `dataset_overview` | §3.1 | reconstructed metabolite dataset, class & cluster breakdown |
| `chemical_space_clustering` | Fig. 1 | five chemical-family clusters (A–E) + t-SNE map + outliers |
| `predict_ld50` | Fig. 2 / §3.3 | live CatBoost acute-LD50; cluster ranking + per-route table |
| `predict_general_toxicity` | Table 2 / §3.5 | hepatotox / DILI / cardiotox / carcinogenicity for cluster E + heatmap |
| `applicability_domain` | Fig. S1/S2 / §2.5 | kNN(k=5)+Gaussian AD % per cluster-E compound, banded |
| `estimate_synthesis_cost` | §3.6 | USD/g (published value, ASKCOS, or heuristic) |
| `predict_molecule_profile` | — | full in-silico tox profile for **any** molecule (name/SMILES) |
| `model_quality` | Table S6 | trained-model RMSE / ROC-AUC |
| `reproduce_all` | — | recomputes headline numbers and compares to the paper |
| `reproduce_claims` | all | the paper's conclusions, each restated with reproduced numbers |

Each tool returns `{"answer": ..., "metadata": ...}`. Figures are saved as PNG to a local
artifacts dir (`HERACLEUM_ARTIFACTS_DIR`) or, if S3 is configured, uploaded and returned as
presigned URLs (same pattern as `chemical-mcp-server` / `tox-antitargets-mcp-server`).

## Reproduction fidelity

`reproduce_all` and `pytest tests/` assert these against the paper:

| Metric | Paper | This server |
|---|---|---|
| Cluster E (furanocoumarins) size | 22 | **22 (exact)** |
| Chemical-space clusters | 5 families | **5, ≥70 % family agreement** |
| Most-toxic cluster | E (furanocoumarins) | **E** |
| Cluster-E IV LD50 range | 62–450 mg/kg | **62–450 (bergamottin/phellopterin 62, umbelliferone 450)** |
| LD50 regression error | RMSE "~45 %" | **RMSE 0.60 (−log mol/kg)** |
| Tox classification ROC-AUC | 0.79–0.93 | **0.80–0.87** |
| Synthesis-cost spread | $0.19–311/g | **$0.19 / $24.9 / $311 (exact)** |

**Documented open-analogue divergences** (faithful method; the small open datasets disagree
with Syntelly's proprietary models):

- *DILI / hepatotoxicity*: the open **TDC `DILI`** model (n=475) predicts most cluster-E
  furanocoumarins as **non**-hepatotoxic, opposite to Syntelly's "all DILI-toxic". The
  applicability domain flags these as moderate-reliability — an honest signal that the open
  set under-covers furanocoumarins. This is the one paper claim (C5) that does **not**
  reproduce, and it is reported as such.
- *Cardiotoxicity*: the open **hERG** proxy is more conservative than Syntelly's cardiotox
  model (it flags furanocoumarins as hERG blockers; the paper found none).
- *Per-route LD50*: TOXRIC's six per-route mouse sets are not openly scriptable, so all
  routes share the open acute-LD50 model unless you supply per-route CSVs (see below). The
  **cluster ranking** (E most toxic) is the robust open reproduction.
- *Full 225-compound set*: Supplementary Tables S1–S5 (SMILES+SynID) are not downloadable,
  so the dataset is reconstructed from the compound **names** in the paper — cluster E (the
  quantitative core) exactly; clusters A–D with representative members.

## Run locally

```bash
git clone https://github.com/chemagents/heracleum-tox-mcp-server
cd heracleum-tox-mcp-server
cp .env.example .env
uv sync
uv pip install --no-deps "PyTDC==0.4.1"     # open datasets; pins old rdkit-pypi, so --no-deps
uv run python build_dataset.py              # reconstruct the dataset (resolves SMILES via PubChem)
uv run python prepare_models.py             # train & cache the open models (downloads TDC data)
uv run python -m server.heracleum_server    # serves http://0.0.0.0:7331/mcp
```

## Run with Docker

```bash
docker compose up -d --build      # host port 7336 -> container 7331
```

To run it inside the CoScientist stack instead, add this repo as a service in
`mcp-servers/docker-compose.yml` (the CoScientist repo already includes such an entry).

The Docker build installs PyTDC and pre-trains the models (best-effort; if there is no
network at build time the server trains them lazily on first request).

## Attach to CoScientist

CoScientist discovers MCP tools via RAG (Postgres + Qdrant). Register this server once:

```bash
# from the CoScientist repo root, with the RAG stack running and .env configured
python scripts/rag_tools/cli.py load mcp-servers/heracleum-tox-mcp-server/rag_registration.json
# or directly:
python scripts/rag_tools/cli.py add \
  --url http://localhost:7336/mcp \
  --name heracleum-tox \
  --description "In-silico toxicology of Heracleum sosnowskyi metabolites; LD50, hepato/DILI/cardio/carcinogenicity, furanocoumarins (Rassabina & Fedorov 2025)"
```

After registration the `ToolRetrieverAgent` surfaces these tools for plant-metabolite /
toxicity / LD50 / furanocoumarin queries, and `ExperimentAgent` (FEDOT.MAS) calls them by
URL. If CoScientist runs in the same Docker network, register the in-network URL instead:
`http://heracleum-tox-mcp-server:7331/mcp`.

See [`REPRODUCTION_QUESTIONS.md`](./REPRODUCTION_QUESTIONS.md) for the exact prompts to ask
CoScientist (one per paper assertion, plus a single "reproduce everything" prompt).

## Exact per-route LD50 reproduction (optional)

The paper predicts LD50 for six mouse routes from TOXRIC. To reproduce those exactly, drop
TOXRIC per-route CSVs (`smiles,y` with `y = -log10(mol/kg)`) named `ld50_<route>.csv`
(`oral,iv,ip,sc,skin,im`) into `HERACLEUM_LD50_DATA_DIR`; route-specific models then train
automatically.

## Tests

```bash
uv run pytest tests -v                 # all (trains models on first run, then cached)
uv run pytest tests -v -m "not slow"   # fast deterministic checks only
```

## License / data

Open datasets via Therapeutics Data Commons (PyTDC) and TOXRIC. Please cite
Rassabina & Fedorov (2025) when using these results, and TDC / the Syntelly platform paper
(Sosnin et al., *Molecules* 2024, 29, 1826) for the methods.
