# Integrating heracleum-tox into CoScientist

This is a standard CoScientist MCP server (FastMCP, HTTP `/mcp`, every tool returns
`{"answer": ..., "metadata": ...}`). It has been **verified end-to-end inside CoScientist**:
a user query flows `OrchestratorAgent → RAG tool-retrieval → ExperimentAgent (FEDOT.MAS) →
this server's tools → synthesized answer`. Integration is three steps; the verification log is
at the bottom so you can see it working before you wire it in.

## 1. Place the server

Clone (or add as a submodule) into the CoScientist `mcp-servers/` directory:

```bash
cd CoScientist/mcp-servers
git clone https://github.com/chemagents/heracleum-tox-mcp-server
```

## 2. Add the docker-compose service

Append this to `mcp-servers/docker-compose.yml`. The build context is the repo root, so use the
provided **`Dockerfile.coscientist`** (the plain `Dockerfile` is for standalone `context: .`):

```yaml
  heracleum-tox-mcp-server:
    build:
      context: ..
      dockerfile: mcp-servers/heracleum-tox-mcp-server/Dockerfile.coscientist
    container_name: heracleum-tox-mcp-server
    env_file:
      - ./heracleum-tox-mcp-server/.env
    environment:
      PYTHONUNBUFFERED: "1"
    ports:
      - "7336:7331"
    restart: unless-stopped
```

```bash
cp heracleum-tox-mcp-server/.env.example heracleum-tox-mcp-server/.env
docker compose up -d --build heracleum-tox-mcp-server
```

The build installs PyTDC and pre-trains the open models (best-effort; otherwise they train
lazily on the first request). To run it outside CoScientist, the repo's root `Dockerfile` +
`docker-compose.yml` already work with `docker compose up`.

## 3. Register it in the RAG

```bash
# from the CoScientist repo root, with the RAG stack (Postgres + Qdrant + embedder) running
python scripts/rag_tools/cli.py load mcp-servers/heracleum-tox-mcp-server/rag_registration.json
```

After this the `ToolRetrieverAgent` surfaces the tools for plant-metabolite / toxicity / LD50 /
furanocoumarin queries, and the `ExperimentAgent` (FEDOT.MAS) calls them by URL. If CoScientist
and this server share a Docker network, register the in-network URL instead:
`http://heracleum-tox-mcp-server:7331/mcp`.

That's it. The 10 tools (`dataset_overview`, `chemical_space_clustering`, `predict_ld50`,
`predict_general_toxicity`, `applicability_domain`, `estimate_synthesis_cost`,
`predict_molecule_profile`, `model_quality`, `reproduce_all`, `reproduce_claims`) are now
available to the agents. See [`REPRODUCTION_QUESTIONS.md`](./REPRODUCTION_QUESTIONS.md) for
example prompts.

---

## Verified end-to-end run (proof it integrates)

**Environment.** CoScientist on OpenRouter (`openrouter/qwen/qwen3-235b-a22b-2507`) via litellm;
RAG stack = Postgres (5432) + Qdrant (6333) + embedding API (5002); `mcp` 1.28.0.

**Registration**

```text
$ python scripts/rag_tools/cli.py load mcp-servers/heracleum-tox-mcp-server/rag_registration.json
✅ Added: heracleum-tox
```

**RAG retrieval** surfaces the tools for a relevant query:

```text
query: "hepatotoxicity and carcinogenicity prediction of plant metabolites"
  - predict_general_toxicity   score=0.46
  - dataset_overview           score=0.37
```

**Full query → FEDOT.MAS calls the server's tools** (CoScientist runtime log):

```text
fedotmas.core.runner   Pipeline run started   pipeline=heracleum_coordinator
fedotmas.plugins.logging  Tool call   agent=heracleum_coordinator tool=transfer_to_agent -> clustering_worker
fedotmas.plugins.logging  Tool call   agent=clustering_worker     tool=chemical_space_clustering args={'n_clusters': 5}
fedotmas.plugins.logging  Tool result agent=clustering_worker     tool=chemical_space_clustering
fedotmas.core.runner   Pipeline complete   total_elapsed=17.1s
```

**Synthesized final answer** (OrchestratorAgent, composed from this server's results):

```text
The metabolite dataset comprises 225 compounds ... grouped into five clusters (A–E):
  Cluster C: fatty acids (132), Cluster A: terpenoids (25),
  Cluster E: furanocoumarins/coumarins (22), Cluster B (22), Cluster D: aromatics (21).
The most toxic cluster is Cluster E (furanocoumarin family).
```

These numbers match the server's reproduction of Rassabina & Fedorov 2025 (Plants 14, 3253)
exactly — see [`README.md`](./README.md).

## Note for CoScientist maintainers (optional, unrelated to this server)

On `mcp >= 1.28`, `tests/unit/test_mcp_patches.py` has 2 failures: those tests assert the
SSE-truncation backport is *active*, but `CoScientist/tools/mcp_patches.py` only activates it on
the broken `mcp <= 1.27` (installed `mcp` 1.28.0 already fixes the bug, so the patch is correctly
inactive). Guarding the two behavioural tests with
`@pytest.mark.skipif(not mcp_patches._mcp_is_broken(), reason="patch only active on mcp<=1.27")`
makes the suite green. No change to this server is required.
