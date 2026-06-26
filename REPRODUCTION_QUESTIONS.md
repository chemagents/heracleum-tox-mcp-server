# Reproducing the paper's *assertions* via CoScientist

Two layers of reproduction:

1. **Numbers / figures** — the MCP tools compute them (live open models + deterministic analyses).
2. **Assertions (conclusions)** — natural-language statements the paper draws *from* those
   numbers. The Orchestrator LLM turns numbers → statements.

How the numbers reach the LLM in CoScientist:

```
user question
  -> OrchestratorAgent              (plans, then delegates)
     -> TaskExecutorAgent
        -> ToolPreparerAgent        (ToolRetrieverAgent + ToolReranker: RAG finds heracleum-tox tools)
        -> ExperimentAgent          (FEDOT.MAS calls the MCP tool over HTTP, gets the JSON result)
  -> OrchestratorAgent              (LLM composes the final natural-language answer)
```

Every tool returns a `finding` field, and **`reproduce_claims`** returns, per claim, a
`reproduced_statement` (the paper's assertion restated with our numbers) plus `evidence` and a
`method` flag (live model / deterministic / published value). The LLM can relay these verbatim
(exact reproduction) or synthesise from `evidence`.

## Fastest path: one question

> "Using the heracleum-tox tools, reproduce all the findings of Rassabina & Fedorov 2025 on the
> toxicological profile of Heracleum sosnowskyi metabolites, and state each conclusion with the
> supporting numbers."

Routes to `reproduce_claims` (all 8 conclusions + numbers) and/or `reproduce_all` (headline
values vs. the paper). The `answer.narrative` field is the full reproduced summary.

## Per-assertion questions

| # | Question to ask CoScientist | Tool | Reproduced assertion |
|---|---|---|---|
| C1 | What metabolites of H. sosnowskyi were collected and how are they grouped? | `dataset_overview` | 225 metabolites (here reconstructed; cluster E exact). |
| C2 | How do the metabolites cluster in chemical space? | `chemical_space_clustering` | Five families: terpenoids, polyphenols/flavonoids, fatty acids, aromatics, furanocoumarins. |
| C3 | Which cluster is most toxic and by which route? | `predict_ld50` | Cluster E (furanocoumarins); worst by IV (62–450 mg/kg). |
| C4 | Which furanocoumarins are most/least toxic by IV? | `predict_ld50` / `dataset_overview` | Most: bergamottin, phellopterin (62); least: scopoletin (350), umbelliferone (450). |
| C5 | Do furanocoumarins cause liver injury but little cardiotoxicity? | `predict_general_toxicity` | Paper: DILI yes, cardiotox low. **Open models diverge** (documented). |
| C6 | How reliable are the predictions, and how good are the models? | `applicability_domain`, `model_quality` | AD kNN(k=5); ROC-AUC 79–93 %. |
| C7 | What is the spread of synthesis cost? | `estimate_synthesis_cost` | $0.19/g (umbelliferone) → $311/g (xanthotoxin). |
| C8 | What is the main toxicological hazard of H. sosnowskyi? | `reproduce_claims` | Furanocoumarins and coumarin derivatives. |

## Bonus: profile any molecule

> "Give the full in-silico toxicology profile (LD50, hepatotoxicity, DILI, cardiotoxicity,
> carcinogenicity, applicability domain, synthesis cost) of bergapten."

Routes to `predict_molecule_profile`.

## Keeping LLM synthesis faithful (optional system prompt)

> "You are reproducing Rassabina & Fedorov 2025 with open-source analogues of Syntelly. Call the
> heracleum-tox tools, then state each conclusion **using only the returned numbers**. Where a
> tool returns a `finding` or `reproduced_statement`, treat it as authoritative. Report any value
> that differs from the paper and say by how much — in particular, surface the documented
> DILI/cardiotoxicity divergence (C5) rather than hiding it."
