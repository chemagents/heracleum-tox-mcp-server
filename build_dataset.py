#!/usr/bin/env python
"""Build the full 225-compound Heracleum sosnowskyi dataset of Rassabina & Fedorov 2025.

Source of truth = the paper's Supplementary Tables S1-S5 (standardized SMILES + SynID for
clusters A-E), extracted by ``parse_supplementary.py`` into ``server/data/supplementary_smiles.csv``.
This script assembles the final bundled dataset:

  * 222 clustered compounds (A 25, B 22, C 132, D 21, E 22) from the Supplementary, using the
    paper's *standardized* SMILES and SynID and the paper's cluster assignment — exact.
  * the 3 explicitly un-clustered molecules (Section 3.2) with their published IV LD50.
  * for cluster E, the published toxicity reference values (Table 2 + Section 3.4/3.6): acute
    LD50 (IV/oral), hepatotoxicity/DILI/cardiotoxicity/carcinogenicity + applicability domains,
    and synthesis cost — merged onto the Supplementary structures by canonical SMILES.

Compound names: cluster E + the 3 outliers use their common names; clusters A-D are resolved
from SMILES via PubChem (cached in ``server/data/names_cache.csv``), falling back to
``<class>:<SynID>`` so the build always completes offline after the cache is populated.

Run:  uv run python build_dataset.py   ->  server/data/heracleum_metabolites.csv
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).resolve().parent
SUPP = HERE / "server" / "data" / "supplementary_smiles.csv"
NAMES_CACHE = HERE / "server" / "data" / "names_cache.csv"
OUT = HERE / "server" / "data" / "heracleum_metabolites.csv"

EXP = "experimental"  # sentinel: a Table-2 cell holding a data-source tag, not an AD%

CLASS_BY_CLUSTER = {
    "A": "terpenoid", "B": "polyphenol_flavonoid", "C": "fatty_acid",
    "D": "aromatic", "E": "furanocoumarin",
}

# --------------------------------------------------------------------------- #
# Cluster E — published reference values (Table 2 + Section 3.4 / 3.6).
# tox cells: (effect, ad_percent_or_None). ad=None means the paper printed a data-source tag.
# Structures come from the Supplementary; these records supply names + reference values,
# merged by canonical SMILES (with a 1:1 fallback for the one epoxide form mismatch).
# --------------------------------------------------------------------------- #
CLUSTER_E = [
    dict(name="bergamottin", klass="furanocoumarin", ld50_iv=62.0, ad_ld50_iv=74,
         hepato=("Toxic", 90), dili=("Toxic", 41), cardio=("Nontoxic", 67), carcino=("Nontoxic", 48)),
    dict(name="phellopterin", klass="furanocoumarin", ld50_iv=62.0, ad_ld50_iv=60,
         hepato=("Toxic", 91), dili=("Toxic", 41), cardio=("Nontoxic", None), carcino=("Nontoxic", 58)),
    dict(name="osthol", klass="coumarin",
         hepato=("Toxic", None), dili=("Toxic", 48), cardio=("Nontoxic", None), carcino=("Toxic", 79)),
    dict(name="pimpinellin", klass="furanocoumarin",
         hepato=("Toxic", 88), dili=("Toxic", 40), cardio=("Nontoxic", 51), carcino=("Nontoxic", 51)),
    dict(name="isopimpinellin", klass="furanocoumarin",
         hepato=("Toxic", 79), dili=("Toxic", 37), cardio=("Nontoxic", None), carcino=("Nontoxic", 45)),
    dict(name="isoimperatorin", klass="furanocoumarin",
         hepato=("Toxic", None), dili=("Toxic", 46), cardio=("Nontoxic", None), carcino=("Toxic", 76)),
    dict(name="imperatorin", klass="furanocoumarin",
         hepato=("Toxic", None), dili=("Toxic", 43), cardio=("Nontoxic", None), carcino=("Nontoxic", 61)),
    dict(name="trioxsalen", klass="furanocoumarin",
         hepato=("Toxic", None), dili=("Toxic", 53), cardio=("Nontoxic", None), carcino=("Toxic", 88)),
    dict(name="oxypeucedanin", klass="furanocoumarin",
         hepato=("Toxic", 38), dili=("Toxic", 29), cardio=("Nontoxic", None), carcino=("Toxic", 26)),
    dict(name="pangelin", klass="furanocoumarin",
         hepato=("Toxic", 61), dili=("Toxic", 41), cardio=("Nontoxic", None), carcino=("Nontoxic", 77)),
    dict(name="xanthotoxin", klass="furanocoumarin", ld50_oral=423.0, syn_cost=311.0,
         hepato=("Toxic", 89), dili=("Toxic", 41), cardio=("Nontoxic", None), carcino=("Toxic", 45)),
    dict(name="heraclenol", klass="furanocoumarin",
         hepato=("Nontoxic", 92), dili=("Toxic", 41), cardio=("Nontoxic", None), carcino=("Nontoxic", 58)),
    dict(name="oxypeucedanin hydrate", klass="furanocoumarin",
         hepato=("Nontoxic", 90), dili=("Toxic", 41), cardio=("Nontoxic", None), carcino=("Nontoxic", 57)),
    dict(name="sphondin", klass="furanocoumarin",
         hepato=("Toxic", 89), dili=("Toxic", 46), cardio=("Nontoxic", None), carcino=("Toxic", 61)),
    dict(name="byakangelicin", klass="furanocoumarin",
         hepato=("Nontoxic", 89), dili=("Toxic", 40), cardio=("Nontoxic", None), carcino=("Nontoxic", 47)),
    dict(name="isobergapten", klass="furanocoumarin",
         hepato=("Toxic", 82), dili=("Toxic", 42), cardio=("Nontoxic", 54), carcino=("Toxic", 47)),
    dict(name="bergapten", klass="furanocoumarin",
         hepato=("Toxic", None), dili=("Toxic", 41), cardio=("Nontoxic", None), carcino=("Toxic", 48)),
    dict(name="psoralen", klass="furanocoumarin", syn_cost=24.9,
         hepato=("Toxic", None), dili=("Toxic", 51), cardio=("Nontoxic", None), carcino=("Toxic", 79)),
    dict(name="isopsoralen", klass="furanocoumarin",
         hepato=("Toxic", 93), dili=("Toxic", 42), cardio=("Nontoxic", 23), carcino=("Toxic", 54)),
    dict(name="quininic acid", klass="aromatic_acid",
         hepato=("Nontoxic", 29), dili=("Toxic", 63), cardio=("Nontoxic", 66), carcino=("Nontoxic", 49)),
    dict(name="scopoletin", klass="coumarin", ld50_iv=350.0,
         hepato=("Nontoxic", None), dili=("Toxic", 51), cardio=("Nontoxic", None), carcino=("Nontoxic", None)),
    dict(name="umbelliferone", klass="coumarin", ld50_iv=450.0, syn_cost=0.19,
         hepato=("Toxic", None), dili=("Toxic", 57), cardio=("Nontoxic", 67), carcino=("Nontoxic", 46)),
]

# The three molecules the paper reports as not falling into any cluster (Section 3.2).
UNCLUSTERED = [
    dict(name="byakangelicol", klass="furanocoumarin", ld50_iv=94.0),
    dict(name="gamma-bisabolene", klass="terpenoid", ld50_iv=401.0, smiles="CC(C)=CCCC(C)=CC1CCC(C)=CC1"),
    dict(name="alpha-terpinolene", klass="terpenoid", ld50_iv=133.0, smiles="CC1=CCC(=CC1)C(C)C"),
]


def canon(smiles: str | None) -> str | None:
    if not smiles:
        return None
    m = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(m) if m is not None else None


def flat(smiles: str | None) -> str | None:
    """Stereo-free canonical SMILES — for merging curated cluster-E records onto the paper's
    standardized structures (PubChem name->SMILES stereo may differ from the Supplementary)."""
    if not smiles:
        return None
    m = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(m, isomericSmiles=False) if m is not None else None


def _load_cache() -> dict[str, str]:
    if NAMES_CACHE.exists():
        df = pd.read_csv(NAMES_CACHE)
        return dict(zip(df["smiles"], df["name"]))
    return {}


def _save_cache(cache: dict[str, str]) -> None:
    pd.DataFrame(sorted(cache.items()), columns=["smiles", "name"]).to_csv(NAMES_CACHE, index=False)


def _pick_name(comp) -> str | None:
    syns = comp.synonyms or []
    cand = [s for s in syns if 2 < len(s) <= 35 and any(c.isalpha() for c in s)
            and not re.fullmatch(r"[0-9\-]+", s)]
    alpha = [s for s in cand if not any(c.isdigit() for c in s)]
    if alpha:
        return alpha[0].lower()
    if cand:
        return cand[0].lower()
    return comp.iupac_name


def resolve_name(smiles: str, cache: dict[str, str], offline: bool) -> str | None:
    if smiles in cache:
        return cache[smiles]
    if offline:
        return None
    try:
        import pubchempy as pcp

        comps = pcp.get_compounds(smiles, "smiles")
        time.sleep(0.2)
        if comps:
            nm = _pick_name(comps[0])
            if nm:
                cache[smiles] = nm
                return nm
    except Exception as exc:  # noqa: BLE001
        print(f"  ! PubChem name failed for {smiles[:40]}…: {exc}", file=sys.stderr)
    return None


def _tox_cols(rec: dict, prefix: str) -> dict:
    effect, ad = rec.get(prefix, (None, None))
    source = EXP if (effect is not None and ad is None) else None
    return {f"ref_{prefix}": effect, f"ref_{prefix}_ad": ad, f"ref_{prefix}_source": source}


def _ref_cols(rec: dict) -> dict:
    cols = {
        "ref_ld50_iv_mgkg": rec.get("ld50_iv"),
        "ref_ld50_iv_ad": rec.get("ad_ld50_iv"),
        "ref_ld50_oral_mgkg": rec.get("ld50_oral"),
        "ref_synthesis_cost_usd_g": rec.get("syn_cost"),
    }
    for p in ("hepato", "dili", "cardio", "carcino"):
        cols.update(_tox_cols(rec, p))
    return cols


def build(offline: bool = False) -> pd.DataFrame:
    if not SUPP.exists():
        sys.exit(f"Missing {SUPP}. Run parse_supplementary.py first.")
    supp = pd.read_csv(SUPP)
    cache = _load_cache()

    # Curated cluster-E records keyed by stereo-free canonical SMILES (+ usage tracking for
    # the 1:1 fallback). Flat matching tolerates stereo differences between PubChem and the paper.
    cured_by_smiles: dict[str, dict] = {}
    for rec in CLUSTER_E:
        smi = flat(rec.get("smiles") or _name_smiles(rec["name"]))
        if smi:
            cured_by_smiles[smi] = rec

    rows: list[dict] = []
    used_names: set[str] = set()
    unmatched_e_rows: list[int] = []

    for _, r in supp.iterrows():
        smi = canon(r["smiles"])
        cl = r["cluster"]
        klass = CLASS_BY_CLUSTER[cl]
        row = {"name": None, "smiles": smi, "class": klass, "paper_cluster": cl,
               "synid": int(r["synid"])}
        if cl == "E" and flat(r["smiles"]) in cured_by_smiles:
            rec = cured_by_smiles[flat(r["smiles"])]
            row["name"] = rec["name"]
            row["class"] = rec["klass"]
            row.update(_ref_cols(rec))
            used_names.add(rec["name"])
        elif cl == "E":
            unmatched_e_rows.append(len(rows))
        if row["name"] is None:
            row["name"] = resolve_name(smi, cache, offline) or f"{klass}:{int(r['synid'])}"
        rows.append(row)

    # Pair the single SMILES-unmatched cluster-E row with the single unused curated record
    # (the epoxide-form mismatch: oxypeucedanin).
    leftover = [rec for rec in CLUSTER_E if rec["name"] not in used_names]
    if len(unmatched_e_rows) == 1 and len(leftover) == 1:
        rec = leftover[0]
        row = rows[unmatched_e_rows[0]]
        row["name"] = rec["name"]
        row["class"] = rec["klass"]
        row.update(_ref_cols(rec))

    # The three explicitly un-clustered molecules.
    for rec in UNCLUSTERED:
        smi = canon(rec.get("smiles") or _name_smiles(rec["name"]))
        row = {"name": rec["name"], "smiles": smi, "class": rec["klass"],
               "paper_cluster": "none", "synid": ""}
        row.update(_ref_cols(rec))
        rows.append(row)

    _save_cache(cache)
    df = pd.DataFrame(rows).drop_duplicates(subset="smiles").reset_index(drop=True)
    # stable column order
    front = ["name", "smiles", "synid", "class", "paper_cluster"]
    df = df[front + [c for c in df.columns if c not in front]]
    return df


_name_cache: dict[str, str] = {}


def _name_smiles(name: str) -> str | None:
    """Resolve a curated compound name to SMILES via PubChem (for the 3 outliers / matching)."""
    if name in _name_cache:
        return _name_cache[name]
    try:
        import pubchempy as pcp

        comps = pcp.get_compounds(name, "name")
        time.sleep(0.2)
        if comps:
            smi = comps[0].canonical_smiles
            _name_cache[name] = smi
            return smi
    except Exception:  # noqa: BLE001
        pass
    return None


def main() -> None:
    offline = "--offline" in sys.argv
    df = build(offline=offline)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Wrote {len(df)} compounds -> {OUT}")
    print(df.groupby("paper_cluster").size().to_string())


if __name__ == "__main__":
    main()
