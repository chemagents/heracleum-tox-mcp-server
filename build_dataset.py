#!/usr/bin/env python
"""Reconstruct the Heracleum sosnowskyi metabolite dataset of Rassabina & Fedorov 2025.

The paper's full 225-compound dataset lives in Supplementary Tables S1-S5 (SMILES +
SynID) which are not openly downloadable. Per the chosen reproduction strategy we
*reconstruct the dataset from the compound names stated in the paper* and resolve each
to a canonical SMILES via PubChem (RDKit-canonicalised), with a manual SMILES fallback.

Cluster E (the 22 furanocoumarins / coumarins of Table 2) is reconstructed **exactly**,
together with every published reference value the paper reports for it:
  - acute LD50 (mouse, intravenous), mg/kg                       (Section 3.4 / Fig. 3)
  - hepatotoxicity / DILI / cardiotoxicity / carcinogenicity     (Table 2)
    each as a binary effect + an applicability-domain percentage
  - synthesis cost, USD per gram                                 (Section 3.6)
The three explicitly un-clustered molecules are also reconstructed exactly.
Clusters A-D (terpenoids, polyphenols/flavonoids, fatty acids, aromatics) are
populated with representative, literature-plausible H. sosnowskyi constituents so the
five-cluster structure (Fig. 1) is reproducible; they are NOT the paper's exact members.

Run:  uv run python build_dataset.py   ->  server/data/heracleum_metabolites.csv
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent / "server" / "data" / "heracleum_metabolites.csv"

# Paper cluster letters -> chemical family (Section 3.2 / Fig. 1)
#   A terpenoids (isoprenoids) | B polyphenolic glycosides, flavonoids, cyclic polyols
#   C fatty acids             | D aromatic compounds      | E furanocoumarins & coumarins
EXP = "experimental"  # sentinel: a Table-2 cell that holds a data-source tag, not an AD%

# --------------------------------------------------------------------------- #
# Cluster E - the paper's quantitative core (Table 2 + Section 3.4 + 3.6).
# tox cells: (effect, ad_percent_or_None). ad=None means the paper printed a
# data-source tag (TOXRIC / PyTDC / NCATS-Flux) i.e. an aggregated experimental value.
# --------------------------------------------------------------------------- #
CLUSTER_E = [
    # name, class, ld50_iv, syn_cost, hepato, dili, cardio, carcino
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
    dict(name="xanthotoxin", klass="furanocoumarin", ld50_oral=423.0, syn_cost=311.0,  # 423 = experimental ORAL (Section 3.4)
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
    dict(name="isopsoralen", klass="furanocoumarin",  # = angelicin
         hepato=("Toxic", 93), dili=("Toxic", 42), cardio=("Nontoxic", 23), carcino=("Toxic", 54)),
    dict(name="quininic acid", klass="aromatic_acid",  # 6-methoxyquinoline-4-carboxylic acid
         hepato=("Nontoxic", 29), dili=("Toxic", 63), cardio=("Nontoxic", 66), carcino=("Nontoxic", 49)),
    dict(name="scopoletin", klass="coumarin", ld50_iv=350.0,
         hepato=("Nontoxic", None), dili=("Toxic", 51), cardio=("Nontoxic", None), carcino=("Nontoxic", None)),
    dict(name="umbelliferone", klass="coumarin", ld50_iv=450.0, syn_cost=0.19,
         hepato=("Toxic", None), dili=("Toxic", 57), cardio=("Nontoxic", 67), carcino=("Nontoxic", 46)),
]

# The three molecules the paper reports as *not* falling into any cluster (Section 3.2).
UNCLUSTERED = [
    dict(name="byakangelicol", klass="furanocoumarin", ld50_iv=94.0),
    dict(name="gamma-bisabolene", klass="terpenoid", ld50_iv=401.0, smiles="CC(C)=CCCC(C)=CC1CCC(C)=CC1"),
    dict(name="alpha-terpinolene", klass="terpenoid", ld50_iv=133.0, smiles="CC1=CCC(=CC1)C(C)C"),
]

# Other furanocoumarins explicitly named in the paper (Intro / 3.1) but not in Table 2.
NAMED_OTHER = [
    dict(name="pabulenol", klass="furanocoumarin"),
    dict(name="columbianetin", klass="furanocoumarin"),
    dict(name="heraclenin", klass="furanocoumarin"),
    dict(name="alloisoimperatorin", klass="furanocoumarin"),
    dict(name="angelicin", klass="furanocoumarin"),
]

# --------------------------------------------------------------------------- #
# Representative members of clusters A-D (NOT the paper's exact lists, which are in
# the un-downloadable supplementary). Chosen as plausible Apiaceae / H. sosnowskyi
# constituents so the five-cluster structure of Fig. 1 reproduces.
# --------------------------------------------------------------------------- #
CLUSTER_A_TERPENOIDS = ["limonene", "alpha-pinene", "beta-pinene", "myrcene", "sabinene",
                        "gamma-terpinene", "beta-caryophyllene", "humulene", "ocimene",
                        "alpha-phellandrene", "terpinen-4-ol", "borneol"]
CLUSTER_B_POLYPHENOLS = ["quercetin", "rutin", "kaempferol", "apigenin", "luteolin",
                         "chlorogenic acid", "myo-inositol", "quinic acid", "catechin", "umbelliferone-glucoside"]
CLUSTER_C_FATTY_ACIDS = ["palmitic acid", "stearic acid", "oleic acid", "linoleic acid",
                         "alpha-linolenic acid", "lauric acid", "myristic acid", "arachidic acid",
                         "behenic acid", "petroselinic acid", "palmitoleic acid", "capric acid",
                         "caprylic acid", "pentadecanoic acid", "margaric acid", "lignoceric acid",
                         "eicosenoic acid", "vaccenic acid"]
CLUSTER_D_AROMATICS = ["vanillin", "ferulic acid", "caffeic acid", "p-coumaric acid",
                       "syringic acid", "eugenol", "benzoic acid", "salicylic acid",
                       "gallic acid", "protocatechuic acid", "sinapic acid", "vanillic acid"]

# Manual SMILES fallback for names PubChem may resolve ambiguously.
SMILES_FALLBACK = {
    "psoralen": "O=c1ccc2cc3ccoc3cc2o1",
    "angelicin": "O=c1ccc2ccc3occc3c2o1",
    "isopsoralen": "O=c1ccc2ccc3occc3c2o1",
    "bergapten": "COc1c2ccoc2cc2oc(=O)ccc12",
    "xanthotoxin": "COc1c2occc2cc2ccc(=O)oc12",
    "umbelliferone": "O=c1ccc2ccc(O)cc2o1",
    "scopoletin": "COc1cc2ccc(=O)oc2cc1O",
    "quininic acid": "COc1ccc2nccc(C(=O)O)c2c1",
}


def resolve_smiles(name: str, fallback: str | None = None) -> str | None:
    from rdkit import Chem

    raw = None
    if fallback:
        raw = fallback
    if raw is None and name in SMILES_FALLBACK:
        raw = SMILES_FALLBACK[name]
    if raw is None:
        try:
            import pubchempy as pcp

            comps = pcp.get_compounds(name, "name")
            time.sleep(0.25)  # be polite to PubChem
            if comps:
                raw = comps[0].isomeric_smiles or comps[0].canonical_smiles
        except Exception as exc:  # noqa: BLE001
            print(f"  ! PubChem failed for {name!r}: {exc}", file=sys.stderr)
    if not raw:
        return None
    mol = Chem.MolFromSmiles(raw)
    return Chem.MolToSmiles(mol) if mol is not None else None


def _tox_cols(rec: dict, prefix: str) -> dict:
    """Flatten a (effect, ad) tuple into effect / ad / source columns."""
    effect, ad = rec.get(prefix, (None, None))
    source = None
    if effect is not None and ad is None:
        source = EXP  # paper printed a data-source tag instead of an AD%
    return {f"ref_{prefix}": effect, f"ref_{prefix}_ad": ad, f"ref_{prefix}_source": source}


def build() -> pd.DataFrame:
    rows: list[dict] = []

    def add(rec: dict, cluster: str | None):
        smi = resolve_smiles(rec["name"], rec.get("smiles"))
        if smi is None:
            print(f"  ! could not resolve SMILES for {rec['name']!r}; skipping", file=sys.stderr)
            return
        row = {
            "name": rec["name"],
            "smiles": smi,
            "class": rec["klass"],
            "paper_cluster": cluster or "",
            "ref_ld50_iv_mgkg": rec.get("ld50_iv"),
            "ref_ld50_iv_ad": rec.get("ad_ld50_iv"),
            "ref_ld50_oral_mgkg": rec.get("ld50_oral"),
            "ref_synthesis_cost_usd_g": rec.get("syn_cost"),
        }
        for p in ("hepato", "dili", "cardio", "carcino"):
            row.update(_tox_cols(rec, p))
        rows.append(row)

    print("Cluster E (exact, 22 compounds):")
    for rec in CLUSTER_E:
        add(rec, "E")
    print("Unclustered (3):")
    for rec in UNCLUSTERED:
        add(rec, "none")
    print("Other named furanocoumarins:")
    for rec in NAMED_OTHER:
        add(rec, "")  # let the clustering assign them
    print("Representative cluster A-D members:")
    for nm in CLUSTER_A_TERPENOIDS:
        add(dict(name=nm, klass="terpenoid"), "A")
    for nm in CLUSTER_B_POLYPHENOLS:
        add(dict(name=nm, klass="polyphenol_flavonoid"), "B")
    for nm in CLUSTER_C_FATTY_ACIDS:
        add(dict(name=nm, klass="fatty_acid"), "C")
    for nm in CLUSTER_D_AROMATICS:
        add(dict(name=nm, klass="aromatic"), "D")

    df = pd.DataFrame(rows).drop_duplicates(subset="smiles").reset_index(drop=True)
    return df


def main() -> None:
    df = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"\nWrote {len(df)} compounds -> {OUT}")
    print(df.groupby("class").size().to_string())


if __name__ == "__main__":
    main()
