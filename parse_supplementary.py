#!/usr/bin/env python
"""Parse Tables S1-S5 of the paper's Supplementary (SMILES + SynID per cluster A-E).

The Supplementary PDF tables wrap long SMILES across lines; this reassembles them.
Extract the text first:  pdftotext -layout plants-3875800-supplementary.pdf supp.txt
Then:  uv run python parse_supplementary.py [supp.txt]  ->  server/data/supplementary_smiles.csv
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("supp.txt")
OUT = Path(__file__).resolve().parent / "server" / "data" / "supplementary_smiles.csv"

TABLE_RE = re.compile(r"Table\s+S([1-5])\b")
ROW_RE = re.compile(r"^\s*(\d+)\.\s+(.*\S)\s+(\d{6,})\s*$")
CLUSTER = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}


def main() -> None:
    lines = SRC.read_text(encoding="utf-8", errors="replace").splitlines()
    rows: list[dict] = []
    cluster: str | None = None
    pending: list[str] = []

    for ln in lines:
        if "Table S6" in ln:
            break
        mt = TABLE_RE.search(ln)
        if mt:
            cluster = CLUSTER[mt.group(1)]
            pending = []
            continue
        if cluster is None:
            continue
        s = ln.strip()
        if not s or s.isdigit():                 # blank or page number
            continue
        if s.startswith("№") or ("SMILES" in s and "SynID" in s):  # column header
            continue
        m = ROW_RE.match(ln)
        if m:
            num, tail, synid = m.group(1), m.group(2), m.group(3)
            raw = ("".join(pending) + tail).replace(" ", "")
            mol = Chem.MolFromSmiles(raw)
            rows.append({
                "cluster": cluster, "n": int(num), "synid": synid,
                "smiles_raw": raw,
                "smiles": Chem.MolToSmiles(mol) if mol is not None else "",
                "valid": mol is not None,
            })
            pending = []
        else:
            pending.append(s)                    # SMILES head fragment (wrapped line)

    df = pd.DataFrame(rows)
    print("rows per cluster:")
    print(df.groupby("cluster").size().to_string())
    print(f"total={len(df)}  expected A25 B22 C132 D21 E22 = 222")
    bad = df[~df.valid]
    print(f"invalid SMILES: {len(bad)}")
    for _, r in bad.iterrows():
        print(f"  {r.cluster}{r.n}: {r.smiles_raw}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"wrote {len(df)} -> {OUT}")


if __name__ == "__main__":
    main()
