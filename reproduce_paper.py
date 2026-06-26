#!/usr/bin/env python
"""Reproduce the headline results & conclusions of Rassabina & Fedorov 2025 and print them.

    python reproduce_paper.py    # trains the open models on first run, then they are cached
"""
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

from server import claims                                   # noqa: E402
from server.dataset import load_dataset                     # noqa: E402
from server.heracleum_server import reproduce_all           # noqa: E402


def main() -> None:
    fn = getattr(reproduce_all, "fn", reproduce_all)        # unwrap the FastMCP tool
    res = fn()["answer"]
    print("HEADLINE NUMBERS vs the paper")
    print("=" * 60)
    print(res["summary"])
    for c in res["checks"]:
        print(f"  [{'PASS' if c['match'] else 'FAIL'}] {c['metric']}: {c['reproduced']} vs paper {c['paper']}")

    print("\nCONCLUSIONS (reproduce_claims)")
    print("=" * 60)
    cl = claims.reproduce_claims(load_dataset())
    ok = sum(c["reproduced"] for c in cl)
    print(f"reproduced {ok}/{len(cl)} claims")
    for c in cl:
        print(f"  [{c['id']}] {'OK' if c['reproduced'] else 'NO'}  {c['reproduced_statement'][:95]}")


if __name__ == "__main__":
    main()
