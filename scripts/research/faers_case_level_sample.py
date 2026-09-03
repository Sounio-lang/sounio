#!/usr/bin/env python3
"""Sample case-level openFDA rows for one Fano + one non-Fano CYP triple.

True case-level FAERS is not committed (size + API rate limits). This script
pulls a *bounded* sample when network is available and writes a local artifact
under results/deep_four/ for residual analysis.

Usage:
  python3 scripts/research/faers_case_level_sample.py
"""
from __future__ import annotations

import csv
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from lib.cyp_fano_canon import is_fano_names  # noqa: E402

OUT = ROOT / "results" / "deep_four" / "faers_case_level_sample.json"
API = "https://api.fda.gov/drug/event.json"
# One representative drug per CYP (same as faers_168)
DRUGS = {
    "CYP1A2": "theophylline",
    "CYP2C9": "warfarin",
    "CYP2C8": "repaglinide",
    "CYP2B6": "bupropion",
    "CYP2C19": "omeprazole",
    "CYP2D6": "codeine",
    "CYP3A4": "simvastatin",
}


def fetch_cases(d1: str, d2: str, d3: str, limit: int = 50) -> list[dict]:
    q = (
        f'search=patient.drug.openfda.generic_name:"{d1}"'
        f'+AND+patient.drug.openfda.generic_name:"{d2}"'
        f'+AND+patient.drug.openfda.generic_name:"{d3}"'
        f"&limit={limit}"
    )
    url = f"{API}?{q}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Sounio-FAERS-case-sample/1.0"})
        with urllib.request.urlopen(req, timeout=45) as resp:
            data = json.loads(resp.read().decode())
        return data.get("results", [])
    except Exception as e:
        return [{"_error": str(e)}]


def order_a_first(case: dict, da: str, db: str) -> str:
    drugs = case.get("patient", {}).get("drug", [])
    date_a = date_b = None
    for d in drugs:
        names = d.get("openfda", {}).get("generic_name", []) or []
        name = (names[0] if names else "").lower()
        start = d.get("drugstartdate") or ""
        if da in name and start:
            date_a = start
        if db in name and start:
            date_b = start
    if date_a and date_b:
        if date_a < date_b:
            return "a_first"
        if date_b < date_a:
            return "b_first"
        return "same"
    return "unknown"


def main() -> int:
    # Fano line (2,3,5): CYP2C9+CYP2C8+CYP2C19 — warfarin+repaglinide+omeprazole
    # Non-Fano e.g. CYP1A2+CYP2C9+CYP2C19 — theophylline+warfarin+omeprazole
    triples = [
        ("CYP2C9", "CYP2C8", "CYP2C19"),
        ("CYP1A2", "CYP2C9", "CYP2C19"),
    ]
    out_rows = []
    for a, b, c in triples:
        da, db, dc = DRUGS[a], DRUGS[b], DRUGS[c]
        fano = is_fano_names(a, b, c)
        cases = fetch_cases(da, db, dc, limit=40)
        time.sleep(0.4)
        n_a = n_b = n_t = 0
        ages = []
        for case in cases:
            if "_error" in case:
                out_rows.append({"triple": f"{a}+{b}+{c}", "fano": fano, "error": case["_error"]})
                break
            ord_ = order_a_first(case, da, db)
            if ord_ == "a_first":
                n_a += 1
                n_t += 1
            elif ord_ == "b_first":
                n_b += 1
                n_t += 1
            age = case.get("patient", {}).get("patientonsetage")
            if age is not None:
                try:
                    ages.append(float(age))
                except (TypeError, ValueError):
                    pass
        asym = abs(n_a - n_b) / n_t if n_t > 0 else None
        out_rows.append(
            {
                "cyp": [a, b, c],
                "drugs": [da, db, dc],
                "fano": fano,
                "n_cases_returned": len([c for c in cases if "_error" not in c]),
                "a_first": n_a,
                "b_first": n_b,
                "temporal": n_t,
                "asymmetry": asym,
                "mean_age": (sum(ages) / len(ages)) if ages else None,
                "n_with_age": len(ages),
            }
        )
        print(json.dumps(out_rows[-1]))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"schema": "faers_case_level_sample.v1", "rows": out_rows}, indent=2) + "\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
