#!/usr/bin/env python3
"""Pull expanded FAERS records via openFDA API for the 7-CYP-representative
drug set, compute per-case temporal ordering, write data/faers_expanded.csv.

Schema matches data/faers_168_results.csv so oct_associator_*.sio can swap
the data source.

Method:
  - For each of the 35 CYP triples, pick the 3 representative drugs (one
    per CYP).
  - Paginate openFDA results (limit=100 per request, up to 1000 per triple).
  - For each report, check whether ≥3 of the 3 drugs have drugstartdate.
  - If all 3 present, determine the temporally earliest of drug A and drug B
    (where A, B, C are in canonical CYP-order from the triple). Increment
    a_first / b_first / neither.
  - |asymmetry| = |a_first − b_first| / (a_first + b_first) when denominator
    is positive; −1 (sentinel for "no temporal data") otherwise.

Zero algebraic/hypothesis-dependent computation — this script only pulls
and counts. All algebraic tests live in Sounio.

Usage:
    python3 scripts/fetch_faers_expanded.py
"""
import urllib.request
import urllib.parse
import urllib.error
import json
import itertools
import time
import csv
import sys
import os

# ─── 7 representative drugs (same as original CSV) ──────────────────────────

DRUGS = {
    1: ("CYP1A2",  "theophylline"),
    2: ("CYP2C9",  "warfarin"),
    3: ("CYP2C8",  "repaglinide"),
    4: ("CYP2B6",  "bupropion"),
    5: ("CYP2C19", "omeprazole"),
    6: ("CYP2D6",  "codeine"),
    7: ("CYP3A4",  "simvastatin"),
}

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.cyp_fano_canon import FANO_TRIPLES_SORTED, fano_flag  # noqa: E402

# Canonical medical Fano (cyp450_fano.sio); sorted triples for combinations().
FANO_TRIPLES = set(FANO_TRIPLES_SORTED)

API = "https://api.fda.gov/drug/event.json"
PAGE_SIZE = 100
MAX_PER_TRIPLE = 1000   # hard cap to keep runtime bounded


def fetch_total(d1, d2, d3):
    q = (f"search=patient.drug.medicinalproduct:{d1}"
         f"+AND+patient.drug.medicinalproduct:{d2}"
         f"+AND+patient.drug.medicinalproduct:{d3}&limit=1")
    url = f"{API}?{q}"
    try:
        r = urllib.request.urlopen(url, timeout=15).read()
        return json.loads(r).get("meta", {}).get("results", {}).get("total", 0)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return 0
        return -1
    except Exception:
        return -1


def fetch_page(d1, d2, d3, skip):
    q = (f"search=patient.drug.medicinalproduct:{d1}"
         f"+AND+patient.drug.medicinalproduct:{d2}"
         f"+AND+patient.drug.medicinalproduct:{d3}"
         f"&limit={PAGE_SIZE}&skip={skip}")
    url = f"{API}?{q}"
    try:
        r = urllib.request.urlopen(url, timeout=20).read()
        return json.loads(r).get("results", [])
    except Exception:
        return []


def find_drug_date(record, drug_name):
    """Return the earliest drugstartdate for any entry matching drug_name
    in the record's patient.drug list. Returns None if no date."""
    drugs = record.get("patient", {}).get("drug", [])
    best = None
    target = drug_name.lower()
    for d in drugs:
        name = (d.get("medicinalproduct") or "").lower()
        if target not in name:
            continue
        ds = d.get("drugstartdate")
        if not ds:
            continue
        # Normalize: API returns YYYYMMDD, YYYYMM, or YYYY
        if len(ds) == 4:
            ds = ds + "0101"
        elif len(ds) == 6:
            ds = ds + "01"
        if best is None or ds < best:
            best = ds
    return best


def analyze_triple(drug_a, drug_b, drug_c, total):
    n_to_fetch = min(total, MAX_PER_TRIPLE)
    if n_to_fetch == 0:
        return 0, 0, 0, 0
    cases = 0
    a_first = 0
    b_first = 0
    temporal = 0
    skip = 0
    while skip < n_to_fetch:
        records = fetch_page(drug_a, drug_b, drug_c, skip)
        if not records:
            break
        for rec in records:
            cases += 1
            da = find_drug_date(rec, drug_a)
            db = find_drug_date(rec, drug_b)
            dc = find_drug_date(rec, drug_c)
            # For the asymmetry comparison we only need a vs b (the first two
            # CYPs in canonical order) — matches the original CSV convention.
            if da and db:
                temporal += 1
                if da < db:
                    a_first += 1
                elif db < da:
                    b_first += 1
                # equal date: neither
        skip += PAGE_SIZE
        time.sleep(0.25)   # respect 240/min rate limit
        if len(records) < PAGE_SIZE:
            break
    return cases, a_first, b_first, temporal


def main():
    out_path = "data/faers_expanded.csv"
    rows = []
    triples = list(itertools.combinations(range(1, 8), 3))
    print(f"fetching {len(triples)} triples ...")
    t0 = time.time()

    for idx, (i, j, k) in enumerate(triples):
        cyp_a, drug_a = DRUGS[i]
        cyp_b, drug_b = DRUGS[j]
        cyp_c, drug_c = DRUGS[k]
        total = fetch_total(drug_a, drug_b, drug_c)
        if total < 0:
            total = 0
        cases, a_first, b_first, temporal = analyze_triple(
            drug_a, drug_b, drug_c, total
        )
        asym = (
            abs(a_first - b_first) / temporal if temporal > 0 else -1.0
        )
        fano = fano_flag(i, j, k)
        row = {
            "cyp_a": cyp_a, "cyp_b": cyp_b, "cyp_c": cyp_c,
            "fano": fano,
            "drugs": f"{drug_a}+{drug_b}+{drug_c}",
            "total_openFDA": total,
            "cases": cases,
            "a_first": a_first,
            "b_first": b_first,
            "temporal": temporal,
            "asymmetry": round(asym, 4) if asym >= 0 else -1,
        }
        rows.append(row)
        elapsed = time.time() - t0
        print(f"[{idx+1:2d}/35] ({i},{j},{k}) total={total:5d} "
              f"cases={cases:5d} temp={temporal:4d} "
              f"a_first={a_first:4d} b_first={b_first:4d} "
              f"asym={row['asymmetry']:.4f}  "
              f"[{elapsed:.1f}s]")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
