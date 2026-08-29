#!/usr/bin/env python3
"""Expand FAERS coverage by multiplying drug-representatives per CYP.

The prior fetchers used ONE drug per CYP (7 representatives, 35 CYP-triples).
This script uses the full data/cyp_drug_mapping.csv (~38 drugs across 7 CYPs)
to enumerate drug-triples within each CYP-triple, query openFDA, and
aggregate at the CYP-triple level.

Per CYP-triple: sum(cases), sum(a_first), sum(b_first), sum(temporal)
across all its drug-triples. Final |asymmetry| computed from the
aggregated totals.

This pushes temporal sample size per CYP-triple from ~tens to ~hundreds
or thousands (depending on triple), enabling a sharper permutation test.

Output: data/faers_drugbank.csv with columns:
  cyp_a, cyp_b, cyp_c, n_drug_triples, cases, a_first, b_first,
  temporal, asymmetry, fano

Usage:
    python3 scripts/fetch_faers_drugbank.py
"""
import urllib.request
import urllib.error
import json
import itertools
import time
import csv
import os

import sys
from pathlib import Path

# Canonical Fano = stdlib/medical/cyp450_fano.sio (NOT alternate CD tables).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.cyp_fano_canon import CYP_NAMES, FANO_TRIPLES_SORTED, fano_flag  # noqa: E402

# Ordered (i<j<k) membership — combinations() yields sorted triples.
FANO_TRIPLES = set(FANO_TRIPLES_SORTED)

API = "https://api.fda.gov/drug/event.json"
PAGE = 100
MAX_PER_DRUG_TRIPLE = 500


def load_cyp_drugs():
    buckets = {i: [] for i in range(1, 8)}
    with open("data/cyp_drug_mapping.csv") as f:
        for r in csv.DictReader(f):
            basis = int(r["cyp_basis"])
            if 1 <= basis <= 7:
                buckets[basis].append(r["drug"].strip().lower())
    return buckets


def fetch_total(d1, d2, d3):
    q = (f"search=patient.drug.medicinalproduct:{d1}"
         f"+AND+patient.drug.medicinalproduct:{d2}"
         f"+AND+patient.drug.medicinalproduct:{d3}&limit=1")
    url = f"{API}?{q}"
    try:
        r = urllib.request.urlopen(url, timeout=15).read()
        return json.loads(r).get("meta", {}).get("results", {}).get("total", 0)
    except urllib.error.HTTPError as e:
        return 0 if e.code == 404 else -1
    except Exception:
        return -1


def fetch_page(d1, d2, d3, skip):
    q = (f"search=patient.drug.medicinalproduct:{d1}"
         f"+AND+patient.drug.medicinalproduct:{d2}"
         f"+AND+patient.drug.medicinalproduct:{d3}"
         f"&limit={PAGE}&skip={skip}")
    url = f"{API}?{q}"
    try:
        r = urllib.request.urlopen(url, timeout=20).read()
        return json.loads(r).get("results", [])
    except Exception:
        return []


def find_drug_date(record, drug_name):
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
        if len(ds) == 4:
            ds = ds + "0101"
        elif len(ds) == 6:
            ds = ds + "01"
        if best is None or ds < best:
            best = ds
    return best


def analyze_drug_triple(drug_a, drug_b, drug_c, total):
    n_fetch = min(total, MAX_PER_DRUG_TRIPLE)
    if n_fetch == 0:
        return 0, 0, 0, 0
    cases = 0
    a_first = 0
    b_first = 0
    temporal = 0
    skip = 0
    while skip < n_fetch:
        records = fetch_page(drug_a, drug_b, drug_c, skip)
        if not records:
            break
        for rec in records:
            cases += 1
            da = find_drug_date(rec, drug_a)
            db = find_drug_date(rec, drug_b)
            if da and db:
                temporal += 1
                if da < db:
                    a_first += 1
                elif db < da:
                    b_first += 1
        skip += PAGE
        time.sleep(0.18)
        if len(records) < PAGE:
            break
    return cases, a_first, b_first, temporal


def main():
    buckets = load_cyp_drugs()
    total_drugs = sum(len(v) for v in buckets.values())
    print(f"CYP buckets: {[(k, len(v)) for k, v in buckets.items()]}")
    print(f"total drugs: {total_drugs}")
    triples = list(itertools.combinations(range(1, 8), 3))
    total_drug_triples = sum(len(buckets[i]) * len(buckets[j]) * len(buckets[k])
                             for (i, j, k) in triples)
    print(f"total drug-triples to scan: {total_drug_triples}")
    print()

    rows = []
    t0 = time.time()
    for tidx, (i, j, k) in enumerate(triples):
        drugs_i = buckets[i]
        drugs_j = buckets[j]
        drugs_k = buckets[k]
        n_drug_triples = 0
        cases = 0
        a_first = 0
        b_first = 0
        temporal = 0
        n_positive_triples = 0
        for d_i in drugs_i:
            for d_j in drugs_j:
                for d_k in drugs_k:
                    total = fetch_total(d_i, d_j, d_k)
                    n_drug_triples += 1
                    if total > 0:
                        n_positive_triples += 1
                        c, a, b, t = analyze_drug_triple(d_i, d_j, d_k, total)
                        cases += c
                        a_first += a
                        b_first += b
                        temporal += t
                    time.sleep(0.18)
        asym = (
            abs(a_first - b_first) / temporal if temporal > 0 else -1.0
        )
        fano = fano_flag(i, j, k)
        row = {
            "cyp_a": CYP_NAMES[i], "cyp_b": CYP_NAMES[j], "cyp_c": CYP_NAMES[k],
            "fano": fano,
            "n_drug_triples": n_drug_triples,
            "n_positive_triples": n_positive_triples,
            "cases": cases,
            "a_first": a_first,
            "b_first": b_first,
            "temporal": temporal,
            "asymmetry": round(asym, 4) if asym >= 0 else -1,
        }
        rows.append(row)
        elapsed = time.time() - t0
        print(f"[{tidx+1:2d}/35] ({i},{j},{k}) drug-triples={n_drug_triples:4d} "
              f"(pos={n_positive_triples:3d}) cases={cases:5d} "
              f"temp={temporal:4d} a={a_first:4d} b={b_first:4d} "
              f"asym={row['asymmetry']:.4f}  [{elapsed:.0f}s]")

    out = "data/faers_drugbank.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
