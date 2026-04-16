#!/usr/bin/env python3
"""Fetch FAERS reports and per-triple co-administered-drug distribution.

For each of the 35 primary drug triples, paginate openFDA up to 1000
reports. In each report, scan the full patient.drug[] list and classify
every drug by its primary CYP using data/cyp_drug_mapping.csv.

Output: data/faers_concomitants.csv with one row per triple, columns:
  cyp_a, cyp_b, cyp_c, drugs, n_reports,
  conc_cyp1..conc_cyp7 = count of reports containing ANY drug mapped to
                         CYP_i OTHER than the 3 primary drugs in the
                         cocktail

The test (done in Sounio):
  For each non-associative triple with associator direction m ∈ {1..7},
  concomitant_rate[CYP_m] vs average concomitant_rate of the other 3
  out-of-triple CYPs. Algebra-mediator hypothesis predicts rate[m]
  > mean(rate[other out-CYPs]).
"""
import urllib.request
import urllib.error
import json
import itertools
import time
import csv
import os
import re

DRUGS = {
    1: ("CYP1A2",  "theophylline"),
    2: ("CYP2C9",  "warfarin"),
    3: ("CYP2C8",  "repaglinide"),
    4: ("CYP2B6",  "bupropion"),
    5: ("CYP2C19", "omeprazole"),
    6: ("CYP2D6",  "codeine"),
    7: ("CYP3A4",  "simvastatin"),
}

CYP_MAP = {}   # drug_name (lower) -> cyp_basis_int

def load_drug_map():
    with open("data/cyp_drug_mapping.csv") as f:
        for r in csv.DictReader(f):
            drug = r["drug"].strip().lower()
            basis = int(r["cyp_basis"])
            CYP_MAP[drug] = basis

API = "https://api.fda.gov/drug/event.json"
PAGE = 100
CAP = 1000


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


def classify_concomitants(record, primary_drugs_lower):
    """Return set of CYP basis indices (1..7) represented by ANY drug in the
    record's patient.drug[] list EXCLUDING the 3 primary drugs. One CYP at
    most once per report (we're measuring presence, not count)."""
    drugs = record.get("patient", {}).get("drug", [])
    seen = set()
    for d in drugs:
        name = (d.get("medicinalproduct") or "").strip().lower()
        if not name:
            continue
        # Skip if this drug IS one of the primary 3 (substring match)
        if any(p in name for p in primary_drugs_lower):
            continue
        # Look for any known drug in the name (substring)
        for known, basis in CYP_MAP.items():
            if len(known) >= 6 and known in name:
                seen.add(basis)
                break
    return seen


def main():
    load_drug_map()
    print(f"drug map has {len(CYP_MAP)} entries covering CYPs {sorted(set(CYP_MAP.values()))}")
    triples = list(itertools.combinations(range(1, 8), 3))
    rows = []
    t0 = time.time()
    for idx, (i, j, k) in enumerate(triples):
        cyp_a, drug_a = DRUGS[i]
        cyp_b, drug_b = DRUGS[j]
        cyp_c, drug_c = DRUGS[k]
        primary = [drug_a, drug_b, drug_c]
        n_reports = 0
        conc = [0] * 8   # 1-indexed; index 0 unused
        skip = 0
        while skip < CAP:
            records = fetch_page(drug_a, drug_b, drug_c, skip)
            if not records:
                break
            for rec in records:
                n_reports += 1
                seen = classify_concomitants(rec, primary)
                for cyp_idx in seen:
                    conc[cyp_idx] += 1
            skip += PAGE
            if len(records) < PAGE:
                break
            time.sleep(0.25)
        rows.append({
            "cyp_a": cyp_a, "cyp_b": cyp_b, "cyp_c": cyp_c,
            "drugs": f"{drug_a}+{drug_b}+{drug_c}",
            "n_reports": n_reports,
            "conc_cyp1": conc[1], "conc_cyp2": conc[2],
            "conc_cyp3": conc[3], "conc_cyp4": conc[4],
            "conc_cyp5": conc[5], "conc_cyp6": conc[6],
            "conc_cyp7": conc[7],
        })
        elapsed = time.time() - t0
        print(f"[{idx+1:2d}/35] ({i},{j},{k}) n={n_reports:4d}  "
              f"conc=[{conc[1]:3d} {conc[2]:3d} {conc[3]:3d} {conc[4]:3d} "
              f"{conc[5]:3d} {conc[6]:3d} {conc[7]:3d}]  [{elapsed:.0f}s]")

    out = "data/faers_concomitants.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
