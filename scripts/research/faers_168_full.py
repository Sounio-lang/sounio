#!/usr/bin/env python3
"""
FAERS 168 Full Validation — All 35 CYP450 Triples

Exhaustively tests all C(7,3) = 35 unordered CYP triples against FAERS.
7 are Fano line triples (predicted order-independent).
28 are non-Fano triples (predicted order-dependent).

For each triple, picks the most common drug per CYP from the mapping table,
queries FAERS, and measures temporal asymmetry.
"""

import json
import csv
import time
import sys
from itertools import combinations
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError

# ============================================================================
# CYP MAPPING AND FANO STRUCTURE
# ============================================================================

CYP_NAMES = {
    1: "CYP1A2", 2: "CYP2C9", 3: "CYP2C8", 4: "CYP2B6",
    5: "CYP2C19", 6: "CYP2D6", 7: "CYP3A4",
}

FANO_LINES = [
    frozenset([1, 2, 4]), frozenset([2, 3, 5]), frozenset([3, 4, 6]),
    frozenset([4, 5, 7]), frozenset([5, 6, 1]), frozenset([6, 7, 2]),
    frozenset([7, 1, 3]),
]

# Best representative drug per CYP (high FAERS frequency, clear single-CYP)
CYP_DRUGS = {
    1: ["theophylline", "caffeine", "tizanidine"],
    2: ["warfarin", "phenytoin"],
    3: ["repaglinide", "paclitaxel"],
    4: ["bupropion", "efavirenz"],
    5: ["omeprazole", "clopidogrel", "lansoprazole"],
    6: ["codeine", "paroxetine", "metoprolol", "tamoxifen"],
    7: ["simvastatin", "midazolam", "cyclosporine"],
}

OPENFDA_BASE = "https://api.fda.gov/drug/event.json"

def query_triple(drug_a, drug_b, drug_c, limit=100):
    """Query FAERS for cases with all three drugs."""
    search = (f'patient.drug.openfda.generic_name:"{drug_a}"'
              f'+AND+patient.drug.openfda.generic_name:"{drug_b}"'
              f'+AND+patient.drug.openfda.generic_name:"{drug_c}"')
    url = f"{OPENFDA_BASE}?search={search}&limit={limit}"
    try:
        req = Request(url, headers={'User-Agent': 'Sounio-168-Validation/1.0'})
        with urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
            return data.get('results', [])
    except (HTTPError, URLError, TimeoutError):
        return []

def extract_order(case, drug_a, drug_b):
    """Which of two drugs was started first?"""
    drugs = case.get('patient', {}).get('drug', [])
    date_a = date_b = None
    for d in drugs:
        names = d.get('openfda', {}).get('generic_name', [])
        name = names[0].lower() if names else ""
        start = d.get('drugstartdate', '')
        if drug_a.lower() in name and start:
            date_a = start
        elif drug_b.lower() in name and start:
            date_b = start
    if date_a and date_b:
        if date_a < date_b: return 'A'
        elif date_b < date_a: return 'B'
        else: return 'same'
    return 'unknown'

def main():
    print("=" * 70)
    print("  FAERS 168 FULL VALIDATION — All 35 CYP Triples")
    print("=" * 70)
    print()

    all_triples = list(combinations(range(1, 8), 3))
    print(f"Total unordered CYP triples: {len(all_triples)}")
    fano_triples = [t for t in all_triples if frozenset(t) in FANO_LINES]
    non_fano_triples = [t for t in all_triples if frozenset(t) not in FANO_LINES]
    print(f"  Fano: {len(fano_triples)}, Non-Fano: {len(non_fano_triples)}")
    print()

    results = []

    for idx, (a, b, c) in enumerate(all_triples):
        is_fano = frozenset([a, b, c]) in FANO_LINES
        fano_str = "FANO" if is_fano else "non-F"

        # Try first drug from each CYP's list
        drug_a = CYP_DRUGS[a][0]
        drug_b = CYP_DRUGS[b][0]
        drug_c = CYP_DRUGS[c][0]

        print(f"  [{idx+1:2d}/35] ({CYP_NAMES[a]},{CYP_NAMES[b]},{CYP_NAMES[c]}) "
              f"{fano_str} | {drug_a}+{drug_b}+{drug_c} ... ", end="", flush=True)

        cases = query_triple(drug_a, drug_b, drug_c)

        # Measure temporal asymmetry for the first two drugs
        a_first = b_first = 0
        for case in cases:
            order = extract_order(case, drug_a, drug_b)
            if order == 'A': a_first += 1
            elif order == 'B': b_first += 1

        temporal = a_first + b_first
        asymmetry = abs(a_first - b_first) / temporal if temporal > 0 else -1

        result = {
            'cyp_a': a, 'cyp_b': b, 'cyp_c': c,
            'fano': is_fano,
            'drugs': f"{drug_a}+{drug_b}+{drug_c}",
            'cases': len(cases),
            'a_first': a_first, 'b_first': b_first,
            'temporal': temporal,
            'asymmetry': asymmetry,
        }
        results.append(result)

        if temporal > 0:
            print(f"cases={len(cases)}, temp={temporal}, asym={asymmetry:.3f}")
        else:
            print(f"cases={len(cases)}, no temporal")

        time.sleep(0.3)

    # ---------------------------------------------------------------
    # ANALYSIS
    # ---------------------------------------------------------------
    print()
    print("=" * 70)
    print("  RESULTS")
    print("=" * 70)

    fano_asym = [r['asymmetry'] for r in results if r['fano'] and r['asymmetry'] >= 0]
    nf_asym = [r['asymmetry'] for r in results if not r['fano'] and r['asymmetry'] >= 0]

    print(f"\nFano triples with temporal data: {len(fano_asym)}/{len(fano_triples)}")
    print(f"Non-Fano with temporal data:     {len(nf_asym)}/{len(non_fano_triples)}")

    if fano_asym:
        print(f"\n  Fano mean asymmetry:     {sum(fano_asym)/len(fano_asym):.3f} "
              f"(n={len(fano_asym)}, values={[round(x,3) for x in fano_asym]})")
    if nf_asym:
        print(f"  Non-Fano mean asymmetry: {sum(nf_asym)/len(nf_asym):.3f} "
              f"(n={len(nf_asym)}, values={[round(x,3) for x in nf_asym]})")

    if fano_asym and nf_asym:
        fm = sum(fano_asym) / len(fano_asym)
        nfm = sum(nf_asym) / len(nf_asym)
        diff = nfm - fm
        print(f"\n  Difference (non-Fano - Fano): {diff:+.3f}")
        print(f"  Direction: {'CONSISTENT' if diff > 0 else 'OPPOSITE'} with prediction")

        # Simple permutation test
        import random
        random.seed(42)
        all_vals = fano_asym + nf_asym
        n_fano = len(fano_asym)
        observed_diff = diff
        n_perm = 10000
        count_extreme = 0
        for _ in range(n_perm):
            random.shuffle(all_vals)
            perm_fano = all_vals[:n_fano]
            perm_nf = all_vals[n_fano:]
            perm_diff = sum(perm_nf)/len(perm_nf) - sum(perm_fano)/len(perm_fano)
            if perm_diff >= observed_diff:
                count_extreme += 1
        p_val = count_extreme / n_perm
        print(f"  Permutation p-value (one-tailed): {p_val:.4f}")
        print(f"  (p < 0.05 would be significant)")

    # Save CSV
    with open('data/faers_168_results.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['cyp_a','cyp_b','cyp_c','fano','drugs','cases','a_first','b_first','temporal','asymmetry'])
        for r in results:
            w.writerow([CYP_NAMES[r['cyp_a']], CYP_NAMES[r['cyp_b']], CYP_NAMES[r['cyp_c']],
                        r['fano'], r['drugs'], r['cases'],
                        r['a_first'], r['b_first'], r['temporal'], round(r['asymmetry'], 4)])
    print("\nResults saved to data/faers_168_results.csv")

if __name__ == "__main__":
    main()
