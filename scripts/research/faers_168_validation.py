#!/usr/bin/env python3
"""
FAERS 168 Validation — Test the Fano Plane CYP450 Predictions

Hypothesis: Drug pairs mediated by CYP enzymes on the same Fano line
show LESS order-dependent adverse events than non-Fano pairs.

Data source: openFDA API (FAERS)
https://open.fda.gov/apis/drug/event/

Usage:
    python3 scripts/faers_168_validation.py
"""

import json
import csv
import time
import sys
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError
from urllib.parse import quote

# ============================================================================
# FANO PLANE STRUCTURE
# ============================================================================

FANO_LINES = [
    (1, 2, 4),  # CYP1A2–CYP2C9–CYP2B6
    (2, 3, 5),  # CYP2C9–CYP2C8–CYP2C19  (gene cluster)
    (3, 4, 6),  # CYP2C8–CYP2B6–CYP2D6
    (4, 5, 7),  # CYP2B6–CYP2C19–CYP3A4
    (5, 6, 1),  # CYP2C19–CYP2D6–CYP1A2
    (6, 7, 2),  # CYP2D6–CYP3A4–CYP2C9   (big three)
    (7, 1, 3),  # CYP3A4–CYP1A2–CYP2C8
]

CYP_NAMES = {
    1: "CYP1A2", 2: "CYP2C9", 3: "CYP2C8", 4: "CYP2B6",
    5: "CYP2C19", 6: "CYP2D6", 7: "CYP3A4",
}

def is_on_fano_line(a, b, c):
    """Check if three CYP basis indices form a Fano line."""
    s = frozenset([a, b, c])
    return any(frozenset(line) == s for line in FANO_LINES)

def classify_triple(a, b, c):
    """0=trivial, 1=associative (Fano), 2=non-associative."""
    if a == b or b == c or a == c:
        return 0
    if is_on_fano_line(a, b, c):
        return 1
    return 2

# ============================================================================
# DRUG-CYP MAPPING
# ============================================================================

def load_drug_mapping(path="data/cyp_drug_mapping.csv"):
    """Load drug → CYP enzyme mapping."""
    drugs = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            drugs[row['generic_name'].lower()] = {
                'cyp': row['primary_cyp'],
                'basis': int(row['cyp_basis']),
                'role': row['role'],
            }
    return drugs

# ============================================================================
# openFDA API QUERIES
# ============================================================================

OPENFDA_BASE = "https://api.fda.gov/drug/event.json"

def query_openfda(drug_a, drug_b, limit=100):
    """Query FAERS for cases with both drugs, return cases with temporal data."""
    # Search for cases mentioning both drugs
    search = f'patient.drug.openfda.generic_name:"{drug_a}"+AND+patient.drug.openfda.generic_name:"{drug_b}"'
    url = f"{OPENFDA_BASE}?search={search}&limit={limit}"

    try:
        req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
            return data.get('results', [])
    except HTTPError as e:
        if e.code == 404:
            return []  # no results
        print(f"  HTTP {e.code} for {drug_a}+{drug_b}", file=sys.stderr)
        return []
    except (URLError, TimeoutError) as e:
        print(f"  Network error for {drug_a}+{drug_b}: {e}", file=sys.stderr)
        return []

def extract_temporal_order(case, drug_a, drug_b):
    """Extract which drug was started first from a FAERS case.
    Returns: 'A_first', 'B_first', 'same_day', or 'unknown'."""
    drugs = case.get('patient', {}).get('drug', [])

    date_a = None
    date_b = None

    for d in drugs:
        name = ""
        openfda = d.get('openfda', {})
        generic_names = openfda.get('generic_name', [])
        if generic_names:
            name = generic_names[0].lower()

        start = d.get('drugstartdate', '')

        if drug_a.lower() in name and start:
            date_a = start
        elif drug_b.lower() in name and start:
            date_b = start

    if date_a and date_b:
        if date_a < date_b:
            return 'A_first'
        elif date_b < date_a:
            return 'B_first'
        else:
            return 'same_day'
    return 'unknown'

# ============================================================================
# VALIDATION LOGIC
# ============================================================================

def analyze_drug_pair(drug_a, drug_b, drug_map):
    """Analyze order-dependence for a specific drug pair."""
    info_a = drug_map.get(drug_a.lower())
    info_b = drug_map.get(drug_b.lower())

    if not info_a or not info_b:
        return None

    cyp_a = info_a['basis']
    cyp_b = info_b['basis']

    if cyp_a == cyp_b:
        return None  # same enzyme, trivial

    # Query FAERS
    cases = query_openfda(drug_a, drug_b, limit=100)

    a_first = 0
    b_first = 0
    same_day = 0
    unknown = 0
    total_reactions_a_first = 0
    total_reactions_b_first = 0

    for case in cases:
        order = extract_temporal_order(case, drug_a, drug_b)
        n_reactions = len(case.get('patient', {}).get('reaction', []))

        if order == 'A_first':
            a_first += 1
            total_reactions_a_first += n_reactions
        elif order == 'B_first':
            b_first += 1
            total_reactions_b_first += n_reactions
        elif order == 'same_day':
            same_day += 1
        else:
            unknown += 1

    temporal_total = a_first + b_first
    if temporal_total == 0:
        asymmetry = 0.0
    else:
        asymmetry = abs(a_first - b_first) / temporal_total

    return {
        'drug_a': drug_a,
        'drug_b': drug_b,
        'cyp_a': CYP_NAMES[cyp_a],
        'cyp_b': CYP_NAMES[cyp_b],
        'basis_a': cyp_a,
        'basis_b': cyp_b,
        'total_cases': len(cases),
        'a_first': a_first,
        'b_first': b_first,
        'same_day': same_day,
        'unknown': unknown,
        'asymmetry': asymmetry,
        'mean_reactions_a_first': total_reactions_a_first / max(a_first, 1),
        'mean_reactions_b_first': total_reactions_b_first / max(b_first, 1),
    }

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("  FAERS 168 VALIDATION")
    print("  Testing Fano Plane CYP450 Predictions Against Real Clinical Data")
    print("=" * 70)
    print()

    drug_map = load_drug_mapping()
    print(f"Loaded {len(drug_map)} drugs with CYP mapping")
    print()

    # Test drug TRIPLES — the Fano test requires 3 CYP enzymes
    # Every pair of CYP enzymes shares a Fano line; only triples discriminate.
    #
    # FANO TRIPLES (predicted: order INDEPENDENT):
    #   Three drugs whose CYP enzymes are all on the same Fano line
    # NON-FANO TRIPLES (predicted: order DEPENDENT):
    #   Three drugs whose CYP enzymes are NOT all on one Fano line

    test_triples = [
        # FANO line (6,7,2): CYP2D6–CYP3A4–CYP2C9
        ("paroxetine", "simvastatin", "warfarin", True),      # 2D6-3A4-2C9 → FANO
        ("fluoxetine", "midazolam", "phenytoin", True),       # 2D6-3A4-2C9 → FANO

        # FANO line (2,3,5): CYP2C9–CYP2C8–CYP2C19
        ("fluconazole", "repaglinide", "omeprazole", True),   # 2C9-2C8-2C19 → FANO
        ("warfarin", "paclitaxel", "clopidogrel", True),      # 2C9-2C8-2C19 → FANO

        # FANO line (5,6,1): CYP2C19–CYP2D6–CYP1A2
        ("omeprazole", "codeine", "theophylline", True),      # 2C19-2D6-1A2 → FANO

        # NON-FANO triples (predicted: order DEPENDENT)
        ("paroxetine", "theophylline", "omeprazole", False),  # 2D6-1A2-2C19 → NOT a line
        ("ketoconazole", "bupropion", "warfarin", False),     # 3A4-2B6-2C9 → NOT a line
        ("fluoxetine", "theophylline", "repaglinide", False),  # 2D6-1A2-2C8 → NOT a line
        ("ciprofloxacin", "simvastatin", "efavirenz", False), # 1A2-3A4-2B6 → NOT a line
        ("paroxetine", "omeprazole", "repaglinide", False),   # 2D6-2C19-2C8 → NOT a line
    ]

    results = []
    print("Querying FAERS for drug triples (this may take a few minutes)...")
    print()

    for entry in test_triples:
        drug_a, drug_b, drug_c, expected_fano = entry
        print(f"  {drug_a} + {drug_b} + {drug_c} ... ", end="", flush=True)

        info_a = drug_map.get(drug_a.lower())
        info_b = drug_map.get(drug_b.lower())
        info_c = drug_map.get(drug_c.lower())

        if not (info_a and info_b and info_c):
            print("skipped (missing drug)")
            continue

        ba, bb, bc = info_a['basis'], info_b['basis'], info_c['basis']
        actual_fano = is_on_fano_line(ba, bb, bc)

        if actual_fano != expected_fano:
            print(f"WARNING: expected_fano={expected_fano} but actual={actual_fano} for ({ba},{bb},{bc})")

        # Query FAERS for the FIRST two drugs (A+B), look at temporal ordering
        # We query A+B+C together to find cases with all three
        search = (f'patient.drug.openfda.generic_name:"{drug_a}"'
                  f'+AND+patient.drug.openfda.generic_name:"{drug_b}"'
                  f'+AND+patient.drug.openfda.generic_name:"{drug_c}"')
        url = f"{OPENFDA_BASE}?search={search}&limit=100"

        cases = []
        try:
            req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
                cases = data.get('results', [])
        except (HTTPError, URLError, TimeoutError):
            pass

        # Extract temporal ordering of the first two drugs
        a_first = b_first = 0
        for case in cases:
            order = extract_temporal_order(case, drug_a, drug_b)
            if order == 'A_first':
                a_first += 1
            elif order == 'B_first':
                b_first += 1

        temporal = a_first + b_first
        asymmetry = abs(a_first - b_first) / temporal if temporal > 0 else -1

        result = {
            'drugs': f"{drug_a}+{drug_b}+{drug_c}",
            'cyps': f"{CYP_NAMES[ba]}-{CYP_NAMES[bb]}-{CYP_NAMES[bc]}",
            'on_fano_line': actual_fano,
            'total_cases': len(cases),
            'a_first': a_first,
            'b_first': b_first,
            'asymmetry': asymmetry,
        }
        results.append(result)

        fano_str = "FANO" if actual_fano else "NON-FANO"
        if asymmetry >= 0:
            print(f"cases={len(cases)}, temporal={temporal}, asym={asymmetry:.3f}, {fano_str}")
        else:
            print(f"cases={len(cases)}, no temporal data, {fano_str}")

        time.sleep(0.5)  # rate limit

    # ---------------------------------------------------------------
    # ANALYSIS
    # ---------------------------------------------------------------
    print()
    print("=" * 70)
    print("  RESULTS")
    print("=" * 70)
    print()

    fano_results = [r for r in results if r['on_fano_line'] and r['asymmetry'] >= 0]
    non_fano_results = [r for r in results if not r['on_fano_line'] and r['asymmetry'] >= 0]

    print(f"Fano line triples (with temporal data): {len(fano_results)}")
    print(f"Non-Fano triples (with temporal data):  {len(non_fano_results)}")
    print()

    for label, group in [("FANO (predicted order-INDEPENDENT)", fano_results),
                          ("NON-FANO (predicted order-DEPENDENT)", non_fano_results)]:
        if group:
            asym_vals = [r['asymmetry'] for r in group]
            mean_asym = sum(asym_vals) / len(asym_vals)
            print(f"  {label}: mean asymmetry = {mean_asym:.3f} (n={len(group)})")
            for r in group:
                print(f"    {r['drugs']:50s} {r['cyps']:30s} asym={r['asymmetry']:.3f} "
                      f"(A→B:{r['a_first']}, B→A:{r['b_first']})")
        print()

    print("=" * 70)
    print("  PREDICTION TEST")
    print("=" * 70)
    print()
    print("  Hypothesis: Non-Fano triples have HIGHER asymmetry than Fano triples")
    print()

    fano_asym = [r['asymmetry'] for r in fano_results]
    non_fano_asym = [r['asymmetry'] for r in non_fano_results]

    if fano_asym and non_fano_asym:
        fano_mean = sum(fano_asym) / len(fano_asym)
        nf_mean = sum(non_fano_asym) / len(non_fano_asym)
        diff = nf_mean - fano_mean

        print(f"  Fano mean asymmetry:     {fano_mean:.3f} (n={len(fano_asym)})")
        print(f"  Non-Fano mean asymmetry: {nf_mean:.3f} (n={len(non_fano_asym)})")
        print(f"  Difference:              {diff:+.3f}")
        print()

        if diff > 0:
            print("  CONSISTENT with prediction: Non-Fano > Fano")
        elif diff < 0:
            print("  OPPOSITE to prediction: Fano > Non-Fano")
        else:
            print("  No difference")
    elif not fano_asym and not non_fano_asym:
        print("  No temporal data available for either group.")
    else:
        print("  Insufficient data in one group for comparison.")

    print()
    print("  NOTE: Signal search with small N. Not definitive.")
    print("  FAERS temporal data is sparse. Null = equally informative.")

if __name__ == "__main__":
    main()
