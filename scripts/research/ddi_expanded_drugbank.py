#!/usr/bin/env python3
"""
Direction A: DDI expanded analysis with DrugBank multi-drug FAERS data.

Replicates the oct_associator_ddi.sio hypothesis tests on the expanded
faers_drugbank.csv dataset, which has orders-of-magnitude more reports
per triple (e.g. 3447 vs 47 for CYP1A2+CYP2C9+CYP2C19).

Key addition: 10000-shuffle permutation test for the Fano/non-Fano
asymmetry difference, resolving the marginal p=0.070 from the original.
"""

import csv
import math
import random
import json
from pathlib import Path

from ddi_campaign_lib import FANO_TRIPLES

def load_data(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            triple = (r["cyp_a"], r["cyp_b"], r["cyp_c"])
            is_fano = triple in FANO_TRIPLES
            temporal = int(r["temporal"])
            cases = int(r["cases"])
            asym = float(r["asymmetry"]) if temporal > 0 else None
            rows.append({
                "triple": triple,
                "fano": is_fano,
                "fano_csv": r["fano"] == "True",
                "cases": cases,
                "temporal": temporal,
                "asymmetry": asym,
                "a_first": int(r["a_first"]),
                "b_first": int(r["b_first"]),
            })
    return rows

def cohens_d(group_a, group_b):
    if len(group_a) < 2 or len(group_b) < 2:
        return 0.0
    mean_a = sum(group_a) / len(group_a)
    mean_b = sum(group_b) / len(group_b)
    var_a = sum((x - mean_a) ** 2 for x in group_a) / len(group_a)
    var_b = sum((x - mean_b) ** 2 for x in group_b) / len(group_b)
    pooled = math.sqrt((len(group_a) * var_a + len(group_b) * var_b) /
                       (len(group_a) + len(group_b)))
    if pooled < 1e-10:
        return 0.0
    return (mean_b - mean_a) / pooled

def eta_squared(fano_vals, nonfano_vals):
    all_vals = fano_vals + nonfano_vals
    if len(all_vals) < 2:
        return 0.0
    grand_mean = sum(all_vals) / len(all_vals)
    ss_total = sum((x - grand_mean) ** 2 for x in all_vals)
    if ss_total < 1e-15:
        return 0.0
    mean_f = sum(fano_vals) / len(fano_vals) if fano_vals else 0
    mean_nf = sum(nonfano_vals) / len(nonfano_vals) if nonfano_vals else 0
    ss_between = (len(fano_vals) * (mean_f - grand_mean) ** 2 +
                  len(nonfano_vals) * (mean_nf - grand_mean) ** 2)
    return ss_between / ss_total

def permutation_test(fano_vals, nonfano_vals, n_perms=10000, seed=42):
    observed = abs(
        (sum(nonfano_vals) / len(nonfano_vals)) -
        (sum(fano_vals) / len(fano_vals))
    )
    all_vals = fano_vals + nonfano_vals
    n_f = len(fano_vals)
    rng = random.Random(seed)
    count_ge = 0
    for _ in range(n_perms):
        rng.shuffle(all_vals)
        perm_f = all_vals[:n_f]
        perm_nf = all_vals[n_f:]
        perm_diff = abs(
            (sum(perm_nf) / len(perm_nf)) -
            (sum(perm_f) / len(perm_f))
        )
        if perm_diff >= observed:
            count_ge += 1
    return count_ge / n_perms, observed

def weighted_asymmetry(row):
    """Compute asymmetry weighted by temporal sample size."""
    if row["temporal"] <= 0:
        return None
    return abs(row["a_first"] - row["b_first"]) / row["temporal"]

def main():
    repo = Path(__file__).resolve().parent.parent.parent
    
    for dataset_name, csv_path in [
        ("original (faers_expanded.csv)", repo / "data" / "faers_expanded.csv"),
        ("DrugBank expanded (faers_drugbank.csv)", repo / "data" / "faers_drugbank.csv"),
    ]:
        print(f"\n{'='*70}")
        print(f"  Dataset: {dataset_name}")
        print(f"{'='*70}")
        
        rows = load_data(csv_path)
        
        fano_disagree = sum(1 for r in rows if r["fano"] != r["fano_csv"])
        print(f"  Fano labeling disagreement: {fano_disagree}/35")
        
        valid = [r for r in rows if r["asymmetry"] is not None]
        fano_asym = [r["asymmetry"] for r in valid if r["fano"]]
        nonfano_asym = [r["asymmetry"] for r in valid if not r["fano"]]
        
        print(f"\n  --- H1: Asymmetry by Fano group (temporal > 0) ---")
        print(f"  Fano    (n={len(fano_asym):2d}): mean|asym| = {sum(fano_asym)/len(fano_asym):.4f}" if fano_asym else "  Fano: no data")
        print(f"  NonFano (n={len(nonfano_asym):2d}): mean|asym| = {sum(nonfano_asym)/len(nonfano_asym):.4f}" if nonfano_asym else "  NonFano: no data")
        
        d = cohens_d(fano_asym, nonfano_asym)
        eta2 = eta_squared(fano_asym, nonfano_asym)
        print(f"  Cohen's d = {d:.4f}")
        print(f"  eta^2     = {eta2:.4f}")
        
        if fano_asym and nonfano_asym:
            direction = "non-Fano MORE asymmetric" if d > 0 else "non-Fano LESS asymmetric"
            print(f"  Direction: {direction}")
        
        print(f"\n  --- Permutation test (10000 shuffles) ---")
        if len(fano_asym) >= 2 and len(nonfano_asym) >= 2:
            p_val, obs_diff = permutation_test(fano_asym, nonfano_asym, n_perms=10000)
            print(f"  Observed |mean diff| = {obs_diff:.4f}")
            print(f"  p-value = {p_val:.4f}")
            if p_val < 0.05:
                print(f"  ** SIGNIFICANT at alpha=0.05 **")
            elif p_val < 0.10:
                print(f"  Marginal (0.05 < p < 0.10)")
            else:
                print(f"  Not significant (p >= 0.10)")
        else:
            p_val = None
            print(f"  Insufficient data for permutation test")
        
        fano_cases = [r["cases"] for r in rows if r["fano"]]
        nonfano_cases = [r["cases"] for r in rows if not r["fano"]]
        print(f"\n  --- H2: Total adverse events ---")
        print(f"  Fano    ({len(fano_cases)} triples): total={sum(fano_cases)}, mean={sum(fano_cases)/len(fano_cases):.1f}")
        print(f"  NonFano ({len(nonfano_cases)} triples): total={sum(nonfano_cases)}, mean={sum(nonfano_cases)/len(nonfano_cases):.1f}")
        
        total_temporal = sum(r["temporal"] for r in rows)
        total_cases = sum(r["cases"] for r in rows)
        print(f"\n  --- Dataset summary ---")
        print(f"  Total triples: {len(rows)}")
        print(f"  With temporal data: {len(valid)}")
        print(f"  Total cases: {total_cases}")
        print(f"  Total temporal-ordered: {total_temporal}")
    
    # Save results JSON
    out_path = repo / "experiments" / "non_assoc_connectomics" / "ddi_drugbank_expanded_results.json"
    results = {
        "dataset": "faers_drugbank.csv",
        "n_triples": 35,
        "n_with_temporal": len(valid),
        "n_fano_temporal": len(fano_asym),
        "n_nonfano_temporal": len(nonfano_asym),
        "mean_asym_fano": sum(fano_asym) / len(fano_asym) if fano_asym else None,
        "mean_asym_nonfano": sum(nonfano_asym) / len(nonfano_asym) if nonfano_asym else None,
        "cohens_d": d,
        "eta_squared": eta2,
        "permutation_p": p_val,
        "permutation_n": 10000,
        "total_cases": total_cases,
        "total_temporal": total_temporal,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out_path.relative_to(repo)}")

if __name__ == "__main__":
    main()
