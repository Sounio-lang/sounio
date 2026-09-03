#!/usr/bin/env python3
"""
Direction D + E: Covariate analysis for DDI associator effect.

E) Partial correlation: associator norm vs asymmetry controlling for
   demographics (mean_age, frac_female) from faers_demographics.csv.

D) Concomitant CYP modulation: does concomitant CYP load modulate
   the associator -> asymmetry relationship?
"""

import csv
import json
from pathlib import Path

from ddi_campaign_lib import FANO_TRIPLES, partial_correlation, pearson_r

def main():
    repo = Path(__file__).resolve().parent.parent.parent
    
    # Load all datasets
    drugbank = {}
    with open(repo / "data" / "faers_drugbank.csv") as f:
        for r in csv.DictReader(f):
            key = (r["cyp_a"], r["cyp_b"], r["cyp_c"])
            drugbank[key] = r
    
    demographics = {}
    with open(repo / "data" / "faers_demographics.csv") as f:
        for r in csv.DictReader(f):
            key = (r["cyp_a"], r["cyp_b"], r["cyp_c"])
            demographics[key] = r
    
    concomitants = {}
    with open(repo / "data" / "faers_concomitants.csv") as f:
        for r in csv.DictReader(f):
            key = (r["cyp_a"], r["cyp_b"], r["cyp_c"])
            concomitants[key] = r
    
    # Build merged dataset
    triples = sorted(drugbank.keys())
    
    print("=" * 70)
    print("  Direction E: Demographics as covariates")
    print("=" * 70)
    
    assoc_norms = []
    asymmetries = []
    ages = []
    frac_females = []
    valid_keys = []
    
    for key in triples:
        db = drugbank[key]
        dm = demographics.get(key)
        temporal = int(db["temporal"])
        if temporal <= 0:
            continue
        asym = float(db["asymmetry"])
        is_fano = key in FANO_TRIPLES
        assoc = 0.0 if is_fano else 2.0
        
        assoc_norms.append(assoc)
        asymmetries.append(asym)
        valid_keys.append(key)
        
        if dm and float(dm.get("mean_age", -1)) > 0:
            ages.append(float(dm["mean_age"]))
        else:
            ages.append(None)
        
        if dm and float(dm.get("frac_female", -1)) >= 0:
            frac_females.append(float(dm["frac_female"]))
        else:
            frac_females.append(None)
    
    n = len(assoc_norms)
    print(f"\n  Valid triples (temporal > 0): {n}")
    
    r_raw, p_raw = pearson_r(assoc_norms, asymmetries)
    print(f"\n  Raw correlation (assoc_norm vs asymmetry):")
    print(f"    r = {r_raw:.4f}, p = {p_raw:.4f}")
    
    # Partial correlation controlling for age
    valid_age = [(i, ages[i]) for i in range(n) if ages[i] is not None]
    if len(valid_age) >= 5:
        idx_a = [v[0] for v in valid_age]
        r_age, p_age = partial_correlation(
            [assoc_norms[i] for i in idx_a],
            [asymmetries[i] for i in idx_a],
            [[ages[i] for i in idx_a]]
        )
        print(f"\n  Partial correlation controlling for mean_age (n={len(idx_a)}):")
        print(f"    r = {r_age:.4f}, p = {p_age:.4f}")
    else:
        r_age, p_age = None, None
        print(f"\n  Insufficient age data for partial correlation")
    
    # Partial correlation controlling for frac_female
    valid_sex = [(i, frac_females[i]) for i in range(n) if frac_females[i] is not None]
    if len(valid_sex) >= 5:
        idx_s = [v[0] for v in valid_sex]
        r_sex, p_sex = partial_correlation(
            [assoc_norms[i] for i in idx_s],
            [asymmetries[i] for i in idx_s],
            [[frac_females[i] for i in idx_s]]
        )
        print(f"\n  Partial correlation controlling for frac_female (n={len(idx_s)}):")
        print(f"    r = {r_sex:.4f}, p = {p_sex:.4f}")
    else:
        r_sex, p_sex = None, None
        print(f"\n  Insufficient sex data for partial correlation")
    
    # Both covariates
    valid_both = [i for i in range(n) if ages[i] is not None and frac_females[i] is not None]
    if len(valid_both) >= 6:
        r_both, p_both = partial_correlation(
            [assoc_norms[i] for i in valid_both],
            [asymmetries[i] for i in valid_both],
            [[ages[i] for i in valid_both], [frac_females[i] for i in valid_both]]
        )
        print(f"\n  Partial correlation controlling for age + sex (n={len(valid_both)}):")
        print(f"    r = {r_both:.4f}, p = {p_both:.4f}")
    else:
        r_both, p_both = None, None
    
    print(f"\n  Interpretation:")
    if r_raw is not None and abs(r_raw) < 0.15:
        print(f"    Raw correlation is weak (|r| < 0.15) — no effect to confound.")
    elif r_age is not None and r_both is not None:
        if abs(r_both) < abs(r_raw) * 0.5:
            print(f"    Effect collapses after demographic adjustment — confounded.")
        else:
            print(f"    Effect survives demographic adjustment — not confounded.")
    
    print(f"\n{'='*70}")
    print(f"  Direction D: Concomitant CYP modulation")
    print(f"{'='*70}")
    
    cyp_loads = []
    asym_d = []
    assoc_d = []
    fano_labels = []
    
    for key in triples:
        db = drugbank[key]
        cc = concomitants.get(key)
        temporal = int(db["temporal"])
        if temporal <= 0:
            continue
        asym = float(db["asymmetry"])
        is_fano = key in FANO_TRIPLES
        
        if cc:
            cyp_cols = [f"conc_cyp{i}" for i in range(1, 8)]
            load = sum(int(cc.get(c, 0)) for c in cyp_cols)
        else:
            load = 0
        
        cyp_loads.append(load)
        asym_d.append(asym)
        assoc_d.append(0.0 if is_fano else 2.0)
        fano_labels.append(is_fano)
    
    n_d = len(cyp_loads)
    print(f"\n  Valid triples: {n_d}")
    
    r_load_asym, p_load_asym = pearson_r(cyp_loads, asym_d)
    print(f"\n  CYP load vs asymmetry:")
    print(f"    r = {r_load_asym:.4f}, p = {p_load_asym:.4f}")
    
    r_load_assoc, p_load_assoc = pearson_r(cyp_loads, assoc_d)
    print(f"\n  CYP load vs associator norm:")
    print(f"    r = {r_load_assoc:.4f}, p = {p_load_assoc:.4f}")
    
    # Split non-Fano by CYP load median
    nf_data = [(cyp_loads[i], asym_d[i]) for i in range(n_d) if not fano_labels[i]]
    if len(nf_data) >= 4:
        median_load = sorted([d[0] for d in nf_data])[len(nf_data) // 2]
        high_load = [d[1] for d in nf_data if d[0] >= median_load]
        low_load = [d[1] for d in nf_data if d[0] < median_load]
        if high_load and low_load:
            mean_hi = sum(high_load) / len(high_load)
            mean_lo = sum(low_load) / len(low_load)
            print(f"\n  Non-Fano triples by CYP load (median split at {median_load}):")
            print(f"    High load (n={len(high_load)}): mean asymmetry = {mean_hi:.4f}")
            print(f"    Low  load (n={len(low_load)}):  mean asymmetry = {mean_lo:.4f}")
            print(f"    Difference = {mean_hi - mean_lo:.4f}")
    
    # Partial: asymmetry ~ assoc_norm controlling for CYP load
    if n_d >= 5:
        r_partial_load, p_partial_load = partial_correlation(assoc_d, asym_d, [cyp_loads])
        print(f"\n  Partial correlation (assoc_norm vs asymmetry | CYP load):")
        print(f"    r = {r_partial_load:.4f}, p = {p_partial_load:.4f}")
    else:
        r_partial_load, p_partial_load = None, None
    
    # Save results
    results = {
        "direction_E": {
            "raw_correlation": {"r": r_raw, "p": p_raw, "n": n},
            "partial_age": {"r": r_age, "p": p_age} if r_age is not None else None,
            "partial_sex": {"r": r_sex, "p": p_sex} if r_sex is not None else None,
            "partial_both": {"r": r_both, "p": p_both} if r_both is not None else None,
        },
        "direction_D": {
            "cyp_load_vs_asymmetry": {"r": r_load_asym, "p": p_load_asym},
            "cyp_load_vs_assoc": {"r": r_load_assoc, "p": p_load_assoc},
            "partial_assoc_asym_given_load": {"r": r_partial_load, "p": p_partial_load} if r_partial_load is not None else None,
        },
    }
    out_path = repo / "experiments" / "non_assoc_connectomics" / "ddi_covariate_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out_path.relative_to(repo)}")

if __name__ == "__main__":
    main()
