#!/usr/bin/env python3
"""
Direction C: Cross-domain Fano profile comparison.

Computes the 7-line Fano profile for DDI data and compares it
with the ABIDE Connectomics Fano profile from the taxonomy analysis.

If the same Fano lines dominate in both domains, this suggests
the algebraic structure is domain-invariant, not an artifact.
"""

import json
import math
import csv
from pathlib import Path

# The 7 Fano lines of the octonion multiplication table.
# Each line is a set of 3 imaginary basis indices {e_i, e_j, e_k}
# such that e_i * e_j = e_k (cyclically).
FANO_LINES = [
    (1, 2, 3),  # line 0: e1*e2=e3
    (1, 4, 5),  # line 1: e1*e4=e5
    (1, 6, 7),  # line 2: e1*e6=e7 (sign: e1*e7=-e6)
    (2, 4, 6),  # line 3: e2*e4=e6
    (2, 5, 7),  # line 4: e2*e5=e7
    (3, 4, 7),  # line 5: e3*e4=e7
    (3, 5, 6),  # line 6: e3*e5=-e6 (alternative sign convention)
]

CYP_TO_BASIS = {
    "CYP1A2": 1,
    "CYP2C9": 2,
    "CYP2C8": 3,
    "CYP2B6": 4,
    "CYP2C19": 5,
    "CYP2D6": 6,
    "CYP3A4": 7,
}


def triple_fano_line(a, b, c):
    """Return which Fano line(s) a triple belongs to, or empty list if non-Fano."""
    triple_set = {a, b, c}
    matches = []
    for i, line in enumerate(FANO_LINES):
        if set(line) == triple_set:
            matches.append(i)
    return matches


def oct_basis(j):
    v = [0.0] * 8
    v[j] = 1.0
    return v


def oct_mul(a, b):
    r = [0.0] * 8
    r[0] = a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3]-a[4]*b[4]-a[5]*b[5]-a[6]*b[6]-a[7]*b[7]
    r[1] = a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2]+a[4]*b[5]-a[5]*b[4]-a[6]*b[7]+a[7]*b[6]
    r[2] = a[0]*b[2]+a[2]*b[0]-a[1]*b[3]+a[3]*b[1]+a[4]*b[6]-a[6]*b[4]+a[5]*b[7]-a[7]*b[5]
    r[3] = a[0]*b[3]+a[3]*b[0]+a[1]*b[2]-a[2]*b[1]+a[4]*b[7]-a[7]*b[4]-a[5]*b[6]+a[6]*b[5]
    r[4] = a[0]*b[4]+a[4]*b[0]-a[1]*b[5]+a[5]*b[1]-a[2]*b[6]+a[6]*b[2]-a[3]*b[7]+a[7]*b[3]
    r[5] = a[0]*b[5]+a[5]*b[0]+a[1]*b[4]-a[4]*b[1]-a[2]*b[7]+a[7]*b[2]+a[3]*b[6]-a[6]*b[3]
    r[6] = a[0]*b[6]+a[6]*b[0]+a[1]*b[7]-a[7]*b[1]+a[2]*b[4]-a[4]*b[2]-a[3]*b[5]+a[5]*b[3]
    r[7] = a[0]*b[7]+a[7]*b[0]-a[1]*b[6]+a[6]*b[1]-a[2]*b[5]+a[5]*b[2]+a[3]*b[4]-a[4]*b[3]
    return r


def assoc_norm(i, j, k):
    a, b, c = oct_basis(i), oct_basis(j), oct_basis(k)
    ab_c = oct_mul(oct_mul(a, b), c)
    a_bc = oct_mul(a, oct_mul(b, c))
    diff = [ab_c[d] - a_bc[d] for d in range(8)]
    return math.sqrt(sum(d*d for d in diff))


def cosine_sim(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(x*x for x in b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return dot / (na * nb)


def kl_divergence(p, q):
    """KL(P || Q) with epsilon smoothing."""
    eps = 1e-10
    return sum(pi * math.log((pi + eps) / (qi + eps)) for pi, qi in zip(p, q))


def jensen_shannon(p, q):
    m = [(pi + qi) / 2 for pi, qi in zip(p, q)]
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


def main():
    repo = Path(__file__).resolve().parent.parent
    
    print("=" * 70)
    print("  Direction C: Cross-Domain Fano Profile Comparison")
    print("=" * 70)
    
    # --- DDI Fano profile ---
    # For each of the 35 triples, compute which Fano line it falls on
    # (if any) and aggregate signal (|asymmetry| × temporal) per line.
    
    print("\n  Computing DDI Fano profile...")
    
    # Load both DDI datasets
    for dataset_name, csv_path in [
        ("DrugBank expanded", repo / "data" / "faers_drugbank.csv"),
    ]:
        ddi_profile_asym = [0.0] * 7
        ddi_profile_cases = [0.0] * 7
        ddi_profile_count = [0] * 7
        
        total_assoc_by_line = [0.0] * 7
        total_non_fano_asym = 0.0
        total_non_fano_cases = 0.0
        n_non_fano = 0
        
        with open(csv_path) as f:
            for r in csv.DictReader(f):
                a = CYP_TO_BASIS[r["cyp_a"]]
                b = CYP_TO_BASIS[r["cyp_b"]]
                c = CYP_TO_BASIS[r["cyp_c"]]
                
                an = assoc_norm(a, b, c)
                temporal = int(r["temporal"])
                cases = int(r["cases"])
                asym = abs(float(r["asymmetry"])) if temporal > 0 else 0.0
                
                lines = triple_fano_line(a, b, c)
                
                if lines:
                    for line_idx in lines:
                        ddi_profile_asym[line_idx] += asym * temporal
                        ddi_profile_cases[line_idx] += cases
                        ddi_profile_count[line_idx] += 1
                else:
                    total_non_fano_asym += asym * temporal if temporal > 0 else 0
                    total_non_fano_cases += cases
                    n_non_fano += 1
        
        # Normalize to probability simplex
        total_asym = sum(ddi_profile_asym)
        total_cases_fano = sum(ddi_profile_cases)
        
        if total_asym > 0:
            ddi_simplex_asym = [x / total_asym for x in ddi_profile_asym]
        else:
            ddi_simplex_asym = [1/7] * 7
        
        if total_cases_fano > 0:
            ddi_simplex_cases = [x / total_cases_fano for x in ddi_profile_cases]
        else:
            ddi_simplex_cases = [1/7] * 7
        
        print(f"\n  DDI Fano profile ({dataset_name}):")
        print(f"  {'Line':>6} {'Triples':>8} {'|asym|*T':>10} {'Cases':>8} {'Asym%':>7} {'Cases%':>7}")
        for i in range(7):
            print(f"  {i:>6} {ddi_profile_count[i]:>8} {ddi_profile_asym[i]:>10.1f} {ddi_profile_cases[i]:>8.0f} {ddi_simplex_asym[i]:>7.3f} {ddi_simplex_cases[i]:>7.3f}")
        print(f"  {'NonF':>6} {n_non_fano:>8} {total_non_fano_asym:>10.1f} {total_non_fano_cases:>8.0f}")
    
    # --- ABIDE Fano profiles ---
    print("\n  ABIDE Fano profiles (from taxonomy analysis):")
    
    abide_profiles = {}
    for label, json_path in [
        ("pilot (n=100, 3 sites)", repo / "experiments" / "non_assoc_connectomics" / "fano_taxonomy_summary.json"),
        ("site-balanced (n=100, 20 sites)", repo / "experiments" / "non_assoc_connectomics" / "fano_taxonomy_site_balanced_summary.json"),
    ]:
        with open(json_path) as f:
            data = json.load(f)
        simplex = data["mean_simplex"]
        abide_profiles[label] = simplex
        print(f"\n  {label}:")
        print(f"  {'Line':>6} {'Weight':>8}")
        for i, w in enumerate(simplex):
            marker = " <--" if w == max(simplex) else ""
            print(f"  {i:>6} {w:>8.4f}{marker}")
    
    # --- Comparison ---
    print(f"\n{'='*70}")
    print(f"  Cross-domain comparison")
    print(f"{'='*70}")
    
    uniform = [1/7] * 7
    
    for abide_label, abide_simplex in abide_profiles.items():
        print(f"\n  ABIDE ({abide_label}) vs DDI:")
        
        cos = cosine_sim(ddi_simplex_asym, abide_simplex)
        js = jensen_shannon(ddi_simplex_asym, abide_simplex)
        js_uniform = jensen_shannon(uniform, abide_simplex)
        
        print(f"    Cosine similarity (DDI asym vs ABIDE): {cos:.4f}")
        print(f"    Jensen-Shannon divergence:              {js:.4f}")
        print(f"    JS divergence (uniform vs ABIDE):       {js_uniform:.4f}")
        
        # Rank correlation of the 7 lines
        ddi_rank = sorted(range(7), key=lambda i: ddi_simplex_asym[i], reverse=True)
        abide_rank = sorted(range(7), key=lambda i: abide_simplex[i], reverse=True)
        
        print(f"    DDI top-3 lines:   {ddi_rank[:3]}")
        print(f"    ABIDE top-3 lines: {abide_rank[:3]}")
        
        overlap = len(set(ddi_rank[:3]) & set(abide_rank[:3]))
        print(f"    Top-3 overlap:     {overlap}/3")
        
        # Dominant line match
        ddi_dominant = ddi_rank[0]
        abide_dominant = abide_rank[0]
        print(f"    DDI dominant line:  {ddi_dominant}")
        print(f"    ABIDE dominant line: {abide_dominant}")
        if ddi_dominant == abide_dominant:
            print(f"    ** Dominant line MATCH **")
        else:
            print(f"    Dominant lines differ")
    
    # Spearman rank correlation
    print(f"\n  Spearman rank correlation (7 lines):")
    for abide_label, abide_simplex in abide_profiles.items():
        # Compute ranks
        def rank_vector(vals):
            order = sorted(range(len(vals)), key=lambda i: vals[i])
            ranks = [0.0] * len(vals)
            for rank, idx in enumerate(order):
                ranks[idx] = rank + 1
            return ranks
        
        r_ddi = rank_vector(ddi_simplex_asym)
        r_abide = rank_vector(abide_simplex)
        n = 7
        d2 = sum((r_ddi[i] - r_abide[i]) ** 2 for i in range(n))
        rho = 1 - 6 * d2 / (n * (n * n - 1))
        print(f"    DDI vs ABIDE ({abide_label}): rho = {rho:.4f}")
    
    # Save results
    results = {
        "ddi_fano_profile_asym": ddi_simplex_asym,
        "ddi_fano_profile_cases": ddi_simplex_cases,
        "ddi_fano_counts": ddi_profile_count,
        "abide_profiles": abide_profiles,
        "comparisons": {}
    }
    
    for abide_label, abide_simplex in abide_profiles.items():
        cos = cosine_sim(ddi_simplex_asym, abide_simplex)
        js = jensen_shannon(ddi_simplex_asym, abide_simplex)
        
        def rank_vector(vals):
            order = sorted(range(len(vals)), key=lambda i: vals[i])
            ranks = [0.0] * len(vals)
            for rank, idx in enumerate(order):
                ranks[idx] = rank + 1
            return ranks
        
        r_ddi = rank_vector(ddi_simplex_asym)
        r_abide = rank_vector(abide_simplex)
        d2 = sum((r_ddi[i] - r_abide[i]) ** 2 for i in range(7))
        rho = 1 - 6 * d2 / (7 * 48)
        
        results["comparisons"][abide_label] = {
            "cosine_similarity": cos,
            "jensen_shannon": js,
            "spearman_rho": rho,
        }
    
    out_path = repo / "experiments" / "non_assoc_connectomics" / "cross_domain_fano_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out_path.relative_to(repo)}")

if __name__ == "__main__":
    main()
