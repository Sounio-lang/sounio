#!/usr/bin/env python3
"""
G₂ Bridge: Optimal Labeling Experiment

For each subject's 7 Laplacian eigenvectors, search all 5040 permutations
of eigenvector-to-octonion-basis assignment. Find the labeling that
maximizes Fano triple-product energy. Compare ASD vs TD.

Question: Does the brain's spectral geometry have a hidden G₂-compatible
structure, and is it clinically discriminative?
"""

import numpy as np
from itertools import permutations
import json, os, sys

# ============================================================================
# Fano structure
# ============================================================================

# 7 oriented Fano lines (1-indexed basis elements)
FANO_LINES = [
    (1, 2, 3), (1, 4, 5), (1, 7, 6),
    (2, 4, 6), (2, 5, 7), (3, 4, 7), (3, 6, 5)
]

# Convert to 0-indexed sets for fast lookup
FANO_TRIPLES_0 = [tuple(sorted([i-1, j-1, k-1])) for (i,j,k) in FANO_LINES]
FANO_SET = set(FANO_TRIPLES_0)

# All 35 unordered triples of {0..6}
ALL_TRIPLES = []
for a in range(7):
    for b in range(a+1, 7):
        for c in range(b+1, 7):
            ALL_TRIPLES.append((a, b, c))

NONFANO_TRIPLES = [t for t in ALL_TRIPLES if t not in FANO_SET]

assert len(FANO_TRIPLES_0) == 7
assert len(NONFANO_TRIPLES) == 28
assert len(ALL_TRIPLES) == 35


def compute_triple_products(frame):
    """
    Precompute all 35 triple products for a 7-frame.
    tp[(a,b,c)] = Σ_v frame[a,v] * frame[b,v] * frame[c,v]
    Returns dict keyed by sorted triple.
    """
    tp = {}
    for (a, b, c) in ALL_TRIPLES:
        tp[(a, b, c)] = np.sum(frame[a] * frame[b] * frame[c])
    return tp


def fano_energy_for_labeling(tp_dict, perm):
    """
    Given precomputed triple products and a permutation σ (mapping
    eigenvector index → basis element index), compute the Fano energy.

    perm[k] = which basis element eigenvector k maps to.
    Fano energy = mean of tp²  for triples where the BASIS indices
    form a Fano line.
    """
    fano_energy = 0.0
    count = 0
    for (bi, bj, bk) in FANO_TRIPLES_0:
        # Which eigenvectors map to basis elements bi, bj, bk?
        # perm[ei] = bi  →  ei = inv_perm[bi]
        # We need inv_perm
        # Actually, let's precompute inv_perm
        pass

    # More efficient: precompute inverse permutation
    inv_perm = [0] * 7
    for i in range(7):
        inv_perm[perm[i]] = i

    fano_sum = 0.0
    for (bi, bj, bk) in FANO_TRIPLES_0:
        ei, ej, ek = inv_perm[bi], inv_perm[bj], inv_perm[bk]
        triple = tuple(sorted([ei, ej, ek]))
        val = tp_dict[triple]
        fano_sum += val * val

    nonfano_sum = 0.0
    for (bi, bj, bk) in NONFANO_TRIPLES:
        ei, ej, ek = inv_perm[bi], inv_perm[bj], inv_perm[bk]
        triple = tuple(sorted([ei, ej, ek]))
        val = tp_dict[triple]
        nonfano_sum += val * val

    fano_energy = fano_sum / 7.0
    nonfano_energy = nonfano_sum / 28.0
    total = fano_sum + nonfano_sum

    return fano_energy, nonfano_energy, total


def find_optimal_labeling(frame):
    """
    Search all 5040 permutations of 7 eigenvectors → 7 basis elements.
    Return the one maximizing Fano triple-product energy.
    """
    tp_dict = compute_triple_products(frame)

    best_fano = -1.0
    best_perm = None
    best_ratio = 0.0
    best_nonfano = 0.0
    best_total = 0.0

    # Stats across all labelings
    all_fano = []
    all_ratios = []

    for perm in permutations(range(7)):
        fe, nfe, total = fano_energy_for_labeling(tp_dict, perm)
        ratio = fe / (fe + nfe) if (fe + nfe) > 1e-30 else 0
        all_fano.append(fe)
        all_ratios.append(ratio)

        if fe > best_fano:
            best_fano = fe
            best_perm = perm
            best_ratio = ratio
            best_nonfano = nfe
            best_total = total

    return {
        'best_fano_energy': best_fano,
        'best_nonfano_energy': best_nonfano,
        'best_ratio': best_ratio,
        'best_total': best_total,
        'best_perm': list(best_perm),
        'mean_fano_all': np.mean(all_fano),
        'std_fano_all': np.std(all_fano),
        'mean_ratio_all': np.mean(all_ratios),
        'max_ratio': max(all_ratios),
        # How exceptional is the best labeling?
        'z_score': (best_fano - np.mean(all_fano)) / max(np.std(all_fano), 1e-30),
    }


# ============================================================================
# Load previously computed data
# ============================================================================

def load_abide_frames(cache_dir="artifacts/research/abide"):
    """
    Load time series, compute correlation, extract eigenvectors.
    Reuses cached .1D files from the previous experiment.
    """
    import pandas as pd

    pheno_path = os.path.join(cache_dir, "phenotypic.csv")
    if not os.path.exists(pheno_path):
        print("  Run g2_abide_experiment.py first to download data.")
        return None, None

    pheno = pd.read_csv(pheno_path)

    asd_frames = []
    td_frames = []

    for _, row in pheno.iterrows():
        fid = row['FILE_ID']
        dx = row['DX_GROUP']  # 1=ASD, 2=TD
        if not fid or fid == 'no_filename':
            continue

        path = os.path.join(cache_dir, f"{fid}_rois_cc200.1D")
        if not os.path.exists(path):
            continue

        try:
            ts = np.loadtxt(path)
            if ts.ndim != 2 or ts.shape[0] < 20 or ts.shape[1] < 8:
                continue
            corr = np.corrcoef(ts.T)
            corr = np.nan_to_num(corr, nan=0.0)
            np.fill_diagonal(corr, 0)
            adj = np.maximum(corr, 0)
            n = adj.shape[0]
            deg = adj.sum(axis=1)
            L = np.diag(deg) - adj
            from scipy import linalg
            eigenvalues, eigenvectors = linalg.eigh(L)
            if len(eigenvalues) < 8:
                continue
            frame = eigenvectors[:, 1:8].T  # (7, n)

            record = {'file_id': fid, 'frame': frame, 'lambda2': eigenvalues[1]}

            if dx == 1:
                asd_frames.append(record)
            elif dx == 2:
                td_frames.append(record)
        except:
            continue

    return asd_frames, td_frames


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("G₂ OPTIMAL LABELING: searching 5040 eigenvector↔basis assignments")
    print("=" * 70)
    print()

    print("[1/3] Loading ABIDE-I eigenvector frames...")
    asd_frames, td_frames = load_abide_frames()

    if asd_frames is None:
        return

    print(f"  Loaded: {len(asd_frames)} ASD, {len(td_frames)} TD")

    max_n = 175  # use all available
    asd_frames = asd_frames[:max_n]
    td_frames = td_frames[:max_n]
    print(f"  Using:  {len(asd_frames)} ASD, {len(td_frames)} TD")

    # Step 2: Find optimal labeling for each subject
    print()
    print("[2/3] Searching optimal labelings (5040 per subject)...")

    asd_results = []
    for i, rec in enumerate(asd_frames):
        result = find_optimal_labeling(rec['frame'])
        result['file_id'] = rec['file_id']
        result['group'] = 'ASD'
        result['lambda2'] = float(rec['lambda2'])
        asd_results.append(result)
        if (i + 1) % 20 == 0:
            print(f"  ASD: {i+1}/{len(asd_frames)}")

    td_results = []
    for i, rec in enumerate(td_frames):
        result = find_optimal_labeling(rec['frame'])
        result['file_id'] = rec['file_id']
        result['group'] = 'TD'
        result['lambda2'] = float(rec['lambda2'])
        td_results.append(result)
        if (i + 1) % 20 == 0:
            print(f"  TD:  {i+1}/{len(td_frames)}")

    # Step 3: Statistical comparison
    print()
    print("[3/3] Permutation test on optimal Fano energy...")

    asd_best = np.array([r['best_fano_energy'] for r in asd_results])
    td_best = np.array([r['best_fano_energy'] for r in td_results])
    asd_ratio = np.array([r['best_ratio'] for r in asd_results])
    td_ratio = np.array([r['best_ratio'] for r in td_results])
    asd_z = np.array([r['z_score'] for r in asd_results])
    td_z = np.array([r['z_score'] for r in td_results])

    def perm_test(a, b, n_perms=10000, seed=42):
        rng = np.random.RandomState(seed)
        obs = abs(a.mean() - b.mean())
        combined = np.concatenate([a, b])
        na = len(a)
        count = 0
        for _ in range(n_perms):
            p = rng.permutation(len(combined))
            d = abs(combined[p[:na]].mean() - combined[p[na:]].mean())
            if d >= obs:
                count += 1
        return count / n_perms

    def cohens_d(a, b):
        sp = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        return (a.mean() - b.mean()) / sp if sp > 1e-15 else 0

    p_best = perm_test(asd_best, td_best)
    p_ratio = perm_test(asd_ratio, td_ratio)
    p_z = perm_test(asd_z, td_z)

    d_best = cohens_d(asd_best, td_best)
    d_ratio = cohens_d(asd_ratio, td_ratio)
    d_z = cohens_d(asd_z, td_z)

    # Report
    print()
    print("=" * 70)
    print("RESULTS: OPTIMAL LABELING ANALYSIS")
    print("=" * 70)
    print()
    print(f"Subjects: ASD n={len(asd_results)}, TD n={len(td_results)}")
    print(f"Labelings searched per subject: 5040 (= 7!)")
    print()

    print("1. Best Fano energy (max over all labelings):")
    print(f"   ASD:  {asd_best.mean():.6f} ± {asd_best.std():.6f}")
    print(f"   TD:   {td_best.mean():.6f} ± {td_best.std():.6f}")
    print(f"   d = {d_best:.3f}, p = {p_best:.4f}")
    print()

    print("2. Best Fano ratio (Fano energy / total, at optimal labeling):")
    print(f"   ASD:  {asd_ratio.mean():.4f} ± {asd_ratio.std():.4f}")
    print(f"   TD:   {td_ratio.mean():.4f} ± {td_ratio.std():.4f}")
    print(f"   d = {d_ratio:.3f}, p = {p_ratio:.4f}")
    print()

    print("3. Z-score of best labeling (how exceptional vs random labeling):")
    print(f"   ASD:  {asd_z.mean():.2f} ± {asd_z.std():.2f}")
    print(f"   TD:   {td_z.mean():.2f} ± {td_z.std():.2f}")
    print(f"   d = {d_z:.3f}, p = {p_z:.4f}")
    print()

    # Additional: distribution of optimal permutations
    # Are certain basis assignments preferred?
    print("4. Optimal permutation structure:")
    asd_perms = [tuple(r['best_perm']) for r in asd_results]
    td_perms = [tuple(r['best_perm']) for r in td_results]

    # Which eigenvector most often maps to e₁ (basis 0)?
    asd_e1 = [p[0] for p in asd_perms]  # which eigvec → basis 0
    td_e1 = [p[0] for p in td_perms]

    from collections import Counter
    print(f"   Eigvec most often assigned to e₁:")
    asd_c = Counter(asd_e1).most_common(3)
    td_c = Counter(td_e1).most_common(3)
    print(f"   ASD: {asd_c}")
    print(f"   TD:  {td_c}")

    # Singlet assignment (basis 6 = e₇)
    asd_e7 = [p.index(6) for p in asd_perms]  # which eigvec → basis 6 (singlet)
    td_e7 = [p.index(6) for p in td_perms]
    asd_e7c = Counter(asd_e7).most_common(3)
    td_e7c = Counter(td_e7).most_common(3)
    print(f"   Eigvec most often assigned to e₇ (SU(3) singlet):")
    print(f"   ASD: {asd_e7c}")
    print(f"   TD:  {td_e7c}")

    print()
    print("-" * 70)
    print("INTERPRETATION")
    print("-" * 70)

    sig_markers = []
    if p_best < 0.05: sig_markers.append(f"best Fano energy (p={p_best:.4f})")
    if p_ratio < 0.05: sig_markers.append(f"Fano ratio (p={p_ratio:.4f})")
    if p_z < 0.05: sig_markers.append(f"z-score (p={p_z:.4f})")

    if sig_markers:
        print(f"  SIGNIFICANT differences found in: {', '.join(sig_markers)}")
        if d_best > 0:
            print("  → ASD connectomes support STRONGER optimal G₂ structure.")
        else:
            print("  → TD connectomes support STRONGER optimal G₂ structure.")
    else:
        print("  No significant differences at p<0.05.")
        if p_best < 0.1 or p_ratio < 0.1 or p_z < 0.1:
            trends = []
            if p_best < 0.1: trends.append(f"Fano energy p={p_best:.3f}")
            if p_ratio < 0.1: trends.append(f"ratio p={p_ratio:.3f}")
            if p_z < 0.1: trends.append(f"z-score p={p_z:.3f}")
            print(f"  Trends: {', '.join(trends)}")
        print("  The optimal G₂ labeling is equally strong in both groups.")

    # Save
    output = {
        'experiment': 'G2_optimal_labeling',
        'parcellation': 'CC200',
        'n_labelings': 5040,
        'n_asd': len(asd_results),
        'n_td': len(td_results),
        'results': {
            'best_fano_energy': {'asd_mean': float(asd_best.mean()), 'td_mean': float(td_best.mean()), 'd': d_best, 'p': p_best},
            'best_ratio': {'asd_mean': float(asd_ratio.mean()), 'td_mean': float(td_ratio.mean()), 'd': d_ratio, 'p': p_ratio},
            'z_score': {'asd_mean': float(asd_z.mean()), 'td_mean': float(td_z.mean()), 'd': d_z, 'p': p_z},
        },
    }
    outpath = "artifacts/research/g2_optimal_labeling_results.json"
    with open(outpath, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {outpath}")


if __name__ == "__main__":
    main()
