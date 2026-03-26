#!/usr/bin/env python3
"""
G₂ Bridge Experiment: ABIDE-I connectome → octonion invariants.

Downloads preprocessed functional connectivity matrices from ABIDE-I,
computes the G₂ 3-form and SU(3) branching quality on Laplacian
eigenvector frames, and runs a permutation test comparing ASD vs TD.

Uses the same mathematics implemented in stdlib/algebra/octonion.sio
and stdlib/hypercomplex_graph/connectome.sio.

References:
  Di Martino+ 2014, "The autism brain imaging data exchange: towards
    a large-scale evaluation of the intrinsic brain architecture in autism"
  Harvey-Lawson 1982, "Calibrated geometries" (G₂ 3-form)
  Baez 2002, "The Octonions" §4.1
"""

import numpy as np
from scipy import linalg
import os, json, sys

# ============================================================================
# ABIDE-I data download
# ============================================================================

ABIDE_PHENO_URL = (
    "https://s3.amazonaws.com/fcp-indi/data/Projects/"
    "ABIDE_Initiative/Phenotypic_V1_0b_preprocessed1.csv"
)

def fetch_abide_phenotypic(cache_dir="artifacts/research/abide"):
    """Download ABIDE-I phenotypic file."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, "phenotypic.csv")
    if not os.path.exists(path):
        print(f"Downloading phenotypic data...")
        import urllib.request
        urllib.request.urlretrieve(ABIDE_PHENO_URL, path)
    import pandas as pd
    df = pd.read_csv(path)
    return df


def fetch_correlation_matrix(file_id, pipeline="cpac", strategy="filt_noglobal",
                              parcellation="rois_cc200", cache_dir="artifacts/research/abide"):
    """Download ROI time series and compute correlation matrix."""
    if not file_id or file_id == 'no_filename':
        return None
    os.makedirs(cache_dir, exist_ok=True)
    fname = f"{file_id}_{parcellation}.1D"
    path = os.path.join(cache_dir, fname)

    if not os.path.exists(path):
        url = (
            f"https://s3.amazonaws.com/fcp-indi/data/Projects/"
            f"ABIDE_Initiative/Outputs/{pipeline}/{strategy}/"
            f"{parcellation}/{fname}"
        )
        try:
            import urllib.request
            urllib.request.urlretrieve(url, path)
        except Exception:
            return None

    try:
        ts = np.loadtxt(path)  # shape: (timepoints, n_rois)
        if ts.ndim != 2 or ts.shape[0] < 20 or ts.shape[1] < 8:
            return None
        corr = np.corrcoef(ts.T)
        # Replace NaN with 0
        corr = np.nan_to_num(corr, nan=0.0)
        return corr
    except:
        return None


# ============================================================================
# Cayley-Dickson sign (matches stdlib/algebra/octonion.sio)
# ============================================================================

def cd_sigma_r(a, b, bits):
    if a == 0 or b == 0: return 1
    if bits <= 1: return -1
    half = 1 << (bits - 1)
    a_hi = 1 if a >= half else 0
    b_hi = 1 if b >= half else 0
    a_lo = a & (half - 1)
    b_lo = b & (half - 1)
    if a_hi == 0 and b_hi == 0: return cd_sigma_r(a_lo, b_lo, bits - 1)
    if a_hi == 0 and b_hi == 1: return cd_sigma_r(b_lo, a_lo, bits - 1)
    if a_hi == 1 and b_hi == 0:
        if b_lo == 0: return cd_sigma_r(a_lo, b_lo, bits - 1)
        return -cd_sigma_r(a_lo, b_lo, bits - 1)
    if b_lo == 0: return -cd_sigma_r(b_lo, a_lo, bits - 1)
    return cd_sigma_r(b_lo, a_lo, bits - 1)

def cd_sigma(a, b):
    return cd_sigma_r(a, b, 3)

# ============================================================================
# G₂ 3-form (matches stdlib/algebra/octonion.sio)
# ============================================================================

# 7 oriented Fano lines: (i,j,k) with 1-indexed basis elements
FANO_LINES = [
    (1, 2, 3), (1, 4, 5), (1, 7, 6),
    (2, 4, 6), (2, 5, 7), (3, 4, 7), (3, 6, 5)
]

def g2_3form_gram(gram):
    """
    G₂ 3-form on a 7×7 Gram matrix.
    For orthonormal frames this always equals 7, so it's NOT useful
    for comparing eigenvector frames (which are always orthonormal).
    """
    phi = 0.0
    for (i, j, k) in FANO_LINES:
        ii, jj, kk = i-1, j-1, k-1
        G = gram[np.ix_([ii,jj,kk],[ii,jj,kk])]
        det = np.linalg.det(G)
        sign = cd_sigma(i, j)
        phi += sign * det
    return phi


def g2_3form_coordinate(frame):
    """
    G₂ 3-form evaluated on coordinate structure of eigenvectors.

    Key insight: Laplacian eigenvectors ψ₁..ψ₇ are always orthonormal,
    so their Gram invariants are trivial. The INTERESTING structure is
    how the eigenvectors distribute their weight across brain regions.

    For each Fano triple (i,j,k), compute:
      φ_{ijk} = Σ_v ψ_i(v) * ψ_j(v) * ψ_k(v)
    This is the "triple product" — how much three eigenmodes co-localize
    at each vertex. For Fano-aligned triples, octonion algebra predicts
    this should differ from non-Fano triples.

    frame: array of shape (7, n) — 7 eigenvectors of dimension n.
    Returns: (phi_fano, phi_nonfano, phi_ratio)
    """
    n = frame.shape[1]

    # Compute triple products for all 35 unordered triples
    fano_set = set()
    for (i,j,k) in FANO_LINES:
        fano_set.add(tuple(sorted([i-1, j-1, k-1])))

    fano_vals = []
    nonfano_vals = []

    for a in range(7):
        for b in range(a+1, 7):
            for c in range(b+1, 7):
                # Triple product: Σ_v ψ_a(v) * ψ_b(v) * ψ_c(v)
                tp = np.sum(frame[a] * frame[b] * frame[c])
                triple = tuple(sorted([a, b, c]))
                if triple in fano_set:
                    fano_vals.append(tp)
                else:
                    nonfano_vals.append(tp)

    fano_energy = np.mean(np.array(fano_vals)**2) if fano_vals else 0
    nonfano_energy = np.mean(np.array(nonfano_vals)**2) if nonfano_vals else 0

    return fano_energy, nonfano_energy


def su3_branching_coordinate(frame):
    """
    SU(3) branching quality from eigenvector coordinate structure.

    Measures how much the 7 eigenvectors partition into 3+3+1 blocks
    in terms of their spatial overlap patterns.

    Block assignment: {ψ₁,ψ₂,ψ₃}→triplet, {ψ₄,ψ₅,ψ₆}→anti-triplet, {ψ₇}→singlet

    Q = within-block spatial overlap / total spatial overlap
    """
    def block(k):
        if k < 3: return 0
        if k < 6: return 1
        return 2

    # Spatial overlap matrix: O[a,b] = Σ_v |ψ_a(v) * ψ_b(v)|
    n = frame.shape[1]
    overlap = np.zeros((7, 7))
    for a in range(7):
        for b in range(7):
            overlap[a, b] = np.sum(np.abs(frame[a] * frame[b]))

    total = np.sum(overlap**2)
    off_block = 0.0
    for a in range(7):
        for b in range(7):
            if block(a) != block(b):
                off_block += overlap[a, b]**2

    if total < 1e-12:
        return 0.0
    return 1.0 - off_block / total


# ============================================================================
# Pipeline: adjacency matrix → G₂ invariants
# ============================================================================

def compute_g2_invariants(corr_matrix, n_eigenvectors=7):
    """
    Full pipeline for one subject:
      correlation matrix → Laplacian → eigenvectors → G₂ invariants

    Returns dict with phi, phi_abs, su3_quality, lambda2.
    """
    # Threshold to adjacency (keep positive correlations)
    adj = corr_matrix.copy()
    np.fill_diagonal(adj, 0)
    adj[adj < 0] = 0  # keep only positive correlations

    n = adj.shape[0]
    if n < 8:
        return None

    # Graph Laplacian
    deg = adj.sum(axis=1)
    L = np.diag(deg) - adj

    # Eigendecomposition (L is symmetric PSD)
    eigenvalues, eigenvectors = linalg.eigh(L)

    # Skip the trivial zero eigenvalue (first eigenvector = constant)
    # Take eigenvectors 1..7 (indices 1 to 7)
    if len(eigenvalues) < 8:
        return None

    lambda2 = eigenvalues[1]
    frame = eigenvectors[:, 1:8].T  # shape (7, n)

    # G₂ coordinate invariants (not Gram-based — those are trivial)
    fano_energy, nonfano_energy = g2_3form_coordinate(frame)

    # Fano/non-Fano ratio: how much Fano triples dominate
    total_energy = fano_energy + nonfano_energy
    fano_ratio = fano_energy / total_energy if total_energy > 1e-15 else 0

    # SU(3) branching quality (coordinate-based)
    q = su3_branching_coordinate(frame)

    return {
        'phi': fano_ratio,  # repurpose: Fano dominance ratio
        'phi_abs': fano_ratio,
        'fano_energy': fano_energy,
        'nonfano_energy': nonfano_energy,
        'su3_quality': q,
        'lambda2': lambda2,
        'n_nodes': n,
    }


# ============================================================================
# Permutation test
# ============================================================================

def permutation_test(group_a, group_b, n_perms=10000, seed=42):
    """
    Two-sided permutation test on |φ| and Q between groups.
    Returns dict with p-values, observed differences, effect sizes.
    """
    rng = np.random.RandomState(seed)

    a_phi = np.array([s['phi_abs'] for s in group_a])
    b_phi = np.array([s['phi_abs'] for s in group_b])
    a_q = np.array([s['su3_quality'] for s in group_a])
    b_q = np.array([s['su3_quality'] for s in group_b])

    obs_diff_phi = abs(a_phi.mean() - b_phi.mean())
    obs_diff_q = abs(a_q.mean() - b_q.mean())

    # Cohen's d
    pooled_std_phi = np.sqrt((a_phi.var(ddof=1) + b_phi.var(ddof=1)) / 2)
    pooled_std_q = np.sqrt((a_q.var(ddof=1) + b_q.var(ddof=1)) / 2)
    d_phi = (a_phi.mean() - b_phi.mean()) / pooled_std_phi if pooled_std_phi > 1e-12 else 0
    d_q = (a_q.mean() - b_q.mean()) / pooled_std_q if pooled_std_q > 1e-12 else 0

    # Permutation
    combined_phi = np.concatenate([a_phi, b_phi])
    combined_q = np.concatenate([a_q, b_q])
    na = len(a_phi)

    count_phi = 0
    count_q = 0
    for _ in range(n_perms):
        perm = rng.permutation(len(combined_phi))
        perm_a_phi = combined_phi[perm[:na]]
        perm_b_phi = combined_phi[perm[na:]]
        perm_a_q = combined_q[perm[:na]]
        perm_b_q = combined_q[perm[na:]]

        if abs(perm_a_phi.mean() - perm_b_phi.mean()) >= obs_diff_phi:
            count_phi += 1
        if abs(perm_a_q.mean() - perm_b_q.mean()) >= obs_diff_q:
            count_q += 1

    return {
        'obs_diff_phi': obs_diff_phi,
        'obs_diff_q': obs_diff_q,
        'p_phi': count_phi / n_perms,
        'p_q': count_q / n_perms,
        'd_phi': d_phi,
        'd_q': d_q,
        'mean_phi_a': a_phi.mean(),
        'mean_phi_b': b_phi.mean(),
        'std_phi_a': a_phi.std(ddof=1),
        'std_phi_b': b_phi.std(ddof=1),
        'mean_q_a': a_q.mean(),
        'mean_q_b': b_q.mean(),
        'std_q_a': a_q.std(ddof=1),
        'std_q_b': b_q.std(ddof=1),
        'n_a': len(group_a),
        'n_b': len(group_b),
        'n_perms': n_perms,
    }


# ============================================================================
# Main experiment
# ============================================================================

def main():
    print("=" * 70)
    print("G₂ BRIDGE EXPERIMENT: ABIDE-I Connectome → Octonion Invariants")
    print("=" * 70)
    print()

    # Step 1: Fetch phenotypic data
    print("[1/4] Fetching ABIDE-I phenotypic data...")
    try:
        pheno = fetch_abide_phenotypic()
    except Exception as e:
        print(f"  ERROR: {e}")
        print("  Falling back to site-specific download...")
        pheno = None

    if pheno is None:
        print("  Cannot fetch phenotypic data. Exiting.")
        return

    # Filter to subjects with usable data
    # DX_GROUP: 1 = ASD, 2 = TD (typically developing)
    asd_df = pheno[pheno['DX_GROUP'] == 1]
    td_df = pheno[pheno['DX_GROUP'] == 2]
    asd_file_ids = asd_df['FILE_ID'].values
    td_file_ids = td_df['FILE_ID'].values
    print(f"  Total: {len(asd_file_ids)} ASD, {len(td_file_ids)} TD subjects")

    # Step 2: Download correlation matrices (sample from each group)
    print()
    print("[2/4] Downloading correlation matrices (CC200 parcellation)...")

    max_per_group = 80  # larger sample for power

    asd_matrices = []
    asd_downloaded = 0
    for fid in asd_file_ids:
        if asd_downloaded >= max_per_group:
            break
        mat = fetch_correlation_matrix(fid)
        if mat is not None and mat.shape[0] >= 8:
            asd_matrices.append((fid, mat))
            asd_downloaded += 1
            if asd_downloaded % 10 == 0:
                print(f"  ASD: {asd_downloaded}/{max_per_group}")

    td_matrices = []
    td_downloaded = 0
    for fid in td_file_ids:
        if td_downloaded >= max_per_group:
            break
        mat = fetch_correlation_matrix(fid)
        if mat is not None and mat.shape[0] >= 8:
            td_matrices.append((fid, mat))
            td_downloaded += 1
            if td_downloaded % 10 == 0:
                print(f"  TD: {td_downloaded}/{max_per_group}")

    print(f"  Downloaded: {len(asd_matrices)} ASD, {len(td_matrices)} TD")

    if len(asd_matrices) < 5 or len(td_matrices) < 5:
        print("  Not enough data. Exiting.")
        return

    # Step 3: Compute G₂ invariants
    print()
    print("[3/4] Computing G₂ invariants per subject...")

    asd_results = []
    for fid, mat in asd_matrices:
        inv = compute_g2_invariants(mat)
        if inv is not None:
            inv['file_id'] = str(fid)
            inv['group'] = 'ASD'
            asd_results.append(inv)

    td_results = []
    for fid, mat in td_matrices:
        inv = compute_g2_invariants(mat)
        if inv is not None:
            inv['file_id'] = str(fid)
            inv['group'] = 'TD'
            td_results.append(inv)

    print(f"  Computed: {len(asd_results)} ASD, {len(td_results)} TD")

    if len(asd_results) < 5 or len(td_results) < 5:
        print("  Not enough valid results. Exiting.")
        return

    # Step 4: Statistical comparison
    print()
    print("[4/4] Permutation test (10,000 permutations)...")

    stats = permutation_test(asd_results, td_results, n_perms=10000, seed=42)

    # Report
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()
    print(f"Sample sizes: ASD n={stats['n_a']}, TD n={stats['n_b']}")
    print()
    print("Fano dominance ratio (triple product energy, Fano / total):")
    print(f"  ASD:  mean = {stats['mean_phi_a']:.6f}  (sd = {stats['std_phi_a']:.6f})")
    print(f"  TD:   mean = {stats['mean_phi_b']:.6f}  (sd = {stats['std_phi_b']:.6f})")
    print(f"  Δ    = {stats['obs_diff_phi']:.6f}")
    print(f"  Cohen's d = {stats['d_phi']:.3f}")
    print(f"  p-value (permutation) = {stats['p_phi']:.4f}")
    print()
    print("SU(3) branching quality Q (coordinate-based spatial overlap):")
    print(f"  ASD:  mean = {stats['mean_q_a']:.6f}  (sd = {stats['std_q_a']:.6f})")
    print(f"  TD:   mean = {stats['mean_q_b']:.6f}  (sd = {stats['std_q_b']:.6f})")
    print(f"  ΔQ   = {stats['obs_diff_q']:.6f}")
    print(f"  Cohen's d = {stats['d_q']:.3f}")
    print(f"  p-value (permutation) = {stats['p_q']:.4f}")
    print()

    # Interpretation
    print("-" * 70)
    print("INTERPRETATION")
    print("-" * 70)

    def interpret_d(d):
        ad = abs(d)
        if ad < 0.2: return "negligible"
        if ad < 0.5: return "small"
        if ad < 0.8: return "medium"
        return "large"

    def interpret_p(p):
        if p < 0.001: return "***"
        if p < 0.01: return "**"
        if p < 0.05: return "*"
        if p < 0.1: return "†"
        return "n.s."

    print(f"  |φ| effect: {interpret_d(stats['d_phi'])} ({interpret_p(stats['p_phi'])})")
    print(f"  Q effect:   {interpret_d(stats['d_q'])} ({interpret_p(stats['p_q'])})")
    print()

    if stats['p_phi'] < 0.05 or stats['p_q'] < 0.05:
        print("  → Octonion algebraic structure differs between ASD and TD.")
        if stats['d_phi'] > 0:
            print("  → ASD: HIGHER Fano dominance (eigenmodes more octonionic).")
        else:
            print("  → ASD: LOWER Fano dominance (eigenmodes less octonionic).")
        if stats['p_q'] < 0.05:
            if stats['d_q'] > 0:
                print("  → ASD: STRONGER 3⊕3̄⊕1 block structure.")
            else:
                print("  → ASD: WEAKER 3⊕3̄⊕1 block structure.")
    else:
        print("  → No significant difference in octonion structure detected.")
        print("    (Consider: finer parcellation, more subjects, or")
        print("     different eigenvector-to-basis assignment.)")

    # Save results
    output = {
        'experiment': 'G2_bridge_ABIDE_I',
        'parcellation': 'CC200',
        'pipeline': 'cpac',
        'strategy': 'nofilt_noglobal',
        'stats': stats,
        'individual_results': {
            'ASD': asd_results,
            'TD': td_results,
        },
    }

    outpath = "artifacts/research/g2_abide_results.json"
    with open(outpath, 'w') as f:
        json.dump(output, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)
    print(f"\n  Results saved to {outpath}")


if __name__ == "__main__":
    main()
