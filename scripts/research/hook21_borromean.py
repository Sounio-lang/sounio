#!/usr/bin/env python3
"""
[2,1]-hook bracket integrated with repo's Borromean paths.

Uses the repo's proven Borromean path generator + path signature machinery,
but replaces the feature extraction with:
1. Sedenion associator (existing, Λ³ component)
2. [2,1]-hook bracket (new, mixed-symmetry component)

The repo proved: octonion associator = 48.9% (chance) on Borromean.
                  Massey triple invariant = 99.8% (defines label).

Question: does the [2,1]-hook bracket of sedenion states detect
the Borromean structure where the octonion associator is blind?
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

# Import the repo's proven Borromean machinery
sys.path.insert(0, '/workspace/sounio/docs/gpu')
from borromean_signature import make_path, invariants, omul, assoc_norm2

# Import our sedenion arithmetic
from hook21_bracket import sed_mul, sed_assoc

# ============================================================
# [2,1]-HOOK ON BORROMEAN PATHS
# ============================================================

def hook21_norm(a, b, c):
    """Correct [2,1] projection norm on sedenions."""
    from itertools import permutations
    perms = list(permutations([a, b, c]))
    signs = [1, -1, -1, 1, 1, -1]
    
    T_perms = [sed_assoc(*p) for p in perms]
    alt = sum(s * t for s, t in zip(signs, T_perms)) / 6.0
    sym = sum(T_perms) / 6.0
    T = T_perms[0]  # [a,b,c]
    hook = T - alt - sym
    return np.linalg.norm(hook)


def borromean_features(X):
    """Extract features from a 3-component path using the repo's signature + our hook.
    
    X: (T, 3) path
    Returns: dict with associator (Λ³) and [2,1]-hook features
    """
    dX = np.diff(X, axis=0)
    T3 = dX.shape[0] // 3
    
    # Three segment increment vectors (the repo's approach)
    seg = np.zeros((3, 3))
    for k in range(3):
        seg[k] = dX[k*T3:(k+1)*T3].sum(0)
    
    # Pack into octonion (repo's approach: imaginary components e1, e2, e4)
    e_oct = np.zeros((3, 8))
    for k in range(3):
        e_oct[k, 1] = seg[k, 0]
        e_oct[k, 2] = seg[k, 1]
        e_oct[k, 4] = seg[k, 2]
    
    # Octonion associator (the bridge feature from repo: should be ~0 on Borromean)
    oct_assoc = np.sqrt(assoc_norm2(e_oct[0], e_oct[1], e_oct[2]))
    
    # Pack into sedenion (our approach: use full 16-dim)
    e_sed = np.zeros((3, 16))
    for k in range(3):
        e_sed[k, 1] = seg[k, 0]
        e_sed[k, 2] = seg[k, 1]
        e_sed[k, 4] = seg[k, 2]
    
    # Sedenion associator norm
    sed_assoc_val = np.linalg.norm(sed_assoc(e_sed[0], e_sed[1], e_sed[2]))
    
    # [2,1]-hook bracket norm
    hook_val = hook21_norm(e_sed[0], e_sed[1], e_sed[2])
    
    # Also compute on the RUNNING trajectory (not just segments)
    # Process path through O-SSM and compute features over the trajectory
    h = np.zeros(16)
    A = np.ones(16) * 0.3
    
    hook_traj = []
    assoc_traj = []
    
    for t in range(2, dX.shape[0]):
        x = np.zeros(16)
        x[:3] = dX[t]
        x[3] = 1.0
        
        h_new = np.tanh(sed_mul(A, h) + x)
        
        if t >= 4:
            a = h.copy()  # save before overwrite
            # Need 3 consecutive states
            # (simplified: use h, h_prev, h_new)
            pass
        
        h = h_new
    
    return {
        'oct_assoc': oct_assoc,
        'sed_assoc': sed_assoc_val,
        'hook21': hook_val,
        'seg_norms': [np.linalg.norm(seg[k]) for k in range(3)],
    }


def run_borromean_hook(N=2000, seed=20260719):
    """Test [2,1]-hook on the repo's proven Borromean paths."""
    rng = np.random.default_rng(seed)
    
    print("=" * 72)
    print("[2,1]-HOOK ON BORROMEAN PATHS (repo's proven setup)")
    print("=" * 72)
    
    # Generate paths (same as repo)
    print(f"Generating {N} Fourier paths...")
    paths = [make_path(rng) for _ in range(N)]
    
    # Compute invariants (repo's approach for slicing)
    print("Computing path signatures...")
    invs = [invariants(X) for X in paths]
    areas = np.array([inv[1] for inv in invs])
    mu = np.array([inv[2] for inv in invs])
    amag = np.abs(areas).max(1)
    mutot = mu.sum(1)
    
    # Borromean slice: pairwise-trivial (small areas)
    slice_idx = np.argsort(amag)[:N // 3]
    print(f"Borromean slice: {len(slice_idx)} of {N} (max|area|={amag[slice_idx].max():.3f})")
    
    # Label by triple invariant (Massey)
    mu_slice = mutot[slice_idx]
    median_mu = np.median(mu_slice)
    labels = (mu_slice > median_mu).astype(int)
    
    # Extract features on the slice
    print("Extracting features on Borromean slice...")
    features = []
    for i, idx in enumerate(slice_idx):
        feat = borromean_features(paths[idx])
        features.append(feat)
    
    # Compare features by label
    oct_assoc = np.array([f['oct_assoc'] for f in features])
    sed_assoc = np.array([f['sed_assoc'] for f in features])
    hook21 = np.array([f['hook21'] for f in features])
    
    print(f"\nResults on Borromean slice (n={len(slice_idx)}):")
    print(f"{'Feature':<20} {'Borromean+':<18} {'Borromean-':<18} {'Cohen d':<10} {'AUROC':<8}")
    print("-" * 74)
    
    for name, vals in [('Oct assoc (Λ³)', oct_assoc), 
                       ('Sed assoc', sed_assoc),
                       ('[2,1]-hook', hook21)]:
        pos = vals[labels == 1]
        neg = vals[labels == 0]
        
        if len(pos) < 5 or len(neg) < 5:
            print(f"{name:<20} (too few samples)")
            continue
        
        m_pos, m_neg = np.mean(pos), np.mean(neg)
        s_pos, s_neg = np.std(pos), np.std(neg)
        pooled = np.sqrt((s_pos**2 + s_neg**2) / 2)
        d = (m_pos - m_neg) / max(pooled, 1e-8)
        
        # AUROC
        all_scores = np.concatenate([pos, neg])
        all_labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        order = np.argsort(all_scores)
        ranks = np.zeros(len(all_scores))
        ranks[order] = np.arange(1, len(all_scores) + 1)
        for v in np.unique(all_scores):
            mask = all_scores == v
            ranks[mask] = ranks[mask].mean()
        n_pos = len(pos)
        n_neg = len(neg)
        au = (ranks[all_labels == 1].sum() - n_pos*(n_pos+1)/2) / (n_pos*n_neg)
        
        print(f"{name:<20} {m_pos:.4f}±{s_pos:.4f}   {m_neg:.4f}±{s_neg:.4f}   {d:+.3f}    {au:.3f}")
    
    # Also check on the FULL dataset (not just Borromean slice)
    print(f"\nFull dataset (n={N}):")
    all_features = [borromean_features(p) for p in paths]
    all_oct = np.array([f['oct_assoc'] for f in all_features])
    all_sed = np.array([f['sed_assoc'] for f in all_features])
    all_hook = np.array([f['hook21'] for f in all_features])
    all_labels = (mutot > np.median(mutot)).astype(int)
    
    print(f"{'Feature':<20} {'Cohen d':<10} {'AUROC':<8}")
    print("-" * 38)
    for name, vals in [('Oct assoc (Λ³)', all_oct),
                       ('Sed assoc', all_sed),
                       ('[2,1]-hook', all_hook)]:
        pos = vals[all_labels == 1]
        neg = vals[all_labels == 0]
        pooled = np.sqrt((np.var(pos, ddof=1) + np.var(neg, ddof=1)) / 2)
        d = (np.mean(pos) - np.mean(neg)) / max(pooled, 1e-8)
        
        all_s = np.concatenate([pos, neg])
        all_l = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        order = np.argsort(all_s)
        ranks = np.zeros(len(all_s))
        ranks[order] = np.arange(1, len(all_s) + 1)
        for v in np.unique(all_s):
            mask = all_s == v
            ranks[mask] = ranks[mask].mean()
        au = (ranks[all_l == 1].sum() - len(pos)*(len(pos)+1)/2) / (len(pos)*len(neg))
        print(f"{name:<20} {d:+.3f}      {au:.3f}")
    
    print("\n" + "=" * 72)


if __name__ == '__main__':
    run_borromean_hook()
