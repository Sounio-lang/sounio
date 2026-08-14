#!/usr/bin/env python3
"""
Sedenion-valued Path Signature with [2,1]-hook level-3 term.

THE CONSTRUCTION
================
A path signature is the iterated-integral expansion of a path.
For a path X: [0,T] → ℝ^d, the signature is:

  S(X) = 1 + ∫dX + ∫∫dX⊗dX + ∫∫∫dX⊗dX⊗dX + ...

The level-3 term captures temporal structure: what happened BEFORE matters.
The Massey product lives at level 3: μ_k = ∫ A_ij dX^k.

We replace the tensor product ⊗ with the SEDENION product:
  S_sed(X) = 1 + ∫dX + ∫∫dX⊗_sed dX + ∫∫∫dX⊗_sed dX⊗_sed dX + ...

At level 3, the sedenion associator [a,b,c] ≠ 0, and its [2,1] projection
captures mixed-symmetry temporal structure — the Massey signal.

HYPOTHESIS: The [2,1]-hook of the sedenion path signature's level-3 term
detects Borromean structure (99.8% in repo's Massey test) where:
- Octonion associator = 48.9% (chance)
- Static sedenion bracket = 0 (this experiment)
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, '/workspace/sounio/docs/gpu')

from borromean_signature import make_path, invariants, assoc_norm2
from hook21_bracket import sed_mul, sed_assoc, _SED_S, _SED_I

# ============================================================
# SEDENION PATH SIGNATURE
# ============================================================

def pack_sedenion(x_3d):
    """Pack a 3D increment into sedenion (indices 1,2,4 from Fano plane)."""
    s = np.zeros(16)
    s[1] = x_3d[0]
    s[2] = x_3d[1]
    s[4] = x_3d[2]
    return s


def sedenion_path_signature(X):
    """Compute the sedenion-valued path signature of X: [0,T] → ℝ³.
    
    Returns level-1, level-2, and level-3 terms.
    
    Level 1: S₁ = Σ dX_t                    (increment sum)
    Level 2: S₂ = Σ dX_s ⊗ dX_t  (s<t)     (sedenion path area)
    Level 3: S₃ = Σ dX_r ⊗ dX_s ⊗ dX_t     (triple iterated integral)
    
    The level-3 term carries the Massey/temporal structure.
    """
    dX = np.diff(X, axis=0)  # (T-1, 3)
    T = dX.shape[0]
    
    # Pack increments as sedenions
    dX_sed = np.array([pack_sedenion(dX[t]) for t in range(T)])
    
    # Level 1: sum of increments
    S1 = dX_sed.sum(axis=0)  # (16,)
    
    # Level 2: left-point Riemann sum of products
    # S₂ = Σ_{t} P_t ⊗ dX_t  where P = running position
    S2 = np.zeros(16)
    P = np.zeros(16)  # running position
    for t in range(T):
        S2 += sed_mul(P, dX_sed[t])
        P += dX_sed[t]
    
    # Level 3: triple iterated integral
    # S₃ = Σ_t S₂_running ⊗ dX_t
    # This accumulates: at each step, multiply the accumulated
    # level-2 area by the current increment
    S3 = np.zeros(16)
    P2 = np.zeros(16)  # running level-2
    P1 = np.zeros(16)  # running level-1 (position)
    
    for t in range(T):
        # Level-3: accumulate P2 ⊗ dX_t (left-point)
        S3 += sed_mul(P2, dX_sed[t])
        # Update running level-2: P1 ⊗ dX_t
        P2 += sed_mul(P1, dX_sed[t])
        # Update position
        P1 += dX_sed[t]
    
    return S1, S2, S3


def path_signature_features(X):
    """Extract [2,1]-hook features from sedenion path signature.
    
    The key insight: the level-3 term S₃ is a sedenion that carries
    the temporal/triple structure. Its [2,1]-hook projection captures
    mixed-symmetry that the Massey invariant detects.
    """
    S1, S2, S3 = sedenion_path_signature(X)
    
    # S₃ itself carries the triple structure
    # Compute its norm (overall level-3 magnitude)
    s3_norm = np.linalg.norm(S3)
    
    # Decompose S₃ into its components via the Fano plane structure
    # The "associator part" of S₃: how much does ordering matter?
    # [S₁, S₂/‖S₂‖, dX] — does the order of S₁, S₂, increment matter?
    
    # Direct [2,1] hook on the three levels of the signature
    # Use S1, S2 (normalized), and the average increment as the triple
    dX = np.diff(X, axis=0)
    mean_inc = pack_sedenion(dX.mean(axis=0))
    
    S2_norm = S2.copy()
    n2 = np.linalg.norm(S2_norm)
    if n2 > 1e-10:
        S2_norm /= n2
    
    # [2,1]-hook of (S1, S2, mean_inc)
    from itertools import permutations
    triples = list(permutations([S1, S2_norm, mean_inc]))
    signs = [1, -1, -1, 1, 1, -1]
    T_perms = [sed_assoc(*p) for p in triples]
    alt = sum(s * t for s, t in zip(signs, T_perms)) / 6.0
    sym = sum(T_perms) / 6.0
    hook = sed_assoc(S1, S2_norm, mean_inc) - alt - sym
    
    # Also: the associator [S1, S2_norm, mean_inc]
    assoc = sed_assoc(S1, S2_norm, mean_inc)
    
    # Level-3 associator: how much does (S2 · S1) · mean_inc ≠ S2 · (S1 · mean_inc)?
    left = sed_mul(sed_mul(S2_norm, S1), mean_inc)
    right = sed_mul(S2_norm, sed_mul(S1, mean_inc))
    l3_assoc = np.linalg.norm(left - right)
    
    return {
        's3_norm': s3_norm,
        'hook_norm': np.linalg.norm(hook),
        'assoc_norm': np.linalg.norm(assoc),
        'l3_assoc': l3_assoc,
        's2_norm': np.linalg.norm(S2),
        's1_norm': np.linalg.norm(S1),
    }


def run(N=2000, seed=20260719):
    rng = np.random.default_rng(seed)
    
    print("=" * 72)
    print("SEDENION PATH SIGNATURE WITH [2,1]-HOOK ON BORROMEAN")
    print("=" * 72)
    
    # Generate paths
    print(f"Generating {N} Fourier paths...")
    paths = [make_path(rng) for _ in range(N)]
    
    # Path signature invariants (repo's proven method)
    print("Computing path signatures (repo's Massey invariants)...")
    invs = [invariants(X) for X in paths]
    areas = np.array([inv[1] for inv in invs])
    mu = np.array([inv[2] for inv in invs])
    amag = np.abs(areas).max(1)
    mutot = mu.sum(1)
    
    # Borromean slice
    slice_idx = np.argsort(amag)[:N // 3]
    mu_slice = mutot[slice_idx]
    median_mu = np.median(mu_slice)
    labels_slice = (mu_slice > median_mu).astype(int)
    
    print(f"Borromean slice: {len(slice_idx)} paths")
    print(f"  Massey μ range: [{mu_slice.min():.2f}, {mu_slice.max():.2f}]")
    
    # Extract sedenion path signature features
    print("Computing sedenion path signature features...")
    
    # Compute for ALL paths (for the full dataset test too)
    all_feats = [path_signature_features(p) for p in paths]
    
    # Slice features
    slice_feats = [all_feats[i] for i in slice_idx]
    
    # Results on Borromean slice
    print(f"\n{'='*72}")
    print("BORROMEAN SLICE RESULTS")
    print(f"{'='*72}")
    print(f"{'Feature':<25} {'d':<10} {'AUROC':<8}")
    print("-" * 43)
    
    feat_names = ['s3_norm', 'hook_norm', 'assoc_norm', 'l3_assoc', 's2_norm', 's1_norm']
    
    for fname in feat_names:
        vals = np.array([f[fname] for f in slice_feats])
        pos = vals[labels_slice == 1]
        neg = vals[labels_slice == 0]
        
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
        
        star = "⚡" if abs(d) > 0.3 else ("*" if abs(d) > 0.1 else "")
        print(f"  {fname:<23} {d:+.3f}    {au:.3f}  {star}")
    
    # Full dataset
    labels_full = (mutot > np.median(mutot)).astype(int)
    print(f"\n{'='*72}")
    print("FULL DATASET RESULTS")
    print(f"{'='*72}")
    print(f"{'Feature':<25} {'d':<10} {'AUROC':<8}")
    print("-" * 43)
    
    for fname in feat_names:
        vals = np.array([f[fname] for f in all_feats])
        pos = vals[labels_full == 1]
        neg = vals[labels_full == 0]
        
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
        
        star = "⚡" if abs(d) > 0.3 else ("*" if abs(d) > 0.1 else "")
        print(f"  {fname:<23} {d:+.3f}    {au:.3f}  {star}")
    
    print(f"\n{'='*72}")
    print("COMPARISON WITH REPO RESULTS")
    print(f"{'='*72}")
    print("  Massey μ₃ (repo):     d=~∞   AUROC=~1.000 (defines label)")
    print("  Oct assoc (repo):      d≈0    AUROC=0.489 (blind)")
    print("  Levy areas (repo):     d≈0    AUROC=0.484 (blind by construction)")
    print("  MLP raw path (repo):   d≈0.1  AUROC=0.561")
    
    print("\n" + "=" * 72)


if __name__ == '__main__':
    run()
