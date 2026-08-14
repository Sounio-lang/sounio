#!/usr/bin/env python3
"""
The [2,1]-hook bracket: the mixed-symmetry non-associative invariant.

THE MATHEMATICAL GAP
====================
The octonion associator [a,b,c] = (a·b)·c - a·(b·c) is the fully
alternating G₂ 3-form (Λ³ component). The repo proved it is ORTHOGONAL
to crossing/Massey obstructions which live in the [2,1] hook.

The tensor cube V^⊗3 of a vector space decomposes into irreducibles:
    V^⊗3 = Λ³ ⊕ [2,1] ⊕ Sym³

where:
- Λ³: fully antisymmetric (the associator lives here)
- [2,1]: mixed symmetry (the Massey product lives here)
- Sym³: fully symmetric

The [2,1] hook is the Schur functor S_{[2,1]}(V). It captures structure
that is "antisymmetric in two indices, then paired with a third" —
exactly the Massey product μ_k = ∫ A_ij dX^k.

WHAT WE CONSTRUCT
=================
A computable [2,1]-bracket operation on octonion states:

  μ(a, b, c) = 2·[a,b,c] - [a,c,b] - [b,a,c]

This projects out the Λ³ (alternating) component and isolates the
mixed-symmetry residual. In representation theory terms:

  Standard Young symmetrizer for [2,1]:
    c_{[2,1]} = (e + (12))·(e - (123))
  
  Applied to the associator tensor T_{ijk} = [e_i, e_j, e_k]:
    μ_{ijk} = T_{ijk} + T_{jik} - T_{ikj} - T_{kij}

On octonions, the associator [a,b,c] = (ab)c - a(bc) is alternating,
so the [2,1] projection of it is IDENTICALLY ZERO — that's the theorem.
But on SEDENIONS, [a,b,c] is NOT alternating, so the [2,1] projection
is NONZERO. This is the sedenion's unique contribution.

EXPERIMENT
==========
Test: does the [2,1]-bracket of sedenion states detect Borromean
(crossing) structure where both the octonion associator AND the sedenion
associator fail?

We compute μ on:
1. Synthetic Borromean paths (from repo's borromean_signature.py)
2. RNA pseudoknots (crossing bracket sequences)
3. N-back EEG (cognitive crossing)
"""

import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(__file__))

# ============================================================
# SEDENION ARITHMETIC (from pseudoknot_experiment.py)
# ============================================================

def _build_sed_sign():
    """Build 16×16 sedenion tables from Cayley-Dickson doubling (corrected).

    CD formula: (a,b)(c,d) = (ac - conj(d)b, da + b conj(c))
    """
    dim = 16
    sign = np.ones((dim, dim))
    idx = np.zeros((dim, dim), dtype=int)

    # Octonion block
    FANO = [(1,2,4),(2,3,5),(3,4,6),(4,5,7),(5,6,1),(6,7,2),(7,1,3)]
    for i in range(8):
        sign[0,i]=1; idx[0,i]=i; sign[i,0]=1; idx[i,0]=i
        sign[i,i]=-1; idx[i,i]=0
    for a,b,c in FANO:
        for p,q,r in [(a,b,c),(b,c,a),(c,a,b)]:
            sign[p,q]=1; idx[p,q]=r
        for p,q,r in [(b,a,c),(c,b,a),(a,c,b)]:
            sign[p,q]=-1; idx[p,q]=r

    OCT_S = sign[:8,:8].copy()
    OCT_I = idx[:8,:8].copy()

    # CD cross-blocks
    for a in range(8):
        for b in range(8):
            if b == 0:
                sign[8+a, b] = 1;  idx[8+a, b] = 8 + a
            else:
                sign[8+a, b] = -OCT_S[a, b]
                idx[8+a, b] = 8 + int(OCT_I[a, b])
            if a == 0:
                sign[a, 8+b] = 1;  idx[a, 8+b] = 8 + b
            else:
                sign[a, 8+b] = OCT_S[b, a]
                idx[a, 8+b] = 8 + int(OCT_I[b, a])
            if b == 0:
                sign[8+a, 8+b] = -1; idx[8+a, 8+b] = a
            else:
                sign[8+a, 8+b] = OCT_S[b, a]
                idx[8+a, 8+b] = int(OCT_I[b, a])

    for i in range(16):
        sign[0,i]=1; idx[0,i]=i; sign[i,0]=1; idx[i,0]=i
    for i in range(1,16):
        sign[i,i]=-1; idx[i,i]=0
    return sign, idx

_SED_S, _SED_I = _build_sed_sign()

def sed_mul(a, b):
    out = np.zeros(16)
    for i in range(16):
        for j in range(16):
            out[int(_SED_I[i,j])] += _SED_S[i,j] * a[i] * b[j]
    return out

def sed_assoc(a, b, c):
    """Sedenion associator [a,b,c] = (a·b)·c - a·(b·c)."""
    return sed_mul(sed_mul(a, b), c) - sed_mul(a, sed_mul(b, c))


# ============================================================
# THE [2,1]-HOOK BRACKET
# ============================================================

def hook_21_bracket(a, b, c, use_sedenion=True):
    """The [2,1]-hook bracket: mixed-symmetry projection of the associator.
    
    Decomposes V^⊗3 into irreducibles: Λ³ ⊕ [2,1] ⊕ Sym³
    Isolates the [2,1] (mixed symmetry) component.
    
    Formula: [2,1](T) = T - Λ³(T) - Sym³(T)
    
    where:
      Λ³(T) = (1/6) Σ_σ sgn(σ) · T_σ   (alternating projection)
      Sym³(T) = (1/6) Σ_σ T_σ           (symmetric projection)
    
    On octonions: T is alternating → Λ³(T)=T, Sym³(T)=0 → [2,1](T)=0
    On sedenions: T is NOT alternating → [2,1](T) ≠ 0
    """
    mul = sed_mul if use_sedenion else None
    
    def assoc(x, y, z):
        return mul(mul(x, y), z) - mul(x, mul(y, z))
    
    # All 6 permutations of (a,b,c) with their signs
    perms = [
        ((a, b, c), +1),
        ((a, c, b), -1),
        ((b, a, c), -1),
        ((b, c, a), +1),
        ((c, a, b), +1),
        ((c, b, a), -1),
    ]
    
    # Compute associator for all permutations
    assoc_perms = [assoc(*p) for p, _ in perms]
    signs = [s for _, s in perms]
    
    # Λ³ = (1/6) Σ sgn(σ) T_σ
    alt_part = sum(s * t for s, t in zip(signs, assoc_perms)) / 6.0
    
    # Sym³ = (1/6) Σ T_σ
    sym_part = sum(assoc_perms) / 6.0
    
    # [2,1] = T - Λ³ - Sym³
    T = assoc(a, b, c)  # the original
    return T - alt_part - sym_part


def hook_21_norm(a, b, c, use_sedenion=True):
    """Norm of the [2,1]-hook bracket."""
    return np.linalg.norm(hook_21_bracket(a, b, c, use_sedenion))


# ============================================================
# VERIFICATION
# ============================================================

def verify_hook_21():
    """Verify the mathematical properties of the [2,1] bracket.

    Uses the CORRECT projector T - Λ³(T) - Sym³(T) via hook_21_bracket().
    On octonions the associator is alternating → [2,1] ≡ 0 (identity, not prediction).
    On sedenions the associator is NOT alternating → [2,1] ≠ 0.
    """
    print("=" * 60)
    print("[2,1]-HOOK BRACKET VERIFICATION (corrected projector)")
    print("=" * 60)

    rng = np.random.default_rng(42)

    # Test 1: On OCTONIONS, [2,1] should be ~0
    # Octonions embed in sedenions as the 0..7 subalgebra, so we use sed_mul
    print("\n1. Octonion [2,1] bracket (should be ~0 — identity, not prediction):")
    for trial in range(5):
        a = np.zeros(16); a[:8] = rng.normal(0, 1, 8)
        b = np.zeros(16); b[:8] = rng.normal(0, 1, 8)
        c = np.zeros(16); c[:8] = rng.normal(0, 1, 8)

        hook = hook_21_bracket(a, b, c, use_sedenion=True)
        assoc_norm = np.linalg.norm(sed_assoc(a, b, c))
        print(f"  trial {trial}: ‖[2,1]‖ = {np.linalg.norm(hook):.2e} "
              f"(assoc ‖{assoc_norm:.4f}‖)")

    # Test 2: On SEDENIONS, [2,1] should be NONZERO
    print("\n2. Sedenion [2,1] bracket (should be > 0):")
    for trial in range(5):
        a = rng.normal(0, 1, 16)
        b = rng.normal(0, 1, 16)
        c = rng.normal(0, 1, 16)

        hook = hook_21_bracket(a, b, c, use_sedenion=True)
        assoc_norm = np.linalg.norm(sed_assoc(a, b, c))
        print(f"  trial {trial}: ‖[2,1]‖ = {np.linalg.norm(hook):.4f} "
              f"(assoc ‖{assoc_norm:.4f}‖)")

    # Test 3: [a,a,b] — full associator vs [2,1] projection
    print("\n3. [a,a,b] sedenion (non-alternativity):")
    for trial in range(3):
        a = rng.normal(0, 1, 16)
        b = rng.normal(0, 1, 16)

        full_assoc = np.linalg.norm(sed_assoc(a, a, b))
        hook_proj = np.linalg.norm(hook_21_bracket(a, a, b, use_sedenion=True))
        print(f"  ‖[a,a,b]_assoc‖ = {full_assoc:.4f}, "
              f"‖[a,a,b]_[2,1]‖ = {hook_proj:.4f}")

    print("\n" + "=" * 60)


# ============================================================
# HOOK-21 ON SYNTHETIC BORROMEAN PATHS
# ============================================================

def generate_borromean_path(T=1000, rng=None):
    """Generate a Borromean-type 3-component path.
    
    Three paths that are pairwise unlinked but globally linked.
    The [2,1] hook should detect the triple linking.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Three component paths (like 3D coordinates over time)
    t = np.linspace(0, 4 * np.pi, T)
    
    # Borromean-like: pairwise linking ≈ 0, but triple linking ≠ 0
    x = np.cos(t) + 0.1 * rng.normal(0, 1, T)
    y = np.sin(t) + 0.1 * rng.normal(0, 1, T)
    z = np.cos(2*t) * np.sin(t) + 0.1 * rng.normal(0, 1, T)
    
    return np.stack([x, y, z], axis=1)  # (T, 3)


def generate_unlinked_path(T=1000, rng=None):
    """Generate three independent (unlinked) paths."""
    if rng is None:
        rng = np.random.default_rng()
    return rng.normal(0, 1, (T, 3))


def compute_hook_features(path, use_sedenion=True):
    """Compute [2,1] hook bracket features from a 3-component path.
    
    Pack 3 components into sedenion (16-dim), compute hook bracket
    over consecutive triples.
    """
    T = path.shape[0]
    
    # Pack into 16-dim sedenion (3 components + zeros)
    def pack(x):
        s = np.zeros(16)
        s[:3] = x
        s[3] = 1.0  # bias
        return s
    
    hook_values = []
    assoc_values = []
    
    for t in range(2, T):
        a = pack(path[t-2])
        b = pack(path[t-1])
        c = pack(path[t])
        
        h = hook_21_norm(a, b, c, use_sedenion=use_sedenion)
        hook_values.append(h)
        
        a_assoc = np.linalg.norm(sed_assoc(a, b, c))
        assoc_values.append(a_assoc)
    
    return {
        'hook_median': np.median(hook_values),
        'hook_mean': np.mean(hook_values),
        'hook_std': np.std(hook_values),
        'assoc_median': np.median(assoc_values),
        'assoc_mean': np.mean(assoc_values),
    }


def run_borromean_experiment(n_samples=100, seed=42):
    """Test: does [2,1] hook discriminate Borromean from unlinked?"""
    rng = np.random.default_rng(seed)
    
    print("\n" + "=" * 60)
    print("[2,1]-HOOK ON BORROMEAN vs UNLINKED PATHS")
    print("=" * 60)
    
    # Generate
    borromean = [generate_borromean_path(500, rng) for _ in range(n_samples)]
    unlinked = [generate_unlinked_path(500, rng) for _ in range(n_samples)]
    
    # Compute features
    print("Computing sedenion [2,1] hook features...")
    
    hook_borr = [compute_hook_features(p) for p in borromean]
    hook_unlk = [compute_hook_features(p) for p in unlinked]
    
    # Compare
    hb = np.array([h['hook_mean'] for h in hook_borr])
    hu = np.array([h['hook_mean'] for h in hook_unlk])
    ab = np.array([h['assoc_mean'] for h in hook_borr])
    au = np.array([h['assoc_mean'] for h in hook_unlk])
    
    print(f"\nResults (n={n_samples} per class):")
    print(f"  Sedenion associator:")
    print(f"    Borromean: {ab.mean():.4f} ± {ab.std():.4f}")
    print(f"    Unlinked:  {au.mean():.4f} ± {au.std():.4f}")
    print(f"    Cohen's d: {(ab.mean() - au.mean()) / np.sqrt((ab.std()**2 + au.std()**2)/2):.3f}")
    
    print(f"  [2,1]-hook bracket:")
    print(f"    Borromean: {hb.mean():.4f} ± {hb.std():.4f}")
    print(f"    Unlinked:  {hu.mean():.4f} ± {hu.std():.4f}")
    d_hook = (hb.mean() - hu.mean()) / np.sqrt((hb.std()**2 + hu.std()**2)/2)
    print(f"    Cohen's d: {d_hook:.3f}")
    
    if abs(d_hook) > 0.2:
        print(f"\n  ⚡ [2,1]-hook discriminates Borromean from unlinked!")
    else:
        print(f"\n  [2,1]-hook does NOT discriminate (d={d_hook:.3f})")
    
    # AUROC
    from numpy import sqrt
    def auroc(scores, labels):
        n_pos = labels.sum(); n_neg = len(labels) - n_pos
        if n_pos == 0 or n_neg == 0: return 0.5
        order = np.argsort(scores)
        ranks = np.zeros(len(scores))
        ranks[order] = np.arange(1, len(scores) + 1)
        for s in np.unique(scores):
            mask = scores == s
            ranks[mask] = ranks[mask].mean()
        return (ranks[labels.astype(bool)].sum() - n_pos*(n_pos+1)/2) / (n_pos*n_neg)
    
    labels = np.concatenate([np.ones(n_samples), np.zeros(n_samples)])
    
    au_assoc = auroc(np.concatenate([ab, au]), labels)
    au_hook = auroc(np.concatenate([hb, hu]), labels)
    
    print(f"\n  AUROC (associator): {au_assoc:.3f}")
    print(f"  AUROC ([2,1] hook): {au_hook:.3f}")
    
    return {'auroc_assoc': au_assoc, 'auroc_hook': au_hook,
            'd_assoc': float((ab.mean()-au.mean())/sqrt((ab.std()**2+au.std()**2)/2)),
            'd_hook': float(d_hook)}


if __name__ == '__main__':
    verify_hook_21()
    results = run_borromean_experiment()
