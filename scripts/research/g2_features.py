#!/usr/bin/env python3
"""
G₂ structure constants as EEG features: 14 generators of the
automorphism group of the octonions.

THE IDEA
========
G₂ is the 14-dimensional exceptional Lie group — the automorphism
group of 𝕆. It preserves the Fano plane structure and the associator
3-form φ (the G₂-invariant Λ³).

The Lie algebra g₂ has 14 generators. Each acts on octonion states
as an infinitesimal automorphism. The structure constants c_{ij}^k
of g₂ encode how these generators compose.

CURRENT STATE: the O-SSM harness extracts 3 features (F1, F2, F3).
HYPOTHESIS: extracting 14 G₂-derived features gives richer psychiatric
signal — each generator probes a different "direction" of non-associative
structure in the EEG trajectory.

CONSTRUCTION
============
The 14 generators of g₂ in the 7-dimensional representation (imaginary
octonions) can be constructed from the derivations of 𝕆:

  D_{ij}(x) = [[e_i, e_j], x] - 3(e_i · [e_j, x] - e_j · [e_i, x])

for each pair (i,j) with i<j. There are C(7,2)=21 pairs, but with
dependencies this reduces to 14 independent derivations.

Each derivation D is a 7×7 antisymmetric matrix. Applied to an
octonion state trajectory, D extracts one "direction" of non-associative
variation.

14 G₂ FEATURES (replacing F1/F2/F3):
  For each derivation D_k (k=1..14):
    G_k = median_t ‖D_k(h_t)‖   (how much the trajectory moves in direction k)

These 14 features form a "G₂ spectrum" of the EEG trajectory.
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))


# ============================================================
# OCTONION BASICS
# ============================================================

_FANO = [(1,2,4),(2,3,5),(3,4,6),(4,5,7),(5,6,1),(6,7,2),(7,1,3)]

def _build_oct():
    sign = np.zeros((8,8)); idx = np.zeros((8,8), dtype=int)
    for i in range(8):
        sign[i,0]=1; idx[i,0]=i; sign[0,i]=1; idx[0,i]=i
        sign[i,i]=-1; idx[i,i]=0
    for a,b,c in _FANO:
        for p,q,r in [(a,b,c),(b,c,a),(c,a,b)]:
            sign[p,q]=1; idx[p,q]=r
        for p,q,r in [(b,a,c),(c,b,a),(a,c,b)]:
            sign[p,q]=-1; idx[p,q]=r
    return sign, idx

_S, _I = _build_oct()

def oct_mul(a, b):
    out = np.zeros(8)
    for i in range(8):
        for j in range(8):
            out[int(_I[i,j])] += _S[i,j] * a[i] * b[j]
    return out

def oct_mul_7(a, b):
    """Multiply two IMAGINARY octonions (7-dim, indices 1-7).
    The result has a real part (inner product) and imaginary part (cross product).
    Returns 7-dim imaginary part only.
    """
    full_a = np.zeros(8); full_a[1:] = a
    full_b = np.zeros(8); full_b[1:] = b
    result = oct_mul(full_a, full_b)
    return result[1:]  # imaginary part only


# ============================================================
# G₂ DERIVATIONS (14 GENERATORS)
# ============================================================

def build_g2_generators():
    """Build the 14 generators of g₂ as 7×7 antisymmetric matrices.

    Uses the derivation formula:
      D_{ij}(x) = [e_i·e_j, x] - 3(e_i·(e_j·x) - e_j·(e_i·x))

    where · is the imaginary octonion product (7-dim).

    We compute D_{ij} for all 21 pairs (i<j) and extract the 14
    independent ones via SVD rank selection.
    """
    # Build all 21 derivations
    derivs = []

    for i in range(7):
        for j in range(i+1, 7):
            # D_{ij} is a linear map on 7-dim imaginary octonions
            # Compute its matrix by applying to each basis vector
            D = np.zeros((7, 7))
            ei = np.zeros(7); ei[i] = 1.0
            ej = np.zeros(7); ej[j] = 1.0
            eiej = oct_mul_7(ei, ej)  # e_i · e_j

            for k in range(7):
                x = np.zeros(7); x[k] = 1.0

                # [e_i·e_j, x] = (e_i·e_j)·x - x·(e_i·e_j)
                lhs = oct_mul_7(eiej, x)
                rhs = oct_mul_7(x, eiej)
                comm = lhs - rhs

                # e_i · (e_j · x) - e_j · (e_i · x)
                ej_x = oct_mul_7(ej, x)
                ei_ejx = oct_mul_7(ei, ej_x)
                ei_x = oct_mul_7(ei, x)
                ej_eix = oct_mul_7(ej, ei_x)
                inner = ei_ejx - ej_eix

                # D_{ij}(x) = comm - 3*inner
                D[:, k] = comm - 3 * inner

            # Check if this derivation is nonzero and independent
            if np.linalg.norm(D) > 1e-10:
                derivs.append(D.flatten())

    # Stack and find 14 independent generators via SVD
    M = np.array(derivs)  # (21, 49)
    U, S, Vt = np.linalg.svd(M, full_matrices=False)

    # Take the top 14 singular vectors (should have exactly 14 nonzero singular values)
    n_gen = min(14, np.sum(S > 1e-8))
    generators = Vt[:n_gen].reshape(n_gen, 7, 7)

    return generators


# ============================================================
# G₂ FEATURES FROM TRAJECTORY
# ============================================================

def compute_g2_features(trajectory_7d, generators):
    """Compute 14 G₂ features from a 7-dim trajectory.

    trajectory_7d: (T, 7) — the imaginary octonion state over time
    generators: (14, 7, 7) — the G₂ generators

    Returns: (14,) — the G₂ spectrum
    """
    T = trajectory_7d.shape[0]
    n_gen = generators.shape[0]
    features = np.zeros(n_gen)

    for k in range(n_gen):
        D = generators[k]  # (7, 7)
        norms = []
        for t in range(T):
            d_h = D @ trajectory_7d[t]  # apply derivation
            norms.append(np.linalg.norm(d_h))
        features[k] = np.median(norms)

    return features


def compute_g2_from_full_trajectory(trajectory_8d, generators):
    """Compute G₂ features from the full 8-dim O-SSM trajectory.

    trajectory_8d: (T, 2, 8) — two octonion states per timestep
    Extracts the imaginary part (7-dim) and computes features.
    """
    # Extract imaginary parts from both states
    T = trajectory_8d.shape[0]
    
    # State 1 imaginary trajectory
    traj1 = trajectory_8d[:, 0, 1:]  # (T, 7)
    # State 2 imaginary trajectory
    traj2 = trajectory_8d[:, 1, 1:]  # (T, 7)
    
    f1 = compute_g2_features(traj1, generators)
    f2 = compute_g2_features(traj2, generators)
    
    # Concatenate: 28 features (14 per state)
    return np.concatenate([f1, f2])


# ============================================================
# VERIFICATION
# ============================================================

def verify_g2():
    """Verify G₂ generators have the right properties."""
    print("=" * 60)
    print("G₂ GENERATOR VERIFICATION")
    print("=" * 60)

    generators = build_g2_generators()
    print(f"\nNumber of generators: {generators.shape[0]}")
    print(f"Each generator: {generators.shape[1]}×{generators.shape[2]}")

    # Check antisymmetry
    for k in range(generators.shape[0]):
        D = generators[k]
        asym_error = np.linalg.norm(D + D.T) / max(np.linalg.norm(D), 1e-10)
        if asym_error > 0.01:
            print(f"  ⚠ Generator {k} is NOT antisymmetric (error={asym_error:.4f})")

    # Check rank (each should be rank 2 or higher)
    for k in range(generators.shape[0]):
        D = generators[k]
        sv = np.linalg.svd(D, compute_uv=False)
        rank = np.sum(sv > 1e-8)
        if k < 3 or rank < 2:
            print(f"  Generator {k}: rank={rank}, singular values={sv[:3].round(4)}")

    # Test: apply to random trajectory
    rng = np.random.default_rng(42)
    traj = rng.normal(0, 0.5, (100, 7))

    features = compute_g2_features(traj, generators)
    print(f"\nG₂ features for random trajectory (100 steps):")
    print(f"  Mean: {features.mean():.4f}")
    print(f"  Std: {features.std():.4f}")
    print(f"  Min: {features.min():.4f}  Max: {features.max():.4f}")
    print(f"  Range: {features.max() - features.min():.4f}")

    # Check: do different generators probe different directions?
    corr = np.corrcoef(features.reshape(-1, 1).T)
    print(f"\nFeature variance (should be diverse):")
    for k in range(min(14, len(features))):
        print(f"  G{k:2d}: {features[k]:.4f}")

    # Test: structured trajectory (oscillatory) vs random
    t = np.linspace(0, 4*np.pi, 100)
    osc_traj = np.stack([
        np.sin(t), np.cos(t), np.sin(2*t), np.cos(2*t),
        np.sin(3*t), np.cos(3*t), np.sin(t)*np.cos(t)
    ], axis=1) * 0.5

    osc_features = compute_g2_features(osc_traj, generators)
    print(f"\nG₂ features for oscillatory trajectory:")
    print(f"  Random:     {features.mean():.4f} ± {features.std():.4f}")
    print(f"  Oscillatory: {osc_features.mean():.4f} ± {osc_features.std():.4f}")

    # Discrimination
    diff = features - osc_features
    print(f"  Difference: {diff.mean():.4f} ± {diff.std():.4f}")
    print(f"  Max |diff|: {np.abs(diff).max():.4f} (generator {np.argmax(np.abs(diff))})")

    print("\n" + "=" * 60)

    return generators


if __name__ == '__main__':
    generators = verify_g2()

    # Save generators for use by other scripts
    np.save('/workspace/sounio/scripts/research/g2_generators.npy', generators)
    print(f"\nG₂ generators saved to scripts/research/g2_generators.npy")
    print(f"Shape: {generators.shape}")
    print(f"Use: generators = np.load('g2_generators.npy')")
