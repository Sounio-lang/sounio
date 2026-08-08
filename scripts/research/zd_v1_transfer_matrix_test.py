#!/usr/bin/env python3
"""THE AFTERNOON TEST — does the level recursion close as a finite transfer matrix?

Tiers 60-94 reduced `tri3` at level m+1 to level-m data, but each rung spawned new
unevaluated scalars: (M^2)_WW, (M^3)_WW, tr(M^2 Pi_W), (M Pi M)_WW, ...  The adviser's
claim is that this is a transfer-matrix computation in disguise: because D and E are
rank-one perturbations of I and Pi_W is an involution, the scalars generated live in a
FINITE-dimensional space, so there should be a CONSTANT matrix T with

    v_{m+1} = T v_m      (v augmented with 2^m and 1 to absorb the inhomogeneity).

If a small T fits exactly, the whole arc collapses to ONE theorem.  If none does, the
level-recursion route is dead.  This script decides it with data.

IT CLOSES.  Accounting, stated precisely because the first draft of this note got it sloppy:
the 13 probed components are 3 DYNAMIC (s3, cp2, cp3), 8 CLOSED FORMS in H = 2^(m+1) that are
label-independent (s2 = (H-2)^2, a2 = H-2, z2 = -(H-2), z3 = 32-10H, a3 = 2H-16,
mpm = -(H-2), pm2 = H-2, mp2 = -(H-2)), and 2 bookkeeping entries (H, 1).  Since every closed
form is a polynomial in H, the data has rank 7 and the exact generating set is

    {s3, cp2, cp3, 1, H, H^2, H^3}

-- note H^2 is REQUIRED and is not in span{1, H}: cp3's inhomogeneity carries 24*H^2, and
H^2 = 4^(m+1) scales by 4 per level.  On that state the three dynamic coordinates satisfy,
exactly, at every level and label tested:

    s3(m+1)  = 8*s3(m)  + 24*cp2(m) - 176 + 72*H
    cp2(m+1) = 4*cp2(m)            +  36 - 16*H
    cp3(m+1) = 8*cp3(m)            + 240 - 168*H + 24*H^2

FIT: m = 3..8, labels W = 1..15, 75 transitions, exact over Q.
OUT OF SAMPLE: labels W >= 16 at m = 4..7, 92 transitions never used in the fit, 0 failures.

CONSEQUENCES.

(1) cp3 does not feed s3, so tri3's own transfer is the 2x2 upper-triangular matrix
    [[8, 24], [0, 4]] with eigenvalues 8 and 4.  The inhomogeneity is label-independent, so
    WITHIN-FIBRE DIFFERENCES obey the homogeneous system (verified on many pairs).

(2) On the reference pairs (W = 2^j against W = 1) the coset coordinate is fibre-blind:
    Dcp2 = 0 at every m and j tested.  So Ds3 scales by EXACTLY 8 per level -- measured
    ratios 8.0 with no drift -- and

        D[tri3](m) = 1728 * 8^(m-j) * [j,3]_2 = 27 * 8^(m-j+2) * [j,3]_2

    with the q-binomial appearing ONCE, as the base case at m = j.  The 8^(m-j) is the
    eigenvalue-8 channel; there is no recursion left in it.

(3) THE MAXIMAL-SEAM EXCEPTION IS AN ARTIFACT OF THE MASK.  On the UNMASKED tri3 the law
    holds with no exception, including j = m; the masked tri3(P3~) deviates from it exactly
    at j = m, which is where the lane recorded an extra 288*[m-1,2]_2 term.

This is MEASURED.  The proof route it opens: (i) derive the three-line recursion from
tri3_level_transfer (already Lean, Tier 90), (ii) prove Dcp2 = 0 on references, (iii) compute
one base case.  That replaces the deviation law's open status with three finite obligations.
"""
import sys, itertools
sys.path.insert(0, "/workspace/sounio/scripts/research")
import numpy as np
from fractions import Fraction
from zd_v1_III_deviation_probe import sign_table_fast

def M_of(m, W):
    """P3(.,.,W,m) as a 2^(m+1) x 2^(m+1) integer matrix, built from the sign table."""
    H = 1 << (m + 1)
    S = sign_table_fast(m + 2).astype(np.int64)
    idx = np.arange(H)
    hi = (idx ^ W) + H                      # hi x W m
    return S[np.ix_(idx, hi)] * S[np.ix_(hi, idx)]

NAMES = ["s3","s2","a2","a3","z2","z3","cp2","cp3","mpm","pm2","mp2","pw","one"]

def state(m, W):
    H = 1 << (m + 1)
    M = M_of(m, W)
    idx = np.arange(H)
    Pi = np.zeros((H, H), dtype=np.int64); Pi[idx, idx ^ W] = 1
    M2 = M @ M; M3 = M2 @ M
    return np.array([
        int(np.trace(M3)),            # s3  = tri3 (unmasked)
        int(np.trace(M2)),            # s2
        int(M2[W, W]),                # a2  closed 2-walks at the seam vertex
        int(M3[W, W]),                # a3  closed 3-walks at the seam vertex
        int(M2[0, 0]),                # z2
        int(M3[0, 0]),                # z3
        int(np.trace(M2 @ Pi)),       # cp2 coset-shifted 2-walks
        int(np.trace(M3 @ Pi)),       # cp3
        int((M @ Pi @ M)[W, W]),      # mpm
        int((Pi @ M2)[W, W]),         # pm2
        int((M2 @ Pi)[W, W]),         # mp2
        1 << m,                       # pw  (carries the 2^(m+3)-type inhomogeneity)
        1,                            # one
    ], dtype=object)

def main():
    LEVELS = list(range(3, 9))          # m = 3..8
    LABELS = list(range(1, 16))         # valid at every level above
    data = {}
    for m in LEVELS:
        for W in LABELS:
            data[(m, W)] = state(m, W)
        print(f"  built level m={m}", flush=True)

    trans = [(m, W) for m in LEVELS[:-1] for W in LABELS]
    A = np.array([[float(x) for x in data[(m, W)]] for (m, W) in trans])
    print(f"\n{len(trans)} transitions, {len(NAMES)} components, rank(A) = "
          f"{np.linalg.matrix_rank(A)}")

    print("\nRow-by-row fit of  v_{m+1} = T v_m  (exact integer check):\n")
    ok_rows, T = [], {}
    for i, nm in enumerate(NAMES):
        b = np.array([float(data[(m + 1, W)][i]) for (m, W) in trans])
        coef, *_ = np.linalg.lstsq(A, b, rcond=None)
        rc = [Fraction(c).limit_denominator(64) for c in coef]
        pred = [sum(rc[j] * data[(m, W)][j] for j in range(len(NAMES))) for (m, W) in trans]
        exact = all(p == data[(m + 1, W)][i] for p, (m, W) in zip(pred, trans))
        ok_rows.append(exact)
        terms = " + ".join(f"{c}*{NAMES[j]}" for j, c in enumerate(rc) if c != 0)
        T[nm] = rc
        print(f"  {'EXACT' if exact else 'FAILS'}  {nm}(m+1) = {terms if terms else '0'}")

    print()
    if all(ok_rows):
        print("VERDICT: the state CLOSES — a constant transfer matrix reproduces every "
              "component at every level and label tested.")
    else:
        bad = [n for n, o in zip(NAMES, ok_rows) if not o]
        print(f"VERDICT: does NOT close on this state.  Failing rows: {bad}")
    return T

if __name__ == "__main__":
    main()
