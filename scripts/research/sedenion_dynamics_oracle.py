#!/usr/bin/env python3
"""Oracle: the substrate DYNAMICS of the sedenion ZD geometry (Frente B, vector 4/2).

The Laplacian L = D - A is the generator of the substrate dynamics. Two exact, integer-certified
invariants of the ZD-geometry graphs:

  1. SPANNING-TREE COMPLEXITY tau (Matrix-Tree theorem): tau = det of any principal (n-1)x(n-1)
     cofactor of L, computed by FRACTION-FREE (Bareiss) integer Gaussian elimination — exact ints.
       * fiber K_{6,6}-3K_{2,2} (12 v, deg 4):  tau = 393216 = (2^2*4^6*6^2*8)/12.
       * 2*K_7 fiber-incidence (7 v, off-diag 2, deg 12): tau = 1075648 = 14^6/7.
  2. RANDOM-WALK RETURN COUNTS (A^k)_{00}: fiber (A^2)=4=deg, (A^4)=48; 2*K_7 (A^2)=24=2^2*6, (A^4)=2976.

Output lines: SPANTREE_FIB, SPANTREE_K7, FIB_A2, FIB_A4, K7_A2, K7_A4, VERTS, DYNAMICS <OK|FAIL>.
Cross-verified vs tests/run-pass/sedenion_dynamics.sio by scripts/ci/sedenion_dynamics_gate.sh.
"""
from collections import defaultdict


def cd_sigma(a, b, bits=4):
    if a == 0 or b == 0: return 1
    if bits <= 1: return -1
    half = 1 << (bits - 1); aH = a >= half; bH = b >= half; aL = a & (half - 1); bL = b & (half - 1)
    if not aH and not bH: return cd_sigma(aL, bL, bits - 1)
    if not aH and bH: return cd_sigma(bL, aL, bits - 1)
    if aH and not bH: return cd_sigma(aL, bL, bits - 1) if bL == 0 else -cd_sigma(aL, bL, bits - 1)
    return -cd_sigma(bL, aL, bits - 1) if bL == 0 else cd_sigma(bL, aL, bits - 1)


def mul(a, b):
    o = {}
    for i, ci in a.items():
        for j, cj in b.items():
            k = i ^ j; o[k] = o.get(k, 0) + cd_sigma(i, j) * ci * cj
            if o[k] == 0: del o[k]
    return o


def vec(c):
    lo, hi, neg = c; return {lo: 1, hi: (-1 if neg else 1)}


def bareiss_det(M):
    """Fraction-free (Bareiss) integer determinant — exact, no rationals."""
    M = [row[:] for row in M]; n = len(M); sign = 1; prev = 1
    for k in range(n - 1):
        if M[k][k] == 0:
            sw = next((r for r in range(k + 1, n) if M[r][k] != 0), None)
            if sw is None: return 0
            M[k], M[sw] = M[sw], M[k]; sign = -sign
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                M[i][j] = (M[i][j] * M[k][k] - M[i][k] * M[k][j]) // prev
        prev = M[k][k]
    return sign * M[n - 1][n - 1]


def spanning_trees(A):
    """Matrix-Tree: tau = det of the (n-1)x(n-1) Laplacian cofactor (delete row/col 0)."""
    n = len(A)
    deg = [sum(A[i]) for i in range(n)]
    L = [[(deg[i] if i == j else 0) - A[i][j] for j in range(n)] for i in range(n)]
    cof = [[L[i][j] for j in range(1, n)] for i in range(1, n)]
    return bareiss_det(cof)


def matpow_diag0(A, k):
    """(A^k)_{00} via repeated integer matmul."""
    n = len(A); P = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    for _ in range(k):
        P = [[sum(P[i][t] * A[t][j] for t in range(n)) for j in range(n)] for i in range(n)]
    return P[0][0]


def main():
    cands = [(lo, hi, neg) for lo in range(1, 8) for hi in range(8, 16) for neg in (0, 1)]
    part = [c for c in cands if any(not mul(vec(c), vec(b)) for b in cands)]
    adj = defaultdict(set)
    for i in range(len(part)):
        for j in range(len(part)):
            if i != j and not mul(vec(part[i]), vec(part[j])): adj[part[i]].add(part[j])
    fib = [v for v in part if (v[0] ^ v[1]) == 9]
    idx = {v: i for i, v in enumerate(fib)}
    Af = [[0] * len(fib) for _ in fib]
    for v in fib:
        for w in adj[v]:
            if w in idx: Af[idx[v]][idx[w]] = 1
    A7 = [[0 if i == j else 2 for j in range(7)] for i in range(7)]

    tau_fib = spanning_trees(Af)
    tau_k7 = spanning_trees(A7)
    fib_a2 = matpow_diag0(Af, 2)
    fib_a4 = matpow_diag0(Af, 4)
    k7_a2 = matpow_diag0(A7, 2)
    k7_a4 = matpow_diag0(A7, 4)
    verts = len(fib)

    print(f"SPANTREE_FIB {tau_fib}")
    print(f"SPANTREE_K7 {tau_k7}")
    print(f"FIB_A2 {fib_a2}")
    print(f"FIB_A4 {fib_a4}")
    print(f"K7_A2 {k7_a2}")
    print(f"K7_A4 {k7_a4}")
    print(f"VERTS {verts}")
    ok = (tau_fib == 393216 and tau_k7 == 1075648 and fib_a2 == 4 and fib_a4 == 48
          and k7_a2 == 24 and k7_a4 == 2976 and verts == 12)
    print(f"DYNAMICS {'OK' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
