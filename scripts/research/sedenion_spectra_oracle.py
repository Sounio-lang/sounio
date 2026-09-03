#!/usr/bin/env python3
"""Oracle: the emergent metric of the sedenion ZD geometry — its graphs have INTEGRAL spectra
(Frente B, vector 4/1). Verifies the spectral moments trace(A^k) of the fiber graph (K_{6,6}-3K_{2,2})
and the 2*K_7 fiber-incidence graph against their proposed integral spectra.

Fiber adjacency spectrum {4, 2^2, 0^6, -2^2, -4}  => Laplacian {0,2^2,4^6,6^2,8}, algebraic conn. 2.
2*K_7 adjacency spectrum {12, -2^6}               => Laplacian {0, 14^6}, algebraic conn. 14.

Output: FIBM2/FIBM4/FIBM6, K7M2/K7M3/K7M4, VERTS, SPECTRA <OK|FAIL>.
"""
from itertools import product
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


def moments(A, K):
    n = len(A)
    P = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    out = []
    for _ in range(K):
        P = [[sum(P[i][t] * A[t][j] for t in range(n)) for j in range(n)] for i in range(n)]
        out.append(sum(P[i][i] for i in range(n)))
    return out


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
    mf = moments(Af, 6)
    A7 = [[0 if i == j else 2 for j in range(7)] for i in range(7)]
    m7 = moments(A7, 4)
    print(f"FIBM2 {mf[1]}")
    print(f"FIBM4 {mf[3]}")
    print(f"FIBM6 {mf[5]}")
    print(f"K7M2 {m7[1]}")
    print(f"K7M3 {m7[2]}")
    print(f"K7M4 {m7[3]}")
    print(f"VERTS {len(fib)}")
    ok = (mf[1] == 48 and mf[3] == 576 and mf[5] == 8448 and m7[1] == 168 and m7[2] == 1680 and m7[3] == 20832 and len(fib) == 12)
    print(f"SPECTRA {'OK' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
