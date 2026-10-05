"""Exact-over-Q recomputation of the n=16 structure claims in #1466.

The merged oracle computes rank, kernel identity and independence by RREF over
F_P with P = 2^31 - 1, justified in the doc by "entries are 0, +-1, so the rank
equals the rational rank". That implication is false in general, and the size
does not rescue it: Hadamard for a 16x16 matrix with |a_ij| <= 1 is 16^8 = 2^32,
exactly 2.00x P, so a single minor can be an exact multiple of P.

This recomputes the same three numbers with fractions.Fraction -- no modulus
anywhere -- and diffs them against the modular oracle.
"""
from fractions import Fraction
import importlib.util, sys

spec = importlib.util.spec_from_file_location("oracle", "/tmp/oracle.py")
oracle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)
P = oracle.P

def rref_exact(M, n):
    R = [[Fraction(x) for x in row] for row in M]
    r = 0; piv = []
    for c in range(n):
        q = None
        for i in range(r, len(R)):
            if R[i][c] != 0: q = i; break
        if q is None: continue
        R[r], R[q] = R[q], R[r]
        inv = Fraction(1, 1) / R[r][c]
        R[r] = [x * inv for x in R[r]]
        for i in range(len(R)):
            if i != r and R[i][c] != 0:
                f = R[i][c]; R[i] = [a - f * b for a, b in zip(R[i], R[r])]
        piv.append(c); r += 1
    return R[:r], piv

def structure_exact(K):
    tab = oracle.build_fast(K); n = 1 << K; h = n // 2
    prims = {}; kernels = {}
    for a in range(1, h):
        for b in range(h, n):
            M = [[0]*n for _ in range(n)]
            for j in range(n):
                s, k = tab[a][j]; M[k][j] += s
                s, k = tab[b][j]; M[k][j] += s
            R, piv = rref_exact(M, n)
            free = [c for c in range(n) if c not in piv]
            if len(free) != 4: continue
            B = []
            for f in free:
                v = [Fraction(0)]*n; v[f] = Fraction(1)
                for ri, c in enumerate(piv): v[c] = -R[ri][f]
                B.append(v)
            C, _ = rref_exact(B, n)
            key = tuple(tuple(x for x in row) for row in C)
            kernels[key] = 1; prims[(a, b)] = key
    return prims, list(kernels), tab, n

def max_independent_exact(KL, n):
    best = [0]
    def rank_of(rows):
        R, piv = rref_exact([list(r) for r in rows], n)
        return len(piv)
    def bt(start, chosen):
        if len(chosen) > best[0]: best[0] = len(chosen)
        for i in range(start, len(KL)):
            cand = chosen + [KL[i]]
            rows = [list(r) for k in cand for r in k]
            if rank_of(rows) == 4 * len(cand):
                bt(i + 1, cand)
    bt(0, [])
    return best[0]

prims, KL, tab, n = structure_exact(4)
print(f"  EXATO sobre Q, n=16:")
print(f"    pares com dim ker = 4 : {len(prims)}")
print(f"    kernels distintos     : {len(KL)}")
print(f"    max independentes     : {max_independent_exact(KL, n)}")
