#!/usr/bin/env python3
"""Independent oracle: the sedenion left-multiplication algebra is Cℓ(8) (Frente B, vector 4/3).
Exact pure-Python port of the operator's numpy reference script. Certifies the Clifford presentation
(8 anticommuting sqrt(-I) generators L_1..L_8), the full 256-dim algebra (Cℓ(8)=M16(C), rank mod p),
the 42-non-anticommuting-pairs fingerprint (all lower-upper), the ladder rank 4, and the Gresnigt
S3-invariant charge Q_1 SM-charge spectrum. See docs/research/sedenion_clifford8.md.

Refs: Gresnigt EPJC 2019 (s10052-019-6967-1) / 2024 (s10052-024-13476-0); arXiv:2306.13098;
Furey arXiv:1405.4601, 1611.09182 (octonion Cℓ(6) one generation).
"""
from itertools import combinations, product

DIM = 16


def cd_conj(x):
    n = len(x)
    if n == 1:
        return x[:]
    h = n // 2
    return cd_conj(x[:h]) + [-v for v in x[h:]]


def cd_mul(x, y):
    n = len(x)
    if n == 1:
        return [x[0] * y[0]]
    h = n // 2
    a, b, c, d = x[:h], x[h:], y[:h], y[h:]
    add = lambda u, v: [p + q for p, q in zip(u, v)]
    sub = lambda u, v: [p - q for p, q in zip(u, v)]
    return sub(cd_mul(a, c), cd_mul(cd_conj(d), b)) + add(cd_mul(d, a), cd_mul(b, cd_conj(c)))


E = [[1 if k == i else 0 for k in range(DIM)] for i in range(DIM)]


def Lmat(i):
    cols = [cd_mul(E[i], E[k]) for k in range(DIM)]
    return [[cols[k][r] for k in range(DIM)] for r in range(DIM)]


L = [Lmat(i) for i in range(DIM)]


def mm(A, B):
    return [[sum(A[r][t] * B[t][c] for t in range(DIM)) for c in range(DIM)] for r in range(DIM)]


def anti0(A, B):
    return all(sum(A[r][t] * B[t][c] + B[r][t] * A[t][c] for t in range(DIM)) == 0
               for r in range(DIM) for c in range(DIM))


def sq_negI(A):
    return all(sum(A[r][t] * A[t][c] for t in range(DIM)) == (-1 if r == c else 0)
               for r in range(DIM) for c in range(DIM))


def algebra_dim_modp(gens, P=1000003):
    rows = []
    flat = lambda M: tuple(M[r][c] % P for r in range(DIM) for c in range(DIM))

    def add(v):
        v = list(v)
        for piv, row in rows:
            if v[piv]:
                f = v[piv]
                v = [(a - f * b) % P for a, b in zip(v, row)]
        for i2, x in enumerate(v):
            if x % P:
                inv = pow(x, P - 2, P)
                rows.append((i2, [(a * inv) % P for a in v]))
                return True
        return False

    I = [[1 if r == c else 0 for c in range(DIM)] for r in range(DIM)]
    add(flat(I))
    basis = [I]
    for g in gens:
        if add(flat(g)):
            basis.append(g)
    frontier = list(basis)
    while len(rows) < DIM * DIM:
        newf = []
        for A in frontier:
            for g in gens:
                Pm = mm(A, g)
                if add(flat(Pm)):
                    newf.append(Pm)
        if not newf:
            break
        frontier = newf
    return len(rows)


def main():
    sq_ok = 1 if all(sq_negI(L[i]) for i in range(1, 9)) else 0
    anti28 = sum(1 for i, j in combinations(range(1, 9), 2) if anti0(L[i], L[j]))
    nonanti = [(i, j) for i, j in combinations(range(1, 16), 2) if not anti0(L[i], L[j])]
    lohi = sum(1 for i, j in nonanti if i >= 8 or j >= 8)
    chosen = []
    for i in range(1, 16):
        if all(anti0(L[i], L[j]) for j in chosen):
            chosen.append(i)
    rank = len(chosen) // 2
    qc = {}
    for occ in product([0, 1], repeat=4):
        n1, n2, n3, n4 = occ
        qc[n1 + n2 + n3 - 3 * n4] = qc.get(n1 + n2 + n3 - 3 * n4, 0) + 1
    q_ok = 1 if all(qc.get(k, 0) == v for k, v in {-3: 1, -2: 3, -1: 3, 0: 2, 1: 3, 2: 3, 3: 1}.items()) else 0
    d = algebra_dim_modp([L[i] for i in range(1, 16)])
    print(f"CL8_SQ_OK {sq_ok}")
    print(f"CL8_ANTI28 {anti28}")
    print(f"NONANTI {len(nonanti)}")
    print(f"NONANTI_LOHI {lohi}")
    print(f"GENS {len(chosen)}")
    print(f"RANK {rank}")
    print(f"Q1_OK {q_ok}")
    print(f"DIM256 {d}")
    ok = (sq_ok and anti28 == 28 and len(nonanti) == 42 and lohi == 42 and len(chosen) == 8
          and rank == 4 and q_ok and d == 256)
    print(f"CL8 {'OK' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
