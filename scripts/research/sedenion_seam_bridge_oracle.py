#!/usr/bin/env python3
"""Oracle: the e8-seam bridge (Frente B). Certifies the FULL six-way equivalence (including the
determinant/eigenvalue formulations via exact integer Bareiss determinants) and the 4-regular quartet
incidence, over all 56 lower x upper pairs of S. See sedenion_seam_bridge.md."""
from collections import Counter

DIM = 16


def cd_sigma(a, b, bits=4):
    if a == 0 or b == 0:
        return 1
    if bits <= 1:
        return -1
    half = 1 << (bits - 1)
    aH, bH, aL, bL = a >= half, b >= half, a & (half - 1), b & (half - 1)
    if not aH and not bH:
        return cd_sigma(aL, bL, bits - 1)
    if not aH and bH:
        return cd_sigma(bL, aL, bits - 1)
    if aH and not bH:
        return cd_sigma(aL, bL, bits - 1) if bL == 0 else -cd_sigma(aL, bL, bits - 1)
    return -cd_sigma(bL, aL, bits - 1) if bL == 0 else cd_sigma(bL, aL, bits - 1)


def Lm(i):
    M = [[0] * DIM for _ in range(DIM)]
    for k in range(DIM):
        M[i ^ k][k] = cd_sigma(i, k)
    return M


L = [Lm(i) for i in range(DIM)]
mm = lambda A, B: [[sum(A[r][t] * B[t][c] for t in range(DIM)) for c in range(DIM)] for r in range(DIM)]
madd = lambda A, B, s=1: [[A[r][c] + s * B[r][c] for c in range(DIM)] for r in range(DIM)]
isnegI = lambda A: all(A[r][c] == (-1 if r == c else 0) for r in range(DIM) for c in range(DIM))
I16 = [[1 if r == c else 0 for c in range(DIM)] for r in range(DIM)]


def anticomm0(i, j):
    P, Q = mm(L[i], L[j]), mm(L[j], L[i])
    return all(P[r][c] + Q[r][c] == 0 for r in range(DIM) for c in range(DIM))


def bareiss_det(M):
    A = [row[:] for row in M]
    n = len(A)
    prev = 1
    sign = 1
    for k in range(n - 1):
        if A[k][k] == 0:
            sw = next((i for i in range(k + 1, n) if A[i][k] != 0), None)
            if sw is None:
                return 0
            A[k], A[sw] = A[sw], A[k]
            sign = -sign
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                A[i][j] = (A[i][j] * A[k][k] - A[i][k] * A[k][j]) // prev
        prev = A[k][k]
    return sign * A[n - 1][n - 1]


LO = range(1, 8)
HI = range(8, 16)
allpairs = {(l, u) for l in LO for u in HI}
seam = {(l, u) for (l, u) in allpairs if u == 8 or u == (l ^ 8)}
offseam = allpairs - seam


def qprod0(l, u, a, b, s):
    def comp(k):
        acc = 0
        if (l ^ a) == k: acc += cd_sigma(l, a)
        if (l ^ b) == k: acc += s * cd_sigma(l, b)
        if (u ^ a) == k: acc += cd_sigma(u, a)
        if (u ^ b) == k: acc += s * cd_sigma(u, b)
        return acc
    return all(comp(k) == 0 for k in range(16))


def is_zd(l, u):
    return any(qprod0(l, u, a, b, s) for a in range(1, 16) for b in range(a + 1, 16) for s in (1, -1))


def main():
    # sparse core (matches souc)
    equiv_ok = 1
    n_nonanti = n_offseam = n_zd = 0
    for (l, u) in allpairs:
        ac = anticomm0(l, u)
        sq = isnegI(mm(mm(L[l], L[u]), mm(L[l], L[u])))
        off = (l, u) in offseam
        zd = is_zd(l, u)
        if not (ac == sq and ac != off and ac != zd):
            equiv_ok = 0
        if not ac: n_nonanti += 1
        if off: n_offseam += 1
        if zd: n_zd += 1
    # FULL six-way including det/spec formulations
    sixway = 1
    for (l, u) in allpairs:
        ac = anticomm0(l, u)
        sq = isnegI(mm(mm(L[l], L[u]), mm(L[l], L[u])))
        LL = mm(L[l], L[u])
        p1 = bareiss_det(madd(LL, I16, -1)) == 0            # +1 in spec
        sing = bareiss_det(madd(L[l], L[u], 1)) == 0 or bareiss_det(madd(L[l], L[u], -1)) == 0
        zd = is_zd(l, u)
        off = (l, u) in offseam
        # all six 'off-seam' conditions equal: (not ac),(not sq),p1,sing,zd,off
        row = [(not ac), (not sq), p1, sing, zd, off]
        if not all(x == row[0] for x in row):
            sixway = 0
    # 4-regular quartet incidence
    P = [(l, u, s) for (l, u) in offseam for s in (1, -1)]

    def pv0(p, q):
        l1, u1, s1 = p
        l2, u2, s2 = q

        def comp(k):
            acc = 0
            if (l1 ^ l2) == k: acc += cd_sigma(l1, l2)
            if (l1 ^ u2) == k: acc += s2 * cd_sigma(l1, u2)
            if (u1 ^ l2) == k: acc += s1 * cd_sigma(u1, l2)
            if (u1 ^ u2) == k: acc += s1 * s2 * cd_sigma(u1, u2)
            return acc
        return all(comp(k) == 0 for k in range(16))
    edges = set()
    for i in range(len(P)):
        for j in range(i + 1, len(P)):
            if pv0(P[i], P[j]) or pv0(P[j], P[i]):
                edges.add((P[i], P[j]))
    quart = {}
    for (a, b) in edges:
        supp = frozenset([a[0], a[1], b[0], b[1]])
        if len(supp) == 4:
            quart[supp] = quart.get(supp, 0) + 1
    q4 = list(quart)
    persub = Counter()
    deg = {p: 0 for p in offseam}
    for q in q4:
        los = [x for x in q if x in LO]
        his = [x for x in q if x in HI]
        persub[sum(1 for l in los for u in his if (l, u) in offseam)] += 1
        for l in los:
            for u in his:
                if (l, u) in deg:
                    deg[(l, u)] += 1
    incidence_ok = 1 if (len(q4) == 42 and set(persub) == {4} and set(deg.values()) == {4}
                         and len(edges) == 168) else 0
    print(f"EQUIV_OK {equiv_ok}")
    print(f"N_NONANTI {n_nonanti}")
    print(f"N_OFFSEAM {n_offseam}")
    print(f"N_ZD {n_zd}")
    print(f"SIXWAY_OK {sixway}")
    print(f"INCIDENCE_OK {incidence_ok}")
    ok = equiv_ok and n_nonanti == 42 and n_offseam == 42 and n_zd == 42
    print(f"BRIDGE {'OK' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
