#!/usr/bin/env python3
"""Oracle: cross-check of Gresnigt's three generation-octonions + the G2 (color-side) monomial
automorphism permuting them (Frente B, vector 4/3). NOT the family S3 (Brown factor); NOT a physics
bridge. See docs/research/sedenion_gresnigt_octonions.md. Ref: Gresnigt arXiv:2306.13098."""
from itertools import combinations


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


def antiL(i, j):
    return all(cd_sigma(i, j ^ c) * cd_sigma(j, c) + cd_sigma(j, i ^ c) * cd_sigma(i, c) == 0
               for c in range(16))


def pprod(ul, uh, us, vl, vh, vs, k):
    a = 0
    if (ul ^ vl) == k: a += cd_sigma(ul, vl)
    if (ul ^ vh) == k: a += vs * cd_sigma(ul, vh)
    if (uh ^ vl) == k: a += us * cd_sigma(uh, vl)
    if (uh ^ vh) == k: a += us * vs * cd_sigma(uh, vh)
    return a


def has_zd(S):
    U = sorted(S)
    prims = [(a, b, s) for a, b in combinations(U, 2) for s in (1, -1)]
    for (ul, uh, us), (vl, vh, vs) in combinations(prims, 2):
        if all(pprod(ul, uh, us, vl, vh, vs, k) == 0 for k in range(16)):
            return True
    return False


def nonanti(S):
    return sum(1 for i, j in combinations(sorted(S), 2) if not antiL(i, j))


PI = [0, 2, 3, 1, 4, 6, 7, 5, 8, 10, 11, 9, 12, 14, 15, 13]
O1 = {1, 4, 5, 8, 9, 12, 13}
O2 = {2, 4, 6, 8, 10, 12, 14}
O3 = {3, 4, 7, 8, 11, 12, 15}


def main():
    aut = 1 if all(cd_sigma(PI[i], PI[j]) == cd_sigma(i, j) and PI[i ^ j] == PI[i] ^ PI[j]
                   for i in range(16) for j in range(16)) else 0
    ord3 = 1 if (all(PI[PI[PI[i]]] == i for i in range(16)) and any(PI[i] != i for i in range(16))) else 0
    fixes = 1 if (PI[8] == 8 and PI[4] == 4 and PI[12] == 12) else 0
    g2 = 1 if (all(1 <= PI[i] <= 7 for i in range(1, 8))
               and all(cd_sigma(PI[i], PI[j], 3) == cd_sigma(i, j, 3) for i in range(8) for j in range(8))) else 0
    octs = 1 if all(not has_zd(S) and nonanti(S) == 6 for S in (O1, O2, O3)) else 0
    img = lambda S: set(PI[x] for x in S)
    cyc = 1 if (img(O1) == O2 and img(O2) == O3) else 0
    print(f"AUT_OK {aut}")
    print(f"ORD3 {ord3}")
    print(f"FIX_QH_E8 {fixes}")
    print(f"G2_DIAGONAL {g2}")
    print(f"OCTS_ZDFREE_NA6 {octs}")
    print(f"CYCLE_O1O2O3 {cyc}")
    print(f"GRESNIGT {'OK' if aut and ord3 and fixes and g2 and octs and cyc else 'FAIL'}")


if __name__ == "__main__":
    main()
