#!/usr/bin/env python3
"""Oracle: the octonion-subalgebra census corroborating Erratum E1 (Frente B, vector 4/3).
Of the 15 basis-aligned 3-dim F2 subspaces of the sedenion index space, exactly ONE is Clifford-pure
(all internal left-mult pairs anticommute) — the base octonion {1..7}; the three subspaces through the
base quaternion {1,2,3} give non-anticommuting-L-pair counts {0,6,12}. Hence the family-S3 octonion
copies of Gresnigt's three-generation construction cannot all be monomial. See sedenion_octonion_census.md.
"""
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
    acc = 0
    if (ul ^ vl) == k: acc += cd_sigma(ul, vl)
    if (ul ^ vh) == k: acc += vs * cd_sigma(ul, vh)
    if (uh ^ vl) == k: acc += us * cd_sigma(uh, vl)
    if (uh ^ vh) == k: acc += us * vs * cd_sigma(uh, vh)
    return acc


def has_zd(S):
    U = sorted(S)
    prims = [(a, b, s) for a, b in combinations(U, 2) for s in (1, -1)]
    for (ul, uh, us), (vl, vh, vs) in combinations(prims, 2):
        if all(pprod(ul, uh, us, vl, vh, vs, k) == 0 for k in range(16)):
            return True
    return False


def nonanti(S):
    return sum(1 for i, j in combinations(sorted(S), 2) if not antiL(i, j))


def subspaces():
    subs = {}
    for a, b, c in combinations(range(1, 16), 3):
        S = frozenset(x for x in [a, b, a ^ b, c, a ^ c, b ^ c, a ^ b ^ c] if x)
        if len(S) == 7:
            subs[S] = True
    return list(subs)


def main():
    subs = subspaces()
    zdfree = sum(1 for S in subs if not has_zd(S))
    quasi = len(subs) - zdfree
    pure = [S for S in subs if nonanti(S) == 0]
    pure_is_lower = 1 if any(set(S) == {1, 2, 3, 4, 5, 6, 7} for S in pure) else 0
    n_c4 = nonanti({1, 2, 3, 4, 5, 6, 7})
    n_c8 = nonanti({1, 2, 3, 8, 9, 10, 11})
    n_c12 = nonanti({1, 2, 3, 12, 13, 14, 15})
    print(f"NSUB {len(subs)}")
    print(f"ZDFREE {zdfree}")
    print(f"QUASI {quasi}")
    print(f"PURE {len(pure)}")
    print(f"PURE_IS_LOWER {pure_is_lower}")
    print(f"QUAT_C4 {n_c4}")
    print(f"QUAT_C8 {n_c8}")
    print(f"QUAT_C12 {n_c12}")
    ok = (len(subs) == 15 and zdfree == 8 and quasi == 7 and len(pure) == 1 and pure_is_lower == 1
          and n_c4 == 0 and n_c8 == 6 and n_c12 == 12)
    print(f"OCTCENSUS {'OK' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
