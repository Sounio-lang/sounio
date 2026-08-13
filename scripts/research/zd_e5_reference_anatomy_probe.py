#!/usr/bin/env python3
"""Reference-side anatomy of M = P3(·,·,1,m) — the measurements behind E5 doc §9.

Self-contained (copies the definitions from formal/lean4/SounioZDFiberAntisym.lean
and SounioCDCocycle.lean; no imports from the repo).  Prints:

  1. row 0 sign law: s_x = (-1)^popcount(x) except s_1 = +1
  2. coboundary defect set size: 24·[m-1,2]_2 unordered pairs
  3. diagonal of M^3 split at W=1
  4. trace decomposition: tr(P^3) = poly(H), and the three mixed terms

All numbers are MEASURED (finite computation), not proved.
"""
from functools import lru_cache


@lru_cache(maxsize=None)
def cd_sigma(a: int, b: int, n: int) -> int:
    if n == 0:
        return -1
    if n == 1:
        return 1 if a == 0 or b == 0 else -1
    if a == 0 or b == 0:
        return 1
    half = 1 << (n - 1)
    a_hi, b_hi = a >= half, b >= half
    if not a_hi and not b_hi:
        return cd_sigma(a % half, b % half, n - 1)
    if not a_hi and b_hi:
        return cd_sigma(b % half, a % half, n - 1)
    if a_hi and not b_hi:
        if b % half == 0:
            return cd_sigma(a % half, 0, n - 1)
        return -cd_sigma(a % half, b % half, n - 1)
    if b % half == 0:
        return -cd_sigma(0, a % half, n - 1)
    return cd_sigma(b % half, a % half, n - 1)


def hi(x: int, llo: int, n: int) -> int:
    return (x ^ llo) + (1 << (n + 1))


def p3(l: int, y: int, llo: int, n: int) -> int:
    return cd_sigma(l, hi(y, llo, n), n + 2) * cd_sigma(hi(l, llo, n), y, n + 2)


def qbin2(n: int) -> int:
    """[n,2]_2 (zero for n < 2)"""
    if n < 2:
        return 0
    return ((1 << n) - 1) * ((1 << (n - 1)) - 1) // 3


def qbin3(m: int) -> int:
    """[m,3]_2 (zero for m < 3)"""
    if m < 3:
        return 0
    return ((1 << m) - 1) * ((1 << (m - 1)) - 1) * ((1 << (m - 2)) - 1) // 21


def popcount(x: int) -> int:
    return bin(x).count("1")


def main() -> None:
    # 1. row 0 law
    bad = 0
    for m in range(1, 6):
        H = 1 << (m + 1)
        for x in range(H):
            want = 1 if x == 0 or x == 1 else (-1) ** popcount(x)
            if p3(0, x, 1, m) != want:
                bad += 1
    print(f"row0 law: failures = {bad} (levels 1..5)")

    # 2. defect count
    for m in (2, 3, 4):
        H = 1 << (m + 1)
        s = [p3(0, x, 1, m) for x in range(H)]
        d = sum(
            1
            for a in range(1, H)
            for b in range(a + 1, H)
            if p3(a, b, 1, m) * s[a] * s[b] != 1
        )
        print(f"defect pairs m={m}: {d}  (24*[m-1,2]_2 = {24 * qbin2(m - 1)})")

    # 3. diagonal split + 4. trace decomposition
    for m in (3, 4):
        H = 1 << (m + 1)
        s = [p3(0, x, 1, m) for x in range(H)]

        def P(a: int, b: int) -> int:
            if a == 0 and b == 0:
                return 1
            if a == 0:
                return s[b]
            if b == 0:
                return -s[a]
            if a == b:
                return -1
            return s[a] * s[b]

        K = [
            sum(p3(a, b, 1, m) * p3(b, c, 1, m) * p3(c, a, 1, m)
                for b in range(H) for c in range(H))
            for a in range(H)
        ]
        from collections import Counter
        dist = Counter(K)
        print(f"diagonal m={m}: {dict(sorted(dist.items()))}")

        ppp = epp = pee = eee = 0
        for a in range(H):
            for b in range(H):
                for c in range(H):
                    pab, pbc, pca = P(a, b), P(b, c), P(c, a)
                    eab = p3(a, b, 1, m) - pab
                    ebc = p3(b, c, 1, m) - pbc
                    eca = p3(c, a, 1, m) - pca
                    ppp += pab * pbc * pca
                    epp += eab * pbc * pca + pab * ebc * pca + pab * pbc * eca
                    pee += eab * ebc * pca + eab * pbc * eca + pab * ebc * eca
                    eee += eab * ebc * eca
        poly = H**3 - 12 * H**2 + 28 * H - 16
        print(
            f"trace m={m}: tr(P^3)={ppp} poly={poly} | "
            f"3EPP={epp} 3PEE={pee} EEE={eee} | delta={epp + pee + eee} "
            f"(target -{1728 * qbin3(m)})"
        )


if __name__ == "__main__":
    main()
