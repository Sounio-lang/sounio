#!/usr/bin/env python3
"""(I) probe: the 2-adic route.

Checks, against the SAME definitions the Lean file uses (`dcoef`/`dterm`/`Ddig`,
`SounioZDFiberAntisym.lean:8178-8188`), three claims:

  A. dcoef(m, j+4) = 2^(2m-2j-3) * (2^(j+1)-1) * (2^(j+2)-1)   -- odd cofactor, so
     v2(c_j) = 2m-2j-3, strictly decreasing in j with gap exactly 2.
  B. F(m, 2^k + y') = c_k - F(m, y')  for y' < 2^k, where F(m,y) = Ddig(m, 8y+1).
  C. v2(F(m,y)) = 2m-2*topbit(y)-3 for y != 0  (hence F(m,y) != 0), and F(m,.) injective.
"""


def psg(x: int) -> int:
    return -1 if bin(x).count("1") % 2 else 1


def dcoef(m: int, i: int) -> int:
    # Nat-truncated subtraction, exactly as in Lean.
    return max(2**i - 4, 0) * max(2**i - 8, 0) * 4 ** (m - i)


def dterm(m: int, W: int, i: int) -> int:
    lo = 2 ** (i - 1) if i >= 1 else 1  # Nat: 0-1 = 0
    if 2 ** (i - 1 if i >= 1 else 0) <= W % 2**i and W % lo != 0:
        return dcoef(m, i) * psg(W >> i)
    return 0


def Ddig(m: int, W: int) -> int:
    return sum(dterm(m, W, i) for i in range(m + 1))


def F(m: int, y: int) -> int:
    return Ddig(m, 8 * y + 1)


def v2(x: int) -> int:
    return (x & -x).bit_length() - 1


def main() -> int:
    bad = 0

    # A
    nA = 0
    for m in range(6, 15):
        for j in range(0, m - 3):
            lhs = dcoef(m, j + 4)
            rhs = 2 ** (2 * m - 2 * j - 3) * (2 ** (j + 1) - 1) * (2 ** (j + 2) - 1)
            nA += 1
            if lhs != rhs:
                bad += 1
                print(f"A FAIL m={m} j={j}: {lhs} vs {rhs}")
    print(f"A: dcoef factorisation, {nA} checks, {bad} mismatches")

    # B
    badB = 0
    nB = 0
    for m in range(6, 13):
        for k in range(0, m - 3):
            ck = dcoef(m, k + 4)
            for yp in range(2**k):
                nB += 1
                if F(m, 2**k + yp) != ck - F(m, yp):
                    badB += 1
                    if badB < 4:
                        print(f"B FAIL m={m} k={k} y'={yp}")
    print(f"B: top-bit peel F(m,2^k+y')=c_k-F(m,y'), {nB} checks, {badB} mismatches")

    # C
    badC = 0
    nC = 0
    for m in range(6, 15):
        seen = {}
        for y in range(2 ** (m - 3)):
            v = F(m, y)
            nC += 1
            if y == 0:
                if v != 0:
                    badC += 1
                    print(f"C FAIL F(m,0)!=0 m={m}")
            else:
                k = y.bit_length() - 1
                if v == 0 or v2(v) != 2 * m - 2 * k - 3:
                    badC += 1
                    if badC < 4:
                        print(f"C FAIL val m={m} y={y} v={v} k={k}")
            if v in seen:
                badC += 1
                print(f"C FAIL collision m={m}: y={seen[v]} and y={y} both -> {v}")
            seen[v] = y
    print(f"C: valuation + injectivity, {nC} labels, {badC} mismatches")

    total = bad + badB + badC
    print(("ALL PASS" if total == 0 else f"FAILURES: {total}"))
    return 0 if total == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
