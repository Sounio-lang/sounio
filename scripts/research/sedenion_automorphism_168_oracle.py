#!/usr/bin/env python3
"""Independent oracle for the sedenion signed-automorphism group (Frente B): the 168 = |PSL(2,7)|
that pervades the sedenion geometry IS the group Aut(S), and it FIXES e8 (the doubling seam).

A linear map M in GL(d,2) (acting on indices 1..2^d-1 as F_2^d vectors, preserving xor) is a signed
automorphism iff sigma(Mi,Mj)*sigma(i,j) is a coboundary delta(eps)(i,j)=eps(i)eps(j)eps(i^j). Uses
RIGOROUS highest-bit-pivot Gaussian elimination over F_2 (order-independent).

Output:
  OCT_AUTOS <n>    octonions {1..7}: signed autos = 168 = |GL(3,2)|
  SED_AUTOS <n>    sedenions {1..15}: signed autos = 168 (of |GL(4,2)|=20160)
  SED_FIX_E8 <n>   sedenion autos fixing e8 (index 8) = 168 (ALL of them)
  ORBIT8 <...>     orbit of e8 under the group (singleton {8} => e8 is the unique fixed point)
  ORBITS <...>     orbit partition of {1..15}
  AUT <OK|FAIL>
"""
from __future__ import annotations
from itertools import product


def cd_sigma(a, b, bits):
    if a == 0 or b == 0:
        return 1
    if bits <= 1:
        return -1
    half = 1 << (bits - 1)
    a_hi, b_hi = a >= half, b >= half
    a_lo, b_lo = a & (half - 1), b & (half - 1)
    if not a_hi and not b_hi:
        return cd_sigma(a_lo, b_lo, bits - 1)
    if not a_hi and b_hi:
        return cd_sigma(b_lo, a_lo, bits - 1)
    if a_hi and not b_hi:
        return cd_sigma(a_lo, b_lo, bits - 1) if b_lo == 0 else -cd_sigma(a_lo, b_lo, bits - 1)
    return -cd_sigma(b_lo, a_lo, bits - 1) if b_lo == 0 else cd_sigma(b_lo, a_lo, bits - 1)


def par(x):
    return bin(x).count("1") & 1


def apply(M, x, d):
    return sum((1 << r) for r in range(d) if par(M[r] & x))


def invertible(M, d):
    rows = list(M)
    rank = 0
    for col in range(d):
        piv = next((r for r in range(rank, d) if rows[r] & (1 << col)), None)
        if piv is None:
            continue
        rows[rank], rows[piv] = rows[piv], rows[rank]
        for r in range(d):
            if r != rank and rows[r] & (1 << col):
                rows[r] ^= rows[rank]
        rank += 1
    return rank == d


def is_auto(M, d):
    n = 1 << d
    piv = {}
    for i in range(1, n):
        Mi = apply(M, i, d)
        for j in range(i + 1, n):
            Mj = apply(M, j, d)
            k = i ^ j
            s = cd_sigma(Mi, Mj, d) * cd_sigma(i, j, d)
            m = (1 << i) | (1 << j) | (1 << k) | (1 if s < 0 else 0)
            while True:
                v = m >> 1
                if v == 0:
                    if m & 1:
                        return False
                    break
                h = v.bit_length() - 1
                if h in piv:
                    m ^= piv[h]
                else:
                    piv[h] = m
                    break
    return True


def autos(d):
    n = 1 << d
    return [list(M) for M in product(range(n), repeat=d) if invertible(M, d) and is_auto(M, d)]


def main():
    oct = autos(3)
    sed = autos(4)
    fix8 = [M for M in sed if apply(M, 8, 4) == 8]
    orb8 = sorted({apply(M, 8, 4) for M in sed})
    seen, orbits = set(), []
    for x in range(1, 16):
        if x in seen:
            continue
        o = sorted({apply(M, x, 4) for M in sed})
        orbits.append(o)
        seen |= set(o)
    print(f"OCT_AUTOS {len(oct)}")
    print(f"SED_AUTOS {len(sed)}")
    print(f"SED_FIX_E8 {len(fix8)}")
    print("ORBIT8 " + ",".join(map(str, orb8)))
    print("ORBITS " + " ".join("{" + ",".join(map(str, o)) + "}" for o in orbits))
    ok = len(oct) == 168 and len(sed) == 168 and len(fix8) == 168 and orb8 == [8]
    print(f"AUT {'OK' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
