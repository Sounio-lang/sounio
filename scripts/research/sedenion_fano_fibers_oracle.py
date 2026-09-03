#!/usr/bin/env python3
"""Oracle: the 7 sedenion ZD fibers ARE the Fano plane PG(2,2), and Aut(S) is its collineation group
(Frente B, vector 1). The 168 signed automorphisms fix e8, so they act on the fiber labels
L = lo^hi in {9..15} = {8^t : t in 1..7} via t -> M(t) on the lower 3 bits = F_2^3\{0} = the 7 Fano
points. Verifies: faithful (168 fiber-permutations), transitive, permutes the 7 Fano lines.

Output:
  AUTOS <n>         signed automorphisms (168)
  FIBER_PERMS <n>   distinct permutations of the 7 fibers (168 = faithful)
  ORBIT1 <n>        orbit size of fiber 1 (7 = transitive)
  FANO_LINES_OK <bool>  group permutes the 7 Fano lines
  FANO <OK|FAIL>
"""
from itertools import product


def cd_sigma(a, b, bits):
    if a == 0 or b == 0: return 1
    if bits <= 1: return -1
    half = 1 << (bits - 1); aH = a >= half; bH = b >= half; aL = a & (half - 1); bL = b & (half - 1)
    if not aH and not bH: return cd_sigma(aL, bL, bits - 1)
    if not aH and bH: return cd_sigma(bL, aL, bits - 1)
    if aH and not bH: return cd_sigma(aL, bL, bits - 1) if bL == 0 else -cd_sigma(aL, bL, bits - 1)
    return -cd_sigma(bL, aL, bits - 1) if bL == 0 else cd_sigma(bL, aL, bits - 1)
def par(x): return bin(x).count("1") & 1
def apply(M, x, d): return sum((1 << r) for r in range(d) if par(M[r] & x))
def invertible(M, d):
    rows = list(M); rank = 0
    for col in range(d):
        piv = next((r for r in range(rank, d) if rows[r] & (1 << col)), None)
        if piv is None: continue
        rows[rank], rows[piv] = rows[piv], rows[rank]
        for r in range(d):
            if r != rank and rows[r] & (1 << col): rows[r] ^= rows[rank]
        rank += 1
    return rank == d
def is_auto(M, d):
    n = 1 << d; piv = {}
    for i in range(1, n):
        Mi = apply(M, i, d)
        for j in range(i + 1, n):
            Mj = apply(M, j, d); k = i ^ j
            s = cd_sigma(Mi, Mj, d) * cd_sigma(i, j, d)
            m = (1 << i) | (1 << j) | (1 << k) | (1 if s < 0 else 0)
            while True:
                v = m >> 1
                if v == 0:
                    if m & 1: return False
                    break
                h = v.bit_length() - 1
                if h in piv: m ^= piv[h]
                else: piv[h] = m; break
    return True


def main():
    autos = [list(M) for M in product(range(16), repeat=4) if invertible(M, 4) and is_auto(M, 4)]
    fps = set(tuple(apply(M, 8 | t, 4) & 7 for t in range(1, 8)) for M in autos)
    orb1 = set(apply(M, 8 | 1, 4) & 7 for M in autos)
    lines = set(frozenset({a, b, a ^ b}) for a in range(1, 8) for b in range(1, 8) if a != b)
    fano_ok = all(frozenset((apply(M, 8 | t, 4) & 7) for t in l) in lines for M in autos for l in lines)
    print(f"AUTOS {len(autos)}")
    print(f"FIBER_PERMS {len(fps)}")
    print(f"ORBIT1 {len(orb1)}")
    print(f"FANO_LINES_OK {fano_ok}")
    ok = len(autos) == 168 and len(fps) == 168 and len(orb1) == 7 and fano_ok
    print(f"FANO {'OK' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
