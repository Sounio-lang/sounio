#!/usr/bin/env python3
"""The deviation law: what is left after §33.5(B) becomes unconditional.

§18.1 is now a theorem (`section_18_1`, Tier 54), so the descent

    D(n, W) = 8 * D(n-1, W)        for W low, against its fibre reference 8g(W)+1

is a consequence of theorems only (Tier 55 `deviation_descent`, modulo the tr(A^2) fibre-constancy
that the descent takes as a hypothesis).  What the law still needs is the BASE CASE: the value of
D at the level where W stops being low, i.e. n = top(W) + 2.

This probe measures what is true at that base, with the questions the new machinery raises:

C1  the descent itself, D(n,W) = 8 D(n-1,W), for every low label
C2  tr(A^2) really is constant on a g-fibre (the hypothesis Tier 55 takes)
C3  the BASE VALUE D(top(W)+2, W) against the law's prediction -27 * 8^(n-j) * [j,3]_2
C4  does the base value depend on g beyond the PARITY of popcount(g)?  (the open question:
    two labels with the same lsb, the same top bit and the same parity, but different g)
    -- an earlier version of this check keyed on (lsb, top) ALONE and reported violations; that
    was the check being wrong, not the law: the parity half is (A), already a theorem.

C5  IS THERE A HIGH-BRANCH TWIN OF §18.1?  For W with top bit n-2, write W_lo = W - 2^(n-2) and
    ask whether t3(n,W) - 8*t3(n-1,W_lo) - 24*t2(n-1,W_lo) is a constant of the level alone.
    §18.1 is the LOW branch; the base case of the deviation lives on the HIGH one, where no
    recursion is on the books.

C6  THE THIRD INVARIANT (§54.3).  `Ncnt_hi` reads N(high) = 4e^2 - 6e - 2 - 4*N(child): the high
    branch carries a MINUS, i.e. it complements rather than blows up, and a complement's triangle
    count needs a third moment beyond the edge count.  Tested, all REFUTED at n = 7,8,9:
      - exact affine t3(n,W) = a*t3' + b*t2' + c for ANY (a,b,c)             -- none
      - the same plus sum of squared degrees, or cubed degrees, or a signed 2-path count -- none
      - the natural "complement" corrections 1^T A 1, 1^T A^2 1, 1^T A^3 1 and the third moment
        of the signed row sums are all LABEL-INDEPENDENT CONSTANTS here, so they carry no
        information at all and cannot be the missing term.
    What IS true: t3 on the high labels is exactly a function of (g, lsb) -- 0 splits.
"""

import sys
from itertools import combinations

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from zd_v1_III_deviation_probe import A_sig_fast, sign_table_fast  # noqa: E402


def g_of(W):
    return (W & (W - 1)) >> 3


def lsb(W):
    return (W & -W).bit_length() - 1


def top(W):
    return W.bit_length() - 1


def gauss3(j):
    """[j choose 3]_2 -- the number of 3-dim subspaces of F_2^j."""
    if j < 3:
        return 0
    num = (2**j - 1) * (2**j - 2) * (2**j - 4)
    return num // 168


def t3(n, W, S):
    A = A_sig_fast(n, W, S).astype(np.int64)
    return int(np.einsum('ab,bc,ca->', A, A, A))


def t2(n, W, S):
    A = A_sig_fast(n, W, S).astype(np.int64)
    return int(np.trace(A @ A))


def main():
    levels = [int(x) for x in (sys.argv[1:] or ["6", "7", "8", "9"])]
    for n in levels:
        S = sign_table_fast(n)
        h = 1 << (n - 2)
        cache = {}

        def T3(W):
            if W not in cache:
                cache[W] = t3(n, W, S)
            return cache[W]

        c1 = c2 = c3 = c4 = 0
        n1 = n3 = n4 = 0
        base_by_key = {}
        if n >= 7:
            Sm = sign_table_fast(n - 1)
        for W in range(1, 1 << (n - 1)):
            g = g_of(W)
            ref = 8 * g + 1
            if ref >= (1 << (n - 1)):
                continue
            D = T3(W) - T3(ref)
            # C2: tr(A^2) constant on the fibre
            c2 += int(t2(n, W, S) != t2(n, ref, S))
            # C1: the descent, only where BOTH W and ref are low at level n
            if n >= 7 and W < h and ref < h:
                Dm = t3(n - 1, W, Sm) - t3(n - 1, ref, Sm)
                c1 += int(D != 8 * Dm)
                n1 += 1
            # C3/C4: the BASE, i.e. the level where W stops being low
            if top(W) == n - 2:
                j = lsb(W)
                pred = 0 if bin(g).count("1") % 2 == 1 else -27 * 8**(n - j) * gauss3(j)
                c3 += int(D != pred)
                n3 += 1
                base_by_key.setdefault((j, top(W), bin(g).count("1") % 2), set()).add(D)
        # C4: same (lsb, top) but different g -> same base value?
        for key, vals in base_by_key.items():
            n4 += 1
            c4 += int(len(vals) != 1)
        print(f"n={n}: C1 descent D(n,W) = 8 D(n-1,W): {c1} viol / {n1} pairs")
        print(f"      C2 tr(A^2) constant on the g-fibre: {c2} viol")
        print(f"      C3 base value = -27*8^(n-j)*[j,3]_2 (or 0): {c3} viol / {n3} base labels")
        print(f"      C4 base value depends only on (lsb, top, parity), not on g: "
              f"{c4} viol / {n4} keys")
        if n >= 7:
            resid = {}
            for W in range(1, 1 << (n - 1)):
                if top(W) != n - 2:
                    continue
                Wlo = W - (1 << (n - 2))
                if Wlo == 0:
                    continue
                r = T3(W) - 8 * t3(n - 1, Wlo, Sm) - 24 * t2(n - 1, Wlo, Sm)
                resid.setdefault(r, 0)
                resid[r] += 1
            top5 = sorted(resid.items(), key=lambda kv: -kv[1])[:5]
            print(f"      C5 high-branch residual t3(n,W) - 8 t3' - 24 t2' over {sum(resid.values())}"
                  f" labels: {len(resid)} distinct values, most common {top5}")


if __name__ == "__main__":
    main()
