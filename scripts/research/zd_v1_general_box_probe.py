#!/usr/bin/env python3
"""The GENERAL high label's box, measured at n = 3..6.

`tri3_fold_high` (Tier 59) reduces the deviation law's base case to a difference of two triangle
sums on ONE index set: for any label with top bit 2^n,

    tr(A^3)(W) = 8 * t3(box_W),      box_W = A_sigma restricted to [1, 2^n)

and at the base level a seam and its fibre reference are both high, so
D(W) = 8 * (t3(box_W) - t3(box_ref)).  `box` is known exactly at W_lo = 0 (it is I - J, Tier 57).
This probe asks what it is otherwise.

C1  entry values, and the count of +1 entries
C2  the coset line l ^ y = W_lo
C3  ⚠ TWO CLAIMS FROM AN EARLIER RUNG, BOTH FALSIFIED HERE -- see the header of §57.3
C4  which box invariants are constant on the g-fibre
C5  the deviation, recovered from the box
"""

import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from zd_v1_III_deviation_probe import A_sig_fast, sign_table_fast  # noqa: E402


def g_of(W):
    return (W & (W - 1)) >> 3


def lsb(W):
    return (W & -W).bit_length() - 1


def gauss3(j):
    return 0 if j < 3 else (2**j - 1) * (2**j - 2) * (2**j - 4) // 168


def dev_pred(N, W):
    """The deviation law.  Parity included -- never re-derive this inline."""
    g = g_of(W)
    if bin(g).count("1") % 2 == 1:
        return 0
    return -27 * 8**(N - lsb(W)) * gauss3(lsb(W))


def main():
    for n in [int(x) for x in (sys.argv[1:] or ["3", "4", "5", "6"])]:
        N = n + 2                      # Lean level n  <->  builder level n+2
        q = 1 << n
        S = sign_table_fast(N)
        IJ = np.eye(q - 1, dtype=np.int64) - np.ones((q - 1, q - 1), dtype=np.int64)
        vals, plus, swapbad, swaptot, classbad, cosetok = set(), set(), 0, 0, 0, 0
        comps = {'#(+1)': {}, '#0': {}, 't2(box)': {}, 't3(box)': {}}
        t3b = {}
        for Wlo in range(1, q):
            W = q + Wlo
            B = A_sig_fast(N, W, S).astype(np.int64)[:q - 1, :q - 1]
            vals |= set(np.unique(B).tolist())
            plus.add(int(np.count_nonzero(B == 1)))
            t3b[W] = int(np.einsum('ab,bc,ca->', B, B, B))
            v = {'#(+1)': int(np.count_nonzero(B == 1)),
                 '#0': int(np.count_nonzero(B == 0)),
                 't2(box)': int(np.trace(B @ B)), 't3(box)': t3b[W]}
            for k in comps:
                comps[k].setdefault(g_of(Wlo), set()).add(v[k])
            byx = {}
            for l in range(1, q):
                for y in range(1, q):
                    if l != y:
                        byx.setdefault(l ^ y, set()).add(int(B[l - 1, y - 1]))
            cosetok += int(byx.get(Wlo) == {1})
            for x, s in byx.items():
                if x != Wlo and s not in ({0, 1}, {-1, 0}):
                    classbad += 1
            for x in range(1, q):
                xp = x ^ Wlo
                if x == Wlo or not (1 <= xp < q):
                    continue
                swaptot += 1
                a, b = byx.get(x), byx.get(xp)
                swapbad += int(not ((a == {0, 1} and b == {-1, 0})
                                    or (a == {-1, 0} and b == {0, 1})))
        print(f"n={n} (builder level {N}), {q-1} high labels, box {q-1}x{q-1}")
        print(f"  C1 entry values {sorted(vals)} | #(+1) takes {len(plus)} value(s): {sorted(plus)}"
              f"   [2(2^n-2) = {2*(q-2)} is only the MINIMUM]")
        print(f"  C2 the coset line l^y = W_lo carries exactly {{1}}: {cosetok}/{q-1} labels")
        print(f"  C3 other l^y in {{0,1}} or {{-1,0}}: {classbad} violations"
              f" | swap x -> x^W_lo exchanges those: {swapbad}/{swaptot}   [both die past n=3]")
        for k, d in comps.items():
            bad = sum(1 for s in d.values() if len(s) > 1)
            print(f"  C4 {k:8s} constant on the g-fibre: {bad}/{len(d)} fibres split")
        bad = tot = 0
        for W, v in t3b.items():
            ref = 8 * g_of(W) + 1
            if ref in t3b:
                tot += 1
                bad += int(8 * (v - t3b[ref]) != dev_pred(N, W))
        print(f"  C5 8*(t3(box_W) - t3(box_ref)) == the deviation law: {bad} viol / {tot}")


if __name__ == "__main__":
    main()
