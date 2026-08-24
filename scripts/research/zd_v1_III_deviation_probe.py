#!/usr/bin/env python3
"""(III) probe: measure the WITHIN-FIBRE deviation of tr(A^3), not tr(A^3) itself.

Reformulation this probe tests. §30 proved `tr(A^2)` is injective in `g(W) = (W&(W-1))>>3`, so
the `tr(A^2)`-fibre of `y` is exactly the labels with `g(W) = y`:

    Fano orbit y : W = 8y+r, r = 1..7          (lsb(W) in {0,1,2}, one GL(3,2) orbit)
    seams        : W = 8(y + 2^i), i < lsb(y)  (lsb(W) = i+3)

so **inside a fibre a class is indexed by `lsb(W)`** -- which is precisely §19's third
level-quantity. Every seam in the fibre of `y` has popcount `popcount(y)+1`, so the whole fibre
has ONE parity, and:

    popcount(y) odd  -> every seam merges with the Fano class   ((c), PROVEN)
    popcount(y) even -> (III) must separate them

Hence (III) is a statement about the DEVIATION

    D(W) = tr(A^3)(W) - tr(A^3)(8*g(W)+1)

namely: D = 0 iff popcount(g) is odd, and D is injective in lsb(W) otherwise. The lane has been
chasing a closed form for tr(A^3) itself (W24-W28, not found); (III) only needs D.
"""

import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def sign_table_fast(n):
    S = np.ones((1, 1), dtype=np.int8)
    for b in range(1, n + 1):
        h = 1 << (b - 1)
        P = S
        T = np.empty((2 * h, 2 * h), dtype=np.int8)
        T[:h, :h] = P
        T[:h, h:] = P.T
        blk = -P.copy()
        blk[:, 0] = P[:, 0]
        T[h:, :h] = blk
        blk2 = P.T.copy()
        blk2[:, 0] = -P.T[:, 0]
        T[h:, h:] = blk2
        S = T
    S[0, :] = 1
    S[:, 0] = 1
    return S


def A_sig_fast(n, Llo, S):
    H = 1 << (n - 1)
    L = Llo | H
    los = np.arange(1, H)
    hi = los ^ L
    SLL = S[np.ix_(los, los)].astype(np.int16)
    SHH = S[np.ix_(hi, hi)].astype(np.int16)
    SLH = S[np.ix_(los, hi)].astype(np.int16)
    SHL = S[np.ix_(hi, los)].astype(np.int16)
    P1 = SLL * SHH
    P3 = SLH * SHL
    res = (P1 == P1.T) & (P3 == P3.T) & (P1 == P3)
    A = np.where(res, -P1, 0).astype(np.int8)
    np.fill_diagonal(A, 0)
    return A


def traces23(A):
    F = A.astype(np.float64)
    t2 = int(np.count_nonzero(A))
    t3 = int(round(float(np.sum(F * (F @ F).T))))
    return t2, t3


def g_of(W):
    return (W & (W - 1)) >> 3


def lsb(W):
    return (W & -W).bit_length() - 1


def sweep(n):
    S = sign_table_fast(n)
    t2 = {}
    t3 = {}
    for Llo in range(1, 1 << (n - 1)):
        A = A_sig_fast(n, Llo, S)
        t2[Llo], t3[Llo] = traces23(A)
    return t2, t3


def main():
    levels = [int(x) for x in (sys.argv[1:] or ["6", "7", "8", "9", "10"])]
    for n in levels:
        t2, t3 = sweep(n)
        labels = list(t2)

        # 0. sanity: tr(A^2) constant on the g-fibre, injective in g  (§30 + Tier 36)
        byg = defaultdict(set)
        for W in labels:
            byg[g_of(W)].add(t2[W])
        nonconst = [y for y, v in byg.items() if len(v) > 1]
        vals = {next(iter(v)) for v in byg.values()}
        print(f"n={n}: {len(labels)} labels, {len(byg)} g-fibres; "
              f"t2 non-constant on {len(nonconst)} fibres; #distinct t2 = {len(vals)}")

        # 1. the deviation
        dev = {}
        for W in labels:
            ref = 8 * g_of(W) + 1
            dev[W] = t3[W] - t3.get(ref, t3[W] if W == ref else None)

        # 2. D = 0 iff popcount(g) odd  (equivalently: the seam's own weight is even)
        bad = 0
        for W in labels:
            y = g_of(W)
            expected_zero = (bin(y).count("1") % 2 == 1) or (W & 7) != 0
            if (dev[W] == 0) != expected_zero:
                bad += 1
                if bad < 4:
                    print(f"  DEV-PARITY FAIL W={W} y={y} lsb={lsb(W)} D={dev[W]}")
        print(f"  D=0 <-> (Fano or popcount(g) odd): {bad} mismatches")

        # 3. injectivity of D in lsb inside each fibre with popcount(g) even
        coll = 0
        nfib = 0
        for y, _ in byg.items():
            if bin(y).count("1") % 2 == 1:
                continue
            members = [W for W in labels if g_of(W) == y and (W & 7) == 0]
            if not members:
                continue
            nfib += 1
            seen = {0: "Fano"}
            for W in members:
                if dev[W] in seen:
                    coll += 1
                    if coll < 4:
                        print(f"  III COLLISION y={y} W={W} lsb={lsb(W)} "
                              f"D={dev[W]} vs {seen[dev[W]]}")
                seen[dev[W]] = f"W={W}"
        print(f"  (III) D injective in lsb on {nfib} even-popcount fibres: {coll} collisions")

        # 4. structure hunt: is D a function of (lsb, n) alone? of (lsb, popcount(g), n)?
        f_lsb = defaultdict(set)
        f_lsb_pc = defaultdict(set)
        f_lsb_t2 = defaultdict(set)
        for W in labels:
            y = g_of(W)
            if (W & 7) != 0 or bin(y).count("1") % 2 == 1:
                continue
            f_lsb[lsb(W)].add(dev[W])
            f_lsb_pc[(lsb(W), bin(y).count("1"))].add(dev[W])
            f_lsb_t2[(lsb(W), t2[W])].add(dev[W])
        print(f"  D determined by lsb alone?           "
              f"{'YES' if all(len(v) == 1 for v in f_lsb.values()) else 'NO'} "
              f"({max((len(v) for v in f_lsb.values()), default=0)} values max)")
        print(f"  D determined by (lsb, popcount g)?   "
              f"{'YES' if all(len(v) == 1 for v in f_lsb_pc.values()) else 'NO'} "
              f"({max((len(v) for v in f_lsb_pc.values()), default=0)} values max)")
        print(f"  D determined by (lsb, t2)?           "
              f"{'YES' if all(len(v) == 1 for v in f_lsb_t2.values()) else 'NO'} "
              f"({max((len(v) for v in f_lsb_t2.values()), default=0)} values max)")

        # 5. THE CLOSED FORM, and the full deviation law
        #    delta(n,j) = -(9/56) * u^3 * (2^j-1)(2^j-2)(2^j-4),  u = 2^(n-j)
        #    t3(W) = t3(8*g(W)+1) + [popcount(g(W)) even] * delta(n, lsb(W))   for seams
        def delta(n, j):
            u = 1 << (n - j)
            num = 9 * u**3 * ((1 << j) - 1) * ((1 << j) - 2) * ((1 << j) - 4)
            assert num % 56 == 0, (n, j)
            return -(num // 56)

        badlaw = 0
        nlaw = 0
        for W in labels:
            y = g_of(W)
            if (W & 7) != 0:
                # Fano label: must agree with its own representative (GL-constancy)
                if t3[W] != t3[8 * y + 1]:
                    badlaw += 1
                    if badlaw < 4:
                        print(f"  LAW FAIL (Fano) W={W} y={y}")
                nlaw += 1
                continue
            nlaw += 1
            d = delta(n, lsb(W)) if bin(y).count("1") % 2 == 0 else 0
            if t3[W] != t3[8 * y + 1] + d:
                badlaw += 1
                if badlaw < 4:
                    print(f"  LAW FAIL W={W} y={y} lsb={lsb(W)} "
                          f"got {t3[W] - t3[8*y+1]} want {d}")
        print(f"  DEVIATION LAW t3(W) = t3(8g+1) + [pc(g) even]*delta(n,lsb): "
              f"{nlaw} labels, {badlaw} mismatches")

        # 6. print the deviation table for the y = 0 fibre (weight-1 seams)
        row = []
        for W in sorted(labels):
            if (W & 7) == 0 and g_of(W) == 0:
                row.append((lsb(W), dev[W]))
        print(f"  y=0 fibre (weight-1 seams), (lsb, D): {row}")

        # 7. the reformulation must reproduce the class count 3*2^(n-5)
        cnt = 0
        for y in range(1 << (n - 4)):
            if bin(y).count("1") % 2 == 1:
                cnt += 1
                continue
            top = (n - 4) if y == 0 else lsb(y)
            cnt += 1 + top
        print(f"  class count from the reformulation: {cnt} vs 3*2^(n-5) = "
              f"{3 * 2**(n-5)}  {'OK' if cnt == 3 * 2**(n-5) else 'MISMATCH'}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
