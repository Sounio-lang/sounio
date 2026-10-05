#!/usr/bin/env python3
"""§18.1 decomposed: the low-branch tr(A^3) recursion is a four-term trace expansion.

    t3(n,W) = 8*t3(n-1,W) + 24*t2(n-1,W) - 12*(2^(n-1) - 4)      for W < 2^(n-2)

is not a fitted integer identity. With h = 2^(n-2), the level-n vertex set [1,2^(n-1)) is the
level-(n-1) vertex set [1,h) DOUBLED (a0 and a0+h) plus the single vertex h, and

    A(n,W) = B + E,        B = J2 (x) A'(n-1,W)

so tr(A^3) = tr(B^3) + 3tr(B^2 E) + 3tr(B E^2) + tr(E^3), and each term is one summand of the
recursion:

    tr(B^3)    = 8*t3'                 <- ALGEBRA: tr((J2 (x) A')^3) = tr(J2^3)*tr(A'^3), tr(J2^3)=8
    3tr(B^2 E) = 24*t2'
    3tr(B E^2) = 0
    tr(E^3)    = -24(h-2) = -12(2^(n-1) - 4)      <- THE CONSTANT, label-independent

E is supported on exactly four families, of total size 12(h-2) ordered pairs:
  (i)   the matching a0 <-> a0+h            sign +1
  (ii)  the coset a0 <-> (a0^W)+h           sign -1, cross blocks only
  (iii) the two copies of the isolated vertex a0 = W
  (iv)  the extra vertex h itself

Only (i) and (ii) can contribute to tr(B^2 E) -- B vanishes on the rows of (iii) and (iv), so that
is a proof, not a measurement -- and their contributions are +4t2' and -4*S(W) with
S(W) = sum_a (A'^2)[a, a^W], for which this probe measures the identity S(W) = -t2'.
"""

import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from zd_v1_III_deviation_probe import A_sig_fast, sign_table_fast  # noqa: E402


def blowup(Ap, n):
    h = 1 << (n - 2)
    N = (1 << (n - 1)) - 1
    B = np.zeros((N, N), dtype=np.int64)
    for e in (0, 1):
        for f in (0, 1):
            B[np.ix_(np.arange(1, h) + e * h - 1, np.arange(1, h) + f * h - 1)] = Ap
    return B


def main():
    levels = [int(x) for x in (sys.argv[1:] or ["7", "8", "9"])]
    for n in levels:
        h = 1 << (n - 2)
        Sn = sign_table_fast(n)
        Sm = sign_table_fast(n - 1)
        nlab = h - 1

        # --- C1: the block identity, over EVERY low label
        bad_blk = 0
        n_blk = 0
        msign = set()
        csign = set()
        cblocks = set()
        for W in range(1, h):
            A = A_sig_fast(n, W, Sn).astype(np.int64)
            Ap = A_sig_fast(n - 1, W, Sm).astype(np.int64)
            nz = Ap != 0
            for e in (0, 1):
                for f in (0, 1):
                    blk = A[np.ix_(np.arange(1, h) + e * h - 1, np.arange(1, h) + f * h - 1)]
                    n_blk += int(nz.sum())
                    bad_blk += int(((blk != Ap) & nz).sum())
            for a0 in range(1, h):
                v = int(A[a0 - 1, a0 + h - 1])
                if v:
                    msign.add(v)
                b0 = a0 ^ W
                if 1 <= b0 < h and b0 != a0:
                    for e in (0, 1):
                        for f in (0, 1):
                            v = int(A[a0 + e * h - 1, b0 + f * h - 1])
                            if v:
                                csign.add(v)
                                cblocks.add((e, f))
        print(f"n={n}: C1 block identity A = J2(x)A' wherever A' != 0 -- "
              f"{n_blk} checks over {nlab} labels, {bad_blk} violations")
        print(f"      matching-edge signs {sorted(msign)}; "
              f"coset-edge signs {sorted(csign)} in blocks {sorted(cblocks)}")

        # --- C2: the four-term expansion, term by term
        bad = [0, 0, 0, 0]
        for W in range(1, h):
            A = A_sig_fast(n, W, Sn).astype(np.int64)
            Ap = A_sig_fast(n - 1, W, Sm).astype(np.int64)
            t2p = int(np.count_nonzero(Ap))
            t3p = int(np.trace(Ap @ Ap @ Ap))
            B = blowup(Ap, n)
            E = A - B
            bad[0] += int(np.trace(B @ B @ B)) != 8 * t3p
            bad[1] += 3 * int(np.trace(B @ B @ E)) != 24 * t2p
            bad[2] += int(np.trace(B @ E @ E)) != 0
            bad[3] += int(np.trace(E @ E @ E)) != -24 * (h - 2)
        print(f"      C2  tr(B^3)=8t3' : {bad[0]} | 3tr(B^2E)=24t2' : {bad[1]} | "
              f"tr(BE^2)=0 : {bad[2]} | tr(E^3)=-24(h-2)={-24*(h-2)} : {bad[3]}   "
              f"(over {nlab} labels)")

        # --- C3: the coset 2-path identity that makes 3tr(B^2E) come out to 24t2'
        badS = 0
        H = 1 << (n - 1)
        for W in range(1, H):
            Ap = A_sig_fast(n, W, Sn).astype(np.int64)
            A2 = Ap @ Ap
            s = sum(int(A2[a - 1, (a ^ W) - 1]) for a in range(1, H) if 1 <= (a ^ W) < H)
            if s != -int(np.count_nonzero(Ap)):
                badS += 1
        print(f"      C3  sum_a (A^2)[a, a^W] = -tr(A^2) : {badS} violations over {H-1} labels")

        # --- C4: the recursion itself, for the record
        badr = 0
        for W in range(1, h):
            A = A_sig_fast(n, W, Sn).astype(np.int64)
            Ap = A_sig_fast(n - 1, W, Sm).astype(np.int64)
            if int(np.trace(A @ A @ A)) != (
                8 * int(np.trace(Ap @ Ap @ Ap))
                + 24 * int(np.count_nonzero(Ap))
                - 12 * ((1 << (n - 1)) - 4)
            ):
                badr += 1
        print(f"      C4  §18.1 itself : {badr} violations over {nlab} low labels")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
