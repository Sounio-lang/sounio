#!/usr/bin/env python3
"""
CD-tower ZD fibers — L1 REDUCED to a fiber-free statement about the sign cocycle.

L1 is the first of the two sigma-lemmas under (c), the parity-collapse law:
    R_{L_seam}(a,b) = R_{L_fano}(tau a, tau b),   L_seam = Y|H,  tau = swap(0, j),  j = lsb(Y).
The previous rung showed L1 is PARITY-FREE (it holds for odd-weight seams too) and that its real
hypothesis is seam-ness. This rung reduces it.

THE TARGET. Since res <=> P1 == P3 and both are +-1, res <=> Q = +1 with
    Q_L(a,b) = sigma(a,b) sigma(a^L,b^L) sigma(a,b^L) sigma(a^L,b)
the product of sigma over the coset square. L1 is the tau-equivariance of Q for L = Y|H, with
a,b restricted to lo-labels. The reduction replaces it by

  (*)  for every Y = 0 mod 8, j = lsb(Y), tau = swap(0,j):  Q_Y(a,b) = Q_{tau Y}(tau a, tau b)
       for ALL a, b -- no fiber, no top bit, no restriction on a,b.

(*) is a cleaner statement than L1 and it is ONE LEVEL DOWN: L1 at level n with label Y|H
reduces to (*) at level n-1 with label Y.

THE CHAIN, verified link by link (no link is assumed):
  K2  the four branch reductions give, exactly, for b != Y:
          Q_{Y|H}(a,b) = - D1_Y(a,b) * D2_Y(b^Y, a)     [at level n-1]
      where D1_Y(a,b) = sigma(a,b) sigma(a^Y,b) and D2_Y(c,y) = sigma(c,y) sigma(c,y^Y) are
      FIRST differences by Y -- so the second difference splits into two first differences.
  K3  the tau-discrepancies of the two factors CANCEL: e1(a,b,Y) = e2(b^Y,a,Y), where
      e1 = D1_{tauY}(tau a,tau b) * D1_Y(a,b) and e2 is the same for D2.
  K4  equivalently, in a single statement: e1(a,b,Y) = e1(a, b^Y, Y) -- the tau-discrepancy of
      the first difference is invariant under shifting the second argument by Y.
  K1  and K4 regrouped is exactly (*), which is measured directly at levels 5,6,7,8.

WHY THE CANCELLATION IS THE CONTENT (negative controls):
  K5  NEITHER D1 NOR D2 is tau-equivariant on its own -- and their violation counts are
      IDENTICAL. So L1 is a genuine cancellation between the two factors, not a factorwise
      property. Without this, K3 would be trivial.
  K6  A GAP IN AN ATTRACTIVE DERIVATION, recorded. By `antisym` one expects
      D2(c,y,Y) = D1(y,c,Y), which would give K4 from K3 in one line. That identity FAILS on the
      degenerate locus (nonzero violation counts). K4 is therefore measured directly, not
      derived that way.
  K7  CONTROL, AND A CORRECTED READING. With a MISMATCHED tau -- j frozen at 3 instead of
      j = lsb(Y) -- the equivariance fails in bulk. That is all this clause measures.
      **The first version of this clause concluded "(*) is a statement about seam labels".
      THAT WAS WRONG**: the failure comes from using the wrong tau, not from Y being a non-seam.
  K8  THE SEAM HYPOTHESIS IS NOT NEEDED. With the matching tau, i.e. j = lsb(Y), (*) holds for
      EVERY Y != 0, seam or not (levels 5,6,7, zero violations). So (*) is strictly more general
      than L1 needs, and the seam condition can be dropped from its statement.

  K0  PARITY. The builders reproduce the in-tree sign_table entrywise.

NOT CLAIMED. L1 is NOT proven, and neither is (*). This is a reduction: L1 (fiber-bound,
lo-labels only, level n) is replaced by (*) (fiber-free, full range, level n-1), verified at
four levels. The b = Y boundary of K2 is excluded from that clause and handled nowhere here;
L1 itself was verified including it by the previous rung. Nothing here is Lean-proven -- though
(*) has the same induction shape as A4_sub, which IS Lean-proven forall n, and that toolkit is
in-tree.

Verdict L1_REDUCED_TO_SEAM_TAU_EQUIVARIANCE_OF_Q__NOT_PROVEN.
Numerical certificate over an exact integer sign table; D3 respected.
"""
import os
import random
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PRIOR = os.path.join(HERE, "cd_tower_zd_fiber_spectral_forall_n_progress_contract.py")
_src = open(PRIOR).read()
exec(_src.split("def main()")[0].split("from collections import defaultdict")[1])  # noqa: S102


def sign_table_fast(n):
    S = np.ones((1, 1), dtype=np.int8)
    for b in range(1, n + 1):
        h = 1 << (b - 1)
        P = S
        T = np.empty((2 * h, 2 * h), dtype=np.int8)
        T[:h, :h] = P
        T[:h, h:] = P.T
        blk = -P.copy(); blk[:, 0] = P[:, 0]
        T[h:, :h] = blk
        blk2 = P.T.copy(); blk2[:, 0] = -P.T[:, 0]
        T[h:, h:] = blk2
        S = T
    S[0, :] = 1
    S[:, 0] = 1
    return S


def sw(x, j):
    b0, bj = x & 1, (x >> j) & 1
    return x ^ (1 | (1 << j)) if b0 != bj else x


def main():
    print("=" * 78)
    print("CD-tower ZD fibers — L1 REDUCED to (*): seam-Y tau-equivariance of Q, fiber-free")
    print("=" * 78)
    ok = {}

    k0 = True
    for n in (6, 7):
        if not np.array_equal(sign_table(n), sign_table_fast(n)):
            k0 = False
    ok["K0"] = k0
    print(f"K0_PARITY   sign_table_fast == in-tree sign_table entrywise (n=6,7) "
          f"{'OK' if k0 else 'FAIL'}")

    tabs = {m: sign_table_fast(m).astype(np.int64) for m in (5, 6, 7, 8)}

    def Q(S, a, b, L):
        return int(S[a, b] * S[a ^ L, b ^ L] * S[a, b ^ L] * S[a ^ L, b])

    # ---- K1  (*) itself -------------------------------------------------------------------
    k1 = True
    for m in (5, 6, 7, 8):
        S, M = tabs[m], 1 << m
        bad = tot = 0
        for Y in range(8, M, 8):
            j = (Y & -Y).bit_length() - 1
            tY = sw(Y, j)
            for a in range(M):
                for b in range(M):
                    tot += 1
                    if Q(S, sw(a, j), sw(b, j), tY) != Q(S, a, b, Y):
                        bad += 1
        k1 = k1 and bad == 0
        print(f"K1_STAR     level {m}: (*) Q_Y(a,b) == Q_tauY(tau a,tau b), seam Y, FULL range: "
              f"violations {bad}/{tot}  {'OK' if bad == 0 else 'FAIL'}")
    ok["K1"] = k1

    # ---- K2  the branch reduction is exact -------------------------------------------------
    k2 = True
    for n in (6, 7, 8):
        Sn, Sm, H = tabs[n], tabs[n - 1], 1 << (n - 1)
        bad = tot = 0
        for Y in range(8, H, 8):
            L = Y | H
            for a in range(1, H):
                for b in range(1, H):
                    if b ^ Y == 0:
                        continue
                    D1 = int(Sm[a, b] * Sm[a ^ Y, b])
                    D2 = int(Sm[b ^ Y, a ^ Y] * Sm[b ^ Y, a])
                    tot += 1
                    if Q(Sn, a, b, L) != -D1 * D2:
                        bad += 1
        k2 = k2 and bad == 0
        print(f"K2_BRANCH   n={n}: Q_(Y|H)(a,b) == -D1_Y(a,b)*D2_Y(b^Y,a) exactly (b != Y): "
              f"violations {bad}/{tot}  {'OK' if bad == 0 else 'FAIL'}")
    ok["K2"] = k2

    # ---- K3 cancellation, K4 sharpened form, K5/K6 controls --------------------------------
    k3 = k4 = True
    k5 = k6 = True
    for m in (5, 6, 7):
        S, M = tabs[m], 1 << m
        D1 = lambda a, b, Y: int(S[a, b] * S[a ^ Y, b])          # noqa: E731
        D2 = lambda c, y, Y: int(S[c, y] * S[c, y ^ Y])          # noqa: E731
        b3 = b4 = t34 = 0
        v1 = v2 = tsep = 0
        b6 = t6 = 0
        for Y in range(8, M, 8):
            j = (Y & -Y).bit_length() - 1
            tY = sw(Y, j)
            for a in range(M):
                for b in range(M):
                    e1 = D1(sw(a, j), sw(b, j), tY) * D1(a, b, Y)
                    c = b ^ Y
                    e2 = D2(sw(c, j), sw(a, j), tY) * D2(c, a, Y)
                    t34 += 1
                    if e1 != e2:
                        b3 += 1
                    e1s = D1(sw(a, j), sw(b ^ Y, j), tY) * D1(a, b ^ Y, Y)
                    if e1 != e1s:
                        b4 += 1
                    tsep += 1
                    if D1(sw(a, j), sw(b, j), tY) != D1(a, b, Y):
                        v1 += 1
                    if D2(sw(a, j), sw(b, j), tY) != D2(a, b, Y):
                        v2 += 1
                    t6 += 1
                    if D2(a, b, Y) != D1(b, a, Y):
                        b6 += 1
        k3 = k3 and b3 == 0
        k4 = k4 and b4 == 0
        k5 = k5 and v1 > 0 and v2 > 0 and v1 == v2
        k6 = k6 and b6 > 0
        print(f"K3_CANCEL   level {m}: e1(a,b,Y) == e2(b^Y,a,Y): violations {b3}/{t34} "
              f"{'OK' if b3 == 0 else 'FAIL'}")
        print(f"K4_SHARP    level {m}: e1(a,b,Y) == e1(a,b^Y,Y): violations {b4}/{t34} "
              f"{'OK' if b4 == 0 else 'FAIL'}")
        print(f"K5_NOTSEP   level {m}: D1 alone NOT tau-equivariant ({v1}/{tsep}), D2 alone NOT "
              f"either ({v2}/{tsep}), counts identical={v1 == v2} => the cancellation is the "
              f"content {'OK' if v1 > 0 and v1 == v2 else 'FAIL'}")
        print(f"K6_GAP      level {m}: the tempting identity D2(c,y,Y) == D1(y,c,Y) FAILS on the "
              f"degenerate locus ({b6}/{t6}) => K4 is measured, not derived from antisym "
              f"{'OK' if b6 > 0 else 'FAIL'}")
    ok["K3"], ok["K4"], ok["K5"], ok["K6"] = k3, k4, k5, k6

    # ---- K7 null control: (*) fails for non-seam Y ------------------------------------------
    k7 = True
    for m in (5, 6):
        S, M = tabs[m], 1 << m
        bad = tot = 0
        for Y in range(1, M):
            if Y % 8 == 0:
                continue
            j = 3 if m > 3 else 1
            tY = sw(Y, j)
            for a in range(M):
                for b in range(M):
                    tot += 1
                    if Q(S, sw(a, j), sw(b, j), tY) != Q(S, a, b, Y):
                        bad += 1
        if bad == 0:
            k7 = False
        print(f"K7_NULL     level {m}: with NON-seam Y, (*) fails {bad}/{tot} => (*) is a "
              f"MISMATCHED tau (j frozen at 3) breaks it -- that is ALL this measures "
              f"{'OK' if bad else 'FAIL'}")
    ok["K7"] = k7

    # ---- K8 the seam hypothesis is not needed ---------------------------------------------
    k8 = True
    for m in (5, 6, 7):
        S, M = tabs[m], 1 << m
        bad = tot = 0
        for Y in range(1, M):
            j = (Y & -Y).bit_length() - 1
            if j == 0:
                continue                      # tau = identity, vacuous
            tY = sw(Y, j)
            for a in range(M):
                for b in range(M):
                    tot += 1
                    if Q(S, sw(a, j), sw(b, j), tY) != Q(S, a, b, Y):
                        bad += 1
        k8 = k8 and bad == 0
        print(f"K8_GENERAL  level {m}: (*) with the MATCHING tau (j = lsb(Y)) holds for EVERY "
              f"Y != 0, seam or not: violations {bad}/{tot} {'OK' if bad == 0 else 'FAIL'} "
              f"=> the seam hypothesis is NOT needed; K7's first reading was wrong")
    ok["K8"] = k8

    print("=" * 78)
    if all(ok.values()):
        print("CD_TOWER_ZDL1_VERDICT L1_REDUCED_TO_SEAM_TAU_EQUIVARIANCE_OF_Q__NOT_PROVEN")
        print("CD_TOWER_ZDL1_NOTE L1 (fiber-bound, lo-labels, level n) is replaced by (*) "
              "(fiber-free, FULL range, level n-1): for seam Y, Q_Y(a,b) = Q_tauY(tau a,tau b). "
              "The chain is verified link by link -- the four branch reductions split the second "
              "difference Q into two FIRST differences exactly (K2), whose tau-discrepancies "
              "cancel (K3), equivalently the discrepancy of D1 is invariant under shifting the "
              "second argument by Y (K4), which regrouped IS (*) (K1, levels 5..8). The "
              "cancellation is the whole content: neither D1 nor D2 is tau-equivariant alone, "
              "with IDENTICAL violation counts (K5). An attractive one-line derivation of K4 "
              "from antisym is recorded as HAVING A GAP -- D2(c,y,Y) = D1(y,c,Y) fails on the "
              "degenerate locus (K6). K7 only shows a MISMATCHED tau breaks it -- its first reading, that (*) is about seam labels, was WRONG -- and K8 shows (*) holds for EVERY Y != 0 with the matching j = lsb(Y), so the seam hypothesis is not needed at all. L1 IS NOT PROVEN and "
              "neither is (*); but (*) has the same induction shape as A4_sub, which is "
              "Lean-proven forall n, and that toolkit is in-tree. Numerical certificate; D3")
        return 0
    print("CD_TOWER_ZDL1_VERDICT INCOMPLETE  failing=" +
          ",".join(k for k, v in ok.items() if not v))
    return 1


if __name__ == "__main__":
    t0 = time.time()
    rc = main()
    print(f"[{time.time() - t0:.1f}s]", file=sys.stderr)
    raise SystemExit(rc)
