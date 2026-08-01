#!/usr/bin/env python3
"""
CD-tower ZD fibers — L2 REDUCED to a fiber-free statement about the tau-discrepancy of sigma.

L2 is the second sigma-lemma under (c), the parity-collapse law. The previous rung wrote its
switching function in CLOSED FORM (lambda(a) = +-(-1)^{p_j(a)}, p_j = parity of the bits below
j) and thereby bypassed the cycle-space wall. It left L2 measured, on the fiber, at level n.

THIS RUNG DOES TO L2 WHAT THE l1_reduction RUNG DID TO L1: drops the fiber and goes one level
down. Write

    g(x,y) = sigma(tau x, tau y) * sigma(x,y)          -- the tau-DISCREPANCY of the cocycle

Then, with j = lsb(Y):

  (diamond)   Qgen'(Y,a,b) = -1   ==>   g(a,b) * g(a^Y, b^Y) = (-1)^{p_j(a) + p_j(b)}

for even-weight Y, a != 0, b != 0, b != Y. No fiber, no top bit, one level down -- exactly the
shape (*) has for L1. Measured at levels 5, 6, 7 (N6), zero violations.

  N1  g is SYMMETRIC, unconditionally. This is what removes the argument swap that the raw
      reduction produces, and it is NOT new work: g(x,y) = g(y,x) is equivalent to
      chi(tau x, tau y) = chi(x,y) for the commutation sign chi(x,y) = sigma(x,y)sigma(y,x),
      which is `chi_tau` in formal/lean4/SounioZDFiberAntisym.lean -- PROVEN forall n.
  N2  g DOES NOT FACTOR. The rectangle test g(x,y)g(x,y0)g(x0,y)g(x0,y0) = 1 fails in bulk, so
      g is not mu(x)nu(y). The coboundary in (diamond) is created by the PAIRING along Y, not
      inherited from g. This is L2's analogue of K5: the cancellation is the whole content.
      Had g factored, L2 would have followed in one line -- and a lambda would then exist for
      ODD-weight Y too, contradicting the triangle obstruction. The probe was run because it
      would have been cheap to be right; it came back negative.
  N3  g is F2-BILINEAR ONLY FOR j <= 2. So the R21-style "find the F2-linear identity" route,
      which closed the ZD locality lemma, is WALLED here for general j.
  N4  THE REDUCTION IS EXACT. The fiber-level discrepancy equals the reduced product
      entrywise -- 0 violations at three levels -- via the R_ll / R_uu branch reductions.
  N5  PIN, AND IT CAUGHT AN ERROR. The reduced resonance predicate is the PROVEN Lean lemma
      Qred_hi_ll: Qgen(W + 2^(m+1), a, b, m+2) = - Qgen'(W, a, b, m+1). NOTE THE MINUS SIGN.
      The first version of this rung dropped it, and the failure locus then came out
      contradicting N7's cross-tab. The clause pins the Lean statement to the measured object.
  N7  THE RESONANCE HYPOTHESIS IS ESSENTIAL, not an artifact of where the previous rung
      happened to define `disc`. Unrestricted, (diamond) FAILS; and every failure is OFF
      resonance, none on it. So L2 is genuinely a statement on the resonance graph -- unlike
      (*), which is unrestricted. That is the structural difference between the two lemmas.
  N8  NULL CONTROLS. Odd-weight Y must fail (it does), and a perturbed mask must fail (it does).
  N0  PARITY. The builder reproduces the in-tree sign_table entrywise.

NOT CLAIMED. L2 is NOT proven, and neither is (diamond). This is a reduction, verified at three
levels, plus one already-proven ingredient (N1). (c) is unchanged in status: its (*) leg is
discharged in Lean, its L2 leg is this. Numerical certificate; D3.

Verdict L2_REDUCED_TO_FIBER_FREE_DIAMOND__NOT_PROVEN.
"""
import os
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
    """tau = swap(bit 0, bit j). Transcribed from SounioZDFiberAntisym.lean's `tau`."""
    return x ^ (1 | (1 << j)) if (x & 1) != ((x >> j) & 1) else x


def pj(x, j):
    return bin(x & ((1 << j) - 1)).count("1") & 1


def gmat(S, N, j):
    t = np.array([sw(x, j) for x in range(N)])
    return S[np.ix_(t, t)] * S


def even_seams(N):
    for Y in range(1, N):
        j = (Y & -Y).bit_length() - 1
        if j and bin(Y).count("1") % 2 == 0:
            yield Y, j


def main():
    print("=" * 78)
    print("CD-tower ZD fibers — L2 REDUCED to (diamond): fiber-free tau-discrepancy of sigma")
    print("=" * 78)
    ok = {}

    n0 = all(np.array_equal(sign_table(n), sign_table_fast(n)) for n in (5, 6))
    ok["N0"] = n0
    print(f"N0_PARITY   sign_table_fast == in-tree sign_table entrywise (n=5,6) "
          f"{'OK' if n0 else 'FAIL'}")

    # ---- N1  g is symmetric ---------------------------------------------------------------
    n1 = True
    n1_n = 0
    for n in (5, 6, 7):
        S = sign_table_fast(n).astype(np.int64)
        N = 1 << n
        for j in range(1, n):
            g = gmat(S, N, j)
            n1 = n1 and not (g != g.T).any()
            n1_n += N * N
    ok["N1"] = n1
    print(f"N1_SYMM     g(x,y) = g(y,x) unconditionally ({n1_n} entries, levels 5,6,7, every j) "
          f"{'OK' if n1 else 'FAIL'} -- this is chi_tau, PROVEN forall n in Lean; it is what "
          f"removes the argument swap the raw reduction produces")

    # ---- N2  g does NOT factor ------------------------------------------------------------
    n2_bad = n2_tot = 0
    for n in (5, 6, 7):
        S = sign_table_fast(n).astype(np.int64)
        N = 1 << n
        for j in range(1, n):
            g = gmat(S, N, j)
            R = g * g[:, [1]] * g[[1], :] * g[1, 1]
            n2_bad += int((R != 1).sum())
            n2_tot += N * N
    n2 = n2_bad > 0
    ok["N2"] = n2
    print(f"N2_NOFACTOR g is NOT mu(x)nu(y): rectangle test fails {n2_bad}/{n2_tot} "
          f"{'OK' if n2 else 'FAIL'} -- so the coboundary is created by the PAIRING along Y, "
          f"not inherited. Had g factored, L2 would follow in one line AND a lambda would exist "
          f"for odd-weight Y, contradicting the triangle obstruction")

    # ---- N3  bilinearity only for small j -------------------------------------------------
    n3_rows = []
    for n in (5, 6):
        S = sign_table_fast(n).astype(np.int64)
        N = 1 << n
        I = np.arange(N)
        for j in range(1, n):
            g = gmat(S, N, j)
            bad = sum(int((g[I ^ x2, :] != g * g[[x2], :]).sum()) for x2 in range(N))
            n3_rows.append((n, j, bad))
    n3 = (all(b == 0 for n_, j_, b in n3_rows if j_ <= 2)
          and all(b > 0 for n_, j_, b in n3_rows if j_ >= 3))
    ok["N3"] = n3
    print(f"N3_BILIN    g is F2-additive in each argument for j <= 2 and NOT for j >= 3 "
          f"{'OK' if n3 else 'FAIL'} -- {'; '.join(f'n={a} j={b}: {c}' for a, b, c in n3_rows)} "
          f"=> the R21-style F2-linear route is WALLED for general j")

    # ---- N4/N5/N6/N7  the reduction, the pin, (diamond), and the hypothesis ---------------
    n4 = n5 = n6 = True
    n4_n = n5_n = n6_n = 0
    n7_bad = n7_offres = n7_tot = 0
    for n in (5, 6, 7):
        Sn = sign_table_fast(n + 1).astype(np.int64)
        S = sign_table_fast(n).astype(np.int64)
        N = 1 << n
        H = 1 << n
        I = np.arange(N)
        J = np.arange(1 << (n + 1))
        A, B = np.meshgrid(I, I, indexing="ij")
        for Y, j in even_seams(N):
            L = Y | H
            t = np.array([sw(x, j) for x in range(N)])
            P1 = (Sn * Sn[np.ix_(J ^ L, J ^ L)])[:N, :N]
            P1f = (Sn[np.ix_(t2 := np.array([sw(x, j) for x in range(1 << (n + 1))]),
                             t2)]
                   * Sn[np.ix_(t2[J ^ L], t2[J ^ L])])[:N, :N]
            D = P1f * P1
            Qfib = (Sn * Sn[np.ix_(J ^ L, J ^ L)] * Sn[np.ix_(J, J ^ L)]
                    * Sn[np.ix_(J ^ L, J)])[:N, :N]
            Qp = S * S[B ^ Y, A ^ Y] * S[B ^ Y, A] * S[A ^ Y, B]
            g = gmat(S, N, j)
            Dred = g * g[B ^ Y, A ^ Y]              # raw reduction: arguments SWAPPED
            Dsym = g * g[A ^ Y, B ^ Y]              # after N1
            p = np.array([(-1) ** pj(x, j) for x in range(N)])
            T = np.outer(p, p)

            side = (B != 0) & ((B ^ Y) != 0)        # Qred_hi_ll's side conditions
            dom = side & (A != 0) & (B != Y)
            n4 = n4 and not (D[dom] != Dred[dom]).any() and not (Dred[dom] != Dsym[dom]).any()
            n4_n += int(dom.sum())
            n5 = n5 and not (Qfib[side] != -Qp[side]).any()
            n5_n += int(side.sum())
            hyp = dom & (Qp == -1)
            n6 = n6 and not (Dsym[hyp] != T[hyp]).any()
            n6_n += int(hyp.sum())
            bad = dom & (Dsym != T)
            n7_bad += int(bad.sum())
            n7_offres += int((bad & (Qfib == 1)).sum())
            n7_tot += int(dom.sum())
    ok["N4"], ok["N5"], ok["N6"] = n4, n5, n6
    print(f"N4_REDUCE   fiber discrepancy == g(a,b)*g(b^Y,a^Y) == g(a,b)*g(a^Y,b^Y) entrywise "
          f"({n4_n} checks, levels 5,6,7) {'OK' if n4 else 'FAIL'} -- the fiber and the top bit "
          f"are GONE; the second equality is N1")
    print(f"N5_PIN      Qgen(Y+H,a,b,n+1) == -Qgen'(Y,a,b,n), the Lean lemma Qred_hi_ll, "
          f"({n5_n} checks) {'OK' if n5 else 'FAIL'} -- NOTE THE MINUS SIGN: the first draft of "
          f"this rung dropped it and N7's cross-tab came out self-contradictory")
    print(f"N6_DIAMOND  (diamond): Qgen'(Y,a,b) = -1 ==> g(a,b)g(a^Y,b^Y) = (-1)^(p_j a + p_j b) "
          f"({n6_n} checks, levels 5,6,7) {'OK' if n6 else 'FAIL'}")
    n7 = (n7_bad > 0) and (n7_offres == 0)
    ok["N7"] = n7
    print(f"N7_HYPNEED  UNRESTRICTED (diamond) FAILS ({n7_bad}/{n7_tot}) and EVERY failure is "
          f"off resonance ({n7_offres} on it) {'OK' if n7 else 'FAIL'} -- the resonance "
          f"hypothesis is essential, not an artifact of where `disc` was defined. This is the "
          f"structural difference from (*), which is unrestricted")

    # ---- N8  null controls ----------------------------------------------------------------
    odd_bad = odd_tot = mask_bad = mask_tot = 0
    for n in (5, 6):
        S = sign_table_fast(n).astype(np.int64)
        N = 1 << n
        I = np.arange(N)
        A, B = np.meshgrid(I, I, indexing="ij")
        for Y in range(1, N):
            j = (Y & -Y).bit_length() - 1
            if not j:
                continue
            g = gmat(S, N, j)
            G = g * g[A ^ Y, B ^ Y]
            Qp = S * S[B ^ Y, A ^ Y] * S[B ^ Y, A] * S[A ^ Y, B]
            dom = (A != 0) & (B != 0) & (B != Y) & ((B ^ Y) != 0) & (Qp == -1)
            p = np.array([(-1) ** pj(x, j) for x in range(N)])
            if bin(Y).count("1") % 2:
                odd_bad += int((G[dom] != np.outer(p, p)[dom]).sum())
                odd_tot += int(dom.sum())
            else:
                for jj in (j + 1, j - 1):
                    if jj < 1:
                        continue
                    q = np.array([(-1) ** pj(x, jj) for x in range(N)])
                    mask_bad += int((G[dom] != np.outer(q, q)[dom]).sum())
                    mask_tot += int(dom.sum())
    n8 = odd_bad > 0 and mask_bad > 0
    ok["N8"] = n8
    print(f"N8_NULL     odd-weight Y breaks (diamond) ({odd_bad}/{odd_tot}); a perturbed mask "
          f"2^(j+-1)-1 breaks it ({mask_bad}/{mask_tot}) {'OK' if n8 else 'FAIL'} -- neither "
          f"the parity hypothesis nor the mask is decorative")

    print("=" * 78)
    if all(ok.values()):
        print("CD_TOWER_ZDL2R_VERDICT L2_REDUCED_TO_FIBER_FREE_DIAMOND__NOT_PROVEN")
        print("CD_TOWER_ZDL2R_NOTE L2 (fiber L = Y|H, level n) is replaced by (diamond) "
              "(fiber-free, level n-1): with g(x,y) = sigma(tau x,tau y)sigma(x,y) and "
              "j = lsb(Y), for even-weight Y, Qgen'(Y,a,b) = -1 implies "
              "g(a,b)g(a^Y,b^Y) = (-1)^(p_j a + p_j b). The reduction is EXACT (N4) and its "
              "resonance predicate is the PROVEN Lean lemma Qred_hi_ll, minus sign included "
              "(N5). One ingredient is already proven forall n: g is symmetric (N1), which is "
              "chi_tau, and it is what kills the argument swap. What carries the content is a "
              "CANCELLATION -- g itself does not factor (N2) -- and the F2-linear route that "
              "closed the ZD locality lemma is walled here for j >= 3 (N3). Unlike (*), "
              "(diamond) genuinely NEEDS its resonance hypothesis: unrestricted it fails, and "
              "every failure is off resonance (N7). L2 IS NOT PROVEN and neither is (diamond). "
              "Numerical certificate; D3")
        return 0
    print("CD_TOWER_ZDL2R_VERDICT INCOMPLETE  failing="
          + ",".join(k for k, v in ok.items() if not v))
    return 1


if __name__ == "__main__":
    t0 = time.time()
    rc = main()
    print(f"[{time.time() - t0:.1f}s]")
    sys.exit(rc)
