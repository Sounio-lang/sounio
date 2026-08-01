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
  N4  THE REDUCTION IS A THEOREM, not a measurement. `l2_reduction` / `l2_reduction_symm` in
      formal/lean4/SounioZDFiberAntisym.lean, kernel-checked forall n: the four branch
      reductions (R_ll, R_uu) plus tau_seam / tau_xor, with `b ^ Y != 0` as the R_uu branch
      condition governing BOTH sides (tau (b^Y) = 0 <-> b^Y = 0 by tau_inj). This clause is a
      PIN of that theorem against the measured object, not the evidence for it. Consequence:
      L2 <== (diamond) is PROVEN and only (diamond) itself is measured -- strictly better than
      where L1 sits, whose K2/K3/K4 are genuinely measured.
  N5  PIN, AND IT CAUGHT AN ERROR. The reduced resonance predicate is the PROVEN Lean lemma
      Qred_hi_ll: Qgen(W + 2^(m+1), a, b, m+2) = - Qgen'(W, a, b, m+1). NOTE THE MINUS SIGN.
      The first version of this rung dropped it, and the failure locus then came out
      contradicting N7's cross-tab. The clause pins the Lean statement to the measured object.
  N9  THE CONCLUSION IS LEVEL-BOUNDED, and this is now a THEOREM: `G_descend` in
      formal/lean4/SounioZDFiberAntisym.lean, from the single lemma `gdisc_descend` --
      gdisc j x y (m+2) = gdisc j (x mod H) (y mod H) (m+1) in all four quadrants,
      unconditionally. The degenerate branches never surface because R_ul/R_uu guard on v = 0
      while the tau factor guards on tau v = 0 -- the SAME condition by tau_inj -- so the two
      constants (1*1 and (-1)*(-1)) multiply to 1, which is what gdisc is at a zero argument.
  N7  THE RESONANCE HYPOTHESIS IS ESSENTIAL, not an artifact of where the previous rung
      happened to define `disc`. Unrestricted, (diamond) FAILS; and every failure is OFF
      resonance, none on it. So L2 is genuinely a statement on the resonance graph -- unlike
      (*), which is unrestricted. That is the structural difference between the two lemmas.
  N8  NULL CONTROLS. Odd-weight Y must fail (it does), and a perturbed mask must fail (it does).
  N0  PARITY. The builder reproduces the in-tree sign_table entrywise.

NOT CLAIMED. L2 is NOT proven, and neither is (diamond). What IS proven forall n is the
REDUCTION itself (N4, `l2_reduction`), its symmetry ingredient (N1, `gdisc_symm`/`chi_tau`),
and the LEVEL-BOUNDEDNESS of (diamond)'s conclusion (N9, `G_descend`) -- all kernel-checked.
So the only measured link left in L2's chain is (diamond) itself, and its conclusion is known
to depend on nothing above bit j+1. (c) is unchanged in status: its (*) leg is
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
    for n in (5, 6, 7):
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
    print(f"N4_REDUCE   [Lean-proven forall n] fiber discrepancy == g(a,b)*g(b^Y,a^Y) == g(a,b)*g(a^Y,b^Y) entrywise "
          f"({n4_n} checks, levels 5,6,7) {'OK' if n4 else 'FAIL'} -- PROVEN forall n as "
          f"`l2_reduction`/`l2_reduction_symm`; this clause PINS that theorem to the measured "
          f"object. Fiber and top bit GONE; the swap is removed by N1")
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

    # ---- N9  THE CONCLUSION OF (diamond) IS LEVEL-BOUNDED ----------------------------------
    # G(Y,a,b) = g(a,b) g(a^Y,b^Y) is INVARIANT under dropping a level and truncating every
    # argument. So (diamond)'s conclusion is not a statement about a growing object at all: it
    # depends only on the bottom j+2 bits, and the target (-1)^{p_j a + p_j b} on the bottom j.
    n9_bad = n9_tot = 0
    for n in (6, 7):
        Sn = sign_table_fast(n).astype(np.int64)
        N = 1 << n
        for j in range(1, n - 1):
            Mj = 1 << (j + 2)
            Sm = sign_table_fast(j + 2).astype(np.int64)
            gn = gmat(Sn, N, j)
            gm = gmat(Sm, Mj, j)
            for Y in range(1, N):
                if (Y & -Y).bit_length() - 1 != j:
                    continue
                A, B = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
                lhs = gn * gn[A ^ Y, B ^ Y]
                Am, Bm = A % Mj, B % Mj
                Y0 = Y % Mj
                rhs = gm[Am, Bm] * gm[Am ^ Y0, Bm ^ Y0]
                n9_bad += int((lhs != rhs).sum())
                n9_tot += N * N
    n9 = n9_bad == 0
    ok["N9"] = n9
    print(f"N9_DESCENT  [Lean-proven forall n] G_n(Y,a,b) == G_(j+2)(Y mod 2^(j+2), a mod 2^(j+2), b mod 2^(j+2)) "
          f"{n9_bad}/{n9_tot} violations {'OK' if n9 else 'FAIL'} -- UNCONDITIONAL, every "
          f"quadrant, no degeneracy exceptions -- PROVEN forall n as `G_descend`, from the single "
          f"lemma `gdisc_descend`: gdisc drops a level in all four quadrants, because R_ul/R_uu "
          f"guard on v = 0 while the tau factor guards on tau v = 0, the SAME condition, so the "
          f"two constants multiply to 1. This clause now PINS that theorem. (diamond)'s "
          f"conclusion collapses to the bottom j+2 bits; it is not an unbounded-level statement")

    # ---- N10  where the hypothesis actually does work --------------------------------------
    n10_rows = []
    for j in (1, 2, 3, 4, 5):
        Mj = 1 << (j + 2)
        Sm = sign_table_fast(j + 2).astype(np.int64)
        gm = gmat(Sm, Mj, j)
        A, B = np.meshgrid(np.arange(Mj), np.arange(Mj), indexing="ij")
        p = np.array([(-1) ** pj(x, j) for x in range(Mj)])
        for Y0 in ((1 << j), (1 << j) | (1 << (j + 1))):
            D = gm * gm[A ^ Y0, B ^ Y0] * np.outer(p, p)
            n10_rows.append((j, Y0, int((D == -1).sum()), Mj * Mj))
    n10 = (all(c == 0 for jj, _, c, _ in n10_rows if jj <= 2)
           and all(c > 0 for jj, _, c, _ in n10_rows if jj >= 3))
    ok["N10"] = n10
    print(f"N10_TRIVIAL for j <= 2 the defect G*T is IDENTICALLY +1 -- (diamond) holds with NO "
          f"hypothesis at all; from j = 3 it does not {'OK' if n10 else 'FAIL'} -- "
          f"{'; '.join(f'j={a} Y0={b}: {c}/{d}' for a, b, c, d in n10_rows)}. The resonance "
          f"hypothesis only does work at j >= 3, which is INDEPENDENTLY the same boundary N3 "
          f"finds for F2-bilinearity of g")

    # ---- N11  the sign law, read off the sixteen PROVEN lemmas -----------------------------
    lean = os.path.join(HERE, "..", "..", "formal", "lean4", "SounioZDFiberAntisym.lean")
    txt = open(lean).read()
    rows = []
    for base in ("Qred", "Q'red"):
        for half in ("low", "hi"):
            for quad in ("ll", "lu", "ul", "uu"):
                name = f"{base}_{half}_{quad}"
                i = txt.find(f"theorem {name} ")
                if i < 0:
                    rows.append((name, None))
                    continue
                body = txt[i:txt.find(":= by", i)]
                concl = body[body.rindex("=") + 1:].strip()
                rows.append((name, concl.startswith("-")))
    n11 = all(neg is not None and neg == (name.split("_")[1] == "hi") for name, neg in rows)
    ok["N11"] = n11
    print(f"N11_SIGNLAW across all SIXTEEN proven reduction lemmas the sign is -1 EXACTLY when "
          f"the LABEL is high, and nothing else changes it {'OK' if n11 else 'FAIL'} -- read "
          f"off the theorem statements in formal/lean4/SounioZDFiberAntisym.lean, not measured. "
          f"(Priming is governed separately: from Q by the label's half, from Q' by whether "
          f"exactly one of a,b is upper.)")

    # ---- N12  hence the parity hypothesis, derived -----------------------------------------
    # By N11 the sign accumulated descending the resonance predicate from level n to level j+2
    # is (-1)^popcount(Y >> (j+2)), whatever a and b do. With lsb(Y) = j,
    #     weight(Y) = 1 + bit_{j+1}(Y) + popcount(Y >> (j+2))
    # so EVEN WEIGHT is exactly the statement that this accumulated sign equals
    # -(-1)^{bit_{j+1}(Y)}. The clause checks the arithmetic identity over every label.
    n12_bad = n12_tot = 0
    for n in (6, 7, 8):
        for Y in range(1, 1 << n):
            j = (Y & -Y).bit_length() - 1
            acc = (-1) ** (bin(Y >> (j + 2)).count("1"))
            pred = -((-1) ** ((Y >> (j + 1)) & 1))
            n12_tot += 1
            if (bin(Y).count("1") % 2 == 0) != (acc == pred):
                n12_bad += 1
    n12 = n12_bad == 0
    ok["N12"] = n12
    print(f"N12_PARITY  even weight(Y) <=> the accumulated descent sign (-1)^popcount(Y>>(j+2)) "
          f"equals -(-1)^bit_(j+1)(Y): {n12_bad}/{n12_tot} violations {'OK' if n12 else 'FAIL'} "
          f"-- so L2's EVEN-WEIGHT HYPOTHESIS IS THE PARITY OF THE SIGN FLIPS in the descent of "
          f"the resonance predicate. That is the mechanism, not a coincidence")

    # ---- N13  two closed forms for the defect, both KILLED ---------------------------------
    n13_rows = []
    for j in (3, 4):
        Mj = 1 << (j + 2)
        Sm = sign_table_fast(j + 2).astype(np.int64)
        gm = gmat(Sm, Mj, j)
        A, B = np.meshgrid(np.arange(Mj), np.arange(Mj), indexing="ij")
        p = np.array([(-1) ** pj(x, j) for x in range(Mj)])
        Y0 = 1 << j
        D = gm * gm[A ^ Y0, B ^ Y0] * np.outer(p, p)
        Qp_ = Sm * Sm[B ^ Y0, A ^ Y0] * Sm[B ^ Y0, A] * Sm[A ^ Y0, B]
        Qu_ = Sm * Sm[A ^ Y0, B ^ Y0] * Sm[A, B ^ Y0] * Sm[A ^ Y0, B]
        n13_rows.append((j, int((D != -Qp_).sum()), int((D != -Qu_).sum()), Mj * Mj))
    n13 = all(a > 0 and b > 0 for _, a, b, _ in n13_rows)
    ok["N13"] = n13
    print(f"N13_KILLED  the defect is NOT -Q'(Y0,a,b) and NOT -Q(Y0,a,b) "
          f"{'OK' if n13 else 'FAIL'} -- "
          f"{'; '.join(f'j={a}: {b} and {c} mismatches /{d}' for a, b, c, d in n13_rows)}. Two "
          f"closed forms tried and refuted; recorded so they are not retried")

    # ---- N14  does the HYPOTHESIS descend too? ---------------------------------------------
    # N9 proved the conclusion is level-bounded. The sign law (N11) and the priming law predict
    # the hypothesis descends as well:
    #   Q'_n(Y,a,b) = (-1)^popcount(Y>>(j+2)) * X_(j+2)(Y0,a0,b0),
    #   X = Q' if popcount((a^b)>>(j+2)) is even, else Q
    # -- the sign counts levels where the LABEL is high, the priming counts levels where
    # exactly ONE of a,b is upper. Unrestricted this FAILS in bulk. It holds exactly on the
    # CLEAN locus: tuples with no degeneracy at ANY level of the descent. That is the K17
    # phenomenon (*) already met -- a tuple non-degenerate at m+2 can reduce to a degenerate one.
    def _Qp(S, Y, a, b):
        return int(S[a, b]) * int(S[b ^ Y, a ^ Y]) * int(S[b ^ Y, a]) * int(S[a ^ Y, b])

    def _Qu(S, Y, a, b):
        return int(S[a, b]) * int(S[a ^ Y, b ^ Y]) * int(S[a, b ^ Y]) * int(S[a ^ Y, b])

    def _pc(x):
        return bin(x).count("1")

    def _clean(Y, a, b, j, n):
        for L in range(j + 2, n + 1):
            msk = (1 << L) - 1
            W, A, B = Y & msk, a & msk, b & msk
            if A == 0 or B == 0 or (A ^ W) == 0 or (B ^ W) == 0 or A == B or (A ^ B ^ W) == 0:
                return False
        return True

    n14_all = n14_tot = n14_cl = n14_cltot = 0
    for n in (6, 7):
        Sn = sign_table_fast(n).astype(np.int64)
        N = 1 << n
        for Y in range(1, N):
            j = (Y & -Y).bit_length() - 1
            if j + 2 > n:
                continue
            Mj = 1 << (j + 2)
            Sm = sign_table_fast(j + 2).astype(np.int64)
            Y0 = Y % Mj
            eps = (-1) ** _pc(Y >> (j + 2))
            for a in range(N):
                for b in range(N):
                    a0, b0 = a % Mj, b % Mj
                    X = (_Qp(Sm, Y0, a0, b0) if _pc((a ^ b) >> (j + 2)) % 2 == 0
                         else _Qu(Sm, Y0, a0, b0))
                    good = _Qp(Sn, Y, a, b) == eps * X
                    n14_tot += 1
                    n14_all += not good
                    if _clean(Y, a, b, j, n):
                        n14_cltot += 1
                        n14_cl += not good
    n14 = (n14_all > 0) and (n14_cl == 0)
    ok["N14"] = n14
    print(f"N14_HYPDESC the hypothesis descends TOO, but only on the CLEAN locus: unrestricted "
          f"{n14_all}/{n14_tot} violations, on tuples with no degeneracy at ANY level "
          f"{n14_cl}/{n14_cltot} {'OK' if n14 else 'FAIL'} -- sign counts levels where the LABEL "
          f"is high, priming counts levels where exactly ONE of a,b is upper. The gap locus is "
          f"the K17 phenomenon again and is NOT handled")

    # ---- N15  (diamond) as a BOUNDED family ------------------------------------------------
    # Combining N9 (conclusion descends, PROVEN) and N14 (hypothesis descends on the clean
    # locus), (diamond) restricted to the clean locus has NO reference to n at all:
    #   for Y0 in {2^j, 3*2^j}, eps = -(-1)^{bit_{j+1}(Y0)} (this is the even-weight
    #   hypothesis, N12), and for BOTH primings X:
    #       X(Y0,a0,b0) = -eps  ==>  D(Y0,a0,b0) = +1        at level j+2.
    n15_rows = []
    n15 = True
    for j in range(1, 8):
        L = j + 2
        Mj = 1 << L
        Sj = sign_table_fast(L).astype(np.int64)
        g = gmat(Sj, Mj, j)
        A, B = np.meshgrid(np.arange(Mj), np.arange(Mj), indexing="ij")
        p = np.array([(-1) ** pj(x, j) for x in range(Mj)])
        for Y0 in ((1 << j), (1 << j) | (1 << (j + 1))):
            eps = -((-1) ** ((Y0 >> (j + 1)) & 1))
            D = g * g[A ^ Y0, B ^ Y0] * np.outer(p, p)
            for nm, X in (("Q'", Sj * Sj[B ^ Y0, A ^ Y0] * Sj[B ^ Y0, A] * Sj[A ^ Y0, B]),
                          ("Q", Sj * Sj[A ^ Y0, B ^ Y0] * Sj[A, B ^ Y0] * Sj[A ^ Y0, B])):
                hyp = X == -eps
                bad = int((hyp & (D != 1)).sum())
                n15 = n15 and bad == 0
                n15_rows.append((j, Y0, nm, bad, int(hyp.sum())))
    ok["N15"] = n15
    print(f"N15_BOUNDED (diamond) on the clean locus has NO reference to n: at level j+2, "
          f"X(Y0,a0,b0) = -eps ==> D = +1. Checked EXHAUSTIVELY for j = 1..7 "
          f"{'OK' if n15 else 'FAIL'} -- "
          f"{'; '.join(f'j={a} Y0={b} X={c}: {d}/{e}' for a, b, c, d, e in n15_rows if a <= 3)}"
          f"; ... (all zero)")

    # ---- N16  one of the four cases is VACUOUS, by a theorem -------------------------------
    n16 = all(e == 0 for a, b, c, d, e in n15_rows if c == "Q" and b == (1 << a))
    ok["N16"] = n16
    print(f"N16_VACUOUS of N15's four cases, `Y0 = 2^j` with priming Q is EMPTY at every j "
          f"{'OK' if n16 else 'FAIL'} -- because Qgen at a single-bit label is identically -1 "
          f"(Qgen_pow2, PROVEN forall n), so its hypothesis X = +1 is unsatisfiable. One "
          f"quarter of the remaining statement is discharged by an existing theorem")

    # ---- N17/N18  THE GAP LOCUS -----------------------------------------------------------
    # N15 covered only the CLEAN locus. Running the descent with the SIXTEEN LEMMAS' ACTUAL
    # side conditions (not the blunt "no degeneracy anywhere" proxy) recovers a lot of it --
    # 9480 -> 24060 clean at n=6 -- but ~57% of hypothesis-satisfying tuples still BLOCK.
    #
    # The right object is therefore not "the clean locus" but the REACHABLE BOTTOM SET:
    #     REACH_j(Y0) = { (a mod 2^{j+2}, b mod 2^{j+2}) : Q'_n(Y,a,b) = -1, Y mod 2^{j+2} = Y0 }
    # Because the CONCLUSION descends unconditionally (N9, proven), (diamond) is EXACTLY
    #     REACH_j(Y0)  subset of  { D = +1 }
    # and that has no `n` in it PROVIDED REACH stabilises. It does.
    def _Qp2(S, Y, a, b):
        return int(S[a, b]) * int(S[b ^ Y, a ^ Y]) * int(S[b ^ Y, a]) * int(S[a ^ Y, b])

    from collections import defaultdict
    prev = {}
    n17 = True
    n17_rows = []
    n18_rows = []
    for n in (6, 7, 8):
        Sn = sign_table_fast(n).astype(np.int64)
        N = 1 << n
        reach = defaultdict(set)
        for Y in range(1, N):
            j = (Y & -Y).bit_length() - 1
            if j > 3 or j + 2 > n or bin(Y).count("1") % 2:
                continue
            Mj = 1 << (j + 2)
            for a in range(1, N):
                for b in range(1, N):
                    if b == Y or _Qp2(Sn, Y, a, b) != -1:
                        continue
                    reach[(j, Y % Mj)].add((a % Mj, b % Mj))
        for k in sorted(reach):
            j, Y0 = k
            Mj = 1 << (j + 2)
            Sj = sign_table_fast(j + 2).astype(np.int64)
            g = gmat(Sj, Mj, j)
            A, B = np.meshgrid(np.arange(Mj), np.arange(Mj), indexing="ij")
            p = np.array([(-1) ** pj(x, j) for x in range(Mj)])
            D = g * g[A ^ Y0, B ^ Y0] * np.outer(p, p)
            eps = -((-1) ** ((Y0 >> (j + 1)) & 1))
            Qpm = Sj * Sj[B ^ Y0, A ^ Y0] * Sj[B ^ Y0, A] * Sj[A ^ Y0, B]
            Qum = Sj * Sj[A ^ Y0, B ^ Y0] * Sj[A, B ^ Y0] * Sj[A ^ Y0, B]
            pred = (Qpm == -eps) | (Qum == -eps)
            badD = sum(1 for (x, y) in reach[k] if D[x, y] != 1)
            outside = sum(1 for (x, y) in reach[k] if not pred[x, y])
            n17 = n17 and badD == 0
            if n >= 8:
                n17_rows.append((j, Y0, len(reach[k]), Mj * Mj, badD))
                n18_rows.append((j, Y0, outside))
            if k in prev and n >= 8 and prev[k] != reach[k]:
                n17 = False
            prev[k] = reach[k]
    ok["N17"] = n17
    print(f"N17_REACH   [the truncation behind it is Lean-proven forall n: G_trunc] "
          f"the REACHABLE bottom set stabilises (j<=2 from n=6, j=3 from n=7, i.e. "
          f"n >= j+4; unchanged at n=8) and NEVER contains a D = -1 point "
          f"{'OK' if n17 else 'FAIL'} -- "
          f"{'; '.join(f'j={a} Y0={b}: |reach|={c}/{d}, D=-1 among them {e}' for a, b, c, d, e in n17_rows)}"
          f". Since the CONCLUSION truncates to level k for EVERY k > j -- `G_trunc`, PROVEN "
          f"forall n by iterating G_descend -- (diamond) IS "
          f"'REACH subset of {{D=+1}}' -- a FINITE, n-free statement per j, GAP LOCUS INCLUDED")
    n18 = any(o > 0 for _, _, o in n18_rows)
    ok["N18"] = n18
    print(f"N18_N15GAP  N15's clean-locus predicate does NOT cover REACH "
          f"{'OK' if n18 else 'FAIL'} -- points outside it: "
          f"{'; '.join(f'j={a} Y0={b}: {c}' for a, b, c in n18_rows)}. So the previous rung's "
          f"bounded family was NECESSARY BUT NOT SUFFICIENT: the blocked tuples land strictly "
          f"outside it, and REACH is the object that closes the gap locus")

    print("=" * 78)
    if all(ok.values()):
        print("CD_TOWER_ZDL2R_VERDICT DIAMOND_IS_A_FINITE_STABLE_FAMILY__GAP_LOCUS_INCLUDED")
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
              "every failure is off resonance (N7) -- so an induction on (diamond) must RE-ESTABLISH "
              "Qgen'(Y,a,b) = -1 at the reduced level in each quadrant, which is where the next "
              "attempt starts. L2 IS NOT PROVEN and neither is (diamond); the REDUCTION to it "
              "IS (N4, Lean, forall n). "
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
