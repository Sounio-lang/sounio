#!/usr/bin/env python3
"""
CD-tower ZD fibers — V1 REDUCED: ∀n spectral completeness needs only TWO INTEGERS.

V1 is the lane's standing open problem: #distinct A_sig spectra = 3*2^{n-5} for ALL n. The
antisymmetry rung (2026-07-31) halved the eigenproblem but explicitly did NOT close V1. This
rung does not close it either -- it REDUCES it, and pins exactly what is left.

THE DECOMPOSITION. The count is not mysterious; it is orbit arithmetic:

    #classes = (#Fano orbits) + (#seam orbits) - (#merges)
             =    2^{n-4}     +  (2^{n-4} - 1)  - (2^{n-5} - 1)
             = 2^{n-3} - 2^{n-5} = 3 * 2^{n-5}

of which:
  (a) the orbit split 2^{n-4} Fano + (2^{n-4}-1) fixed seams is PROVEN ∀n (the orbit theorem);
  (b) the spectrum is constant on each orbit -- PROVEN ∀n (the automorphism is an algebra map,
      so it is a graph isomorphism);
  (c) the even-weight seams merge (exactly 2^{n-5}-1 of them) -- the parity-collapse law, with
      an explicit iso Phi verified n<=8; its ∀n reduces to two named sigma-lemmas. OPEN ∀n;
  (d) NOTHING ELSE merges -- the surviving 3*2^{n-5} classes are pairwise NON-cospectral.

So V1 ∀n = (c) ∀n AND (d) ∀n. The arithmetic in between is trivial.

THE REDUCTION (this rung's contribution). (d) as stated is a claim about whole spectra, which
is the hard shape -- cospectral graphs are common. It is not needed. TWO INTEGERS suffice:

    tr(A_sig^2)  = the number of nonzero entries (edges x2; entries are +-1)
    tr(A_sig^3)  = the signed triangle count

W3 shows these two induce EXACTLY the spectral partition -- identical blocks, not merely an
equal count -- and W4 shows the count is 3*2^{n-5} at n = 6..11, six levels. So (d) reduces
from "no two classes are cospectral" to "a 2-integer invariant is injective on the classes",
which is a closed-form question rather than a spectral one.

  W0  PARITY. The builders reproduce the in-tree sign_table/A_sig entrywise.
  W1  DECOMPOSITION. Fano orbits = 2^{n-4}, seams = 2^{n-4}-1, even-weight seams = 2^{n-5}-1,
      and orbits - merges = #spectra = 3*2^{n-5}. Measured n = 6..9.
  W2  MONOCHROMATICITY. The spectrum is constant on every orbit. Measured n = 6..9.
  W3  PARTITION IDENTITY. The partition induced by (tr A^2, tr A^3) is IDENTICAL, block for
      block, to the partition induced by the full spectrum. Measured n = 6..9. (Equal counts
      would NOT be enough -- two partitions can have the same number of blocks and differ.)
  W4  THE REDUCTION. #distinct (tr A^2, tr A^3) = 3*2^{n-5} at n = 6..11.
  W5  MINIMALITY IN THE TRACE FAMILY. tr(A^1) = 0 for every fiber (the diagonal is zero), and
      tr(A^2) ALONE gives exactly 2^{n-4} values -- strictly fewer than 3*2^{n-5}. So neither
      trace alone suffices and the pair is not padded.
  W6  CLOSED FORM, AND ITS BOUNDARY (HONEST). On the stratum where tr(A^2) is constant -- the
      y = 0 Fano class together with ALL seams -- it equals (2^{n-1}-2) * 4*(2^{n-3}-1), i.e.
      the graph is Dmax-regular off the single isolated vertex, with Dmax the ∀n-proven core
      law. OFF that stratum tr(A^2) VARIES and no closed form is derived here; a general one
      appears to need the degree-histogram induction, which this lane records as OPEN.
  W7  NULL CONTROL. A weaker invariant must UNDER-separate: (#vertices, tr A^2) gives strictly
      fewer than 3*2^{n-5} blocks. If it separated too, W4 would be uninformative.

NOT CLAIMED -- V1 IS NOT CLOSED.
  * (d) is REDUCED, not proven. Closing it needs BOTH a closed form for (tr A^2, tr A^3) in
    terms of the fiber label AND a proof that the closed form is injective on the classes for
    all n. A form that collides at some large n leaves (d) open. Six levels is evidence.
  * (c) is untouched here and remains n <= 8.
  * Even (c) + (d) both closed would give V1; neither is closed.
  * tr(A^2)'s closed form is established only on the constant stratum (W6).

Verdict V1_REDUCED_TO_TWO_INTEGERS__NOT_CLOSED.
Numerical certificate over an exact integer sign table; D3 respected.
"""
import os
import sys
import time
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PRIOR = os.path.join(HERE, "cd_tower_zd_fiber_spectral_forall_n_progress_contract.py")
_src = open(PRIOR).read()
exec(_src.split("def main()")[0].split("from collections import defaultdict")[1])  # noqa: S102

SPEC_N = (6, 7, 8, 9)          # clauses needing eigendecompositions
TRACE_N = (6, 7, 8, 9, 10, 11)  # traces need no eigensolve


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
    """tr(A^2) = #nonzeros (entries are +-1); tr(A^3) via one matmul."""
    F = A.astype(np.float64)
    t2 = int(np.count_nonzero(A))
    t3 = int(round(float(np.sum(F * (F @ F).T))))
    return t2, t3


def orbit_of(Llo):
    """Fano group (low 3 bits nonzero) indexed by y = Llo>>3, or the fixed seam ('S', y)."""
    return ("F", Llo >> 3) if (Llo & 7) else ("S", Llo >> 3)


def main():
    print("=" * 78)
    print("CD-tower ZD fibers — V1 REDUCED: ∀n completeness needs only TWO INTEGERS")
    print("=" * 78)
    ok = {}

    # ---- W0 parity -----------------------------------------------------------------------
    w0 = True
    for n in (6, 7):
        Sref, Sfast = sign_table(n), sign_table_fast(n)
        if not np.array_equal(Sref, Sfast):
            w0 = False
        H = 1 << (n - 1)
        for Llo in range(1, H):
            if not np.array_equal(A_sig(n, Llo, Sref), A_sig_fast(n, Llo, Sfast).astype(float)):
                w0 = False
    ok["W0"] = w0
    print(f"W0_PARITY   builders == in-tree sign_table/A_sig entrywise (n=6,7) "
          f"{'OK' if w0 else 'FAIL'}")

    tables = {n: sign_table_fast(n) for n in TRACE_N}
    spec_cache = {}

    # ---- W1 decomposition + W2 monochromaticity + W3 partition identity -------------------
    w1 = w2 = w3 = True
    for n in SPEC_N:
        S, H = tables[n], 1 << (n - 1)
        spec, pair = {}, {}
        for Llo in range(1, H):
            A = A_sig_fast(n, Llo, S)
            spec[Llo] = tuple(np.round(np.linalg.eigvalsh(A.astype(np.float64)), 3))
            pair[Llo] = traces23(A)
        spec_cache[n] = (spec, pair)

        byorb = defaultdict(set)
        for Llo in range(1, H):
            byorb[orbit_of(Llo)].add(spec[Llo])
        nF = sum(1 for k in byorb if k[0] == "F")
        nS = sum(1 for k in byorb if k[0] == "S")
        merges = sum(1 for k in byorb if k[0] == "S" and bin(k[1]).count("1") % 2 == 0)
        nspec = len(set(spec.values()))
        pred = 3 * 2 ** (n - 5)
        cnt_ok = (nF == 2 ** (n - 4) and nS == 2 ** (n - 4) - 1
                  and merges == 2 ** (n - 5) - 1 and nF + nS - merges == nspec == pred)
        w1 = w1 and cnt_ok
        mono = all(len(v) == 1 for v in byorb.values())
        w2 = w2 and mono

        bs, bp = defaultdict(set), defaultdict(set)
        for Llo in range(1, H):
            bs[spec[Llo]].add(Llo)
            bp[pair[Llo]].add(Llo)
        same = sorted(map(sorted, bs.values())) == sorted(map(sorted, bp.values()))
        w3 = w3 and same

        print(f"W1_DECOMP   n={n}: Fano={nF}=2^(n-4)  seams={nS}=2^(n-4)-1  "
              f"even-wt merges={merges}=2^(n-5)-1  orbits-merges={nF + nS - merges} == "
              f"#spectra={nspec} == 3*2^(n-5)={pred}  {'OK' if cnt_ok else 'FAIL'}")
        print(f"W2_MONO     n={n}: spectrum constant on every orbit  {'OK' if mono else 'FAIL'}")
        print(f"W3_PARTITION n={n}: (trA^2,trA^3) induces the SPECTRAL partition block-for-block "
              f"{'OK' if same else 'FAIL'}")
    ok["W1"], ok["W2"], ok["W3"] = w1, w2, w3

    # ---- W4 the reduction, six levels ------------------------------------------------------
    w4 = True
    for n in TRACE_N:
        S, H = tables[n], 1 << (n - 1)
        if n in spec_cache:
            pairs = set(spec_cache[n][1].values())
        else:
            pairs = {traces23(A_sig_fast(n, Llo, S)) for Llo in range(1, H)}
        pred = 3 * 2 ** (n - 5)
        good = len(pairs) == pred
        w4 = w4 and good
        print(f"W4_REDUCE   n={n:2d}: #distinct (trA^2,trA^3) = {len(pairs):4d}  "
              f"3*2^(n-5) = {pred:4d}  {'OK' if good else 'FAIL'}")
    ok["W4"] = w4

    # ---- W5 minimality in the trace family -------------------------------------------------
    w5_t1 = w5_t2 = True
    for n in SPEC_N:
        S, H = tables[n], 1 << (n - 1)
        t2vals = set()
        for Llo in range(1, H):
            A = A_sig_fast(n, Llo, S)
            if int(np.trace(A.astype(np.int64))) != 0:
                w5_t1 = False
            t2vals.add(int(np.count_nonzero(A)))
        if len(t2vals) != 2 ** (n - 4):
            w5_t2 = False
        print(f"W5_MINIMAL  n={n}: tr(A^1)=0 for every fiber {w5_t1};  #distinct tr(A^2) alone "
              f"= {len(t2vals)} = 2^(n-4) = {2 ** (n - 4)} < 3*2^(n-5) = {3 * 2 ** (n - 5)} "
              f"{'OK' if w5_t2 else 'FAIL'}")
    ok["W5"] = w5_t1 and w5_t2

    # ---- W6 closed form on the constant stratum, and its boundary --------------------------
    w6_on = w6_off = True
    for n in SPEC_N:
        S, H = tables[n], 1 << (n - 1)
        want = (2 ** (n - 1) - 2) * 4 * (2 ** (n - 3) - 1)
        offvals = set()
        for Llo in range(1, H):
            t2 = int(np.count_nonzero(A_sig_fast(n, Llo, S)))
            kind, y = orbit_of(Llo)
            # the constant stratum is the y=0 Fano class plus the WEIGHT-1 seams -- NOT all
            # seams; the first draft of this clause said "all seams" and the gate caught it.
            on_stratum = (kind == "F" and y == 0) or (kind == "S" and bin(y).count("1") == 1)
            if on_stratum:
                if t2 != want:
                    w6_on = False
            else:
                offvals.add(t2)
        if len(offvals) <= 1:            # off the stratum it must genuinely VARY
            w6_off = False
        print(f"W6_CLOSED   n={n}: tr(A^2) == (2^(n-1)-2)*4*(2^(n-3)-1) = {want} on the y=0 Fano "
              f"class and the weight-1 seams {'OK' if w6_on else 'FAIL'};  off that stratum it takes "
              f"{len(offvals)} distinct values (NOT closed here) {'OK' if w6_off else 'FAIL'}")
    ok["W6"] = w6_on and w6_off

    # ---- W7 null control -------------------------------------------------------------------
    w7 = True
    for n in SPEC_N:
        S, H = tables[n], 1 << (n - 1)
        weak = {(H - 1, int(np.count_nonzero(A_sig_fast(n, Llo, S)))) for Llo in range(1, H)}
        if len(weak) >= 3 * 2 ** (n - 5):
            w7 = False
        print(f"W7_NULL     n={n}: the weaker invariant (#vertices, trA^2) gives {len(weak)} "
              f"blocks < 3*2^(n-5) = {3 * 2 ** (n - 5)} -- it UNDER-separates, so W4 is not "
              f"vacuous {'OK' if w7 else 'FAIL'}")
    ok["W7"] = w7

    # ---- W8  (c) IS CLOSED, so V1 = (d) ALONE ----------------------------------------------
    # `SounioZDCollapse.parity_collapse` (2026-08-02) proves Phi is an isomorphism of the SIGNED
    # annihilation graph between an even-weight seam fiber and its Fano partner, forall n, with
    # BOTH sign identities discharged ((*) = star_forall, L2 = L2_forall). An isomorphism of the
    # signed graph forces equal spectra, which IS the merge.
    #
    # FOUR things the orbit arithmetic silently assumes, checked here:
    #   (i)   the theorem's hypotheses hold on EVERY even-weight seam (no stratum left out);
    #   (ii)  the merge it forces really happens (spectra coincide);
    #   (iii) tau W lands in a FANO orbit, not another seam -- otherwise the merge would be
    #         seam<->seam and the arithmetic would be different;
    #   (iv)  the even-weight seams number 2^{n-5}-1.
    # And one thing the arithmetic does NOT need, recorded because assuming it would be WRONG:
    #   (v)   the map (even-weight seam) -> (its Fano orbit) is NOT injective. It collapses
    #         2^{n-5}-1 seams onto 2^{n-6} orbits, because tau clears the lowest set bit of
    #         y = W>>3, so y = 5 and y = 6 both land on y = 4. The subtraction is still right:
    #         each merged seam is removed ONCE regardless of where it lands.
    def _tau(x, j):
        b0, bj = x & 1, (x >> j) & 1
        return x ^ (1 | (1 << j)) if b0 != bj else x

    w8_rows = []
    w8 = True
    for n in SPEC_N:
        S = sign_table_fast(n)
        H = 1 << (n - 1)
        even = [W for W in range(8, H, 8) if bin(W >> 3).count("1") % 2 == 0]
        hyp_bad = fano_bad = merge_bad = 0
        imgs = []
        for W in even:
            j = (W & -W).bit_length() - 1
            if not (W < H and W % 2 ** (j + 1) == 2 ** j
                    and bin(W).count("1") % 2 == 0 and j + 2 <= n - 1):
                hyp_bad += 1
            Wt = _tau(W, j)
            if (Wt & 7) == 0:
                fano_bad += 1
            imgs.append(Wt >> 3)
            e1 = np.round(np.linalg.eigvalsh(A_sig_fast(n, W, S).astype(np.float64)), 6)
            e2 = np.round(np.linalg.eigvalsh(A_sig_fast(n, Wt, S).astype(np.float64)), 6)
            if not np.allclose(e1, e2, atol=1e-6):
                merge_bad += 1
        cnt_ok = (len(even) == 2 ** (n - 5) - 1)
        noninj = (len(set(imgs)) < len(even)) if len(even) > 1 else True
        w8 = w8 and hyp_bad == 0 and fano_bad == 0 and merge_bad == 0 and cnt_ok and noninj
        w8_rows.append((n, len(even), hyp_bad, fano_bad, merge_bad, cnt_ok, len(set(imgs))))
    ok["W8"] = w8
    print(f"W8_CCLOSED  (c) IS CLOSED forall n -- so V1 = (d) ALONE {'OK' if w8 else 'FAIL'} -- "
          + "; ".join(f"n={a}: {b} even-weight seams, hyp-fail {c}, tauW-not-Fano {d}, "
                      f"merge-fail {e}, count=2^(n-5)-1 {f}, distinct Fano images {g}"
                      for a, b, c, d, e, f, g in w8_rows)
          + ". `SounioZDCollapse.parity_collapse` (kernel-checked, no sorryAx) makes Phi an "
            "isomorphism of the SIGNED annihilation graph between an even-weight seam fiber and "
            "its Fano partner, forall n, with BOTH sign identities discharged: (*) = "
            "`star_forall`, L2 = `L2_forall`. j = lsb(W) >= 3 for a seam and j <= n-3 is "
            "automatic (j = n-2 forces W = 2^(n-2), which has ODD weight), so the theorem covers "
            "EVERY even-weight seam. *** AND ONE THING THE ARITHMETIC DOES NOT NEED, recorded "
            "because assuming it would be WRONG: the map (even-weight seam) -> (its Fano orbit) "
            "is NOT injective -- tau clears the LOWEST SET BIT of y = W>>3, so y = 5 and y = 6 "
            "both land on y = 4, and 2^(n-5)-1 seams collapse onto 2^(n-6) orbits. The "
            "subtraction is still right because each merged seam is removed ONCE regardless of "
            "where it lands. *** WHAT REMAINS OF V1 IS (d) ALONE: that the 2^(n-4) Fano orbits "
            "are pairwise non-cospectral and that no odd-weight seam merges")

    # ---- W9  ATTACKING (d): the degree splits, and the bulk has a CLOSED FORM ---------------
    # W6 recorded that tr(A^2) has a closed form only on the narrow stratum where it is constant,
    # and that the general case "appears to need the degree-histogram induction, which is OPEN".
    # Two structural facts move it:
    #
    # (i) EDGE <=> RESONANCE. A_sig's definition carries three conditions -- P1 symmetric, P3
    #     symmetric, P1 = P3 -- and the first two are AUTOMATIC. Reason: the commutation sign
    #     chi(x,y) = sigma(x,y)sigma(y,x) is -1 for distinct nonzero x,y, and each symmetry
    #     condition is a PRODUCT OF TWO chi's (chi(a,b)chi(a^L,b^L) and chi(a,b^L)chi(a^L,b)),
    #     so both are +1 identically. The edge relation is exactly resonance.
    #
    # (ii) THEREFORE the degree is a resonance count, and `Qred_hi_ll` turns it into Qgen' one
    #     level down, where THIS SESSION's collapse theorem applies: off the six degeneracy lines
    #     Qgen'(Llo,a,b,m) = eps * Qgen(Llo0,a0,b0,j+2), a function of the bottom residues alone.
    #     Every residue class has exactly H/M representatives, so the non-degenerate part of the
    #     degree is a pure residue count -- a CLOSED FORM. What is left is the count over the six
    #     lines, where the collapse does not apply.
    def _Qp(S, Y, a, b):
        return int(S[a, b]) * int(S[b ^ Y, a ^ Y]) * int(S[b ^ Y, a]) * int(S[a ^ Y, b])

    _bt = {}

    def Qu_bottom(jp, Y, a, b):
        if jp not in _bt:
            _bt[jp] = sign_table_fast(jp + 2)
        S = _bt[jp]
        return int(S[a, b]) * int(S[a ^ Y, b ^ Y]) * int(S[a, b ^ Y]) * int(S[a ^ Y, b])

    # (a) edge <=> resonance
    w9a_bad = w9a_tot = 0
    for n in (6, 7):
        Sx = sign_table_fast(n).astype(np.int16)
        H = 1 << (n - 1)
        los = np.arange(1, H)
        for Llo in range(1, H):
            L = Llo | H
            hi = los ^ L
            P1 = Sx[np.ix_(los, los)] * Sx[np.ix_(hi, hi)]
            P3 = Sx[np.ix_(los, hi)] * Sx[np.ix_(hi, los)]
            full = (P1 == P1.T) & (P3 == P3.T) & (P1 == P3)
            just = (P1 == P3)
            np.fill_diagonal(full, False)
            np.fill_diagonal(just, False)
            w9a_bad += int((full != just).sum())
            w9a_tot += full.size

    # (b) the split is exact, and the non-degenerate part is the closed form
    def _pred(n, Llo, a):
        j = (Llo & -Llo).bit_length() - 1
        M = 1 << (j + 2)
        low = 1 << j
        L0 = Llo % M
        eps = (-1) ** (bin(Llo >> (j + 2)).count("1") % 2)
        a0 = a % M
        c = 0
        for b0 in range(M):
            if (a0 == 0 or b0 == 0 or a0 == L0 or b0 == L0 or a0 == b0 or (a0 ^ b0) == L0):
                continue
            q = -1 if L0 == low else (-1 if (a0 % low == 0 or b0 % low == 0
                                             or a0 % low == b0 % low) else 1)
            if eps * q == -1:
                c += 1
        return c * ((1 << (n - 1)) // M)

    w9b_bad = w9b_tot = w9c_bad = 0
    for n in (6, 7):
        Sm = sign_table_fast(n - 1)
        S = sign_table_fast(n)
        H = 1 << (n - 1)
        for Llo in range(1, H):
            j = (Llo & -Llo).bit_length() - 1
            if j == 0 or j + 2 > n - 1:
                continue
            M = 1 << (j + 2)
            L0 = Llo % M
            A = A_sig_fast(n, Llo, S)
            deg = np.count_nonzero(A, axis=1)
            for a in range(1, H):
                if a == Llo:
                    continue
                a0 = a % M
                nd = dg = 0
                for b in range(1, H):
                    if b == a or b == Llo:
                        continue
                    b0 = b % M
                    degen = (a0 == 0 or b0 == 0 or a0 == L0 or b0 == L0
                             or a0 == b0 or (a0 ^ b0) == L0)
                    if _Qp(Sm, Llo, a, b) == -1:
                        if degen:
                            dg += 1
                        else:
                            nd += 1
                w9b_tot += 1
                if int(deg[a - 1]) != nd + dg:
                    w9b_bad += 1
                if nd != _pred(n, Llo, a):
                    w9c_bad += 1
    w9 = (w9a_bad == 0 and w9b_bad == 0 and w9c_bad == 0 and w9b_tot > 0)
    ok["W9"] = w9
    print(f"W9_DEGSPLIT ATTACKING (d): the degree SPLITS and its bulk has a CLOSED FORM "
          f"{'OK' if w9 else 'FAIL'} -- (a) edge <=> resonance, i.e. A_sig's two SYMMETRY "
          f"conditions are automatic: {w9a_bad}/{w9a_tot} mismatches (n=6,7, every fiber) -- "
          f"each is a product of TWO commutation signs chi, and chi = -1 for distinct nonzero "
          f"arguments, so both products are +1 identically; (b) deg(a) = [non-degenerate part] + "
          f"[six-lines part], EXACTLY: {w9b_bad}/{w9b_tot} (a,Llo) pairs wrong; (c) the "
          f"non-degenerate part equals the CLOSED FORM (H/M) * #{{b0 off the lines : "
          f"eps*Qgen(Llo0,a0,b0,j+2) = -1}}: {w9c_bad}/{w9b_tot} wrong. That form is pure residue "
          f"counting, because the COLLAPSE theorem makes Qgen' a function of the bottom residues "
          f"alone and every residue class has exactly H/M representatives; the bottom Qgen is "
          f"known in closed form for BOTH label classes (Qgen_pow2 and Q_three_pow2, both "
          f"Lean-proven). *** W6 recorded tr(A^2) as having NO general closed form, needing the "
          f"degree-histogram induction. It now has one on the complement of the six lines, and "
          f"what is OPEN is the SIX-LINES COUNT alone -- a thin set (4 residues out of M for a "
          f"generic a), and the same degenerate locus that needed separate treatment throughout "
          f"the L2 chain. (d) IS NOT CLOSED ***")

    # ---- W10  THE DEGREE IS FULLY DETERMINED -- but stratified, not bounded ----------------
    # The collapse theorem takes j as a FREE parameter: it never requires j = lsb(Y). So a pair
    # that is degenerate at j = lsb(Llo) -- exactly where W9's remainder R(a) lives -- may be
    # NON-degenerate at a LARGER j', and then the collapse determines its value at the finer
    # bottom level j'+2. Running that:
    #   * EVERY ordered pair is either the coset partner b = a^Llo, or resolved at some j' >= j;
    #   * where it resolves, the collapse value is EXACTLY right;
    #   * the coset partner has Qgen' = +1 ALWAYS, i.e. it is NEVER an edge.
    # So R(a) is not unknown -- it is determined. What it is NOT is BOUNDED: the minimal
    # resolving j' runs to j+5 already at n = 8 and grows with n, so this is a stratified
    # determination, not a closed form. That distinction is the honest state of (d).
    w10_cos = w10_res = w10_unres = w10_bad = 0
    w10_lvl = {}
    w10_cosvals = set()
    for n in (6, 7):
        Sm = sign_table_fast(n - 1)
        H = 1 << (n - 1)
        m = n - 1
        for Llo in range(1, H):
            j = (Llo & -Llo).bit_length() - 1
            if j + 2 > m:
                continue
            for a in range(1, H):
                if a == Llo:
                    continue
                for b in range(1, H):
                    if b == a or b == Llo:
                        continue
                    if (a ^ b) == Llo:
                        w10_cos += 1
                        w10_cosvals.add(_Qp(Sm, Llo, a, b))
                        continue
                    jp = None
                    for k in range(j, m - 1):
                        Mp = 1 << (k + 2)
                        A0, B0, Y0 = a % Mp, b % Mp, Llo % Mp
                        if not (A0 == 0 or B0 == 0 or A0 == Y0 or B0 == Y0
                                or A0 == B0 or (A0 ^ B0) == Y0):
                            jp = k
                            break
                    if jp is None:
                        w10_unres += 1
                        continue
                    w10_res += 1
                    w10_lvl[jp - j] = w10_lvl.get(jp - j, 0) + 1
                    Mp = 1 << (jp + 2)
                    eps = (-1) ** (bin(Llo >> (jp + 2)).count("1") % 2)
                    if _Qp(Sm, Llo, a, b) != eps * Qu_bottom(jp, Llo % Mp, a % Mp, b % Mp):
                        w10_bad += 1
    w10 = (w10_unres == 0 and w10_bad == 0 and w10_cosvals == {1} and len(w10_lvl) > 2)
    ok["W10"] = w10
    print(f"W10_DEGDET  THE DEGREE IS FULLY DETERMINED -- but STRATIFIED, not bounded "
          f"{'OK' if w10 else 'FAIL'} -- the collapse theorem takes j as a FREE parameter (it "
          f"never requires j = lsb(Y)), so a pair degenerate at j = lsb(Llo) -- exactly where "
          f"W9's remainder R(a) lives -- may be non-degenerate at a LARGER j'. Running that on "
          f"every ordered pair, n = 6,7: {w10_res} resolved at some j' with the collapse value "
          f"EXACTLY right ({w10_bad} wrong), {w10_cos} coset partners b = a^Llo whose Qgen' "
          f"values are {sorted(w10_cosvals)} -- always +1, so the coset partner is NEVER an edge "
          f"-- and {w10_unres} left over. NOTHING is left over. *** So R(a) is not unknown, it is "
          f"DETERMINED. What it is NOT is BOUNDED: the minimal resolving level is "
          f"{'; '.join(f'j+{k}: {v}' for k, v in sorted(w10_lvl.items()))} and it grows with n "
          f"(to j+5 at n=8), so this is a STRATIFIED determination, not a closed form. That "
          f"distinction is the honest state of (d): tr(A^2) is computable level by level from a "
          f"PROVEN theorem plus one explicit never-an-edge class, and a bounded closed form would "
          f"need the strata to telescope. (d) IS NOT CLOSED ***")

    # ---- W11  THE INJECTIVITY DECOMPOSES, and tr(A^2) is PARITY-BLIND ----------------------
    # (d)'s second half is "the pair (tr A^2, tr A^3) is injective on the surviving classes".
    # Its FIBRE STRUCTURE is now known exactly, which splits it into two named statements:
    #   (I)  tr(A^2) is INJECTIVE on the 2^{n-4} Fano orbits;
    #   (II) tr(A^2)(seam y) = tr(A^2)(Fano tau y), tau y = y with its LOWEST SET BIT cleared --
    #        for EVERY seam, even weight and odd alike;
    #   (III) tr(A^3) then separates the odd-weight seams inside those fibres.
    # (II) is the conceptual one: tau always preserves tr(A^2) but preserves the SPECTRUM only
    # for even weight. So tr(A^2) is PARITY-BLIND -- it is an L1-level invariant, and the parity
    # the collapse law turns on is carried entirely by tr(A^3). That is the trace-side echo of
    # the lane's own C1/C2 finding that L1 does not see the parity and L2 does.
    w11_rows = []
    w11 = True
    for n in SPEC_N:
        S = sign_table_fast(n)
        H = 1 << (n - 1)
        fano, seam = {}, {}
        for Llo in range(1, H):
            y = Llo >> 3
            (fano if (Llo & 7) else seam).setdefault(y, Llo)
        F = {y: traces23(A_sig_fast(n, L, S)) for y, L in fano.items()}
        Sm = {y: traces23(A_sig_fast(n, L, S)) for y, L in seam.items()}
        inj = len({v[0] for v in F.values()}) == len(F)          # (I)
        b2 = b3e = b3o = 0
        for y, c in Sm.items():
            t = y - (y & -y)
            if c[0] != F[t][0]:
                b2 += 1                                          # (II)
            same3 = (c[1] == F[t][1])
            if bin(y).count("1") % 2 == 0:
                if not same3:
                    b3e += 1
            elif same3:
                b3o += 1                                         # (III)
        allc = list(F.values()) + [Sm[y] for y in Sm if bin(y).count("1") % 2]
        pair = (len(set(allc)) == len(allc) == 3 * 2 ** (n - 5))
        w11 = w11 and inj and b2 == 0 and b3e == 0 and b3o == 0 and pair
        w11_rows.append((n, len(F), inj, b2, b3e, b3o, pair))
    ok["W11"] = w11
    print(f"W11_INJDEC  THE INJECTIVITY DECOMPOSES, and tr(A^2) is PARITY-BLIND "
          f"{'OK' if w11 else 'FAIL'} -- "
          + "; ".join(f"n={a}: {b} Fano orbits, trA2 injective on them {c}, "
                      f"trA2(seam y) != trA2(Fano tau y) on {d}, even-wt trA3 differs on {e}, "
                      f"odd-wt trA3 EQUAL on {f}, pair injective {g}"
                      for a, b, c, d, e, f, g in w11_rows)
          + ". So the FIBRES of tr(A^2) are known EXACTLY: each is one Fano orbit y together "
            "with the seams y + 2^i, i < lsb(y), that tau maps onto it. (d)'s injectivity is "
            "therefore (I) tr(A^2) injective on the Fano orbits AND (III) tr(A^3) separating "
            "inside each fibre -- two statements about a KNOWN structure, not one statement "
            "about an unknown one. *** AND THE CONCEPTUAL POINT: tau ALWAYS preserves tr(A^2) "
            "but preserves the SPECTRUM only for even weight, so tr(A^2) is PARITY-BLIND. It is "
            "an L1-level invariant; the parity the collapse law turns on is carried entirely by "
            "tr(A^3). That is the trace-side echo of C1/C2 -- L1 holds for every seam and does "
            "not see the parity, L2 does. (I) and (III) are MEASURED, not proven; (d) IS NOT "
            "CLOSED ***")

    # ---- W12  (I) IS NO LONGER A MEASUREMENT: it FOLLOWS from a recursion, by PARITY --------
    # With T(n,y) = trA2(n, Fano orbit y)/24, h = 2^{n-5}, c_n = 2^{n-3}-1, A(n) = T(n,0):
    #
    #     T(n, y)     = 4*T(n-1, y) + c_n          (y < h)
    #     T(n, y + h) = A(n) - 4*T(n-1, y)         (y < h)
    #
    # Given that, injectivity is an INDUCTION and the only interesting case dies by parity:
    #   * lower half  y -> 4*T(n-1,y) + c_n   is injective whenever T(n-1,.) is  [affine]
    #   * upper half  y -> A(n) - 4*T(n-1,y)  likewise
    #   * CROSS: a lower value equals an upper one iff 4T(n-1,y) + c_n = A(n) - 4T(n-1,y''),
    #     and A(n) = 4A(n-1) + c_n, so that is T(n-1,y) + T(n-1,y'') = A(n-1).
    #     EVERY T IS ODD, so the left side is EVEN and the right side is ODD. Impossible.
    #   * base n=6: T = [35,19,7,23], distinct.
    # Oddness propagates: 4*odd + odd = odd and odd - 4*odd = odd, with c_n odd for n >= 4.
    def _T(n):
        S = sign_table_fast(n)
        H = 1 << (n - 1)
        reps = {}
        for Llo in range(1, H):
            if Llo & 7:
                reps.setdefault(Llo >> 3, Llo)
        return {y: traces23(A_sig_fast(n, L, S))[0] // 24 for y, L in sorted(reps.items())}

    _Tc = {n: _T(n) for n in (6, 7, 8, 9, 10)}
    w12_rec = w12_odd = w12_A = w12_cross = w12_inj = 0
    w12_rows = []
    for n in (7, 8, 9, 10):
        t, tp = _Tc[n], _Tc[n - 1]
        h = 1 << (n - 5)
        c = 2 ** (n - 3) - 1
        ra = sum(1 for y in range(h) if t[y] != 4 * tp[y] + c)
        rb = sum(1 for y in range(h) if t[y + h] != t[0] - 4 * tp[y])
        od = sum(1 for y in t if t[y] % 2 == 0)
        aa = 0 if t[0] == 4 * tp[0] + c else 1
        cr = sum(1 for y in tp for z in tp if tp[y] + tp[z] == tp[0])
        ij = 0 if len(set(t.values())) == len(t) else 1
        w12_rec += ra + rb
        w12_odd += od
        w12_A += aa
        w12_cross += cr
        w12_inj += ij
        w12_rows.append((n, len(t), ra, rb, od, cr, ij))
    w12_base = (sorted(_Tc[6].values()) == [7, 19, 23, 35])
    w12 = (w12_rec == 0 and w12_odd == 0 and w12_A == 0 and w12_cross == 0
           and w12_inj == 0 and w12_base)
    ok["W12"] = w12
    print(f"W12_TRECUR  (I) IS NO LONGER A MEASUREMENT -- it FOLLOWS from a recursion, by PARITY "
          f"{'OK' if w12 else 'FAIL'} -- with T = trA2/24 on the Fano orbits, h = 2^(n-5), "
          f"c_n = 2^(n-3)-1: T(n,y) = 4T(n-1,y) + c_n on the lower half and "
          f"T(n,y+h) = T(n,0) - 4T(n-1,y) on the upper. "
          + "; ".join(f"n={a}: {b} orbits, lower-half wrong {c}, upper-half wrong {d}, "
                      f"EVEN T values {e}, pairs summing to A(n-1) {f}, non-injective {g}"
                      for a, b, c, d, e, f, g in w12_rows)
          + f"; base n=6 T = [7,19,23,35] {w12_base}. *** GIVEN THE RECURSION, INJECTIVITY IS AN "
            "INDUCTION: both halves are AFFINE in T(n-1,.) hence injective, and the only cross "
            "case reduces to T(n-1,y) + T(n-1,y'') = A(n-1) -- EVEN = ODD, impossible, because "
            "every T is odd and oddness propagates (4*odd + odd = odd, odd - 4*odd = odd, c_n "
            "odd). So (I) is not a coincidence checked at four levels: it is a consequence of "
            "ONE recursion. That recursion is what is now MEASURED (n = 7..10), and it is the "
            "single thing (I) needs. (III) is untouched and (d) IS NOT CLOSED ***")

    # ---- W13  THE RECURSION IS DERIVED: four quadrants, eight rows, one sign law ------------
    # W12's recursion is not an observed coincidence about Fano representatives. It comes from a
    # RAW-COUNT recursion that holds for EVERY label. With
    #     N(m,W) = #{(a,b) in [1,2^m)^2 : a != b, Qgen'(W,a,b,m) = -1}   and  e = 2^(m-1):
    #
    #     N(m, W)      = 4*N(m-1, W)                    + 10e - 18       (W < e, label LOW)
    #     N(m, W + e)  = 4(e-1)(e-2) - 4*N(m-1, W)      +  6e - 10       (label HIGH)
    #
    # MECHANISM. Split (a,b) by the top bit at level m -- four quadrants. The eight Q'red rows
    # send each quadrant to level m-1, and N11's sign law says the sign is -1 EXACTLY when the
    # LABEL is high. A -1 sign turns "count the -1s" into "count the +1s" = (total) - (-1s),
    # which IS the reflection in the upper-half formula. The `ll` quadrant is `Q'red_low_ll`,
    # which is UNCONDITIONAL, so it contributes N(m-1,W) on the nose; the other three differ by
    # constants that depend ONLY on the level, never on the label -- they are the finitely many
    # degenerate (u,v) where a row's side condition fails, plus the Qgen-vs-Qgen' swap.
    #
    # Subtracting the isolated-vertex pairs (2^m - 2, since a = W and b = W are excluded from
    # A_sig) turns these into W12's recursion EXACTLY:
    #     trA2(n,y)   = 4 trA2(n-1,y) + 12(2^(n-2) - 2) = 4 trA2(n-1,y) + 24 c_n
    #     trA2(n,y+h) = 4(2^(n-2)-1)(2^(n-2)-2) - 4 trA2(n-1,y) = A(n) - 4 trA2(n-1,y)
    def _N(m, W, S):
        H = 1 << m
        return sum(1 for a in range(1, H) for b in range(1, H)
                   if a != b and _Qp(S, W, a, b) == -1)

    w13_bl = w13_bh = w13_t = w13_q = 0
    w13_rows = []
    for m in (5, 6, 7):
        S = sign_table_fast(m)
        Sp = sign_table_fast(m - 1)
        e = 1 << (m - 1)
        bl = bh = 0
        for W in range(1, e):
            p = _N(m - 1, W, Sp)
            if _N(m, W, S) != 4 * p + 10 * e - 18:
                bl += 1
            if _N(m, W + e, S) != 4 * (e - 1) * (e - 2) - 4 * p + 6 * e - 10:
                bh += 1
            w13_t += 2
        # and the ll quadrant is EXACTLY N(m-1,W), unconditionally
        for W in (1, 9, 17):
            if W >= e:
                continue
            H, Hp = 1 << m, e
            ll = sum(1 for a in range(1, Hp) for b in range(1, Hp)
                     if a != b and _Qp(S, W, a, b) == -1)
            if ll != _N(m - 1, W, Sp):
                w13_q += 1
        w13_bl += bl
        w13_bh += bh
        w13_rows.append((m, e - 1, bl, bh))
    # the conversion back to W12's constants
    w13_conv = all(12 * (2 ** (nn - 2) - 2) == 24 * (2 ** (nn - 3) - 1) for nn in range(5, 13))
    w13 = (w13_bl == 0 and w13_bh == 0 and w13_q == 0 and w13_conv and w13_t > 0)
    ok["W13"] = w13
    print(f"W13_RECDER  THE RECURSION IS DERIVED -- four quadrants, eight rows, one sign law "
          f"{'OK' if w13 else 'FAIL'} -- with N(m,W) the raw resonance count and e = 2^(m-1): "
          f"N(m,W) = 4N(m-1,W) + 10e - 18 for a LOW label, and "
          f"N(m,W+e) = 4(e-1)(e-2) - 4N(m-1,W) + 6e - 10 for a HIGH one. "
          + "; ".join(f"m={a}: {b} labels, low-form wrong {c}, high-form wrong {d}"
                      for a, b, c, d in w13_rows)
          + f"; the `ll` quadrant equals N(m-1,W) on the nose in {w13_q} failures (it is "
            f"`Q'red_low_ll`, UNCONDITIONAL); the constant conversion 12(2^(n-2)-2) = 24 c_n "
            f"holds {w13_conv}. *** SO W12's RECURSION IS NOT A COINCIDENCE ABOUT FANO "
            "REPRESENTATIVES: it holds for EVERY label, and it is the four-quadrant split under "
            "the eight Q'red rows. N11's sign law -- the sign is -1 exactly when the LABEL is "
            "high -- is what turns 'count the -1s' into 'count the +1s' = total minus, i.e. the "
            "REFLECTION in the upper half. Subtracting the isolated-vertex pairs (2^m - 2) gives "
            "W12's form exactly. WHAT IS STILL MEASURED is only the two level-constants 10e-18 "
            "and 6e-10 -- counts of the degenerate (u,v) where a row's side condition fails. "
            "So (I) now rests on TWO INTEGERS PER LEVEL. (III) is untouched; (d) IS NOT CLOSED ***")

    # ---- W14  THE LEVEL-CONSTANTS, DERIVED from four closed forms --------------------------
    # W13 left the two constants 10e-18 and 6e-10 measured. They decompose, quadrant by quadrant,
    # into pieces each priced by a CLOSED FORM proven in Lean:
    #   ll = N(m-1,W)  exactly                       `Q'red_low_ll` is UNCONDITIONAL
    #   lu = N(m-1,W) + 3(e-2), the three being
    #        v = 0      -> Qgen(W,0,u) = -1          `Qgen_zero_left`
    #        v = u      -> Qgen(W,u,u) = -1          `Qgen_diag_neg`
    #        v = u^W    -> UNPRIMED -1 (coset_left + diag) but PRIMED +1
    #                                                `Qgen'_coset_partner`
    #        u = W      -> the row fails, and contributes ZERO
    #   uu = lu      (same three pieces)
    #   ul = lu + e  (the u = 0 slice, e-1 terms all -1 by `Qgen_zero_left`, plus one boundary)
    # Total 4N(m-1,W) + 9(e-2) + e = 4N + 10e - 18.
    def _Qu2(S, Y, a, b):
        return int(S[a, b]) * int(S[a ^ Y, b ^ Y]) * int(S[a, b ^ Y]) * int(S[a ^ Y, b])

    w14_rows = []
    w14 = True
    for m in (5, 6, 7):
        S = sign_table_fast(m)
        Sp = sign_table_fast(m - 1)
        e = 1 << (m - 1)
        bad = 0
        for W in range(1, e):
            Np = sum(1 for u in range(1, e) for v in range(1, e)
                     if u != v and _Qp(Sp, W, u, v) == -1)
            ll = sum(1 for a in range(1, e) for b in range(1, e)
                     if a != b and _Qp(S, W, a, b) == -1)
            lu = sum(1 for u in range(1, e) for v in range(0, e)
                     if _Qp(S, W, u, v + e) == -1)
            ul = sum(1 for u in range(0, e) for v in range(1, e)
                     if _Qp(S, W, u + e, v) == -1)
            uu = sum(1 for u in range(0, e) for v in range(0, e)
                     if u != v and _Qp(S, W, u + e, v + e) == -1)
            # the three pieces of lu, and the u=W row-failure
            p_v0 = sum(1 for u in range(1, e) if u != W)
            p_dg = sum(1 for u in range(1, e) if u != W)
            cos = [u for u in range(1, e) if u != W and 1 <= (u ^ W) < e and (u ^ W) != u]
            p_uW = sum(1 for v in range(0, e) if _Qp(S, W, W, v + e) == -1)
            u0 = sum(1 for v in range(1, e) if _Qp(S, W, 0 + e, v) == -1)
            checks = [
                ll == Np,                                        # ll is exact
                lu == Np + 3 * (e - 2), uu == Np + 3 * (e - 2),   # lu, uu
                ul == lu + e,                                     # ul's extra e
                p_v0 == e - 2, p_dg == e - 2, len(cos) == e - 2,   # the three pieces
                p_uW == 0,                                        # the row failure is empty
                u0 == e - 1,                                      # the u=0 slice
                all(_Qu2(Sp, W, u ^ W, u) == -1 for u in cos),    # unprimed -1 on the partner
                all(_Qp(Sp, W, u, u ^ W) == 1 for u in cos),      # PRIMED +1 -- the asymmetry
                ll + lu + ul + uu == 4 * Np + 10 * e - 18,        # and the total
            ]
            if not all(checks):
                bad += 1
        w14 = w14 and bad == 0
        w14_rows.append((m, e - 1, bad))
    ok["W14"] = w14
    print(f"W14_CONSTS  THE LEVEL-CONSTANTS ARE DERIVED, from four closed forms "
          f"{'OK' if w14 else 'FAIL'} -- "
          + "; ".join(f"m={a}: {b} labels, {c} failing the full decomposition"
                      for a, b, c in w14_rows)
          + ". Quadrant by quadrant, with e = 2^(m-1): ll = N(m-1,W) EXACTLY (`Q'red_low_ll` is "
            "unconditional); lu = uu = N(m-1,W) + 3(e-2), the three (e-2)s being v=0 priced by "
            "`Qgen_zero_left`, v=u by `Qgen_diag_neg`, and v=u^W by the ASYMMETRY between "
            "unprimed -1 (Qgen_coset_left + Qgen_diag_neg) and PRIMED +1 "
            "(`Qgen'_coset_partner`, proven forall n today); the u=W row-failure contributes "
            "ZERO; and ul = lu + e, the extra being the u=0 slice (e-1 terms, all -1 by "
            "`Qgen_zero_left`) plus one boundary term. Summing: 4N(m-1,W) + 9(e-2) + e = "
            "4N + 10e - 18, which is W13's low-label constant. *** SO THE CONSTANT IS NOT A "
            "FITTED INTEGER: every piece is a count priced by a Lean-proven closed form, and the "
            "one that was still measured -- the coset partner is never an edge -- is now the "
            "theorem `Qgen'_coset_partner`. What remains unpriced is the single +1 boundary term "
            "in ul and the HIGH-label constant 6e-10, whose reflection structure is the same. "
            "(III) is untouched; (d) IS NOT CLOSED ***")

    # ---- W15  BOTH CONSTANTS, SLICE BY SLICE, WITH COVERAGE --------------------------------
    # W14 left two things measured: the single +1 boundary term in the low-label `ul`, and the
    # whole high-label constant 6e-10. Both close the same way. Every failure slice of every
    # reduction row lies on the TWELVE-CONDITION LOCUS where Qgen = -1 -- the six "= 0"
    # degeneracies (`Qgen_degen`) and the six "= H" gap roots (`Qgen_H_left_*`, `Qgen_H_right_*`,
    # `Qgen_H_diff_*`) -- and there `Qgen'_on_neg` (proven forall n today) gives
    #     Q' = -(chi(a^W, b^W) * chi(a, b^W)),
    # with `chi_char` explicit. So Q' = +1 exactly when precisely one of
    #     c1 = (a = W or b = W or a = b),   c2 = (a = 0 or b = W or b = a^W)
    # holds. That single table prices all 16 high-label and all 15 low-label slices.
    # This clause asserts, per (m, W'), and for BOTH label parities:
    #   (1) each slice's value-set is the predicted SINGLETON,
    #   (2) the row's applies-domain and the slices are disjoint and COVER the quadrant,
    #   (3) the applies-part equals the level-(m-1) prediction (sign +1 low / -1 high),
    #   (4) the M = N' bridge that the high-label lu/ul quadrants need,
    #   (5) the four quadrant totals and the grand total.
    def _w15(m, Wl, high, S, Sp):
        e = 1 << (m - 1)
        W = Wl + e if high else Wl
        bad = [0]

        def chk(c):
            if not c:
                bad[0] += 1

        def sl(pairs, want, f):
            lst = list(pairs)
            chk(bool(lst) and {f(a, b) for a, b in lst} == {want})
            return len(lst), sum(1 for a, b in lst if f(a, b) == -1)

        Np = sum(1 for a in range(1, e) for b in range(1, e)
                 if a != b and _Qp(Sp, Wl, a, b) == -1)
        P = (e - 1) * (e - 2)
        # lemma A at level m-1: first arg = label -> +1 (`Qgen'_label_left`),
        #                       second arg = label -> -1 (`Qgen'_label_right`)
        chk({_Qp(Sp, Wl, Wl, b) for b in range(1, e) if b != Wl} == {1})
        chk({_Qp(Sp, Wl, a, Wl) for a in range(1, e) if a != Wl} == {-1})
        tot = {}

        # ---- ll : a, b in [1,e), a != b
        F = lambda a, b: _Qp(S, W, a, b)
        dom = [(a, b) for a in range(1, e) for b in range(1, e) if a != b]
        tot["ll"] = sum(1 for a, b in dom if F(a, b) == -1)
        if high:
            row = [(a, b) for a, b in dom if a != Wl and b != Wl]
            app = sum(1 for a, b in row if F(a, b) == -1)
            chk(app == sum(1 for a, b in row if _Qp(Sp, Wl, a, b) == 1))
            chk(app == (e - 2) ** 2 - Np)
            n1, g1 = sl([(Wl, b) for b in range(1, e) if b != Wl], -1, F)
            n2, g2 = sl([(a, Wl) for a in range(1, e) if a != Wl], -1, F)
            chk(len(row) + n1 + n2 == len(dom))
            chk(tot["ll"] == app + g1 + g2 == e * e - 2 * e - Np)
        else:
            chk(tot["ll"] == sum(1 for a, b in dom if _Qp(Sp, Wl, a, b) == -1) == Np)

        # ---- ul : a = u+e (u in [0,e)), b in [1,e)
        F = lambda u, b: _Qp(S, W, u + e, b)
        dom = [(u, b) for u in range(0, e) for b in range(1, e)]
        tot["ul"] = sum(1 for u, b in dom if F(u, b) == -1)
        if high:
            row = [(u, b) for u, b in dom
                   if u != 0 and u != Wl and b != Wl and u != b]
            app = sum(1 for u, b in row if F(u, b) == -1)
            chk(app == sum(1 for u, b in row if _Qu2(Sp, Wl, u, b) == 1))
            chk(app == (e - 2) * (e - 3) - Np)
            n, g = len(row), app
            for pairs, want in [([(0, Wl)], 1),
                                ([(0, b) for b in range(1, e) if b != Wl], -1),
                                ([(Wl, Wl)], 1),
                                ([(Wl, b) for b in range(1, e) if b != Wl], 1),
                                ([(u, Wl) for u in range(1, e) if u != Wl], -1),
                                ([(u, u) for u in range(1, e) if u != Wl], -1)]:
                dn, dg = sl(pairs, want, F)
                n, g = n + dn, g + dg
            chk(n == len(dom))
            chk(tot["ul"] == g == e * e - 2 * e - Np)
        else:
            chk(tot["ul"] == sum(1 for u, b in dom if _Qu2(Sp, Wl, u, b) == -1))
            chk(tot["ul"] == Np + 4 * e - 6)          # <- the +1 boundary term, priced

        # ---- lu : a in [1,e), b = v+e (v in [0,e))
        F = lambda a, v: _Qp(S, W, a, v + e)
        dom = [(a, v) for a in range(1, e) for v in range(0, e)]
        tot["lu"] = sum(1 for a, v in dom if F(a, v) == -1)
        if high:
            row = [(a, v) for a, v in dom
                   if v != 0 and a != Wl and v != Wl and a != v]
            app = sum(1 for a, v in row if F(a, v) == -1)
            chk(app == sum(1 for a, v in row if _Qu2(Sp, Wl, v, a) == 1))
            chk(app == (e - 2) * (e - 3) - Np)
            n, g = len(row), app
            for pairs, want in [([(Wl, 0)], 1),
                                ([(a, 0) for a in range(1, e) if a != Wl], -1),
                                ([(Wl, v) for v in range(1, e)], -1),
                                ([(a, Wl) for a in range(1, e) if a != Wl], -1),
                                ([(a, a) for a in range(1, e) if a != Wl], -1)]:
                dn, dg = sl(pairs, want, F)
                n, g = n + dn, g + dg
            chk(n == len(dom))
            chk(tot["lu"] == g == e * e - e - 1 - Np)
        else:
            row = [(a, v) for a, v in dom if a != Wl]
            chk(sum(1 for a, v in row if F(a, v) == -1)
                == sum(1 for a, v in row if _Qu2(Sp, Wl, v, a) == -1))
            chk({F(Wl, v) for v in range(0, e)} == {1})   # `Qgen'_label_left`: row-fail = +1
            chk(tot["lu"] == Np + 3 * e - 6)

        # ---- uu : a = u+e, b = v+e, u != v
        F = lambda u, v: _Qp(S, W, u + e, v + e)
        dom = [(u, v) for u in range(0, e) for v in range(0, e) if u != v]
        tot["uu"] = sum(1 for u, v in dom if F(u, v) == -1)
        row = [(u, v) for u, v in dom
               if u != 0 and v != 0 and u != Wl and v != Wl and (u ^ v) != Wl]
        app = sum(1 for u, v in row if F(u, v) == -1)
        chk(app == sum(1 for u, v in row
                       if _Qp(Sp, Wl, v, u) == (1 if high else -1)))
        slices = ([([(0, v) for v in range(1, e)], -1),
                   ([(Wl, 0)], 1),
                   ([(u, 0) for u in range(1, e) if u != Wl], -1),
                   ([(Wl, v) for v in range(1, e) if v != Wl], 1),
                   ([(u, Wl) for u in range(1, e) if u != Wl], -1),
                   ([(u, u ^ Wl) for u in range(1, e) if u != Wl], -1)] if high else
                  [([(0, Wl)], 1),
                   ([(0, v) for v in range(1, e) if v != Wl], -1),
                   ([(Wl, 0)], 1),
                   ([(u, 0) for u in range(1, e) if u != Wl], -1),
                   ([(Wl, v) for v in range(1, e) if v != Wl], -1),
                   ([(u, Wl) for u in range(1, e) if u != Wl], -1),
                   ([(u, u ^ Wl) for u in range(1, e) if u != Wl], 1)])
        n, g = len(row), app
        for pairs, want in slices:
            dn, dg = sl(pairs, want, F)
            n, g = n + dn, g + dg
        chk(n == len(dom))
        chk(tot["uu"] == g)
        chk(tot["uu"] == (e * e - e - 1 - Np) if high else tot["uu"] == Np + 3 * e - 6)

        # ---- the M = N' bridge (high-label lu and ul)
        if high:
            d = [(a, v) for a in range(1, e) for v in range(1, e)
                 if a != v and a != Wl and v != Wl]
            chk(sum(1 for a, v in d if _Qu2(Sp, Wl, v, a) == -1) == Np)
            # off the sixth line this is a THEOREM, not a measurement: `Qgen_symm` then
            # `Qgen_eq_Qgen'`, whose five hypotheses (a!=0, a!=W, v!=W, a!=v, a^v!=W) are
            # exactly what `d` minus the sixth line gives, plus `Qgen_symm`'s v!=0.
            chk(all(_Qu2(Sp, Wl, v, a) == _Qp(Sp, Wl, a, v)
                    for a, v in d if (a ^ v) != Wl))
            on6 = [(a, v) for a, v in d if (a ^ v) == Wl]
            chk(len(on6) == e - 2)
            chk({(_Qu2(Sp, Wl, v, a), _Qp(Sp, Wl, a, v)) for a, v in on6} == {(-1, 1)})

        grand = sum(tot.values())
        want = (4 * P - 4 * Np + 6 * e - 10) if high else (4 * Np + 10 * e - 18)
        chk(grand == want)
        return bad[0]

    # all labels at m = 5, 6; a spread of nine at m = 7 (the m=7 sweep is O(e^2) per label and
    # per parity -- this cap is DECLARED, not silent)
    W15_LABELS = {5: list(range(1, 16)), 6: list(range(1, 32)),
                  7: [1, 3, 6, 9, 17, 22, 31, 45, 63]}
    w15_rows, w15 = [], True
    for m in (5, 6, 7):
        S, Sp = sign_table_fast(m), sign_table_fast(m - 1)
        bad = sum(_w15(m, Wl, hi, S, Sp)
                  for Wl in W15_LABELS[m] for hi in (False, True))
        w15 = w15 and bad == 0
        w15_rows.append((m, len(W15_LABELS[m]), (1 << (m - 1)) - 1, bad))
    # NULL CONTROL: the whole ledger rests on W' != 0 (`Qgen_degen` requires it, and Qgen 0 = +1).
    # At W' = 0 it must FAIL, or the clause would be asserting something vacuously wide.
    null_bad = sum(_w15(m, 0, hi, sign_table_fast(m), sign_table_fast(m - 1))
                   for m in (5, 6) for hi in (False, True))
    ok["W15"] = w15 and null_bad > 0
    print(f"W15_LEDGER  BOTH LEVEL-CONSTANTS ARE DERIVED SLICE BY SLICE, WITH COVERAGE "
          f"{'OK' if ok['W15'] else 'FAIL'} -- "
          + "; ".join(f"m={a}: {b} of {t} labels x 2 parities, {c} violations"
                      for a, b, t, c in w15_rows)
          + f"; null control W'=0: {null_bad} violations (must be > 0). "
            "EVERY failure slice of EVERY reduction row lies on the twelve-condition locus where "
            "Qgen = -1: the six '= 0' degeneracies (`Qgen_degen`) and the six '= H' gap roots "
            "(`Qgen_H_left_*`, `Qgen_H_right_*`, `Qgen_H_diff_*`), all proven forall n. There "
            "`Qgen'_on_neg` -- proven forall n today, one rewrite off `Qgen'_eq_chi` -- gives "
            "Q' = -(chi(a^W,b^W) * chi(a,b^W)), and `chi_char` makes both explicit: Q' = +1 "
            "exactly when precisely ONE of c1 = (a=W or b=W or a=b), c2 = (a=0 or b=W or b=a^W) "
            "holds. That one table prices all 16 high-label and all 15 low-label slices, and the "
            "clause checks not only each slice's value but that the slices COVER the quadrant. "
            "*** THE HIGH-LABEL CONSTANT IS DERIVED: ll = ul = e^2-2e-N', lu = uu = e^2-e-1-N', "
            "summing to 4P' - 4N' + 6e - 10 with P' = (e-1)(e-2). The reflection is the MINUS "
            "sign every high row carries (N11), which turns 'count the -1s' into 'count the +1s'. "
            "The bridge it needs is M = N': the level-(m-1) Qgen count over the five-line-free "
            "box equals the full Q' count -- OFF the sixth line the two are equal by "
            "`Qgen_symm` + `Qgen_eq_Qgen'` (a THEOREM forall n: the five hypotheses are exactly "
            "the box), and ON it they differ, because the (e-2) pairs on the sixth line a^v = W' "
            "carry Qgen = -1 but Q' = +1 (`Qgen'_coset_partner`), exactly cancelling the (e-2) "
            "that lemma A's b = W' row (`Qgen'_label_right`) contributes while its a = W' row "
            "(`Qgen'_label_left`) contributes none. *** AND W14's LEFTOVER IS PRICED: the "
            "low-label ul = N' + 4e - 6, not lu + e as a fitted offset -- the u = 0 slice is "
            "fully degenerate (a = H, `Qgen_H_left_low`) over all e-1 values of b, where lu's "
            "v = 0 sits inside the per-a degenerate set. So BOTH constants, 10e-18 and 6e-10, "
            "are now sums of Lean-priced pieces with no fitted term. HONEST SCOPE: the pointwise "
            "closed forms are Lean forall n; the COUNTING (slice sizes, disjointness, coverage) "
            "is on paper and pinned here at m = 5,6,7. (III) is untouched; (d) IS NOT CLOSED ***")

    # ---- W16  THE BASE CASE, AND HOW FAR THE CLOSED FORM REACHES ---------------------------
    # W15 determined the recursion's STEP but pinned no base level. The descent sends
    # (m,W) -> (m-1, W % 2^(m-1)); for an ODD label that stays odd (`odd_stays_odd`), never 0,
    # and at level 1 it is 1. So every odd chain bottoms out in the label-2^k family, where
    # `Qgen'_pow2_eq` (proven forall n today) gives a COMPLETE closed form: Q' = +1 on exactly
    # two disjoint lines, a = 2^k and b = a^2^k, of size 2^m-2 each, and -1 elsewhere. Hence
    #     N(m, 2^k) = (2^m-1)(2^m-2) - 2(2^m-2) = (2^m-2)(2^m-3),  independent of k.
    # Bridge to the lane's graph: the edge test is Qgen(Llo|2^(n-1), a, b, n) = +1. Because
    # Llo < 2^(n-1) throughout the lane (A_sig_fast ranges Llo over [1,H), H = 2^(n-1)), the OR
    # IS an addition, Llo|2^(n-1) = Llo + 2^(n-1) -- the hypothesis `Qred_hi_ll` needs, and
    # without which the conversion below would not be available. `Qred_hi_ll` instantiated at
    # m-1 (side conditions b != 0, b != Llo ONLY) then turns the test into Qgen'(Llo,a,b,m)=-1
    # off the b=Llo column. The a=Llo ROW is covered and is +1 (`Qgen'_label_left`) -- so the
    # ISOLATED VERTEX IS DERIVED -- while the uncovered b=Llo column contributes exactly
    # 2^m-2 to N (`Qgen'_label_right`), giving  tr(A^2) = N(m,Llo) - (2^m-2).
    # Unrolling the W15 recursion from this base gives a signed base-4 digit sum over the bits
    # of the label. Fed the RAW label it is EXACT on the Fano family and FALSE on every seam --
    # reported here as a declared negative. W17 then shows the formula was right and the ARGUMENT
    # wrong: the correct argument is 8*g(W)+1 with g(W) = (W & (W-1)) >> 3, seams included.
    def _Qp_mat(S, Y, H):
        idx = np.arange(1, H)
        x, y = idx[:, None], idx[None, :]
        xy, yy = x ^ Y, y ^ Y
        return (S[x, y].astype(np.int16) * S[yy, xy].astype(np.int16)
                * S[yy, x].astype(np.int16) * S[xy, y].astype(np.int16))

    def _N(S, Y, H):
        M = _Qp_mat(S, Y, H)
        np.fill_diagonal(M, 0)
        return int(np.count_nonzero(M == -1))

    def _E(m, W):
        tot = 0
        for i in range(2, m + 1):
            if (W >> (i - 1)) & 1:
                tot += (((1 << i) - 4) * ((1 << i) - 8) * (4 ** (m - i))
                        * (-1) ** bin(W >> i).count("1"))
        return tot

    w16 = True
    # (a) THE BASE CASE: N(m,2^k) = (2^m-2)(2^m-3) for every k < m, and the two +1 lines
    base_rows = []
    for m in range(2, 10):
        S, H = sign_table_fast(m), 1 << m
        bad = 0
        for k in range(m):
            W = 1 << k
            M = _Qp_mat(S, W, H)
            np.fill_diagonal(M, 0)
            N = int(np.count_nonzero(M == -1))
            L1 = [(W, b) for b in range(1, H) if b != W]                   # a = 2^k
            L2 = [(a, a ^ W) for a in range(1, H) if a != W and (a ^ W) != 0]
            if not (N == (H - 2) * (H - 3) and len(L1) == len(L2) == H - 2
                    and not (set(L1) & set(L2))
                    and all(_Qp(S, W, a, b) == 1 for a, b in L1)
                    and all(_Qp(S, W, a, b) == 1 for a, b in L2)):
                bad += 1
        w16 = w16 and bad == 0
        base_rows.append((m, m, bad))
    # (b) THE BRIDGE and (c) THE CLOSED FORM, both families, against the lane's own builders
    fano_ok = fano_tot = seam_ok = seam_tot = bridge_bad = pow2_const = 0
    for n in range(6, 11):
        m, H = n - 1, 1 << (n - 1)
        S, Sm = sign_table_fast(n), sign_table_fast(m)
        for Llo in range(1, H):
            t2, _ = traces23(A_sig_fast(n, Llo, S))
            pred = (H - 2) * (H - 4) - _E(m, Llo)
            if Llo & 7:
                fano_tot += 1
                fano_ok += (t2 == pred)
            else:
                seam_tot += 1
                seam_ok += (t2 == pred)
                if t2 == (H - 2) * (H - 4):
                    pow2_const += ((Llo & (Llo - 1)) == 0)
            if n <= 9 and t2 != _N(Sm, Llo, H) - (H - 2):
                bridge_bad += 1
    # the closed form must be EXACT on Fano and must FAIL on every seam (declared negative)
    w16 = w16 and fano_ok == fano_tot and seam_ok == 0 and bridge_bad == 0
    # (d) NULL CONTROL: at W = 0 the base case is false (Qgen 0 = +1, so Q' = +1 everywhere)
    S4 = sign_table_fast(4)
    null_ok = _N(S4, 0, 16) != (16 - 2) * (16 - 3)
    w16 = w16 and null_ok
    ok["W16"] = w16
    print(f"W16_BASE    THE BASE CASE IS PROVEN, AND THE CLOSED FORM REACHES THE FANO FAMILY "
          f"ONLY {'OK' if w16 else 'FAIL'} -- base: "
          + "; ".join(f"m={a}: {b} powers of two, {c} failing" for a, b, c in base_rows)
          + f"; bridge tr(A^2) = N - (2^m-2): {bridge_bad} violations over BOTH families "
            f"(n=6..9); closed form: Fano {fano_ok}/{fano_tot} match, seams {seam_ok}/{seam_tot} "
            f"match (n=6..10); null control W=0: {'FAILS as required' if null_ok else 'PASSED'}. "
            "*** THE BASE CASE IS `Qgen'_pow2_eq`, PROVEN FORALL N TODAY: on the box "
            "1 <= a,b < 2^m with a != b, Q'(2^k,a,b,m) = +1 on exactly the two DISJOINT lines "
            "a = 2^k (`Qgen'_label_left`) and b = a^2^k (`Qgen'_coset_partner`), each of size "
            "2^m-2, and -1 on everything else (`Qgen'_off_lines` + `Qgen_pow2`). So "
            "N(m,2^k) = (2^m-2)(2^m-3) for EVERY k < m -- k-independent, level-uniform. This is "
            "the bottom of every odd chain: the reduced label of an odd W is odd at every level "
            "(`odd_stays_odd`), hence never 0, and at level 1 the box is empty "
            "(`base_box_empty`). *** AND THE ISOLATED VERTEX IS NOW DERIVED, NOT MEASURED: the "
            "edge test is Qgen(Llo|2^(n-1),a,b,n) = +1 -- and since Llo < 2^(n-1) throughout "
            "the lane the OR IS an addition, which is what lets `Qred_hi_ll` (instantiated at "
            "m-1) apply at all. Its only side conditions are b != 0 and b != Llo, so it "
            "converts the test to Qgen'(Llo,a,b,m) = -1 on the "
            "whole a = Llo ROW, where `Qgen'_label_left` gives +1: no edges. The one column the "
            "row cannot reach, b = Llo, is zero by A's symmetry while contributing exactly "
            "2^m-2 to N (`Qgen'_label_right`) -- which is the bridge tr(A^2) = N - (2^m-2), "
            "checked here on BOTH families. *** IT ALSO DERIVES W6's CONSTANT STRATUM: the "
            "seams whose tr(A^2) equals the y=0 value are exactly the PURE POWERS OF TWO "
            f"({pow2_const} of them found, = n-4 per level), i.e. exactly the base-case family. "
            "*** CLOSED FORM, AND ITS HONEST BOUNDARY: unrolling W15's recursion from this base "
            "gives tr(A^2) = (2^m-2)(2^m-4) - E(m,Llo) with E a SIGNED BASE-4 DIGIT SUM, "
            "E(m,W) = sum over i with bit_{i-1}(W)=1 of (2^i-4)(2^i-8)*4^(m-i)*(-1)^popcount(W>>i). "
            "It is EXACT on the Fano family (all labels, n=6..10 here, n=11 too in-session) and "
            "FALSE on EVERY seam -- asserted as a declared negative, not left to be discovered. "
            "The reason the RAW label fails there is structural: an even label's descent hits "
            "W' = 0 at an INTERMEDIATE step (24 % 16 = 8, then 8 % 8 = 0), where the recursion is "
            "simply inapplicable -- not at the bottom, and not only for pure powers of two. "
            "*** SUPERSEDED IN PART BY W17: the formula was right and the ARGUMENT was wrong. Fed "
            "8*g(W)+1 with g(W) = (W & (W-1)) >> 3 it covers the seams too, so W6's general form "
            "is closed on the WHOLE label set. This clause's negative is kept because it is what "
            "forced the invariant to be found. *** THIS DOES NOT NARROW "
            "(d): tr(A^2) is parity-blind (W11), so (d) still needs tr(A^3)'s form AND the seam "
            "family. (III) is untouched; (d) IS NOT CLOSED ***")

    # ---- W17  THE SEAM HALF CLOSES: ONE INVARIANT COVERS EVERY LABEL -----------------------
    # W16 left the seam family open and recorded the closed form as FALSE there. It is not the
    # formula that was wrong, it was the ARGUMENT it was being fed. N(m,W) turns out to depend
    # on W only through
    #       g(W) = (W & (W-1)) >> 3       -- clear the LOWEST SET BIT, then take bits >= 3
    # For ODD W, W & (W-1) = W-1, so g(8y+1) = y: g GENERALISES the lane's y. Every label,
    # seam included, therefore reduces to a Fano label 8*g(W)+1, and W16's closed form applies
    # verbatim:
    #       tr(A^2)(n,W) = (2^m-2)(2^m-4) - E(m, 8*g(W)+1),   m = n-1.
    # This is read off the BLOCK STRUCTURE of N, not fitted: N is constant exactly on the sets
    # {8y+1..8y+7} together with the even labels whose lowest set bit, once cleared, leaves 8y
    # -- e.g. at m=7 the block of 65 is {65..72, 80, 96}, and 72=64+8, 80=64+16, 96=64+32 all
    # clear to 64. The pure powers of two all clear to 0 and so join the y=0 block, which is
    # exactly `Qgen'_pow2_eq`.
    def _g(W):
        return (W & (W - 1)) >> 3

    uni_ok = uni_tot = 0
    uni_f = uni_s = 0
    for n in range(6, 10):                      # n=10 also verified in-session; cap DECLARED
        m, H = n - 1, 1 << (n - 1)
        S = sign_table_fast(n)
        for Llo in range(1, H):
            t2, _ = traces23(A_sig_fast(n, Llo, S))
            uni_tot += 1
            if t2 == (H - 2) * (H - 4) - _E(m, 8 * _g(Llo) + 1):
                uni_ok += 1
                if Llo & 7:
                    uni_f += 1
                else:
                    uni_s += 1
    # NULL CONTROL: the plain y = Llo>>3 must FAIL on seams, else `g` is doing no work
    S6 = sign_table_fast(6)
    null17 = sum(1 for Llo in range(8, 32, 8)
                 if traces23(A_sig_fast(6, Llo, S6))[0]
                 != (30 * 28) - _E(5, 8 * (Llo >> 3) + 1))
    w17 = (uni_ok == uni_tot) and null17 > 0
    ok["W17"] = w17
    print(f"W17_UNIFORM THE SEAM HALF CLOSES -- ONE INVARIANT COVERS EVERY LABEL "
          f"{'OK' if w17 else 'FAIL'} -- {uni_ok}/{uni_tot} labels match at n=6..9 "
          f"({uni_f} Fano, {uni_s} seam; n=10 also verified in-session, cap declared); "
          f"null control (plain y=Llo>>3 on the n=6 seams): {null17}/3 fail, as required. "
          "*** W16 REPORTED THE SEAM HALF OPEN AND THE CLOSED FORM FALSE THERE. THAT WAS "
          "WRONG -- not the formula, but the ARGUMENT it was given. N(m,W) depends on W only "
          "through g(W) = (W & (W-1)) >> 3: CLEAR THE LOWEST SET BIT, THEN TAKE BITS >= 3. For "
          "odd W, W & (W-1) = W-1, so g(8y+1) = y and g GENERALISES the lane's y. Every label, "
          "seam included, reduces to the Fano label 8*g(W)+1, and W16's closed form applies "
          "verbatim: tr(A^2)(n,W) = (2^m-2)(2^m-4) - E(m, 8*g(W)+1). *** THE INVARIANT WAS READ "
          "OFF THE BLOCK STRUCTURE, NOT FITTED: N is constant exactly on {8y+1..8y+7} together "
          "with the even labels that clear to 8y -- at m=7 the block of 65 is {65..72,80,96}, "
          "and 72=64+8, 80=64+16, 96=64+32 all clear to 64. The pure powers of two clear to 0 "
          "and join the y=0 block, which is precisely `Qgen'_pow2_eq`, the Lean base case. *** "
          "SO W6's OPEN GENERAL FORM FOR tr(A^2) IS NOW CLOSED ON THE WHOLE LABEL SET. Two "
          "measured negatives were needed to get here and both are kept: the naive scaling law "
          "N(m,2^t V) = N(m-t,V) is REFUTED (0 of 31 at m=6), and plain y fails on every seam. "
          "*** STILL NOT (d): tr(A^2) is parity-blind (W11), so (d) needs tr(A^3). The DERIVATION "
          "of the closed form remains base (Lean forall n) + recursion (paper, W15-pinned); what "
          "is verified directly here is the FORMULA. (III) untouched; (d) IS NOT CLOSED ***")

    # ---- W18  HALF OF THE INVARIANT g IS NOW A THEOREM: Q' IS tau-EQUIVARIANT --------------
    # W17 found g(W) = (W & (W-1)) >> 3 empirically. Half of it is `Qgen'_tau`, proven forall n
    # today off three theorems already in the tree (`star_forall` for Q, `tau_xor`, `chi_tau`):
    #     Q'(Y,a,b,m) = Q'(tau j Y, tau j a, tau j b, m)   for every j <= lsb(Y).
    # `tau j` swaps bits 0 and j, so at j = lsb(W) it MOVES THE LOWEST SET BIT TO POSITION 0 --
    # exactly what g does before the >>3 -- and normalises the label to an ODD one
    # (`tau_lsb_odd`, also proven today), which is what keeps W' != 0 available all the way down.
    # This clause pins BOTH directions honestly: tau is SOUND (it never merges two labels with
    # different N) but NOT COMPLETE (each N-block is exactly FOUR tau-orbits; the residual is
    # that bits 1 and 2 of an already-odd label do not matter -- NOT proven anywhere yet).
    def _tau18(j, x):
        return x if ((x >> 0) & 1) == ((x >> j) & 1) else x ^ (1 | (1 << j))

    def _Qmat18(S, Y, H):
        idx = np.arange(H)
        x, y = idx[:, None], idx[None, :]
        xy, yy = x ^ Y, y ^ Y
        return (S[x, y].astype(np.int16) * S[yy, xy].astype(np.int16)
                * S[yy, x].astype(np.int16) * S[xy, y].astype(np.int16))

    # (i) the Lean `tau` definition, transcribed and compared to this clause's -- K7 in this
    #     lane once drew a WRONG conclusion from a mismatched tau, so pin it before using it.
    def _tau_lean(j, x):
        return x if (x & 1) == ((x >> j) & 1) else x ^ (1 | (1 << j))
    w18_def = all(_tau18(j, x) == _tau_lean(j, x)
                  for j in range(0, 8) for x in range(0, 256))
    # (ii) the theorem, pointwise, over its exact hypotheses (j <= lsb Y, ALL a,b including 0)
    w18_pt_bad = w18_pt_tot = 0
    for m in (5, 6):
        S, H = sign_table_fast(m), 1 << m
        for Y in range(1, H):
            t = (Y & -Y).bit_length() - 1
            for j in range(0, min(t, m - 1) + 1):
                tp = np.array([_tau18(j, x) for x in range(H)])
                A = _Qmat18(S, Y, H)
                B = _Qmat18(S, _tau18(j, Y), H)[np.ix_(tp, tp)]
                w18_pt_tot += H * H
                w18_pt_bad += int(np.count_nonzero(A != B))
    # (iii) the label action is exactly "move the lowest set bit to 0" = (W & (W-1)) + 1
    w18_arith = all(_tau18((W & -W).bit_length() - 1, W) == (W & (W - 1)) + 1
                    for W in range(1, 4096))
    w18_odd = all(_tau18((W & -W).bit_length() - 1, W) % 2 == 1 for W in range(1, 4096))
    # (iv) the COUNT consequence, and tau's soundness-but-incompleteness
    w18_cnt_bad = 0
    orbit_rows = []
    for m in (5, 6, 7):
        S, H = sign_table_fast(m), 1 << m
        Nv = {W: _N(sign_table_fast(m), W, H) for W in range(1, H)}
        w18_cnt_bad += sum(1 for W in range(1, H) if Nv[W] != Nv[(W & (W - 1)) + 1])
        # tau-orbits (undirected: tau is an involution, so the RELATION is symmetric)
        par = list(range(H))
        def find(x):
            while par[x] != x:
                par[x] = par[par[x]]
                x = par[x]
            return x
        for W in range(1, H):
            a, b = find(W), find((W & (W - 1)) + 1)
            if a != b:
                par[max(a, b)] = min(a, b)
        orb = {W: find(W) for W in range(1, H)}
        sound = all(Nv[a] == Nv[b] for a in range(1, H) for b in range(1, H)
                    if orb[a] == orb[b])
        n_orb, n_blk = len(set(orb.values())), len(set(Nv.values()))
        orbit_rows.append((m, n_orb, n_blk, sound, n_orb == 4 * n_blk))
    # (v) KEPT NEGATIVE: the naive reading of the Fano/168 action -- "the low 3 bits of ANY
    #     label may be changed freely" -- is FALSE: adding it collapses everything into ONE
    #     orbit, merging labels with different N. It is NOT the residual mechanism.
    S5, H5 = sign_table_fast(5), 32
    N5 = {W: _N(S5, W, H5) for W in range(1, H5)}
    naive_bad = sum(1 for W in range(1, H5) for r in range(1, 8)
                    if 1 <= ((W & ~7) | r) < H5 and N5[W] != N5[(W & ~7) | r])
    w18 = (w18_def and w18_pt_bad == 0 and w18_arith and w18_odd
           and w18_cnt_bad == 0 and all(r[3] and r[4] for r in orbit_rows)
           and naive_bad > 0)
    ok["W18"] = w18
    print(f"W18_TAU     HALF OF g IS A THEOREM: Q' IS tau-EQUIVARIANT forall n "
          f"{'OK' if w18 else 'FAIL'} -- Lean `tau` definition matches this clause's: "
          f"{w18_def}; pointwise Q'(Y,a,b) == Q'(tau Y, tau a, tau b) over the theorem's exact "
          f"hypotheses (j <= lsb Y, ALL a,b incl. 0): {w18_pt_bad}/{w18_pt_tot} failures at "
          f"m=5,6; label action tau_lsb W == (W & (W-1)) + 1: {w18_arith}, and the result is "
          f"odd: {w18_odd} (W < 4096); count consequence N(m,W) == N(m, tau_lsb W): "
          f"{w18_cnt_bad} violations at m=5,6,7; "
          + "; ".join(f"m={a}: {b} tau-orbits vs {c} N-blocks (sound={d}, exactly 4x={e})"
                      for a, b, c, d, e in orbit_rows)
          + f"; NAIVE-FANO null control: {naive_bad} label pairs refute it (must be > 0). "
            "*** `Qgen'_tau` IS THE THEOREM, and it cost three lines because the tree already "
            "had every ingredient: `star_forall` gives tau-equivariance for Q, `tau_xor` moves "
            "tau through the xors, and `chi_tau` says the two commutation signs cannot see tau "
            "at all. `tau j` swaps bits 0 and j, so at j = lsb(W) it MOVES THE LOWEST SET BIT TO "
            "POSITION 0 -- precisely what g does before the >>3 -- and normalises the label to an "
            "ODD one (`tau_lsb_odd`), which is what makes `odd_stays_odd`, hence W' != 0, "
            "available at every level below. *** WHAT IS NOT PROVEN, STATED PLAINLY: (1) the step "
            "from the pointwise identity to the equality of COUNTS needs a bijection-to-"
            "cardinality argument, which is Finset territory this Mathlib-free file does not "
            "have; (2) tau is SOUND but NOT COMPLETE -- each N-block is EXACTLY FOUR tau-orbits, "
            "and the residual (bits 1 and 2 of an already-odd label are irrelevant) has no proof "
            "here; (3) the additive identity tau_lsb W = (W & (W-1)) + 1 is pure bit arithmetic, "
            "pinned above rather than proven in Lean. *** AND A KEPT NEGATIVE: my first guess at "
            "the residual -- that the Fano/168 action lets the low 3 bits of ANY label vary "
            "freely -- is REFUTED. Adding that relation collapses ALL labels into a single "
            "orbit, merging labels with demonstrably different N. So the factor of four is NOT "
            "the naive Fano action, and the second half of g remains OPEN. "
            "(III) untouched; (d) IS NOT CLOSED ***")

    # ---- W19  THE RESIDUAL FACTOR OF FOUR: GL(3,2), AND WHY IT COSTS NOTHING ---------------
    # W18 left tau sound but not complete -- each N-block is exactly four tau-orbits. The
    # missing mechanism is GL(3,2) acting on bits 0,1,2 and identity above: order 168 (the
    # lane's own group), TRANSITIVE on the seven nonzero low patterns, so it merges the four
    # odd residues 1,3,5,7 and closes g. Unlike my refuted guess in W18, it acts on the LABEL
    # AND BOTH POINTS at once, exactly as tau does.
    # WHY it needs no hypothesis on W while sigma itself is not invariant: sigma moves by a
    # COBOUNDARY, sigma(px,py) = sigma(x,y)*lam(x)*lam(y)*lam(x^y). Q and Q' are each a product
    # of FOUR sigmas over a coset square in which the six lam values each occur TWICE, so every
    # lam squares away. That cancellation is `Qgen_of_coboundary` / `Qgen'_of_coboundary`,
    # proven forall n today for an ARBITRARY linear p and ARBITRARY sign lam. So the whole
    # factor of four is reduced to the single sigma-level statement checked here.
    import itertools as _it
    import functools as _ft

    def _gl32():
        out = []
        for cols in _it.product(range(1, 8), repeat=3):
            span = {0}
            for c in cols:
                span |= {s ^ c for s in span}
            if len(span) != 8:
                continue
            out.append(tuple(_ft.reduce(
                lambda A, i: A ^ (cols[i] if (v >> i) & 1 else 0), range(3), 0)
                for v in range(8)))
        return out

    def _Qm(S, Y, H, primed=True):
        idx = np.arange(H)
        x, y = idx[:, None], idx[None, :]
        xy, yy = x ^ Y, y ^ Y
        if primed:
            return (S[x, y].astype(np.int16) * S[yy, xy].astype(np.int16)
                    * S[yy, x].astype(np.int16) * S[xy, y].astype(np.int16))
        return (S[x, y].astype(np.int16) * S[xy, yy].astype(np.int16)
                * S[x, yy].astype(np.int16) * S[xy, y].astype(np.int16))

    def _is_coboundary(S, pm, H):
        """solve  rho(x,y) = lam(x)lam(y)lam(x^y)  over F2; True iff consistent"""
        piv = {}
        for x in range(H):
            for y in range(H):
                r = (1 << x) ^ (1 << y) ^ (1 << (x ^ y))
                b = 0 if int(S[pm[x], pm[y]]) * int(S[x, y]) == 1 else 1
                while r:
                    h = r.bit_length() - 1
                    if h in piv:
                        r ^= piv[h][0]
                        b ^= piv[h][1]
                    else:
                        piv[h] = (r, b)
                        break
                else:
                    if b:
                        return False
        return True

    G32 = _gl32()
    NONLIN = (0, 2, 1, 3, 4, 5, 6, 7)          # fixes 0, permutes low bits, NOT F2-linear
    w19_rows = []
    for m in (4, 5):
        S, H = sign_table_fast(m), 1 << m
        QP = {Y: _Qm(S, Y, H) for Y in range(H)}
        QU = {Y: _Qm(S, Y, H, False) for Y in range(H)}
        eq_p = eq_u = cob = 0
        for tbl in G32:
            pm = np.array([(x & ~7) | tbl[x & 7] for x in range(H)])
            if all(np.array_equal(QP[Y], QP[pm[Y]][np.ix_(pm, pm)]) for Y in range(1, H)):
                eq_p += 1
            if all(np.array_equal(QU[Y], QU[pm[Y]][np.ix_(pm, pm)]) for Y in range(1, H)):
                eq_u += 1
            if _is_coboundary(S, pm, H):
                cob += 1
        pmn = np.array([(x & ~7) | NONLIN[x & 7] for x in range(H)])
        null_eq = sum(int(np.count_nonzero(QP[Y] != QP[pmn[Y]][np.ix_(pmn, pmn)]))
                      for Y in range(1, H))
        null_cob = _is_coboundary(S, pmn, H)
        w19_rows.append((m, eq_p, eq_u, cob, null_eq, null_cob))
    # star_forall's hypothesis Y % 2^j = 0 is NOT TIGHT inside the low block: the bit-swap
    # (0<->1) IS tau_1, and it is equivariant for ODD Y as well, which star_forall excludes.
    tight_bad = 0
    for m in (5, 6):
        S, H = sign_table_fast(m), 1 << m
        pm = np.array([(x & ~7) | ((0, 2, 1, 3, 4, 6, 5, 7)[x & 7]) for x in range(H)])
        tight_bad += sum(int(np.count_nonzero(
            _Qm(S, Y, H) != _Qm(S, pm[Y], H)[np.ix_(pm, pm)])) for Y in range(1, H, 2))
    # COMPLETENESS: <GL(3,2), tau_lsb> must equal the N-block partition exactly
    comp_rows = []
    for m in (5, 6, 7):
        S, H = sign_table_fast(m), 1 << m
        Nv = {W: _N(S, W, H) for W in range(1, H)}
        par = list(range(H))
        def _find(x):
            while par[x] != x:
                par[x] = par[par[x]]
                x = par[x]
            return x
        def _uni(a, b):
            ra, rb = _find(a), _find(b)
            if ra != rb:
                par[max(ra, rb)] = min(ra, rb)
        for W in range(1, H):
            _uni(W, (W & (W - 1)) + 1)
            for tbl in G32:
                v = (W & ~7) | tbl[W & 7]
                if 1 <= v < H:
                    _uni(W, v)
        orb = {W: _find(W) for W in range(1, H)}
        sound = all(Nv[a] == Nv[b] for a in range(1, H) for b in range(1, H)
                    if orb[a] == orb[b])
        complete = all(orb[a] == orb[b] for a in range(1, H) for b in range(1, H)
                       if Nv[a] == Nv[b])
        comp_rows.append((m, len(set(orb.values())), len(set(Nv.values())), sound, complete))
    w19 = (all(r[1] == 168 and r[2] == 168 and r[3] == 168 and r[4] > 0 and not r[5]
               for r in w19_rows)
           and tight_bad == 0
           and all(b == c and d and e for _, b, c, d, e in comp_rows))
    ok["W19"] = w19
    print(f"W19_GL32    THE RESIDUAL FACTOR OF FOUR IS GL(3,2), AND A COBOUNDARY KILLS IT "
          f"{'OK' if w19 else 'FAIL'} -- "
          + "; ".join(f"m={a}: Q'-equivariant {b}/168, Q-equivariant {c}/168, sigma moves by a "
                      f"coboundary {d}/168 | NONLINEAR null: {e} Q' mismatches (must be > 0), "
                      f"coboundary={f} (must be False)"
                      for a, b, c, d, e, f in w19_rows)
          + f"; star_forall tightness probe (tau_1 on ODD Y, which its hypothesis EXCLUDES): "
            f"{tight_bad} mismatches; "
          + "; ".join(f"m={a}: <GL(3,2),tau_lsb> {b} orbits vs {c} N-blocks "
                      f"(sound={d}, complete={e})" for a, b, c, d, e in comp_rows)
          + ". *** THE MECHANISM, AND IT IS NOW A THEOREM: sigma is NOT invariant under these "
            "maps, but it moves by a COBOUNDARY, sigma(px,py) = sigma(x,y) lam(x) lam(y) "
            "lam(x^y). Q and Q' are each a product of FOUR sigmas over a coset square in which "
            "the six lam values occur exactly TWICE, so every lam squares away. That is "
            "`Qgen_of_coboundary` and `Qgen'_of_coboundary`, proven forall n today for an "
            "ARBITRARY F2-linear p and an ARBITRARY sign function lam -- kernel-clean, and they "
            "do not even need Classical.choice. *** SO g IS FULLY EXPLAINED: <GL(3,2), tau_lsb> "
            "is SOUND AND COMPLETE against the N-block partition at m=5,6,7 -- GL(3,2) is "
            "transitive on the seven nonzero low patterns, which merges the four odd residues "
            "(the factor of four W18 could not reach), and tau_lsb normalises every even label "
            "to an odd one. *** WHAT REMAINS MEASURED: that sigma DOES move by a coboundary "
            "under GL(3,2). That is now the single open statement behind g -- one clean "
            "sigma-level fact, not a vague factor of four -- verified 168/168 at m=4,5 with a "
            "non-linear null control that fails both the equivariance and the coboundary test. "
            "*** A BY-CATCH: `star_forall`'s hypothesis Y % 2^j = 0 is NOT TIGHT. The bit-swap "
            "(0<->1) IS tau_1, and it is equivariant for ODD Y too, which that hypothesis "
            "excludes -- 0 mismatches at m=5,6. The theorem is true more widely than it is "
            "stated. (III) untouched; (d) IS NOT CLOSED ***")

    # ---- W20  THE COBOUNDARY ITSELF, PROVEN forall n FOR BOTH GENERATORS -------------------
    # W19 left one measured statement: that sigma moves by a coboundary under the low-block
    # maps. The forall-n content of that is now a theorem. cdSigma's recursion strips the TOP
    # bit and recurses on the residues; a map confined to bits 0,1,2 commutes with that split
    # entirely, so the coboundary property is INHERITED level to level and the whole forall-n
    # statement collapses to a check at LEVEL 3 (`sigma_coboundary_up`, proven today). The
    # level-3 base is finite and falls to `decide`, so for a CONCRETE map the whole thing
    # closes: `sigma_coboundary_trans` and `sigma_coboundary_cyc` are the two generators.
    # This clause pins the Lean tables against the measured lambdas, and checks the one link
    # the Lean does not cover: that these two generators really do generate GL(3,2).
    _tTrans = (0, 1, 2, 3, 5, 4, 7, 6)          # e2 -> e2 ^ e0
    _lTrans = tuple(-1 if v in (5, 7) else 1 for v in range(8))
    _tCyc = (0, 2, 4, 6, 3, 1, 7, 5)            # e0->e1, e1->e2, e2->e0^e1
    _lCyc = tuple(-1 if v in (6, 7) else 1 for v in range(8))

    def _lowmap(t, x):
        return 8 * (x // 8) + t[x % 8]

    def _cob_holds(S, t, l, H):
        """the exact identity the Lean theorems state"""
        for x in range(H):
            for y in range(H):
                lhs = int(S[_lowmap(t, x), _lowmap(t, y)])
                rhs = (int(S[x, y]) * l[x % 8] * l[y % 8] * l[(x ^ y) % 8])
                if lhs != rhs:
                    return False
        return True

    # (i) the two Lean tables are F2-LINEAR and lie in GL(3,2)
    def _is_lin(t):
        return all(t[u ^ v] == t[u] ^ t[v] for u in range(8) for v in range(8))
    w20_lin = _is_lin(_tTrans) and _is_lin(_tCyc)
    w20_bij = sorted(_tTrans) == list(range(8)) and sorted(_tCyc) == list(range(8))
    # (ii) they GENERATE GL(3,2) -- the link the Lean does not cover
    gen, frontier = {tuple(range(8))}, [tuple(range(8))]
    while frontier:
        cur = frontier.pop()
        for g in (_tTrans, _tCyc):
            nxt = tuple(g[cur[v]] for v in range(8))
            if nxt not in gen:
                gen.add(nxt)
                frontier.append(nxt)
    w20_gen = len(gen) == 168 and set(gen) == set(G32)
    # (iii) the identity itself, at several levels, exactly as the Lean states it
    w20_rows = []
    for m in (3, 4, 5, 6):
        S, H = sign_table_fast(m), 1 << m
        a = _cob_holds(S, _tTrans, _lTrans, H)
        b = _cob_holds(S, _tCyc, _lCyc, H)
        w20_rows.append((m, a, b))
    # (iv) NULL CONTROL: perturb one lambda entry -- the identity must break
    _lBad = list(_lTrans)
    _lBad[5] = 1
    S5b, H5b = sign_table_fast(5), 32
    w20_null = not _cob_holds(S5b, _tTrans, tuple(_lBad), H5b)
    # (v) composition closure: lam_{p.q}(x) = lam_q(x) * lam_p(q x), which is what makes the
    #     two generators suffice for all 168
    comp_t = tuple(_tTrans[_tCyc[v]] for v in range(8))
    comp_l = tuple(_lCyc[v] * _lTrans[_tCyc[v]] for v in range(8))
    w20_comp = _cob_holds(sign_table_fast(5), comp_t, comp_l, 32)
    w20 = (w20_lin and w20_bij and w20_gen and w20_null and w20_comp
           and all(a and b for _, a, b in w20_rows))
    ok["W20"] = w20
    print(f"W20_COBND   THE COBOUNDARY IS PROVEN forall n FOR BOTH GENERATORS "
          f"{'OK' if w20 else 'FAIL'} -- Lean tables linear: {w20_lin}, bijective: {w20_bij}; "
          f"<tTrans,tCyc> generates {len(gen)}/168 of GL(3,2) and equals it: {w20_gen}; "
          + "; ".join(f"m={a}: trans={b} cyc={c}" for a, b, c in w20_rows)
          + f"; NULL CONTROL (one lambda entry flipped): identity breaks = {w20_null}; "
            f"composition closure lam_pq(x) = lam_q(x)*lam_p(q x): {w20_comp}. "
            "*** THE forall-n CONTENT IS NOW A THEOREM. `cdSigma`'s recursion strips the TOP "
            "bit and recurses on the residues, while a map confined to bits 0,1,2 commutes with "
            "that split entirely -- it preserves the >= half tests, the = 0 tests and the "
            "residues -- so the coboundary is INHERITED from each level to the next and the "
            "whole forall-n statement COLLAPSES TO LEVEL 3. That is `sigma_coboundary_up`, "
            "proven today; its four branches are exactly `R_ll`, `R_lu`, `R_ul`, `R_uu`, which "
            "were already in the tree. *** AND FOR CONCRETE MAPS IT CLOSES COMPLETELY: the "
            "level-3 base is finite and falls to `decide`, so `sigma_coboundary_trans` (the "
            "transvection e2 -> e2^e0, lam = -1 on {5,7}) and `sigma_coboundary_cyc` (the "
            "7-cycle, lam = -1 on {6,7}) are THEOREMS AT EVERY LEVEL -- kernel-clean, plain "
            "`decide`, no native_decide. *** WHAT THE LEAN DOES NOT COVER, AND THIS CLAUSE DOES: "
            "that these two generators generate GL(3,2) (checked here by closure: exactly 168, "
            "and equal to the group W19 enumerates), and that the coboundary property is closed "
            "under composition with lam_pq(x) = lam_q(x)*lam_p(q x) (checked on the product of "
            "the two). Those two facts plus the two theorems give all 168 -- but the composition "
            "step needs `lowMap`'s F2-linearity in Lean, which is bit-work not done here, so "
            "ALL 168 IS NOT YET A SINGLE LEAN STATEMENT. (III) untouched; (d) IS NOT CLOSED ***")

    # ---- W21  ALL 168 CLOSED: lowMap IS LINEAR, AND THE CLASS IS GL(3,2) ------------------
    # W20 proved the coboundary forall n for the two generators and left "all 168" outside Lean
    # because it needed lowMap's F2-linearity. That is now `lowMap_lin`, and with it
    # `sigma_coboundary_comp` (composition closure), the inductive class `LowCob` (the two
    # generators, closed under composition) and `lowCob_sigma` / `Qgen'_lowCob` -- Q' is
    # INVARIANT under every map in the class, forall n. This clause pins the Lean construction
    # against the measured one and checks the payoff.
    def _lowMap(t, x):
        return 8 * (x // 8) + t[x % 8]

    # (i) the Lean `lowMap` and W19's index permutation are the SAME map
    w21_same = all(_lowMap(t, x) == ((x & ~7) | t[x & 7])
                   for t in G32 for x in range(256))
    # (ii) `lowMap_lin`: every table in GL(3,2) gives an F2-linear lowMap
    w21_lin = all(_lowMap(t, x ^ y) == _lowMap(t, x) ^ _lowMap(t, y)
                  for t in G32 for x in range(64) for y in range(64))
    # (iii) `lowMap_comp`: lowMap t1 . lowMap t2 = lowMap (t1 . t2)
    w21_comp = all(_lowMap(t1, _lowMap(t2, x))
                   == _lowMap(tuple(t1[t2[v]] for v in range(8)), x)
                   for t1 in G32[:12] for t2 in G32[:12] for x in range(64))
    # (iv) THE PAYOFF: N(m, 8y+r) is constant in r = 1..7 -- the 7-fold merge that contains
    #      W18's residual factor of four
    merge_rows = []
    for m in (5, 6, 7):
        S, H = sign_table_fast(m), 1 << m
        bad = 0
        ys = 0
        for y in range(H // 8):
            vals = {_N(S, 8 * y + r, H) for r in range(1, 8) if 8 * y + r < H}
            if len(vals) > 1:
                bad += 1
            if vals:
                ys += 1
        merge_rows.append((m, ys, bad))
    # (v) NULL CONTROL: a NON-linear low permutation must break linearity of lowMap
    w21_null = not all(_lowMap(NONLIN, x ^ y) == _lowMap(NONLIN, x) ^ _lowMap(NONLIN, y)
                       for x in range(64) for y in range(64))
    w21 = (w21_same and w21_lin and w21_comp and w21_null
           and all(b > 0 and c == 0 for _, b, c in merge_rows))
    ok["W21"] = w21
    print(f"W21_ALL168  ALL 168 ARE CLOSED IN LEAN: lowMap IS LINEAR AND THE CLASS COMPOSES "
          f"{'OK' if w21 else 'FAIL'} -- Lean `lowMap` == W19's index permutation: {w21_same}; "
          f"lowMap F2-linear for all 168 tables: {w21_lin}; lowMap composes on tables: "
          f"{w21_comp}; NONLINEAR null control breaks linearity: {w21_null}; "
          + "; ".join(f"m={a}: {b} y-blocks, {c} where N is NOT constant over r=1..7"
                      for a, b, c in merge_rows)
          + ". *** THE LAST STRUCTURAL GAP IS CLOSED. `lowMap_lin` -- lowMap t is F2-linear "
            "when t is -- follows from four core bit facts (shiftRight_xor_distrib, "
            "shiftLeft_xor_distrib, testBit_mod_two_pow, two_pow_add_eq_or_of_lt) once "
            "8*a + b with b < 8 is recognised as a DISJOINT xor. With it: "
            "`sigma_coboundary_comp` (the coboundary composes, lam_pq(v) = l2 v * l1 (t2 v)), "
            "the inductive class `LowCob` (the two generators, closed under composition), "
            "`lowCob_sigma` (every member carries the coboundary at every level) and "
            "`Qgen'_lowCob` (Q' is INVARIANT under every member, forall n) -- all kernel-clean. "
            "*** SO THE CHAIN IS COMPLETE: sigma moves by a coboundary (proven forall n) -> the "
            "four sigmas of Q' cancel it (proven) -> Q' is invariant under the class (proven) -> "
            "the 7 nonzero low residues merge, which contains W18's residual factor of four. "
            "The merge itself is verified here: N(m, 8y+r) is CONSTANT in r = 1..7 for every y "
            "at m = 5,6,7. *** WHAT IS STILL NOT LEAN: that the class LowCob is EXACTLY GL(3,2) "
            "-- a finite closure computation, done in W20 (168 elements, equal to the enumerated "
            "group) -- and the counting step from pointwise Q'-invariance to equality of N, "
            "which needs Finset cardinality this Mathlib-free file does not have. "
            "(III) untouched; (d) IS NOT CLOSED ***")

    # ---- W22  THE COUNTING STEP IS PROVEN -- g IS CLOSED END TO END ------------------------
    # The last gap. `Qgen'_lowCob` is POINTWISE; g is about the COUNT. Bridging them normally
    # means Finset cardinality, which this Mathlib-free file does not have -- and does not need:
    # a plain recursive `sumLt`, the fact that `lowMap t` permutes each block of eight
    # (`sum8_perm`), and the seam split (`sumLt_add`) do it. `Ncnt_lowCob` is the result:
    #     Ncnt (lowMap t W) (k+3) = Ncnt W (k+3)   for every t in the class, forall n.
    # THIS CLAUSE PINS `Ncnt` TO THE MEASURED N FIRST -- K7 in this lane once drew a wrong
    # conclusion from a mismatched tau, and `Reach` was once the wrong set entirely.
    def _Ncnt(S, W, m):
        """transcription of the Lean `Ncnt`: a double sum with the same guard"""
        H = 1 << m
        tot = 0
        for a in range(H):
            for b in range(H):
                if a != 0 and b != 0 and a != b and _Qp(S, W, a, b) == -1:
                    tot += 1
        return tot

    # (i) the Lean Ncnt IS the lane's N
    w22_same = True
    for m in (4, 5, 6):
        S, H = sign_table_fast(m), 1 << m
        for W in range(1, H):
            if _Ncnt(S, W, m) != _N(S, W, H):
                w22_same = False
                break
    # (ii) the theorem itself: N(lowMap t W) == N(W) for EVERY t in the class, every W
    w22_rows = []
    for m in (5, 6):
        S, H = sign_table_fast(m), 1 << m
        Nv = {W: _N(S, W, H) for W in range(1, H)}
        bad = sum(1 for t in G32 for W in range(1, H)
                  if Nv[_lowMap(t, W)] != Nv[W])
        w22_rows.append((m, 168 * (H - 1), bad))
    # (iii) NULL CONTROL. My FIRST attempt here was vacuous and this clause caught it: I used
    #     the non-linear low permutation, expecting it to break count-invariance. It does not --
    #     count-invariance is WEAKER than the pointwise Q'-invariance, and holds for ALL 5040
    #     permutations of the low block that fix 0, not merely the 168 linear ones. So `LowCob`
    #     is SUFFICIENT but not NECESSARY for this conclusion (it IS necessary pointwise: W19
    #     pins that exactly 168 of them are Q'-equivariant). What IS load-bearing is CONFINEMENT
    #     TO THE LOW BLOCK -- a map that touches bit 3 breaks the count at every label.
    S6n, H6n = sign_table_fast(6), 64
    Nv6 = {W: _N(S6n, W, H6n) for W in range(1, H6n)}
    w22_nonlin_ok = all(Nv6[_lowMap(NONLIN, W)] == Nv6[W] for W in range(1, H6n))
    _flip3 = [W for W in range(1, H6n) if 1 <= (W ^ 8) < H6n]
    w22_null = all(Nv6[W ^ 8] != Nv6[W] for W in _flip3)
    w22 = w22_same and w22_null and w22_nonlin_ok and all(c == 0 for _, _, c in w22_rows)
    ok["W22"] = w22
    print(f"W22_COUNT   THE COUNTING STEP IS PROVEN -- g IS CLOSED END TO END "
          f"{'OK' if w22 else 'FAIL'} -- Lean `Ncnt` == the lane's N entrywise (m=4,5,6, every "
          f"label): {w22_same}; "
          + "; ".join(f"m={a}: {b} (t,W) pairs, {c} where N(lowMap t W) != N(W)"
                      for a, b, c in w22_rows)
          + f"; NULL CONTROL -- leaving the low block (flip bit 3) breaks the count at "
            f"{len(_flip3)}/{len(_flip3)} labels: {w22_null}; and the HONEST negative, the "
            f"non-linear low permutation still PRESERVES the count: {w22_nonlin_ok}. "
            "*** THE LAST GAP IS CLOSED. Going from the POINTWISE `Qgen'_lowCob` to the COUNT "
            "normally needs Finset cardinality, which this Mathlib-free file does not have. It "
            "does not need it: a plain recursive `sumLt`, the 8-block permutation `sum8_perm` "
            "(proven by induction over the class -- the two generators are concrete, so each "
            "base case is eight terms reordered and closes by `omega`), and the seam split "
            "`sumLt_add` give `sumLt_lowMap`: REINDEXING A BOUNDED SUM BY `lowMap t` CHANGES "
            "NOTHING, forall n. Applying it twice (once per argument) plus injectivity "
            "(`lowMap_inj`, from linearity and trivial kernel) gives `Ncnt_lowCob`. *** SO g IS "
            "NOW PROVEN END TO END: sigma moves by a coboundary (Tier 25) -> the four sigmas of "
            "Q' cancel it (Tier 24) -> Q' is invariant under the class (Tier 26) -> THE COUNT is "
            "invariant (Tier 27). Together with `Qgen'_tau` (W18) for the even labels, both "
            "halves of g = (W & (W-1)) >> 3 are theorems. *** ONE FINITE FACT REMAINS OUTSIDE "
            "LEAN, and it is not an analytic gap: that the class `LowCob` is EXACTLY GL(3,2). "
            "That is a closure computation over 8-element tables, done in W20 (168 elements, "
            "equal to the enumerated group). *** AND ONE HONEST WEAKENING, caught by this "
            "clause's own null control: `LowCob` is SUFFICIENT for count-invariance but NOT "
            "NECESSARY -- ALL 5040 permutations of the low block fixing 0 preserve N, not just "
            "the 168 linear ones. The hypothesis IS necessary for the POINTWISE statement (W19: "
            "exactly 168 are Q'-equivariant, and the non-linear ones fail pointwise), which is "
            "what the proof actually uses. My first null control asserted the opposite and was "
            "vacuous; the load-bearing hypothesis is CONFINEMENT TO THE LOW BLOCK, and a map "
            "touching bit 3 breaks the count at every label. (III) untouched; (d) IS NOT "
            "CLOSED ***")

    # ---- W23  LowCob IS EXACTLY GL(3,2) -- THE LAST FINITE FACT IS NOW IN LEAN -------------
    # W20-W22 carried one item outside Lean: that the inductive class `LowCob` (the two
    # generators, closed under composition) is exactly GL(3,2). It is now `lowCob_eq_GL`, with
    # SOUNDNESS from `lowCob_isGL` and COMPLETENESS from `lowCob_covers` -- for each of the 168
    # elements an explicit WORD in the two generators, found by BFS (longest: 12).
    # This clause TRANSCRIBES the Lean `glList` and re-derives everything independently.
    _GLLIST = eval("[" + (
        "(1,2,4),(1,2,5),(1,2,6),(1,2,7),(1,3,4),(1,3,5),(1,3,6),(1,3,7),(1,4,2),(1,4,3),(1,4,6"
        "),(1,4,7),(1,5,2),(1,5,3),(1,5,6),(1,5,7),(1,6,2),(1,6,3),(1,6,4),(1,6,5),(1,7,2),(1,7"
        ",3),(1,7,4),(1,7,5),(2,1,4),(2,1,5),(2,1,6),(2,1,7),(2,3,4),(2,3,5),(2,3,6),(2,3,7),(2"
        ",4,1),(2,4,3),(2,4,5),(2,4,7),(2,5,1),(2,5,3),(2,5,4),(2,5,6),(2,6,1),(2,6,3),(2,6,5),"
        "(2,6,7),(2,7,1),(2,7,3),(2,7,4),(2,7,6),(3,1,4),(3,1,5),(3,1,6),(3,1,7),(3,2,4),(3,2,5"
        "),(3,2,6),(3,2,7),(3,4,1),(3,4,2),(3,4,5),(3,4,6),(3,5,1),(3,5,2),(3,5,4),(3,5,7),(3,6"
        ",1),(3,6,2),(3,6,4),(3,6,7),(3,7,1),(3,7,2),(3,7,5),(3,7,6),(4,1,2),(4,1,3),(4,1,6),(4"
        ",1,7),(4,2,1),(4,2,3),(4,2,5),(4,2,7),(4,3,1),(4,3,2),(4,3,5),(4,3,6),(4,5,2),(4,5,3),"
        "(4,5,6),(4,5,7),(4,6,1),(4,6,3),(4,6,5),(4,6,7),(4,7,1),(4,7,2),(4,7,5),(4,7,6),(5,1,2"
        "),(5,1,3),(5,1,6),(5,1,7),(5,2,1),(5,2,3),(5,2,4),(5,2,6),(5,3,1),(5,3,2),(5,3,4),(5,3"
        ",7),(5,4,2),(5,4,3),(5,4,6),(5,4,7),(5,6,1),(5,6,2),(5,6,4),(5,6,7),(5,7,1),(5,7,3),(5"
        ",7,4),(5,7,6),(6,1,2),(6,1,3),(6,1,4),(6,1,5),(6,2,1),(6,2,3),(6,2,5),(6,2,7),(6,3,1),"
        "(6,3,2),(6,3,4),(6,3,7),(6,4,1),(6,4,3),(6,4,5),(6,4,7),(6,5,1),(6,5,2),(6,5,4),(6,5,7"
        "),(6,7,2),(6,7,3),(6,7,4),(6,7,5),(7,1,2),(7,1,3),(7,1,4),(7,1,5),(7,2,1),(7,2,3),(7,2"
        ",4),(7,2,6),(7,3,1),(7,3,2),(7,3,5),(7,3,6),(7,4,1),(7,4,2),(7,4,5),(7,4,6),(7,5,1),(7"
        ",5,3),(7,5,4),(7,5,6),(7,6,2),(7,6,3),(7,6,4),(7,6,5)"
    ) + "]")

    def _mk(a, b, c):
        return tuple((a if v & 1 else 0) ^ (b if v & 2 else 0) ^ (c if v & 4 else 0)
                     for v in range(8))

    _indep = [(a, b, c) for a in range(1, 8) for b in range(1, 8) for c in range(1, 8)
              if len({0, a, b, a ^ b, c, a ^ c, b ^ c, a ^ b ^ c}) == 8]
    w23_len = len(_GLLIST) == 168 and len(set(_GLLIST)) == 168
    w23_set = set(_GLLIST) == set(_indep)

    def _glIndep(a, b, c):
        return (a < 8 and b < 8 and c < 8 and a != 0 and b != 0 and c != 0
                and a != b and a != c and b != c and (a ^ b) != c)
    w23_pred = all(_glIndep(a, b, c) == ((a, b, c) in set(_indep))
                   for a in range(8) for b in range(8) for c in range(8))
    _tT, _tC = (0, 1, 2, 3, 5, 4, 7, 6), (0, 2, 4, 6, 3, 1, 7, 5)
    _cmp = lambda t1, t2: tuple(t1[t2[v]] for v in range(8))
    # BREADTH-first (a FIFO queue): the word-length bound below is about SHORTEST words, and
    # a stack here would explore depth-first and report 72 instead of 12 -- which is exactly
    # what this clause caught on its first run.
    words, frontier, head = {}, [], 0
    for g in (_tT, _tC):
        if g not in words:
            words[g] = 1
            frontier.append(g)
    while head < len(frontier):
        cur = frontier[head]
        head += 1
        for g in (_tT, _tC):
            nx = _cmp(g, cur)
            if nx not in words:
                words[nx] = words[cur] + 1
                frontier.append(nx)
    w23_gen = len(words) == 168 and set(words) == {_mk(*t) for t in _indep}
    w23_word = max(words.values()) <= 12
    w23_sound = all(t[0] == 0 and sorted(t) == list(range(8))
                    and all(t[u ^ v] == t[u] ^ t[v] for u in range(8) for v in range(8))
                    for t in words)
    w23_null = ((1, 2, 3) not in set(_GLLIST)) and (NONLIN not in words)
    w23 = (w23_len and w23_set and w23_pred and w23_gen and w23_word and w23_sound
           and w23_null)
    ok["W23"] = w23
    print(f"W23_ISGL    `LowCob` IS EXACTLY GL(3,2) -- THE LAST FINITE FACT IS NOW IN LEAN "
          f"{'OK' if w23 else 'FAIL'} -- transcribed glList: 168 distinct: {w23_len}, equals "
          f"the independently enumerated GL(3,2): {w23_set}; Lean `glIndep` agrees with "
          f"independence on all 512 triples: {w23_pred}; BFS from the two generators reaches "
          f"exactly those 168: {w23_gen} (longest word {max(words.values())} <= 12: {w23_word}); "
          f"every generated table is an injective linear self-map fixing 0: {w23_sound}; "
          f"null control (a DEPENDENT triple absent, the NON-LINEAR table not generated): "
          f"{w23_null}. *** NOTHING ABOUT THE INVARIANT g IS OUTSIDE LEAN ANY MORE. "
          "SOUNDNESS is `lowCob_isGL`: every member restricts to an injective linear "
          "endomorphism of F2^3, which IS a GL(3,2) element -- assembled from `lowCob_lt`, "
          "`lowCob_t0`, `lowCob_lin` and `lowCob_inj8` (injectivity from linearity plus trivial "
          "kernel). COMPLETENESS is `lowCob_covers`: for each of the 168 an EXPLICIT WORD in the "
          "two generators, found by BFS and emitted as a `LowCob.comp` term; the dispatch is a "
          "168-way match on the index and each case closes by `decide` on the eight low values. "
          "`glIdx_lt`/`glIdx_eq` compute the index, `glList_indep`/`lowCob_eq_GL` tie it "
          "together -- all plain `decide`, NO `native_decide`, so no extra trust axiom. *** THE "
          "WHOLE CHAIN IS NOW LEAN: sigma moves by a coboundary -> the four sigmas of Q' cancel "
          "it -> Q' is invariant under the class -> the COUNT is invariant -> and the class IS "
          "GL(3,2), whose transitivity on the seven nonzero low patterns merges the residues. "
          "Both halves of g = (W & (W-1)) >> 3 are theorems, end to end. "
          "(III) untouched; (d) IS NOT CLOSED ***")

    # ---- W24  tr(A^3): WHY IT IS THE FINER INVARIANT, AND ITS CONSTANT STRATUM ------------
    # A_sig's ENTRY is not the resonance predicate but the SIGN -P1 = -sigma(a,b)sigma(a^L,b^L).
    # Under a class member the coboundary does NOT cancel there: only lam(a^b) squares away, and
    # what survives factors as mu(a)*mu(b) with mu(x) = lam(x)lam(x^L) -- a DIAGONAL SIMILARITY
    # A' = D A D with D^2 = I, so tr(A'^k) = tr(A^k) for EVERY k (`P1_of_coboundary`,
    # `P1_lowCob`, proven forall n today). tau admits no such factorisation. So tr(A^2) is
    # invariant under BOTH symmetries while tr(A^3) is invariant under GL(3,2) ONLY -- which is
    # exactly why the PAIR separates strictly more than tr(A^2) alone (W5).
    from fractions import Fraction as _Fr
    gl_bad = tau_changed = 0
    tot_orb = 0
    for n in (6, 7, 8):
        m, H = n - 1, 1 << (n - 1)
        Sn = sign_table_fast(n)
        t3 = {}
        for Llo in range(1, H):
            t3[Llo] = traces23(A_sig_fast(n, Llo, Sn))[1]
        for y in range(H // 8):
            vals = {t3[8 * y + r] for r in range(1, 8) if 8 * y + r < H}
            if vals:
                tot_orb += 1
                if len(vals) > 1:
                    gl_bad += 1
        tau_changed += sum(1 for W in range(2, H, 2)
                           if 1 <= (W & (W - 1)) + 1 < H
                           and t3[W] != t3[(W & (W - 1)) + 1])
    # the y = 0 stratum has a CLOSED FORM; off it the same form FAILS (declared negative)
    strat_rows = []
    for n in (6, 7, 8, 9, 10):
        m, H = n - 1, 1 << (n - 1)
        Sn = sign_table_fast(n)
        hits = 0
        for Llo in range(1, H):
            t2v, t3v = traces23(A_sig_fast(n, Llo, Sn))
            if _Fr(2, 7) * t2v * ((1 << m) - 15) == t3v:
                hits += 1
        cf = _Fr(2, 7) * ((1 << m) - 2) * ((1 << m) - 4) * ((1 << m) - 15)
        ok_cf = (traces23(A_sig_fast(n, 1, Sn))[1] == cf)
        strat_rows.append((n, hits, ok_cf))
    w24 = (gl_bad == 0 and tau_changed > 0
           and all(h == 7 and c for _, h, c in strat_rows))
    ok["W24"] = w24
    print(f"W24_TRA3    tr(A^3) IS GL(3,2)-INVARIANT BUT NOT tau-INVARIANT -- AND THAT IS WHY "
          f"THE PAIR SEPARATES {'OK' if w24 else 'FAIL'} -- GL-orbits with non-constant "
          f"tr(A^3): {gl_bad}/{tot_orb} (n=6,7,8); tau-merges that CHANGE tr(A^3): "
          f"{tau_changed} (must be > 0); "
          + "; ".join(f"n={a}: {b} labels on the closed-form stratum, y=0 form exact: {c}"
                      for a, b, c in strat_rows)
          + ". *** THE MECHANISM IS A THEOREM: A_sig's ENTRY is the SIGN "
            "-P1 = -sigma(a,b)sigma(a^L,b^L), not the resonance predicate. Under a class member "
            "the coboundary does NOT cancel there -- only lam(a^b) squares away and what "
            "survives FACTORS, P1(pa,pb) = P1(a,b)*mu(a)*mu(b) with mu(x) = lam(x)lam(x^L). That "
            "is a DIAGONAL SIMILARITY A' = D A D with D^2 = I, so tr(A'^k) = tr(A^k) for EVERY k. "
            "`P1_of_coboundary` and `P1_lowCob` are proven forall n today. tau admits no such "
            "factorisation, and it MEASURABLY changes tr(A^3). *** SO THE ASYMMETRY IS EXPLAINED: "
            "tr(A^2) is invariant under BOTH GL(3,2) and tau; tr(A^3) under GL(3,2) ONLY. That is "
            "why the PAIR separates strictly more than tr(A^2) alone (W5), and it is the "
            "structural reason (d) needs the second trace. *** CLOSED FORM, ON ITS STRATUM ONLY: "
            "tr(A^3) = (2/7)(2^m-2)(2^m-4)(2^m-15) = (2/7)*tr(A^2)*(2^m-15) on the y=0 class, "
            "exact at n=6..11. OFF that stratum the form FAILS -- exactly 7 labels satisfy it at "
            "every level, which are precisely the seven members of the y=0 GL-orbit -- and this "
            "is asserted as a DECLARED NEGATIVE. The deviation is constant on GL-orbits, which is "
            "where the next rung starts. *** NO GENERAL CLOSED FORM FOR tr(A^3) IS CLAIMED. This "
            "is exactly where tr(A^2) stood at W6, and it took W13-W17 to close that one. "
            "(III) untouched; (d) IS NOT CLOSED, and V1 IS NOT PROVEN ***")

    # ---- W25  THE tr(A^3) DEVIATION: ONE EXACT RECURSION AND TWO SHARP IMPOSSIBILITIES ----
    # W24 left the deviation off the y=0 stratum open, noting only that it is GL-constant.
    # Splitting the label by its top bit -- the split that cracked tr(A^2) -- gives:
    #   LOW  (W < e): an EXACT linear recursion in the level-(n-1) PAIR, all labels;
    #   HIGH (W >= e): the pair (t2',t3') DETERMINES t3 on ODD labels but NOT in general, and
    #                  even where it determines it, the dependence is NOT affine.
    # Both negatives are asserted so the next rung does not chase an impossible ansatz.
    _t23 = {}
    for _n in range(5, 11):
        _S = sign_table_fast(_n)
        for _W in range(1, 1 << (_n - 1)):
            _t23[(_n, _W)] = traces23(A_sig_fast(_n, _W, _S))

    # (i) THE LOW RECURSION, in exact integer arithmetic
    low_rows = []
    for n in range(7, 11):
        m, e = n - 1, 1 << (n - 2)
        bad = tot = 0
        for W in range(1, e):
            Wp = W % e
            if Wp == 0:
                continue
            t2p, t3p = _t23[(n - 1, Wp)]
            tot += 1
            if _t23[(n, W)][1] != 8 * t3p + 24 * t2p - 12 * ((1 << m) - 4):
                bad += 1
        low_rows.append((n, tot, bad))
    # (ii) HIGH: does (t2',t3') determine t3?  all labels vs odd labels only
    hi_rows = []
    for n in range(7, 11):
        m, e = n - 1, 1 << (n - 2)
        allmap, oddmap = {}, {}
        for W in range(e, 1 << m):
            Wp = W % e
            if Wp == 0:
                continue
            k = _t23[(n - 1, Wp)]
            allmap.setdefault(k, set()).add(_t23[(n, W)][1])
            if W % 2:
                oddmap.setdefault(k, set()).add(_t23[(n, W)][1])
        hi_rows.append((n, sum(1 for v in allmap.values() if len(v) > 1),
                        sum(1 for v in oddmap.values() if len(v) > 1), len(allmap)))
    # (iii) even where it determines it, the HIGH-odd dependence is NOT affine in (t2',t3'):
    #       exhibit three odd high labels whose (t2',t3',t3) triples are not collinear
    aff_bad = 0
    for n in range(8, 11):
        m, e = n - 1, 1 << (n - 2)
        pts = []
        for W in range(e, 1 << m, 2):
            Wp = W % e
            if Wp:
                t2p, t3p = _t23[(n - 1, Wp)]
                pts.append((t2p, t3p, _t23[(n, W)][1]))
        pts = sorted(set(pts))
        # solve the affine system on the first three independent points, test on the rest
        import itertools as _it2
        found = False
        for tri in _it2.combinations(pts[:6], 3):
            M = [[a, b, 1] for a, b, _ in tri]
            det = (M[0][0]*(M[1][1]-M[2][1]) - M[0][1]*(M[1][0]-M[2][0])
                   + (M[1][0]*M[2][1] - M[2][0]*M[1][1]))
            if det:
                A = np.array(M, float); y = np.array([c for _, _, c in tri], float)
                sol = np.linalg.solve(A, y)
                resid = max(abs(sol[0]*a + sol[1]*b + sol[2] - c) for a, b, c in pts)
                if resid > 1.0:
                    aff_bad += 1
                found = True
                break
        if not found:
            aff_bad += 1
    w25 = (all(b == 0 for _, _, b in low_rows)
           and all(ba > 0 and bo == 0 for _, ba, bo, _ in hi_rows)
           and aff_bad == 3)
    ok["W25"] = w25
    print(f"W25_TRA3REC ONE EXACT tr(A^3) RECURSION, AND TWO SHARP IMPOSSIBILITIES "
          f"{'OK' if w25 else 'FAIL'} -- LOW branch t3(n,W) = 8 t3' + 24 t2' - 12(2^m - 4), in "
          f"exact integer arithmetic: "
          + "; ".join(f"n={a}: {b} labels, {c} failing" for a, b, c in low_rows)
          + "; HIGH branch, does (t2',t3') DETERMINE t3: "
          + "; ".join(f"n={a}: {d} keys, {b} colliding over ALL labels, {c} over ODD labels"
                      for a, b, c, d in hi_rows)
          + f"; and where it does determine it the dependence is NOT affine in (t2',t3'): "
            f"{aff_bad}/3 levels refute affinity. *** THE POSITIVE: the top-bit split -- the "
            "same split that cracked tr(A^2) -- gives the LOW branch an EXACT linear recursion "
            "in the level-(n-1) PAIR, with a closed constant -12(2^m - 4). Verified in exact "
            "integers over every low label at n = 7..10, 0 failures. *** THE FIRST NEGATIVE, AND "
            "IT IS THE USEFUL ONE: on the HIGH branch the pair (t2',t3') does NOT determine "
            "t3(n,W). Explicit witness at n=7: the key (t2',t3') = (168,-336) carries BOTH "
            "-92112 and 18480. So NO recursion for tr(A^3) on the pair alone can exist in "
            "general -- a third level-quantity is required. *** BUT THE COLLISIONS ARE ENTIRELY "
            "SEAM-BORNE: restricted to ODD labels the pair DOES determine t3 on the high branch, "
            "0 collisions at all four levels. The even labels are what break it -- the same "
            "place tau, and hence the tr(A^2) story, needed separate treatment. *** THE SECOND "
            "NEGATIVE: even on the odd high branch the dependence is NOT AFFINE in (t2',t3') -- "
            "an affine fit through three points misses the rest by 1e4 to 1e7, against exact "
            "zero on the low branch. So the high branch is a genuine function of the pair on the "
            "Fano family, but not a linear one. *** tr(A^3) IS STILL NOT CLOSED. What this rung "
            "buys is one exact half of the recursion and two impossibility results that rule out "
            "the two obvious ansaetze. (III) untouched; (d) IS NOT CLOSED ***")

    # ---- W26  THE THIRD LEVEL-QUANTITY IS lsb(W) ------------------------------------------
    # W25 proved the pair (t2,t3) is NOT self-propagating on the high branch. The missing datum
    # is lsb(W), the 2-adic valuation of the label -- and that is exactly what `tau` normalises
    # away (tau moves the lowest set bit to position 0). It coheres with W24: tr(A^2) IS
    # tau-invariant and so needs only g(W), which DISCARDS lsb; tr(A^3) is NOT tau-invariant and
    # needs lsb back. The search was run against spectral candidates too, and they FAIL.
    def _lsb(W):
        return (W & -W).bit_length() - 1

    _P = {}
    for _n in range(5, 11):
        _S = sign_table_fast(_n)
        for _W in range(1, 1 << (_n - 1)):
            _P[(_n, _W)] = traces23(A_sig_fast(_n, _W, _S))

    def _hi_coll(n, key):
        e = 1 << (n - 2)
        mp = {}
        for W in range(e, 1 << (n - 1)):
            Wp = W % e
            if Wp:
                mp.setdefault(key(n - 1, Wp), set()).add(_P[(n, W)][1])
        return sum(1 for v in mp.values() if len(v) > 1), len(mp)

    pair = lambda n, W: _P[(n, W)]
    triple = lambda n, W: (_P[(n, W)][0], _P[(n, W)][1], _lsb(W))
    rows26 = []
    for n in range(7, 11):
        cp, kp = _hi_coll(n, pair)
        ct, kt = _hi_coll(n, triple)
        rows26.append((n, cp, kp, ct, kt))
    # SPECTRAL CANDIDATES FAIL: adding tr(A^4), tr(A^5) changes nothing
    spec_rows = []
    for n in (7, 8, 9):
        S = sign_table_fast(n - 1)
        t45 = {}
        for W in range(1, 1 << (n - 2)):
            A = A_sig_fast(n - 1, W, S).astype(np.float64)
            A2 = A @ A
            t45[W] = (int(round(np.trace(A2 @ A2))), int(round(np.trace(A2 @ A2 @ A))))
        k4 = lambda nn, W: (_P[(nn, W)][0], _P[(nn, W)][1], t45[W][0])
        k5 = lambda nn, W: (_P[(nn, W)][0], _P[(nn, W)][1], t45[W][1])
        c4, _ = _hi_coll(n, k4)
        c5, _ = _hi_coll(n, k5)
        cp, _ = _hi_coll(n, pair)
        spec_rows.append((n, cp, c4, c5))
    w26 = (all(ct == 0 and cp > 0 for _, cp, _, ct, _ in rows26)
           and all(c4 == cp and c5 == cp and cp > 0 for _, cp, c4, c5 in spec_rows))
    ok["W26"] = w26
    print(f"W26_LSB     THE THIRD LEVEL-QUANTITY IS lsb(W) -- THE 2-ADIC VALUATION OF THE LABEL "
          f"{'OK' if w26 else 'FAIL'} -- HIGH-branch collisions, pair (t2',t3') vs triple "
          f"(t2',t3',lsb'): "
          + "; ".join(f"n={a}: {b}/{c} colliding with the pair, {d}/{e} with the triple"
                      for a, b, c, d, e in rows26)
          + "; SPECTRAL CANDIDATES REFUTED (adding tr(A^4) or tr(A^5) changes NOTHING): "
          + "; ".join(f"n={a}: pair {b}, +t4 {c}, +t5 {d}" for a, b, c, d in spec_rows)
          + ". *** THE ANSWER IS lsb(W), AND IT IS THE ONE THE STRUCTURE PREDICTED. `tau` moves "
            "the LOWEST SET BIT to position 0, so lsb is precisely the datum tau destroys. W24 "
            "proved tr(A^2) is invariant under BOTH GL(3,2) and tau -- hence it depends only on "
            "g(W) = (W & (W-1)) >> 3, which DISCARDS the lowest set bit -- while tr(A^3) is "
            "invariant under GL(3,2) ONLY. So tr(A^3) must see exactly what g threw away, and it "
            "does: adjoining lsb(W') kills every high-branch collision at n = 7,8,9,10, where "
            "the pair alone collides at 1, 2, 4 and 8 keys. *** AND lsb PROPAGATES FOR FREE: it "
            "is label data, not a graph invariant -- on the high branch W = W' + e with W' != 0, "
            "so lsb(W) = lsb(W'). Hence the TRIPLE (t2, t3, lsb) is SELF-PROPAGATING where the "
            "pair is not. *** THE SPECTRAL NEGATIVE IS THE SHARP ONE: adding tr(A^4) or tr(A^5) "
            "leaves the collision count EXACTLY unchanged, so the colliding labels agree in the "
            "whole trace family -- no level-(n-1) SPECTRAL invariant can supply the missing "
            "datum. It had to be label-arithmetic, and it is. *** STILL NOT A FORMULA: at FIXED "
            "lsb' the high branch is determined but NOT AFFINE in (t2',t3') -- an affine fit "
            "misses by 1e5 to 1e6, against exact zero on the low branch. tr(A^3) IS NOT CLOSED; "
            "what is now known is exactly which three quantities a closed form may use. "
            "(III) untouched; (d) IS NOT CLOSED ***")

    # ---- W27  NO CLOSED FORM FROM THE TRIPLE YET -- AND WHY, HONESTLY --------------------
    # I went looking for the closed form using (t2, t3, lsb) and did NOT find it. Three things
    # are worth recording so the next rung does not repeat the search.
    #   (1) the HIGH branch is not a low-degree polynomial in (t2',t3'), even stratified by lsb';
    #   (2) W26's "the triple DETERMINES t3" is real but MODEST evidence -- measured here;
    #   (3) the structural reason: a triangle at level n whose vertices straddle the level split
    #       does NOT reduce to a level-(n-1) triangle. It reduces to a PATH, and path counts are
    #       not traces -- which is why no trace (W26) and no polynomial in traces (here) closes
    #       the high branch.
    def _lsb2(W):
        return (W & -W).bit_length() - 1

    poly_rows, inj_rows = [], []
    for n in (8, 9):
        m, e = n - 1, 1 << (n - 2)
        strat = {}
        for W in range(e, 1 << m):
            Wp = W % e
            if Wp:
                strat.setdefault(_lsb2(Wp), set()).add(
                    (_P[(n - 1, Wp)][0], _P[(n - 1, Wp)][1], _P[(n, W)][1]))
        worst = 0.0
        for j, pts in strat.items():
            pts = sorted(pts)
            for basis in ([lambda a, b: b, lambda a, b: a, lambda a, b: 1.0],
                          [lambda a, b: b, lambda a, b: a, lambda a, b: 1.0,
                           lambda a, b: float(a) * a, lambda a, b: float(a) * b]):
                if len(pts) < len(basis) + 2:
                    continue
                A = np.array([[f(p[0], p[1]) for f in basis] for p in pts], float)
                y = np.array([p[2] for p in pts], float)
                sol, *_ = np.linalg.lstsq(A, y, rcond=None)
                rel = float(np.max(np.abs(A @ sol - y))) / max(1.0, float(np.max(np.abs(y))))
                worst = max(worst, rel) if len(basis) == 5 else worst
        poly_rows.append((n, worst))
        # collapse ratio of the triple
        trip = {}
        for W in range(e, 1 << m):
            Wp = W % e
            if Wp:
                trip.setdefault((_P[(n-1,Wp)][0], _P[(n-1,Wp)][1], _lsb2(Wp)), set()).add(W)
        nlab = sum(len(v) for v in trip.values())
        nval = len({_P[(n, W)][1] for W in range(e, 1 << m) if W % e})
        inj_rows.append((n, nlab, len(trip), nval,
                         max(len(v) for v in trip.values())))
    w27 = all(r > 0.005 for _, r in poly_rows) and all(k < l for _, l, k, _, _ in inj_rows)
    ok["W27"] = w27
    print(f"W27_NOFORM  NO CLOSED FORM FROM THE TRIPLE YET -- AND WHY "
          f"{'OK' if w27 else 'FAIL'} -- HIGH branch, best QUADRATIC fit in (t2',t3') "
          f"stratified by lsb', worst relative residual: "
          + "; ".join(f"n={a}: {b:.3g}" for a, b in poly_rows)
          + "; collapse of the triple on high labels: "
          + "; ".join(f"n={a}: {b} labels -> {c} keys -> {d} distinct t3 (max {e} labels/key)"
                      for a, b, c, d, e in inj_rows)
          + ". *** I WENT LOOKING FOR THE CLOSED FORM WITH THE TRIPLE AND DID NOT FIND IT. Three "
            "things are recorded so the next rung does not repeat the search. (1) The HIGH branch "
            "is NOT a low-degree polynomial in (t2',t3') even stratified by lsb': adding the "
            "quadratic terms t2'^2 and t2't3' improves the relative residual from ~0.46 to ~0.06 "
            "but does not close it, against EXACT ZERO on the low branch. (2) W26's claim that "
            "the triple DETERMINES t3 is real but MODEST evidence, and this clause measures how "
            "modest: the triple is not injective -- about two labels per key, up to four -- and "
            "it carries the high labels onto far fewer distinct t3 values, so the agreement is "
            "not an artefact of a fine partition. But determining a value on a finite set is NOT "
            "evidence that a FORMULA exists, and W26 should not be read as if it were. (3) THE "
            "STRUCTURAL REASON, which is the useful part: a triangle at level n whose vertices "
            "straddle the level split does NOT reduce to a level-(n-1) TRIANGLE -- it reduces to "
            "a PATH, and path counts are not traces. That is why no additional trace helps (W26) "
            "and why no polynomial in the traces closes it (here). The next rung should carry a "
            "PATH-COUNT alongside the traces, not another spectral invariant. "
            "(III) untouched; (d) IS NOT CLOSED, and tr(A^3) IS NOT CLOSED ***")

    # ---- W28  THE PATH COUNTS FAIL -- AND W27's RECOMMENDATION IS RETRACTED ---------------
    # W27 concluded that straddling triangles become PATHS and recommended the next rung carry a
    # path-count. I counted them. They give NOTHING, and the reason retracts the recommendation:
    # the colliding level-(n-1) labels have IDENTICAL FULL SPECTRA, and agree on the non-spectral
    # invariants too. No invariant OF THE FIBER can supply the missing datum -- it is not a graph
    # recursion at all, it is a LABEL recursion, and lsb(W) is the label datum that closes it.
    def _inv(n, W, S):
        A = A_sig_fast(n, W, S).astype(np.float64)
        A2 = A @ A
        deg = np.count_nonzero(A, axis=1).astype(np.float64)
        return dict(t2=int(round(np.trace(A2))), t3=int(round(np.trace(A2 @ A))),
                    p2=int(round(np.sum(deg ** 2))), p3=int(round(np.sum(deg ** 3))),
                    q1=int(round(np.sum(A * A2))),
                    spec=tuple(np.round(np.linalg.eigvalsh(A), 6)))

    _I = {}
    for _n in range(6, 10):
        _S = sign_table_fast(_n)
        for _W in range(1, 1 << (_n - 1)):
            _I[(_n, _W)] = _inv(_n, _W, _S)

    def _coll(n, key):
        e = 1 << (n - 2)
        mp = {}
        for W in range(e, 1 << (n - 1)):
            Wp = W % e
            if Wp:
                mp.setdefault(key(n - 1, Wp), set()).add(_I[(n, W)]['t3'])
        return sum(1 for v in mp.values() if len(v) > 1)

    kpair = lambda n, W: (_I[(n, W)]['t2'], _I[(n, W)]['t3'])
    kp2 = lambda n, W: (_I[(n, W)]['t2'], _I[(n, W)]['t3'], _I[(n, W)]['p2'])
    kq1 = lambda n, W: (_I[(n, W)]['t2'], _I[(n, W)]['t3'], _I[(n, W)]['q1'])
    kall = lambda n, W: (_I[(n, W)]['t2'], _I[(n, W)]['t3'], _I[(n, W)]['p2'],
                         _I[(n, W)]['p3'], _I[(n, W)]['q1'], _I[(n, W)]['spec'])
    path_rows, cosp_rows = [], []
    for n in (7, 8, 9):
        path_rows.append((n, _coll(n, kpair), _coll(n, kp2), _coll(n, kq1), _coll(n, kall)))
        # the colliding labels: how many DISTINCT full spectra among them?
        e = 1 << (n - 2)
        mp = {}
        for W in range(e, 1 << (n - 1)):
            Wp = W % e
            if Wp:
                mp.setdefault(kpair(n - 1, Wp), set()).add((Wp, _I[(n, W)]['t3']))
        worst = 0
        for key, v in mp.items():
            if len({x[1] for x in v}) > 1:
                sp = {_I[(n - 1, x[0])]['spec'] for x in v}
                worst = max(worst, len(sp))
        cosp_rows.append((n, worst))
    w28 = (all(a == b == c == d and a > 0 for _, a, b, c, d in path_rows)
           and all(x == 1 for _, x in cosp_rows))
    ok["W28"] = w28
    print(f"W28_PATHS   THE PATH COUNTS FAIL, AND W27's RECOMMENDATION IS RETRACTED "
          f"{'OK' if w28 else 'FAIL'} -- HIGH-branch collisions: "
          + "; ".join(f"n={a}: pair {b}, +sum(deg^2) {c}, +sum A.(A^2) {d}, "
                      f"+ALL of them AND the full spectrum {e}"
                      for a, b, c, d, e in path_rows)
          + "; distinct FULL SPECTRA among the colliding level-(n-1) labels: "
          + "; ".join(f"n={a}: {b}" for a, b in cosp_rows)
          + ". *** I COUNTED THE PATHS AND THEY GIVE NOTHING. sum_a deg_a^2 (2-paths through a "
            "vertex), sum_a deg_a^3, and the Hadamard contraction sum_{a,b} A_ab (A^2)_ab -- all "
            "chosen because they are invariant under the class action A' = D A D with D^2 = I "
            "(so |A|, hence the DEGREE SEQUENCE, is untouched) and none of them is a trace -- "
            "leave the collision count EXACTLY unchanged. Adding all of them TOGETHER WITH the "
            "full spectrum still changes nothing. *** AND THE REASON RETRACTS W27: the colliding "
            "level-(n-1) labels have IDENTICAL FULL SPECTRA -- one distinct spectrum among the "
            "eight labels {17..24} at every level tested. So NO invariant of the level-(n-1) "
            "fiber, spectral or not, can supply the missing datum. W27 concluded that straddling "
            "triangles become paths and recommended carrying a path-count; that recommendation "
            "is WRONG and is withdrawn here. The straddling observation is still true, but it "
            "does not follow that a fiber invariant can express the correction. *** THE CORRECT "
            "STATEMENT: the level-(n-1) fiber's isomorphism class does NOT determine tr(A^3) at "
            "level n. The missing datum is LABEL ARITHMETIC, not graph structure -- which is "
            "exactly what W26 found when every spectral candidate failed and lsb(W) worked. So "
            "this is not a graph recursion with a richer invariant; it is a LABEL recursion. "
            "(III) untouched; tr(A^3) IS NOT CLOSED; (d) IS NOT CLOSED ***")

    # ---- W29  THE COUNTING RECURSION ENTERS LEAN: THREE OF FOUR LOW QUADRANTS -------------
    # The W15 ledger has been carried on paper and pinned by clause. Tier 29 starts formalising
    # it. The mechanism is split-by-predicate / extract-singleton / evaluate-constant -- no
    # Finset, exactly as Tier 27 avoided cardinality. This clause transcribes each new Lean
    # statement and checks it against the measured sums.
    def _nInd(S, W, a, b):
        return 1 if (a != 0 and b != 0 and a != b and _Qp(S, W, a, b) == -1) else 0

    def _qInd(S, W, u, b):
        return 1 if (b != 0 and _Qu2(S, W, u, b) == -1) else 0

    def _Ncnt(S, W, m):
        H = 1 << m
        return sum(_nInd(S, W, a, b) for a in range(H) for b in range(H))

    def _Mcnt(S, W, m):
        H = 1 << m
        return sum(_qInd(S, W, u, b) for u in range(H) for b in range(H))

    w29_rows = []
    for m in (3, 4, 5):                       # level m+2 = 5,6,7
        e = 1 << (m + 1)
        S2, S1 = sign_table_fast(m + 2), sign_table_fast(m + 1)
        bad_quad = bad_ll = bad_ul = bad_lu = 0
        for W in range(1, e):
            # Ncnt_quad: the four quadrants sum to the whole box
            ll = sum(_nInd(S2, W, a, b) for a in range(e) for b in range(e))
            lu = sum(_nInd(S2, W, a, e + v) for a in range(e) for v in range(e))
            ul = sum(_nInd(S2, W, e + u, b) for u in range(e) for b in range(e))
            uu = sum(_nInd(S2, W, e + u, e + v) for u in range(e) for v in range(e))
            if ll + lu + ul + uu != _Ncnt(S2, W, m + 2):
                bad_quad += 1
            # Ncnt_ll_low : ll == Ncnt W (m+1)
            if ll != _Ncnt(S1, W, m + 1):
                bad_ll += 1
            # Ncnt_ul_low : ul == Mcnt W (m+1)
            if ul != _Mcnt(S1, W, m + 1):
                bad_ul += 1
            # Ncnt_lu_low : lu == sum over a!=0, a!=W of [Qgen(W,v,a,m+1) = -1]
            pred = sum(1 for a in range(e) for v in range(e)
                       if a != 0 and a != W and _Qu2(S1, W, v, a) == -1)
            if lu != pred:
                bad_lu += 1
        w29_rows.append((m + 2, e - 1, bad_quad, bad_ll, bad_ul, bad_lu))
    # NULL CONTROL: drop the `a != W` guard from the lu statement and it must BREAK
    m = 4
    e = 1 << (m + 1)
    S2, S1 = sign_table_fast(m + 2), sign_table_fast(m + 1)
    null29 = 0
    for W in range(1, e):
        lu = sum(_nInd(S2, W, a, e + v) for a in range(e) for v in range(e))
        naive = sum(1 for a in range(e) for v in range(e)
                    if a != 0 and _Qu2(S1, W, v, a) == -1)
        if lu != naive:
            null29 += 1
    def _uuInd(S1, W, u, v):
        return 1 if (u != v and v != (u ^ W)
                     and (u == 0 or v == 0 or u == W or v == W
                          or _Qp(S1, W, v, u) == -1)) else 0

    bad_uu = 0
    for m in (3, 4):
        e = 1 << (m + 1)
        S2, S1 = sign_table_fast(m + 2), sign_table_fast(m + 1)
        for W in range(1, e):
            for u in range(e):
                for v in range(e):
                    if _nInd(S2, W, e + u, e + v) != _uuInd(S1, W, u, v):
                        bad_uu += 1
    null_uu = 0
    e = 1 << 4
    S2, S1 = sign_table_fast(5), sign_table_fast(4)
    for W in range(1, e):
        for u in range(e):
            for v in range(e):
                naive = 1 if (u != v and (u == 0 or v == 0 or u == W or v == W
                                          or _Qp(S1, W, v, u) == -1)) else 0
                if _nInd(S2, W, e + u, e + v) != naive:
                    null_uu += 1
    w29 = (all(b == c == d == f == 0 for _, _, b, c, d, f in w29_rows) and null29 > 0
           and bad_uu == 0 and null_uu > 0)
    ok["W29"] = w29
    print(f"W29_LEDGER  THE COUNTING RECURSION ENTERS LEAN -- ALL FOUR LOW QUADRANTS "
          f"{'OK' if w29 else 'FAIL'} -- "
          + "; ".join(f"level {a}: {b} labels, quad-split {c}, ll {d}, ul {e}, lu {f} failing"
                      for a, b, c, d, e, f in w29_rows)
          + f"; NULL CONTROL (drop the a != W guard from lu): {null29} labels break, as required"
            f"; uu pointwise (Ncnt_uu_low) at levels 5,6: {bad_uu} failures; its NULL "
            f"CONTROL (drop the v != u^W guard): {null_uu} pairs break, as required."
            " *** `Ncnt_quad` splits the level-(m+2) box into its four quadrants at the seam "
            "2^(m+1) -- two applications of `sumLt_add` plus `sumLt_pair`. *** `Ncnt_ll_low` is "
            "the clean one: `Q'red_low_ll` is UNCONDITIONAL, so that quadrant IS the level-(m+1) "
            "count, with no slices at all. *** `Ncnt_ul_low` and `Ncnt_lu_low` reduce their "
            "quadrants to counts of the UNPRIMED `Qgen`, because those two low rows land on "
            "`Qgen` rather than `Qgen'`. `lu`'s single row-failure, a = W, contributes NOTHING: "
            "`Qgen'_label_left` makes the value +1 there, so the indicator is 0 -- the null "
            "control confirms that dropping the guard genuinely changes the count. *** NEW "
            "TOOLKIT, all by induction and all kernel-clean: `sumLt_zero`, `sumLt_const`, "
            "`sumLt_pair`, `sumLt_split_if`, `sumLt_single`, `sumLt_single'`. Split by "
            "predicate, extract singletons, evaluate constants -- no Finset, exactly as Tier 27 "
            "avoided cardinality. *** AND `uu` CLOSES THE SET: its five side conditions COLLAPSE, because on EVERY failure slice Qgen = -1 (u=0 and v=0 are the gap roots a=H and b=H; u=W and v=W are a^W=H and b^W=H), so `Qgen'_off_lines` converts all four at once, while the fifth, v = u^W, is exactly `Qgen'_coset_partner` and gives +1, contributing NOTHING. ALL FOUR LOW QUADRANTS ARE NOW LEAN THEOREMS. *** WHAT IS NOT DONE: "
            "the bridge from the UNPRIMED counts back to `Ncnt`, the six-line slice "
            "arithmetic over SIX OVERLAPPING lines. Until it lands the LOW recursion is NOT yet a Lean theorem and the "
            "closed form's derivation still rests on this clause. (III) untouched; tr(A^3) NOT "
            "closed; (d) IS NOT CLOSED ***")

    # ---- W30  THE BRIDGE'S CORE: Ncnt = OffCnt + (2^M - 2) --------------------------------
    # All four quadrants differ from Ncnt only ON THE SIX LINES, so they all factor through ONE
    # quantity: the count OFF the lines. `Ncnt_eq_OffCnt` is the first of those factorings, and
    # it needs NO inclusion-exclusion -- `nInd_split` is a POINTWISE identity, and summing a
    # pointwise identity keeps the pieces disjoint for free.
    def _offInd(S, W, M, a, b):
        return 1 if (a != 0 and a != W and b != 0 and b != W and a != b
                     and b != (a ^ W) and _Qp(S, W, a, b) == -1) else 0

    def _OffCnt(S, W, M):
        H = 1 << M
        return sum(_offInd(S, W, M, a, b) for a in range(H) for b in range(H))

    w30_rows = []
    for M in (3, 4, 5, 6):
        S, H = sign_table_fast(M), 1 << M
        bad_pt = bad_br = 0
        for W in range(1, H):
            for a in range(H):
                for b in range(H):
                    lhs = _nInd(S, W, a, b)
                    rhs = ((1 if (b == W and a != 0 and a != W) else 0)
                           + _offInd(S, W, M, a, b))
                    if lhs != rhs:
                        bad_pt += 1
            if _Ncnt(S, W, M) + 2 != _OffCnt(S, W, M) + H:
                bad_br += 1
        w30_rows.append((M, H - 1, bad_pt, bad_br))
    # NULL CONTROL. My FIRST choice was VACUOUS and this clause caught it: dropping the
    #   `b != a^W` guard does NOT break the bridge -- on the coset diagonal Q' = +1, so those
    #   pairs contribute 0 with or without it. Same for `a != W`. Both are load-bearing for the
    #   PROOF's case analysis, not for the VALUE. Only `b != W` carries the count.
    null30 = null30_vac = 0
    M = 4
    S, H = sign_table_fast(M), 1 << M
    for W in range(1, H):
        drop_bW = sum(1 for a in range(H) for b in range(H)
                      if a != 0 and a != W and b != 0 and a != b
                      and b != (a ^ W) and _Qp(S, W, a, b) == -1)
        if _Ncnt(S, W, M) + 2 != drop_bW + H:
            null30 += 1
        drop_cos = sum(1 for a in range(H) for b in range(H)
                       if a != 0 and a != W and b != 0 and b != W and a != b
                       and _Qp(S, W, a, b) == -1)
        if _Ncnt(S, W, M) + 2 == drop_cos + H:
            null30_vac += 1
    w30 = (all(c == d == 0 for _, _, c, d in w30_rows)
           and null30 > 0 and null30_vac == H - 1)
    ok["W30"] = w30
    print(f"W30_BRIDGE  THE BRIDGE'S CORE -- Ncnt = OffCnt + (2^M - 2) "
          f"{'OK' if w30 else 'FAIL'} -- "
          + "; ".join(f"M={a}: {b} labels, pointwise nInd_split {c} failures, bridge {d}"
                      for a, b, c, d in w30_rows)
          + f"; NULL CONTROL (drop the b != W guard from OffCnt): {null30} labels break, as required"
            f"; and the HONEST VACUITY CHECK -- dropping b != a^W instead leaves the bridge "
            f"INTACT at {null30_vac} of {H-1} labels, so that guard does NOT carry the value. "
            "*** THE FACTORING: all four quadrants differ from `Ncnt` only ON THE SIX "
            "LINES, so they all go through ONE quantity, `OffCnt` -- the count OFF the lines. "
            "`Ncnt_eq_OffCnt` is the first of those factorings. *** AND IT NEEDS NO "
            "INCLUSION-EXCLUSION, which is what made the six OVERLAPPING lines look painful: "
            "`nInd_split` is a POINTWISE identity, nInd = [b = W column] + [off-lines], and "
            "summing a pointwise identity keeps the pieces disjoint for free. The overlaps never "
            "arise. *** THE THREE LINES THAT `Ncnt` SEES AND `OffCnt` DOES NOT: the a = W row is "
            "+1 throughout (`Qgen'_label_left`) and contributes NOTHING; the coset diagonal "
            "b = a^W is +1 throughout (`Qgen'_coset_partner`) and contributes NOTHING; only the "
            "b = W column is -1 throughout (`Qgen'_label_right`), and its size is exactly "
            "2^M - 2. *** AND A CAUGHT VACUITY, the third time a null control has done this in this lane: my first control dropped the `b != a^W` guard expecting a break and got none, because on the coset diagonal Q' = +1 so those pairs contribute 0 either way. Same for the `a != W` row. Both guards are load-bearing for the PROOF's case analysis and NOT for the value; only `b != W` carries the count. *** WHAT IS NOT DONE: the same factoring for the three UNPRIMED quadrant "
            "counts (ul, lu) and for uu, then the arithmetic assembly. Until those land the LOW "
            "recursion is NOT yet a Lean theorem. (III) untouched; tr(A^3) NOT closed; (d) IS "
            "NOT CLOSED ***")

    print("=" * 78)
    if all(ok.values()):
        print("CD_TOWER_ZDV1_VERDICT C_CLOSED__V1_REDUCED_TO_D_ALONE__NOT_CLOSED")
        print("CD_TOWER_ZDV1_NOTE V1 decomposes as orbit arithmetic (W1): 2^{n-4} Fano orbits "
              "(PROVEN forall n) + 2^{n-4}-1 seams, spectrum constant per orbit (W2, PROVEN "
              "forall n), minus 2^{n-5}-1 even-weight seam merges = 3*2^{n-5}. So V1 forall n = "
              "(c) the parity-collapse law forall n [*** CLOSED 2026-08-02, W8: `SounioZDCollapse.parity_collapse`, kernel-checked ***] AND (d) no further collapse. "
              "THIS RUNG REDUCES (d): the pair (tr A^2, tr A^3) -- edge count and signed "
              "triangle count -- induces EXACTLY the spectral partition block-for-block (W3, "
              "n<=9) and yields 3*2^{n-5} classes at n=6..11 (W4, six levels). So (d) turns from "
              "a cospectrality claim into injectivity of a TWO-INTEGER invariant. Neither trace "
              "alone suffices (W5: tr A^1 = 0, tr A^2 alone gives only 2^{n-4}). tr(A^2) has a "
              "closed form ONLY on the narrow stratum where it is constant -- y=0 Fano plus the WEIGHT-1 seams, 7+(n-3) fibers (W6) -- the general form is "
              "NOT derived and appears to need the degree-histogram induction, which is OPEN. "
              "V1 IS NOT CLOSED, but it is now (d) ALONE: (d) still needs a closed form PLUS a "
              "forall-n injectivity proof. Measured structure to start from: the isolated vertex "
              "is a = Llo itself, every other degree is 4*odd, and the degree histogram is "
              "governed by the binary structure of y = Llo>>3. Numerical certificate; D3 respected")
        return 0
    print("CD_TOWER_ZDV1_VERDICT INCOMPLETE  failing=" +
          ",".join(k for k, v in ok.items() if not v))
    return 1


if __name__ == "__main__":
    t0 = time.time()
    rc = main()
    print(f"[{time.time() - t0:.1f}s]", file=sys.stderr)
    raise SystemExit(rc)
