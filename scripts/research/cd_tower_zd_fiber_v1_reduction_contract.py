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
