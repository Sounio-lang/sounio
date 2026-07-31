#!/usr/bin/env python3
"""
CD-tower ZD fibers — the FIBER ANTISYMMETRY LEMMA: the explicit low-rank factorisation
that cd_tower_zd_fiber_spectral_forall_n_progress_contract.py lists as OPEN.

That contract's verdict names the missing piece verbatim:
    "The ∀n proof needs the explicit algebraic low-rank factorisation (Walsh/character-sum
     type) -- OPEN."
and its companion doc records "not found in-session". This rung finds it, and the finding is
elementary: the factorisation is a rank-2 folding induced by a sign involution on the fiber.

Notation. n >= 6; H = 2^{n-1}; a fiber is L = L_lo | H with L_lo in [1,H). The signed resonance
matrix A_sig is indexed by lo-labels l in [1,H); the hi-partner of l is h(l) = l ^ L. Write
tau(x,y) = cd_sigma(x,y,n-1) for the level-(n-1) sign. The contract's own builder sets
    P1(l,y) = sigma(l,y)*sigma(h(l),h(y)),  P3(l,y) = sigma(l,h(y))*sigma(h(l),y),
    res     = (P1==P1^T) & (P3==P3^T) & (P1==P3),   A_sig = -P1 on res, 0 elsewhere.

  A1  THE LEMMA (measured, six levels, ALL fibers): A_sig(l ^ L_lo, y) = -A_sig(l, y)
      for every l with l ^ L_lo != 0, for n = 6..11. ~1.2e9 entry comparisons, 0 violations.

  A2  MASK INVARIANCE + TWO VACUOUS CLAUSES. res(l ^ L_lo, y) == res(l, y) for y != L_lo.
      Separately: P1 and P3 are ALWAYS symmetric, so two of the three clauses in the existing
      resonance predicate are vacuous and res reduces to (P1 == P3). Reported as a finding
      about the in-tree predicate, not as a repair.

  A3  THE ALGEBRAIC CORE (exact identity, no case analysis, y != L_lo):
          P1(l ^ L_lo, y) = -P3(l, y)      and      P3(l ^ L_lo, y) = -P1(l, y).
      Proof, by the CD top-bit recursion, using that l, y, m = l^L_lo, m' = y^L_lo all have
      top bit 0 and that h(l) = m + H, h(y) = m' + H with m' != 0:
          sigma(l,y)      = tau(l,y)          sigma(h(l),h(y)) = tau(m',m)
          sigma(l,h(y))   = tau(m',l)         sigma(h(l),y)    = -tau(m,y)
      so P1(l,y) = tau(l,y)tau(m',m) and P3(l,y) = -tau(m',l)tau(m,y); substituting l -> m
      (whose partner is l) gives P1(m,y) = tau(m,y)tau(m',l) = -P3(l,y), and symmetrically.
      Hence the involution l -> l ^ L_lo SWAPS P1 and P3 and negates them. With A2, res is
      preserved, and on res: A(m,y) = -P1(m,y) = P3(l,y) = P1(l,y) = -A(l,y). QED for A1.

  A4  THE ISOLATED VERTEX IS DERIVED, NOT OBSERVED. The single exceptional column y = L_lo has
      m' = 0, which flips the bL==0 branch of the recursion; res then reduces to
      tau(l,L_lo) == tau(l^L_lo,L_lo), while the level-(n-1) sub-lemma
      tau(l, L_lo) = -tau(l ^ L_lo, L_lo) holds -- so res fails identically and row/column
      L_lo is ZERO. The isolated vertex sits at exactly l = L_lo, in every fiber. This is the
      source of the "-1" in 2^{n-2}-1; the sub-lemma is the same identity one level down.

  A5  THE EXPLICIT FACTORISATION. Let reps = one label per pair {l, l^L_lo}, l != L_lo
      (|reps| = 2^{n-2}-1), M = A_sig[reps,reps], and J the (2^{n-2}-1) x (2^{n-1}-1) matrix
      with J[k,rep_k] = +1, J[k,rep_k ^ L_lo] = -1, 0 elsewhere. Then
          A_sig = J^T M J        and        J J^T = 2 I.
      This IS the C^T S C the prior rung asked for, with C = J a 0/+-1 incidence matrix.
      Consequence, PROVEN for all n given A1: rank(A_sig) <= 2^{n-2}-1. Equality is still
      MEASURED (n <= 10, all fibers) -- the lower bound is not derived here.

  A6  EXACT SPECTRAL HALVING. Because J J^T = 2I, the nonzero spectrum of A_sig equals the
      nonzero spectrum of 2M. The eigenproblem descends, for all n, to an explicit
      (2^{n-2}-1)-dimensional matrix. Verified n = 6..9, all fibers.

  A7  DEFLATION GUARD. The lemma is not implied by the prior rung's V2 doubling containment:
      for fibers with L_lo >= 2^{n-2} the pairing l <-> l ^ L_lo carries lo-labels inside
      V2's block [1, 2^{n-2}) to labels outside it, so a statement about that block alone
      cannot yield it. Measured as a crossing count.

  A8  NULL CONTROL. The antisymmetry is a property of the fiber's OWN label: substituting a
      perturbed label L' != L_lo must BREAK it. If a perturbed label ever preserved it, A1
      would be vacuous. Measured as a nonzero violation count for every L' != L_lo.

  A0  PARITY. The vectorised sign_table/A_sig builders used here reproduce the in-tree
      builders of cd_tower_zd_fiber_spectral_forall_n_progress_contract.py entrywise.

NOT CLAIMED. This closes the low-rank / factorisation half only. It does NOT close ∀n spectral
completeness (#distinct spectra = 3*2^{n-5}): an explicit rank and factorisation do not exclude
cospectral fibers at large n. A6 reduces that question to the halved matrix M; it does not
answer it. The rank EQUALITY (lower bound) remains measured, not proven.

Verdict ZD_FIBER_ANTISYMMETRY_LEMMA__FACTORISATION_FOUND_LOWRANK_BOUND_PROVEN.
Numerical certificate over an exact integer sign table; D3 respected.
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PRIOR = os.path.join(HERE, "cd_tower_zd_fiber_spectral_forall_n_progress_contract.py")

# --- the in-tree reference builders, imported by execution (the prior rung has no __all__) ---
_src = open(PRIOR).read()
exec(_src.split("def main()")[0].split("from collections import defaultdict")[1])  # noqa: S102

FULL_N = (6, 7, 8, 9, 10)     # full battery
ANTISYM_N = (6, 7, 8, 9, 10, 11)  # A1 reaches one level further (cheap)
SPEC_N = (6, 7, 8, 9)         # A6 needs dense eigensolves


def sign_table_fast(n):
    """Vectorised CD sign table, int8, built by the same top-bit recursion as cd_sigma."""
    S = np.ones((1, 1), dtype=np.int8)
    for b in range(1, n + 1):
        h = 1 << (b - 1)
        P = S
        T = np.empty((2 * h, 2 * h), dtype=np.int8)
        T[:h, :h] = P                      # aH=0,bH=0 -> sigma(aL,bL)
        T[:h, h:] = P.T                    # aH=0,bH=1 -> sigma(bL,aL)
        blk = -P.copy(); blk[:, 0] = P[:, 0]
        T[h:, :h] = blk                    # aH=1,bH=0 -> +/- sigma(aL,bL), + iff bL==0
        blk2 = P.T.copy(); blk2[:, 0] = -P.T[:, 0]
        T[h:, h:] = blk2                   # aH=1,bH=1 -> +/- sigma(bL,aL), - iff bL==0
        S = T
    S[0, :] = 1
    S[:, 0] = 1
    return S


def parts_fast(n, Llo, S):
    """P1, P3 for fiber L_lo, indexed by lo-label - 1."""
    H = 1 << (n - 1)
    L = Llo | H
    los = np.arange(1, H)
    hi = los ^ L
    SLL = S[np.ix_(los, los)].astype(np.int16)
    SHH = S[np.ix_(hi, hi)].astype(np.int16)
    SLH = S[np.ix_(los, hi)].astype(np.int16)
    SHL = S[np.ix_(hi, los)].astype(np.int16)
    return SLL * SHH, SLH * SHL


def A_sig_fast(n, Llo, S):
    P1, P3 = parts_fast(n, Llo, S)
    res = (P1 == P1.T) & (P3 == P3.T) & (P1 == P3)
    A = np.where(res, -P1, 0).astype(np.int8)
    np.fill_diagonal(A, 0)
    return A


def pair_index(H, Llo):
    """(row idx, partner idx) for every l with l ^ L_lo != 0; and the non-L_lo columns."""
    los = np.arange(1, H)
    part = los ^ Llo
    keep = part != 0
    cols = np.array([y - 1 for y in range(1, H) if y != Llo])
    return los[keep] - 1, part[keep] - 1, cols


def main():
    print("=" * 78)
    print("CD-tower ZD fibers — FIBER ANTISYMMETRY LEMMA: the explicit low-rank factorisation")
    print("=" * 78)
    ok = {}

    # ---- A0 parity against the in-tree builders -----------------------------------------
    a0 = True
    for n in (6, 7):
        Sref, Sfast = sign_table(n), sign_table_fast(n)
        if not np.array_equal(Sref, Sfast):
            a0 = False
        H = 1 << (n - 1)
        for Llo in range(1, H):
            if not np.array_equal(A_sig(n, Llo, Sref), A_sig_fast(n, Llo, Sfast).astype(float)):
                a0 = False
    ok["A0"] = a0
    print(f"A0_PARITY   vectorised builders == in-tree sign_table/A_sig entrywise (n=6,7) "
          f"{'OK' if a0 else 'FAIL'}")

    tables = {n: sign_table_fast(n) for n in ANTISYM_N}

    # ---- A1 the lemma -------------------------------------------------------------------
    a1 = True
    tot_v = tot_c = 0
    for n in ANTISYM_N:
        S, H = tables[n], 1 << (n - 1)
        v = c = 0
        for Llo in range(1, H):
            A = A_sig_fast(n, Llo, S)
            r, p, _ = pair_index(H, Llo)
            v += int(np.count_nonzero(A[p, :] + A[r, :]))
            c += r.size * (H - 1)
        tot_v += v
        tot_c += c
        a1 = a1 and v == 0
        print(f"A1_LEMMA    n={n:2d}  A(l^L_lo,y) == -A(l,y)   violations={v}/{c}  "
              f"{'OK' if v == 0 else 'FAIL'}")
    ok["A1"] = a1
    print(f"A1_TOTAL    {tot_v} violations over {tot_c} comparisons, all fibers, "
          f"n={ANTISYM_N[0]}..{ANTISYM_N[-1]}  {'OK' if a1 else 'FAIL'}")

    # ---- A2 mask invariance + vacuous clauses -------------------------------------------
    a2_mask = a2_vac1 = a2_vac3 = True
    for n in FULL_N:
        S, H = tables[n], 1 << (n - 1)
        for Llo in range(1, H):
            P1, P3 = parts_fast(n, Llo, S)
            if not np.array_equal(P1, P1.T):
                a2_vac1 = False
            if not np.array_equal(P3, P3.T):
                a2_vac3 = False
            r, p, cols = pair_index(H, Llo)
            res = P1 == P3
            if not np.array_equal(res[np.ix_(p, cols)], res[np.ix_(r, cols)]):
                a2_mask = False
    # A2b — the same vacuity in the SIBLING rung's own 4-term predicate, measured on its code
    sib = os.path.join(HERE, "cd_tower_zd_fiber_signed_localization_contract.py")
    ns = {}
    exec(open(sib).read().split("def _mul")[0], ns)  # noqa: S102
    sigma = ns["cd_sigma"]
    b12 = b34 = tot = 0
    for n in (6, 7):
        H = 1 << (n - 1)
        N = 1 << n
        # memoise THEIR cd_sigma into a table, then vectorise; the values are their code's
        Sg = np.array([[sigma(a, b, n) for b in range(N)] for a in range(N)], dtype=np.int16)
        for Llo in range(1, H):
            L = Llo | H
            lo = np.arange(1, H)
            hi = lo ^ L
            P1 = Sg[np.ix_(lo, lo)] * Sg[np.ix_(hi, hi)]
            P2 = Sg[np.ix_(lo, lo)].T * Sg[np.ix_(hi, hi)].T
            P3 = Sg[np.ix_(lo, hi)] * Sg[np.ix_(hi, lo)]
            P4 = Sg[np.ix_(lo, hi)].T * Sg[np.ix_(hi, lo)].T
            tot += P1.size
            b12 += int(np.count_nonzero(P1 != P2))
            b34 += int(np.count_nonzero(P3 != P4))
    a2_sib = (b12 == 0 and b34 == 0)
    ok["A2"] = a2_mask and a2_vac1 and a2_vac3 and a2_sib
    print(f"A2_MASK     res(l^L_lo,y) == res(l,y) for y != L_lo, all fibers n={FULL_N} "
          f"{'OK' if a2_mask else 'FAIL'}")
    print(f"A2_VACUITY  P1 always symmetric={a2_vac1}, P3 always symmetric={a2_vac3} "
          f"=> 2 of the 3 clauses of the resonance predicate as used in the A_sig builder are "
          f"vacuous; res reduces to (P1 == P3)  {'OK' if a2_vac1 and a2_vac3 else 'FAIL'}")
    print(f"A2_SIBLING  the same vacuity measured on the 4-term predicate of "
          f"cd_tower_zd_fiber_signed_localization_contract.py's own resonant(): "
          f"P1!=P2 in {b12}/{tot}, P3!=P4 in {b34}/{tot} (n=6,7)  "
          f"{'OK' if a2_sib else 'FAIL'}")

    # ---- A3 the algebraic core ----------------------------------------------------------
    a3 = True
    for n in FULL_N:
        S, H = tables[n], 1 << (n - 1)
        for Llo in range(1, H):
            P1, P3 = parts_fast(n, Llo, S)
            r, p, cols = pair_index(H, Llo)
            if not np.array_equal(P1[np.ix_(p, cols)], -P3[np.ix_(r, cols)]):
                a3 = False
            if not np.array_equal(P3[np.ix_(p, cols)], -P1[np.ix_(r, cols)]):
                a3 = False
    ok["A3"] = a3
    print(f"A3_CORE     P1(l^L_lo,y) = -P3(l,y) and P3(l^L_lo,y) = -P1(l,y)  (y != L_lo), "
          f"all fibers n={FULL_N}  {'OK' if a3 else 'FAIL'}")

    # ---- A4 the isolated vertex, derived -------------------------------------------------
    a4_iso = a4_sub = True
    for n in FULL_N:
        S, H = tables[n], 1 << (n - 1)
        Sp = S[:H, :H]                       # level-(n-1) sign table = tau
        for Llo in range(1, H):
            A = A_sig_fast(n, Llo, S)
            z = np.flatnonzero(np.count_nonzero(A, axis=1) == 0)
            if not (z.size == 1 and z[0] == Llo - 1):
                a4_iso = False
            lv = np.arange(1, H)
            lv = lv[(lv ^ Llo) != 0]
            if not np.array_equal(Sp[lv, Llo], -Sp[lv ^ Llo, Llo]):
                a4_sub = False
    ok["A4"] = a4_iso and a4_sub
    print(f"A4_ISOLATED the unique zero row/col sits at exactly l = L_lo, every fiber "
          f"{'OK' if a4_iso else 'FAIL'};  level-(n-1) sub-lemma "
          f"tau(l,L_lo) = -tau(l^L_lo,L_lo) {'OK' if a4_sub else 'FAIL'}")

    # ---- A5 explicit factorisation + rank ------------------------------------------------
    a5_fac = a5_rank = True
    for n in FULL_N:
        S, H = tables[n], 1 << (n - 1)
        target = 2 ** (n - 2) - 1
        for Llo in range(1, H):
            A = A_sig_fast(n, Llo, S).astype(float)
            reps = [l for l in range(1, H) if l != Llo and l < (l ^ Llo)]
            if len(reps) != target:
                a5_fac = False
            idx = [r - 1 for r in reps]
            M = A[np.ix_(idx, idx)]
            J = np.zeros((len(reps), H - 1))
            for k, rp in enumerate(reps):
                J[k, rp - 1] = 1.0
                J[k, (rp ^ Llo) - 1] = -1.0
            if not np.allclose(J @ J.T, 2 * np.eye(len(reps))):
                a5_fac = False
            if not np.allclose(J.T @ M @ J, A):
                a5_fac = False
            if int(np.linalg.matrix_rank(A, tol=1e-6)) != target:
                a5_rank = False
        print(f"A5_FACTOR   n={n:2d}  A_sig = J^T M J with J J^T = 2I, |reps| = 2^(n-2)-1 = "
              f"{target}  {'OK' if a5_fac else 'FAIL'};  rank == {target} (measured, all "
              f"fibers) {'OK' if a5_rank else 'FAIL'}")
    ok["A5"] = a5_fac and a5_rank
    print("A5_BOUND    rank(A_sig) <= 2^(n-2)-1 is DERIVED for all n from A1+A4 (rows pair "
          "up to sign; row L_lo is zero). Equality is MEASURED (n<=10), not derived.")

    # ---- A6 spectral halving -------------------------------------------------------------
    a6 = True
    for n in SPEC_N:
        S, H = tables[n], 1 << (n - 1)
        for Llo in range(1, H):
            A = A_sig_fast(n, Llo, S).astype(float)
            reps = [l for l in range(1, H) if l != Llo and l < (l ^ Llo)]
            idx = [r - 1 for r in reps]
            M = A[np.ix_(idx, idx)]
            ea = np.sort(np.round(np.linalg.eigvalsh(A), 6))
            em = np.sort(np.round(2 * np.linalg.eigvalsh(M), 6))
            ea = ea[np.abs(ea) > 1e-6]
            em = em[np.abs(em) > 1e-6]
            if ea.shape != em.shape or not np.allclose(ea, em):
                a6 = False
    ok["A6"] = a6
    print(f"A6_HALVING  nonzero spec(A_sig) == spec(2M): the eigenproblem descends to an "
          f"explicit (2^(n-2)-1)-dim matrix, n={SPEC_N}, all fibers  {'OK' if a6 else 'FAIL'}")

    # ---- A7 deflation guard vs V2 --------------------------------------------------------
    a7 = False
    n = 8
    S, H = tables[n], 1 << (n - 1)
    block = 1 << (n - 2)                    # V2's block is lo-labels [1, 2^{n-2})
    crossings = 0
    probes = 0
    for Llo in range(block, H):             # fibers with the top lo-bit set
        probes += 1
        inside = np.arange(1, block)
        crossings += int(np.count_nonzero((inside ^ Llo) >= block))
    a7 = crossings > 0 and probes > 0
    print(f"A7_DEFLATE  n={n}: over {probes} fibers with L_lo >= 2^(n-2), the pairing "
          f"l <-> l^L_lo carries {crossings} inside-block labels OUTSIDE V2's block "
          f"[1,2^(n-2)) => V2's doubling containment cannot imply the lemma "
          f"{'OK' if a7 else 'FAIL'}")
    ok["A7"] = a7

    # ---- A8 null control -----------------------------------------------------------------
    a8 = True
    worst = None
    for n in (6, 7):
        S, H = tables[n], 1 << (n - 1)
        for Llo in range(1, H):
            A = A_sig_fast(n, Llo, S)
            for Lp in range(1, H):
                if Lp == Llo:
                    continue
                los = np.arange(1, H)
                part = los ^ Lp
                keep = part != 0
                r, p = los[keep] - 1, part[keep] - 1
                v = int(np.count_nonzero(A[p, :] + A[r, :]))
                if v == 0:
                    a8 = False
                    worst = (n, Llo, Lp)
    # A8b — the complementary arm: RIGHT pairing, WRONG matrix. Take A built at fiber L'' and
    # test it against L_lo's pairing. If a foreign fiber's matrix ever satisfied L_lo's
    # antisymmetry, the lemma would be a property of the ambient sign table, not of the fiber.
    a8b = True
    worst_b = None
    for n in (6, 7):
        S, H = tables[n], 1 << (n - 1)
        for Llo in range(1, H):
            r, p, _ = pair_index(H, Llo)
            for Lpp in range(1, H):
                if Lpp == Llo:
                    continue
                B = A_sig_fast(n, Lpp, S)
                if int(np.count_nonzero(B[p, :] + B[r, :])) == 0:
                    a8b = False
                    worst_b = (n, Llo, Lpp)
    ok["A8"] = a8 and a8b
    print(f"A8_NULL_a   RIGHT matrix, WRONG pairing: every perturbed label L' != L_lo breaks "
          f"the antisymmetry of A_sig(L_lo), n=6,7  {'OK' if a8 else 'FAIL'}"
          + (f"  [preserved at {worst}]" if worst else ""))
    print(f"A8_NULL_b   RIGHT pairing, WRONG matrix: no foreign fiber's A_sig(L'') satisfies "
          f"L_lo's antisymmetry, n=6,7 => the lemma binds the fiber, not the ambient sign "
          f"table  {'OK' if a8b else 'FAIL'}" + (f"  [preserved at {worst_b}]" if worst_b else ""))

    print("=" * 78)
    if all(ok.values()):
        print("CD_TOWER_ZDFAN_VERDICT "
              "ZD_FIBER_ANTISYMMETRY_LEMMA__FACTORISATION_FOUND_LOWRANK_BOUND_PROVEN")
        print("CD_TOWER_ZDFAN_NOTE the involution l -> l ^ L_lo swaps P1 and P3 and negates "
              "them (A3), which preserves the resonance mask (A2) and flips the sign of "
              "A_sig (A1, 6 levels, all fibers, 0/1.2e9 violations). This yields the explicit "
              "factorisation A_sig = J^T M J with J J^T = 2I (A5) -- the C^T S C the prior "
              "rung listed as OPEN -- hence rank(A_sig) <= 2^{n-2}-1 for ALL n, with the "
              "isolated vertex at l = L_lo DERIVED from the level-(n-1) sub-lemma (A4), and "
              "an exact spectral halving to a (2^{n-2}-1)-dim matrix (A6). NOT CLAIMED: ∀n "
              "spectral completeness (#spectra = 3*2^{n-5}) -- A6 reduces that question, it "
              "does not answer it; rank EQUALITY remains measured (n<=10). Two of the three "
              "clauses of the in-tree resonance predicate are vacuous (A2). Numerical "
              "certificate over an exact integer sign table; D3 respected")
        return 0
    print("CD_TOWER_ZDFAN_VERDICT INCOMPLETE  failing=" +
          ",".join(k for k, v in ok.items() if not v))
    return 1


if __name__ == "__main__":
    t0 = time.time()
    rc = main()
    print(f"[{time.time() - t0:.1f}s]", file=sys.stderr)
    raise SystemExit(rc)
