#!/usr/bin/env python3
"""
CD-tower ZD fibers — the geometry localizes to the SIGNED resonance graph (∀n narrowing).

Companion to cd_tower_zd_fiber_spectral_classifier_contract.py (the adjacency spectrum is a
complete geometry invariant, n<=8). This rung NARROWS the open ∀n problem by halving it and
localizing the geometry to a signed cocycle where the ∀n machinery already lives.

Each fiber's annihilation graph is (per cd_tower_collapse_isomorphism.py, closed-form verified
n<=8) the Z2 SIGNED DOUBLE COVER of a "lo-graph" on the lo-labels: vertices (lo, s), s=+-1, with
(lo,s)~(lp,t) iff RESONANT(lo,lp) and s*t = eps(lo,lp), where (resonant, eps=-P1) come from four
cd_sigma products. Write A_R = unsigned resonance adjacency (eps ignored) and A_sig = the
eps-SIGNED adjacency, both on the H-1 = 2^{n-1}-1 lo-vertices.

  L1  DOUBLE-COVER REDUCTION.  spec(full annihilation graph G_n(L)) = spec(A_R) ∪ spec(A_sig),
      for EVERY fiber (verified via the true algebra product, all fibers, n=6,7). This halves the
      vertex count and is a structural identity: the closed-form adjacency has block form
      [[A+,A-],[A-,A+]] (A+ = eps+1 edges, A- = eps-1 edges), whose spectrum is
      spec(A+ + A-) ∪ spec(A+ - A-) = spec(A_R) ∪ spec(A_sig) -- so it holds ∀n wherever the
      closed-form rule does (that rule: verified n<=8).
  L2  THE GEOMETRY IS IN THE SIGN.  #distinct A_sig spectra = 3*2^{n-5} (the FULL nauty-complete
      classification) for n=6,7,8; while #distinct A_R spectra is strictly SMALLER (half:
      4,8,16... i.e. 3*2^{n-5}/... coarser). So the SIGNED resonance graph A_sig alone is a
      complete invariant; the unsigned graph A_R is not. The fiber geometry lives entirely in the
      SIGN COCYCLE eps = -P1 (a product of cd_sigma values).
  L3  LOCALIZATION (the narrowing).  The ∀n spectral-classification problem reduces to the signed
      resonance graph A_sig -- HALF the vertices, and its signs are the cd_sigma cocycle whose
      forall-n law (the seam-flip law) is already proven ∀n in Lean (SounioSeamFlip.lean). This
      does NOT prove ∀n completeness; it narrows it to a signed-cocycle spectral question.
      (Honest: A_R is NOT universal -- it varies by class, e.g. base-Fano/seam fibers are
      cocktail-party K_{(2^{n-2}-1)x2} but others are not -- so the geometry is not "all in A_sig
      because A_R is constant"; rather A_sig is a finer, complete invariant and A_R is a coarser
      one, and the classification is realised by A_sig alone.)

Verdict ZD_FIBER_GEOMETRY_LOCALIZES_TO_SIGNED_GRAPH. Numerical (machine precision); ∀n OPEN.
D3 respected.
"""
import numpy as np
from collections import Counter


def cd_sigma(a, b, bits):
    if a == 0 or b == 0:
        return 1
    if bits <= 1:
        return -1
    half = 1 << (bits - 1)
    aH, bH, aL, bL = a >= half, b >= half, a & (half - 1), b & (half - 1)
    if not aH and not bH:
        return cd_sigma(aL, bL, bits - 1)
    if not aH and bH:
        return cd_sigma(bL, aL, bits - 1)
    if aH and not bH:
        return cd_sigma(aL, bL, bits - 1) if bL == 0 else -cd_sigma(aL, bL, bits - 1)
    return -cd_sigma(bL, aL, bits - 1) if bL == 0 else cd_sigma(bL, aL, bits - 1)


def resonant(lo, lp, L, n):
    hi, hq = lo ^ L, lp ^ L
    P1 = cd_sigma(lo, lp, n) * cd_sigma(hi, hq, n)
    P2 = cd_sigma(lp, lo, n) * cd_sigma(hq, hi, n)
    P3 = cd_sigma(lo, hq, n) * cd_sigma(hi, lp, n)
    P4 = cd_sigma(lp, hi, n) * cd_sigma(hq, lo, n)
    return (P1 == P2 == P3 == P4, -P1)


def _mul(a, b, bits):
    out = {}
    for i, ci in a.items():
        for j, cj in b.items():
            k = i ^ j
            out[k] = out.get(k, 0) + cd_sigma(i, j, bits) * ci * cj
            if out[k] == 0:
                del out[k]
    return out


def full_adj(n, Llo):
    H = 1 << (n - 1)
    N = 1 << n
    L = Llo | H
    V = [{lo: 1, hi: (-1 if neg else 1)}
         for lo in range(1, H) for hi in range(H, N) for neg in (0, 1) if (lo ^ hi) == L]
    m = len(V)
    A = np.zeros((m, m))
    for i in range(m):
        for j in range(i + 1, m):
            if not _mul(V[i], V[j], n) and not _mul(V[j], V[i], n):
                A[i, j] = A[j, i] = 1
    return A


def lo_graphs(n, Llo):
    H = 1 << (n - 1)
    L = Llo | H
    los = list(range(1, H))
    m = len(los)
    AR = np.zeros((m, m))
    AS = np.zeros((m, m))
    for a in range(m):
        for b in range(a + 1, m):
            ok, e = resonant(los[a], los[b], L, n)
            if ok:
                AR[a, b] = AR[b, a] = 1
                AS[a, b] = AS[b, a] = e
    return AR, AS


def spec(A):
    return tuple(np.round(np.linalg.eigvalsh(A), 3).tolist())


def main():
    print("=" * 72)
    print("CD-tower ZD fibers — the geometry localizes to the SIGNED resonance graph")
    print("=" * 72)

    # L1 — double-cover reduction, ALL fibers, n=6,7 (via the true algebra product)
    l1 = True
    for n in (6, 7):
        H = 1 << (n - 1)
        for Llo in range(1, H):
            AR, AS = lo_graphs(n, Llo)
            if Counter(spec(full_adj(n, Llo))) != Counter(spec(AR)) + Counter(spec(AS)):
                l1 = False
                break
        print(f"L1_DOUBLE_COVER n={n}: spec(full) == spec(A_R) ∪ spec(A_sig) for all {H-1} fibers "
              f"{'OK' if l1 else 'FAIL'}")
        if not l1:
            break

    # L2 — the geometry is in the sign: A_sig complete, A_R coarser (n=6,7,8)
    l2 = True
    for n in (6, 7, 8):
        H = 1 << (n - 1)
        SR, SS = set(), set()
        for Llo in range(1, H):
            AR, AS = lo_graphs(n, Llo)
            SR.add(spec(AR)); SS.add(spec(AS))
        nauty = 3 * 2 ** (n - 5)
        ok = (len(SS) == nauty and len(SR) < nauty)
        l2 = l2 and ok
        print(f"L2_GEOMETRY_IN_SIGN n={n}: #A_sig spectra={len(SS)} (=nauty {nauty}, COMPLETE); "
              f"#A_R spectra={len(SR)} (coarser) {'OK' if ok else 'FAIL'}")

    l3 = l1 and l2
    print("=" * 72)
    if l3:
        print("CD_TOWER_ZDLOC_VERDICT ZD_FIBER_GEOMETRY_LOCALIZES_TO_SIGNED_GRAPH")
        print("CD_TOWER_ZDLOC_NOTE the fiber annihilation graph is the Z2 signed double cover of a "
              "lo-graph, so spec(G_n) = spec(A_R) ∪ spec(A_sig) (L1, all fibers n=6,7; structural ∀n via "
              "the [[A+,A-],[A-,A+]] block form). The SIGNED resonance graph A_sig alone is a COMPLETE "
              "geometry invariant (#spectra = 3*2^{n-5}, n=6,7,8) while the unsigned A_R is strictly "
              "coarser (L2). So the geometry lives in the SIGN COCYCLE eps=-P1 (a product of cd_sigma), "
              "and the ∀n classification problem NARROWS to the signed resonance graph -- HALF the vertices "
              "-- whose cocycle obeys the seam-flip law already proven ∀n in Lean. Honest: this NARROWS, "
              "does not close, ∀n; A_R varies by class (not universal). Numerical certificate; D3 respected")
        return 0
    print("CD_TOWER_ZDLOC_VERDICT INCOMPLETE")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
