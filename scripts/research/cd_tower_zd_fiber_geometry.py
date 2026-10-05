r"""
DEEP DIVE (item c): the fiber GEOMETRY carried by each orbit of the frozen 168 -- a hierarchical
"seam-stratified" structure on the growing zero-divisor graph.  Companion to
cd_tower_auto_action_on_zd_fibers.py (which proves the ORBIT structure); here we compute WHAT geometry
each orbit carries, as a function of its seam-tower coordinate y.

RECALL.  The frozen signed-monomial auto group (perm part PSL(2,7)) acts on the 2^{n-1}-1 ZD-fibers of
A_n with 2^{n-4} size-7 orbits, each indexed by a seam-tower coordinate y in F2^{n-4} (the label's bits
{3..n-2}; the octonion part x=L&7 ranges over the orbit).  Since M is an algebra automorphism it carries
a fiber's annihilation graph isomorphically, so every fiber in an orbit has ONE geometry = a function of
y.  This file computes that geometry (per-fiber intra-annihilation graph -- verts / edges / full degree
histogram) cheaply via a single representative fiber per y (matches analyze_fibers' per-fiber counts).

FINDINGS (VERIFIED n=6,7,8; closed forms fitted + the core law checked over all 4+8+16=28 orbits).
  Global fiber constants (independent of y):
    verts = 4(2^{n-2}-1),    max-degree Dmax = 4(2^{n-3}-1),    verts = 2*Dmax + 4.
  y = 0 (pure-octonion orbit): the fiber is Dmax-REGULAR (all verts at degree Dmax) -- the densest,
    most symmetric orbit; edges = verts*Dmax/2.
  CORE LAW (the main structural result).  For y != 0, let 2^b be the OUTERMOST (highest) active seam bit
  of y.  The number of vertices sitting at the maximal degree Dmax -- the fiber's "core" -- is
        core(y) = 4 (2^{n-1-b} - 1).
  So the core SHRINKS as the outermost active seam gets newer, and the NEWEST seam (b = n-2, the last
  doubling) always leaves a minimal core of exactly 4 vertices at degree Dmax, no matter how large n is.
  The full degree histogram is SELF-SIMILAR across n (a seam-indexed hierarchical / "meta-fractal"
  stratification: the sub-Dmax layers of an (n, y) fiber replicate a lower-level fiber's histogram).
  Example histograms (deg: count):
    n=8, y=  0:  {124: 252}                         (Dmax-regular)
    n=8, y=  8:  {60:192, 124:60}                   (outer seam 8  -> core 60)
    n=8, y= 64:  {4:248, 124:4}                     (outer seam 64 = newest -> core 4)
    n=8, y=120:  {4:8, 20:32, 84:192, 116:16, 124:4} (all seams -> nested strata)

INTERPRETATION / RELATION TO PRIOR WORK (honest).  This says the frozen 168's orbits are exactly the
STRATA of a hierarchical ZD geometry that grows and refines with n.  de Marrais's later corpus
("Placeholder Substructures", meta-fractals) describes fractal/self-similar ZD patterns in the 2^n-ions
combinatorially; the SELF-SIMILARITY here RESONATES with that language, but our object is different and
specific -- the degree-stratification of the ZD fibers INDEXED BY THE FINITE AUTOMORPHISM GROUP'S ORBITS
(the seam-tower coordinate y), which de Marrais does not compute (see the 2026-07-11 de Marrais deep-read
note: he has no automorphism-orbit decomposition of the ZD set).  Do NOT claim the fractal ZD phenomenon
itself as novel (de Marrais has it); claim the ORBIT<->STRATUM identification and the CORE LAW.

CORE-LAW RECURSION (identified + VERIFIED n=6,7,8,9; the forall-n proof reduces to ONE lemma).
  Reparametrize by k = n-1-b (doublings above the outermost active seam).  Empirically (verified n<=9):
        core(k) = 2*core(k-1) + 4,   core(1) = 4    =>   core(k) = 4(2^k - 1).
  The one-step structure is explicit: writing the top seam bit 2^{n-2}, the level-n core splits as
        core_n  =  { v in core_n : top bit OFF in both lo and hi_lo }   (= core_{n-1} LIFTED, unchanged)
               (+) { v in core_n : top bit ON }                          (= core_{n-1} + 4 new vertices).
  Both halves verified n=6->7 and n=7->8 (the topbit-off set equals core_{n-1} exactly; the topbit-on set
  has core_{n-1}+4 elements).  So the forall-n proof is reduced to a single ANNIHILATION-DOUBLING lemma:
  under A_n = A_{n-1}^2 (Moreno's inductive doubling, moreno.txt: A_n = R^2 with (a,b)(c,d) =
  (ac - d*b, da + bc*)), (i) a lifted level-(n-1) max-degree primitive stays max-degree, and (ii) the
  newly-excited (top-bit-on) max-degree primitives number exactly core_{n-1}+4.  This lemma is about the
  ANNIHILATION GRAPH, a genuinely different object from the sign cocycle / associator that all the other
  forall-n proofs in this lane rest on -- Moreno gives the doubling FRAMEWORK (ZD <-> associator, ZD-set
  ~= G2) but NOT this combinatorial per-primitive law, which would have to be derived.  NOT done here.

STATUS -- HONEST: closed forms (verts, Dmax) + CORE LAW + the doubling RECURSION STRUCTURE are
VERIFIED n=6,7,8,9 (all y; 4+8+16+... orbits).  This is a consistent recurrence across FOUR levels, not
a fit -- but it is NOT a proof.  Tag VERIFIED(n<=9) + CONJECTURE(forall n); the gap is the single
annihilation-doubling lemma above.  Independent of it, the ORBIT theorem (companion file) is PROVEN
forall n -- THAT is the robust, application-facing result (see the 2026-07-11 applications recon:
particle-physics "invariants up the CD tower" + sedenion-NN structured priors point at the orbit
structure, not the core law); the core law is a refinement whose forall-n proof is deliberately banked.
"""
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


def _mul(a, b, bits):
    out = {}
    for i, ci in a.items():
        for j, cj in b.items():
            k = i ^ j
            out[k] = out.get(k, 0) + cd_sigma(i, j, bits) * ci * cj
            if out[k] == 0:
                del out[k]
    return out


def _vec(c):
    lo, hi, neg = c
    return {lo: 1, hi: (-1 if neg else 1)}


def fiber_geom(n, Lfull):
    """Intra-fiber annihilation graph of fiber L=Lfull at level n: (verts, edges, degree Counter)."""
    H = 1 << (n - 1)
    N = 1 << n
    V = [_vec((lo, hi, neg))
         for lo in range(1, H) for hi in range(H, N) for neg in (0, 1) if (lo ^ hi) == Lfull]
    deg = [0] * len(V)
    for i in range(len(V)):
        for j in range(i + 1, len(V)):
            if not _mul(V[i], V[j], n) and not _mul(V[j], V[i], n):
                deg[i] += 1
                deg[j] += 1
    keep = [d for d in deg if d > 0]
    return len(keep), sum(keep) // 2, Counter(keep)


def main():
    ok = True
    for n in (6, 7, 8):
        H = 1 << (n - 1)
        seam_bits = [1 << b for b in range(3, n - 1)]
        seam_vals = [0]
        for b in seam_bits:
            seam_vals = seam_vals + [v | b for v in seam_vals]
        Dmax = 4 * ((1 << (n - 3)) - 1)
        fullv = 4 * ((1 << (n - 2)) - 1)
        good = True
        for y in sorted(seam_vals):
            v, e, h = fiber_geom(n, (y ^ 1) | H)
            core = h.get(max(h), 0)
            if y == 0:
                pred = fullv
            else:
                b = max(bb for bb in range(3, n - 1) if y & (1 << bb))
                pred = 4 * ((1 << (n - 1 - b)) - 1)
            good = good and core == pred and v == fullv and max(h) == Dmax and v == 2 * Dmax + 4
        ok = ok and good
        print(f"n={n}: verts=4(2^(n-2)-1)={fullv}=2*Dmax+4, Dmax=4(2^(n-3)-1)={Dmax}; "
              f"CORE LAW core=4(2^(n-1-b)-1) over {len(seam_vals)} orbits: {'OK' if good else 'FAIL'}")
    print("\nZD-FIBER GEOMETRY (per orbit):",
          "verts/Dmax closed forms + CORE LAW (outermost active seam governs the Dmax core) "
          "VERIFIED n=6,7,8; forall-n = CONJECTURE (needs the CD fiber-doubling recursion)." if ok
          else "MISMATCH")
    return ok


if __name__ == "__main__":
    main()
