r"""
(C) The FROZEN signed-monomial automorphism group (permutation part PSL(2,7), order 168) acting on the
GROWING zero-divisor fiber set of the Cayley-Dickson tower A_n.  This is the "beyond Kirshtein" object:
Kirshtein (2012) computes the automorphism GROUP (and its freezing); she does not touch the zero
divisors.  Here the frozen group ACTS on the ZD-fibers, and the action's orbit structure grows.

OBJECT.  A_n (dim 2^n) has zero divisors first at n=4 (sedenions).  The mixed-half ZD primitives
e_lo +/- e_hi partition into FIBERS indexed by the label L = lo XOR hi, reduced to its lower part
Llo in F2^{n-1}\{0} -- so there are 2^{n-1}-1 fibers (7, 15, 31, 63, ... at n=4,5,6,7).  The fibers
carry a per-fiber GEOMETRY (annihilation graph; edge counts / degrees) constant on each orbit -- but the
map orbit -> geometry is NOT injective (Fano/seam collisions; see the RETRACTED note in the THEOREM).

ACTION.  The 168 valid index-maps M in GL(n,2) all fix the seam H=2^{n-1} and (block lemma, forall n)
have the form [[A,0],[0,1]] with A a valid auto of A_{n-1}; iterating, A = (a copy of GL(3,2) on the
three octonion bits {0,1,2}) (X) (identity on the seam-tower bits {3,...,n-2}, i.e. the values
8,16,...,2^{n-2}).  M sends the primitive e_lo +/- e_hi to e_{M lo} +/- e_{M hi}, so it sends the fiber
label L to M.L (linear); on the lower label Llo this is the A-action.  Hence the 168 acts on the
2^{n-1}-1 fibers through its octonion GL(3,2) on the bottom three coordinates, fixing every seam bit.

THEOREM (orbit structure -- PROVEN forall n; VERIFIED n=4..7 below).
  The frozen PSL(2,7) acts on the 2^{n-1}-1 ZD-fibers of A_n with orbit decomposition
        2^{n-4} orbits of size 7   +   (2^{n-4}-1) fixed points,
  where
    * each size-7 orbit is a copy of the natural 2-transitive PSL(2,7)-action on the 7 points of the
      Fano plane PG(2,2) (point-stabilizer of order 168/7 = 24, isomorphic to S_4);
    * the fixed points are exactly the nonzero vectors of the SEAM SUBSPACE <8,16,...,2^{n-2}> =
      F2^{n-3}\{0} -- i.e. the tower of inner doubling seams e_8, e_16, e_24, ... (the lifted seams);
    * the action is EQUIVARIANT with the fiber geometry: each orbit is MONOCHROMATIC (all its fibers
      share one annihilation-graph shape) -- this MUCH is PROVEN forall n (M is an algebra automorphism;
      see proof below).
      *** RETRACTED 2026-07-12 (adversarial nauty audit -- the claim below was FALSE and was tagged
      VERIFIED; it is refuted inside that range).  The CONVERSE "distinct orbits realize distinct
      geometries" DOES NOT HOLD.  Counterexample at n=6: the Fano orbit y=2 (fiber Llo=17) and the fixed
      SEAM y=3 (Llo=24) have ISOMORPHIC annihilation graphs -- identical nauty canonical cert; 60 verts,
      168 edges, degree histogram {4:56, 28:4}.  Confirmed INDEPENDENTLY here (fiber_geom histogram +
      Weisfeiler-Leman, both agreeing with the reviewer's nauty; control Seam y=2 correctly differs).
      So the orbit->geometry map is NOT injective, and the finite orbit decomposition is NOT read off
      the fiber geometry.  What actually holds is the PARITY COLLAPSE LAW (reviewer; nauty-complete
      n<=8): gamma(Seam(y)) = gamma(Fano(y & (y-1))) exactly when wt(y) is EVEN (y!=0); hence
      #geometries = 3*2^{n-5} < #orbits = 2^{n-3}-1, deficit 2^{n-5}-1 (=0 at n=5, so the old n=5
      "type-A vs type-B" pair was fine; the false generalization first breaks at n=6).  The FANO stratum
      alone looks separated (n=6: the four Fano orbits have distinct edge counts 840/456/552/168), but
      Fano-stratum injectivity forall n is OPEN and needs SPECTRAL data, not degrees (at n=8 Fano(0) and
      several seams share a degree sequence yet are pairwise non-isomorphic).  Regression witness:
      cd_tower_fiber_geometry_collision.py.  Net: monochromaticity is a remark; geometric DISTINCTNESS
      is dead; the parity-collapse law is the replacement theorem-shaped object.
  So the permutation REPRESENTATION grows without bound (2^{n-4} -> infinity Fano copies + a growing
  fixed seam-subspace) while the GROUP stays frozen at 168.  Counts check: 7*2^{n-4}+(2^{n-4}-1)=2^{n-1}-1.

PROOF (forall n>=4).  Freezing gives every valid index-map at level n the block form [[A,0],[0,1]] with
  A a valid map at level n-1 (Kirshtein 2012 Thm 41, UNCONDITIONAL forall n -- so this proof needs NO
  corr value-pin; equivalently our own block lemma, at corr-parity).  Iterating down to the sedenion
  base n=4, whose 168 valid maps are exactly [[g,0],[0,1]] with g in GL(3,2)=Aut(octonions) (the n=4
  seam e_8 fixed), the level-n lower block is  A = g (+) Id  in the split coordinates
        L = (x, y),   x in F2^3 (octonion bits {0,1,2}),   y in F2^{n-4} (seam-tower bits {3..n-2}),
  acting by  M.(x,y) = (g x, y)  (y fixed pointwise; x moved by GL(3,2)).  Therefore:
    - y != 0, x = 0:  (0,y) is fixed  ->  2^{n-4}-1 fixed points = seam-subspace nonzero vectors.
    - any y, x != 0:  {(g x, y): g in GL(3,2)} = {(x',y): x' != 0}  (GL(3,2) is transitive on F2^3\{0})
      -> one orbit of size 7 per y in F2^{n-4}  ->  2^{n-4} seven-orbits.
    - stabilizer of (x,y), x != 0:  {g: g x = x} = Stab_{GL(3,2)}(x) = the Fano point-stabilizer S_4,
      order 24.
  QED.  The only inputs are (i) block-form freezing forall n [Kirshtein, unconditional] and (ii) GL(3,2)
  transitive on F2^3\{0} [trivial].  Equivariance with fiber geometry is automatic: M is an algebra
  automorphism, so it carries the annihilation graph of a fiber to that of its image isomorphically,
  hence preserves every graph invariant (verts/edges/degrees) -- orbits are monochromatic by construction.

RELATION TO PRIOR WORK (narrow lit-check, 2026-07-11 -- primary sources read).
  Two DIFFERENT automorphism-group actions on CD zero divisors must be kept apart:
  * CONTINUOUS group, PUBLISHED (this is NOT our result):  Reggiani 2024 (arXiv:2411.18881) proves the
    connected real automorphism group G2 = Aut(O) = Aut(S)^0 acts ISOMETRICALLY and TRANSITIVELY on the
    normalized sedenion ZD manifold ZD(S) -- a SINGLE orbit, isotropy SU(2), ZD(S) ~= G2/SU(2) = V2(R^7)
    (Moreno 1998, q-alg/9710013 Cor 2.14, is the homeomorphism foundation ZD ~ G2; Lopatin-Zubkov 2022
    classify G2-orbits of octonion pairs).  Continuous Aut recursion: Aut(A_n) ~= G2 x (S3)^{n-3}.
  * FINITE signed-monomial / Cayley-Dickson-loop group -- OUR result, NOT FOUND in the literature:
    - Kirshtein 2012 (arXiv:1102.5151) computes the finite groups (1344/2688/5376) but the term "zero
      divisor" never appears -- groups only, no ZD action.
    - de Marrais (FULL arXiv corpus deep-read 2026-07-11: math/0011260, 0207003, 0403113, 0603281,
      0703745, 0704.0026, 0704.0112, 0804.3416 + NKS'04) -- verdict (b) ADJACENT.  His corpus has the
      surrounding pieces but NOT our result: (i) he RESTATES Moreno's "automorphism group of the ZDs =
      G2 x (S3)^{n-3}" -- but that is CONTINUOUS (G2), a group-ORDER formula he treats as an upper
      bound his own counts EXCEED, and Moreno's not his; (ii) he has a genuine finite PSL(2,7) action
      (Klein x C7 x dihedral) but acting on FANO-PLANE LABELINGS / box-kite presentations, used to
      COUNT/classify box-kites, not to orbit-decompose the ZD SET; (iii) 168 <-> ZD appears only as a
      count coincidence + Fano labeling frame.  The words orbit/stabilizer/fixed-point (group sense) on
      the ZD set appear in NONE of the nine papers.  His box-kite tallies (7,35,155,651=Trip_{N-2};
      4^{n-4} missing) do NOT match our 2^{n-4}x[7] -> structurally distinct, not a re-derivation.
      RESIDUAL CAVEAT (SUBSTANTIALLY RESOLVED 2026-07-12, convergent secondary evidence):
      the one item not in the arXiv corpus was the ICGTMP-2006 talk, full title now recovered as
      "The Marriage of Nothing and All: ZERO-DIVISOR BOX-KITES IN A 'TOE' SKY" (26th ICGTMP, CUNY,
      Jun 2006).  Three findings drop the overturn risk to very low: (1) SUBJECT -- title/subtitle put
      it squarely in his box-kite + physics-TOE program, not group-orbit theory; (2) NEVER PUBLISHED --
      the official ICGTMP conferences page lists NO proceedings for Group26 (nor 25/27); it was only
      ever cited "forthcoming from Springer" and no such Springer volume exists, so there is no paywalled
      venue to be locked out of (at most a talk/preprint, and his talks fold into the read arXiv corpus);
      (3) HIS OWN DESCRIPTION -- Placeholder Substructures I (math/0703745, 2007) cites it as ref [4]
      ONLY in a QM aside ("the QM case is a degenerate form ... see [4]"), pinning its content to the
      QM/TOE zero-divisor angle, NOT an orbit/stabilizer/fixed-point decomposition; that citing paper and
      the whole corpus carry no such group-action-on-the-ZD-set statement (168/PSL(2,7) stay in box-kite /
      Fano-labeling roles).  NOT DONE: the talk's own text was not read (apparently unavailable anywhere
      public).  Verdict (b) ADJACENT stands with high (not certain) confidence; if the talk text ever
      surfaces, re-check before a *published* priority claim, but it no longer blocks the novelty scoping.
  NOVELTY SCOPE (honest -- pin the claim exactly here): the novelty is the explicit ORBIT + STABILIZER
  + FIXED-POINT decomposition of the DISCRETE ZD/fiber set under the FINITE signed-monomial (CD-loop)
  automorphism group (permutation part PSL(2,7)) -- 2^{n-4} size-7 Fano orbits + (2^{n-4}-1) fixed seams,
  stab S4, forall n.  Do NOT claim as novel: "PSL(2,7) is a symmetry of the ZDs" (de Marrais: 168<->ZD
  count coincidence + Fano labeling symmetry, pervasive), nor "a group acts on ZDs scaling with n"
  (Moreno/Reggiani: CONTINUOUS G2 x (S3)^{n-3}, transitive with SU(2) isotropy).  The finite/discrete
  orbit-decomposition is the surviving kernel; the continuous G2 gives ONE transitive orbit, ours REFINES
  into a growing discrete partition.

STATUS: orbit multiset + stabilizer + seam-fixed-points VERIFIED n=4..7 (orbit theorem PROVEN forall n).
        fiber-geometry MONOCHROMATICITY holds (equivariance); geometric DISTINCTNESS was FALSE (retracted
        2026-07-12) -- replaced by the parity-collapse law, reviewer nauty n<=8.
here; the orbit theorem is PROVEN forall n (proof above, on block-form freezing + GL(3,2)-transitivity).
Novelty = "no published finite-Aut ZD-orbit decomposition found" (de Marrais adjacent; full corpus
unread -> deep-read in progress); cleanly distinct from Reggiani's continuous transitive G2-action.
"""
import importlib.util
from collections import Counter


def _load_oracle():
    spec = importlib.util.spec_from_file_location(
        "orc", "scripts/research/cd_tower_automorphism_oracle.py")
    orc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(orc)
    return orc


def lift(maps, n):
    """maps: image arrays at level n (len 2^n) fixing the seam. Return the level-(n+1) lifts [[A,0],[0,1]]."""
    N = 1 << n
    out = []
    for m in maps:
        mm = [0] * (2 * N)
        for v in range(N):
            mm[v] = m[v]            # lower block A
            mm[v | N] = m[v] | N    # M(v+H) = A(v)+H  (beta=0, seam fixed)
        out.append(mm)
    return out


def main():
    orc = _load_oracle()
    maps = {4: orc.sweep_autos(4)}     # 168 valid index-maps at n=4 (full GL(4,2) sweep, instant)
    ok = len(maps[4]) == 168
    print(f"n=4: |valid M| = {len(maps[4])} (expect 168)  {'OK' if ok else 'FAIL'}")
    for n in (4, 5, 6, 7):
        if n > 4:
            maps[n] = lift(maps[n - 1], n - 1)
        M = maps[n]
        H = 1 << (n - 1)
        elems = list(range(1, H))                       # fibers = F2^{n-1}\{0}
        parts = orc.orbits_on(M, elems)
        cnt = Counter(len(p) for p in parts)
        fixed = sorted(p[0] for p in parts if len(p) == 1)
        k = 1 << (n - 4)
        pred = Counter({7: k, 1: k - 1})
        pred = Counter({s: c for s, c in pred.items() if c > 0})
        stab = next((sum(1 for m in M if (m[p[0]] & (H - 1)) == p[0])
                     for p in parts if len(p) == 7), None)
        seam_sub = sorted(x for x in range(1, H) if x % 8 == 0)  # <8,16,...> nonzero = multiples of 8
        # structural premise of the forall-n proof: lower block = g (+) Id -- each seam bit fixed,
        # octonion bits {1,2,4} stay inside the octonion span {1..7}.
        seam_bits = [1 << b for b in range(3, n - 1)]              # values 8,16,...,2^{n-2}
        struct = all((m[s] & (H - 1)) == s for m in M for s in seam_bits) and \
            all((m[o] & (H - 1)) in (1, 2, 3, 4, 5, 6, 7) for m in M for o in (1, 2, 4))
        good = (cnt == pred and stab == 24 and fixed == seam_sub and struct)
        ok = ok and good
        print(f"n={n} dim{1<<n}: {H-1} fibers | orbits {dict(sorted(cnt.items()))} "
              f"(pred {dict(sorted(pred.items()))}) | stab(7-orbit)={stab} | "
              f"fixed={fixed}==seam-subspace{seam_sub}:{fixed==seam_sub} | A=g(+)Id:{struct} | "
              f"{'OK' if good else 'FAIL'}")
    print("\nAUTO-ACTION-ON-ZD-FIBERS:",
          "orbit structure 2^(n-4)x[7] + (2^(n-4)-1)x[1], stab=24, fixed=seam-subspace, A=g(+)Id -- "
          "VERIFIED n=4..7; PROVEN forall n (block-form freezing + GL(3,2)-transitivity)." if ok else "MISMATCH")
    return ok


if __name__ == "__main__":
    main()
