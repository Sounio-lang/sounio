# Conjecture 6.8 (diam Γ_C^Z(𝕊) = 3): proof strategy and obstruction map

**Deep-research synthesis, 2026-09-01.** This document maps the full landscape of
the existence problem underlying Guterman–Zhilina's Conjecture 6.8, records the
verified reduction, the odd-degree existence mechanism, the exact stratification,
and — crucially — a catalogue of six standard existence mechanisms with the
*specific* obstruction that defeats each. The odd-degree phenomenon is measured
universally; its uniform proof is the open core.

## 1. The reduction (verified against [GZ26])

Conjecture 6.8 ⟺ every pair of zero-divisor vertices x=(a,b), x′=(a′,b′) at
distance ≥ 3 admits a **witness**: u ∈ Im C(x), w ∈ Im C(x′), u,w ≠ 0, [u,w]=0
(then x — u — w — x′ is a length-3 path).

**The vertex (zero-divisor) constraint is vacuous** [proved]. GZ vertices are
lines of elements with zero-divisor imaginary part. For the path we need u,w to
be vertices; u ∈ Im C(x) is pure, so u is a vertex iff u ∈ Z(𝕊). Using GZ Prop
4.1 (1,a,b,c,d,ab,ac,ad orthonormal), for
u = k₀x + k₁(c,d) + k₂(d,−c) + k₃(ac,−ad) + k₄(ad,ac):
n(u₁)=n(u₂)=Σkᵢ² (identically), Re(u₁)=0, ⟨u₁,u₂⟩ = −k₁k₂+k₂k₁+k₃k₄−k₄k₃ = 0.
By GZ Lemma 4.12, u ∈ Z(𝕊). **Every nonzero u ∈ Im C(x) is a zero divisor**, so
the witness locus is cut by [u,w]=0 alone.

## 2. The witness locus as a determinantal variety

Fix frames of Im C(x), Im C(x′). The map w ↦ [u,w] is a matrix M(u) (15 nonzero
rows in W₁₁, 5 columns), **linear in u**. A witness w exists iff ker M(u) ≠ 0, so

  D(x,x′) = { [u] ∈ ℝP⁴ : rank M(u) ≤ 4 }  (the 5×5 minor ideal).

The commutator tensor C[a][i][j] = [f^U_i, f^W_j]_a is exact integer, verified
against census witnesses (‖Σ C k m‖ = 2·10⁻⁸ at a real witness). Real point of D
⟹ real u ⟹ real w ∈ ker M(u) ⟹ real witness.

## 3. The odd-degree existence mechanism

**Measured (Singular primdecGTZ over ℚ, ~17 configs fully decomposed + ~30 via
(dim, mult)):** the **reduced top-dimensional part** of D has **odd** degree in
every configuration:
- Generic pairs (d ≥ 3): D is a curve (proj. dim 1), reduced degree **7**,
  splitting as 2 conics + 1 cubic (2+2+3) — the ternary cubic is the odd part.
- Symmetric config (e₁,e₂)v(−2e₂,e₃+e₄−e₆−e₇): reduced degree 7 = line + 3 conics
  (1+2+2+2). (The pencil of R16 is this line, a symmetry artifact.)
- Aligned degenerate pairs (still d ≥ 3, e.g. par duro (e₁,e₂)v(e₂,e₃), and
  (e₁,e₂)v(e₂,e₁)): dim D = 2 or 3; top component a real hyperplane {k₀=0} or a
  rational surface, reduced degree 1 or 3 (odd). Scheme degree can be even by
  multiplicity (e.g. hyperplane with multiplicity 2), but the reduced degree is
  odd.

**Existence (conditional on uniform oddness).** A reduced real projective variety
of dimension δ and odd degree d has a real point: a generic real linear subspace
of codimension δ meets it in a conjugation-stable 0-cycle of odd degree, so an odd
(≥1) number of real points. This covers all strata.

## 4. Why the standard mechanisms fail (the obstruction map)

The existence of a witness evades every standard topological/enumerative forcing
tool. Each was checked; each has a *specific* obstruction:

| # | Mechanism | Specific obstruction |
|---|-----------|----------------------|
| 1 | Stiefel–Hopf (nonsingular bilinear ℝ⁵×ℝ⁵→ℝ¹¹ forbidden) | condition range p−m<i<n = 6<i<5 is **empty** ⟹ vacuous at p=11; the effective rank-7 target is (u,w)-dependent, not a fixed reduction |
| 2 | Eisenbud 1-genericity ⟹ CM determinantal of expected codim/degree | expected codim of rank≤4 in 15×5 is 11; observed **3** ⟹ M(u) is far from 1-generic (rank laws impose syzygies) |
| 3 | Euler class on the sphere cover S⁴×S⁴ | H⁷(S⁴×S⁴)=0 ⟹ Euler class **0** ⟹ no obstruction; forcing lives only in the ℤ/2 descent to ℝP⁴ |
| 4 | Twisted Euler class w₇ = α⁴β³+α³β⁴ on B = ℝP⁴×ℝP⁴ | E₇ lives only on B∖Z; w₇\|_{B∖Z} = 0 (i* = 0, R18). The section's zeros are on Γ ⊂ B∖Z where the class dies |
| 5 | A single mod-2 characteristic number = deg D | deg D **varies** (7, 3, 1 across strata) ⟹ not a deformation-invariant number; only the *parity* is invariant |
| 6 | Miracle flatness (deg constant on connected stratum) | D is **not** Cohen–Macaulay: codim 3 but proj.dim 5 (Auslander–Buchsbaum depth 0 ≠ 2) ⟹ family not flat via the minor ideal |

The recurring lesson: the object is **not** a bundle class (attempts 3,4 chase the
Euler class; R13 chased an excision image — both are the "reach for a
characteristic class because the number looks topological" error). The real
invariant is the **parity of the real degree**, whose source is complex-conjugation
stability, not a Chern class.

## 5. The viable route and its two remaining lemmas

The only route not immediately obstructed is **degree constancy on the connected
generic stratum**, using the connectedness input from [KY] (Khalil–Yiu: G₂ =
Aut(O) acts freely transitively on unit orthogonal ZD pairs; the ZD variety is a
bundle over S⁶ with S⁵ fibres, connected).

**Lemma A (connectedness).** The generic stratum U = {pairs with dim D = 1} is a
connected, dense open subset of the (connected) ZD-pair space. [Needs: the bad
locus B = {dim D ≥ 2} has real codimension ≥ 2. Measured: dim D = 1 on ~17 random
pairs; B appears only at hand-picked aligned basis pairs (measure zero).]

**Lemma B (degree constancy).** deg D is constant on U. [Needs a flatness
substitute since D is not CM (obstruction #6): e.g. constancy of the Hilbert
polynomial of the reduced curve √D on U, or a resolution whose Betti numbers are
constant on U. Measured constant = 7.]

Given A + B: deg D ≡ 7 (odd) on U ⟹ real witness for every generic d ≥ 3 pair.
The bad locus B is handled separately: each such pair carries a real linear
component of D (hyperplane), giving real witnesses there too. (This last needs a
uniform statement over B; verified on the examples computed.)

## 6. What would close it

Either:
(i) **prove Lemma B** — the reduced witness curve has constant (hence odd) degree
    on the connected generic stratum, via a resolution/Hilbert-polynomial argument
    that survives non-CM-ness; plus Lemma A (bad-locus codimension) and the
    bad-locus real-component statement; or
(ii) **a direct real-algebraic parity theorem**: the mod-2 real degree of this
    specific structured determinantal locus is 1, proved from the Cayley–Dickson
    multiplication (a Cayley–Dickson analogue of a real Bezout/conjugation parity).

The number 7 = dim Im O is suggestive of a structural origin (via GZ Thm 4.15,
components of Γ_O ↔ lines in Im O ≅ ℝP⁶); a map D → ℝP⁶ of the right degree would
give (ii).

## Appendix: verification artifacts
- Exact tensor + frames: `data/conj68_frames_13370001.txt`, census FRAME_U/W emit.
- Exact decomposition (component equations incl. the cubic):
  `data/conj68_ulocus_decomp_13370001.txt`, `data/conj68_ulocus_degrees_8configs.txt`.
- Tooling: Singular (`minor(M,5)`, `primdecGTZ`, `mres` for pd), verified in ℚ.
- Reduction & rank laws: `conj68_manuscript_draft.md` §§1–5, 8.
- Full attack log incl. all retractions: `conj68_attack_log_2026-08-31.md` (R1–R22).

[GZ26] Guterman–Zhilina, arXiv:2608.26903.  [KY] Khalil–Yiu (1997), cited therein.
