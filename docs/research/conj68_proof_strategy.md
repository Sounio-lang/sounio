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

## 7. The subalgebra reframe (deep-research, 2026-09-01) — and its precise scope

**Key structural observation (A. advisor).** The degrees 7, 3, 1 are dim Im 𝕆,
dim Im ℍ, dim Im ℂ = 2^k − 1 — odd for a **structural** reason (composition
subalgebras of 𝕆 have dimension 2^k), not an enumerative one. Let A(x,x′) be the
ℝ-subalgebra of 𝕆 generated by {a, b, a′, b′}. Then, for generic (random) pairs:

**deg D = dim Im A** [measured, exact octonion linear algebra + Singular]:
- dim A = 8 (generates 𝕆): D is the curve 2 conics + 1 cubic, deg 7 = dim Im 𝕆.
- dim A = 4 (quaternion): deg 3 = dim Im ℍ.
- dim A = 2 (complex): deg 1.
Oddness is then free (2^k − 1). The stratification is explained: more aligned
pair → smaller generated subalgebra → smaller degree → larger D.

**Scope and limit [measured].** The identity deg D = dim Im A holds for GENERIC
pairs (Zariski-dense: generating 𝕆 is an open condition, dim A = 8). It is NOT a
universal scheme identity: on the measure-zero **aligned basis locus**
(e_i,e_j)v(e_k,e_l) it can deviate — e.g. one aligned pair with dim A = 8 gave
component degrees (2,1),(2,1),(2,2), sum 6 (the generic cubic having split into a
line + a conic), and a super-degenerate aligned pair has D = ℙ⁴ (likely d ≤ 2).

**What is robust across ALL strata [measured, ~50 configs]:** D has an
**odd-degree component** — a cubic generically, or a line/hyperplane when it
degenerates. This is what gives the real point (§3).

**The preservation mechanism — a concrete proof route.** Under specialization
from a generic pair, the generic cubic component (degree 3, odd) can only break
into pieces summing to 3, i.e. (line + conic) = (odd + even) or stay a cubic —
**an odd-degree component always survives**. So:
1. *Generic step:* for pairs with dim A = 8, D contains an irreducible cubic
   (degree 3) — to be proved from the Cayley–Dickson structure (the generic value
   deg D = dim Im A = 7 with the 2+2+3 split).
2. *Specialization step:* the cubic's degree (3, odd) is preserved mod 2 under
   degeneration (a limit of a degree-3 cycle is a degree-3 cycle; its odd part
   cannot vanish), so every pair retains an odd-degree component.
3. *Real point:* odd-degree real component ⟹ real witness (§3).

This replaces the six failed mechanisms (§4) and the non-CM flatness route (§5):
the parity is carried by a **specific geometric component (the cubic)** whose
existence is structural (tied to dim Im 𝕆 = 7) and whose oddness is preserved by
specialization — not by any single characteristic number.

**Remaining to prove:** (1) the generic cubic exists (deg D = dim Im A = 7 for
dim A = 8), from the CD multiplication — the clean algebraic core; (2) the mod-2
preservation of an odd component under specialization to every d ≥ 3 pair.

## 8. The octonion witness equations (deep-research Step 1, verified)

Writing u=(u₁,u₂), w=(w₁,w₂) (doubly pure), the Cayley–Dickson product gives the
commutator explicitly:

  [u,w] = ( [u₁,w₁] + 2·Im(ū₂ w₂),  2·(w₂u₁ − u₂w₁) ).   [verified numerically = 0
  at a real witness: ‖·‖ = 0 for both components]

So a witness ⟺ the two **octonion** equations
  (A)  [u₁, w₁] = −2·Im(ū₂ w₂),
  (B)  w₂ u₁ = u₂ w₁.

**The Φ_u reduction (toward deg D = 7).** For n(u₁) ≠ 0, right multiplication
R_{u₁} is invertible on 𝕆, so (B) solves w₂ uniquely from w₁:
  w₂ = Φ_u(w₁),  Φ_u = R_{u₁}⁻¹ ∘ L_{u₂},
where L_{u₂}(w₁)=u₂w₁, R_{u₁}(w₂)=w₂u₁, and R_{u₁}⁻¹ = R_{ū₁}/n(u₁) (octonion
inverse). Φ_u is rational (polynomial after clearing n(u₁)) in u. A witness then
exists iff there is a nonzero w₁ with:
  (i) (w₁, Φ_u(w₁)) ∈ Im C(x′)  (the frame constraint — a linear condition), and
  (ii) equation (A) holds with w₂ = Φ_u(w₁).
D is the u-locus where this linear-algebra system in w₁ degenerates. Because Φ_u
carries one factor of 1/n(u₁) and the frame/(A) constraints are linear, the
degeneracy determinant is a form in u whose degree (after clearing n(u₁)²) is the
source of deg D = 7. **This is the concrete analytic core to finish: compute the
degree of that degeneracy form and show it equals dim Im 𝕆 = 7 when x,x′ generate
𝕆.** The Φ_u operator makes it a question about left/right octonion
multiplication operators (Zorn-vector / Moufang identities), i.e. pure 𝕆 algebra
— no sedenion determinantal computation.

Status: equations (A),(B) and the Φ_u reduction are exact and verified; the degree
computation from Φ_u is the remaining analytic step (Step 1 of §7).

## 9. Two attacks on deg D = 7 (2026-09-01) — verdict

**Attack 1 (Thom–Porteous / Chern).** D is the corank-1 locus of O^5 → E⊗O(1)
(E = effective rank-7 target), so deg D = c_3(E⊗O(1)) with
c_3(E⊗O(1)) = 35h³ + 15 c₁(E)h² + 5 c₂(E)h + c₃(E); matching 7 needs
15c₁+5c₂+c₃ = −28. **But E is exactly the bundle E₇ that fails to exist over Z
(§6); computing its Chern classes reintroduces the dead object.** This route is
structurally the wrong tool — the same error as chasing w₇. Abandoned.

**Attack 2 (Φ_u / octonion operators — the live route).** Verified: writing
Im C(x′) as the graph {(w₁, Ψ(w₁))}, equation (B) becomes (Ψ − Φ_u)(w₁) = 0 with
Φ_u = R_{u₁}⁻¹L_{u₂}. Both blocks (A) and (B) have rank 5 generically (neither
alone forces a witness); D is where the **combined** 16×5 system drops to rank 4.
After clearing n(u₁), the entries are degree 2 in u. The degree of the combined
degeneracy form is deg D = 7, but extracting "= dim Im 𝕆" from here requires the
algebra of the operators R_{u₁}⁻¹L_{u₂} (Moufang identities, the Zorn-vector
matrix representation of octonion left/right multiplication). This is a genuine
analytic project, not an in-session computation.

**Honest verdict.** deg D = 7 is exact and measured on ~20 configs; the two
avenues to *prove* it are (1) dead (reintroduces E₇) and (2) live but a real
research computation in octonion operator algebra. The problem is reduced to its
sharpest form — the parity/degree lives in the operators R_{u₁}, L_{u₂} on 𝕆 —
but is not closed here. No proof is fabricated.
