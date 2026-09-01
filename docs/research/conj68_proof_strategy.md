<!-- docs:meta
topic_id: repo.docs.research.conj68-proof-strategy
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.conj68-proof-strategy
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

>>> RESOLVED 2026-09-01: Conjecture 6.8 is a THEOREM — Zhilina, arXiv 2608.26890, Theorem 4.13 (diam Γ_C^Z(𝕊)=3), a companion paper to 2608.26903 that we had cited but never fetched. Proof = dimension count: find (a',b')∈Im C(x')∩span{(a,-b),(b,a),(ab,0),(0,ab)}^⊥ (5∩codim4≥1), then Lemma 4.12 gives d(x,(a',b'))≤2; length-3 path. VERIFIED computationally 8/8. See refs/zhilina_diameter_commutativity_2608.26890_THEOREM.md. Our §§2-5 rank laws stand independently.

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

## 10. ℚ(t) computation FALSIFIES "deg D = dim Im A" (2026-09-01, exact)

Using exact arithmetic over the function field ℚ(t) for a generic 1-parameter
family x=(e₁,e₂), x′(t)=((1+t²)e₄, (1−t²)e₅+2t e₆) (dim A = 8, generates 𝕆):

**deg D = 9** over ℚ(t) (scheme mult), with reduced curve components (Singular,
t=2 and t=3 rational members) (dim2,deg): (2,1),(2,2),(2,2),(2,1) — reduced total
degree **6 (EVEN)**. Compare 13370001 (also dim A = 8): reduced 2+2+3 = **7 (ODD)**.

**Consequences (correcting §7–§9):**
- **"deg D = dim Im A = 7" is FALSE.** Among configs that generate 𝕆 (dim A = 8),
  the reduced degree varies (6, 7) and even its parity varies. The subalgebra
  reframe gave the right *intuition* (odd components appear) but the wrong
  value-identity. The degree is not the subalgebra dimension.
- Neither the reduced top-dimensional degree nor the scheme multiplicity is
  uniformly odd (measured: reduced 6 and 7; scheme mult 1, 2, 7, 9).
- **The only robust fact across all configs:** D contains an **odd-degree,
  ℝ-rational component** — a cubic (deg 3), a pair of lines (deg 1), or a
  hyperplane (deg 1). In the deg-9 family the generic cubic has split into
  line + conic, leaving two degree-1 lines. That odd component gives the real
  point (§3).

**Status.** The exact ℚ(t) machinery (the native operators) did its job: it
disproved a clean but false statement before it hardened into the manuscript. The
open core is now precisely: **prove D always contains an odd-degree ℝ-rational
component** — a statement about the *component structure* of the determinantal
locus, not about its total degree (which is genuinely config-dependent). The Φ_u
operator reduction (§8) remains the tool; the question is why an odd-degree piece
is always present in the degeneracy locus of (Ψ − R_{u₁}⁻¹L_{u₂}).

## 11. Scoreboard of clean reframes — all falsified by exact computation except one

Exact computation (Sounio native operators + Singular over ℚ and ℚ(t)) served
primarily to FALSIFY clean-but-false claims before they hardened:

| Clean reframe | Verdict |
|---|---|
| deg D = 7 uniform | FALSE — ℚ(t) generic family gives 9 |
| reduced top-dim degree odd | FALSE — that family's reduced degree is 6 (even) |
| scheme multiplicity odd | FALSE — (e₁,e₂)v(e₂,e₁) gives mult 2 |
| deg D = dim Im A | FALSE — dim A = 8 configs give both 7 and 9 |
| **∃ odd-degree ℝ-rational component of D** | **survives** (~50 configs); no structural mechanism |

**Honest status of Conjecture 6.8.** All evidence supports it (a real witness in
every configuration examined). But it is a ∀-statement over a continuous ~8-dim
moduli of pairs (mod G₂); computation at any scale gives evidence, not proof. The
proof is structural, and the clean structural reframes above are all falsified.
The surviving open core — "D always contains an odd-degree ℝ-rational component"
— is a statement about the component structure of the degeneracy locus of
(Ψ − R_{u₁}⁻¹L_{u₂}) (§8), not about any single numerical invariant. It has not
yielded to: the six standard mechanisms (§4), Thom–Porteous (dead bundle, §9),
miracle flatness (non-CM, §5), or the subalgebra reframe (§7, falsified here).

This is the genuine research frontier. The most valuable computational
contribution of this work is the **obstruction map (§4) plus this scoreboard**:
they tell the next attempt exactly which avenues are dead, saving the effort.

## 12. External model consult (OpenRouter, strong+exotic) — no valid mechanism

Fanned the corrected question ("prove D always has an odd-degree ℝ-rational
component"; total degree varies 6/7/9) to the strongest available models:
GPT-5.4, DeepSeek V4 Pro, Nemotron-Ultra-550B, Minimax M3, Kimi K2, Grok 4.3,
Gemini 2.5 Pro, DeepSeek R1. **None produced a valid mechanism**; each founders on
the exact deg-6 (even) counterexample:

- **GPT-5.4** (best): correctly moves to the incidence variety
  Y = {([u],[w]) : [u,w]=0} ⊂ ℝP⁴×ℝP⁴ and proposes deg₂(π₁)=1 (unique generic
  commuting partner). Two gaps: (i) "deg₂(π₁)=1" is birationality of Y→D and does
  NOT imply D(ℝ)≠∅; (ii) it admits the deg-6 case gives mod-2 total 0 and
  hand-waves "there must be another odd contribution."
- **DeepSeek V4 Pro**: novel idea — alternativity ⇒ eigenvalues of C_u=[u,·] in
  ±λ pairs + a structural zero ⇒ a fixed component. FALSE premise: [u,·] maps
  Im C(x′)→𝕊, not to itself, so it is not an endomorphism and has no eigenvalues.
- **Grok 4.3**: dimensionally-confused sketches (a curve→ℝP² "topological degree";
  a 15×5 "determinant").
- **Nemotron/Minimax/Gemini/R1/Qwen**: errored or stalled on the basic setup.

**Takeaway.** The frontier models converge on the *right object* (the incidence
variety Y and a component-level, not total-degree, invariant) but none closes it —
consistent with this being a genuine open problem. The residue worth a dedicated
effort:
1. Work on Y ⊂ ℝP⁴×ℝP⁴ (bidegree class), not on deg D.
2. The invariant must be component-level: "∃ odd-degree ℝ-component," which is
   NOT a mod-2 total (the total is even for deg-6 configs — two lines + two
   conics, 1+2+2+1). So a mod-2 homology class argument alone cannot work; the
   mechanism must produce an actual odd component (e.g. a canonical rational
   curve/line in D of odd degree), likely from a natural ℝ-rational construction
   tied to the Cayley–Dickson/Moufang structure.
3. The DeepSeek instinct (a structurally forced component), repaired to the pair
   (w₁,w₂) where [u,·] does act, may be the way in — but was not completed by any
   model.

This closes the external-consult avenue: the strongest models available (incl.
GPT-5.4, DeepSeek V4 Pro) do not solve it. The problem is genuinely open.

## 13. Comprehensive repo sweep (2026-09-01) — the exact ZD machinery, and the honest verdict

A full repo + primary-source sweep (prompted by the founder: "you didn't look at
everything") found a large body of EXACT Cayley–Dickson zero-divisor machinery I
had not connected, and three primary Zhilina papers cached from a prior session
(now preserved in `docs/research/refs/`):
- `zhilina_doubly_alternative_zd_2608.26893.txt` — centralizer structure.
- `zhilina_orthogonality_graphs_part1_2608.28176.txt`.
- `zhilina_orthogonality_graphs_part2_2608.28163.txt` — the diameter-3 technique.

**Genuinely useful assets found (repo, exact/Lean-proven):**
- **2-cycle criterion + nullity-histogram law** (routon_zd_spec, nullity_histogram_law_spec,
  cd_tower_zd_graph_invariants_spec): closed forms for `dim ker(L_a)` of canonical
  ZDs; `det(L_a)=det(A)∏_cycles(1−p(k))`, p(k)∈{±1} from CD signs; L2:
  `nullity(L_a)=nullity(R_a)` pointwise.
- **`SounioZDChi.lean` (Lean, no sorry): χ(x,y)=σ(x,y)σ(y,x) = +1 iff x=0∨y=0∨x=y,
  else −1.** ⟹ distinct nonzero basis units ALWAYS anticommute ⟹ the commutativity
  graph has **no monomial edges** ⟹ the finite-basis reduction that works on the
  orthogonality side (84 vertices, PSL(2,7)) **cannot** be used for commutativity.
- **Centralizer bridge (2608.26893 Lemma 2.19/2.20):** C(x)=F⊕Fx⊕O(x) for n(x)≠0
  (sedenion ZDs, Euclidean norm ⟹ n(x)≠0, so dim Im C(x)=5, confirming our setup).
- **Diameter-3 technique (2608.28163 Cor 4.8):** "special elements" (a component
  = e₀) act as hubs; every nonspecial element is distance-1 from a special one,
  giving diameter 3 — but this is the ORTHOGONALITY graph. GZ could NOT transpose
  it to commutativity (they get only 3≤diam≤4); that gap is exactly Conjecture 6.8.

**The honest verdict.** Conjecture 6.8 (commutativity diameter) is an **open
target** in this repo, not a solved result — the founder's own
`open_problems_scan_2026-08-31.md` lists it as open problem #1, and its proposed
reduction ("Aut(𝕆)/Khalil–Yiu transitive on ZD pairs") is an **unverified
parenthetical that the sweep shows is false**: the symmetry (G₂×S₃, PSL(2,7)=168)
is transitive on the 84 single ZD vertices, NOT on pairs, so there is **no finite
reduction**. All the exact machinery is orthogonality-side; the commutativity
witness reduces to a rank/degeneracy problem on Im C(x)×Im C(x') (≤5×5) — exactly
the determinantal locus D of §§2–12, whose degree we computed (6,7,9) and whose
odd-component existence remains the open core. The comprehensive sweep corrected a
real oversight (the exact ZD corpus) but confirmed: the solution is not sitting
pre-assembled in the repo; 6.8 is genuinely open, and the repo's own proposed
shortcut is falsified here.

## 14. A TRUE new piece (2026-09-01): odd-dimensional skew degeneracy

Prompted by the founder's remark that "original is just another combination," a
new — and, unlike §§7–13's falsified reframes, TRUE — structural lemma was
assembled from pieces already owned:

**Lemma (skew degeneracy).** For fixed u, the form B_u(w,w') := ⟨[u,w],w'⟩ =
φ(u,w,w') is **skew-symmetric** in (w,w') (Lemma 2.1: φ alternating). Restricted
to Im C(x′) × Im C(x′) it is a skew form on a **5-dimensional (odd)** space, hence
**degenerate**: it has a nonzero kernel (measured nullity = 1 for all sampled u).
So for EVERY u ∈ Im C(x) there is a nonzero w ∈ Im C(x′) with **[u,w] ⊥ Im C(x′)**.
By the symmetric statement (φ(·,w,·) skew on Im C(x)), one likewise gets, for a
compatible (u,w), **[u,w] ⊥ Im C(x)**.

**Verified (exact):** B_u is skew (‖B+Bᵀ‖ ≈ 1e-13), nullity 1 (odd) for all 200
random u; and the two-sided condition [u,w] ⊥ (Im C(x) ⊕ Im C(x′)) is achievable
(min ‖proj‖ = 0.0000).

**But it is not sufficient (measured).** At the (u,w) achieving
[u,w] ⊥ (Im C(x) ⊕ Im C(x′)), the residual ‖[u,w]‖ ≈ 1.98 ≠ 0: the component in
the **6-dimensional complement** (Im C(x)⊕Im C(x′))^⊥ survives. The true witness
has ‖[u,w]‖ ≈ 1e-14. So odd-dimensional skew degeneracy kills the 10 conditions of
the projection onto Im C(x)⊕Im C(x′) but leaves the 6-dim perp component.

**Significance.** This is the first genuinely TRUE new structural statement of the
session (the odd-dimensionality parity that matters is the **dimension of the
space, 5**, not the degree of the variety), and it gives a cleaner reduction:
> Conjecture 6.8 ⟸ for some (u,w) ∈ Im C(x)×Im C(x′), the ≤6-dimensional
> component of [u,w] in (Im C(x) ⊕ Im C(x′))^⊥ vanishes (the projection onto
> Im C(x)⊕Im C(x′) already vanishes for free, by odd-dim skew degeneracy).
This is the open remainder: kill the perp component. Whether a further parity /
dimension argument (the perp space ∩ W₁₁ has its own dimension) forces it is the
next question — the honest live lead, in the "combination" spirit.

## 15. The residual edge (2026-09-01): odd parity is one-shot; residual is even

Following the §14 skew lemma to its conclusion. Measured exactly:
- The residual space (Im C(x)⊕Im C(x'))^⊥ ∩ W₁₁ where [u,w] lands after the skew
  reduction is **4-dimensional (EVEN)** in every config (and the associator
  relations [u,w,w],[w,u,u] are automatically satisfied by [u,w], so they don't
  reduce it further). So there is **no second odd-parity** to exploit: the odd
  parity of §14 is **one-shot** — spent on the two 5-dimensional (odd) factors
  Im C(x), Im C(x') — leaving an even residual where parity forces nothing. This
  is a structural EXPLANATION of why all degree/parity arguments failed all
  session (§11): the usable odd parity is exhausted by the first reduction.
- Attempt to close via a rank-4 section over ℝP⁴ (w₄ ∈ H⁴(ℝP⁴;ℤ/2)=ℤ/2): the
  canonical section R(u) = residual of [u, w(u)] with w(u)=ker(B_u) (skew kernel,
  a degree-2 Pfaffian vector) FAILS: its real zeros number 141/142/148 across
  configs (not uniform, not clean parity), and **none of them are actual
  witnesses** — because w(u) only enforces [u,w]⊥Im C(x'), so [u,w(u)] retains
  components in Im C(x), x̃, x̃' outside the 4-dim residual; R=0 ≠ [u,w]=0.

**Net.** §14's skew lemma is a genuine TRUE reduction (kills the projection onto
Im C(x)⊕Im C(x') for free, by odd-dim degeneracy), but the residual is even and
the natural rank-4-over-ℝP⁴ closing construction does not have witnesses as its
zeros. The conjecture is not closed by this edge. The keeper is §14 (the true
lemma + the "dimension parity, not degree parity" insight) and this §15
explanation of why the parity is one-shot.
