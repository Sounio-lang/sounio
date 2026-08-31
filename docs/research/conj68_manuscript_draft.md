# The commutator map on sedenion zero-divisor commutants: rank laws and a characteristic-class approach to the commutativity-diameter conjecture

**Draft v0.1 — 2026-08-31.** Working manuscript; verification artifacts in this
repository (see Appendix). Status of each claim marked: [T] theorem with proof,
[K] kernel-verified canonical instance, [M] measured (exact integer arithmetic),
[L] technical lemma stated with evidence, proof in preparation.

## 1. Introduction

Let 𝕊 denote the sedenion algebra (the 16-dimensional real Cayley–Dickson
algebra), and let Γ_C^Z(𝕊) be the commutativity graph on lines of elements
whose imaginary part is a zero divisor, as introduced by Guterman–Zhilina
[GZ26, arXiv:2608.26903]. They prove 3 ≤ diam Γ_C^Z(𝕊) ≤ 4 and conjecture
(their Conjecture 6.8, supported by floating-point experiments):

> **Conjecture (GZ).** diam Γ_C^Z(𝕊) = 3.

The conjecture is equivalent to: for every pair of zero divisors x, x′ ∈ 𝕊
with d(x,x′) ≥ 3, there exist u ∈ Im C(x), w ∈ Im C(x′), u,w ≠ 0, with
[u,w] = 0. Here Im C(x) = ℝx ⊕ O(x) (dimension 5; O(x) is Moreno's
4-dimensional orthogonalizer).

We study the bilinear commutator map T(u,w) = [u,w] on Im C(x) × Im C(x′) and
prove sharp rank laws for it, all mechanically verified. We then examine — and
**close as a negative result** — a characteristic-class approach to existence: the
relevant relative Euler class is computed and shown to vanish (§6), so witness
existence is not topologically forced by this bundle and must be established
algebraically (§8). Every algebraic identity below has (i) a short proof, (ii)
exact integer verification on random instances, and (iii) a Lean 4 kernel-checked
canonical instance (no `sorry`, Mathlib-free). The rank laws (§§2–5) stand on
their own as the paper's positive contribution.

Throughout: x = (a,b), x′ = (a′,b′) normalized zero divisors (n(a) = n(b) = 1
etc.), z = ab, z′ = a′b′ their Γ_O-invariants, x̃ = (b,−a) the hexagon
companion (GZ Lemma 3.6), φ(u,w,v) := ⟨[u,w], v⟩.

## 2. The trilinear form is alternating

**Lemma 2.1** [T][K]. φ is fully alternating on pure sedenions: it is
antisymmetric in (u,w) by definition and cyclic by the Cayley–Dickson
inner-product identities ⟨p, qr⟩ = ⟨pr̄, q⟩ = ⟨q̄p, r⟩; an antisymmetric
cyclic trilinear form is alternating. In particular
**T(u,w) ⊥ u and T(u,w) ⊥ w** for all pure u, w. ∎
*(Lean: `p12_polarized`, full polarization over the pure basis, 15³ instances.)*

**Remark 2.2 (a vacuity trap).** For pure u,w one has wu = conj(uw), hence
u∘w = uw + wu = 2Re(uw)·e₀ is always real: the "relation" ⟨T, u∘w⟩ = 0 is
vacuous. We record this because we fell into it (attack log, Rodada 10);
every pointwise relation below is accompanied by a non-vacuity certificate.

## 3. Linear relations: the rank-11 theorem

**Theorem 3.1** [T][K]. im T ⊥ span{x, x′, x̃, x̃′}, hence rank T ≤ 11
(measured: = 11 on all 143 sampled configurations [M]).
*Proof.* For x, x′: φ(u,w,x) = φ(x,u,w) = ⟨[x,u],w⟩ = 0 since u ∈ C(x).
For x̃: the key lemma [x̃, u] ∈ ℝ·(0,1) for every u ∈ Im C(x) (from the
double-hexagon multiplication table: [x̃,x] = 4(0,1) and x̃ commutes with the
four generators of O(x)), plus the fact that every element of Im C(x′) is
doubly pure, hence ⊥ (0,1). ∎
*(Lean: `complement_orthogonal_canonical`, `lemmaB_canonical`,
`frames_doubly_pure`, `complement_independent`.)*

## 4. The sector ghost law: the rank-9 theorem

On O(x) × O(x′) ("sector"), with frames u = (c, cz), c ⊥ {1,a,b,z} and
w = (g, gz′), g ⊥ {1,a′,b′,z′}:

**Theorem 4.1** [T][K]. im T|_{O×O} ⊥ (n(A′)z + n(A)z′, 0) and
(0, n(A′)z + n(A)z′) — the norm-normalized sum of invariants. Hence
rank T|_{O×O} ≤ 9 (measured: = 9 generically, with strata {2,3,7} [M]).
*Proof.* Ghost 1: φ(u,w,(z+z′,0)) = ⟨[(z+z′,0),u],w⟩ by cyclicity;
[(z,0),u] = (−2cz,−2c) and [(z′,0),u] = ([z′,c], 2(cz)z′); right-multiplication
by the unit z′ is an isometry, so ⟨(cz)z′, gz′⟩ = ⟨cz,g⟩ cancels the cross
term; the remainder is ⟨cz′+z′c, g⟩ = −2⟨c,z′⟩Re(g) = 0 by purity of g.
Ghost 2: analogous, ending in −4⟨c,z′⟩⟨z′,g⟩ = 0 since g ⊥ z′. ∎
*(Lean: `ghost_orthogonal_canonical`, `ghost_normalized_n2` — the latter with
n(A) = 1, n(A′) = 2 pinning the coefficients. Discovery pipeline: algebraic
dictionary → mod-p kernel → rational reconstruction → exact ℤ verification,
25/25 configurations, `VERIFY_FAIL = 0`.)*

**Proposition 4.2** [T][M]. The orthogonalizer of a zero divisor (A,B) admits
the closed form O((A,B)) = {(n(A)c, −(Bc)A/n(A)-normalized) : c ⊥ {1,A,B,AB}};
equivalently the partner of c is d(c) = −(Bc)A/n(A) (derived from the pure
component equations ac = −db, da = bc; verified by bilateral annihilation on
all sampled configurations).

## 5. The associator relations: the pointwise rank-7 structure

**Theorem 5.1** [T][K][M]. For all pure u, w:
⟨[u,w], [u,w,w]⟩ = 0 and ⟨[u,w], [w,u,u]⟩ = 0,
where [p,q,r] = (pq)r − p(qr) is the associator.
*Proof.* By Lemma 2.1 and w² = −n(w), ⟨[u,w],[u,w,w]⟩ = ⟨[u,w], (uw)w⟩.
The term ⟨uw, (uw)w⟩ vanishes by the self-negating move
⟨uw,(uw)w⟩ = ⟨(uw)w̄, uw⟩ = −⟨(uw)w, uw⟩. For the other term put v = uw, so
wu = conj v and ⟨wu, (uw)w⟩ = −⟨v̄w, v⟩ = −⟨v², w⟩ = −2Re(v)⟨uw,w⟩ + n(v)⟨1,w⟩,
and ⟨uw, w⟩ = n(w)⟨u,1⟩ = 0, ⟨1,w⟩ = 0. Symmetrically for [w,u,u]. ∎
*Non-vacuity* [M]: rank{u, w, [u,w,w], [w,u,u]} = 4 on 2000/2000 exact
integer samples.
*(Lean: `p3_bilinear` for the conj identity; probe:
`conj68_pointwise_relations_probe.sio`.)*

## 6. The obstruction argument

Let B = ℝP⁴ × ℝP⁴ = P(Im C(x)) × P(Im C(x′)), γ₁, γ₂ the tautological line
bundles, α, β the generators of H*(B; ℤ/2) = ℤ/2[α,β]/(α⁵,β⁵). Let
W₁₁ = Im𝕊 ∩ {x,x̃,x′,x̃′}^⊥ (Theorem 3.1). The section s([u],[w]) = T(u,w)
takes values in W₁₁ and satisfies the four pointwise orthogonality relations
of Lemma 2.1 and Theorem 5.1, whose right-hand vectors have parities
(γ₁, γ₂, γ₁, γ₂) in (u,w).

Off the degeneracy locus Z (below), the four projected constraint vectors are
independent in W₁₁, so s is a section of a rank-7 bundle
E₇ ≅ ℝ¹¹ ⊖ 2γ₁ ⊖ 2γ₂ twisted by γ₁γ₂. Since rank E₇ = 7 < 8 = dim B, the
primary obstruction to a nowhere-zero section is its twisted Euler class in
H⁷; its mod-2 reduction is

  w₇(E₇ ⊗ γ₁γ₂) = Σᵢ wᵢ(E₇)(α+β)^{7−i},  w(E₇) = (1+α²+α⁴)(1+β²+β⁴),

**Theorem 6.1** [T][K]. w₇(E₇ ⊗ γ₁γ₂) = α⁴β³ + α³β⁴ ≠ 0.
*(Hand computation; Sounio exact polynomial arithmetic
`conj68_euler_class.sio`; Lean `euler7_primary_obstruction_nonzero`.)*

A nonzero primary obstruction would force every section to vanish somewhere,
PROVIDED E₇ were an honest rank-7 bundle over all of B. **It is not, and this
matters:** E₇ lives only over B ∖ Z, and there the obstruction vanishes.
Indeed, since Z₁, Z₂ have codimension 4 and Z₁∩Z₂ is a single point (codim 8,
invisible to H⁷), the Thom computation of Prop 6.1b gives
im(H⁷(B, B∖Z) → H⁷(B)) = span{α⁴β³, α³β⁴} = **all** of H⁷(B; ℤ/2) (which has
exactly these two monomials). By exactness i*: H⁷(B) → H⁷(B∖Z) is therefore the
zero map, so **w₇|_{B∖Z} = 0**. The primary obstruction of the honest bundle
E₇ over the *open* manifold B ∖ Z is zero — as it must be, top-degree
obstructions over open manifolds comb out to the ends. Consequently Theorem 6.1
is **not, by itself, a proof**: w₇ ≠ 0 is a necessary consistency check on the
relative obstruction, not a derivation of it. The actual content is:

**The correct forcing invariant, and its vanishing.** The existence of a witness
is *not* controlled by the absolute obstruction over the open manifold B ∖ Z.
Let M = B ∖ N̊(Z), a compact 8-manifold with boundary ∂M = ∂N(Z); by (F3-type)
Lemma L(ii) below, s is nowhere zero on N(Z), hence gives a nowhere-zero
boundary section s|_{∂M}. A witness is a zero of s in the interior of M, and the
obstruction that *forces* one is the **relative Euler class**
  e_rel(E₇ ⊗ γ₁γ₂, s|_{∂M}) ∈ H⁷(M, ∂M; ℤ̃) ≅ H₁(M; ℤ̃),
(Lefschetz duality). Removing the codimension-4 locus Z from the 8-manifold B
leaves π₁ unchanged, so mod 2 this group is H₁(M; ℤ/2) ≅ H₁(B; ℤ/2) = (ℤ/2)²,
and it is exactly the class the census reads: **e_rel ≡ [Γ] = 0** (measured,
R14). The invariant that would force a witness is measured to vanish.

**Theorem 6.2 (the characteristic-class route is closed) [M, T].** For the
commutator section s on B = ℝP⁴×ℝP⁴, the relative primary obstruction to a
global nowhere-zero section vanishes: e_rel = [Γ] = 0. Two independent reasons:
1. **No local obstruction at Z, on dimension grounds.** Extending a nowhere-zero
   section across a codimension-k locus whose fibre sphere is S^{r−1} has first
   obstruction in π_{k−1}(S^{r−1}), which is nonzero only for k ≥ r. Here k = 4
   (codim Z_i) and r = 7 (rank E₇): since 4 < 7 the linking 3-sphere maps into
   S⁶ (or S⁷ on Z) and π₃(S⁶) = π₃(S⁷) = 0. **There is no Z-supported obstruction
   for any section whatsoever** — a fortiori none for s.
2. **The absolute class w₇ carries no information about s.** H⁷(B; ℤ/2) is
   2-dimensional with basis {α⁴β³, α³β⁴}, and the two Thom pushforwards of the
   normal bundles of Z₁, Z₂ (mod-2 Thom classes α⁴, β⁴) are exactly that basis.
   Hence im(H⁷(B, B∖Z) → H⁷(B)) = H⁷(B) *automatically*, and **every** degree-7
   class "appears supported on Z". The equality w₇ = α⁴β³ + α³β⁴ ∈ im(j*) is a
   tautology of low-dimensional cohomology, not a geometric localization of the
   obstruction of s. (This corrects the reading of R13, which had kept this route
   looking alive.)

*Consequence.* Theorem 6.1 (w₇ ≠ 0 for the *virtual* bundle over B) is a
consistency check with no forcing power: over B ∖ Z, where the honest E₇ lives,
w₇ restricts to 0, and the relative Euler class that would count interior zeros
is [Γ] = 0. **The former Corollary 6.2 (diam = 3 via this obstruction) is
withdrawn.** Witness existence is therefore an *algebraic*, not a topological,
fact — see §8.

*Scope, honestly.* (i) The claim is that **this** route, as constructed, is
dead — not that no topological proof exists. (ii) e_rel = [Γ] presumes the census
enumerated all components of Γ; the tracer is handle-limited (~5 components per
configuration, R17b) so [Γ] = 0 is measured, not proven. This does not revive
forcing: unfound components would have to carry an odd class while the found ones
already sum to zero, for which there is no evidence.

**Lemma L(ii) [T].** On Z the section is canonically nonvanishing whenever
d(x,x′) ≥ 3: s(x,w) = [x,w] = 0 would give w ∈ Im C(x) ∩ Im C(x′), i.e. d ≤ 2.
Off Z the four constraint projections are independent in W₁₁ (2312/2312 samples,
rank 9; all 85 rank drops at parallel points, each by exactly one [M]); on Z the
local rank is 8 = dim B. (This is the boundary datum for e_rel above.)

## 7. Supporting evidence and negative results

- 175+ configurations swept with strong exact/numerical hunts: witnesses in
  every one; on the 84 basis configurations all witnesses are ±1-integer
  combinations of hexagon frames (support ≤ 2) [M].
- Sector-only witnesses do NOT exist for generic configurations (floors
  0.05–0.9 under 120-restart hunts); orthogonal common-invariant witnesses
  (case-(i)) also fail generically: the witness variety is genuinely mixed —
  consistent with the obstruction argument, which produces zeros of the full
  section without locating them [M].
- The complex structure R (right multiplication by (0,1); GZ Lemma 3.6)
  preserves every O(x) (143/143) but does not intertwine T: four candidate
  equivariance identities fail outside 6 degenerate configurations [M].
- **The linear pencil is a symmetry artifact, not a general construction** [M].
  At the maximally symmetric config (e₁,e₂) (a, b, ab = e₁, e₂, e₃ all
  basis-aligned) the odd components of the (4,4) witness curve are two (1,1)
  projective lines — a 1-parameter *pencil* of witnesses in closed integer form
  (verified exactly in ℤ). But a genericity sweep (8 configurations, both zero
  divisors sampled, one per process; tracer validated on the (e₁,e₂) config as a
  positive control reproducing 2 lines + 1 conic) found **0 line components
  (KRANK=MRANK=2) and 20 conic components (KRANK=MRANK=3)** (raised the witness cap 40→300 and confirmed the tracer still resolves the (e₁,e₂) lines under the raised cap — a sensitivity check, so the null is not a truncation artifact): generically
  the odd components are conics, and the pencil is the split-conic degeneration
  at the symmetric point. Consequently a witness-existence proof cannot rest on
  a uniform linear pencil. It rests instead on the **odd degree** of the whole
  witness curve (§8): the pencil/conic/cubic split varies with the configuration,
  but the total degree 7 does not.

## 8. The odd-degree existence argument

With the topological forcing closed (§6), existence is exhibited by an
enumerative parity — but of the witness *variety's degree*, not of an Euler
class. This is the route the data support, and it is close to complete.

Fix a pair (x, x′). In frame coordinates for Im C(x), Im C(x′) the commutator
gives a matrix M(u) (rows = the ≤ 11 nonzero components of [u, ·] in W₁₁,
columns indexed by the frame of Im C(x′)), with entries **linear in u**. A
nonzero w ∈ Im C(x′) with [u, w] = 0 is exactly a kernel vector of M(u), so the
witness locus is the determinantal variety

  D(x,x′) = { [u] ∈ ℙ(Im C(x)) = ℝP⁴ : rank M(u) ≤ 4 }.

**Measured, exactly (Singular, primary decomposition over ℚ; ~17 configurations,
including the special stratum a = 2e₃ − 2e₄):**

**Fact 8.1 [M].** D(x,x′) is a curve (projective dimension 1) of **degree 7**.
Its reduced structure has degree 7 as well (the scheme's non-reduced part is
supported on lower-dimensional embedded components — a conjugate pair of complex
points and the origin — which do not touch the curve; deg D = deg √D = 7). The
curve splits, over ℚ, as two conics + one **cubic** for generic pairs
(2 + 2 + 3 = 7), or one **line** + three conics for flatter pairs (1 + 2 + 2 + 2
= 7): in every case an **odd-degree** component is present, forced by the odd
total degree.

**Theorem 8.2 (existence, conditional on uniformity of Fact 8.1).** deg D = 7 is
odd. A reduced real projective curve of odd degree has a real point: a generic
real hyperplane H (defined over ℝ) meets D in a conjugation-stable 0-cycle of
degree 7, whose complex points pair off, leaving an odd — hence positive —
number of real points. A real [u] ∈ D is a real line in Im C(x); M(u) is a real
matrix of rank ≤ 4, so ker M(u) contains a real nonzero w ∈ Im C(x′). Then
(u, w) is a witness with u, w ≠ 0 and [u, w] = 0. Hence every pair admits a
witness, and **diam Γ_C^Z(𝕊) = 3**.

*(Note: the argument never uses d(x,x′) ≥ 3 — it produces a witness for every
pair, which for d ≤ 2 is unsurprising. The hypothesis enters only in reading the
witness as a length-3 path.)*

**What remains for a proof.** Two points, both about making Fact 8.1 uniform:
1. **deg D = 7 for every admissible pair.** The right argument is *not* a
   Thom–Porteous class in the rank-7 bundle E₇ — that bundle is exactly the one
   §6 shows fails to exist over Z. Instead: over the connected space of
   admissible pairs (connectivity should follow from the transitivity results of
   [GZ26]), the family {D(x,x′)} is flat wherever dim D = 1, so deg D is locally
   constant, hence ≡ 7 from the measured value. The genuine gap is the **bad
   locus** where dim D jumps to 2 (there the degree says nothing): we have not
   found such a pair (dim D = 1 on every sampled configuration, mod-p rank test),
   but ruling it out is open.
2. **The curve is reduced of odd degree.** Measured (deg D = deg √D = 7) on the
   sampled configurations; the general statement is needed for the hyperplane
   section in Theorem 8.2.

This is the enumerative parity R14 sensed as "c₇ odd" — but attached to the
correct object. The degree of the witness curve, an intersection number, is a
deformation invariant; its being odd forces a real point. The Euler class of §6
was the wrong odd invariant (it lived on B ∖ Z where it vanishes); the *degree*
is the right one.

## Appendix: verification artifacts (this repository)

| Claim | Sounio (exact) | Lean 4 (kernel) |
|---|---|---|
| Basis sweep 85/85, D3EXACT=43 | tests/run-pass/sedenion_conj68_basis_probe.sio | — |
| Rank-11 relations | examples/research/conj68_rank_structure.sio | SounioConj68RankBound.lean |
| Ghost law 25/25 + 143/143 | conj68_dictionary_kernel.sio, conj68_rank_structure.sio | ghost_* theorems |
| Alternating φ + associators | conj68_pointwise_relations_probe.sio | p12_polarized, p3_bilinear |
| w₇ = α⁴β³+α³β⁴ | conj68_euler_class.sio | euler7_primary_obstruction_nonzero |
| Locus Z = Z₁∪Z₂, drop-by-1 | conj68_loci_probe.sio | — |
| Negative results (R5, R8) | conj68_hidden_identity_probe.sio, conj68_sylvester_scan.sio | — |

Full attack log with all retractions: docs/research/conj68_attack_log_2026-08-31.md.

[GZ26] A. Guterman, S. Zhilina, *Relation graphs of the sedenion algebra*,
arXiv:2608.26903 (2026). And the companion series arXiv:2608.28176,
2608.28163, 2608.26893, 2608.26890.
