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
prove sharp rank laws for it, all mechanically verified. We then present a
characteristic-class argument that reduces the conjecture to a technical lemma
about two explicit codimension-4 degeneracy strata, which we characterize
exactly. Every algebraic identity below has (i) a short proof, (ii) exact
integer verification on random instances, and (iii) a Lean 4 kernel-checked
canonical instance (no `sorry`, Mathlib-free).

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

A nonzero primary obstruction forces every section to vanish somewhere —
i.e., every configuration admits a witness — PROVIDED the bundle E₇ is
honest over the relevant skeleton. This is the content of:

**Technical Lemma L** [L]. Let Z = Z₁ ∪ Z₂ with Z₁ = {[x]} × ℝP⁴,
Z₂ = ℝP⁴ × {[x′]} (each ≅ ℝP⁴, codimension 4 in B). Then
(i) off Z the four constraint projections are independent in W₁₁ (measured:
2312/2312 samples over 40 configurations at rank 9 of the full 9-row stack;
every one of the 85 rank drops occurred at parallel points and dropped by
exactly one [M]);
(ii) on Z the local rank is 8 = dim B, and the section s is canonically
nonvanishing near Z whenever d(x,x′) ≥ 3 (s(x,w) = [x,w] = 0 would give
w ∈ Im C(x) ∩ Im C(x′), i.e. d ≤ 2);
(iii) [in preparation] a nowhere-zero section over B would yield a nowhere-
zero section of the honest rank-7 bundle over B ∖ N(Z) whose relative primary
obstruction class maps onto w₇ under H⁷(B, N(Z)) → H⁷(B), using (ii) and the
codimension-4 excision exact sequence.

**Corollary 6.2** (conditional on L(iii)). Every pair of zero divisors admits
a length-≤3 commuting path: **diam Γ_C^Z(𝕊) = 3**, proving the GZ conjecture.

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
