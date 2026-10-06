<!-- docs:meta
topic_id: repo.docs.research.cd-tower-bdi-dugger-isaksen-gate-2026-08-07
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.cd-tower-bdi-dugger-isaksen-gate-2026-08-07
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Literature gate: Biss–Dugger–Isaksen and the Dugger et al. eigentheory

**Date:** 2026-08-07. **Trigger:** §58.4 of `cd_tower_zd_fiber_v1_reduction_spec_2026-07-31.md` — the
Fable-5 strategy review named three papers that were not on my reading list at all and made clearing
them a precondition for any novelty claim. This is that clearance. Papers read directly (full text
via arXiv), not from abstracts.

- **[DDD]** Biss, Dugger, Isaksen, *Large annihilators in Cayley–Dickson algebras* (math/0511691).
- **[DDDD]** Biss, Christensen, Dugger, Isaksen, *Large annihilators … II* (math/0702075).
- **[Eig]** Biss, Christensen, Dugger, Isaksen, *Eigentheory of Cayley–Dickson algebras*
  (arXiv:0905.2987).

---

## Verdict

**The gate is CLEAR for the spectral-completeness headline, with one residual risk that is now named
precisely instead of vaguely.** None of the three papers defines a graph, an adjacency matrix, or a
graph spectrum; none uses a binary-index/XOR condition; all three work with continuous elements of
`A_n` and real-linear operators on the `2^n`-dimensional algebra.

And there is a positive finding that outweighs the relief — see §4.

---

## 1. What [DDD] actually proves

Annihilator **dimensions** of general elements. `dim Ann(x) ≤ 2ⁿ − 4n + 4`; every multiple of 4 in
range occurs (Thm 1.2); `ZD_{2ⁿ−4n+4}(A_n)` is `2^{n−4}` disjoint copies of the Stiefel variety
`V₂(ℝ⁷)` (Thm 1.3). The technique is the `C_n`-vector-space structure: `A_n` is a vector space over
`C_n = ⟨1, i_n⟩`, `L_x` is conjugate-linear and anti-Hermitian for `x ⟂ C_n`, whence dimensions are
multiples of 4.

Their own scope statement: *"Although a complete description of zero-divisors seems to be out of
reach…"* and *"These strata are … unknown even in the case of `A₅`."*

**No basis-index combinatorics.** Standard basis vectors appear only inside existence arguments
(Prop 13.1). **No graph, no spectrum, no XOR.**

## 2. What [DDDD] actually proves — and what "the splitting" is

The abstract's *"a certain splitting that simplifies computations surprisingly"* is:

    every element of H⊥_{n+1} is uniquely {a,b} := (1/√2)(a, −i_n a) + (1/√2)(b, i_n b),  a,b ∈ C⊥_n

i.e. the **±i_n eigen-splitting of one doubling step**, with the multiplication rule
`{a,b}{x,y} = √2{ax, by}` under `C`-orthogonality (Prop 1.2/4.1). Main results: `dim Ann{a,b} =
dim Ann a + dim Ann b`, plus 4 exactly on the **D-locus** (Thms 1.3/1.5); and a stability theorem
`T^c_n` (Thm 8.12).

**Is that our hi/lo?** No. Theirs is a real-linear decomposition of a subspace of the algebra into
two copies of `C⊥_n`; ours is an involution on binary **index labels** (`x ↦ x ⊕ W`, plus the
`+2^{n+1}` seam shift). Both descend from the same doubling, which is why they rhyme, but they are
objects of different type — theirs has no index set in it at all.

**⚠ RESIDUAL RISK, now named precisely.** [DDDD]'s zero-divisor criterion in the splitting is:
*"if `a` and `b` are `C`-orthogonal, then `{a,b}` is always a zero-divisor."* `C`-orthogonality is
the vanishing of the projection of `ab*` onto `C_n`. **Restricted to standard basis elements this
becomes a condition on indices** — `e_i e_j*` is `± e_k`, so it lies in `C_n` exactly when `k` is
`0` or the index of `i_n`. So a basis-level shadow of their criterion exists, and our
`seam_coincidence` lives in that register. This does not make the two the same statement — ours is
about the seam predicate on a fiber of assessors, and covers `isZD`, `hasXorAnnih` and `anti0`
together on the full distinct-nonzero box — but **the honest position is that `seam_coincidence`'s
mathematical content may be a basis-level specialisation of [DDDD] §5, and the paper must either
prove it is not or cite it as such.** The ∀n anchor-free Lean proof is ours regardless; the
*content* is what is at issue. This is the one item the gate does not close.

## 3. What [Eig] actually proves — and why it is not our spectrum

The eigentheory is of `M_a = L_{a*}L_a / |a|²`, a **positive semidefinite real operator on the
`2^n`-dimensional algebra**. Results: `M_a` diagonalisable with non-negative eigenvalues (3.9);
`ker M_a = ker L_a`, so **`a` is a zero-divisor iff `0` is an eigenvalue** (3.8); eigenvalues sum to
`2^n` (3.17); all lie in `[0, 2^{n−3}]` (4.8); every eigenspace has real dimension a multiple of 4
(3.20); and Thm 8.2 — a top-dimensional zero-divisor has eigenvalues `0` and `2^k` (`0 ≤ k ≤ n−3`)
with multiplicities `2ⁿ−4n+4`, `8`, and `4`.

**Our `A_σ` is not `L_a` and not `M_a`.** Theirs acts on the algebra; ours is the adjacency matrix of
a graph on the discrete fiber, with `±1` entries given by products of the CD sign cocycle. The
sizes, the index sets and the entries all differ, and neither paper contains a construction that
would produce ours. **No graph, no adjacency matrix, no XOR condition, and no isomorphism
classification appears in any of the three papers.**

## 4. The positive finding: their §9 is our framing

[Eig] closes with open questions. Two of them:

> **Question 9.3.** Fix `n`. Describe the space of all possible spectra of elements in `A_n`.
> *"Results such as Theorem 8.2 suggest that the answer is complicated. We don't even have a guess."*
>
> **Question 9.4.** Fix `n`. Describe the space of all possible spectra of zero-divisors in `A_n`.

So *"describe the possible spectra attached to zero-divisors in `A_n`"* is an **explicitly stated
open question by this school**, unanswered since 2009, in a register adjacent to ours. That is the
citation that frames the completeness theorem — it establishes the question as one the field asks.

**⚠ We do NOT answer 9.3 or 9.4 and must never say we do.** Their spectrum is `M_a`'s on the algebra;
ours is a graph's on the fiber. The correct sentence is that our result is a *discrete analogue* in
the combinatorial register, and that the relation between the two spectra is itself open.

## 5. Actions

1. Cite [DDD], [DDDD], [Eig] in the paper's related-work section, with §3's distinction stated
   explicitly and §4's Questions 9.3/9.4 quoted as the framing — plus the disclaimer that we do not
   answer them.
2. **Settle the `seam_coincidence` question of §2** before claiming it as content: either derive it
   from [DDDD] §5 (and then present it as a formalisation with a new proof), or exhibit a pair the
   `C`-orthogonality criterion does not decide.
3. Still owed, from §58.4 and not covered here: **de Marrais** (box-kites — highest rediscovery risk
   for the fibration), **Zhilina** (sedenion relation graph vs our `n = 4` annihilation graph), the
   van Dam–Haemers DS survey (framing), and the two-graph catalogues.
