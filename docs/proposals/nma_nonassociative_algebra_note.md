<!-- docs:meta
topic_id: repo.docs.proposals.nma-nonassociative-algebra-note
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.proposals.nma-nonassociative-algebra-note
-->

# Algebraic Structure of NMA Consistency — Note

**Status:** Working note (for `epistemic-meta-analysis.md` Wave 2.6 spin-out)
**Date:** 2026-04-23
**Question:** Does the algebraic object formed by NMA treatment comparisons actually justify an octonion-associator framing, or is it a smaller structure?

## 1. What NMA consistency actually is, algebraically

Fix a set of treatments T = {t₁, …, tₙ} and let cᵢⱼ denote the comparison effect between tᵢ and tⱼ (e.g. log-OR, log-RR, mean difference).

Two assumptions give NMA its structure:

- **Reversal:** cⱼᵢ = −cᵢⱼ (additive scale) or cⱼᵢ = 1/cᵢⱼ (multiplicative scale).
- **Transitivity / consistency:** for any triple (i, j, k), cᵢⱼ + cⱼₖ = cᵢₖ (additive) or cᵢⱼ · cⱼₖ = cᵢₖ (multiplicative).

**Additive formulation.** The set of pairwise comparisons that satisfy consistency is exactly the set of coboundaries of the 0-cochain "treatment effect" on the complete graph Kₙ. The consistency condition is the cocycle condition for the 1-cocycle (c_ij)_{i<j} on Kₙ with values in ℝ (or in ℝᵏ for multivariate outcomes). Inconsistency is exactly a nonzero H¹ class — the cocycle fails to be a coboundary.

Concretely, the inconsistency of a closed loop i → j → k → i is the sum

  Δ(i,j,k) = cᵢⱼ + cⱼₖ + cₖᵢ.

Under consistency, Δ = 0 for every triangle. Under inconsistency, Δ ≠ 0. Node-splitting and design-by-treatment interaction tests are statistics on this quantity.

**This is an abelian-group / group-cohomology object. It is associative. There is no intrinsic associator.**

**Multiplicative formulation.** If effects are ratios and composition is multiplication, the consistent comparisons form a commutative groupoid: the treatment set is the object set, each comparison is an invertible morphism, and consistency is composition associativity plus reversal giving inverses. Again associative.

## 2. Where does non-associativity have a chance to enter?

Three routes, each with a prerequisite:

### Route A — Multivariate effects with non-commutative composition

If each comparison cᵢⱼ is a vector in ℝᵏ (correlated outcomes per comparison — e.g. efficacy + safety jointly), and composition of comparisons is modelled by a non-commutative operation (rotation composition, quaternion-like product), then closed loops produce a non-trivial commutator [cᵢⱼ, cⱼₖ].

- **Prerequisite:** a principled reason why comparison composition is non-commutative. Existing multivariate NMA uses vector addition (associative + commutative) — any departure needs physical justification.
- **Target algebra:** quaternions (ℍ, dim 4) are the natural non-commutative normed division algebra. Octonions only enter if we also need non-associativity, which requires Route B as well.

### Route B — Sequential / path-dependent effects

If treatments have order-dependent effects (A-then-B ≠ B-then-A — tolerance, priming, order effects in crossover designs), comparison composition is non-commutative; if there is path memory (effect of A-then-B-then-C depends on whether you "collapsed" A-then-B first), composition is non-associative.

- **Prerequisite:** crossover or sequential-therapy NMA, not standard parallel-arm NMA.
- **Target algebra:** free non-associative magma over the treatment set, potentially quotiented by empirically-supported identities. Octonions arise only if the treatment set embeds into the 𝕆 Fano-plane multiplication table — extremely unlikely to be data-driven.

### Route C — Hypercomplex encoding as imposed structure

Treat each of n ≤ 7 treatments as an imaginary octonion basis element and interpret comparisons as octonion products. This is the "just embed it in 𝕆" move.

- **Status:** This is an imposed structure, not one arising from the data. The embedding is a modelling choice that imports octonion non-associativity onto a problem that does not natively require it.
- **Tested empirically:** may still be useful as a *detector* — if the imposed octonion associator correlates with inconsistency statistics, the embedding has practical value even without intrinsic motivation. But this is methodologically the weakest of the three routes.

## 3. Does octonion dimension 8 specifically get us anything?

**Hurwitz theorem:** the only normed division algebras over ℝ are ℝ, ℂ, ℍ, 𝕆 (dim 1, 2, 4, 8). Each step in the Cayley-Dickson chain doubles dimension and loses one property: commutativity at ℍ, associativity at 𝕆, alternativity / division at 𝕊 (sedenions).

**For NMA:** there is no natural treatment-count that maps to 8. Four treatments yield 6 comparisons (dim 6); five yield 10; six yield 15. Neither 6 nor any other small NMA dimension aligns with 𝕆.

**Where 𝕆's dimension is motivated elsewhere in the research program:** 7-sphere, G₂ automorphism group, triality for brain connectome work. These are genuine structural arguments. They do not transfer to NMA.

**Verdict:** 𝕆 is not the natural algebra for NMA consistency. Grok-code's critique stands.

## 4. Honest paths forward

**(P1) Keep 𝕆 for brain/connectomics work; don't claim it for NMA.** The octonion thread in the broader research program has legitimate structural motivation (G₂, triality, 7-sphere, non-associative graph labels on the connectome). NMA does not share that motivation. The two should be decoupled.

**(P2) For NMA, use the algebra the problem actually has.** Primary claim: inconsistency is a group-cohomology defect (H¹ of the treatment graph). Secondary claim: in multivariate or sequential NMA, quaternion-valued or free-non-associative composition may expose inconsistency structures that the standard scalar cocycle misses. Let the simulation decide which.

**(P3) Reframe the Wave 2.6 paper.** Working title shifts from *"Non-Associative Consistency in NMA"* (octonion-branded) to something like *"Cohomological Inconsistency in Network Meta-Analysis"* (additive/multivariate) or *"Path-Dependent Consistency in Sequential NMA"* (non-associative, quaternion or magma). Both are defensible. Neither requires 𝕆.

**(P4) Validation sim is still gatekeeper.** Whichever algebraic frame wins, the Python/NumPy prototype must show that the chosen detector flags inconsistency cases that node-splitting / design-by-treatment miss. If it doesn't, drop the whole spin-out regardless of algebra choice.

## 5. Immediate action

- Remove "octonion" and "𝕆" language from `docs/proposals/epistemic-meta-analysis.md` main body.
- Retain "non-associative consistency detector" as a method placeholder, with this note cited as the algebraic justification to be completed before paper submission.
- Decouple NMA work stream from the connectomics/G₂ program explicitly in both plans.

## 6. Open algebraic questions

- **Q-alg-1.** What is H¹(Kₙ, ℝᵏ) for multivariate NMA, and is there a natural multivariate inconsistency statistic generalizing the scalar cocycle defect?
- **Q-alg-2.** Does crossover / sequential-therapy NMA data exist in sufficient density to power a Route-B study, or is this a theory-only paper?
- **Q-alg-3.** Is there a published statistics of associators on groupoids (measure-valued associator distributions) that the Route-A/B work could build on?
- **Q-alg-4.** Is the Bucher indirect-comparison statistic a Route-0 (additive H¹) phenomenon in disguise? If so, there's an existing bridge to cohomological framing already in the NMA literature worth citing.
