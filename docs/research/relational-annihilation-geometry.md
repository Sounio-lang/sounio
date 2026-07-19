<!-- docs:meta
topic_id: repo.docs.research.relational-annihilation-geometry
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.relational-annihilation-geometry
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# The geometry of relational annihilation — sedenion zero-divisors as a non-reductive model of rupture between subjects

*Follow-on in the rupture-algebra program. The associator gives a measure of semantic/cognitive rupture
(`nonassociativity-as-rupture.md`, `petitot-semantic-potential.md`); the zero-divisor locus gives a measure
of **relational** rupture — annihilation. This note computes its exact geometry and states the clinical
correspondence as a falsifiable hypothesis, with the crank-landmine held in check throughout.*

## The object

A **zero-divisor pair** in the sedenions 𝕊: two elements `x, y ≠ 0` with `x·y = 0`. The subjects remain
**nonzero** — fully present — yet their **relation annihilates**. This is why the model is *non-reductive*:
it does not say a person "becomes nothing"; it locates the collapse in the **bond**, not the subject. It is
the algebraic form of a phenomenology where someone is entirely present and the connective tissue to the
world (or to a doubled self) has gone to zero.

Why sedenions specifically: in ℝ,ℂ,ℍ,𝕆 (division algebras) there is **no** annihilation — every nonzero
relation is invertible, one can always "return." Annihilation is *born* at 𝕆→𝕊 (the catastrophe of
`rupture-as-singularity.md`), and 𝕊 = 𝕆 ⊕ 𝕆 is a **doubling** — the split between an observing and a
suffering self (Baumeister's "escape from the self"); the annihilating pair lives *across* the two copies.

## The computed geometry (`box_kite.py`)

- **Counting (stated precisely — the unit matters).** Among the `2·C(15,2) = 210` two-unit-sum *elements*
  `a = e_i ± e_j`, exactly **84 are zero divisors** = the **42 assessor planes × 2 diagonals** each. (84 is
  the count of zero-divisor *elements* of this form, not of annihilating pairs — the ordered/unordered pair
  count is a different object; the census `42 / 84 / 168 = |PSL(2,7)|` fixes the vocabulary.)
- **Specificity — the load-bearing fact.** For a fixed zero-divisor subject `a = e₁+e₁₀`, its annihilating
  partners form a **dimension-4 (codimension-12) subspace of ℝ¹⁶ — measure zero.** And most subjects are not
  zero divisors at all (`{det L_x = 0}` is itself a measure-zero hypersurface). So annihilation is
  **generically impossible**: it requires exact alignment, twice over. This — not the metaphor — is what
  grounds the clinical reading: annihilation is not the limit of increasing distress; it is a
  configuration.
- **Structure:** the 42 assessor planes partition into **7 box-kites × 6 assessors** (strut constant
  `S = lo ⊕ (hi∧7)`, `S = 1..7`) — de Marrais's octahedral skeleton, reproduced from scratch; the 7 =
  the 7 Fano lines, the octonion structure surfacing beneath the sedenion annihilation.

## The clinical correspondence (hypothesis, not claim)

- **Altered states / overexcitability / creative peaks** map to the *associator* side (high non-associativity
  = path-dependent, non-context-free recombination); Dabrowski's **positive disintegration** is *crossing
  the bifurcation set* (`petitot-semantic-potential.md`), and the gifted nervous system's overexcitability
  is *living nearer the singularity* — the double edge of creativity and vulnerability.
- **Relational annihilation / suicidal behavior** maps to the *zero-divisor* side. The **falsifiable
  structural prediction:** annihilation is not caused by generic pain but requires a **specific
  configuration** — the box-kite. This is the algebraic echo of **Joiner's interpersonal theory** (suicidal
  desire as the *conjunction* of thwarted belongingness **and** perceived burdensomeness, with acquired
  capability — a specific configuration, not undifferentiated suffering). The testable content: does the
  box-kite's combinatorial specificity map onto the clinical conjunction's specificity? (Prior formal work
  exists — cusp-catastrophe models of suicidality — but none with a zero-divisor annihilation structure.)

**The falsifiable prediction (this is what converts algebra to science).** Do *not* claim `det L_x` **is**
suffering — that is analogy. Claim instead that suffering phenomena exhibit the **formal signature** of
composition failure (rare, structured, configuration-specific), and derive a test: **risk (e.g.
suicidality) should be conjunctive and low-dimensional** — interaction terms should dominate main effects,
and the risk set should occupy a small-dimensional slice of predictor space (the empirical analog of the
dimension-4-in-16 / codimension-12 annihilator).  This is testable on a large cohort (e.g. COMPASS, N≈6,543), in a **pre-registrable** form:
**(a) interaction dominance** — variance explained by order-k≥2 terms exceeds that of main effects
(functional-ANOVA / Sobol indices); **(b) low intrinsic dimension** of the high-risk set, `d ≪ p`
(two-NN or participation ratio on the high-risk subset); **(c) configurational sparsity** — the
high-risk fraction falls far below an additive model calibrated on the same marginals.
**Falsifier:** if an additive main-effects model matches or beats the interaction model out-of-sample
and `d ~ p`, the signature is absent. Do **not** anchor (b) on the literal 4/16 ratio — the algebra
motivates the qualitative form, not a numeric value in a clinical predictor space. This connects to the
discrete-curvature program (`ABIDE`/Ollivier-Ricci as fragility
geometry). If risk turns out additive and high-dimensional, the annihilation model is wrong — which is the
point: it can be wrong.

## Why the framing is, in the end, hopeful

If it is the **relation** that annihilates (not the person), and if annihilation requires a **specific
box-kite configuration** (not generic despair), then intervention is **breaking the configuration** —
restoring an invertible relation. The algebra places the collapse exactly where care can act: in the bond,
and in a specific structure that can be disrupted rather than an all-or-nothing state of the subject. That
is the point of modeling this at all — not to aestheticize suffering, but to locate where it can be moved.

## Honest scope & landmines

This is a **model**, not a clinical instrument. The clinical correspondence is a **hypothesis** — the value
is (i) the non-reductive framing (relation, not subject, annihilates), (ii) the structural prediction
(annihilation is combinatorially specific — the box-kite), (iii) the unification (one algebra for
creativity, altered states, and collapse). The dominant landmine (from the program synthesis) is at maximum
force here: octonion-consciousness has a crank-adjacent reputation, and modeling suicide algebraically can
read as aestheticizing. The only defenses are the ones taken here — foreground the **falsifiable structural
claim** (box-kite specificity ↔ clinical conjunction), ground in **real clinical theory** (Dabrowski,
Joiner, catastrophe suicidology), and keep the register **non-reductive and care-oriented**, never
mystical. Harness `box_kite.py`.
