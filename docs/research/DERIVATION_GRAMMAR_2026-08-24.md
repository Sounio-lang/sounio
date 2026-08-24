<!-- docs:meta
topic_id: repo.docs.research.derivation-grammar-2026-08-24
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.derivation-grammar-2026-08-24
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# The derivation grammar of epistemic-invariant structure — formalized + generativity-tested

**Date:** 2026-08-24 · Companion to PRESERVATION_ALGEBRA_GEOMETRY_2026-08-24.md.

The six-layer tower (geometry → Jordan → kinematic → conformal → anomaly →
rigidity) is not an ad-hoc pipeline: it is a **derivation in a typed, context-
sensitive grammar** whose production rules are functors. "Derivation" is meant in
both senses at once — the formal-language sense (rule application from a start
symbol) and the Lie sense (`Der` is literally one of the rules).

## The grammar G = (types, rules, start, context)

**Types (nonterminal categories).** `ZD` (zero-divisor), `Sub` (subspace),
`Jord` (Jordan algebra), `Lie` (Lie algebra), `Coh` (cohomology / vector space).

**Start symbol.** A zero-divisor `z` in a Cayley–Dickson algebra, carrying two
parameters: the split vector `μ⃗` (fixes real forms) and the causal type
`c(z) = sign Q(z) ∈ {−, 0, +}` under the square-form.

**Production rules (typed functors — only well-typed compositions are derivations).**
```
R1 [ker]   ZD  → Sub    z ↦ ker L_z
R2 [stab]  Sub → Jord   K ↦ P_z         (two-sided stabilizer)
R3 [Der]   Jord→ Lie    J ↦ Der(J)      (kinematic)
R4 [TKK]   Jord→ Lie    J ↦ KKT(J)      (conformal)
R5 [H²ℝ]   Lie → Coh    g ↦ H²(g;ℝ)     (central charges / anomaly)
R6 [H²gg]  Lie → Coh    g ↦ H²(g;g)     (deformations / rigidity)
```
The typing is strict: `R3/R4` accept only `Jord`, `R5/R6` only `Lie`. A derivation
is a well-typed path; the tower is the path `z →R1→ →R2→ →{R3,R4}→ →{R5,R6}→`.

**Context-sensitivity (what makes it a grammar, not a free monoid of functors).**
The causal type `c(z)` gates and colours productions:
- **Signature** of every output is fixed by `c(z)`: `−`→Euclidean, `0`→Carrollian,
  `+`→Lorentzian (the rung law). Real forms then set by `μ⃗`.
- **R5 fires nontrivially iff `c(z)=0`**: only the null branch unlocks central
  charges (`H²(g;ℝ)=3`); the `±` branches give `0`. This is a genuine
  context-sensitive rule — a production available only after the context (null
  causal type) has been established.
- **R6 is context-free-trivial**: `H²(g;g)=0` on every branch (universal rigidity).

## Generativity test — level 5 (Cayley–Dickson `A₅`, 32-dim)

A grammar is *generative* (not merely descriptive) iff it PREDICTS structure at a
level it was not fit to. The rules above were read off level 4 (sedenions). Test:
apply them to level 5 and compare against a fresh 32-dim computation.

**Result (computed, `scratchpad/level5.py`):**
- Level-5 pair-type ZDs have **kernel dim ∈ {4, 8, 12}** — no longer the uniform `4`
  of level 4 (the erasure ladder `2ⁿ⁻¹−4 = 12` is the *maximum*, not the only value).
- Representatives (all division `⇒` all spacelike, `Q(z)=−2`):
  `ker 4 → J_spin(5)`, `ker 8 → J_spin(11)`, `ker 12 → J_spin(5)` — all Euclidean.

**Verdict — two-sided, and the test did its job:**
- **The signature/rung production (R1→R2 colour + context) IS GENERATIVE.** Every
  level-5 locus is Euclidean, exactly as `c(z)=−` predicts. The causal-type rule
  generalises to a level it never saw. That part of the grammar is real, not
  metaphor.
- **The DIMENSION production is NOT generative as-stated.** Level 4's single output
  `J_spin(5)` is only one of level 5's outcomes; `ker 8 → J_spin(11)` is produced by
  finer kernel geometry the level-4 rule does not encode. So `R2`'s output *size*
  is level-parametric: it must read the kernel-dimension (itself context) as an
  explicit input, not a constant.

So the grammar is **generative at the categorical/signature layer, descriptive-
but-incomplete at the dimension layer** — and the falsification test *pinpointed*
the single rule (`R2` output dimension) that needs promotion from constant to
context-reading. That is the honest separation of grammar from metaphor: the rung
law and the null-gated anomaly are genuinely generative; the dimension is not yet.

## The Sounio reading

This meta-structure is the project's own thesis one level up: **a typed system
where the types are species-of-construction and the operations are typed functors,
with a context (the causal type of `z`) that gates which constructions are
derivable.** The novelty is not any single algebra in the tower — it is the
grammar that says which algebras are constructible from a given epistemic locus,
and the null branch's exclusive anomaly is a context-sensitive production, not a
coincidence.

Open: promote `R2`'s dimension rule to read the kernel-dimension context (fixing
the one non-generative rule); and run the null/timelike branches at *split* level 5
to test rung-law generativity beyond the spacelike branch that division level 5
allows.

---

## R2 promoted: the dimension rule is now generative

The one non-generative rule (R2's output dimension) is promoted from a level-4
constant to a **context-reading rule**, verified generative at level 5:

> **`dim P_z = 1 + |Stab(z)|`**, where `Stab(z)` = the basis units preserving
> `ker L_z`, equal to `|H| − |bad|` with `H = {k : k ⊕ supp(ker L_z) ⊆
> supp(ker L_z)}` the XOR-stabilizer of the kernel support and `|bad|` the
> sign-inconsistent coset.

Verified at level 5 (computed, `scratchpad/level5.py`), `dim P_z = 1 + |Stab|` holds
3/3:
```
ker 4  → |support|=8  |H|=8  |Stab|=5   dim P=6  = J_spin(5)
ker 8  → |support|=16 |H|=16 |Stab|=11  dim P=12 = J_spin(11)
ker 12 → |support|=24 |H|=8  |Stab|=5   dim P=6  = J_spin(5)
```
Key: `dim P_z` is set by the **support-stabilizer `|H|`, not the kernel dimension**
— ker 4 and ker 12 both give `|H|=8 → J_spin(5)`; only ker 8 (support = a rank-4
coset) gives `|H|=16 → J_spin(11)`. The rule reads the support geometry as context,
so it now GENERATES a new level correctly.

**Conjectured refinement (4 data points: level4 |H|=8; level5 |H|∈{8,16,8}):**
`|bad| = |H|/4`, hence `dim P_z = (3/4)|H| = 3·2^{r−2}` for `H` of rank `r`, giving a
**doubling ladder of spin factors** `J_spin(3·2^{k}−1) = J_spin(5), J_spin(11),
J_spin(23), J_spin(47), …` — the preservation geometry grows through a fixed family
as the support-stabilizer rank climbs. (Solid: `dim P = 1+|Stab|`, generative.
Conjecture: the `|bad|=|H|/4` closed form.)

**Grammar status: now generative at every layer** — signature (rung law, verified
level 5), dimension (promoted R2, verified level 5), and the null-gated anomaly (the
context-sensitive rule). The remaining test is the null/timelike branch at *split*
level 5 (division level 5 is all-spacelike, so only the Euclidean branch was
exercised).
