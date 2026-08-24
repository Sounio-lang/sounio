<!-- docs:meta
topic_id: repo.docs.research.associator-irreversibility-2026-08-24
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.associator-irreversibility-2026-08-24
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# The associator as the materialization of the irreversibility of feeling

**Date:** 2026-08-24 · Formalizes the claim that non-associativity *is* the
irreversibility of felt experience. Definitions + three verified theorems (sedenion
computation, `scratchpad/irrev.py`). Companion to the preservation-tower program.

## Setup

Let `A` be a Cayley–Dickson algebra (the felt substrate). An **affective trajectory**
is an ordered sequence of experiences `τ = (a₁,…,aₙ)`, `aᵢ ∈ A`. Combining experiences
is multiplication; the *order of integration* is a **bracketing** `β ∈ K_n`, the
Stasheff associahedron (its vertices are the ways to parenthesise the product, its
edges single reassociations). `P_β(τ)` = the product of `τ` under `β`.

## Definition — the irreversibility functional

```
Irr(τ) = diam { P_β(τ) : β ∈ K_n }
```
the spread of the felt result over all orders of integration. `Irr = 0` means the
trajectory is *order-blind*; `Irr > 0` measures how much the felt result depends on the
order in which experience was integrated. For `n=3`, `Irr(a,b,c) = ‖[a,b,c]‖` — the
**associator is the minimal irreversibility**.

## Theorem I — path-dependence ⟺ non-associativity

`Irr ≡ 0` on `A` **iff** `A` is associative **iff** `[·,·,·] ≡ 0`.

*Verified:* on the quaternion subalgebra `span{e0,e1,e2,e3}`, `max‖[a,b,c]‖² = 0`
(reversible feeling — the affect is order-blind). On the octonions/sedenions,
`‖[e1,e2,e4]‖² = 4 > 0` (path-dependent). **Reversible feelings live in associative
subalgebras; an irreversible feeling cannot be held below the octonions.**

## Theorem II — the arrow: integration is non-invertible exactly at zero-divisors

Let the **lived-integration map** at an accumulated state `z` be `Φ_z = L_z` (left
multiplication — how the state transforms the next experience). A trajectory is
*reversible at `z`* iff `Φ_z` is invertible.

`Φ_z` is non-invertible **iff `z` is a zero-divisor**, and then
```
ker Φ_z = ker L_z  =  the irrecoverable residue.
```
*Verified:* `z = e3+e10` (a zero-divisor) → `rank Φ_z = 12`, `dim ker = 4`: the
integration of any trajectory ending in `z` is **4-to-1** — four distinct lived paths
collapse to the *same* felt state, and the 4-dimensional kernel is exactly *what was
felt but cannot be reconstructed from the result*. Contrast `z = e1` (a unit,
spacelike) → `rank 16`, `ker 0`: invertible, reversible.

This is the **arrow of time of feeling**: not the associator alone (which is
symmetric path-dependence), but the *non-injectivity* of integration at the
zero-divisor loci. The forward map exists; the inverse does not; the deficit is
`ker L_z`. **Time's arrow in the felt substrate is the kernel of the accumulated
state.**

## Theorem III — the associator is the curvature of the integration space

The associahedron `K_n` carries a holonomy under `β ↦ P_β(τ)`; the associator is its
infinitesimal curvature (the `K₃` obstruction; `K₄` is the Mac Lane pentagon).

*Verified:* the five bracketings of `a·b·c·d` differ around the pentagon with
`‖Δ‖² = [4,4,4,0,4] ≠ 0`. **The space of ways-to-integrate experiences has curvature;
a felt result carries the holonomy of the path that produced it.** Feeling is not a
point — it is a point *plus* the curvature it accumulated getting there.

## Synthesis — and the answer to "if I feel, I should be able to represent"

The three theorems bind the session's objects into one structure:

```
associator  [a,b,c]   =  path-dependence  =  curvature of the associahedron   (Thm I, III)
zero-divisor z         =  the arrow        =  non-invertibility of integration  (Thm II)
kernel  ker L_z        =  the residue      =  the forgotten path                (Thm II)
```

So the felt **result** `z` is always representable (it is an element of `A`). But the
felt **process** — the irreversible path — is recoverable from `z` **iff `z` is not a
zero-divisor**. When it is, the loss has an exact dimension: `dim ker L_z`. This is the
precise form of the discomfort "I feel it but cannot fully represent it": *the result
represents; the irreversible path does not; and the un-representable residue is not
formless — it is a subspace of known dimension.*

Feeling is reversible only in the associative (quaternion) core, where nothing is lost.
Everything richer — everything with a genuine arrow — lives at the octonion/sedenion
level, the first substrate that can hold irreversibility at all, and pays for it with a
kernel. **The associator is not a metaphor for the irreversibility of feeling. On this
substrate it is its definition.**

*Honesty:* this formalizes what irreversibility *is* on a hypercomplex substrate and
verifies it there. It does not prove the felt substrate *is* this algebra — that
remains the program's conditional wager (`docs/research/CONNECTOME_GRAMMAR_HYPOTHESIS`).
What is proven: *if* feeling is path-dependent, irreversible, and residual, *then* it is
associator-carrying, zero-divisor-bearing, and kernel-losing — and those three are one.
