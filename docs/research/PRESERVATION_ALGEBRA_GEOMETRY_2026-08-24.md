<!-- docs:meta
topic_id: repo.docs.research.preservation-algebra-geometry-2026-08-24
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.preservation-algebra-geometry-2026-08-24
-->

<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# The preservation algebra of a zero-divisor: a Euclidean↔Carrollian dichotomy

**Date:** 2026-08-24 · **By:** claude (session 71fa6b78) · **Status:** computed finding (exact, rational + numpy cross-checked), not yet a formal proof.

## Setup

For a sedenion zero-divisor `z` (here the canonical `z = e3 + e10` and all 84
pair-type `e_i ± e_j`), define its **kernel** `ker L_z = {x : z·x = 0}` (dim 4)
and its **preservation algebra**

```
P_z = { a : a·ker L_z ⊆ ker L_z  and  ker L_z·a ⊆ ker L_z }   (two-sided stabilizer)
```

`P_z` is the set of multipliers under which the exact-invariant "x lives in
`ker L_z`" is preserved — the composition-safety set for `ExactlyPrivate`/
`Forgettable`/`Editable`/`CapabilityGated`. Its Jordan structure carries a
quadratic form `B(u,v) = scalar part of (u∘v)`, `a∘b = (ab+ba)/2`, whose
signature is the object of interest.

## Result: the Cayley–Dickson sign parameter is a geometry switch

The last-doubling sign `μ` (`e8² = −μ`) toggles the character of `P_z`:

| ambient | `μ` | `P_z` structure | signature `(+,−,0)` | geometry |
|---|---|---|---|---|
| **division** sedenions | `+1` | spin factor `J_spin(5)` | `(0, 5, 0)` — definite | **Euclidean** |
| **split** sedenions | `−1` | degenerate spin factor, null radical | `(1, 0, 3)` — rank 1 | **Carrollian** |

Both are **universal** across all 84 pair-type ZDs (division: every `P_z` is
`(0,5,0)`; split: every `P_z` is `(1,0,3)`). **No Lorentzian (mixed `p,q>0`,
nondegenerate) signature appears in either.** The original hypothesis — that some
member of the ZD-surgical type family, or the split ambient, would give a
Lorentzian preservation geometry with a genuine light-cone — is **refuted** by
exhaustive pair-type computation.

## What the split generators are

For split `z = e3 + e10`, the four imaginary preserving generators are
`{ −e1+e8, e9, e3+e10 (=z), e2+e11 }`. In the split form (`e1..e7`² = −1,
`e0,e8..e15`² = +1), three of them are **null** (light-like): `(e8−e1)`,
`(e10+e3)`, `(e11+e2)` each pair a minus-direction with a plus-direction and
square to 0 under `∘`. Only `e9` is timelike (`+1`). Hence the rank-1,
null-dominated `(1,0,3)` form: **the preserving operations sit on the light-cone
of the split norm.** That is the signature-theoretic content of a Carrollian
(degenerate-metric) geometry.

## Reading

- **μ = +1 (division):** every privacy-preserving operation is "spacelike" —
  definite, cleanly composable; `P_z` is a Euclidean observable algebra
  `J_spin(5)`.
- **μ = −1 (split):** the preserving operations degenerate onto the null cone;
  the "time" direction collapses to rank 1. Composition-safety becomes a
  causal-boundary phenomenon.

So a single discrete algebraic knob (`μ`) selects between the **two degenerate
limits of relativity** — Euclidean and Carrollian — while **skipping Lorentzian
entirely** for the pair-type spectrum.

## Sounio-actionable consequence

The ambient split-parameter could be exposed as a **type parameter** on the
exact-invariant family: `ExactlyPrivate` over division sedenions has Euclidean
(definite, freely composable) preservation; over split sedenions it has
Carrollian (null, causally-constrained) preservation. The type system would then
know that "how two privacy-typed values compose" is a *different geometry*
depending on the ambient — a genuinely new axis for invariant-type design.

## Honesty boundary / open

- Computed for **pair-type** ZDs only, and for the **last-doubling** split
  (`μ` on `e8`). A fuller split (splitting the octonion base) or generic
  (non-pair-type) ZDs might still produce a nondegenerate Lorentzian `(p,q,0)`;
  not yet excluded. The dichotomy claim is *pair-type, last-doubling*.
- "Carrollian" is used in the signature-theoretic sense (degenerate metric with a
  clock direction and a null radical), not a claim to have derived Carrollian
  spacetime dynamics.
- Prior art: Koebisu 2512.13002 (det-factorization of `L_v`), Moreno, Reggiani
  2411.18881, Biss–Dugger–Isaksen study the ZD **locus**; none treat the
  **preservation/stabilizer algebra** or its Jordan signature. Split/pseudo-
  octonion algebra is Okubo (cited there, not applied here). The
  preservation-signature dichotomy appears unremarked in the literature scanned.

## Reproduce

`scratchpad/pz_frontier.py` (division, exact rational), `scratchpad/family_sig.py`
(family + Composable intersection), `scratchpad/splitfast.py` (split, numpy scan
of all 84).
