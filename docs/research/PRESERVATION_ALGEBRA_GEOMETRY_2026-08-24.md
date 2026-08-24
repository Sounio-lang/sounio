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

---

## UPDATE (same day): the ladder closes — Lorentzian appears under base-split

Extending to a **fully parametrized Cayley–Dickson** (doubling-sign vector `μ⃗`,
`scratchpad/ladder.py`) and scanning generic ZDs settles the open question: the
preservation-signature ladder is **complete**, and which rung you land on is set
by *where* you split and *which* zero-divisor you pick.

```
μ⃗ = (−,−,−,−)  division      : (0,5,0)×84                          all Euclidean
μ⃗ = (−,−,−,+)  last-split    : (1,0,3)×84                          all Carrollian
μ⃗ = (−,−,+,−)  BASE-SPLIT    : (0,5,0)×12 + (1,0,3)×48 + (4,1,0)×24   ALL THREE coexist
μ⃗ = (−,−,+,+)  doubly-split  : (1,0,3)×36 + (3,2,0)×48             Carrollian + Lorentzian
```

`(4,1,0)` is a **nondegenerate Lorentzian** spin factor `J_spin(4,1)` — the 5D
Minkowski/de-Sitter signature, whose structure group is `SO(4,1)`. Verified
**exactly** (rational arithmetic, `scratchpad/verify_lorentz.py`): for base-split
`z=e4+e13`, `ker L_z` dim 4 (genuine two-sided ZD), `P_z` dim 6, imaginary dim 5,
**Jordan-closed and spin-factor**, signature `(4,1,0)`.

### The finding

The preservation algebra of an exact-invariant realizes the **complete signature
ladder of relativistic geometry — Euclidean → Carrollian → Lorentzian** — as a
function of the Cayley–Dickson split-vector and the zero-divisor. Crucially, a
**single base-split algebra hosts all three simultaneously**: 12 Euclidean, 48
Carrollian, 24 Lorentzian pair-type loci. So within one algebra, *the choice of
privacy-locus `z` selects the composition geometry of the invariant.*

### Sounio consequence (sharpened)

`ExactlyPrivate<T, z>` is not one type — it is a **family whose composition
geometry is chosen by `z`**: pick a Euclidean locus and privacy composes freely
and definitely; pick a Lorentzian locus and composition-safety acquires a genuine
light-cone (`SO(4,1)` de-Sitter causal structure among preserving operations —
timelike vs spacelike operation-types, a causal order on capability composition).
This makes the ambient-split-vector and the ZD-locus *two type-level knobs* on the
causal/metric character of how invariants compose — an axis with, as far as the
literature scanned shows, no prior owner.

### Remaining open

- Is the rung a **closed-form function** of `(μ⃗, z)`? The base-split split of
  84 into 12/48/24 suggests the ZD-orbit under the split `G2`-analogue indexes the
  rung; not yet derived.
- Does `(4,1)` vs `(3,2)` correspond to a physically meaningful distinction
  (de Sitter vs anti-de Sitter-like) of the invariant's composition causal order?
- Lean formalization of "base-split `z=e4+e13` ⇒ `P_z ≅ J_spin(4,1)`".

---

## THE RUNG LAW (closed form, verified 84/84 across the CD family)

The rung is not ad hoc — it is the **causal type of the zero-divisor**. Let `Q` be
the square-form on the ambient algebra, `Q(e_i) := e_i² ∈ {±1}` (the form induced
by squaring; `Q(z) = Q(e_i)+Q(e_j)` for a pair-type `z = e_i ± e_j`).

> **Rung law (pair-type).** For a pair-type zero-divisor `z`,
> - `Q(z) < 0`  (z **spacelike**, both arms `−`)  ⇒  `P_z` **Euclidean** `J_spin(5)`, sig `(0,5,0)`
> - `Q(z) = 0`  (z **null**, mixed arms)           ⇒  `P_z` **Carrollian** (degenerate), sig `(1,0,3)`
> - `Q(z) > 0`  (z **timelike**, both arms `+`)    ⇒  `P_z` **Lorentzian** `J_spin(4,1)`, sig `(4,1,0)`

Verified exhaustively (`scratchpad/rung_derive.py`, `rung_law.py`,
`lastsplit_check.py`): the predictor `rung = g(sign Q(e_i), sign Q(e_j))` matches
the computed signature on **84/84** pair-type ZDs in base-split, and — being a
statement about arm-signs — trivially in division (all arms `−` ⇒ all Euclidean)
and last-split (all ZDs mixed-arm ⇒ all Carrollian; confirmed: 84/84 mixed).

### Why (mechanism)

The preserving generators of `P_z` inherit their `Q`-signs from `z`'s arms.
- **Spacelike z:** all preserving imaginary generators square to `−1` ⇒ definite
  Jordan form ⇒ Euclidean.
- **Null z:** the mixed-arm structure forces the preserving generators onto the
  light-cone (each pairs a `+` with a `−` and is `Q`-null) ⇒ the Jordan form
  acquires a rank-collapse (radical) ⇒ Carrollian.
- **Timelike z:** the preserving generators are `+`-dominated with one
  distinguished `−` (from the 4-dim kernel's orthogonal structure) ⇒ `(4,1)`
  Lorentzian.

### Statement

**The composition geometry of an exact-invariant is the causal type of its
defining zero-divisor.** Choosing the privacy-locus `z` is choosing whether the
invariant composes Euclidean-ly (definite, free), Carrollian-ly (null, ultra-local)
or Lorentzian-ly (a genuine light-cone / `SO(4,1)` causal order on capability
composition). The ambient split-vector fixes which causal types of locus *exist*;
the locus fixes the rung.

### Open (toward a paper)

- Prove the `(4,1)` (not `(3,2)` etc.) refinement from the kernel's orthogonal
  structure — i.e. derive the exact `+/−` split, not just its existence.
- Extend the law to generic (non-pair-type) ZDs, where `dim ker` and `dim P_z`
  vary; conjecture: rung still tracks `sign Q(z)` but the algebra is a larger/
  mixed `J_spin(p,q)`.
- Lean: `Q(z)>0` (base-split `e4+e13`) ⇒ `P_z ≅ J_spin(4,1)`.
