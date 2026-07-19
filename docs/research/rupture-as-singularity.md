<!-- docs:meta
topic_id: repo.docs.research.rupture-as-singularity
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.rupture-as-singularity
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Rupture as singularity — the Thom/Petitot bridge, realized in Cayley–Dickson algebra

*Follow-on to `nonassociativity-as-rupture.md`. Pulls the thread it flagged as next: the
singularity-theoretic bridge between the two faces of rupture (semantic — Thom/Petitot morphodynamics;
epistemic — sedenion annihilation). The bridge is not a metaphor. It is an identity, and it is computed.*

## The identity

In catastrophe theory (Thom), rupture is **crossing a bifurcation set**: a smooth family of potentials
`V(·; c)` parametrized by controls `c` degenerates on a subvariety (where the Hessian is singular), and
meaning/behavior jumps when `c` crosses it. Petitot builds semantic opposition this way — the semiotic
square as a multi-well potential, opposition = the bifurcation set.

In a Cayley–Dickson algebra, **the zero-divisor set is the singular locus of left-multiplication**: `x` is
a zero divisor iff `L_x` (the linear map `y ↦ x·y`) is singular iff `det L_x = 0` (Dugger–Isaksen). This
is *literally* a bifurcation set: a family of operators `L_x` parametrized by `x` that degenerates on a
subvariety. **Rupture-as-singularity is the same event on both sides — `det L_x = 0`.**

## What is computed (`catastrophe_cd.py`)

The catastrophe set of Cayley–Dickson multiplication, scanned across the doubling tower:

| Algebra | dim | `det L_x = |x|^dim`? | catastrophe set (singular 2-unit sums `e_i ± e_j`) |
|---|---|---|---|
| ℍ quaternion | 4 | **holds** (division) | **0** / 6 |
| 𝕆 octonion | 8 | **holds** (division) | **0** / 42 |
| 𝕊 sedenion | 16 | fails | **84** / 210 |
| 𝕋 trigintaduonion | 32 | fails | **588** / 930 |

- **The catastrophe set is empty for the division algebras ℝ,ℂ,ℍ,𝕆** — `det L_x = |x|^dim` never vanishes
  off the origin — and is **born at the 𝕆→𝕊 doubling.** Cayley–Dickson doubling is an **unfolding
  sequence**: order → commutativity → associativity → **division** is lost, and the loss of division at 𝕊
  *is* the appearance of Thom's bifurcation set. It then grows under further doubling (84 → 588).
- **The count validates the construction:** exactly **84** two-unit zero divisors in 𝕊 — the Cawagas (2004)
  number. The compiler's canonical XOR/`cd_sigma` basis reproduces the standard sedenion zero-divisor
  structure exactly. Annihilation verified directly: `a = e₁+e₁₀` gives a `b` with `‖a·b‖ = 1.8e-16`
  (nonzero × nonzero → 0).
- **The bifurcation path.** Interpolating `x(t)` from a generic element to a zero divisor,
  `det L_x(t): +0.098 → +0.014 → +0.0072 → +0.0040 → +0.00045 → 0` — the determinant crosses zero exactly
  as `x` enters the rupture set. This is Thom's "crossing the bifurcation set," in the algebra.

## The two faces, joined

- **`det L_x` is the positive, graded rupture object** the synthesis (§4) claimed no prior formalism had.
  Thom detects rupture by a *degeneracy* (Hessian determinant vanishes — a negative object); here the same
  event is the vanishing of a **positive graded quantity** (`det L_x`, the "distance to rupture") that can
  be measured everywhere and reaches zero at the catastrophe. Positivization, concretely.
- **The associator is the germ; the zero-divisor variety is its unfolding.** The associator
  `[a,b,c] ≠ 0` (order-1 rupture, the G₂ 3-form) is lost at 𝕆; the zero divisors (order-2 rupture,
  annihilation) are born at 𝕊 — the very next doubling. The non-associativity of 𝕆 is the germ whose
  Cayley–Dickson unfolding produces the 𝕊 catastrophe set. Both are singular loci of the multiplication:
  the associator is where multiplication fails to be *flat* (associative); the zero divisor is where `L_x`
  fails to be *invertible*. **Semantic rupture (Thom/Petitot's bifurcation of a meaning-potential) and
  epistemic rupture (algebraic annihilation) are the same singularity structure** — which is exactly the
  bridge the synthesis said was unwritten.

## Honest scope

This establishes the **structural identity** (zero-divisor set = bifurcation set of `L_x`; born as a
catastrophe at 𝕆→𝕊; det = positive distance-to-rupture) rigorously and by computation, and reproduces the
Cawagas count as a check. It does **not** yet build the *semantic* potential explicitly — i.e. exhibit a
Petitot-style meaning-potential whose bifurcation set is realized by (or dual to) the zero-divisor variety
of a specific algebra. That construction — a concrete morphodynamic model whose organizing center is the
Cayley–Dickson associator, testing whether the semiotic square's non-Booleanizable topology (Petitot's
impossibility theorem) matches the G₂/Fano combinatorics of the annihilation locus — is the next thread.
The exceptional-singularity resonance (Wildgen's four-actant E₆–E₈; G₂ as the common symmetry of both the
associator and the zero-divisor manifold) is the place to look. Harness `catastrophe_cd.py`.
