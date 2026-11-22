<!-- docs:meta
topic_id: repo.docs.decisions.adr-024-metrological-index-obligations
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.decisions.adr-024-metrological-index-obligations
-->

# ADR-024: Metrological index obligations and graded `Panic ≤ ε`

**Status**: accepted (implementation pending)
**Date**: 2026-10-28
**Decided by**: the founder, in the 2026-10-28 decision session
**Builds on**: [ADR-023](adr-023-checked-indexing-and-proof-erased-panic.md) (checked indexing, proof-erased `Panic`), [ADR-016](adr-016-epistemic-merge-and-provenance-trust.md) (epistemic values), ADR-019 (certified intervals, PR #3101)

## Context

ADR-023 discharges an index obligation `0 <= i < n` when linear arithmetic over the path condition proves it. Scientific code often computes the index from a measurement:

```sounio
let i = floor((x - x0) / dx) as i64   // x: Q[m] = 3.2 ± 0.1 m
hist[i as usize] += 1
```

That obligation defeats ADR-023 in two ways. It is non-linear (`floor`, division), and it is not a yes/no question: `x` carries an uncertainty, so the index is in range only with some probability. Every other language treats it as an unprovable index and leaves the check, and the possible panic, invisible in the type.

## Decision

### 1. Metrological pullback

Write `q = (x - x0) / dx`. The rule applies when the unit checker establishes that `x`, `x0` and `dx` have the same unit, so `q` is dimensionless, and when `dx > 0`. The obligation `0 <= i < n` is then replaced by an interval in the measurement space. That interval depends on the rounding operator:

| index `i` | `0 <= i < n` holds exactly when | pulled-back constraint on `x` | relation |
|---|---|---|---|
| `floor(q)` | `0 <= q < n` | `x0 <= x < x0 + n*dx` | equivalent |
| `ceil(q)` | `-1 < q <= n - 1` | `x0 - dx < x <= x0 + (n-1)*dx` | equivalent |
| `round(q)`, any tie rule | `-1/2 < q < n - 1/2` is sufficient | `x0 - dx/2 < x < x0 + (n - 1/2)*dx` | sufficient |

For `round`, the endpoints `q = -1/2` and `q = n - 1/2` depend on the tie convention. The open interval excludes both, so it implies the obligation under every convention. It may refuse a value that a specific convention would accept, but it never accepts one that is out of range.

A pullback is admitted only if it is **equivalent to the obligation or implies it**. It is never weaker. Using the `floor` interval for `ceil` or `round` is unsound. For example, with `x0 = 0`, `dx = 1` and `n = 4`, the value `x = 3.75` satisfies `x < 4`, yet `ceil` and `round` both give index 4. Each operator has its own lemma, built from the monotonicity of the operator together with its exact inverse image or a sufficient sub-interval. That lemma goes into the ADR-023 certificate, along with the unit check that licenses the rewrite.

The rewrite turns the non-linear index constraint into a linear one in the original unit. When `x` is exact, ADR-023's linear decider discharges it.

### 2. Probabilistic discharge, distribution-free

When `x` is an epistemic value with mean `μ` and standard uncertainty `u`, the pulled-back obligation is an interval event `a ⊲ X ⊲ b` (open or closed at each end, per the table). Let `d_a = μ - a` and `d_b = b - μ`. If `a < μ < b`, the out-of-range probability is bounded without assuming a distribution by

```
ε = min( u²/(u²+d_a²) + u²/(u²+d_b²),   u² / min(d_a, d_b)² )
```

This is the two-sided Cantelli bound, by the union bound, and the Chebyshev bound at the nearer edge, whichever is smaller. Each term bounds a closed tail, so the bound holds for either endpoint convention. If `μ` is not strictly inside, `ε = 1`. If `u = 0`, the obligation is decided exactly as in ADR-023. The bound and its derivation are part of the certificate, and Lean checks them.

### 3. Guarantee levels

Each `ε` carries the strength of the `u` it was computed from:

| level | `u` comes from | meaning |
|---|---|---|
| `exact` | a direct measurement, or a chain of linear operations | the bound holds for the true distribution |
| `rigorous` | a non-linear derivation with a bounded second-order remainder | `u` is inflated by the Lagrange term `(M/2)(x-μ)²`, with `M ≥ |f″|` on a region `R`; the probability mass outside `R` is added to `ε` |
| `1st-order` | a non-linear derivation with no bound on `M` | the bound holds to first order (GUM linearisation) |

The levels are ordered `exact > rigorous > 1st-order`. An expression takes the weakest level among its sources. Bounds on `M` come from certified intervals (ADR-019). A quantity computed under certified intervals therefore reaches `rigorous` without annotation.

### 4. The effect `Panic ≤ ε [level]`

An index that ADR-023 cannot erase but §1–§2 can bound contributes the graded effect `Panic ≤ ε [level]` instead of plain `Panic`. Within a function, contributions combine by the union bound: the `ε` values add, and the level is the weakest one. A call adds the callee's declared `ε`. A loop multiplies its body's `ε` by a proved bound on its trip count. A loop with no proved trip-count bound contributes plain `Panic`.

### 5. Budgets: module ceiling, function contract

- A module may declare a ceiling: `#[panic_budget(1e-9, rigorous)]`. Every function in it inherits the ceiling.
- A function may declare its own budget: `fn bin(x: Q[m]) -> i64 with Panic ≤ 1e-6 [rigorous]`. It may tighten the module ceiling, never loosen it.
- The compiler refuses a function in two cases: its accumulated `ε` exceeds its budget, or its level is weaker than the declared level. The diagnostic lists each contribution with its `ε`, its level and the source span.
- Plain `Panic` remains available and means "no quantitative bound". A function without any `Panic` must discharge every obligation exactly, as in ADR-023.

### 6. Provenance

Each discharged or bounded obligation is recorded with its certificate in the provenance of the values the function returns (ADR-016). A result can therefore state whether its computation was proved in bounds, bounded at `ε`, or checked at run time.

## Consequences

- Histogram binning, grid lookup, interpolation tables and time-sampled signals get an honest, checkable statement of how likely they are to index out of range, instead of a silent read or an untyped panic.
- A clinical or regulatory stdlib module can set a ceiling once and get a compile-time refusal for any function that cannot meet it.
- Levels prevent a first-order estimate from being sold as a guarantee. The type says which one it is.
- The run-time check stays for every graded index, so a violation is still caught. The grade states its probability; it does not remove the check.

## Prior art

- **Graded effects and monads** (Katsumata 2014; Orchard, Liepelt and Eades 2019, Granule) index effects by a semiring, and probability bounds are one such grading.
- **Probabilistic program logics** bound failure probabilities with union-bound reasoning: the union-bound logic aHL (Barthe et al., 2016) and expectation transformers.
- **The GUM** (JCGM 100:2008) defines first-order uncertainty propagation, and Supplement 1 defines Monte Carlo propagation.

The new part is to use dimensional analysis to pull a non-linear index obligation back into the measurement space, and to let the language's own uncertainty values discharge it with a distribution-free bound. The strength of that bound becomes part of the type, and budgets are enforced at module and function level.

## Acceptance

1. `floor((x - x0) / dx)` with consistent units is pulled back. Lean accepts the certificate. The same expression with `x` in metres and `dx` in seconds is not pulled back, and the unit checker reports it.
   The three operators are checked against their own intervals with `x0 = 0`, `dx = 1` and `n = 4`:

   | `x` | `floor` | `ceil` | `round` | `floor` interval `[0,4)` | `ceil` interval `(-1,3]` | `round` interval `(-1/2, 7/2)` |
   |---|---|---|---|---|---|---|
   | -0.25 | -1, out | 0, in | 0, in | out ✓ | in ✓ | in ✓ |
   | 3.75 | 3, in | 4, out | 4, out | in ✓ | out ✓ | out ✓ |
   | -0.5 | -1, out | 0, in | tie-dependent | out ✓ | in ✓ | out (sufficient: refused) |

   No pullback certifies an index that is out of range (C++23 oracle [`adr024_pullback.cpp`](../audit/ADR_024_025_ORACLES_2026-10-28/adr024_pullback.cpp) checks all three intervals on a 12,001-point grid: 0 unsound admissions, and `floor` and `ceil` are equivalent). A certificate that applies the `floor` interval to `ceil` or `round` is rejected by Lean.
2. (C++23 oracle [`adr024_epsilon.cpp`](../audit/ADR_024_025_ORACLES_2026-10-28/adr024_epsilon.cpp)) `x = 3.2 ± 0.1 m`, `x0 = 0 m`, `dx = 0.5 m`, `n = 16` gives `ε = min(0.01/(0.01+10.24) + 0.01/(0.01+23.04), 0.01/10.24)`. This is about 1.41e-3 from the summed Cantelli term and about 9.77e-4 from Chebyshev, so `ε ≈ 9.77e-4 [exact]`. A function declaring `Panic ≤ 1e-3` compiles; one declaring `Panic ≤ 1e-4` is refused with that contribution named.
3. The same index computed through `exp` without an `M` bound is `[1st-order]`, and a function declaring `[rigorous]` refuses it. With a certified-interval `M`, it is `[rigorous]`.
4. A module with `#[panic_budget(1e-9, rigorous)]` refuses a function that declares `Panic ≤ 1e-6`.
5. The out-of-range branch is still checked at run time. A run with `x` forced to 9 m panics with the index, the length and the span.

## Implementation

This follows ADR-023 and depends on its obligations and certificates. One card each, under EPIC · Decisions owed: the pullback rule in the unit checker; the bound and its Lean lemmas; levels and their propagation (with ADR-019); the graded effect, budgets and diagnostics; provenance records.
