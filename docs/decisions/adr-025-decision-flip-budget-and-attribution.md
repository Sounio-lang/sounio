<!-- docs:meta
topic_id: repo.docs.decisions.adr-025-decision-flip-budget-and-attribution
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.decisions.adr-025-decision-flip-budget-and-attribution
-->

# ADR-025: Epistemic decisions — flip budgets, attribution and the value of remeasuring

**Status**: accepted (implementation pending)
**Date**: 2026-10-28
**Decided by**: the founder, in the 2026-10-28 decision session
**Builds on**: [ADR-024](adr-024-metrological-index-obligations.md) (distribution-free bounds, guarantee levels, budgets), [ADR-016](adr-016-epistemic-merge-and-provenance-trust.md) (epistemic values and provenance)

## Context

A clinical rule is a branch on measured values:

```sounio
let crcl = (140.0 - age) * weight / (72.0 * scr)   // Cockcroft–Gault, mL/min
if crcl < 30.0 { reduce_dose() } else { standard_dose() }
```

Every value on the right carries an uncertainty, and so `crcl` does too. The branch is taken on the mean as if it were exact. When the mean sits near the threshold, measurement noise alone decides which dose the patient gets, and nothing in the program says so. Every programming language has this blind spot. Sounio already knows the uncertainty of `crcl`, so in Sounio the blind spot is a choice.

## Decision

### 1. Epistemic branches

An `if`, `match` guard or `while` condition whose operands are epistemic values is an **epistemic decision**. The compiler marks it statically. Conditions over exact values are unchanged.

### 2. Flip bound

For a condition `g(x₁…x_k) ⋈ t` (`⋈` one of `<`, `<=`, `>`, `>=`), let `s = g(μ)` be the score at the means. Let `u_s² = Σᵢⱼ cᵢ cⱼ Cov(xᵢ, xⱼ)` with `cᵢ = ∂g/∂xᵢ`, using the existing first-order and correlated propagation. The **flip** event is that the branch taken at the means differs from the branch the true values would take. It is bounded without assuming a distribution, by the one-sided Cantelli inequality:

```
ε_flip = u_s² / (u_s² + (s − t)²)
```

The bound carries an ADR-024 guarantee level (`exact`, `rigorous`, `1st-order`). It is `exact` for a linear `g`. For non-linear `g` it is `1st-order` unless certified intervals bound the second-order remainder. Equality tests on epistemic values are refused: `ε_flip` is not bounded below 1.

### 3. Attribution

`u_s²` is split among the inputs by Euler allocation: `wᵢ = cᵢ · Cov(xᵢ, s) / u_s²`. The shares sum to 1 under correlation. The **attribution set** of a decision is the smallest set of inputs whose shares sum to at least 0.9. It names the inputs whose uncertainty is responsible for the fragility.

### 4. Value of remeasuring

When an input's type declares its alternative assays (`scr: Q[mg/dL] assays { jaffe: 0.2, enzymatic: 0.05 }`), the compiler can price a remeasurement. It recomputes `ε_flip` with that input's uncertainty replaced by the alternative, and reports, for each assay, the bound the decision would then have. This is a value-of-information computation at the point of the decision.

### 5. Budget and refusal

A decision gets its budget from the ADR-024 mechanism. The module sets a ceiling (`#[flip_budget(0.05, 1st-order)]`), and a function or a single `if` may tighten it (`if crcl < 30.0 within flip <= 0.01 { … }`). The budget is **checked at run time, at each decision**, because `ε_flip` depends on the measured values. When `ε_flip` exceeds the budget, or the level is below the required one, the decision does not choose a branch. It evaluates to a **refusal**:

```
Refused { flip: 0.227 [1st-order], attribution: [scr 0.984, weight 0.016],
          remeasure: [scr enzymatic → flip 0.022] }
```

The refusal is a value of the effect `Refuse`. Under ADR-018 this is the `refusal` class: refusing replaces a silent guess. A function containing an epistemic decision must declare `Refuse` or handle it. The compiler checks this statically, so no caller can forget that a decision may decline to decide.

### 6. Provenance

Every epistemic decision that does choose a branch records its `ε_flip`, its level, its attribution set and the budget it met, in the provenance of the values it produces (ADR-016). A downstream result can state how fragile the decisions behind it were.

## Worked example

Computed with the formulas above (C++23 oracle [`docs/audit/ADR_024_025_ORACLES_2026-10-28/adr025_flip.cpp`](../audit/ADR_024_025_ORACLES_2026-10-28/adr025_flip.cpp)). Inputs: age 70 (exact), weight 70 ± 1 kg, independent inputs, threshold 30 mL/min.

| SCr (mg/dL) | CrCl | u(CrCl) | `ε_flip` | attribution |
|---|---|---|---|---|
| 1.8 ± 0.20 (Jaffe) | 37.81 | 4.24 | **0.227** | scr 0.984, weight 0.016 |
| 1.8 ± 0.05 (enzymatic) | 37.81 | 1.18 | **0.022** | scr 0.791, weight 0.209 |
| 2.2 ± 0.20 | 30.93 | 2.85 | **0.903** | scr 0.976 |
| 2.2 ± 0.05 | 30.93 | 0.83 | **0.441** | scr 0.717, weight 0.283 |

Take a budget of 0.05. The first patient is refused, with the instruction to remeasure SCr enzymatically, and that remeasurement would make the decision admissible (0.022). The second patient is too close to the threshold for any available assay: the refusal reports that even the enzymatic assay leaves 0.441. That is the honest answer, and today no program gives it.

## Consequences

- Fragile clinical, regulatory and scientific decisions become visible, priced and refusable, instead of being silent coin tosses decided by measurement noise.
- The refusal points to the remedy: which input to remeasure, and with which assay.
- Cantelli is conservative. Near a threshold the bound is loose, and the refusal errs towards asking for more data. A tighter bound needs a declared distribution; that is a separate decision.
- Exact-valued programs are unaffected.

## Prior art

- **Robustness verification** of classifiers bounds how far inputs can move before a decision flips (certified robustness, randomized smoothing), but for learned models, not for program branches over measured values.
- **Value-of-information analysis** is standard in decision theory and health economics (EVPI/EVSI), but it is computed by hand, outside the program that makes the decision.
- **Sensitivity attribution**: Euler allocation is the risk-contribution rule of portfolio theory; Sobol indices and Shapley values attribute the variance of a model's output.
- **Probabilistic programming** reasons about distributions over executions, but does not refuse a branch as too fragile at the point of decision.

The new part is to treat a branch on measured values as a decision with a guaranteed flip bound, an attribution and a priced remeasurement. All three are computed by the language at the branch. A budget turns a fragile decision into a typed refusal that callers cannot ignore.

## Acceptance

1. The four table rows above are reproduced by the Sounio implementation to 3 significant figures, against the C++ oracle.
2. With `#[flip_budget(0.05, 1st-order)]`, patient 1 at Jaffe precision returns `Refused` whose attribution is led by `scr` and whose remeasure entry reports about 0.022 for the enzymatic assay. Patient 1 at enzymatic precision takes the standard-dose branch, with `ε_flip ≈ 0.022` in provenance.
3. A function with an epistemic decision that neither declares nor handles `Refuse` is refused at compile time.
4. `if crcl == 30.0` on an epistemic `crcl` is refused at compile time.
5. A decision over exact values compiles and runs exactly as before, with no `Refuse` and no runtime cost.

## Implementation

Cards under EPIC · Decisions owed:
- marking epistemic decisions and the static `Refuse` check;
- the runtime flip bound and the Euler attribution, reusing the correlated-propagation machinery;
- assay declarations and remeasurement pricing;
- budgets and policy, shared with ADR-024;
- provenance records;
- the worked example as a run-pass test with its C++ oracle.
