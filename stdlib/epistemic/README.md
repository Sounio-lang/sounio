# Epistemic Computing Standard Library

**Every value knows its uncertainty. Every computation propagates variance.**

## Overview

The `epistemic` module provides first-class uncertainty quantification for Sounio. The checked public surface is `epistemic::knowledge`: the `Epistemic` struct and the `ep_*` free functions, anchored by `tests/stdlib/epistemic/test_knowledge_madaros_import_e2e.sio`, `tests/stdlib/epistemic/test_knowledge_stdlib.sio`, and `tests/run-pass/ep_gum_covariance.sio`.

- **Value**: point estimate (`val: f64`)
- **Variance**: uncertainty of the value (`variance: f64`, σ²)
- **Confidence**: reliability of the estimate (`confidence: i64`, 0..1000; `ep_measured` stores 900)

Provenance is not a field of `Epistemic`. Source tracking lives in `stdlib/epistemic/provenance.sio` and `stdlib/epistemic/affine` (anchor: `tests/run-pass/affine_shared_source_add.sio`).

## Modules

### `knowledge.sio` — Core Type

```sio
use epistemic::knowledge::{ep_measured, ep_div, ep_val, ep_std, ep_is_credible}

// Create epistemic values (val, std_dev). Variance is stored as std_dev^2
// and confidence defaults to 900/1000.
let dose = ep_measured(500.0, 25.0)
let volume = ep_measured(10.0, 0.01)

// Arithmetic propagates variance by the GUM delta method (uncorrelated).
let concentration = ep_div(&dose, &volume)
// ep_val(&concentration) = 50.0, ep_std(&concentration) = 2.5

// Confidence gate: the threshold is an integer on the 0..1000 scale.
if ep_is_credible(&concentration, 900) {
    // proceed
}
```

The method form is equivalent under Madaros multi-module (anchor: `tests/run-pass/madaros_knowledge_method_form.sio`): `Epistemic::measured(500.0, 25.0)`, `dose.div(&volume)`, `concentration.val()`, `concentration.std()`.

`prob_gt`, `ci95`, and a provenance string argument on `measured` are not on the checked surface.

### `propagate.sio` — Variance Propagation

Free functions over `Epistemic` (anchor: `tests/stdlib/epistemic/test_propagate_stdlib.sio`):

```sio
use epistemic::knowledge::{ep_measured}
use epistemic::propagate::{sum, product, exp, ln}

let x = ep_measured(2.0, 0.1)
let y = ep_measured(3.0, 0.1)

let s = sum(x, y)        // variances add: 0.01 + 0.01 = 0.02
let p = product(x, y)    // delta method: y^2 var(x) + x^2 var(y)
let exp_x = exp(x)       // Var(e^X) ≈ e^(2X) * Var(X)
let log_x = ln(x)        // Var(ln X) ≈ Var(X) / X^2
```

**Propagation Rules:**

| Function | Variance Formula |
|----------|------------------|
| `sum(X, Y)` | `Var(X) + Var(Y)` |
| `diff(X, Y)` | `Var(X) + Var(Y)` |
| `product(X, Y)` | `Y²Var(X) + X²Var(Y)` |
| `quotient(X, Y)` | `Var(X)/Y² + X²Var(Y)/Y⁴` |
| `exp(X)` | `e^(2X) · Var(X)` |
| `ln(X)` | `Var(X) / X²` |
| `ep_sqrt_ep(X)` | `Var(X) / (4X)` |
| `ep_square(X)` | `4X² · Var(X)` |

Monte Carlo: `monte_carlo_identity` and `monte_carlo_square` are the checked entry points. The generic `monte_carlo(x, f, n)` fn-pointer form is flagged as fragile in the module header and has no run-pass anchor.

### `gum.sio` — GUM Uncertainty Budget

```sio
use epistemic::gum::{
    gum_type_a, gum_type_b_uniform, gum_combine2,
    gum_value, gum_std_u, gum_u95,
}

let ua = gum_type_a(0.070710, 5)    // type A: std_dev / sqrt(n), nu = n - 1
let ub = gum_type_b_uniform(0.5)    // type B, uniform: half_width / sqrt(3)
let r = gum_combine2(98.3, ua, ub)  // root-sum-square combination
let v = gum_value(r)                 // 98.3
let uc = gum_std_u(r)                // combined standard uncertainty
let u95 = gum_u95(r)                 // expanded uncertainty at 95%
```

Anchor: `tests/stdlib/epistemic/test_gum_stdlib.sio`. Also available: `gum_type_b`, `gum_type_b_triangular`, `gum_type_b_expanded`, `gum_with_sensitivity`, `gum_combine3`, `gum_dof`, `gum_k95`, `gum_u99`.

### `active.sio` — Active Inference

```sio
use epistemic::knowledge::{ep_measured}
use epistemic::active::{exploration_priority, ucb_select}

let drug_a = ep_measured(0.65, 0.10)
let drug_b = ep_measured(0.60, 0.02)

// Index of the value with the most to learn (highest variance).
let priority = exploration_priority(&[drug_a, drug_b])

// Upper-confidence-bound selection; returns Option<usize>.
let chosen = ucb_select(&[drug_a, drug_b], 1.0)
```

Anchor: `tests/stdlib/epistemic/test_active_stdlib.sio`, which pins `precision`, `coefficient_of_variation`, `relative_uncertainty`, `expected_info_gain`, and `prediction_error`. `expected_free_energy(current, expected_posterior_var, expected_reward, reward_weight)` and `update_belief(prior, observation, observation_variance)` exist in the module but are not exercised by that test.

### `meta.sio` — Meta-Analysis (no run-pass anchor)

`fixed_effects(&[Epistemic])`, `random_effects(&[Epistemic])`, and `bayesian_pool(&[Epistemic], prior_mean, prior_variance)` are defined and return `MetaResult { pooled, heterogeneity, k, method, weights }`, with `Heterogeneity { q, df, i_squared, tau_squared, p_value }`. The module imports `Beta`, `BetaConfidence`, and `Provenance` from `epistemic::knowledge`, which does not export them, and nothing under `tests/stdlib/epistemic/` or `tests/run-pass/` exercises it. Treat it as unanchored until that import is reconciled.

### `merkle.sio` — Provenance hashing (no public API)

The module implements fixed-size hash helpers, but nothing in it is `pub` and there is no `MerkleDAG` type. A Merkle DAG example is not on the checked surface.

## The `Epistemic` Type

```
Epistemic                       (stdlib/epistemic/knowledge.sio)
├── val: f64                    -- point estimate
├── variance: f64               -- uncertainty (σ²)
└── confidence: i64             -- 0..1000 (1000 = certain, 900 = ep_measured default)

Constructors: ep_measured(val, std_dev), ep_certain(val), ep_new(val, variance, confidence)
Accessors:    ep_val, ep_variance, ep_std (returns √variance), ep_confidence
Arithmetic:   ep_add, ep_sub, ep_mul, ep_div, ep_scale, ep_shift, ep_square, ep_sqrt_ep
Covariance:   ep_add_cov, ep_sub_cov, ep_mul_cov, ep_div_cov
Fusion/gates: ep_merge, ep_is_credible, ep_gate
```

`BetaConfidence` (an alpha/beta posterior) is not part of `Epistemic`. A separate `EpistemicBeta` type lives in `stdlib/epistemic/beta_confidence.sio` (`eb_new`, `eb_mean`, `eb_fuse_independent`), anchored by `tests/run-pass/beta_confidence_rule.sio` under Madaros.

## Units

Units-as-type-parameters (`Knowledge<mg>`, `500.0_mg`) are not on the checked surface. Dimensional values use `stdlib/units/lib.sio` (`Quantity`, `quantity_new`, `dim_mass()`, `dim_time()`, `quantity_div`); see `examples/units/dimensional_report.sio`.

## Design Principles

1. **Variance over error bars**: σ² is stored because variance is additive.
2. **Confidence is an integer 0..1000** on the checked `Epistemic` surface, not a distribution.
3. **Provenance is tracked separately** (`provenance.sio`, `affine`), not as a field of `Epistemic`.
4. **Confidence only degrades**: `ep_add` and friends keep the minimum of the inputs' confidence.

## References

- Taylor, J.R. "Introduction to Error Analysis"
- JCGM 100:2008, "Guide to the Expression of Uncertainty in Measurement"
- Gelman, A. et al. "Bayesian Data Analysis"
- Friston, K. "Active Inference and Free Energy"
- Pearl, J. "Causality: Models, Reasoning, and Inference"

## License

MIT / Apache-2.0 (same as Sounio)
