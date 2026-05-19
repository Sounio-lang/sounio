# Stats

## Overview

Comprehensive statistics module: distributions, inference, regression (linear/logistic/robust), Bayesian analysis, causal inference, multivariate methods, and timeseries.

## Epistemic Differentiators

- Stable v1 workflow facade in [`stats::v1`](./v1.sio)
- Lower-level epistemic modules can return diagnostics as [`BetaConfidence`](../epistemic/knowledge.sio) values
- [`LinearRegression::builder()`](./regression/linear.sio) with epistemic fit tracking VIF, Cook's D, leverage
- Uncertainty propagation in bootstrap resampling and MCMC
- [`KnowledgeLinearModel`](./regression/linear.sio) wraps results with provenance

## Quickstart

```sio
use stats::v1::{describe, compare_groups, regress_simple, regression_evidence_mean};

let summary = describe(values, n);
let comparison = compare_groups(control, n_control, treated, n_treated);
let regression = regress_simple(exposure, outcome, n);

assert(summary.mean > 0.0);
assert(comparison.ci_lower < comparison.ci_upper);
assert(regression.model.r_squared > 0.0);
assert(regression_evidence_mean(&regression) > 0.0);
```

## Sounio-Native Workflow Surface

The `stats::v1` module is the recommended public entry point for ordinary
analysis scripts. It is intentionally workflow-shaped rather than a Stata
command clone:

| Workflow | Function | Example |
|----------|----------|---------|
| Descriptive summary | `describe(values, n)` | [`examples/stats/v1_descriptive_workflow.sio`](../../examples/stats/v1_descriptive_workflow.sio) |
| Group comparison | `compare_groups(a, n_a, b, n_b)` | [`examples/stats/v1_group_comparison_workflow.sio`](../../examples/stats/v1_group_comparison_workflow.sio) |
| Simple regression | `regress_simple(x, y, n)` | [`examples/stats/v1_regression_workflow.sio`](../../examples/stats/v1_regression_workflow.sio) |

Each workflow also exposes a `V1AssumptionReport` with separate evidence
channels:

| Channel | Meaning |
|---------|---------|
| `sample_evidence` | observation count support |
| `balance_evidence` | group-size balance for comparison workflows |
| `variance_evidence` | variance stability or group variance-ratio support |
| `fit_evidence` | simple-regression fit quality |
| `overall_evidence` | mechanical `BetaConfidence` combination of the available channels |

Use `BetaConfidence` helpers from `stats::v1` when a workflow needs an explicit
evidence carrier. These helpers are not calibrated coverage probabilities and
are not full assumption diagnostics; they are v1 evidence channels for
conservative workflow gating. The variance and fit channels are deliberately
heuristic and scale-dependent in v1; future validated diagnostics should replace
them where clinical or publication-grade inference is required.

v1 diagnostics are first-pass workflow gates, not publication-grade assumption
tests. Still missing for that stronger tier: Levene-style variance testing,
Shapiro/residual normality checks, and robust regression.

```sio
use stats::v1::{beta_confidence, confidence_mean};

let assumptions = beta_confidence(9.0, 1.0);
assert(confidence_mean(&assumptions) > 0.80);
```

## Benchmarks

See [`BENCHMARKS.md`](../../benchmarks/README.md) for performance data.

## Validation Status

See [`VALIDATION_REPORT.md`](../../benchmarks/stdlib_validation/VALIDATION_REPORT.md) for test coverage.

## Modules

| Module | Description |
|--------|-------------|
| [`v1`](./v1.sio) | Stable Sounio-native workflow facade |
| [`regression/linear`](./regression/linear.sio) | Linear regression with epistemic tracking |
| [`regression/logistic`](./regression/logistic.sio) | Logistic regression |
| [`regression/robust`](./regression/robust.sio) | Robust regression methods |
| [`descriptive`](./descriptive.sio) | Descriptive statistics |
| [`distributions`](./distributions.sio) | Probability distributions |
| [`inferential`](./inferential.sio) | Hypothesis testing |
| [`bayesian/`](./bayesian/) | Bayesian methods |
| [`epistemic/`](./epistemic/) | Epistemic statistics (bootstrap, assumptions) |
| [`multivariate/`](./multivariate/) | PCA, clustering |
| [`timeseries/`](./timeseries/) | ARIMA, Kalman filter |
| [`clinical/`](./clinical/) | Meta-analysis, power analysis |

## License

MIT / Apache-2.0 (same as Sounio)
