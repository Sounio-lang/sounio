# Stats

## Checked dynamic scientific workflows

The borrowed count-model APIs provide fixed-alpha fitting and information,
ML dispersion profiling, fixed-alpha bootstrap and
[ML profile-refit bootstrap](regression/DYNAMIC_COUNT_PROFILE_BOOTSTRAP.md).
The latter reestimates coefficients and dispersion under both models in every
replicate, preserving explicit failure, precision and work-budget diagnostics.

[Dynamic HC3 and single-way cluster covariance](regression/DYNAMIC_ROBUST_COVARIANCE.md)
uses caller-owned buffers above the legacy fixed-size limits. These are
computational covariance and contrast outputs; availability does not certify
finite-sample inference or scientific assumptions.

[Count dataset preparation](../genomics/COUNT_DATASET.md) connects keyed raw
integer counts, normalization offsets and reusable per-gene fitting to
[complete scientific documents](../viz/SCIENTIFIC_DOCUMENT.md). Numerical
concordance, empirical calibration, resource measurements and merge status
are separate evidence boundaries.


## Overview

Comprehensive statistics module: distributions, inference, regression (linear/logistic/robust), Bayesian analysis, causal inference, multivariate methods, and timeseries.

## Epistemic Differentiators

- Assumption checks returned as [`BetaConfidence`](../epistemic/knowledge.sio) values
- [`LinearRegression::builder()`](./regression/linear.sio) with epistemic fit tracking VIF, Cook's D, leverage
- Uncertainty propagation in bootstrap resampling and MCMC
- [`KnowledgeLinearModel`](./regression/linear.sio) wraps results with provenance

## Quickstart

```sio
use stats::regression::linear::LinearRegression;

// Builder pattern with method chaining
let model = LinearRegression::new()
    .with_data(x, y, n)
    .fit_epistemic();

// Access provenance
assert(model.provenance.method == "OLS");
assert(model.provenance.n_obs == n);

// Predict with uncertainty
let y_pred = model.value.predict(5.0);
```

## Benchmarks

See [`BENCHMARKS.md`](../../benchmarks/README.md) for performance data.

## Validation Status

See [`VALIDATION_REPORT.md`](../../benchmarks/stdlib_validation/VALIDATION_REPORT.md) for test coverage.

## Modules

| Module | Description |
|--------|-------------|
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
