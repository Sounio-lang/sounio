# Conditional weighted linear inference

Module: `stats::regression::weighted_inference`.

## Entry point and design

```sounio
pub fn weighted_linear_inference(
    design: &[f64;2048], response: &[f64;256], weights: &[f64;256],
    m: i64, p: i64, rcond: f64, confidence: f64,
    assume_gaussian_precision: bool
) -> WeightedLinearInference with Mut, Div, Panic
```

The physical row stride of the design is **8**. Valid dimensions are
`1 <= p <= 8`, `p <= m <= 256`. Only active rows and columns are read.
Include a column of ones explicitly for an intercept. No intercept,
centering term, sample exclusion, or variance model is inferred.

The fit uses the checked positive-weight QR least-squares path. Active design
and response entries must be finite; active weights must be finite and
strictly positive. The numerical rank must equal p for a successful fit.
The rank cutoff follows the weighted solver's contract: rcond must be finite
and in [0,1); zero requests its dimension-dependent default.
See [the weighted solver contract](../../linalg/WEIGHTED_LEAST_SQUARES.md).

The result preserves coefficients and fit metadata; rowwise fitted/residual
arrays remain in the point-fit API. The inference result adds:

- `covariance: [f64;64]`, with physical row stride 8.
- `standard_errors, t_statistics, p_values, ci_lower, ci_upper: [f64;8]`.
- `covariance_status, inference_status, calibration: i64`.
- `level: f64`.
- `variance_model_assessed: bool`, which remains false.

Only the active coefficient block has a statistical meaning. Inactive
covariance rows/columns and inactive coefficient-inference entries are zero.
Read the appropriate availability status before interpreting any array entry.

## Model declaration

The flag `assume_gaussian_precision=true` declares the model

```text
y = A beta + error
error_i independent Gaussian with mean 0 and variance sigma^2 / w_i
sigma^2 unknown; relative precisions w_i supplied by the caller
```

This is a declaration by the caller. The implementation does not assess
Gaussian errors, independence, linear model adequacy, or whether the weights
are proportional to true precisions. It does not turn arbitrary objective,
survey, importance, reliability, or frequency weights into validated
inverse-variance weights. Estimated weights add uncertainty that this
conditional model does not account for.

With the flag false, the numerical fit is still returned. Covariance and
inference remain unavailable. This preserves the separation between
minimizing a weighted criterion and making a statistical sampling claim.
[NIST's WLS discussion](https://www.itl.nist.gov/div898/handbook/pmd/section1/pmd143.htm)
describes relative weights and the role of inverse error variances;
the [WLS criterion](https://www.itl.nist.gov/div898/handbook/pmd/section4/pmd432.htm)
treats the supplied design and weights as fixed quantities.

## Mathematical quantities

For a successful full-rank fit with `df = m-p > 0`, let W=diag(w).
Under the declared model,

```text
SSE_w       = sum_i w_i (y_i - fitted_i)^2
sigma_hat^2 = SSE_w / df
Cov_hat     = sigma_hat^2 (A' W A)^(-1)
SE_j        = sqrt(Cov_hat[j,j])
t_j         = beta_hat_j / SE_j
p_j         = two-sided Student tail at |t_j|, with df degrees of freedom
CI_j        = beta_hat_j +/- t_(1-(1-confidence)/2, df) * SE_j
```

These formulas specify the quantities; they are not an instruction to
construct or invert normal equations numerically. The implementation uses
the checked QR factors and scaling metadata. Replacing every weight by the
same positive factor leaves the coefficients and this unknown-scale
coefficient covariance unchanged in exact arithmetic.

The tests are for each coefficient's null value **zero**. The intervals are
**pointwise coefficient confidence intervals**, conditional on the declared
model. They are not simultaneous intervals, prediction intervals, posterior
credible intervals, or confidence that the variance model is correct.
NIST's [linear and multiple-linear fitting reference](https://itl.nist.gov/div898/software/dataplot/refman1/auxillar/line_fit.htm)
distinguishes parameter inference and joint intervals from other regression
quantities.

The accepted confidence range is **[0.5, 0.999999], inclusive**.
Finite-precision Student tails and critical values remain numerical
evaluations, with the separately bounded native validation documented in
the sprint receipt.

## Availability and error precedence

The numerical fit's `status` follows the weighted solver:

| Fit status | Meaning |
|---|---|
| 0 | Available full-rank fit |
| 1 | Invalid dimensions |
| 2 | Nonfinite active design or response |
| 3 | Numerical rank deficiency |
| 4 | Arithmetic failure |
| 5 | Invalid rcond |
| 6 | Invalid active weight |

The fit is attempted before model-dependent work. Fit failure returns
covariance_status=4, inference_status=4, calibration=0 and level=0.

| covariance_status | Meaning |
|---|---|
| 0 | Covariance and SE arrays available |
| 1 | Gaussian precision model not declared |
| 2 | No residual degrees of freedom |
| 4 | Fit or covariance arithmetic failure |

| inference_status | Meaning |
|---|---|
| 0 | Student statistics, tails and intervals available |
| 1 | Gaussian precision model not declared |
| 2 | No residual degrees of freedom |
| 3 | Degenerate zero standard error; Student inference unavailable |
| 4 | Fit, covariance or inference arithmetic failure |
| 5 | Unsupported or nonfinite confidence request |

After a successful fit, opting out takes precedence over checking confidence
or residual degrees of freedom. When the model is declared, absent residual
degrees of freedom takes precedence over an invalid confidence request.
An invalid confidence request does not discard a successfully computed
covariance or its standard errors.

`calibration=1` records a requested conditional Gaussian-precision calibration
after a successful fit. It does not by itself assert that inference is
available. `calibration=0` applies to opt-out and failed fits.
`level` equals the requested confidence only when inference_status=0;
otherwise it is zero. `variance_model_assessed` is always false.

All unavailable t, p and interval arrays are zero, including inactive entries.
A covariance failure clears the covariance and SE arrays. An inference-only
arithmetic failure retains available covariance/SE but clears t, p and CI.
A finite positive SE alone does not guarantee representable intervals: a
nonpositive half-width or rounded bounds with lower >= upper returns
inference_status=4 while retaining available covariance/SE.
A perfect zero-response fit can have available zero covariance and SE while
Student inference is unavailable: the zero payloads must not be read as
p=0 or as zero-width certified intervals.

## Numerical scaling and validation boundary

The shared six-factor scaling helper balances finite factors before restoring
the final magnitude. It accepts finite signed values, nonnegative multipliers
and strictly positive divisors. Invalid inputs or overflow return status4;
final binary64 underflow to zero is allowed by that helper, with callers
deciding whether zero makes their own output unavailable. A caller must still
read its result status; the helper is not a correctly-rounded product proof.

Successful factor output has inactive and lower-triangular scratch slots
cleared. Factor/scaling buffers are available only when their fit status is
zero; buffers left after failure must not be interpreted as valid factors.

The inference implementation is not a correctly-rounded transcendental
library or a proof of coverage under violated assumptions. It supplies no
heteroscedasticity-robust sandwich covariance, autocorrelation correction,
clustered inference, errors-in-variables model, or uncertainty for estimated
weights.

`tests/run-pass/stats_weighted_inference_status.sio` covers availability,
precedence, inactive poisoned entries, invalid requests, degeneracy,
covariance overflow, factor-output padding and scalar restoration edges.
The synthetic consumer `tests/run-pass/viz_weighted_inference.sio` renders
conditional Student coefficient intervals through production Canvas/SVG,
with their model declaration visible. Independent C++23 references and Forge
execution determine acceptance; source and documentation alone are not
evidence of native execution or broader statistical calibration.
