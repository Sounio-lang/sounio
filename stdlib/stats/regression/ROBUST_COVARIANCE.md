# HC3 covariance for checked multiple OLS

Module: `stats::regression::robust_covariance`.

## Public API

`multiple_linear_hc3(design:&[f64;2048],response:&[f64;256],
n:i64,p:i64,rcond:f64)->LinearHc3Result` declares Mut,Div,Panic.
The design has physical row stride8. Dimensions, active-input validation,
rcond policy, explicit caller-supplied intercept and coefficient fit are
inherited from the checked QR kernel. It supports1<=p<=8 and p<=n<=256.

LinearHc3Result contains:

* fit:LeastSquaresResult, the original checked OLS fit including its status,
  coefficients, reported fitted values, residuals and rank metadata;
* covariance_status:i64, covariance:[f64;64] with physical stride8,
  standard_errors:[f64;8], leverage:[f64;256];
* method:i64, always3 identifying HC3, including errors;
* assumptions_assessed:bool, alwaysfalse.

Inactive successful covariance/SE/leverage entries remain zero. This result
contains covariance and standard errors only. It supplies no t statistics,
p values, confidence intervals or calibrated degrees of freedom for inference.

## Mathematical target

For an unweighted full-column-rank design X, OLS residuals u and
B=(X'X)^(-1)X', the HC3 estimator is

```text
h_i = x_i' (X'X)^(-1) x_i
a_i = B[:,i] u_i/(1-h_i)
V_HC3 = sum_i a_i a_i'
SE_j = sqrt(V_HC3[j,j]).
```

There is no additional HC1 factor n/(n-p). The OLS coefficients are unchanged.
HC3 is a heteroscedasticity-consistent covariance construction; it does not
make the coefficient estimator resistant to outliers.

## QR computation and scales

One existing qr_least_squares_factor call returns the OLS fit plus the
upper-triangular R and original-column maximum-absolute scales S.
The factorization is (X S^(-1))P=Q R.

For each row, form z=P' S^(-1) x_i, solve R' q=z, and compute h_i=q'q.
Then solve R v=q and restore the original coefficient influence as
S^(-1) P v times u_i/(1-h_i). Both triangular dot sums are compensated.
No X'X, normal-equation solve, explicit inverse or explicit Q is constructed.

The influence scaling uses the shared balanced six-factor product to avoid
premature product/ratio overflow and underflow. An influence Gram buffer
stores at most256 rows by8 coefficients. Standard errors use LASSQ norms.
Covariance uses compensated normalized Gram products and balanced restoration,
without directly squaring large or tiny raw residuals.

## Availability statuses and failure payload

| covariance_status | meaning |
|---|---|
| 0 | covariance/SE/leverage available under the numerical policy |
| 1 | underlying fit unavailable; inspect fit.status |
| 2 | n<=p, insufficient residual degrees of freedom |
| 3 | computed leverage reaches or exceeds1; HC3 denominator unavailable |
| 4 | detected arithmetic, representability or factor-metadata failure |

The original fit is retained independently, including a valid fit when its
covariance is unavailable. On every covariance failure all covariance,
standard-error and leverage payloads are zero. Metadata method3 and
assumptions_assessedfalse remain. Fit validation/rank failure takes precedence
over residual degrees of freedom or leverage.

Leverages are evaluated from rounded QR factors. No leverage is clipped or
replaced by an epsilon, and no zero residual is used to define0/0 when the
computed leverage is1. Values extremely near1 are ill conditioned; rounding
can dominate1-h and HC3. This implementation does not certify exact detection
of every mathematically unit-leverage row or uniform relative precision near1.

A fit whose REPORTED raw residuals are all exactlyzero has covariance0 and
standard errors0 if every computed leverage is below1. Perfect algebraic
data can still have nonzero reported residuals from floating-point fitting;
those actual reported residuals are propagated. No tolerance turns them into
zero. Zero covariance does not create an inferential interval or significance
claim. HC3 is unavailable for computed unit leverage even with zero residual.

## Representability boundary

A nonzero original design entry whose column normalization becomeszero returns4.
A nonzero triangular quotient that becomeszero returns4. A nonzero adjusted
influence that becomeszero returns4 even if other observations would dominate.
These are conservative contribution boundaries, not a statement that a
mathematical covariance does not exist. Arithmetic overflow also returns4.

Representable subnormal influences are permitted. Nonzero SE whose diagonal
covariance underflows tozero is unavailable with4; off-diagonal rounding tozero is permitted, including a normalized Gram dot
that underflows or cancels tozero before restoration. Tiny terms in norm/Gram/leverage reductions can round
away. This is not a correctly-rounded covariance certificate, and it does not
promise recovery of every representable aggregate from individually tiny terms.

The residual measurand is the checked QR fit's reported
response-minus-reported-fitted binary64 value. It is not a residual computed
from an unrounded ideal solution. Original design units, column pivoting and
coefficient units are restored separately; scaling a column changes coefficient
and covariance units accordingly.

## Scientific assumptions and scope

Use requires a scientifically defensible conditional mean model and the
independence conditions relevant to heteroscedasticity-consistent estimation,
with suitable finite moments and regularity for asymptotic interpretation.
The function does not assess those conditions. In particular it does not
correct dependence within subjects, families, sites or other clusters.
A clustered estimator would aggregate score vectors within each cluster
before forming its outer products and requires a separate contract.

The Gaussian relative-precision WLS inference module, this OLS HC3 estimator,
and the legacy Huber/Tukey point estimators are distinct statistical procedures.
This API does not validate the legacy M-estimator standard errors, estimate
unknown WLS precision weights, supply cluster corrections or guarantee
finite-sample coverage.

## Storage and execution boundary

All arrays are fixed; no heap allocation. HC3 scratch includes R64, scales8,
pivot-seen8, one2048-element influence buffer and small8-element row vectors.
The returned OLS fit/result arrays and nested QR scratch also contribute to
peak storage. No measured RSS, stack ceiling or performance superiority is
claimed from source inspection.

Source completion is not native numerical acceptance. Typechecking and
execution must be coordinated through Forge/Slurm. Independent numerical
reference work follows ADR009, using C++23 rather than Python/Rust/Julia.

## Native acceptance — 2026-10-19

The24-case independent C++23/quad reference and status/metamorphic consumers
pass in Slurm15625 within105/105, zero failures/skips/errors. The portable
linear_hc3_fixture.cpp emitter reproduces the reference fixture byte-for-byte.
An isolated HC0-weighting control compiles but fails the reference in15626.
These bounded cases do not establish inferential calibration or cluster support.
See docs/plans/HC3_COUNT_TSV_ACCEPTANCE_2026_10_19.md for receipts and limits.
