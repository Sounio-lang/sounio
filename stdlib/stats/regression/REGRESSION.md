# Simple OLS and influence contract

## Conditional weighted coefficient inference

The separate weighted_linear_inference API adds original-order covariance,
standard errors, Student p-values and pointwise intervals when the caller
explicitly declares independent Gaussian relative-precision errors with an
unknown common scale. Separate availability statuses preserve the point fit
when inference is unavailable; the variance model remains unassessed.
Confidence support is[0.5,0.999999]. Final Forge15430:90/90.
See [full contract](WEIGHTED_INFERENCE.md) and
[receipt](../../../docs/plans/STATS_WEIGHTED_INFERENCE_EXECUTION_2026_10_19.md).
The point-fit APIs below keep their numerical-only contracts.


## Separate checked multivariable route

`weighted_multiple_linear_fit(design,response,weights,n,p,rcond)` provides
positive-weight numerical fitting through the shared weighted QR kernel.
Weights belong to the objective; no variance model, frequency interpretation,
SE, p-value or confidence interval is assessed. It reports raw residual norm
and the separate residual norm normalized by max(weight).
See [weighted contract](../../linalg/WEIGHTED_LEAST_SQUARES.md) and
[86/86execution receipt](../../../docs/plans/STATS_WEIGHTED_QR_EXECUTION_2026_10_19.md).

`stats::regression::multiple::multiple_linear_fit` borrows a fixed2048cell
design and256response, with physical row stride8 and active dimensions
1<=p<=8,p<=n<=256. Supply an explicit column of ones for an intercept.
It returns the shared checked column-pivoted Householder QR result:
coefficients, fitted/residual arrays, residual norm, rank, algebraic df,
pivot labels and explicit statuses. Standard errors, p-values and confidence
intervals are not supplied by this route.

See [least-squares contract](../../linalg/LEAST_SQUARES.md) and
[83/83native execution receipt](../../../docs/plans/STATS_MULTIVARIABLE_QR_EXECUTION_2026_10_18.md).
The simple OLS/influence contract below remains scoped to its original APIs.

Implementation: stats::regression::ols; compatibility API: stats::regression::linear.
This is bounded simple regression with an intercept and one predictor, not general
multiple regression. Numerical acceptance is recorded separately in the sprint
execution receipt. Unassessed assumptions are not evidence.

## Inputs and computation

- Fit: 3 through 100 finite paired observations. Influence: 4 through 100.
- Borrowed fixed arrays; only the first n entries are used.
- Anchored differences, scale normalization, compensated centered sums and a
  scaled residual norm avoid normal-equation determinants and raw-unit squares.
- Unrepresentable differences, coefficients or nonzero standard errors produce
  an explicit numeric failure. This is not a guarantee for every finite f64 input.
- Slope inference uses Student t with n-2 degrees of freedom and two-sided tails.
  The 95% interval is conditional on the ordinary regression assumptions.
- Constant response has undefined R-squared; perfect residual fit has unavailable
  Student inference. Coefficients can remain available in both cases.
- fit_simple_linear_regression retains value-array arguments for compatibility;
  callers concerned with copies should use the borrowed ols_fit_simple API.

## Actual case deletion

Each observation is excluded and the regression is fitted again. With residual e,
hat diagonal h, and deleted residual standard deviation s_deleted:

    DFFITS = (e / s_deleted) * sqrt(h) / (1-h)
    externally_studentized_residual = (e / s_deleted) / sqrt(1-h)
    Cook_D = 0.5 * ((e / s_full) * sqrt(h) / (1-h))^2

A deleted singular design, h=1, zero deleted residual scale, or nonrepresentable
statistic has a per-row status. No epsilon is substituted for these denominators.
Cook validity and DFFITS validity are independent. Overall partial status means
some rows require inspection, not that their zero-valued storage is a statistic.

The cutoff 2*sqrt(2/n) is a recorded screening convention for two fitted
parameters. Crossing it is not a p-value, an assumption test or permission to
delete a sample. The scientific consumer labels the lines as screening.

Time is O(n) for a fit and O(n^2) for influence, with n capped at 100. Actual
deletion trades bounded work for resistance to subtractive cancellation in
SSE - e^2/(1-h). No runtime or memory speedup has been measured for this batch.
There is no heap allocation inside these kernels.

## Status codes

| Code | Meaning |
|---|---|
| 0 | Available |
| 1 | Invalid sample size or inactive array slot |
| 2 | Nonfinite input |
| 3 | Rank-deficient predictor |
| 4 | Zero residual scale; inference unavailable |
| 5 | Nonrepresentable intermediate/result |
| 6 | Influence partially available |
| 7 | Hat diagonal equals one |
| 8 | Deleted residual scale equals zero |

Check fit status before coefficients, inference status before t/p/F/CI, and
per-row status before influence. Inactive fields use finite zero storage and
inactive p-values use one. Those values are sentinels.

## Compatibility and assessment

LinearModel adds fit_status, inference_status and r_squared_defined.
ModelDiagnostics adds influence status, per-row statuses and screening metadata.
RegressionAssumptions adds assessment_status (1 unassessed, 2 invalid fit).
KnowledgeLinearModel adds variance_status: code 5 marks a squared standard error that cannot be represented, while the standard error remains available.
These record layout changes require downstream literal/ABI review.

The old data-independent DFFITS array, fixed 1.96 interval and heuristic
Shapiro/Breusch-Pagan/Durbin-Watson confidence claims are removed. Their existing
BetaConfidence fields and overall confidence hold Beta(1,1) compatibility
sentinels, not calibrated probabilities. Normality, linearity, variance equality
and serial independence have not been tested. VIF=1 applies only to the single
predictor in a valid fit; the intercept VIF is inactive.

The C++23 reference uses reorthogonalized two-column QR and actual deletion
prediction differences, independently of the Sounio centered-sum calculation.
The focused regression fixture and figure must execute natively on Slurm;
source presence and reference generation alone do not establish acceptance.
