# Public t-test contracts

Import from `stats::inferential`:

- `t_test_one_sample(data, n, mu0)`
- `t_test_independent(x, nx, y, ny)`: pooled equal-variance model.
- `t_test_welch(x, nx, y, ny)`: unequal variances; fractional df retained.
- `t_test_paired(x, y, n)`: one-sample inference on paired differences.

The legacy array signatures accept at most 100 observations per sample.
Valid sample sizes are 2..100. These are bounded adapters, not a streaming
statistics API. No rows are silently dropped.

## Result semantics

`TTestResult` exposes estimate `mean_diff`, `se_diff`, `df`, `t_stat`,
`p_value`, `ci_lower`, `ci_upper`, `cohens_d`, `confidence_level`,
`method`, `n_x`, `n_y`, and `status`.

Numerical fields are meaningful only when status is T_TEST_OK.
For one sample the estimate and CI are centered on mean(data)-mu0.
For independent samples they describe mean(x)-mean(y).
For paired samples they describe the mean paired difference.

CI confidence is currently fixed at 0.95. Critical values come from the
Student quantile at the actual df, including the analytic Cauchy case df=1.
These are t-model intervals, not bootstrap or distribution-free intervals.
P-values are bilateral Student probabilities.

Method constants: T_METHOD_ONE_SAMPLE, T_METHOD_POOLED, T_METHOD_WELCH,
T_METHOD_PAIRED. For paired inference, n_x and n_y record the number of pairs
in each input. n_y is zero for one-sample inference.

The effect-size denominator is sample SD for one sample, pooled SD for pooled
inference, sqrt((variance_x+variance_y)/2) for Welch, and SD of paired differences
for paired inference (standardized paired difference, often called d_z).
These standardizations are not interchangeable.

## Invalid results

| Status | Meaning |
|---|---|
| T_TEST_OK | All reported quantities passed numerical checks |
| T_TEST_INVALID_SAMPLE | Sample length outside 2..100 |
| T_TEST_NONFINITE | Input observation or null mean is non-finite |
| T_TEST_ZERO_VARIANCE | The required standard error/effect scale is zero |
| T_TEST_NUMERIC_FAILURE | Intermediate/output overflow or another detected numerical failure |

Moments use Welford accumulation. Welch df use normalized variance weights,
avoiding fourth-power scaling and an arbitrary absolute epsilon.
Underflow/overflow and arbitrary extreme inputs are not fully certified by the
sampled validation. A 0.95 interval does not establish the model assumptions.

## Evidence and consumer

See `docs/plans/STDLIB_MULTIDOMAIN_EXECUTION_2026_10_17.md`.
The C++23 reference integrates Student density independently of incomplete beta.
`tests/run-pass/stats_inference_forest.sio` feeds the result into explicit
confidence bounds in the shared Canvas/SVG forest path.
