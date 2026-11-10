# Checked Jarque-Bera normality summary

`stats::normality::normality(&x, n)` accepts 2 <= n <= 256 finite f64
observations. It computes population central moments (divisor n), skewness S,
Pearson kurtosis K, excess K-3 and JB = n/6*(S*S+(K-3)^2/4).
The returned p is exp(-JB/2), the **asymptotic** chi-square(2) survival.
Small samples can have a computable statistic without calibrated finite-sample
inference. No sample-size threshold establishes that calibration here.
Independent, identically distributed sampling is not assessed. Regression
residuals require their model/sampling assumptions to be considered separately.
No p-value is converted into a Beta posterior or probability of normality.

## Availability

Always read `status` before using the numerical payload:

| status | Meaning |
|---|---|
| 0 | Statistic and asymptotic reference tail available |
| 1 | n outside 2..256 |
| 2 | Nonfinite input in the selected prefix |
| 3 | Constant observations, so standardized moments are undefined |
| 4 | Arithmetic failure |

Unavailable numerical fields are zero sentinels, including p; neither zero nor
one is an inference when status is nonzero. Metadata `n` echoes the request,
and `asymptotic=true` identifies the reference convention, not validity.
A representable p can underflow to zero for an extreme statistic.
Only the first n entries are read. One fixed 256-value scratch array and scalar
accumulators are used, with no dynamically allocated sample-sized buffers.

## Numerical policy

Finite input is checked before accumulation. Anchor differences preserve small
representable contrasts at large offsets. When subtracting opposite-sign extremes
overflows, divide by the maximum absolute input before subtracting. Scale the
centered differences again before forming powers, and use compensated sums for
the mean and second, third and fourth moments. This prevents fourth-power
overflow and tiny-unit moment underflow. It cannot recover variation already
rounded away in the caller's input.
The nonpositive exponential is evaluated by range reduction and repeated
squaring; it never forms an overflowing positive reciprocal.

## Compatibility and scope

The existing function, original five fields and inline tests remain. Status,
sample size and asymptotic metadata are added. Code constructing the public
record directly must add these fields; no such source consumer was found.
Callers that previously used p without checking status must migrate.

This API is Jarque-Bera. Shapiro-Wilk, Anderson-Darling, Levene and the legacy
epistemic p-to-Beta heuristics are not certified by this work.
The subsequent checked Koenker heteroscedasticity and simple-OLS residual
consumer are documented separately in [HETEROSCEDASTICITY.md](HETEROSCEDASTICITY.md).
They retain independent procedure statuses and do not certify the private heuristics.

Independent reference: `tests/reference/normality_reference.cpp`, C++23
long-double raw central moments and standard-library pow/exp, with 12 datasets.
Candidate uses scaled f64 moments; the oracle does not reuse that algorithm.
Native acceptance and the controlled old-body failure are recorded in
`docs/plans/STATS_NORMALITY_EXECUTION_2026_10_18.md`.
