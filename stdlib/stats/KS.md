# Checked Kolmogorov–Smirnov consumers

## Public contract

`stats::ks_test` exposes:

```sounio
ks_one_normal(x: &[f64;256], n: i32, mu: f64, sigma: f64) -> KSResult
ks_two(x: &[f64;256], nx: i32, y: &[f64;256], ny: i32) -> KSResult
```

Both require `Mut, Div, Panic`. Each sample prefix has length 1..256.
Inactive array suffixes are ignored. Inputs are copied; callers' arrays are
preserved.

`KSResult` carries `d, p, ne: f64`, `status, n_x, n_y: i32`,
`asymptotic: bool` and `has_ties: bool`. Check status before interpreting
numerical payloads.

| Status | Meaning |
|---|---|
| 0 | Statistic and approximate calibration computed |
| 1 | Sample size outside 1..256 |
| 2 | Nonfinite observation in an active prefix |
| 3 | Invalid normal reference: nonfinite mu/sigma or sigma <= 0 |
| 4 | Numerical failure |

An error status is not evidence of distributional agreement. Payloads associated
with errors are unavailable, even where initialized to zero. Status 0 describes
the algorithm's completion; it does not certify the sampling model.

## Hypotheses and interpretation

The one-sample statistic compares an ECDF with a **fully specified** normal CDF.
The caller must supply reference mean and standard deviation fixed independently
of the tested sample. Passing its own sample mean and SD does not yield a valid
ordinary KS normality p-value: estimation changes the null distribution. Use an
appropriate fitted-normal procedure with its own calibration for that question.
See [NIST's KS definition and restrictions](https://www.itl.nist.gov/div898/handbook/eda/section3/eda35g.htm).

The two-sample statistic is the supremum absolute gap between the two ECDFs.
Under an independent continuous common null, its asymptotic scaling has effective
sample size `ne = nx*ny/(nx+ny)`; the one-sample value is `ne = n`.
Independence is a modeling requirement, not something these functions check.
See [Geyer's empirical-process explanation](https://www.stat.umn.edu/geyer/f07/5601/examp/kolmogorov.html).

Equal observations are processed as complete groups when evaluating the
two-sample gap. Thus the statistic remains meaningful for ties.
`has_ties` reports repeated values in the pooled observations, including values
shared across samples. **The returned calibration is still the continuous-null
approximation; it is not adjusted or certified for ties/discrete distributions.**
The flag does not perform a permutation test.

A large p is not the posterior probability that the null is true, nor a
certificate of equivalence. No confidence band is constructed.

## Calibration and arithmetic

The implementation evaluates the two-sided limiting Kolmogorov survival at

```
lambda = (sqrt(ne) + 0.12 + 0.11/sqrt(ne)) * D
```

This retains the module's Stephens-style effective-size correction.
It is an approximation, including for small sample sizes; no exact finite-n
distribution or finite-n error bound is supplied. Applying the correction with
two-sample effective size is an implementation policy, not an assertion of exact
two-sample calibration. The original methodological reference is
[Stephens (1970), JRSS B 32:115–122](https://doi.org/10.1111/j.2517-6161.1970.tb00821.x).

For small lambda, a positive theta-transformed CDF expansion avoids cancellation
in the alternating survival expansion. For larger lambda the implementation
sums the alternating survival terms directly. Both paths have a fixed 32-term
bound. At lambda <= 0.04 the CDF underflows binary64 and the returned survival
rounds to 1. Representable small survival probabilities use direct negative
exponentials. This numerical kernel does not turn the approximation into an exact
test or validate all other statistical consumers.

## Scientific figure

`tests/run-pass/viz_ks_ecdf.sio` is a real consumer of `ks_two`, Canvas, and
the SVG renderer. Its synthetic tied samples are `[0,0,1]` and `[0,1,1]`.
The expected statistic is 1/3. It plots complete ECDF steps, with vertical jumps
after each whole tie group and support tails visible beyond 0 and 1.
A vertical connector has length computed from the returned statistic.
The displayed D and p labels are formatted from the actual result.

The figure explicitly labels the synthetic size, the continuous-null asymptotic
calibration, and unassessed ties. Its p is illustrative metadata; this small tied
example does not validate inferential calibration. The consumer asserts scene
structure, the grouped-step gap, status/tie metadata, and a known Canvas axis
pixel, then emits SVG through the production scene renderer. Validation status
belongs to the sprint execution receipt; the existence of this source alone is
not execution evidence.

## Resource boundaries

Samples are bounded by fixed arrays. The two-sample merge visits ordered tie
groups rather than creating one coordinate array per observation. Scratch memory
is fixed by the public array bounds. No bootstrap, random generator, external
numerical library, or dynamic allocation is needed by the statistical core.
This documents implementation structure, not a measured speed/RSS improvement.
