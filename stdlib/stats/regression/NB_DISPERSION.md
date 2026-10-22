# NB2 moment dispersion

Implementation: `stats::regression::nb_dispersion::nb2_moment_dispersion`.

This estimates the NB2 dispersion alpha in

    Var(Y) = mu + alpha * mu^2

from caller-supplied nonnegative counts and strictly positive means. It does
not estimate the means, fit a log-linear model, shrink gene-wise dispersions,
or test differential expression.

## Estimator

For active i = 0..n-1,

    raw = sum((y_i - mu_i)^2 - mu_i) / sum(mu_i^2)

    alpha_signed = raw * n / (n - parameters_declared)

`parameters_declared` is the caller's count of mean parameters already used.
Zero means the means are treated as supplied, not estimated here. The scale
n/(n-p) is a finite-sample adjustment, not a proof that the result is unbiased.
`alpha_nonnegative` is the projection of `alpha_signed` onto [0, infinity).
A negative signed value sets `truncated` and is underdispersion relative to
this moment equation, not a hypothesis test that alpha is zero.

`pearson_x2` is sum((y_i-mu_i)^2 / mu_i). Sums are compensated. Inactive
entries beyond n are not read.

## Status

| Status | Meaning |
|---|---|
| 0 | available |
| 1 | invalid n or declared parameter count |
| 2 | nonfinite active mean |
| 3 | active mean not strictly positive |
| 4 | negative active count |
| 5 | arithmetic failure or nonpositive mu^2 weight |
| 6 | residual degrees of freedom below 1 |

Precedence is that order. Failures keep n and the declared parameter count
when those inputs were read, and clear the numerical payload.
`nb_assumptions_assessed` is always false.

## Limits

Counts convert to binary64. Integers above 2^53 can round. This is not an
exact integer-contrast estimator, a Cox-Reid adjusted likelihood, a DESeq2 or
edgeR shrinkage estimator, a standard error, or an RNA-seq significance test.
Independence, mean specification, and exposure adequacy are unassessed.
