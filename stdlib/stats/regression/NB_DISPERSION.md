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
| 5 | arithmetic failure or nonpositive aggregate mu^2 weight |
| 6 | reserved defensive residual-df failure; unreachable after the current status-1 validation |

Precedence is that order. Failures keep n and the declared parameter count
when those inputs were read, and clear the numerical payload.
`nb_assumptions_assessed` is always false.

## Binary64 arithmetic policy

A positive mean whose squared value rounds to zero does not independently
invalidate the estimate. The sum of the rounded squared means must still be
positive and finite. This admits mixed-scale inputs when other rows provide
positive aggregate weight. It does not recover squared means that all round
to zero before summation, even if their exact mathematical sum would be
representable. Such inputs still return status 5.

For a zero count, the Pearson contribution is evaluated directly as `mu`,
using the identity `(0-mu)^2/mu = mu`. This preserves a representable positive
contribution when `mu*mu` underflows. Positive counts retain `square/mu`.
The public moment, squared-mean weight, Pearson sum and dispersion must each
be finite under the existing bounded arithmetic policy. The function does not
promise availability for every mathematically representable final ratio.

The NB2 variance convention is `V(mu) = mu + alpha*mu^2`, as also documented
in the [statsmodels variance-family reference](https://www.statsmodels.org/stable/generated/statsmodels.genmod.families.varfuncs.NegativeBinomial.html).
That reference supplies the parameterization, not a validation of this
moment estimator or its declared degrees-of-freedom adjustment.

## Limits

Counts convert to binary64. Integers above 2^53 can round. This is not an
exact integer-contrast estimator, a Cox-Reid adjusted likelihood, a DESeq2 or
edgeR shrinkage estimator, a standard error, or an RNA-seq significance test.
Independence, mean specification, and exposure adequacy are unassessed.
