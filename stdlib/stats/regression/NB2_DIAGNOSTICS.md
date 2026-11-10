# Supplied-parameter NB2 count diagnostics

## Scope and evidence state

Implementation entry point:
`stats::regression::nb2_diagnostics::nb2_count_diagnostic`.

This scalar kernel compares Poisson and NB2 Pearson residual scaling for a
supplied nonnegative count, positive count mean and nonnegative dispersion.
NB2 uses the convention

    Var(Y) = mu + alpha * mu^2

This mean/dispersion parameterization is also stated in the official
[statsmodels NegativeBinomial source documentation](https://www.statsmodels.org/stable/_modules/statsmodels/genmod/families/family.html#NegativeBinomial).
That source supplies the parameterization, not validation of our arithmetic,
moment estimator or diagnostic implementation.

The kernel does not estimate mu or alpha, fit an NB likelihood, calculate
deviance, shrink dispersion, test a hypothesis, construct a confidence interval,
or assess a statistical model. A supplied parameter is not an assessed parameter.

Current-source typecheck and native demonstration execution are complete:
Slurm15783 completed0:0 after rebuilding the seed and modular Madaros.
Four targets returned TYPECHECK_OK: nb2_diagnostics, nb_dispersion, poisson and
the count diagnostic example. The example compiled and executed with exit0,
including its in-memory Canvas render and SVG emission. This is demonstration
execution, not numerical concordance or a Canvas pixel comparison.

The root extracted the actual HTML/SVG, validated the SVG using xmllint (exit0),
converted the SVG with sips to a1380×620 PNG and inspected the full figure.
Both panels, two residual colors, legends, units and negative labels were
visible without clipping. A clipped QuickLook preview was not used as the
accepted full-figure inspection.

No tests were added or run. Independent numerical concordance, boundary tests,
Canvas pixel/geometry comparison and repository CI/merge remain pending.
The historical acceptance of earlier kernels or renderers does not extend those
pending claims. Exact evidence:
[NB2 diagnostics execution receipt](../../../docs/plans/NB2_DIAGNOSTICS_EXECUTION_2026_10_21.md).
Proof, implementation, typecheck, execution and scientific acceptance remain
different claims.

## Public API

```sounio
pub struct Nb2CountDiagnostic {
    pub status:i64,
    pub variance_status:i64,
    pub variance:f64,
    pub poisson_pearson_residual:f64,
    pub nb2_pearson_residual:f64,
    pub poisson_limit:bool,
}
pub fn nb2_count_diagnostic(
    count:i64, mean:f64, alpha:f64
)->Nb2CountDiagnostic with Mut,Div,Panic
```

The implementation uses scalar storage and no heap allocation.

## Input and statuses

The supported count range is0..9007199254740992 inclusive. This limit ensures
the accepted i64 count is represented exactly as binary64; larger counts are
rejected rather than silently rounded. Mean must be finite and strictly
positive. Alpha must be finite and nonnegative. Both+0 and-0 alpha select the
Poisson branch.

| status | Meaning |
|---|---|
| 0 | Both Pearson residual payloads are available |
| 1 | Negative count |
| 2 | Nonfinite or nonpositive mean |
| 3 | Nonfinite or negative alpha |
| 4 | Count exceeds the supported lossless-conversion range |
| 5 | Residual arithmetic unavailable |

The public status constants are NB2_DIAG_OK, NB2_DIAG_NEGATIVE_COUNT,
NB2_DIAG_INVALID_MEAN, NB2_DIAG_INVALID_ALPHA, NB2_DIAG_INEXACT_COUNT and
NB2_DIAG_ARITHMETIC_UNAVAILABLE, in the numeric order above. Validation precedence
is negative count, invalid mean, invalid alpha, then count too large.

On a nonzero status, variance and both residual fields are zero,
variance_status is2, and poisson_limit is false. These zeros are unavailable
payloads, not observed zero residuals or a zero variance.

Variance availability is separate when status is0:

| variance_status | Meaning |
|---|---|
| 0 | Staged binary64 variance is finite and positive |
| 1 | Staged variance overflowed or produced an invalid result; variance is0 |
| 2 | Not computed/available because the overall diagnostic failed |

The corresponding public constants are NB2_VARIANCE_AVAILABLE,
NB2_VARIANCE_ARITHMETIC_UNAVAILABLE and NB2_VARIANCE_NOT_COMPUTED.

Thus status0 with variance_status1 is a valid outcome: residual scaling can
remain available even when the squared-unit variance cannot be represented.
Callers must inspect both statuses and must not display variance0 as a valid
NB2 variance when its status is1 or2.

## Arithmetic contract

Let y be the lossless binary64 conversion of count and delta=y-mu.

The Poisson Pearson residual is

    delta / sqrt(mu)

For alpha equal to zero, the NB2 residual is copied directly from the Poisson
residual. The two output fields are bit-identical in this branch, and
poisson_limit is true. The flag denotes this explicit input branch; it is not
set merely because a small positive alpha happens to round to a Poisson-like
result.

For positive alpha, the NB2 residual is evaluated by two scaled forms:

    mu >= 1: (delta/mu) / sqrt(alpha + 1/mu)
    mu < 1:  (delta/sqrt(mu)) / sqrt(1 + alpha*mu)

These forms express delta/sqrt(mu+alpha*mu²) without first requiring the raw
squared-unit variance to be representable. Divisors and square-root arguments
must be finite and positive; unavailable residual arithmetic returns status5.
The canonical square root is used. This is an implementation arithmetic policy,
not a theorem or a claim of correctly rounded residuals for all binary64 inputs.

The separately reported variance follows the explicit staged binary64 order:

    alpha_mu = alpha * mu
    extra = mu * alpha_mu
    variance = mu + extra

A finite positive staged result is reported with variance_status0. An invalid or
overflowed result is reported with variance_status1 and variance0, without
discarding otherwise available residuals. Underflow of an extra contribution to
zero is accepted: it is not independently treated as a diagnostic failure.

Staging and rounding matter near representability boundaries. The variance is
NOT claimed to be the correctly rounded value of the exact real expression,
and variance_status0/1 is NOT an if-and-only-if test of whether that exact
expression is representable. Nor does this policy recover every contribution
lost through ordinary binary64 rounding.

## Scientific interpretation and consumer contract

Both residuals are descriptive transformations of a supplied count and mean.
Poisson scaling uses variance mu; NB2 scaling uses the supplied alpha convention.
A residual nearer zero after increasing alpha does not establish a better model,
a calibrated uncertainty interval or an accepted dispersion estimate.

The current count diagnostic example can compare these two residuals using
Poisson means fitted with exposure offsets and a plug-in nonnegative moment
dispersion estimated from the same observations. It must label that plug-in
construction, retain the signed moment estimate/truncation report separately,
and avoid claiming an NB maximum-likelihood fit. Means and dispersion derived
from the same data do not acquire independent calibration from this comparison.

Mean is an expected COUNT, not count/exposure rate. If exposure-adjusted rates
are plotted elsewhere, count means must still be supplied here. Pearson
residuals are dimensionless. Any visualization of mean±sqrt(variance) is sampled
standard-deviation geometry, not a CI or a claim of NB predictive coverage.
An unavailable variance must not be used to draw such a mark.

Independence, count likelihood, mean specification, exposure adequacy and
dispersion-model assumptions are unassessed. There is no deviance output,
likelihood-ratio statistic, p-value, interval or shrinkage procedure in this API.

## Executable identity and bounded resource observation

Frozen source commit:4a65e767af7ce955cfde85f25484ea7f8a5f44f2.
Compiler SHA256:
0c896942018ff4be9a86d69f30e3a8aadac955f32451c445a07fbf502ae37fc3.
Program SHA256:
694a9bab24cde523bced34c36a67d5070635b601c04da575cc22ee1458fc8a13.
Although the compiler bytes match a previously observed hash, this checkpoint
rebuilt the current-source seed/Madaros; it is not an unverified reuse claim.

Exact-ELF profiling Slurm15785 completed0:0. GNU time recorded elapsed0.01s and
maximum RSS8748KiB for the single eight-row demonstration. This observation
includes the whole native program, including scene construction/Canvas work/SVG
output. It is not a scalar-kernel benchmark, comparative performance study or
general memory/runtime guarantee.

## Continuation

Independent numerical agreement and boundary behavior require their own
authorized evidence. No correctness result is inferred from a formula, a clean
typecheck, a successful demonstration or a legible figure. Canvas execution is
recorded, while pixel comparison remains pending. No tests or full corpus were
run for this checkpoint. The broader statistics, genomics, mathematics, physics
and execution/resource sprint remains active.
