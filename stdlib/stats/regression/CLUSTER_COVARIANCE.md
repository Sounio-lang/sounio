# Single-way clustered OLS covariance: CR0 and CR1S

Module: `stats::regression::robust_covariance`.
This is covariance-only estimation from a checked unweighted OLS fit.

## API

`multiple_linear_cluster(design:&[f64;2048],response:&[f64;256],
cluster_ids:&[i64;256],n:i64,p:i64,rcond:f64,method:i64)->LinearClusterResult`
declares Mut,Div,Panic. Physical design stride8; dimensions and fit validation
are inherited from qr_least_squares_factor:1<=p<=8, p<=n<=256.
The intercept is supplied explicitly. Only active design/response/label prefixes
are consumed. Arbitrary signed i64 labels, including negative/extreme values,
are compared by equality; numeric magnitude and ordering have no statistical role.

Result fields:

* fit:LeastSquaresResult, preserving the checked OLS fit and its own status;
* covariance_status:i64, covariance:[f64;64] with physical stride8,
  standard_errors:[f64;8];
* clusters:i64, method:i64, correction_factor:f64;
* assumptions_assessed:bool, alwaysfalse.

Method0 is CR0; method1 is the full Stata-style correction called CR1S here.
No other method number is accepted. No t/p/CI, cluster-adjusted inferential
degrees of freedom or hypothesis-test result is returned.

## Estimators and naming

For OLS observation influence b_i=(X'X)^(-1)x_i and the fit's REPORTED
response-minus-reported-fitted binary64 residual u_i, form

```text
s_g = sum_{i in cluster g} b_i u_i
V_CR0 = sum_g s_g s_g'
f_CR1S = [G/(G-1)] [(n-1)/(n-p)]
V_CR1S = f_CR1S V_CR0
SE_j = sqrt(V[j,j]).
```

Summation WITHIN a cluster occurs before its outer product. Summing individual
observation outer products would instead discard the required within-cluster
cross terms and implement a different estimator.

The full correction follows [Cameron and Miller (2015), formula12](https://cameron.econ.ucdavis.edu/research/Cameron_Miller_JHR_2015.pdf).
Naming is explicit because [clubSandwich's vcovCR documentation](https://jepusto.github.io/clubSandwich/reference/vcovCR.html)
uses CR1 for the G/(G-1) correction alone, and CR1S for the full factor above.
This API's method1 does not mean that shorter CR1 convention and does not
claim parity with every command's absorbed-effects/weighting conventions.

## QR path and accumulation

One checked factorization supplies (X S^(-1))P=Q R. Reuse the existing private
compensated triangular solves: for a row, z=P' S^(-1)x_i, R'q=z, Rv=q.
The original coefficient influence is S^(-1)P v times its reported residual.
Balanced scalar restoration avoids premature products/scale-ratio overflow.

Labels are mapped in first-appearance order with bounded equality lookup.
Row-to-group indices are retained. Coefficient-major cluster sums and
corrections each use a fixed2048-element array. Every coefficient's signed
within-cluster sum uses compensated accumulation.

For CR1S, scale each completed cluster vector by sqrt(f_CR1S) before its Gram
calculation. This avoids first requiring an uncorrected covariance diagonal
to be representable when the final corrected value could be representable.
The returned correction_factor is the full specified f, not its square root.
Rounding through sqrt(f) means this is not an exact-bit multiplication theorem.

HC3 and clustering share the final LASSQ-standard-error and normalized-Gram
helper. The HC3 arithmetic loop order is preserved by the extraction; this
extension adds no HC3 leverage clamp or changed correction. Cluster CR0/CR1S
does not divide by1-h and does not inherit HC3's unit-leverage rejection.

## Availability and precedence

| covariance_status | meaning |
|---|---|
| 0 | requested clustered covariance available |
| 1 | fit unavailable; inspect fit.status |
| 2 | n<=p |
| 3 | fewer than2 distinct active cluster labels |
| 4 | detected arithmetic/representability or factor-metadata failure |
| 5 | method not0 or1 |

Precedence: checked fit (including size/rcond/input/rank), method, residual
degrees of freedom, cluster count, arithmetic. Thus an invalid fit is returned
before examining method or labels.

Any unavailable covariance has covariance/SE entirelyzero and
correction_factor0. A successful fit remains available independently.
The requested method is retained even when invalid. clusters remains0 until
labels have been grouped; once known, G is retained as diagnostic metadata on
status3 or4. On success correction_factor is1 for CR0 and f_CR1S for CR1S.
Inactive covariance/SE slots remainzero.

Both methods require G>=2 and n>p. G<=p is permitted. Covariance can be singular
and generally has rank at mostG-1 when the OLS score sums cancel; status0
does not assert invertibility, positive-definiteness, or valid Wald inference.

## Numerical boundaries and exact zeros

The point fit's reported residuals define the input measurand. No residual
tolerance manufactures perfect fit. If those residuals or all completed cluster
scores are exactlyzero, zero covariance/SE is valid when all other requirements
hold. Algebraically perfect data can have nonzero floating residuals, which are
propagated. Nearly cancelling cluster scores can be sensitive to rounding.

Nonzero normalized design entries, triangular quotients, individual observation
influences, or CR1S-scaled group scores that round entirely tozero make the
covariance unavailable with4. This is a conservative contribution policy;
representable subnormal values are allowed. Individual or accumulated signed
overflow also returns4, even when a different summation order might have
cancelled earlier. No complete representable-aggregate accuracy is promised.

Norms use LASSQ. Covariance uses compensated normalized Gram products and
balanced restoration; a positive SE with diagonal covariance rounding tozero
returns4. Off-diagonal normalized products/dots and final values may round or
cancel tozero. No absolute covariance floor or clipping is inserted.
Changing labels/row order can change floating summation order and final bits;
equality of the statistical partition does not promise bitwise equality.

## Scientific interpretation

The chosen partition must support the assumed independence across clusters,
an appropriate conditional mean model and the regularity conditions needed
for asymptotic covariance interpretation. Within-cluster dependence is allowed
by the estimator. The code does not infer, test or assess these assumptions,
choose clusters, handle overlapping/multiple clustering dimensions, or estimate
a correlation model.

G>=2 is an arithmetic availability rule, not an adequate-number-of-clusters
certificate. The CR1S multiplier is the stated convention, not calibration for
few clusters. Positive standard errors do not establish cluster independence.
No finite-sample coverage, t reference distribution or p-value is supplied.

This API is unweighted OLS. It does not automatically extend to WLS, Poisson,
Huber/Tukey estimating equations, absorbed fixed effects or estimated precision
weights. These require their own bread/scores and contracts.

## Resources and verification boundary

No heap, explicit Q, normal equations or inverse is introduced. Cluster scratch
includes two2048-element f64 arrays, label/row-group arrays of256 i64 entries,
R64, scales8 and small8-element row vectors. The returned fit/result and nested
QR scratch also contribute to peak storage. Equality lookup is O(nG), bounded
by the256-row domain; triangular/Gram work is O(np^2+Gp^2).
Shared helpers and bounded source storage do not establish an RSS or speed gain.

Source completion/typechecking are distinct from native numerical acceptance.
Independent reference calculations use C++23 under ADR009. Sounio checks and
execution are coordinated through Forge/Slurm; no whole-corpus or calibration
claim follows from preparing this module.
