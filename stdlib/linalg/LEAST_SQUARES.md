# Bounded column-pivoted Householder least squares

Module: `linalg::least_squares`.

## API and physical storage

```sounio
pub fn qr_least_squares(
    a:&[f64;2048], b:&[f64;256], m:i64, p:i64, rcond:f64
) -> LeastSquaresResult with Mut, Div, Panic
```

a is a fixed row-major matrix with **physical row stride 8**, independent of
the active number of columns. A[row,column] is a[row*8+column]. The active
domain is 1<=p<=8, p<=m<=256. Only rows0..m-1 and columns0..p-1 are read.
The response uses b[0..m]. Inactive row cells, inactive columns and the response
suffix are ignored, including NaN/Inf poison. Inputs are never mutated.

No intercept is inserted. A caller requiring one supplies an explicit column
of ones. All columns and response must use the intended common observation
ordering. The function does not determine weights, missing-value deletion,
units, regularization, covariance, or a statistical sampling model.

It minimizes ||A*beta-b||2 for a numerically full-column-rank active matrix.
It does not form normal equations or explicitly construct Q.

## Result

```sounio
pub struct LeastSquaresResult {
    pub status:i64,
    pub m:i64,
    pub p:i64,
    pub rank:i64,
    pub df:i64,
    pub coefficients:[f64;8],
    pub fitted:[f64;256],
    pub residuals:[f64;256],
    pub residual_norm:f64,
    pub rcond_used:f64,
    pub r_diagonal_ratio:f64,
    pub pivot:[i64;8],
}
```

For status0, coefficients are in original column order; fitted and residuals
are in original observation order. Each residual is the binary64 subtraction
b[row]-reported_fitted[row]. residual_norm is
the Euclidean residual norm in response units. df=m-p is the algebraic
residual dimension, without a claim that statistical residual degrees of
freedom or a noise model have been assessed. Square full-rank systems may
succeed with df0.

pivot[j] is the original column index at pivoted position j. Only its first
p entries are active; inactive coefficient, fitted, residual and pivot cells
remain zero. rank=p on success.

## Validation and failures

|Status|Meaning|
|---|---|
|0|Available full-column-rank least-squares result|
|1|Invalid active dimensions|
|2|Nonfinite active matrix or response entry|
|3|Rank deficient under the declared equilibrated QR threshold|
|4|Arithmetic failure or unavailable represented numerical output|
|5|Invalid rcond|

Validation order is dimensions, rcond, then **all** active finite inputs.
Consequently a zero column does not mask nonfinite input elsewhere. Invalid
dimensions or rcond trigger no array reads.

rcond=0 selects m*binary64_epsilon. Otherwise rcond must be finite and strictly
between zero and one; negative zero follows the default-zero branch. The
used value is returned only for success or rank deficiency.

All failures preserve the requested m and p. Rank-deficient results additionally
retain computed rank, active pivot labels, rcond_used and the accepted-prefix
diagonal-ratio diagnostic. Their df, coefficients, fitted values, residuals
and residual norm remain zero. Other errors zero every numeric field except
m and p. Read status before using payloads. No partial solution is returned,
and rank deficiency does not trigger a silent pseudoinverse or regularized fit.

## Equilibration, pivoting and rank policy

Each original column is divided by its maximum absolute active entry. The
response is divided by its maximum absolute active entry; an all-zero response
uses scale1. A zero column remains a zero column in the QR workspace, allowing
other independent columns to be processed before rank is reported.

At step k, every remaining column norm is recomputed over rows k..m-1 using
a scaled sum-of-squares algorithm. The largest is chosen, with ties keeping
the first current column. Full active column swaps include the already
computed upper R rows. Norms are not updated by subtractive downdates.

A compact Householder reflector uses beta=-sign(alpha)*norm, an implicit
leading vector entry1, and tau=(beta-alpha)/beta. The R diagonal is written
before the trailing-column and response updates. Reflector dot products use
compensated sums. Lower active column storage retains the reflector entries;
no dense Q is formed.

The rank test is

```text
abs(R[k,k]) <= rcond_used * abs(R[0,0])
```

using the newly recomputed diagonal magnitude. An exactly zero remaining
norm is deficient even if the product threshold underflows. Rank is the
number of accepted QR diagonals. This numerical-rank decision applies to
the **column-equilibrated matrix**, rather than a cutoff on the original
unequilibrated singular spectrum. Column equilibration reduces sensitivity
to column units; it does not provide an exact floating-point scale-invariance
certificate. CPQR rank thresholds are
heuristics; this API does not claim an SVD rank certificate.

r_diagonal_ratio is min(abs(Rdiag))/max(abs(Rdiag)) over accepted diagonals.
For rank failure it describes only that accepted prefix; rank0 returns0.
It is a heuristic diagnostic, **not** a reciprocal condition-number estimate,
error bound, or proof of numerical rank.

## Solution and output restoration

Compensated triangular back substitution solves for pivoted coefficients in
equilibrated coordinates. Coefficients are restored in original order using
the response and original column scales. Three algebraic multiplication/division
orders are tried; an order with an overflowed or prematurely subnormal
intermediate is avoided when another normal intermediate is available.
Nonzero restoration that cannot produce a finite nonzero value is unavailable
with status4. Exact zero coefficients are allowed.

Fitted values are recomputed from the borrowed original matrix divided by
the original column scales and the normalized coefficients. They are not
computed from overwritten QR reflector storage or by multiplying the
already restored possibly extreme raw coefficients. A single final multiplication
by the response scale produces the reported fitted value, permitting ordinary
binary64 rounding into a subnormal or zero. A nonfinite prediction is unavailable.
Each reported residual is computed directly as
b[row]-that_reported_fitted_value, with finite-subtraction checking. LASSQ
accumulates those raw residuals and restores their final Euclidean norm.

This residual is defined against the rounded reported prediction, rather than
an unrounded normalized QR solution or a separately scaled projection residual.
In particular, subnormal response fits are not rejected because a normalized
roundoff residual would disappear on restoration. There is no epsilon-to-zero
rule: ordinary binary64 subtraction and its cancellation/rounding apply.
The API does not return a residual sum of squares, which could overflow or
underflow even when its norm is representable.

An unrepresentable coefficient, nonfinite individual prediction or residual,
or unavailable final residual norm invalidates the entire numerical payload
with status4. Coefficient restoration does not silently supply zero for a
detected nonzero underflowed coefficient. In contrast, the single final fitted
multiplication follows IEEE binary64 underflow rounding, including zero.
Exact cancellation producing represented zero is also permitted. Underflow
during matrix normalization, inner products or cancellation can still lose
components; status0 is not a correctly-rounded result certificate.

Householder QR improves the numerical route compared with normal equations,
but it does not remove ill-conditioning or guarantee useful coefficients for
every nearly dependent design. No statistical standard errors, confidence
intervals, likelihood, or model-assumption tests are supplied by this kernel.

## Resource and execution boundary

The weighted-fit checkpoint additionally repairs the first component of each
Householder application: its separately compensated tail dot gives
`new_top=(alpha/beta)*old_top-tau*tail_dot`. Lower updates use the full
projection. This preserves tiny top projections lost by subtracting the
nearly equal full projection from old_top in the covered critical cases.
The scalar restoration body is now shared in least_squares_scale.sio.
Native15405 passes86/86, including all prior83fixtures; restoring the actual
previous top-update formulas fails both new numerical regressions.
See [execution receipt](../../docs/plans/STATS_WEIGHTED_QR_EXECUTION_2026_10_19.md).

One [f64;2048] matrix workspace, one [f64;256] transformed response, two small
[f64;8] scale/solution vectors, and the bounded result arrays. Original matrix
data are borrowed again for prediction rather than duplicated in another
full matrix. No heap, explicit Q, or sample history is allocated.

Work is O(m*p^2), including recomputed norms, with fixed p<=8 and m<=256.
These are source resource bounds, not measured runtime/RSS improvements;
compiler temporaries and stack-frame overhead are additional.

Independent C++23 references using a different factorization and native
Sounio execution through Forge establish acceptance for the cases actually
executed. Source completion alone is not native execution, a full-corpus
result, or a numerical proof for the whole least-squares domain.

## Algorithm background

- [LAPACK DGEQP3](https://netlib.org/lapack/lapack_routine/dgeqp3.f) describes column-pivoted QR.
- [LAPACK Working Note114](https://www.netlib.org/lapack/lawnspdf/lawn114.pdf) discusses the practical role and limitations of QR pivoting for revealing rank.

The API above declares its own bounded diagonal-threshold policy and resource
contract. Acceptance receipts concern this implementation and covered cases.
