# Positive-weight bounded QR least squares

Module: `linalg::weighted_least_squares`.

## API and criterion

```sounio
pub fn weighted_qr_least_squares(
    a:&[f64;2048], b:&[f64;256], weights:&[f64;256],
    m:i64, p:i64, rcond:f64
) -> WeightedLeastSquaresResult with Mut, Div, Panic
```

The matrix is row-major with **physical stride8**, independent of p.
A[row,column] is a[row*8+column]. The active domain is
1<=p<=8 and p<=m<=256. Only that active rectangle and the first m
response/weight entries are inspected. Inactive cells and suffixes may
contain poison. Inputs are borrowed and never changed.

The objective is a positive weighted squared-error criterion:

```text
min_beta sum_i (w_i / max(w)) * (b_i - A_i beta)^2
```

Common positive weight scaling leaves its mathematical minimizer unchanged.
No intercept is inserted; supply a column of ones when intended.

Weights must be finite and strictly positive. Zero-weight deletion, negative
weights, missing-data processing, frequency-weight semantics and constraints
are not supplied. Positive weights define a numerical criterion; they are
not automatically validated inverse variances or sampling weights.
variance_model_assessed is always false. No covariance, standard error,
p-value, confidence interval, likelihood or coverage is inferred.

## Result and units

WeightedLeastSquaresResult has these public fields:

|Field|Type and meaning|
|---|---|
|status|i64; availability/error code|
|m,p|i64; requested dimensions|
|rank|i64; accepted weighted/equilibrated QR rank|
|df|i64; m-p on success, algebraic residual dimension only|
|coefficients|[f64;8]; original column order|
|fitted,residuals|[f64;256]; original observation order|
|residual_norm|f64; Euclidean norm of reported raw residuals|
|normalized_weighted_residual_norm|f64; weighted norm using weights/max(weight)|
|weight_scale|f64; max active weight|
|rcond_used|f64; actual default or requested threshold|
|r_diagonal_ratio|f64; accepted-diagonal heuristic|
|pivot|[i64;8]; original column labels at pivoted positions|
|variance_model_assessed|bool; always false|

Coefficients and predictions use original design/response units.
Each residual is the binary64 subtraction b_i-reported_fitted_i.
Both norms retain response units because the weight ratio is dimensionless.
The normalized weighted norm is not sqrt(sum w_i residual_i^2), unless
weight_scale=1. Only p coefficients/pivots and m fitted/residual values are
active; unused output cells are zero. A square fit may succeed with df0.

## Validation and unavailable results

|Status|Meaning|
|---|---|
|0|Available fit|
|1|Invalid dimensions|
|2|Nonfinite active matrix or response value|
|3|Rank deficiency under the delegated equilibrated QR threshold|
|4|Arithmetic, transformation or representability failure|
|5|Invalid rcond|
|6|Nonfinite or nonpositive active weight|

Precedence is size, rcond, all active matrix/response values, then all active
weights. A bad weight or zero column cannot mask nonfinite matrix/response
input elsewhere. Invalid size/rcond reads no array element.
rcond0 selects m*binary64_epsilon; otherwise it must be finite and strictly
between zero and one. Negative zero follows the default.

Every failure preserves requested m,p. Rank failure may additionally retain
rank, active pivots, rcond_used, accepted-prefix r_diagonal_ratio and
weight_scale; df and all solution/norm fields are zero. Other errors zero
every numeric field except m,p. No partial solution, dropped-row fit,
pseudoinverse or regularization is fabricated. Read status first.

## Transformation and delegated QR

Original columns are divided by their respective maximum absolute entries.
The response uses its maximum absolute entry, or1 when all zero.
A zero column remains zero and reaches the QR rank calculation, allowing
other independent columns to be processed before a deficiency is reported.

Each row factor is sqrt(w_i)/sqrt(max(w)), using canonical shared sqrt.
It is deliberately not sqrt(w_i/max(w)): forming that ratio first can
underflow even when the square-root factor is representable.

The normalized matrix and response entries are multiplied by that row
factor. A nonzero raw input disappearing during outer normalization, or a
nonzero normalized coordinate whose weighted product rounds to zero,
produces status4. The kernel does not silently erase a nonzero entry.
Nonzero subnormal products retain ordinary binary64 rounding; this check
does not promise exact relative precision for them.

The transformed system delegates to the existing column-pivoted Householder
qr_least_squares kernel, which may further equilibrate its columns and
response. Its returned coefficients are in the outer normalized coordinates.
Restoring the outer original-column and response scales yields original-unit
coefficients.

The numerical-rank criterion concerns the weighted and reequilibrated
matrix. Strictly positive weights do not change mathematical rank, but
numerical rank can change with relative scales. r_diagonal_ratio remains a
heuristic, not a reciprocal condition number or singular-value certificate.

## Shared restoration and reported residuals

least_squares_scale.sio exports LeastSquaresNumber and least_squares_restore.
Its numerical body is moved unchanged from the QR kernel, apart from symbol
names and public visibility. It tries three product/division orders and
accepts normal finite intermediates before their final operation, avoiding
avoidable intermediate underflow or overflow. The helper presumes already
validated finite values and positive finite scale/divisor; it is not a
separately validated general-purpose scaling API.

A separate intentional QR arithmetic repair accompanies this block. Both the
trailing-matrix and response Householder top entries accumulate the reflector
tail dot product separately and use
(alpha/beta)*old_top-tau*tail_dot, since 1-tau=alpha/beta. Lower entries still
use the full projection tau*(old_top+tail_dot). This prevents subtracting two
rounded equal large terms from erasing a small meaningful tail when alpha=0.
The helper extraction and this numerical repair are distinct changes. A
before-control source with the former top update is retained for native
regression comparison; source reasoning alone does not establish execution.

Coefficient restoration is strict: a nonzero coefficient that cannot be
restored as a finite nonzero value yields status4. Exact represented zero
coefficients are accepted.

Fitted values are recomputed with compensated sums from the borrowed
original matrix normalized by its original scales and the outer normalized
coefficients. The transformed design is not mistaken for original data.
One final response-scale multiplication permits IEEE-rounded subnormal
or zero fitted values. Nonfinite predictions are unavailable.
Residuals are b-minus-those-reported-predictions, with no epsilon zeroing.

Both norms use LASSQ scaled sums of squares in the same observation pass.
For the weighted norm each raw residual magnitude is multiplied once by its
row factor before LASSQ. A factor in (0,1] cannot overflow a finite residual.
A nonzero residual whose weighted product rounds to zero produces status4:
this conservative individual-contribution policy may reject a case whose
aggregate norm would otherwise be representable. It does not silently omit
that contribution. Nonzero subnormal weighted products remain allowed.

No raw residual is directly squared. Final norm restoration uses the shared
helper; an unavailable required norm invalidates the numerical payload.
The API does not expose a squared-error sum that could overflow despite a
finite norm. Binary64 inner-product cancellation, normalization rounding
and ill-conditioning remain limitations; status0 is not a correctly-rounded
or universal relative-error certificate.

## Storage and execution boundary

The wrapper uses one [f64;2048] transformed design, one [f64;256] response,
one [f64;8] original-column-scale vector and a bounded result.
It borrows original design data again for prediction. Delegation also incurs
the existing QR scratch matrix, response and result storage; this wrapper
does not claim the memory cost of a single QR invocation alone.

No heap, normal equations, explicit Q, covariance matrix or sample history.
Work is O(m*p^2) plus bounded transform/prediction/norm passes. These are
source bounds, not measured RSS, stack or speedup claims; backend temporaries
are additional.

Independent C++23 references and native Forge execution establish numerical
acceptance only for the cases actually run. Source completion alone is
neither native execution nor an assessed variance model.
## Method background and execution receipt

The [NIST weighted least-squares criterion](https://www.itl.nist.gov/div898/handbook/pmd/section4/pmd432.htm)
defines the weighted residual objective. Interpreting weights as estimated
inverse variances requires additional modeling; this API leaves that assessment
explicitly unavailable. See
[native86/86execution receipt](../../docs/plans/STATS_WEIGHTED_QR_EXECUTION_2026_10_19.md).

least_squares_scale is a shared implementation utility. Kernel calls supply
finite values and positive finite scale/divisor arguments; its exported
visibility serves cross-module reuse rather than a general scaling-domain
validation contract.
