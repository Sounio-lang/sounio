# Reusing least-squares triangular factors

This internal numerical interface exports the triangular factor and column
scales from the same decomposition used by the checked fit. It adds no
covariance or inferential claim by itself. The existing LeastSquaresResult
and WeightedLeastSquaresResult layouts are unchanged.

## Unweighted API

Module `linalg::least_squares`:

```sounio
pub fn qr_least_squares_factor(
    a:&[f64;2048], b:&[f64;256], m:i64, p:i64, rcond:f64,
    r_out:&![f64;64], scales_out:&![f64;8]
) -> LeastSquaresResult with Mut, Div, Panic
```

The input physical row stride remains8 and all original domain/status rules
from LEAST_SQUARES.md apply. The fit and factors come from one QR pass;
there is no second decomposition or cross-product matrix.

When and only when the returned status is0:

- r_out is an upper triangular p-by-p R factor stored with physical stride8.
- Its active columns are in pivoted order, matching returned pivot[j].
- scales_out contains original column maximum-absolute scales in original
  column order, not pivoted order.
- All lower-triangular, inactive-column and inactive-row R entries are zero.
- Inactive scale entries are zero.

Let S be the diagonal matrix of exported original column scales and let P
place original column pivot[j] at pivoted position j. In exact arithmetic
the factor represents

```text
(A S^-1) P = Q R
```

with the usual rectangular orthonormal factor Q, which is not exported.
The supplied response and its normalization do not enter the exported
column scales or R. The signs of R are the actual Householder signs;
diagonal magnitudes must not replace signed entries in triangular solves.

## Weighted API

Module `linalg::weighted_least_squares`:

```sounio
pub fn weighted_qr_least_squares_factor(
    a:&[f64;2048], b:&[f64;256], weights:&[f64;256],
    m:i64, p:i64, rcond:f64,
    r_out:&![f64;64],
    outer_scales_out:&![f64;8],
    inner_scales_out:&![f64;8]
) -> WeightedLeastSquaresResult with Mut, Div, Panic
```

The rules in WEIGHTED_LEAST_SQUARES.md apply, including strictly positive
weights, normalized weight criterion, numerical rank and variance-model
limitations. On status0:

- r_out is the upper R of the weighted, twice column-equilibrated design.
- outer_scales_out contains maximum-absolute original design column scales.
- inner_scales_out contains maximum-absolute columns of the row-weighted,
  outer-normalized transformed design.
- Both scale vectors use original column order.
- R uses the returned pivot order and stride8.
- Lower/inactive R entries and inactive scale entries are zero.

Define F=diag(sqrt(w_i)/sqrt(max(w))) and the original-order diagonal scale
matrices So and Si. The factor represents

```text
(F A So^-1 Si^-1) P = Q R
```

Thus the physical normalized-weighted design has, at pivoted column j, the
combined scale outer_scales_out[pivot[j]]*inner_scales_out[pivot[j]].
Consumers should restore those two scales separately rather than forming a
possibly overflowing/underflowing product. No response scaling belongs in R.

For covariance work, the appropriate response-unit residual scale and any
sampling-model assumptions are separate from this factor interface.
Normalized weights and model-dependent scale parameters must be handled
consistently. This interface does not certify inverse-variance weights,
Gaussian errors, independent observations, finite-sample calibration, or
coverage.

## Balanced downstream scaling helper

Module least_squares_scale additionally exports:

```sounio
pub fn least_squares_scaled_product(
    value:f64, multiplier1:f64, multiplier2:f64,
    divisor1:f64, divisor2:f64, divisor3:f64
) -> LeastSquaresNumber with Mut, Div, Panic
```

It evaluates value*multiplier1*multiplier2/(divisor1*divisor2*divisor3)
without first forming the possibly overflowing numerator or denominator
products in raw units. All inputs must be finite; the multipliers must be
nonnegative and the divisors strictly positive. Violations return status4
with zero payload, including invalid scales paired with an otherwise zero
numerator. Valid zero numerator factors return an available zero.

Each nonzero scalar is normalized to a mantissa in [1,2) and an integer
binary exponent using exact power-of-two scaling. Bounded mantissa products
and exponent addition/subtraction form the combined scale. Final normal
outputs use a power-of-two factor and one final multiplication. For
subnormals, the mantissa is first multiplied by the minimum positive normal
value, followed by a final power-of-two multiplication into the subnormal
range, including the half-minimum tie. The physical e-324 literal parser
is not required.

Nonfinite/overflowed results return status4. Final binary64 underflow,
including a rounded zero, is allowed with status0. The caller must decide
whether that zero makes its own covariance, diagonal or scale unavailable;
this helper does not impose a statistical availability rule. The value
factor supplies the result sign.

This scalar method avoids premature raw scale products; it does not prove
correct rounding for every six-factor input. Mantissa multiplication and
division remain rounded binary64 operations. It allocates no arrays or heap.
The former least_squares_restore body and its stricter nonzero restoration
policy remain unchanged, and prior fit paths keep using that original helper.

## Availability and ownership

All factor-buffer contents are available **only if the final returned fit has
status0**, regardless of whether a buffer happens to contain finite numbers.

A QR failure does not export a completed factor. In the weighted path, an
inner QR may succeed and write R/inner scales before later outer restoration
or norm computation fails. Such failed WLS buffers are unavailable and may
contain partial or previous values. Their contents are not error payloads,
and no failure-time clearing or preservation guarantee is supplied.

Pass distinct output buffers and obey exclusive-reference aliasing rules.
Do not alias factor outputs with inputs or with another output. A successful
call overwrites all factor/scale entries, including inactive zero cells.

Rank-deficient fit metadata remains governed by the original fit contract.
Its diagnostic pivot/rank fields do not make triangular factor buffers
available. No rank-deficient pseudoinverse, explicit Q or normal-equation
matrix is exported.

## Source preservation and storage

Each former public fit body is now a private shared engine. The previous
public entry point calls it with factor export disabled; the new factor
entry point calls it with export enabled. Existing numerical statements,
validation order, result layout, coefficient restoration, reported residual
definitions and norm arithmetic are retained.

For QR, export copies upper R and the original scale vector only after the
full fit succeeds. For WLS, the requested-factor branch delegates to the
same QR factor engine, forwards its R/inner scales and copies original outer
scales only after the final WLS fit succeeds.

The old QR entry point supplies bounded dummy [f64;64] and [f64;8] buffers:
576 bytes of additional array payload. The old WLS entry point supplies
[f64;64] plus two [f64;8] dummy buffers:640 bytes, in addition to delegated
QR storage. Factor-enabled callers supply those workspaces themselves.
Backend frames, fit-result arrays, delegated scratch and temporaries remain
additional costs; no measured RSS or speedup claim follows from extraction.

No explicit Q, matrix inverse or A-transpose-A product is added, and there is
no duplicate QR solely to obtain factors. Downstream statistical inference
can use triangular solves under its own model/status contract.

Native acceptance must verify both preservation of prior fit behavior and
the factor layout/scaling contract. Source comparison alone does not prove
native execution, covariance accuracy or inferential validity.
