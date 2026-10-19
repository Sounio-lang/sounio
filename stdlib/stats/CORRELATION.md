# Checked Pearson and averaged-rank Spearman estimates

Module: `stats::correlation_core`.

## API

```sounio
pub struct CorrelationEstimate {
    pub status: i64,
    pub n: i64,
    pub coefficient: f64,
    pub residual_ratio: f64,
}
pub fn pearson_checked(x:&[f64;256], y:&[f64;256], n:i64)
    -> CorrelationEstimate with Mut, Div, Panic
pub fn spearman_checked(x:&[f64;256], y:&[f64;256], n:i64)
    -> CorrelationEstimate with Mut, Div, Panic
```

These functions return descriptive estimates only, without a p-value,
confidence interval, posterior interpretation, or assertion of independence.
Pearson measures linear association. Spearman applies Pearson to averaged
ranks and measures monotonic association. Neither measure establishes
causality, accounts for clustered observations, or removes confounding.

## Domain and status

n must be in 2..256 and is validated before array access. Every active
pair is checked for finiteness before centering, sorting, or ranking.
The unused suffix is ignored. Inputs are never mutated.

| Status | Meaning |
|---|---|
| 0 | Available estimate |
| 1 | n outside 2..256 |
| 2 | Nonfinite active input |
| 3 | At least one active variable is exactly constant |
| 4 | Arithmetic failure after validation |

The returned n always preserves the requested count, including errors.
Unavailable estimates have coefficient=0 and residual_ratio=0; read status
before interpreting those fields. An unavailable constant variable is not
reported as an available zero correlation. Signed zeros compare equal and
belong to the same tie group.

A two-point estimate is allowed when both variables vary, but it does not
supply residual degrees of freedom for conventional Pearson t inference.
Inference is a separate contract.

## Stable Pearson computation

Each private scratch vector uses its first observation as an anchor. When
subtractions remain finite, the largest absolute anchored difference scales
the vector. This preserves representable small spreads around a large common
offset and permits subnormal-scale inputs without raw squared units.

If an anchored subtraction overflows for finite observations of opposite
sign, the vector is first divided by its largest raw magnitude, then the
scaled anchor is subtracted. This fallback represents the large overall
spread without requiring an overflowing difference. Very small components
relative to that overall spread can round away.

A compensated mean centers the bounded coordinates. Compensated centered
sums of squares and cross-products determine

```text
r = Sxy / sqrt(Sxx) / sqrt(Syy)
```

No raw-unit square, raw covariance determinant, or product of raw standard
deviations is required. Constant detection uses equality of the original
observations before normalization. A subsequent collapsed or invalid norm
is an arithmetic failure, rather than a claim that the original data were
constant.

Tiny coefficient excursions beyond [-1,1] of at most 1.5e-14 are brought to
the endpoint; larger excursions fail with status4. This is a bounded
floating-point correction policy, not a formal proof of a global error bound.

## Residual ratio and near-perfect association

In exact arithmetic, residual_ratio=sqrt(1-r^2). It is evaluated independently
from the rounded coefficient, using the centered projection:

```text
b = Sxy/Sxx
e_i = centered_y_i - b*centered_x_i
residual_ratio = norm(e)/sqrt(Syy)
```

The residual norm uses scaled sum-of-squares accumulation. This avoids
subtracting r^2 from one after r rounds to an endpoint, and can retain a
positive residual ratio for a coefficient that rounds to +1 or -1.
Equal or opposite centered vectors give b=+1 or -1 and exact zero residuals
when their compensated sums share those exact relationships.

The residual calculation still has binary64 centering, slope, and subtraction
error. Near-perfect relationships at roughly one input ULP may therefore
have substantial relative error in residual_ratio. The result does not
promise arbitrarily accurate inference in that regime. Exchanging x and y
leaves the exact ratio unchanged, but rounding can differ. The same small
endpoint-correction threshold applies to a ratio just above one.

No general numerical precision claim follows from coefficient==1 alone.

## Spearman ranks, complexity and memory

Each variable is ordered by deterministic index heapsort: finite value is
the primary key and original position breaks equal-value sort ties.
After sorting, all observations equal by floating-point comparison receive
the average of their occupied one-based ranks. Ties are not broken in the
statistical ranks. No subtraction is used to compare extreme finite values.

Spearman then runs the same private Pearson implementation on its two rank
buffers. It does not use the no-ties shortcut 1-6*sum(d^2)/(n*(n^2-1)), which
would be incorrect with averaged ties.

Pearson uses two fixed 256-element f64 scratch arrays: 4096 bytes of array
payload. Spearman uses those two rank buffers and one 256-element i64 index
buffer reused for the second variable: 6144 bytes of array payload. The
private Pearson pass centers rank buffers in place, avoiding another pair
of scratch copies. Stack-frame metadata and backend temporaries are additional.

No heap allocation. Pearson is O(n); Spearman is O(n log n) with linear
tie-group assignment. These are algorithmic resource bounds, not measured
RSS or benchmark speedups.

Independent C++23 reference evaluation and native Sounio execution through
Forge are the acceptance route. Source completion alone establishes neither
native execution nor full-corpus validation.
