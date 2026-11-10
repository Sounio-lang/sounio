# Labelled Pearson and Spearman correlation permutations

Module: `stats::correlation_permutation`.

## Statistical contract

This is a two-sided test based on absolute Pearson correlation or
averaged-rank Spearman correlation under an **exchangeability/independence
null for the paired labels**. The x observations remain fixed; y labels
are permuted. The library does not assess that null or the sampling design:
exchangeability_assessed is always false.

Zero population correlation alone does not justify exchangeability. The
procedure is not generally a valid test of zero correlation under
heteroskedastic dependence, clustered observations, longitudinal dependence,
or other nonexchangeable designs. It does not perform blocked or stratified
permutations, a residual permutation scheme, or a studentized weak-null test.

## API

```sounio
pub fn correlation_permutation(
    x:&[f64;256], y:&[f64;256], n:i64,
    statistic:i64, draws:i64, seed:i64, exact:bool
) -> CorrelationPermutationResult with Mut, Div, Panic
```

Result fields:

| Field | Type and meaning |
|---|---|
|status|i64; availability/error code below|
|statistic|i64; requested 1 Pearson or 2 averaged-rank Spearman|
|method|i64; 1 Monte Carlo, 2 complete labelled enumeration|
|n|i64; requested count|
|coefficient|f64; observed correlation estimate|
|p_value|f64; two-sided enumerated or plus-one MC proportion|
|hits,draws|i64; accepted extreme-or-equal labelings and completed labelings|
|seed_used|i64; normalized initial MC state, zero for enumeration|
|mc_available|bool; Monte Carlo error metadata available|
|mc_se,mc_lo,mc_hi|f64; MC hit-probability standard error and Wilson95 bounds|
|exchangeability_assessed|bool; always false|

Only the active prefixes are inspected, and inputs are never mutated.
All outputs have bounded storage, without a vector of resampled statistics.

## Validation and errors

| Status | Meaning |
|---|---|
|0|Available result|
|1|n outside 2..256|
|2|Nonfinite active observation|
|3|At least one variable is exactly constant|
|4|Arithmetic failure|
|5|MC draws outside 1..100000|
|6|Complete enumeration requested for n>8|
|7|statistic outside 1..2|

Validation order is statistic, size, requested method capacity/budget, then
coefficient preparation (finite inputs, constants and arithmetic). No array
element is accessed for invalid statistic, size or capacity/budget.
Enumeration ignores the supplied seed and draws entirely.

An error preserves requested n and statistic. Every other numerical result
field, including method, coefficient, hits, draws and seed_used, is zero;
both boolean fields are false. Read status before using numerical payloads.

## Preparation and tail comparisons

The shared core has a reusable preparation API:

```sounio
correlation_prepare(x,y,n,ranked,a,b) -> CorrelationEstimate
correlation_prepare_indexed(x,y,n,ranked,a,b,indices) -> CorrelationEstimate
```

a and b are exclusive references to [f64;256] workspaces; the indexed form
accepts an exclusive [i64;256] workspace, permitting one index buffer to
serve ranking and later permutations. The shared inputs are fixed256
references, n is i64, ranked is bool, and all functions declare Mut,Div,Panic.

A successful Pearson preparation returns anchor-centered and scaled
coordinates using the checked core's bounded arithmetic. Spearman ranks
each vector once with deterministic heapsort and averaged ties, then
subtracts (n+1)/2 from those ranks **without scaling**. These are exact
half-integer centered ranks. Failed preparation does not make output
workspace contents available. Only active output prefixes are written.
Callers must supply distinct input/output workspaces and distinct a/b
exclusive references, respecting the language's borrowing contract.

The previous pearson_checked and spearman_checked numerical algorithms
remain unchanged: their centered moment/projection body is extracted into
a read-only helper rather than recomputed differently. Preparation's
unscaled-rank estimate may differ from spearman_checked in final floating
rounding while representing the same statistic.

A permutation changes neither centered vector norm nor its mean. Therefore
the tail compares absolute dot products directly, without recalculating a
coefficient, sorting, centering or computing a norm per draw.

- Spearman compares abs(dot)>=abs(observed_dot) exactly on the half-integer
  rank arithmetic. Quarter-integer products and their bounded sums remain
  exactly representable for n<=256, including ties.
- Pearson uses compensated dot products and the policy
  abs(dot)>=abs(observed_dot)*(1-64*epsilon_binary64). This relative allowance
  is part of the declared floating-point statistic comparison. It is not a
  general proof of exact tail classification for every ill-conditioned
  cancellation pattern. No absolute tolerance is silently added.

## Complete labelled enumeration

exact=true is supported for 2..8 observations. An in-place lexicographic
index iterator visits all n! permutations of labels, including the identity.
Equal observed values do not remove their label multiplicity. The result has

```text
method = 2
draws = n!
p_value = hits / n!
seed_used = 0
mc_available = false
mc_se = mc_lo = mc_hi = 0
```

“Exact” means complete finite label enumeration. Pearson still evaluates a
floating-point prepared statistic with the stated comparison policy.
Spearman's half-integer tail arithmetic is exact within the bounded rank
domain. Neither interpretation establishes the scientific null assumption.

## Monte Carlo draws and reproducibility

exact=false requires 1..100000 draws. Every draw resets the identity labels
and applies a full descending Fisher-Yates shuffle, j=n-1 through 1.
Sampling is with replacement across draws; successive permutations are not
a partially updated shuffle chain.

The deterministic Park-Miller stream uses state=(16807*state)%2147483647.
For a width w, draw=state-1 is rejected when it is at or above
floor(2147483646/w)*w; accepted draw%w selects the swap index. This avoids
modulo bias for uniform generator outputs. It is not a cryptographic RNG
or a proof that outputs from a deterministic PRNG are independent.

Initial state follows the shared legacy normalization seed%2147483647,
adding 2147483646 when nonpositive. A final nonpositive result is explicitly
mapped to 2147483646, repairing the negative congruence class that otherwise
reaches the absorbing zero state. In particular -2147483646 and -4294967293
produce the same valid stream as positive seed2147483646. seed_used records
that normalized initial state. Products remain within i64 range.

The reported MC p is (hits+1)/(B+1), so zero hits do not produce zero p.

## Monte Carlo metadata

q=hits/B is the uncorrected hit fraction. mc_se=sqrt(q*(1-q)/B).
mc_lo and mc_hi are the Wilson95 interval for that hit probability, including
zero and all-hit cases. They use z=1.959963984540054.

These bounds describe Monte Carlo hit-probability precision under the usual
binomial sampling approximation. They are not an interval for the correlation
coefficient, a posterior probability of the null, or a claim of scientific
coverage under invalid exchangeability. mc_se can be zero at an endpoint
while the Wilson interval still has positive width.

## Execution and validation boundary

Pearson preparation is O(n); Spearman preparation is O(n log n).
Each draw is O(n). Complete enumeration is O(n*n!).
The permutation function has two [f64;256] workspaces and one [i64;256]
workspace, 6144 bytes of array payload. The same index buffer is reused
for rank preparation, enumeration or shuffling. No heap, per-draw ranking,
per-draw coefficient normalization, or sample history is allocated.

Array payload excludes scalar records, compiler temporaries and backend
stack-frame overhead. Resource bounds are algorithmic; no measured runtime
or RSS improvement is asserted.

Independent C++23 references and native Forge execution establish numerical
acceptance for their explicitly covered cases. Source completion alone
does not establish native execution, a full-corpus result, or inferential
validity for a user's sampling design.

## Method references

- Phipson and Smyth (2010), [Permutation p-values should never be zero](https://gksmyth.github.io/pubs/PermPValuesPreprint.pdf): random-permutation p-value correction.
- DiCiccio and Romano (2017), [Robust permutation tests for correlation and regression coefficients](https://www.tandfonline.com/doi/abs/10.1080/01621459.2016.1202117): independence and zero-correlation nulls require distinct calibration.

Studentized weak-null inference, blocked/cluster permutations and empirical
coverage/power studies remain future implementation work; they are not supplied
by the simple reassociation contract above.
