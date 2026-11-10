# Seeded bootstrap of the mean

## API and result

`stats::bootstrap::bootstrap_mean(&x, n, b, conf, seed)` borrows a fixed
256-value array. Accepted sizes: 2..256 observations, 2..1024 resamples,
finite confidence strictly between zero and one. Only the first n entries
are read. Caller data is unchanged.

BootResult contains mean, se, lo, hi, b, status and seed_used.
status=0 means available; 1 invalid size/resample budget; 2 invalid confidence;
3 nonfinite data; 4 nonfinite arithmetic. On failure all numerical payloads
and b/seed_used are zero sentinels, not estimates. Always inspect status.

Requests above the resample capacity now fail explicitly instead of silently
clamping. Quantiles now use linear interpolation h=(b-1)p (type 7).
These changes, the integer index mapping and seed normalization intentionally
change numerical output from the legacy implementation.

## Computation and reproducibility

Point and resample means use compensated sums after normalization by the maximum
absolute input. Bootstrap SE is the sample standard deviation of the resample
means about their own mean, using Welford updates in normalized coordinates.
The percentile interval uses probabilities (1-conf)/2 and (1+conf)/2.
Heapsort replaces insertion sort: O(n*b + b*log(b)) work, one 1024-double
resample buffer, no heap allocation or per-resample sample array.
This is a source resource bound, not a measured peak-stack/RSS benchmark.

MINSTD recurrence is computed in i64, state=(16807*state)%2147483647.
Normalize the signed seed by its remainder modulo 2147483647; if that remainder
is nonpositive, add 2147483646. If the result is still nonpositive, map it to
2147483646: this repairs the negative remainder -2147483646 that otherwise
enters the absorbing zero state. Seeds -2147483646 and -4294967293 now match
positive seed2147483646; other normalized streams are preserved. Zero, exact
multiples and i64 minimum require no overflowing absolute value. The valid
normalized seed is returned.
Draw state-1 and reject values >= floor(2147483646/n)*n before reduction modulo n.
This removes the extra modulo mapping bias; the generator remains a deterministic
31-bit linear generator, with finite period and serial dependence. It is not a
cryptographic generator or a claim of best available random-stream quality.
Same data/order, settings, implementation and pinned backend reproduce the output.
No cross-backend bitwise portability guarantee is made.

## Statistical and floating-point limits

The ordinary empirical bootstrap treats observations as exchangeable iid samples
from the target population. It does not establish independence, correct clustering,
handle paired/longitudinal structures, or fix selection bias. Finite-resample
percentiles have Monte Carlo error. The default capacity is a bounded execution
contract, not a universal adequacy threshold for tail precision.

This is an ordinary percentile interval, not BCa, studentized, exact coverage or
an assumption-free guarantee. Constants legitimately yield zero SE and a
degenerate interval. Percentile intervals need not contain the original estimate;
the current forest builder requires containment, so consumers must check it
rather than alter the interval.

Scaling protects intermediate sums/squares against many overflows and underflows.
Tiny components relative to the largest magnitude can still be rounded away;
no arbitrary precision, exact-sum or correctly-rounded guarantee is made.
A mathematically nonzero result below the f64 range may round to zero.
Nonfinite final results are rejected explicitly.

## Existing parallel APIs

`stats/resampling.sio`, `bootstrap_diff.sio`, `bootstrap_corr.sio` and
`stats/epistemic/bootstrap.sio` are separate legacy implementations. This
checkpoint does not certify or silently redirect them. Their consolidation,
Checked mean BCa is now separately accepted in [BCA.md](BCA.md), sharing this
kernel and returning explicit availability and bias/tail metadata. The legacy
mean-BCa adapter delegates to it; other legacy routines, correlation/general BCa,
coverage studies and broader API consolidation remain explicit sprint work.
Checked exhaustive/Monte Carlo permutation inference is now separately accepted
in [PERMUTATION.md](PERMUTATION.md), with an explicit exchangeability null and
[execution receipt](../../docs/plans/STATS_PERMUTATION_EXECUTION_2026_10_17.md).
