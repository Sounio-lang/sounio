# Dynamic integer-count normalization

`dynamic_median_ratio_size_factors` takes a row-major IntegerBuffer with
explicit gene/sample dimensions, caller factor/log-factor outputs and checked
scratch. It does not create a new owner, assume an intercept, read a file,
choose a zero policy, or infer study/sample identities.

Counts must be nonnegative integers at most2^53 before any f64 conversion.
Mode0 uses genes positive in every sample. Mode1 uses geometric means over
positive counts and per-sample positive ratios. Mode1 is an explicit alternative,
not an automatic fallback or an assertion of equivalence to DESeq2 poscounts.
No eligible ratios in any required sample returns unavailable, preserving outputs.

Log ratios are evaluated relative to stored integer anchors. For
abs((count-anchor)/anchor)<=0.5, checked log1p retains the small represented
integer contrast. Outside that neighborhood, the implementation uses one checked
logarithm of count/anchor rather than subtracting logarithms of two large counts.
Both positive counts are at most2^53, so this nonlocal quotient is finite and
nonzero in[2^-53,2^53] and is separated from1 in the selected branch. Size factors are exact
sample medians of the represented log ratios, then centered to geometric mean1.
Floating logarithms and exponentials are not exact-real or calibrated biological
estimates. All output factors/logs commit together after full success.

Workspace is2G+2S doubles and2G integers, with checked shape arithmetic.
The matrix is supplied by the caller. Median selection is deterministic three-way
partitioning with an8N visit budget and full heapsort fallback; even medians use
the maximum of the selected lower prefix and the selected upper median. This
preserves the exact order statistics without sorting each entire sample.
The input counts remain unchanged; scratch order may change.

## Accepted focused concordance and resource observation

The same20,000-gene ×512-sample structured fixture completed in80.61 seconds
with full heapsorting and in34.79 seconds with budgeted selection plus the
nonlocal one-log ratio evaluation. The reported maximum RSS was80384KiB for
both. An accepted selection-only intermediate ran64.16 seconds. These are
fixed-program observations under the central Slurm/compiler route, not a
general speedup guarantee, an equal-condition statistical benchmark or a claim
that total memory was reduced. The caller matrix and scratch geometry did not
grow, and no extra logarithm cache was introduced.

[tests/reference/deep_normalization_precision_reference.cpp](../../tests/reference/deep_normalization_precision_reference.cpp)
provides an independent400-decimal-digit reference from exact integer inputs:
raw logarithms/geometric means, std::sort log-ratio medians and final centering.
The26 accepted cases check native scientific15 output rather than relying on
a completion marker. They include contrasts adjacent to2^53, reciprocal far
quotients, median parity and ties, zero modes0/1, no automatic fallback, strict
preflight before zero exclusion, and transactional errors.

For nonzero tiny log factors the comparison uses absolute1e-29 plus relative
1e-11; factors use relative1e-12. Reference zeros from the bounded symmetry
cases at Big400 scale below1e-300 receive a separate absolute2e-14 allowance
for binary64 centering roundoff. This zero convention is not a proof of exact
zero for arbitrary datasets and is not used to hide the adjacent-integer
contrasts around1e-16. Missing/duplicate outputs and status/eligibility
mismatches are rejected by the C++ verifier.

Exact source/compiler/program/receipt identities and run commands belong to the
central [deep-stdlib execution plan](../../docs/plans/STDLIB_DEEP_2026_10_23.md)
and its evidence records. No compiler, test or numerical oracle was executed
locally by the implementation agent. Focused concordance does not establish a
full-domain error bound, full corpus/CI acceptance, a complete RNA-seq model,
a dispersion estimator, assessed composition assumptions or DESeq2 parity.
