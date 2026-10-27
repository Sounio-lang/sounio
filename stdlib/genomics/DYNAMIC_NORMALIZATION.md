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

## Reusable strided adapter

The earlier API already accepts dynamic gene/sample counts. The additive
`dynamic_median_ratio_size_factors_into` adapter changes the storage and result
contract, preserving the estimator, eligibility, median and centering policies:

```text
dynamic_median_ratio_size_factors_workspace(G,S)
  -> {status,genes,samples,numeric_elements,integer_elements,payload_bytes}
dynamic_size_factors_slot() -> reusable result plus checked-log scratch
dynamic_median_ratio_size_factors_into(counts,G,S,row_stride,col_stride,offset,
  mode,factors,log_factors,work,anchors,slot) -> status
```

The caller owns all buffers. An active count is at
`offset + gene*row_stride + sample*col_stride`. Nonnegative strides include
repeated read-only views; padding is ignored. Bounds and products are checked
before indexing, and all active counts are validated before ratio evaluation.
The query reports2G+2S double elements andG integer anchors, together with
checked payload bytes. Each eligible gene stores its positive integer anchor;
zero identifies an excluded gene. There is no additional per-cell log cache or
new8-sample/256-gene capacity limit.

Construct `dynamic_size_factors_slot` once and reuse it. The adapter captures
scalar log results from its one reusable CheckedLog child and propagates every
checked-log failure as arithmetic status4. It does not allocate a checked-log
result per gene/sample visit. Selection still returns one result per sample;
no claim of zero allocations for the complete routine is made. Slot scalar
getters expose status/used/excluded; `dynamic_size_factors_snapshot` creates an
explicit detached metadata snapshot. Reusing/resetting the slot does not modify
that snapshot.

Statuses remain0success,1shape/layout/mode/budget,2negative active count,
3no eligible ratios in a required sample,4arithmetic,5active count above2^53.
Both public factor/log-factor buffers remain unchanged on failure; workspace,
anchors and result metadata are scratch and may change. The old packed function,
scalar workspace query and2G integer eligibility/anchor layout remain available
through an adapter. It uses the same checked leaf path, so this change also
removes repeated checked-log record construction from the legacy call.

`normalization_into_reference.cpp` independently uses high-precision direct
integer logarithms, sorted log-ratio medians and centering. Its five cases cover
both zero policies, strided poisoned padding,513×17 dimensions, adjacent2^53
counts and repeated views. Native factor/log results are also required to be
bit-identical between the new and legacy adapters for those inputs. The status
fixture covers malformed layouts/workspaces, exact-count boundaries and
transactional failures. Neither comparison certifies biological assumptions.

`profile_normalization_into.sh` is a Slurm-only paired whole-program harness.
It compiles the unchanged20,000×512 legacy workload once per source variant,
then alternates three pairs of fresh-process executions. It retains compile/run
wall/user/system time, RSS, statuses and source/compiler/ELF/stdout hashes.
It requires accepted markers and identical scientific stdout. Measurements are
specific to that structured program and do not isolate the normalization kernel
or certify general speed, peak memory or a global numerical error bound.

The focused Slurm observation for this adapter (job16802) retained the same
whole-program scientific stdout across all pairs:

| Trial | Before wall seconds | After wall seconds | Before RSS KiB | After RSS KiB |
|---|---:|---:|---:|---:|
|1|35.48|35.69|79872|80384|
|2|35.66|35.72|80384|80384|
|3|35.50|35.65|80384|80384|

These observations show no execution-time or RSS improvement for that structured
workload. They support retaining the interface/storage/checking changes without
claiming a measured speedup or memory reduction. The matrix and legacy scratch
still dominate the caller-visible buffer contract; the new adapter'sG-integer
anchors reduce the declared integer scratch requirement relative to the old2G
layout, which remains available for compatibility. Complete artifacts are under
`/scratch/sounio-stdlib-advanced-20261023/normalization_perf01` on the Slurm host;
that scratch path is an execution receipt location rather than a portable input.
