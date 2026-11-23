# Ordered line index reduction

`viz::dynamic_line_reduce::dynamic_line_reduce(x, y, available, xmin, xmax,
buckets, indices)` selects original indices for rendering an ordered line. It
allocates no memory and copies no numerical payload. It does not fit a model,
estimate uncertainty, or preserve statistics for subsequent analysis. Keep the
original data for scientific calculations.

X and Y are borrowed `NumericBuffer` owners; availability is a borrowed
`IntegerBuffer` of exactly the same length. Availability must contain only 0 or
1. Active X and Y must be finite. Active X must be nondecreasing globally,
including across unavailable runs; equal X is allowed. Inactive numerical
payloads are ignored. Every owner must have construction status zero.

The viewport has finite strictly increasing endpoints and a finite positive
binary64 span. The explicit bucket count is in 1..2^53. For an active sample
inside the closed viewport, bucket projection evaluates
`((x-xmin)/(xmax-xmin))*buckets`, truncates the nonnegative result, and maps the
upper endpoint/rounded upper boundary to the final bucket. This staged binary64
projection is the contract; it does not certify exact real arithmetic at every
bucket boundary. Points outside the viewport are not projected.

Within every active segment and pixel bucket, selection retains the first,
first minimum Y, first maximum Y, and last original index, removes duplicates,
and orders candidates chronologically. Each segment's original first and last
points are retained, as are its nearest exterior neighbors on either side of
the viewport. This policy can retain extra outside points compared with a
minimal clipped representation. The first unavailable original index of every
contiguous gap run is retained as a path-break sentinel, including gap runs
outside the viewport. A renderer must inspect availability and break the path
at these sentinels; it must not draw through unavailable points. Selection is
strictly increasing in original index.

`LineReduction` reports `status`, `input_count`, `required`, `written`, and
`buckets`. Status 0 means success; 1 owner or shape failure; 2 invalid viewport
or bucket count; 3 invalid availability or nonfinite active sample; 4 decreasing
active X; 6 insufficient output length. Validation precedes selection. A count
pass computes the exact requirement, followed by a commit pass only when the
writable `indices.length` is sufficient. Insufficient storage reports the exact
requirement and written=0. Other failures report required=0 and written=0.
Every failure leaves all output elements unchanged. Success writes only the
selected prefix and leaves the suffix unchanged. Output length/capacity and
source owners are never resized. Linear borrowing prevents aliasing the output
owner with the availability input in safe callers.

Work is linear in the input length, with constant reducer scratch space and
bounded four-candidate sorting per bucket run. The caller controls output
storage with the integer owner's explicit payload budget. Bucket count does
not cause a bucket-sized allocation. This module does not yet migrate existing
fixed-array scene storage or renderers to dynamic payloads.

The small native fixture exercises gaps, ignored inactive NaNs, spikes,
transactional insufficient output and invalid flags/order. The million-sample
fixture generates ordered inputs with isolated spikes and three gap runs at
runtime, then emits the selected provenance. The standalone C++23 checker uses
independent integer bucket boundaries and map/set selection to compare every
emitted index. These sources are prepared for central Slurm validation; source
preparation alone is not a native execution or benchmark claim.

## Reusable incremental state

`line_reducer_init(xmin,xmax,buckets)` constructs an opaque state with fixed
viewport and two independent private state records, once. `line_reduction_slot()`
constructs the caller's reusable result. Neither processing path constructs a
bucket or candidate array inside its sample/bucket loop: the four chronological
candidates use a scalar sorting network and all bucket updates are fieldwise.
The batch API retains its original signature, status precedence and full-output
transactionality. This is a plumbing change, not a different selection policy.

`line_reducer_feed_into(state,x,y,availability,indices,result)` preflights the
entire chunk, then performs count and commit passes. Every selected index is an
ordinal in the complete concatenated source. Only the FIRST `result.written`
elements of the supplied output belong to this call. Drain them before the next
call; a different output owner is permitted. Prior output prefixes are never
read, rewritten or required. `required` is the exact LOCAL count needed for this
call, not the lifetime count. `input_count` is the committed lifetime input count.
The scalar queries `line_reducer_consumed` and `line_reducer_selected` report
committed lifetime input and selected-index counts without rich snapshots.

The current bucket is pending until the next bucket/gap or finish, so a valid
feed can consume rows and write zero indices. Empty feeds preserve that bucket,
gap continuity and the previous active X. Active X must remain nondecreasing
across chunk boundaries and gaps. Inactive numeric payloads remain ignored.
When a gap spans chunks, only its first global ordinal is emitted.

`line_reducer_finish_into(state,indices,result)` drains the last bucket and
closes the stream. It is idempotent: subsequent finishes write0 and require0.
Feeding a closed stream returns7. Status5 reports global-ordinal overflow before
addition. Existing statuses1/2/3/4/6 retain their owner/domain/payload/order/
capacity meanings. An invalid initial viewport yields status2 on processing.

A bad chunk or insufficient capacity does NOT consume any part of the chunk,
change committed state or modify any destination element. Retry the same chunk
with sufficient capacity; do not skip it. Private scratch and the explicit
result may change on failure. A failed finish leaves the stream open and the
output unchanged, allowing retry. Output suffixes are untouched on success.
Future invalid chunks cannot revoke already drained output: validate a complete
scientific document separately before painting when whole-document atomicity is
required. Reducer output is for display, never substituted for statistical input.

Memory beyond caller buffers is constant state, but the required selection can
be linear in input length when there are many gap runs or separate segments.
There is no claim that arbitrary input reduces to a fixed number of pixels.
A streaming caller can bound retained output by draining local prefixes, with
capacity retries explicit. Repeated one-row feeds reuse both private records.

New native sources cover every two-chunk split of a tied/spiked/gapped dyadic
fixture and compare all original indices with an independent C++23 map/set
oracle. A separate legacy-API-only fixture checks every selected ordinal for
2,000,001 alternating active/gap rows and emits a bounded checksum. Running the
same source with the saved old reducer is a measured control; no arena failure
or performance improvement is asserted before native evidence is available.
