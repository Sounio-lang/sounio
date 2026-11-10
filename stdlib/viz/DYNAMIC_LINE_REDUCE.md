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
