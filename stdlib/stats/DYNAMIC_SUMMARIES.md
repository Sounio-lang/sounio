# Dynamic histogram and summaries

The numerical APIs use caller-owned typed buffers and never allocate inside a
kernel. Active observations must be finite; the mask is exactly 0 or 1. Inactive
observations are ignored. Invalid preflight leaves public outputs unchanged.

`dynamic_histogram_counts` uses explicit increasing edges. Interior bins are
left-closed and right-open; the last includes its right endpoint. Integer counts,
underflow observations, and overflow observations are retained separately.
Density takes an explicit integer normalizer: use all available observations for
unconditional density, or in-range observations for conditional density.
No binning rule or normalization choice is inferred. Balanced division preserves
representable densities even when the full bin width exceeds binary64 range.
Unrepresentable positive density returns status 4 without committing density;
the separate integer count result remains valid.

`dynamic_summary_order` validates then copies active observations and heapsorts
once in O(N log N) time and O(N) caller workspace.
`dynamic_quantile_sorted` validates a finite sorted prefix and implements type 7:
h=(N-1)q, with convex interpolation between adjacent order statistics. Endpoints
are exact. Available N must not exceed 2^53. The rank and interpolant follow
binary64 arithmetic; no exact real quantile guarantee is made.
`dynamic_box_sorted` publishes type-7 quartiles and Tukey 1.5-IQR observed
whiskers, outlier count, and separate fence availability. Overflowing fences use
quarter-scaled comparisons; finite representable fences use direct arithmetic
to preserve small values. These are descriptive summaries, not confidence
intervals. The renderer accepts these explicit method and policy identifiers.

Numerical implementations are checked against a separate C++23
Boost.Multiprecision reference, including endpoint bin membership, exact
quartiles/outliers, and a width-overflow density. Gate results are tracked in
the sprint ledger; writing this document does not establish a passing gate.
