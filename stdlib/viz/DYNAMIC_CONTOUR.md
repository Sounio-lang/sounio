# Checked dynamic bilinear contours

`viz::dynamic_contour` renders explicit levels of a supplied scalar grid. It
borrows monotone X and Y owners, a strided NumericMatrix/owner, a logical
row-major cell mask, and a strictly increasing nonempty levels owner. At least
two X and Y values are required. Y coordinates increase with matrix row; row0
is the lowest data Y, independently of raster orientation. No density estimator,
field smoothing, or uncertainty model is invented.

All axis coordinates are finite strictly increasing with finite total spans.
Each mask element is0 or1. Every corner of every active cell must be finite;
masked-only payloads are ignored. The matrix descriptor must fit its current
owner. A compact DynamicPlotSpec supplies explicit strictly increasing finite
viewport domains and opaque RGB8 style, with interval_kind/level zero. Existing
shared segment projection/clipping is used by both backends.

Classification is high iff value>=level. Ordinary cells use marching squares.
Ambiguous saddles use the exact sign of
`Q=(a-level)(c-level)-(b-level)(d-level)`, with a,b,c,d ordered bottom-left,
bottom-right,top-right,top-left. Positive Q pairs bottom/right and top/left;
negative Q pairs bottom/left and right/top. Exact Q0 emits two crossing branches
joining opposite edge roots, describing the genuine bilinear saddle rather than
choosing an arbitrary diagonal.

The shared `contour_predicate::contour_saddle_sign` calculates exact Q sign for
finite binary64 inputs. Each input is decomposed by exact powers of two into a
53-bit integer significand and exponent. Six exact signed products for
`ac-bd-La-Lc+Lb+Ld` are accumulated in separate positive/negative radix2^24
180-limb arrays, then compared. There is no floating product/difference filter,
heap allocation, or rounded-cancellation topology fallback. This is an exact
predicate on supplied binary64 values, not a claim that measured data are exact.

Flat edges are handled explicitly. Every edge with both endpoints equal to the
level is emitted in full, except an all-four-equal cell emits no lines and
increments flat_cells. A flat edge shared by two active nonflat cells belongs
to the lower row-major cell. Masked or all-flat neighbors cannot own it. A cell
with one flat edge and opposite-edge values strictly on either side has an
additional straight branch joining the opposite root to the same coordinate on
the flat edge. Three equal-level corners therefore produce an L, and a flat
edge with an opposite crossing produces a T. An isolated equal-level vertex
can produce a zero-length candidate; it is omitted and counted as degenerate.

Edge interpolation is ordinary checked floating-point geometry, separate from
the exact topological predicate. When a direct endpoint difference overflows,
half-scaled differences form the interpolation ratio. Nonfinite or out-of-range
ratios/results reject the entire call before output. This does not certify
arbitrary subpixel geometry accuracy. Segments are clipped to the data viewport
and projected using the shared integer rules. Canvas additionally clips to its
raster before scissored antialiasing. SVG uses round caps inside a user-space
pixel-cell clip rectangle. The caller supplies an outer root and unique clip id.
Original cell, level index, and branch provenance accompany every SVG segment;
levels use15-significant-digit scientific metadata, without a round-trip claim.

Both entrypoints accept maxwork and maxsegments. Work is cells*levels per pass,
checked before multiplication; each call performs one full count/preflight pass
and at most one rendering pass. Prefix validation additionally scans axes and
active cell corners. The work count is bounded by maxwork and at most INT64_MAX/2
so counters cannot overflow. Pixel traversal is additionally bounded by checked
raster/plot extents; maxwork is not a wall-time budget. No segments are stored
or allocated. The exact number of nondegenerate raw segments is counted before
any paint/fragment; insufficient maxsegments returns the exact requirement and
paints nothing. Viewport/raster clipping can reduce painted segments.

`DynamicContourResult` includes status,input_cells,work_required,
segments_required,painted,masked_cells,flat_cells,degenerate_segments.
Status0 available,1 invalid spec/raster/id/budget sign,2 owner/shape,
3 invalid source,5 interpolation/projection unavailable,6 work budget/overflow,
7 insufficient segment budget. work_required=-1 denotes an unrepresentable
work product. Masked cells count once; flat cells and degenerate candidates
count per level. Unavailable calls emit/paint no partial figure. Count results
on status7 are available; other nonzero states must not be treated as a complete
contour description.

Prepared sources include96 exact predicate cases with integer/power-of-two
construction and independent C++23 IEEE bit-decoding/cpp_int evaluation. The
actual rendering fixture covers positive/negative/exact-zero saddles, a T,
three-zero L, all-flat cell, shared ridge ownership including masked/flat
neighbors, an isolated vertex, and ordinary contours. Exact integer/dyadic
geometry permits independent SVG and full Canvas pixel comparison. Budget and
invalid-prefix calls must not add geometry. These sources await central native
validation; no estimator, uncertainty, or performance certification is implied.
