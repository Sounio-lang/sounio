# Compact borrowed dynamic plots

`viz::dynamic_plot` renders numerical buffer owners directly, without copying
payloads into the fixed-array scene IR. `DynamicPlotSpec` contains only layout,
raw X/Y domains, opaque RGB8 style, interval kind and level. It is not an owner
of payloads or a persistent raw-pointer view. Each entrypoint borrows the checked
owners for the duration of the call. There is no 1024-sample limit, hidden
allocation, automatic downsampling, or automatic axis construction.

`dynamic_line_canvas(canvas, spec, x, y, available, indices, count)` and
`dynamic_line_svg(spec, x, y, available, indices, count, id)` consume the first
`count` original indices produced by `dynamic_line_reduce` or another checked
selection. Indices must be strictly increasing and in bounds. Every original
gap run's first index must be present; the renderer rejects a selection that
could join a line through a removed gap. Availability is exactly 0/1 and all
active source samples must be finite, including unselected samples. Inactive
numerical payloads are ignored. A standalone active sample creates no segment.
Use original data for calculations; rendering selection is not a statistical
estimator. Line specs require interval_kind=0 and level=0.

`dynamic_interval_canvas(canvas, spec, x, center, lower, upper, available)` and
`dynamic_interval_svg(spec, x, center, lower, upper, available, id)` render
supplied intervals directly, without applying line reduction. All owners have
matching lengths; each active interval must have finite ordered bounds with
lower<=center<=upper. Kind values match existing IR: 1 confidence, 2 prediction,
3 credible, 4 standard uncertainty. Kinds1..3 require a supplied level strictly
between0 and1; kind4 requires level0. These declarations describe supplied
bounds and do not validate their scientific calibration. No confidence interval
is inferred from a standard uncertainty. Callers must provide honest axes,
units, legends and scientific assumptions.

The shared geometry uses the closed plot pixel-center rectangle. Lines reuse
checked clipped segment projection and round-cap SVG lines; Canvas uses raster
intersection clipping plus scissored antialiasing. Intervals use one-pixel
vertical spans, five-pixel endpoint caps only at true endpoints inside the Y
viewport, and three-by-three center markers only when the true center is in the
Y viewport. An outside X center can retain a cap/marker footprint overlapping
the plot boundary. Footprints are intersected before drawing. SVG rectangles
cover the exact corresponding pixel cells (integer center coordinates minus
one half). The SVG fragment adds a user-space clip rectangle to contain line
cap fringes. Canvas additionally intersects the raster before drawing, bounding
paint traversal even when the plot extends off-raster.

Domains must be finite ordered with finite spans; constant domains are supported
by shared projection when samples agree with the declared value. Pixel layouts
use the existing checked geometry contract (extents1..1000000, bounded endpoints).
Numerically unrepresentable selected line projection is rejected before any
paint or fragment emission. The SVG entrypoints emit fragments: callers provide
the outer root and a nonnegative clip id unique within that root.

`DynamicPlotResult` reports status, input_count, selected_count and
visible_primitives. Status0 is available,1 invalid spec/raster/id,2 owner/shape
or count failure,3 invalid availability/payload/interval,4 invalid indices or
missing gap sentinel,5 projection arithmetic unavailable. Every failure is
preflighted before mutation/output. Visible primitives count projected segments
or interval rectangles; Canvas counts those intersecting its raster, so a
cropped raster can legitimately report fewer primitives than SVG. The interval
selected_count equals source length, including unavailable rows. The return
value is availability, not an assessment of biological or statistical quality.

The native fixture connects a1025-sample owner through the reducer to both real
backends, with isolated signed spikes and a multi-row gap. It also renders
supplied standard-uncertainty intervals with cropped endpoints and verifies
invalid-prefix rejection. The independent C++23 checker compares line
provenance/endpoint geometry, exact interval rectangles and all interval Canvas
pixels, clip cells and metadata. It checks Canvas line containment and color
bounds, not an exact antialiasing bitmask. Prepared sources are pending central
native validation; they do not establish performance or full-scene migration.
