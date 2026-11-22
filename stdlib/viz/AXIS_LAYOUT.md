# Actual canonical tick-label layout

Module: viz::axis_layout. The checked plan is now consumed by the public
IR fitted-axis constructor and both Canvas/SVG axis renderers. The legacy
fixed-budget viz_add_axis API remains unchanged.

## Geometry contract

The canonical label contract is an ASCII eight-by-eight pixel cell, as in
axis_label, Canvas font rendering and SVG textLength. It is not a measurement
of arbitrary fonts, Unicode shaping or optical glyph bounds.

axis_measure_labels(ticks,vertical,x,y,extent,document_w,document_h) returns
AxisLabelPlan. Tick positions use the shared axis_offset. Horizontal labels
are centered at ticks and begin eight pixels below the axis; vertical labels
end eight pixels before the axis and are centered vertically at each tick.
text_y is the top cell row; SVG baseline is text_y+7. Width is 8*label.n.

The plan validates finite increasing values, distinct labels and supported
geometry/document bounds. Adjacent cells require two pixels of separation.
Status0 means fit,1 invalid input/tick labels,2 overlap/insufficient spacing,
3 text or axis footprint outside the document. Failures have n=0 and
intervals=0; returned rectangle arrays are diagnostic only.
required_left/right/top/bottom report additional document margin necessary
for the observed extent violation, not automatic plot translation.
first_problem identifies the first detected label issue when available.

axis_fit_labels accepts a raw domain, maximum intervals1..8, orientation,
geometry, document size and ScaleSpec. It tries interval budgets downward
using axis_scaled_ticks until every label fits. It never changes raw bounds,
drops an endpoint or silently suppresses labels. Scale availability and tick
representation still depend on the existing checked scale/axis contracts.
Titles, legends, surrounding panels and glyph ink outside the canonical cell
contract are not covered. Fitting labels alone does not validate a data-series
transform or promise that the rest of a figure fits.

## Public integrated constructor

visual IR exports viz_add_fitted_axis(scene,lo,hi,max_intervals,vertical,layout,
document_w,document_h,spec,style), returning FittedAxisResult with status, node,
intervals and required margins. Status0 available,1 invalid,2 label overlap,
3 outside document,4 scene capacity. All validation precedes mutation. The
constructor preserves the raw bounds and stores only the chosen interval
count/scale in the existing axis node; no new IR buffers are introduced.
The canonical cell/stroke contract requires opaque RGB8 style with stroke1.

Both backends consume axis_measure_labels. Fitted axes revalidate against
the actual output document before painting any axis/tick/label. A changed
raster or export size can therefore make the axis unavailable: Canvas paints
no axis and SVG emits explicit data-axis-layout-status metadata. It does not
automatically refit/alter interval counts during rendering.

Legacy viz_add_axis retains its fixed112 horizontal guard and old rendering
overlap/outside behavior; its positions still come from the same planner.
No arbitrary-font metrics or title/legend fitting is inferred.

## Fixture and C++23 oracle

tests/run-pass/stdlib_deep_viz_axis_layout.sio exercises fit/overlap/document
margin statuses, renders actual Canvas labels, emits their actual SVG text,
and records every nonzero Canvas pixel as AXIS_PIXEL x y color.
It uses a fixed signed linear domain with three labels after fitted planning.
tests/reference/deep_viz_axis_layout.cpp independently constructs the exact
three label-cell rectangles from the canonical strings and integer tick
positions. It checks the actual SVG text positions/length/content and every
reported Canvas pixel against those rectangles, requiring paint in all three
labels. It does not certify glyph raster identity, font equality or arbitrary
formatting accuracy. Future fitted-constructor wiring can reuse this consumer
and oracle without changing the expected geometry.

Neither fixture nor C++ source has been compiled/executed by this author.
Native evidence belongs to the coordinator's Slurm gate. A Darwin candidate
was requested first and returned a compiling helper that incorrectly panicked
on invalid inputs and added irrelevant effects; the planner instead returns
explicit statuses. No successful check of final source is claimed.

The additional stdlib_deep_viz_fitted_axis.sio uses the real public constructor
and production IR/Canvas/SVG route for a tight signed domain with negative and
negative-exponent labels. It preserves the legacy fixed-budget refusal,
checks fitted construction and atomic rejection, emits actual SVG and native
Canvas pixels, and exercises resized-document refusal in both backends.
deep_viz_fitted_axis.cpp checks independent exact text cells, axis/tick centers,
colors, native cell containment and resized SVG unavailability. It does not
claim a full glyph bitmap oracle. These new integrated sources have not been
compiled or executed by this author; a fresh coordinator receipt is required.
