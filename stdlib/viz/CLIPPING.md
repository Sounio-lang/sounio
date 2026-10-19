# Scientific line viewports, clipping and series legends

## Existing IR integration

viz::ir::viz_line_set_viewport(scene, node, xmin, xmax, ymin, ymax) validates a
linear line-chart viewport and enables VIZ_FLAG_CLIP_PLOT. It returns false before
mutation for invalid node/kind/slot, unsupported stroke width, geometry or domain.
The domain is stored in the existing series. Other nodes sharing that series
therefore see the same domain; callers keep explicit axes consistent with it.

The supported mark is a one-pixel, opaque RGB polyline. The current style path
uses RGB directly; alpha and wider strokes are separate rendering work.
The chart layout gives endpoint distances: closed pixel-center bounds
[x,x+w] by [y,y+h]. Coordinates stay within +/-1,000,000 and extents in 1..1,000,000.

Scatter/bar/ribbon clipping, log/symlog transforms, automatic domain/axis linking,
adaptive ticks and multiseries layouts are not certified by this checkpoint.

## Shared clipping and normalization

viz::clip::clip_series_segment normalizes against the finite, strictly increasing
domain with finite positive x/y spans, then calls clip_unit_segment.
Both Canvas and SVG use this same function and clip_pixel quantization.

clip_unit_segment clips an entire segment to the closed unit square using
Liang-Barsky parameter bounds. It does not independently clamp original endpoints,
which could draw false corner segments. An active clipping face is bound exactly;
accepted intersection residue is clamped to the unit interval.
A degenerate point inside/on the square is visible; a point outside is rejected.

ClipSegment.status:
- 0: visible segment or point;
- 1: geometrically outside;
- 2: invalid/numeric domain, including nonfinite input and normalized coordinates
  outside +/-1,000,000.

The normalized bound is an explicit numerical contract that limits cancellation;
it is not a claim of robust clipping for arbitrary binary64 endpoint ratios.
Raw scientific values can still be tiny/large when their normalized values fit.
Pixel positions use nearest-integer quantization after the accepted clip.
The independent bounded geometric comparison uses absolute tolerance 1e-9 in
normalized space, plus exact assertions for reference boundary coordinates.

A segment touching a nonfinite sample is invalid. Renderers omit both adjacent
segments and never connect across that gap. SVG includes visible/outside/invalid
segment counts. The pure helper exposes the status to native callers.

## Raster and SVG containment

display::canvas_ext::canvas_draw_line_aa_scissored receives already clipped integer
endpoints and a PixelScissor. It validates its coordinate bounds, computes
major-axis rational coverage using i64 arithmetic, and clips every blended pixel,
including the antialiasing fringe. Work and storage do not depend on the distance
to the original out-of-view endpoints. Coverage is quantized to 0..255.

SVG emits the same accepted endpoints inside a user-space clipPath covering the
closed pixel cells: x-0.5..x+w+0.5 and y-0.5..y+h+0.5. This contains the stroke
fringe consistently with the raster scissor. Round line caps preserve a segment
that degenerates to a point, following the SVG2 stroke-linecap definition:
https://www.w3.org/TR/SVG2/painting.html#LineCaps . Individual segments preserve gaps and avoid false
connections. SVG font/stroke rasterization does not promise Canvas pixel identity.
SVG path compaction is future artifact-size work; no size/speed improvement was
benchmarked in this implementation checkpoint.

## Shared legend geometry

viz::legend::legend_layout defines swatch and text positions, vertical centering
and the number of visible 8x8 ASCII glyph cells for both backends.
The swatch starts at x+4; text starts at x+20, separated from the sample.
Available text cells are (width-24)/8. Visible text is truncated by whole cells;
the full label remains in IR. SVG uses textLength to reserve the same span.
Full-label SVG tooltips, Unicode layout and wrapping remain future work.

viz::ir::viz_add_series_legend validates a source chart, printable ASCII label
(1..64), usable geometry and at least one visible cell, then snapshots its style.
Allowed source kinds are line/scatter/bar/forest; this acceptance tests line charts.
A later style change requires refreshing the legend; there is no live binding.
Invalid requests return -1 without scene mutation. It uses one node and no series.

Legacy legend controls also use the shared rendering geometry. XML-special
characters are escaped. The native fixture checks swatch pixels, a glyph pixel,
short-label truncation, preserved full IR content and transactional rejection.

## Native consumer and evidence

tests/run-pass/viz_scientific_clip.sio samples the nominal electron Compton shift
at 13 angles from 0 to pi. It draws a piecewise linear curve in a window of
0.5..2.5 radians and 1..4 picometers, with explicit shared axes and a series legend.
It is a physical-function consumer, not an experimental dataset or uncertainty fit.

tests/reference/scientific_clip_reference.cpp generates the numerical expectations
using independent long-double Cohen-Sutherland clipping, and integer AA coverage
references. Tests include 20 segment cases, degenerate points, parallel boundaries,
corner contacts, reversal, nonfinite gaps, normalized bounds, large raw scales,
raster margins, x/y major axes and a negative slope.

Execution receipts and exact snapshot/compiler identity:
docs/plans/SCIENTIFIC_CLIPPING_LEGENDS_EXECUTION_2026_10_17.md.

Compact clipped export now uses independent M/L subpaths in one path per chart.
See SVG_PATHS.md and SCIENTIFIC_SVG_PATHS_EXECUTION_2026_10_17.md for the
format change, preserved geometry, measured size and raster limits.
