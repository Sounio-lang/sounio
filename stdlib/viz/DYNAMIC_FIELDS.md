# Borrowed dynamic scatter and heatmaps

`viz::dynamic_fields` renders checked numerical owners directly. It allocates
no memory, creates no per-point allocations, does not sort or compress input, and
has no 1024-row or Mat8 dimension ceiling. Callers retain owner lifetime and
provide the outer SVG document, axes, color scale, units, and scientific labels.
These entrypoints emit SVG fragments or paint an existing Canvas.

Scatter takes a compact `DynamicPlotSpec`, X/Y `NumericBuffer` owners, and an
`IntegerBuffer` availability mask of the same length. Availability is exactly
0/1. All active X/Y values must be finite; inactive numerical payloads are
ignored. Interval kind and level must be zero. `dynamic_scatter_canvas` and
`dynamic_scatter_svg` use the same checked projection and centered three-by-three
filled footprint. A center just outside the viewport can retain its overlapping
footprint. Footprints are intersected with the closed plot pixel-center bounds;
Canvas additionally intersects the raster. No outside value is clamped into a
false boundary sample. SVG stores each original `data-source-index`.

Heatmap takes `DynamicHeatmapSpec {x,y,w,h,color_min,color_max}`, a
`NumericBuffer`, its checked pointer-free `NumericMatrix` descriptor, and an
`IntegerBuffer` mask in logical row-major order (`row*cols+col`), independent of
physical matrix strides. The finite color domain is strictly increasing with a
finite span. Active cells must be finite; inactive payloads are ignored. The
matrix descriptor is checked against the current owner even when cells are
masked. Rows increase from top to bottom. `dynamic_heatmap_canvas` and
`dynamic_heatmap_svg` call the existing canonical Viridis functions; finite
values outside the declared color range saturate at its endpoints. This is
color mapping, not alteration of the stored data.

The closed plot rectangle has w+1 by h+1 pixel centers. Logical cell edges are
`origin + index*(extent+1)/count` using integer division, with positive cell
widths/heights guaranteed before drawing. A grid with more columns or rows than
available pixel centers fails with status6; density aggregation is never
implicit. Multiplications are bounded by the existing geometry limit of one
million pixels per extent. Empty dimensions produce no cells and no division.
SVG rectangles cover matching pixel cells with positions shifted by minus one
half. Each rectangle carries logical row, column, and row-major source index;
this index is not a physical storage offset for a strided matrix.

`DynamicFieldResult` contains status, input_count, masked, outside, and painted.
Status0 is available,1 invalid spec/raster,2 owner/descriptor/shape failure,
3 invalid availability or nonfinite active payload,5 unavailable projection,
6 heatmap grid too dense for the explicit pixel budget. Full-prefix validation
precedes every mark or paint operation. Failures return zero counters except
scatter's known input_count; no apparently valid partial figure is emitted.
Canvas outside includes marks with no raster intersection; SVG counts visibility
in the plot only, so a cropped raster can have different painted/outside counts.
Masked rows/cells never increment painted. Counts describe rendering operations,
not unique pixels, density estimates, uncertainty, or biological quality.

Prepared native regression sources exercise a 4096-row scatter owner with
inactive NaNs and boundary-overlapping footprints, plus a 9-by-11 matrix with
nontrivial row/column strides and offset. The independent C++23 checker compares
every actual SVG rectangle's original provenance, exact geometry and color,
and every nonzero Canvas pixel. It uses the canonical Viridis endpoint colors
for the fixture's binary values. This does not certify every palette interior
or arbitrary projection boundary. No execution or performance claim follows
from preparing the sources; authoritative validation is coordinated centrally.
