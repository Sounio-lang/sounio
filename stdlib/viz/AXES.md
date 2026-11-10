# Shared scientific linear axes

## Public surface

- `viz::axis::{axis_ticks, axis_label, axis_offset}`: backend independent numeric plan.
- `viz::ir::viz_add_axis`: explicit axis node in the existing scene, rendered by Canvas and SVG.
- `viz_node_set_label`: existing short ASCII unit/title attachment.

An axis consumes one of the existing 64 node slots and **no series slot**.
No scene or series structure grew. Horizontal values increase rightward;
vertical values increase upward. Rectangle width/height denotes endpoint distance.

## Domains and failure

Domains must be finite and strictly increasing. The caller chooses 1..8 equal
intervals (2..9 ticks). Endpoints retain the input binary64 values. Opposite-sign
interpolation avoids evaluating hi-lo, which can overflow for valid endpoints.

AxisTicks.status is 0 on success, 1 for invalid domain/count, 2 for nonfinite or
nonincreasing computed ticks, 3 if adjacent values have identical formatted labels.
Every failed plan has n=0. axis_label returns an empty label for nonfinite input.

The builder returns -1 before mutating the scene for an invalid plan, unsupported
geometry, insufficient label spacing or exhausted node capacity. Coordinates
are bounded to +/-1,000,000 and extents to 1..1,000,000 pixels. Horizontal tick
spacing must be at least 112 pixels; vertical spacing at least 12 pixels.
Caller reserves margins for endpoints and titles and uses ASCII units.

## Formatting and backend parity

Labels use six significant decimal digits in scientific notation; zero and
negative zero share `0`. The rounding rule is half up after binary64
normalization, so this is a display policy, not an exact decimal-conversion API.
Values remain f64 in the scene. Distinct values that collapse to one adjacent
label are rejected instead of silently printed as duplicate ticks.

Both backends call the same tick, label and integer-placement functions.
Canvas uses its 8x8 ASCII bitmap font. SVG uses textLength and
lengthAdjust to reserve the same horizontal span, while retaining native text.
SVG unit/title content uses the existing XML escaping path.
Glyph rasterization is font-dependent: geometry/label parity does not imply
identical SVG and Canvas pixels.

## Automatic planning and binding

The explicit primitives above now also support the automatic covering planner
and line-chart binding in [AUTO_AXES.md](AUTO_AXES.md).
`axis_nice_cover` chooses decimal steps and a covering domain; the plot adapter
creates both axes and applies those exact bounds to the clipped line.
Additional line series can share the resulting viewport on the same rectangle.
The binding is a snapshot and must be rebuilt after data or geometry changes.

## Supported scope and follow-up

Explicit log/symlog transforms are available in SCALES.md. Automatic nonlinear ticks, offset notation for narrow domains, locale formatting,
adaptive collision layout and Unicode fonts remain separate work.
Explicit line clipping and shared ASCII legend layout are documented in CLIPPING.md;
scatter/bar/ribbon clipping and live axis/domain dependency tracking remain future
work. Axes are opt-in. The inferential forest consumer passes actual CI endpoints
to the explicit API; the automatic DFFITS consumer uses the covering planner.
