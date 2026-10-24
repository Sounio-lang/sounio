# Checked forest geometry and explicit pages

This module represents caller-supplied intervals; it does not compute or
calibrate an interval. Existing confidence/prediction/credible kinds and level
metadata remain distinct. Legacy variance forests represent mean plus/minus
one standard deviation, with canonical checked square root.

## Row and footprint policy

viz::forest::forest_layout(n,x,y,w,h) returns status, capacity, step and first.
The plot rectangle contains pixel centers x..x+w and y..y+h, inclusively.
A seven-pixel marker requires seven distinct pixel-center rows. The capacity
is min(1024,(h+1)/7). Status0 is available,1 invalid input/geometry,2 insufficient
row capacity. n0 is a valid empty legacy layout.

Where historical row spacing was at least seven pixels, its positions are
retained. Otherwise spacing is seven pixels when the full row set fits.
There is no implicit row truncation, overlap fallback or off-panel continuation.

forest_project validates finite ordered bounds and projects a row.
Status0 is visible,1 interval outside,2 invalid input.
A constant domain centers the matching coordinate. Interval endpoints beyond
a manually supplied domain are clipped without a fabricated end cap.
forest_clip_rect intersects actual filled interval, cap and 7x7 marker
footprints with the closed plot pixel rectangle. Both Canvas and SVG use it;
SVG rectangle coordinates describe the corresponding pixel cells at x-.5,y-.5.
The center marker may be partially clipped at a domain endpoint.
This does not claim pixel-identical antialiasing across arbitrary backends.

Every active row, interval kind/level and plot domain is validated before
either renderer paints/emits marks. Invalid raw scene mutations refuse the
whole forest rather than showing a misleading partial result.

## Builders and effects

viz_add_forest_plot keeps its arguments, now with Mut,Div,Panic for validation
and canonical sigma. Invalid variances, arithmetic or layout fail before
scene mutation. Auto X bounds include the full sigma envelope.

viz_add_forest_interval_plot keeps its arguments, now with Mut,Div.
It rejects a whole source that does not fit; return -1 leaves counts/IDs intact.
All bounds must be finite, ordered around the estimate and have a finite
aggregate X span. Constant aggregate domains are supported without fake padding.

viz_add_forest_interval_page accepts the same source arrays, total, first
(zero-based), kind, level, layout, style. It returns ForestPageResult:
status, chart, first, count, next, total, has_more.
Status0 available;1 is reserved for initial invalid state;2 invalid interval
data/kind/level/domain;3 invalid geometry/source range;4 no available row;
5 scene capacity. A caller continues explicitly at next while has_more.
No page is claimed to include the full source when has_more is true.

The complete input prefix is validated even when requesting an early page.
All pages use one global source envelope, supporting meaningful comparisons.
Only the chosen page is copied into scene storage; no additional data arrays
or heap allocation are introduced. Forest node min_value/max_value store
first source row/total source rows; SVG exposes these as page metadata.
Caller code must present page order or navigation; this API does not add UI.

## Validation boundary

New run-pass sources are stdlib_deep_viz_forest.sio (existing APIs only,
designed to fail on old crowded-row acceptance) and
stdlib_deep_viz_forest_pages.sio (checked planner/pagination contracts).
tests/reference/deep_viz_forest.cpp is a C++23 independent exact integer
footprint oracle/checker for the first fixture's actual SVG. It checks eight
primitives, their pixel-cell rectangles/colors and forest metadata.
It does not validate all statistical values, text layout or every renderer.

No local/workspace compiler execution has been performed by this change's
author. Authoritative builds, execution and before/after evidence belong to
the coordinator's Slurm gate. Source preparation is not an accepted result.
