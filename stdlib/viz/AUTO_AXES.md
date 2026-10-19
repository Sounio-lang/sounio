# Automatic linear axes and shared plot domains

## Public API

- `viz::nice_axis::axis_nice_cover(lo,hi,budget)` returns `NiceAxis`.
- `viz::plot::viz_auto_plot_axes(&!scene,chart,style)` returns `PlotAxes`.
- `viz::plot::viz_plot_share_domain(&!scene,source,target)` returns bool.

The explicit-axis API in AXES.md remains the rendering contract. These functions
add numeric planning and bind its output to the existing line viewport and axes.
No scene, node or series record grew. Two axes consume two node slots, no series.

## Covering plan

Inputs must be finite, strictly increasing, with budget 1..8 intervals.
The selected step belongs to 1,2,2.5,5 times a decimal power. Integer multiples
cover both endpoints outward and must fit the budget. The planner advances the
step when outward rounding needs more intervals. The final existing axis_ticks
plan must have finite increasing values with distinct six-significant-digit
labels. Existing renderers recompute the same tick plan.

This is an outward covering-domain policy, not a claim to reproduce D3 ticks.
D3's primary documentation describes the separate conventions for ticks inside
an interval and nice domains covering an interval:
https://d3js.org/d3-array/ticks . The step family here additionally includes 2.5.

NiceAxis.status:
0 available; 1 invalid input; 2 representation limit; 3 adjacent labels coincide;
4 insufficient interval budget. Failure has intervals=0 and inactive zero bounds.
A one-interval budget cannot cover a range straddling zero with this lattice.

Representation limits are explicit: finite input span and expanded span,
positive representable decimal step, and normalized integer indices bounded by
9007199254740990. Extreme finite endpoints, subnormal spacing and narrow
large-offset domains can be unavailable; no log, padding epsilon or fabricated
label is substituted. Offset formatting is future work. step is nominal; binary64
interpolation can differ from exact decimal arithmetic by rounding.

## Plot binding

The source must be an existing one-pixel line chart with 2..1024 rows and valid
finite stored domains. Plot layout must have width >=112, height >=24, bounded
geometry compatible with line clipping, and room for the added axis origins.
Horizontal budget is min(8,width/112); vertical is min(8,height/24).
Neither count nor bounds is manually specified by the consumer.

Both plans and capacity are validated before changing the active graph.
Success creates a horizontal axis at (x,y+h+18), a vertical axis at (x-18,y),
and updates the source line viewport to the same planned bounds.
The caller reserves outer margins for labels, units/title, legends and canvas.
This helper does not infer the output canvas dimensions or fit text around them.

PlotAxes.status:
0 available; 1 invalid chart/geometry; 2 insufficient node slots;
3 unavailable x plan; 4 unavailable y plan; 5 unexpected construction failure.
On failure active node count, node ID allocation and series domain are unchanged.
Inactive storage past node_count is not a transaction journal.

Share-domain requires a clipped source line and identical target plot rectangle.
The validated target gets the source's exact four bounds and clipping.
It allocates no new slot; rejection preserves target limits. The source domain
is authoritative: a larger target can be clipped. Compute a union before binding
when all series must contribute to the limits.

This binding is a snapshot. Editing data, manually modifying an axis or changing
layout later requires rebuilding/rebinding. It is not a live scene dependency
engine. Log/symlog, scatter/bar/ribbon clipping and offset/Unicode label policies
are not implemented by this API.

## Numerical evidence and scientific consumer

The independent C++23 reference enumerates decimal candidates in long double,
rather than using the Sounio normalization loops. Generated checks cover 12
domains, including 2.5 steps and scales around 1e-299 and 1e299; failure fixtures
cover nonfinite, constant, budget and label limits.

The DFFITS consumer uses real case-deletion regression output and shares the
automatic viewport with positive/negative screening lines. It checks axis-domain
equality, overlay equality, rejected capacity/geometry/label changes, and executes
Canvas and SVG. See the sprint execution receipt for actual native job identity
and acceptance; this API contract alone is not an execution result.

Explicit nonlinear plot construction and checked log/symlog are available in
SCALES.md; the automatic builder in this document remains linear-only.
