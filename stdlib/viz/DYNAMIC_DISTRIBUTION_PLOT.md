# Explicit dynamic histograms and supplied box summaries

`viz::dynamic_distribution_plot` consumes borrowed checked owners and a compact
`DynamicPlotSpec`. It computes no statistics, allocates no payload, and has no
fixed sample/group ceiling. Both domains must be finite strictly increasing
with finite spans; these entrypoints use linear coordinates. Scale transformed
histogram heights are not implicitly substituted for count/density values.

Histogram Canvas/SVG entrypoints accept finite strictly increasing bin edges
(length B+1), heights (length B), availability0/1 (length B), and kind1=count or
kind2=density. Active heights must be finite and nonnegative. Masked heights are
ignored; every edge is validated because it defines neighboring bins. There is
no automatic binning, density normalization, division by bin width, or integer
count exactness claim for supplied f64 heights. The renderer emits supplied-kind
and supplied-normalization metadata. Stats kernels own these scientific choices.

A bin covers its actual edge interval, with its vertical footprint from zero to
the supplied height. Data intervals are clipped to the declared viewport before
projection, including positive-only Y windows. The pixel rectangle uses
half-open projected edge/baseline differences, matching accepted bar rectangle
semantics. Zero-height, wholly outside, or subpixel-rounded empty bins emit no
rectangle and increment the explicit empty count. Adjacent edge projections do
not overlap pixel columns. The caller must provide axes/units and disclose
whether heights are counts or densities, especially for unequal bin widths.

Box entrypoints accept group positions, q1/median/q3, lower/upper whisker bounds,
and availability owners with equal lengths. Every active summary must be finite
and satisfy lower<=q1<=median<=q3<=upper. `DynamicBoxPolicy` supplies an even box
width4..40, positive quantile_method and positive whisker_policy declarations.
The renderer emits these declarations verbatim;7 conventionally denotes type7
quantiles. It does not recompute or assess a declared whisker policy. Policy1 denotes Tukey1.5IQR whiskers; policy2 denotes explicitly supplied/custom
bounds. The fixture uses policy2, without claiming a fence calculation. Any specific fence-based scientific policy
must be selected and calculated by a statistics kernel before rendering.

Boxes have hollow outlines, a median line, a central whisker stem and half-width
endpoint caps. These are distribution summaries, not confidence intervals. No
uncertainty category is invented. True horizontal features outside the Y
viewport are omitted rather than moved to the boundary. Vertical stems/sides
are clipped. Full horizontal footprints are intersected with the closed plot
pixel-center rectangle, including group centers outside its X domain. There
is no outlier detection or automatic suppression; outlier markers belong to a
separate explicit source series. Collapsed quartiles/whiskers remain valid.

Both backends use identical integer filled-rectangle primitives; SVG describes
the corresponding pixel cells with coordinates shifted by minus one half.
Canvas additionally clips to its raster before painting. Every active prefix is
validated before any mark. SVG fragments include original bin/group index and
part provenance. Box part0 is the whisker stem,1/2 endpoint caps,3/4 q1/q3 edges,
5/6 box sides,7 median. Histogram uses part0.

`DistributionCounts` reports status,input_count,masked,empty,painted. Painted
counts bins/groups with any visible footprint, not their number of primitives
or unique pixels. Status0 available,1 invalid spec/kind/policy/raster,2 owner or
shape,3 invalid availability/data/order,5 unavailable projection. Every failure
has zero paint counters and produces no marks. Canvas raster cropping may yield
additional empty groups compared with SVG. Source arrays remain unchanged.

Prepared native sources cover unequal bins, zero height, masking, clipped
positive histogram extents, boundary box footprints and omitted outside caps.
An invalid later summary plus a changed earlier summary tests full-prefix
failure before drawing. The independent C++23 checker compares all actual SVG
rectangles and every nonzero Canvas pixel. Preparing these files alone is not
native acceptance, estimator validation, or a performance claim.
