# Checked logarithmic and symlog scientific plots

## Numerical surface

`viz::scale::{ScaleSpec,ScaleValue,scale_forward,scale_inverse,axis_scaled_ticks}`.

ScaleSpec.mode is 0 linear, 1 positive log10, 2 smooth symlog.
constant must be finite and positive for every mode; it affects symlog only.

    linear(x) = x
    log10(x) = ln(x)/ln(10), x > 0
    symlog(x,c) = sign(x)*ln(1+abs(x)/c)
    symlog_inverse(t,c) = sign(t)*c*expm1(abs(t))

Symlog accepts zero and negative finite values. Log10 rejects zero and negative
values; negative-only log axes are not implemented. There is no epsilon shift.
The transform uses a short-range log1p series and expm1 series to avoid cancellation
near zero. Overflowing x/c is handled through logarithmic decomposition; a large
inverse coordinate is combined with ln(c) before exponentiation when needed.

ScaleValue.status: 0 available; 1 invalid mode/constant; 2 nonfinite input;
3 nonpositive log10 domain; 4 nonrepresentable output, including nonzero-to-zero
underflow. Failure has inactive value=0. Nonzero subnormals remain representable
but can reduce inverse precision; status 0 does not guarantee relative round-trip
accuracy independent of conditioning.

math::pure::ln_checked returns CheckedLog (0 available, 1 nonpositive domain,
2 nonfinite input/result). It shares the existing logarithm core. Legacy ln
retains its signature, effects and domain-sentinel policy. The checked API has
Mut/Div effects and does not add Panic to existing render entrypoints.

## Axis semantics

axis_scaled_ticks takes raw-unit endpoints and explicit 1..8 intervals.
It creates equally spaced coordinates AFTER transformation and inverse-maps
their labels to the raw units. It does not claim nice decade/minor-tick selection.
Endpoint labels retain the exact raw inputs; intermediate labels are checked for
finite/increasing values and distinct six-significant-digit display strings.
AxisTicks status follows AXES.md (1 invalid, 2 representation, 3 duplicate label).
Axis node flags are 2*mode+orientation; node.value stores c. Axis node min/max
remain raw units; series viewport min/max are transformed units.

Canvas and SVG call the same tick/label functions and integer offsets.
SVG emits scale names and a six-digit constant display label as metadata.
That label is for display; the IR and source configuration retain full f64 c.

## Plot construction

`viz::scaled_plot::{ScaledDomain,ScaledPlot,viz_add_scaled_plot}`.

ScaledDomain declares raw x/y limits, both ScaleSpecs and explicit interval counts.
The builder creates one clipped line and two matching axes in the existing IR.
It consumes three node slots and one series slot; no scene record grows.
Inputs have 2..1024 paired finite samples. All first n samples must transform
successfully; invalid log values or nonfinite samples reject the whole call.
This API does not silently filter samples or turn them into gaps. Existing
linear gap-capable export remains available separately.

Data transform once into the inactive destination IR series. No second pair of
1024-element transformed input arrays is allocated. On failure active node/series
counts, IDs and active domains are unchanged. Inactive storage past series_count
can be overwritten and is not a rollback journal. Raw caller arrays are retained.

Geometry uses the existing line-clip bounds and one-pixel stroke. Axis labels need
>=112 px per horizontal interval and >=12 px per vertical interval; the caller
reserves outer margins. Plot statuses: 1 invalid geometry/count, 2 capacity,
3 unavailable x plan, 4 unavailable y plan, 5 invalid sample transformation.

Segments interpolate straight between transformed samples, as a sampled log plot;
this is not an adaptive curve tessellator. Values outside valid explicit domains
are clipped AFTER transformation. Build again when raw data/domain/scale changes.

## Compatibility protections

Linear viewport and automatic-linear-axis APIs reject charts carrying scale bits.
The linear share-domain API rejects either a scaled source or target. A caller
must not copy raw values into an already transformed series. Scaled overlay/
shared-axis update helpers and live dependency tracking remain future work.
Manual IR mutation can bypass these API checks.

Reserved line-chart flags: 512/1024 encode x mode; 2048/4096 encode y mode;
the combined mask is 7680. Existing clip flag 256 is retained.
For scaled lines node.min_value/node.max_value store x/y constants.
Axis flags 0/1 retain the old linear orientation convention.

## Independent evidence and limits

C++23 long-double log10/log1p/expm1 generates 51 transformation and inverse cases,
raw-unit logarithmic/symlog tick values, invalid-domain and overflow controls.
Tolerance is relative 2e-11 in ordinary cases, with a two-subnormal-ulp budget
and its inverse propagation for quantized transform coordinates.

For x=1e-16,c=1e300, rounding the transform to binary64 already produces inverse
absolute error about 1.63402854931100443e-24. C++ computes a propagated two-ulp
bound 9.88131291682493140e-24. This is a representation bound, not a relaxation
of the normal-range criterion or a claim that all finite inputs round-trip.

Scientific consumers use actual regression DFFITS (linear x/symlog y, c=0.1)
and nominal electron Compton shift (positive log10/log10).
Source arrays, active-state rejection, metadata, Canvas execution and SVG output
are covered by the focused fixtures. See the execution receipt for native job
identity and accepted counts. Source presence alone is not acceptance.

Conceptual references:
https://d3js.org/d3-scale/symlog and https://d3js.org/d3-scale/log .
This is an independent Sounio implementation with its own supported domains and
explicit tick policy, not a full D3 API/behavior compatibility claim.
