# Shared transformed dynamic rendering

`viz::dynamic_scaled_plot` connects raw checked numerical owners to the existing
shared linear, positive-log10 and smooth symlog `ScaleSpec` transforms. It uses
`scale_valid` and `scale_forward`; no logarithm approximation is duplicated.
Constructors and callers should use these same specs for fitted raw-unit axes.

`dynamic_scale_xy` accepts raw X/Y, availability, caller-owned transformed X/Y,
and X/Y specs. `dynamic_scale_intervals` accepts raw X/center/lower/upper and four
caller-owned transformed owners. Output lengths must equal source length.
Owners are borrowed; the helpers allocate and resize nothing. Callers explicitly
budget 2N or 4N transformed f64 elements in addition to raw data. Availability is
exactly0/1. Masked payloads are neither inspected nor transformed, and masked
output elements remain unchanged. Every active value must be finite; intervals
must be ordered. A full validation pass checks all transforms before a commit
pass writes any output. Failure leaves every transformed output unchanged.

Helper statuses are0 success,1 invalid scale,2 owner/shape,3 invalid mask or raw
payload/order,4 unavailable transform (including nonpositive logarithmic input
or representability failure). `DynamicScaleResult` reports n and the number of
active rows written. In particular, logarithmic lower bounds at or below zero
reject all four outputs; they are never replaced with a positive plotting edge.

`dynamic_scaled_plot(raw_spec,x_scale,y_scale)` returns `DynamicScaledPlot` with
status, original raw endpoints, the same specs, and transformed plot domains.
Endpoints are transformed and checked for finite spans. Distinct raw endpoints
must remain distinct. The public descriptor is reconstructed/checked by high
level wrappers so inconsistent transformed endpoints cannot silently change the
plot. Style and interval declarations follow `DynamicPlotSpec`.

`dynamic_scaled_line_canvas/svg` transforms into explicit caller buffers, calls
the original-index reducer on **transformed X and its transformed viewport**,
then invokes the real dynamic line renderer. The wrapper cannot accidentally
bucket in raw X under a logarithmic axis. Segments connect transformed vertices;
this is straight-line interpolation in display coordinates, not a claim about a
continuous raw-space model. Gaps retain their original indices and masks.

`dynamic_scaled_interval_canvas/svg` transforms all four supplied coordinates
and renders them directly without reducing interval rows. Monotone transforms
preserve endpoint ordering. Transformed endpoints depict the original supplied
interval under a changed axis; they do not create new uncertainty, coverage, or
a symmetric uncertainty in transformed units. Existing kind/level metadata is
retained as the caller's declaration, without assessment of calibration.

`DynamicScaledRender` reports status, stage, n, required, selected, and
visible_primitives. Stage0 is descriptor validation,1 transform,2 reduction,
3 rendering. Nonzero status belongs to that stage's checked API. Canvas/SVG
paint and fragment emission occur only after preceding stages succeed. Caller
transformed workspaces can have been successfully committed when a later
reduction/rendering stage fails; this does not violate the transactional
transform-helper contract. Reduction shortage reports exact required capacity
without truncating indices. No hidden scratch allocations occur.

SVG wrappers append scale-mode/constant metadata only after successful series
emission, using a caller-supplied unique plot id. Constants use15-significant-
digit scientific output so tiny scale constants are not reported as zero;
this is descriptive metadata, not a binary64 round-trip certificate. Axes remain
explicit caller-owned scene nodes sharing the same ScaleSpecs.

Prepared regression sources exercise log X, symlog Y, an explicitly masked
invalid raw row, active-invalid XY rejection with untouched outputs, logarithmic
interval rejection with four untouched outputs, original-index reduction,
real Canvas/SVG geometry, and fitted axes using the same specs. The standalone
C++23 checker computes transformed geometry with std::log/log1p and independent
bucket provenance. Carefully separated interior coordinates avoid ambiguous
integer boundaries; exact raw viewport endpoints use explicit endpoint rules.
The checker also compares all interval Canvas pixels and verifies line scissor
and colors. Prepared sources alone are not native acceptance or an accuracy,
calibration, or performance certification.
