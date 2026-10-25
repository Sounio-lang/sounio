# Reusable viewport-bound contour capture

`viz::dynamic_contour` provides a caller-owned `PreparedContour` for preparing a
sampled bilinear contour once and exporting the same integer geometry to Canvas
and SVG. This is geometry reuse, not statistical data reduction or a new physical
interpolation model. The existing direct contour APIs retain their signatures,
classification, clipping, and deterministic segment order.

The capture is **viewport-bound**: its input `DynamicPlotSpec` domains, origin,
extents, RGB, and contour levels are copied into private owned state. It retains
no grid, mask, level-buffer, matrix-descriptor, or caller-spec pointers. Changing
any of those inputs requires another successful prepare. Input owners can be
closed after preparation; exporting a previous capture does not inspect them.

## Lifecycle and explicit payload budget

1. `prepared_contour_new(segment_capacity, level_capacity, max_bytes)` allocates
   at most `64*segment_capacity + 8*level_capacity` payload bytes once, with checked
   multiplication and addition. It returns a linear owner; inspect
   `prepared_contour_status` and always consume it with `prepared_contour_close`,
   including partial allocation failures.
2. Construct one `prepared_contour_work()` and reuse it. Its private cell,
   exact-predicate limb arrays, projection, and raster/scissor scratch do not
   allocate per cell or segment. This fixed managed workspace and allocator/RSS
   overhead are outside the owner's payload budget.
3. `prepared_contour_prepare_into(&!owner, &spec, &xs, &ys, &values, &matrix,
   &cell_mask, &levels, maxwork, &!work)` validates the entire active input,
   counts all topological segments, and validates their projection before any
   owner writes. It then commits using the same deterministic topology code.
4. `prepared_contour_canvas_into(&!canvas, &owner, &!work)` and
   `prepared_contour_svg_into(&owner, nonnegative_id, &!work)` borrow the snapshot.
   Each export resets output counters and performs zero cell/predicate evaluations.
   The SVG caller supplies a unique document-local ID. The emitter writes a
   fragment with a pixel-cell clip path, not an enclosing document.

Each raw segment occupies eight i64 values: projected endpoint coordinates,
row-major source cell, source level index, cell-local branch, and visibility.
Outside segments retain their provenance and consume capacity; their visibility
is 1 and their endpoint placeholders are zero. Visible status 0 includes a
projected point. The owner does not duplicate world-space coordinates.
`prepared_contour_segment_into` reads a record into reusable caller scratch and
clears all fields on an invalid lookup.

`prepared_contour_bytes` reports retained payload bytes, including successful
partial allocations; it is neither peak RSS nor total memory use.
`prepared_contour_generation` increments only after a successful prepare. An
empty-capacity owner can query the exact requirements without allocating payload.

## Validation, statuses, and transactionality

Owner construction statuses are 0 available, 1 invalid sizes/budget, 2 insufficient
budget, and 3 allocation failure. A successfully constructed owner starts with no
prepared capture.

Prepare statuses preserve the direct contour conventions: 0 success, 1 invalid
plot/request or generation exhaustion, 2 unavailable storage, 3 invalid data,
5 invalid topology/projection arithmetic, 6 work budget, and 7 segment or level
capacity. `work.capacity_known` is true only after complete counting/projection;
then `segments_required` and `levels_required` are exact, including on status 7.
Earlier errors do not invent a capacity estimate. All prepare failures leave the
owner's previous capture, levels, generation, and readiness unchanged. Work is
scratch and may change on failure.

Inputs require increasing finite grid axes and contour levels, a checked strided
matrix, a 0/1 row-major cell mask, finite corners of every active cell, finite
spans, and a nonconstant plot domain. Masked-only values may be nonfinite. The
explicit work limit is cells times levels; it is checked before traversal and
bounds segment/count arithmetic. The caller must keep immutable borrows valid
through preparation and must not modify a Canvas payload through an alias.

Both exports validate capture/request/raster before painting or emitting output;
an invalid SVG ID emits no bytes. Canvas additionally clips the stored endpoints
to its raster intersection and uses the existing scissored antialiasing primitive.
The stored plot geometry and SVG clip use closed pixel-center bounds. SVG and
Canvas consume identical captured plot endpoints; backend antialiasing/font
rasterization are not asserted to be identical.

## Preserved topology and reusable helpers

High corners use `value >= level`. Ambiguous cells use the exact IEEE-f64 dyadic
saddle predicate. An exactly zero saddle emits crossing branches; it does not
choose an arbitrary diagonal. Flat edges, T junctions, three-zero-corner L shapes,
masked/all-flat neighbor ownership, degenerate endpoint counting, and branch order
are preserved. Shared Into helpers are additive:

- `contour_sign_slot`, `contour_saddle_sign_into`;
- `clip_segment_slot`, `clip_unit_segment_into`;
- `uncertainty_segment_slot`, `uncertainty_segment_into`,
  `uncertainty_raster_segment_into`.

Their legacy wrappers remain available. Direct contour exports also reuse cell
and projection scratch, preventing the new exact-predicate workspace from being
constructed inside their cell loops. Repeated standalone rich wrapper calls are
not the allocation-free interface.

## Executable acceptance and limits

Run `bash scripts/stdlib/prepared_contour_gate.sh NEW_OUTPUT_DIRECTORY` with the
canonical resolver's explicit `MADAROS_RAW_BIN`. Heavy compilation/execution is
performed on Slurm, uid 1000, with a 524288-KiB stack limit. Setting
`SOUNIO_SEED_COMPILER` also checks the actual import closure with that seed.
The gate checks legacy contour geometry/pixels, 96 exact cpp_int saddle cases,
new prepared geometry and every nonzero Canvas pixel, capacity and late-error
atomicity, detached input lifetimes, clipping/outside records, repeated prepare,
and zero-output invalid SVG requests. Coordinate, level-provenance, and missing
completion mutants must be rejected. The optional PPM is reconstructed solely
from the fixture's actual native Canvas pixel records, not by redrawing SVG.

The separate benchmark uses the same linear field, quarter-grid isovalue, 257²
Canvas, data setup, and dual exports before/after. It does not benchmark a full
scientific pipeline, all saddle patterns, all fields, GPU rendering, or a generic
contour library. Native runtime, compiler build provenance, mathematical
concordance, CI, and human review remain separate evidence.
