# Two datasets per scientific panel

`viz::scientific_document` now supplies opt-in two-layer entrypoints in addition
to the existing single-layer two-panel document. This is a bounded compositor,
not an arbitrary scene graph: each panel has a primary scatter and one explicit
secondary scatter or connected-input-order line. It neither refits nor sorts,
normalizes, reduces or copies numerical datasets. The four sets of owners may
have independent lengths and masks. They share the declared domain of their
panel. Different viewports require a different document.

`ScientificOverlayPlan { document, left_kind, right_kind }` wraps an existing
validated `ScientificDocumentPlan`; secondary kinds are0scatter and1line.
`ScientificOverlayText` wraps the existing four document text slices and four
additional legend slices (`left_first`, `left_second`, `right_first`,
`right_second`, each with `_start` and `_len`). All text remains borrowed from
one checked `IntegerBuffer`, printable ASCII and XML-escaped. The8px cell
contract is preserved. Every legend has its own bounded column: no truncation,
no overlap-by-guessing and no implicit reduction of font size.

Series identity owns color across both panels: primary is RGB70,180,235 and
secondary is RGB235,130,70. The opt-in renderer derives color-overridden specs
from the same validated geometry; the legacy document palette is unchanged.
This avoids swapping the two models' identities when moving between panels.
Legend glyphs
match actual scatter squares or connected lines. Legend position and data
geometry use the same canonical plan in Canvas and SVG. Dataset names and
scientific meaning remain caller declarations, not authenticated provenance.
State whether means/residuals came from different models and retain the actual
fit/buffer association.

`scientific_overlay_validate`, `scientific_overlay_svg` and
`scientific_overlay_canvas` accept plan, text, label owner, the two primary
x/y/mask triplets, then the left and right secondary x/y/mask/index/count sets.
For scatter mode the secondary index owner/count is ignored; it is not used to
filter or reorder data. In line mode the existing shared `dynamic_line_validate`
checks the index prefix, strict original-index ordering and every required gap
sentinel. Explicit line mode connects supplied original order; it is not a claim
that the line is a fitted curve or that X was sorted. No unordered scatter data
are silently converted into lines.

ALL document/legend and all four payloads are validated before any SVG byte or
Canvas paint, including the last secondary sample. The line/scatter validators
are exactly the same ones the backends subsequently use. Caller buffers remain
borrowed and unchanged throughout the call. Canvas dimensions/pointer are
checked before background replacement. Rejection leaves the entire raster
unchanged. SVG output is one complete standalone root; no browser is needed.
Primary and secondary groups have unique internal IDs; clip IDs100/101 belong
to the two optional line layers within that root. Combining separately emitted
SVG roots inside another DOM requires an external ID namespace policy.

Status0 means available; original document statuses retain their meaning.
Status31 rejects an unsupported secondary kind;32 rejects a legend slice or its
layout budget;40+left-status and50+right-status identify the secondary renderer
preflight failure. Canvas shape failure is3. `ScientificOverlayResult` reports
`left_primary`, `right_primary`, `left_secondary`, `right_secondary`.
Primary/scatter values count painted source marks, while secondary line values
count visible segments, not source rows. SVG metadata explicitly distinguishes
`scatter` from `line-input-order`; it must not be read as four identical row
counts. Masking and source-index provenance retain the underlying renderer's
semantics. Coincident marks can overlap.

The original document APIs now share the frame emitters with this compositor;
their signatures, output order, layout and plotting policy remain unchanged.
The renderer supplies no confidence interval or calibration. Its interval-kind
is none. Genuine intervals require the existing interval renderer and an
explicit scientific contract; computational covariance alone does not authorize
CI labels.

Numerical owners are not copied into fixed1024-element IR arrays. Compositor
scratch is constant-size plans/results/text metadata, excluding caller data,
Canvas raster and any explicit line-index storage. Validation/rendering traverse
all active prefixes and SVG artifact size grows with visible primitives.
Existing backend managed-runtime costs and overplotting remain; this API is
not an unlimited-resource or performance promise.

The native source fits1028 synthetic raw counts through two fixed-alpha NB2 models:
two disjoint groups versus an explicit intercept-only design. Both panels use
the actual returned means and prepared signed deviance residuals; the secondary
uses a separate declared availability mask. Independent score roots are group
means2/4 and pooled mean3. The C++23 oracle checks actual rectangles, original
indices, clipped footprints, colors, line endpoints/gaps, unique IDs, text bounds,
legend glyph binding, late-error zero bytes and an actual Canvas pixel. Negative
controls alter copies of native output; they never replace or edit the source
native artifact. Optional `SOUNIO_VIZ_CANVAS_DUMP=1` exports actual Canvas RLE for
lossless decoding by `scientific_canvas_export.cpp`, without SVG redraw.
Execution/concordance and resource receipts are recorded separately; source
preparation alone is not a passing-gate or human-acceptance claim.

## Focused executed scope

Slurm16804 checked both changed libraries, compiled and ran the new actual-fit
consumer and both existing scientific-document consumers. All native results
and both existing independent concordance checks passed. The new independent
C++23 oracle passed both roots, all layer geometry/provenance and56 text bounds,
plus three rejected copies with a moved mark, deleted mark or wrong RGB.
With identical production libraries, Slurm16809 compiled and executed the final
captioned fixture, checked eight numerical mean/residual payloads against the
independent score roots before using the numeric budget for geometry, and ran
the same three negative controls. The figure declares synthetic data and the
secondary displayed subset explicitly. Actual Canvas pixels were exported
losslessly from that executable, decoded
under Slurm and inspected by an agent: model colors remain consistent across
panels, legends/ticks are readable and no text clipping was observed. This is
focused execution and agent inspection, not full-corpus CI, broad inferential
calibration, arbitrary-layer support or human acceptance. Source/ELF/compiler
identities, actual SVG/PNG and the bounded resource observations are preserved
separately by the coordinator.
