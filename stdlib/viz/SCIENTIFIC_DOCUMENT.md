# Borrowed scientific documents

`viz::scientific_document` composes two scatter panels into a complete standalone
SVG or an existing Canvas, using the accepted dynamic scatter renderer. It does
not copy data into the fixed-capacity IR. The caller owns all numeric and integer
buffers and supplies the two raw linear domains explicitly. There is no automatic
statistical computation, density aggregation, downsampling, interval inference,
filesystem write or browser dependency.

`scientific_two_panel_plan(width,height,xmin0,xmax0,ymin0,ymax0,xmin1,xmax1,ymin1,ymax1)`
returns a `ScientificDocumentPlan`. Dimensions are bounded to width640..1000000,
height360..1000000; these are validity bounds, not a promise that every size fits.
All domains must be finite, strictly increasing and have finite spans. Constant
data require a caller-declared nonconstant viewport. Shared margins use actual
canonical six-significant-digit tick labels and the existing eight-pixel cell
contract; up to four intervals are tried, then fewer. Panel axes share geometric
alignment but each retains its declared domain. Status1 means invalid dimensions
or domain,2 means no adequate axis/label plan. No generic font measurement is
claimed.

`ScientificDocumentText` contains four start/length slices into one borrowed
`IntegerBuffer`: title, left panel title, right panel title and footer. Every
slice must be nonempty printable ASCII32..126, in bounds, and fit its entire
allocated line at eight pixels per character. No label or legend is silently
truncated. Panel titles should identify X/Y variables and units; the caller must
declare scientific caveats in the footer. The renderer escapes ampersand, angle
brackets and both quotation marks for SVG. It draws the same character positions
through the native Canvas font, while SVG requests canonical textLength. This
is a layout contract, not identical font-raster certification.

`scientific_document_validate(plan,text,bytes,x0,y0,mask0,x1,y1,mask1)` preflights
the exact shared scatter validators for BOTH entire source prefixes, plus every
label and reconstructed plan. Masks are0/1. Active values must be finite;
inactive payloads retain the existing renderer's ignored-payload semantics.
A malformed final row or footer rejects before any SVG byte or Canvas paint.
Render functions repeat the same validation and borrow unchanged inputs:
`scientific_document_svg(...)` emits one complete SVG;
`scientific_document_canvas(canvas,...)` requires exact document dimensions and
a valid raster. Only after success does Canvas replace its background.

Return `ScientificDocumentResult` reports status,stage,left_painted,right_painted.
Status0 available,1 plan,2 text/owner,3 Canvas shape/pointer;10+left-scatter-status
or20+right-scatter-status identifies a rejected panel. Stage is reserved0.
The complete SVG reports actual input/masked/outside counts for both panels,
and a completion marker after both panels and footer. Unique panel ids are
internal to this standalone root. Concatenating roots into another document
requires the caller to manage that surrounding document.

Optional plan flags `identity_left` and `zero_right` request descriptive reference
lines: identity requires exactly equal left X/Y domains; zero requires the right
Y domain to contain0. They are not uncertainty intervals. Generic documents emit
interval-kind none. The default background is RGB22,28,36, series colors are
RGB70,180,235 and235,130,70. Mark clipping/RGB/provenance comes from dynamic_fields;
axes use pixelcell rectangles and shared integer tick offsets.

No new data-size ceiling or hidden data owner is introduced. Validation and
rendering traverse all samples; artifact size is proportional to visible
scatter marks. This wrapper does not solve overplotting and does not claim a
general performance gain or unlimited managed-runtime resources.

Native sources exercise1025 points, complete XML escaping, a late-row
failure producing zero SVG bytes, exhaustive unchanged-Canvas assertions,
actual colored pixels and both successful panels. The independent C++23 oracle
checks actual SVG marks/provenance/footprints/colors, all text bounds, completion
and a native pixel report. It does not compare every font pixel. In focused
Slurm16417, both scientific-document consumers passed native execution and
independent actual-output concordance. Slurm16422 exported actual Canvas pixels
losslessly, with full RLE coverage checks and rejection of missing-run copies.
The corrected native font and actual PNGs were inspected; these are bounded
agent inspections, not human acceptance or a full-suite result. Source/compiler
identities, actual SVG/PNG artifacts, scope and single-program measurements are
in the [focused evidence bundle](../../docs/evidence/stdlib-inference-document-2026-10-23/README.md).
