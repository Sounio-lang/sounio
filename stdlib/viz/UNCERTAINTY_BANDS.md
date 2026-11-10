# Sampled standard-uncertainty marks

## Scope and implementation state

The prepared implementation upgrades the existing `viz_add_uncertainty_band` path to represent
**sampled mean±1sigma marks** consistently in Canvas and SVG. The historical API
name is retained. It does not specify a continuous ribbon between samples.
Variance is sigma² in squared Y units; the mean and the resulting bounds have
Y units. A one-standard-deviation interval is not a confidence interval, credible
interval, prediction interval or coverage-probability claim.

Current-compiler typecheck and native SVG example execution are complete:
Slurm15776 returned TYPECHECK_OK for18 targets and compiled/executed the SVG
example with exit0. The compiler was built from current source in15772 and reused
after exact unchanged bin/scripts/self-hosted/stdlib tree verification.
Root structural and visual SVG inspection is complete: extracted native SVG passed
xmllint and its actual QuickLook preview was inspected. This establishes neither
scientific numerical agreement nor Canvas pixel parity. Independent numerical
concordance, Canvas runtime validation, full corpus and CI/merge remain pending.
No tests were added or run. Historical114-entry acceptance does not cover this
increment. See [execution receipt](../../docs/plans/SAMPLED_UNCERTAINTY_EXECUTION_2026_10_20.md).

## Input and builder contract

`viz_add_uncertainty_band(scene,xs,ys,vars,n,layout,style)` preserves its arguments
and node return convention. The active prefix has n in0..1024; n0 is a valid
empty series and n1 is a single sampled uncertainty mark.

Every active X, mean and variance must be finite, and every variance nonnegative.
Sigma uses the canonical mathematical square root, not a fixed-iteration local
approximation. Both mean-sigma and mean+sigma must be finite. Zero variance is
valid; no negative variance is silently converted to zero uncertainty.

Validate the whole prefix and capacity before mutating active scene counts,
IDs or series data. An invalid request returns-1 transactionally. Inactive array
entries are not read. The automatic Y domain includes every lower and upper
bound, rather than only means. Domain and projection remain explicitly bounded;
this is not a guarantee for every possible binary64 range.

No lower/upper arrays or additional uncertainty payload are added to Visual IR.
The existing means/variances are retained; scalar helpers recompute checked
bounds when rendering. This preserves the data meaning and avoids another pair
of1024-element IR buffers.

### Source effect change

The builder now requires `with Mut, Div, Panic`; the existing builder declared
only Mut. The direct `epiviz_band` path and its knowledge wrapper also require
the effects necessary for checked scalar projection and indexing. This is a
source API effect change: callers must propagate the declared effects. It is
not an allocation or inferential-method change. Public `viz_html_emit_scene` now
also declares Panic (alongside IO/Mut/Div); `viz_canvas_render_scene` carries
Panic with its existing effects. `viz_workbench_build_scene` propagates
Mut/Div/Panic. Callers outside this workbench must propagate these effects too.

## Shared rendering semantics

Each accepted sample produces a vertical, three-pixel-wide sampled mark spanning
mean-sigma through mean+sigma, subject to the shared pixel quantization and
closed plot bounds. Samples are not connected into a filled ribbon.
The center line is handled separately with shared segment clipping. SVG uses
round caps so a segment reduced to a point remains visible.

A sampled strip requires its X center inside the CLOSED declared X domain.
An X center outside is omitted even if its three-pixel footprint could overlap.
In Y, visibility instead depends on intersection of the full uncertainty interval
with the declared window: the mean may be outside while an interval fragment
remains visible. The accepted strip footprint is intersected with plot bounds.
Clipping does not invent a mean at a boundary. Constant domains map to an integer
center without epsilon padding.

The plot uses closed pixel-center bounds[x,x+w]×[y,y+h]. Intersection occurs
before Canvas pixel loops, bounding raster work to the plot. Invalid samples
must not reach floating-to-integer conversion, and invalid endpoints must not
cause a center line to bridge an invalid gap.

Canvas retains a checkerboard sampled fill. SVG uses a2×2 parity pattern and
compressed per-sample geometry to carry the same checkerboard phase. SVG/Canvas
font and edge antialiasing are not promised pixel-identical. The pattern is a
rendering convention, not a numerical confidence or probability statement.

No log/symlog uncertainty transformation or nonlinear propagation rule is added.
Standard deviations are not transformed by treating variance as a raw endpoint.
Any future nonlinear uncertainty contract requires a separate explicit decision.

## Shared scalar API

The implementation uses `viz::uncertainty`:

- `uncertainty_bounds(mean,variance)->UncertaintyBounds{lower,upper,status}`:
  status0 valid or2 invalid.
- `uncertainty_prefix(xs,means,vars,n)->bool`: validates the active1024-capacity
  prefixes before any builder mutation.
- `uncertainty_strip(sx,mean,variance,xmin,xmax,ymin,ymax,x,y,w,h)`:
  returns x/y/w/h/status.
- `uncertainty_segment(ax,ay,bx,by,xmin,xmax,ymin,ymax,x,y,w,h)`:
  returns x0/y0/x1/y1/status.

Strip and segment statuses are0 visible,1 outside,2 invalid. Bounds/prefix/strip
require Mut/Div/Panic; segment requires Mut/Div. These helpers carry no IR
dependency. `uncertainty_domain` supplies the finite ordered-domain check;
`uncertainty_raster_segment` additionally clips the already bounded center segment
to the Canvas intersection. Its public integer endpoints and scissor bounds
must lie within ±1,000,000; out-of-range values return invalid before integer
subtraction. There is no new uncertainty viewport setter in this slice; explicitly
assigned series domains use the same checked projection/clipping path.

SVG identifies its resources as uncertainty-pattern-NODEINDEX and
uncertainty-clip-NODEINDEX. Its semantic comment is
`<!-- uncertainty variance plus-minus-one-sigma sampled-not-CI n=N -->`.
The emitted group also carries data-uncertainty-input="variance",
data-uncertainty-span="plus-minus-one-sigma" and
data-uncertainty-sampling="sampled-not-CI". Individual strips and center segments
retain their sample/segment indices. Builder metadata is standard uncertainty
(interval_kind4), level0. No implicit band axes are drawn; add explicit scene axes.
This describes supplied variance semantics, not an assessed uncertainty model.

## Direct rendering entry points

Direct `epiviz_band` and `epiviz_knowledge_series` validate n before reading or
unpacking their fixed1024-element arrays. Invalid lengths produce no unsafe
array access. Validation and clipping apply even when callers bypass the IR
builder. Empty input produces no marks; a singleton does not require a fabricated
second point.

The knowledge wrapper reads the declared variance field. It does not reinterpret
a confidence field as a coverage level for these marks.

## Evidence boundaries and continuation

Completed evidence is limited to18 current-compiler typechecks and execution of
the native SVG demonstration. The checked source commit is
52ba99b87d49da04e97524fa0d8911b1d1d4d46b. Compiler SHA256:
0c896942018ff4be9a86d69f30e3a8aadac955f32451c445a07fbf502ae37fc3.
Native HTML SHA256:
82b72b8c718a8ab2743c038d289f4c704f4bb541f6e49a3a5cbf2ad4015f614e.

Independent C++23 numerical agreement and boundary coverage, Canvas runtime
validation, full corpus and CI/merge remain separate pending states. Root SVG
structure/visual inspection is complete within the execution receipt's scope. A demonstration without assertions is not a numerical test suite.

The change retains prior statistical, genomic, physics and NB2 work. It is a
bounded semantic repair within the mass stdlib sprint, with no new statistical
inference or measured resource superiority claim.
