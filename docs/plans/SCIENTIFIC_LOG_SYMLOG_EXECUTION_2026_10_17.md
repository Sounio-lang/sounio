<!-- docs:meta
topic_id: repo.docs.plans.scientific-log-symlog-execution-2026-10-17
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.plans.scientific-log-symlog-execution-2026-10-17
-->

# Log10, smooth symlog and checked logarithm — 2026-10-17

## Accepted native identity

Run `stdlib-scale-20261017-final1`, Slurm **14905**.
COMPLETED **0:0**, elapsed **00:01:20**.
JUnit **23 tests, 0 failures, 0 skips, 0 errors**.
Snapshot:
`a6547926e6c43f6a96d648c5756280f8952ab0977c88838389a4012cce9bc2f6`.

Pinned committed Madaros v0.80 decompressed ELF:
`125c25703ac4062fc6befb7acf77e945abb13405f0adb5ab09fbc5a8d4b5b29f`.
Compressed compiler:
`feaaf7fe181f1ee3efd30e7fe23b1709937ea7e93b52e812276fb093b6ccf52a`.
This is not a compiler rebuilt from current sources.
2 CPUs, 8192 MiB, GPU off, five-minute limit, clean-v1, UID/GID 1000/1000.
Evidence root:
`/var/lib/cephfs-slurm/training/sounio-foundry/runs/stdlib-scale-20261017-final1`.

Isolated worktree: `/workspace/sounio-sprint-stats-viz`.
Original remote-main baseline:
`f5b2c5850be9f3d3995719ffcade44a6d851658a`.
Remote main checked as `106f7a988d5b007472d0eead724f58ac50f92200`.
Its stdlib differences from that baseline are optimize/nelder_mead.sio and
polynomial/lib.sio, outside this focused closure.
The active canonical checkout was not modified. No commit/push/PR/merge or CI
acceptance is claimed.

## Implemented

- `math/pure.sio`: extracted the existing logarithm body into ln_core, preserving
  the legacy ln wrapper, signature, effects and domain policy. New ln_checked
  returns a status for nonpositive/nonfinite inputs instead of the legacy sentinel.
  No logarithm approximation was duplicated; existing renderer effects remain.
- `viz/scale.sio`: checked linear, positive log10 and smooth symlog
  sign(x)*ln(1+abs(x)/c), and inverses. Positive finite c is explicit.
  Stable short-range log1p/expm1 series avoid cancellation. Log decomposition
  handles overflowing x/c; large inverse coordinates combine with ln(c).
- Shared axis tick planning uniformly spaces transformed coordinates and labels
  inverse-mapped values in raw units. Six-digit scientific display and duplicate
  label checks remain. This is explicit interval planning, not automatic nonlinear
  decade/minor-tick selection.
- `viz/scaled_plot.sio`: validates input/domain/geometry/capacity, then transforms
  into the inactive destination IR series, creating one clipped line and two axes.
  Active state survives rejection; raw caller arrays are retained. No extra pair
  of 1024-element transformed buffers and no scene-record growth.
- Existing Canvas/SVG renderers consume the same raw-unit axis plan. SVG records
  scale names and a display label for c. The actual f64 constant remains in IR.
  Compact independent-subpath SVG export is retained.
- Linear viewport/auto-axis/share-domain APIs reject scaled charts instead of
  applying raw limits to transformed samples.
- Public contract: `stdlib/viz/SCALES.md`.

New scaled plots require finite valid first-n samples (2..1024). Any nonpositive
log sample or failed transformation rejects the whole call. The builder does
not filter samples into gaps. The separate legacy linear gap path remains.

## Independent numerical evidence

`tests/reference/scale_reference.cpp` builds with C++23, -O2 -Wall -Wextra
-Werror -pedantic. Long-double log10/log1p/expm1 generates the Sounio fixture.
It covers 51 transformations and inverses, logarithmic/symlog raw-unit ticks,
invalid parameters, zero/negative log domains, nonfinite inputs and overflow.

Ordinary comparisons use relative tolerance 2e-11. For subnormal transformed
coordinates, the reference computes a two-ulp absolute coordinate allowance and
its inverse propagation; it does not silently impose an impossible uniform
round-trip relative bound.

For x=1e-16,c=1e300, C++ reports:
- rounded coordinate: 9.99999983659714433e-317;
- inverse error from that rounding alone: 1.63402854931100443e-24;
- propagated two-subnormal-ulp bound: 9.88131291682493140e-24.

The native numerical fixture executes all 51 cases, including extreme x/c values,
and finishes SCALE_REFERENCE_PASS. This is bounded binary64 evidence, not an
all-input proof or an ulp-optimal logarithm claim.

## Native scientific consumers and independent geometry

The plot fixture uses:
1. Actual case-deletion DFFITS, linear x and symlog y with c=0.1 (printed in title).
2. Nominal electron Compton shift, positive log10 x and y, with original radian/pm
   axis labels. Samples span twelve doubled angles beginning at 0.001 rad.

Canvas executes both figures. It checks an axis pixel and the active-state
rejection paths; SVG includes the selected scales and raw-unit labels.
The actual final SVGs and their Quick Look previews were inspected. PNG previews
have square white padding and are not Canvas exports.

A separate C++23 tool, scale_scene_reference.cpp, recomputes Compton values with
long-double sine using the repository's physical constants, then independently
applies log10 and integer projection. It matches all twelve points / eleven
segments exactly in the native SVG.

The same tool compares complete SVG documents byte-for-byte for selected legacy
fixtures (automatic/explicit DFFITS, axes, clipping/Compton, legends, interval
forests and dense path cases); all comparisons pass. It was authored after the
native snapshots and is separate evidence, not a file claimed inside them.

## Controlled identity-transform comparison

Run `stdlib-scale-20261017-before1`, Slurm **14906**.
FAILED **1:0**, elapsed **00:01:20**.
JUnit **21 passes, 2 failures, zero skips/errors**.
Snapshot:
`cea0c33a7530425547d380618cb8bd831432e579b2202e01d3760fe71b9afa82`.

Manifests differ only in `stdlib/viz/scale.sio`. The controlled variant retains
the new API but bypasses the nonlinear transformations with identity coordinates.
Both new fixtures compile and fail runtime assertions; the numerical fixture
reaches the first positive-log case. All earlier 21 tests remain passing.
This is a controlled mathematical counterfactual, not an unmodified old checkout.

## Integration corrections

- 14895: existing 21 pass; numerical fixture parse fails; nonlinear SVG axes
  missing while the chart and Canvas execute.
- 14897: explicit typed scalar reads from scene fix nonlinear SVG axis metadata/
  constant access; integrated plot passes. Numerical fixture still fails parse.
- 14900: renamed reserved variable linear in the generator; fixtures split into
  bounded helper functions. Numerical execution reaches the subnormal case and
  exposes the unconditioned round-trip criterion.
- 14905: independent C++ quantization allowance, explicit c in the figure title,
  all 23 native cases pass.

The source adaptation uses explicit scalar projections for pinned-compiler record
access. This does not claim a compiler root-cause repair or source-built compiler
validation.

## Delivery and continuation

Local source/log/JUnit/reference/hash/SVG custody:
`/Users/demetriosagourakis/Documents/Sounio-stdlib-sprint-2026-10-16/log-symlog-2026-10-17/`.

Snapshot binding only. Rebuild after editing input/domain/scale. No scaled
overlay/live-domain helper, adaptive curve tessellation, negative-only log API,
automatic nonlinear nice ticks, alpha/wide-stroke contract or Unicode typography.
No runtime/RSS/binary-size gain is measured in this batch.
The previous compact SVG size evidence remains separate.

Next priority returns to statistical assumption/resampling procedures, with
genomics count-matrix, mathematics boundary and physics covariance implementation
tracks preserved in the master sprint. The complete sprint remains in progress.

Conceptual references: https://d3js.org/d3-scale/symlog and
https://d3js.org/d3-scale/log . Implementation, supported domains and tick policy
are independently defined in the Sounio contract, not a full D3 compatibility claim.
