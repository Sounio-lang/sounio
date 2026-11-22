<!-- docs:meta
topic_id: repo.docs.plans.scientific-axes-execution-2026-10-17
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.plans.scientific-axes-execution-2026-10-17
-->

# Shared scientific axes execution — 2026-10-17

## Scope and source

Continuation of the original massive standard-library sprint across statistics,
scientific visualization, mathematics, genomics and physics. Statistics and
scientific rendering retain priority. This checkpoint implements explicit
linear axes in the existing Visual IR and both existing renderers.

Remote worktree: /workspace/sounio-sprint-stats-viz.
Baseline remote main: f5b2c5850be9f3d3995719ffcade44a6d851658a.
Remote main rechecked: 92d70d1eb6c178895fc0f72429352b7d11d2db7c.
After fetching main, git diff --name-only HEAD..origin/main over stdlib/viz,
stdlib/stats, stdlib/math/pure.sio, stdlib/genomics and stdlib/physics was empty.
The other agent's live checkout was not modified. No commit, push, PR or merge.

## Implemented contract

- New viz::axis::{axis_ticks, axis_label, axis_offset}: one numeric plan shared by
  Canvas and SVG; 1..8 explicit linear intervals, exact domain endpoints,
  overflow-safe opposite-sign interpolation, six significant decimal digits,
  bounded scientific notation including runtime binary64 subnormals.
- Invalid/nonfinite/decreasing domains, bad interval counts, unrepresentable
  distinct ticks, and adjacent identical formatted labels produce explicit status.
- VIZ_NODE_AXIS and viz_add_axis: horizontal or vertical; transactional rejection
  for invalid plans, geometry, inadequate spacing and exhausted node capacity.
- An axis consumes one existing node slot and no series slot. No VizScene or
  VizSeriesData capacity or structure grew.
- Both backends consume identical values, labels and integer tick placement.
  SVG title content follows existing XML escaping. Canvas uses the 8x8 bitmap font;
  SVG reserves the same horizontal text span, but glyph rasterization differs.
- Consumer: observations [1,2,4,5,8], one-sample t-test against mu0=3, actual 95% CI,
  explicit interval forest, shared horizontal axis with domain from CI endpoints.
  Estimate is 1; CI endpoints are approximately -2.4004369 and 4.4004369.
  The vertical-axis raster test is separate from the categorical forest figure.

Contract and limitations: stdlib/viz/AXES.md.

## Rendering regression and repair

In diagnostic job 14820, the scene's series[0].n was asserted to be 1 after
forest construction, x-axis construction, title assignment and y-axis construction.
The SVG nevertheless emitted forest n=166 and extra rows; the native Canvas
marker check failed. Merely replacing the local series copy by a reference did
not solve the failure (job 14822).

The accepted source reads the row count as an explicit i64 directly from the scene
before rendering and uses that scalar consistently in both forest renderers.
Series fields are borrowed rather than copying the whole array-bearing record.
The same native program then preserves the single-row geometry and marker.

A controlled before package retained the accepted fixture and changed only the
two forest renderer functions back from the explicit row-count scalar to d.n.
Job 14831 produced 12 passes and 1 failure (the scientific-axis consumer).
The accepted after package produced 13 passes and 0 failures.
This is evidence for the bounded source repair with the pinned compiler;
the underlying current-source compiler root cause has not been established.

The fixture also exposed the pinned compiler turning the literal
4.94065645841246544e-324 into zero. The subnormal formatting check now constructs
the smallest subnormal at runtime as smallest_normal / 4503599627370496.0,
and checks its label against C++23. No compiler fix is claimed.

## Independent C++23 reference

tests/reference/scientific_axis_reference.cpp generates expected labels using
C++23 scientific stream formatting at six significant digits, with exponent and
zero normalization matching the explicit display policy. It generates tick
values with long-double convex interpolation. The fixture compares four domains:
[-1e308,1e308], [1e-300,9e-300], [-7,13], and [1e300,1.5e300], at relative
tolerance 1e-13. The zero reference requires exact zero.
Label cases include zero, negative zero, signed fractions, +/-decimal exponent
ranges, rounding carry at 9.999999 and the smallest runtime subnormal.

Reproduce the generated assertion helpers:
c++ -std=c++23 -O2 tests/reference/scientific_axis_reference.cpp -o /tmp/axis-reference
/tmp/axis-reference > /tmp/axis-reference.sio

The checked fixture adds diagnostic printing to the generated assertions.
Its native checks also cover nonfinite input, duplicate-label rejection,
transactional rejection, 64-node exhaustion, no series-slot consumption,
exact Canvas marker/tick/glyph pixels, SVG label values and XML escaping.
No Python, Rust or Julia numerical reference was used.

## Native acceptance

| Run | Slurm | State | Focused JUnit |
|---|---|---|---|
| stdlib-axes-20261017-before | 14831 | FAILED 1:0; expected regression | 13 tests, 1 failure, 0 skips/errors |
| stdlib-axes-20261017-final | 14832 | COMPLETED 0:0, 46 seconds | 13 tests, 0 failures/skips/errors |

Earlier accepted source checkpoint: stdlib-axes-20261017-k, job 14828,
13 tests, zero failures/skips/errors.
Intermediate runs caught an incorrect borrowed title argument, a diagnostic
helper missing IO, the subnormal literal issue, and the rendering regression;
they are not acceptance receipts.

Final snapshot SHA-256:
d20d628bfc69e0ea375f4ffaeeb5fd6e52f74ccd7e97e2425f2c03bcb6bafd76

Committed Madaros v0.80.0 SHA-256:
125c25703ac4062fc6befb7acf77e945abb13405f0adb5ab09fbc5a8d4b5b29f

Slurm: CPU only, 2 CPUs, 8192 MiB, GPU off, clean-v1 environment, actual UID/GID 1000.
This uses the committed pinned compiler, not a rebuild from current compiler sources.
The runtime package MANIFEST.sha256 was regenerated and sha256sum -c passed
before final acceptance. Snapshot identity, not the standalone package's empty
Git metadata, identifies the executed source.

Thirteen tests retain the previous twelve multi-domain cases and add
viz_scientific_axes. This is not a whole-stdlib, full corpus, CI, compiler
fixed-point or current-source rebuild certification. No speed/RSS gain was
measured for the borrowed series reads; no performance claim is made.

## Core source hashes

62ffc4869f84176f6e4ea35d9ff9b30ce04ff6742eb1eaf09a71789b15594e2f  stdlib/viz/axis.sio
dbdd8130335db726b86740b789237da2127879d159798e3e789d56190ba7ef4e  stdlib/viz/ir.sio
d6ed713ed446a6c82ef714db390af48dfdb8811ac3252440e0a4b0836c23e893  stdlib/viz/viz_canvas.sio
69a9b647512e425dfd81d91e314113f02ba17d0d9ba77202d75953ec20970c87  stdlib/viz/viz_html.sio
e4aedca1474f49069a7894634622d82a3ada657aa2796408ea8417fcb531cb1b  tests/run-pass/viz_scientific_axes.sio
38fa9cf105d53d98c7e7f61e9a3dccc37dbc9ea3229223ecc006dbf5ae774077  tests/reference/scientific_axis_reference.cpp

## Artifacts and continuation

Native outputs:
 /var/lib/cephfs-slurm/training/sounio-foundry/runs/stdlib-axes-20261017-final

Local custody:
 /Users/demetriosagourakis/Documents/Sounio-stdlib-sprint-2026-10-16/scientific-axes-2026-10-17

inferential-scientific-axis.svg is extracted from the final native stdout,
not reconstructed in JavaScript. XML structure and a Quick Look preview were
inspected. The preview PNG is only a thumbnail; SVG is the figure artifact.

Next: clipping, legend and shared chart-domain transforms; automatic nice ticks,
log/symlog and narrow-domain formatting; regression DFFITS and honest diagnostics;
then figures using real input paths for statistics, count data and physics.
The existing packaging/bundle infrastructure remains supporting work.
