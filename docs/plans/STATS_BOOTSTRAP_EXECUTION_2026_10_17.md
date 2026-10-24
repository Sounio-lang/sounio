<!-- docs:meta
topic_id: repo.docs.plans.stats-bootstrap-execution-2026-10-17
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.plans.stats-bootstrap-execution-2026-10-17
-->

# Seeded percentile bootstrap — 2026-10-17

## Accepted native boundary

Run `stats-viz-bootstrap-20261017-final3`, Slurm **14939**.
Snapshot `9ee6d4d691980b08920e79c5b036bff21bccd3077bf9c2482d10224edbd606cb`.
JUnit **27 tests, 0 failures, 0 skips, 0 errors**.
Job COMPLETED, exit 0:0, elapsed 00:01:32. Final snapshot manifest matches the completed source package.
CPU 2, memory 8192 MiB, GPU off, 5-minute limit, UID/GID 1000/1000.
Pinned committed Madaros v0.80 ELF:
`125c25703ac4062fc6befb7acf77e945abb13405f0adb5ab09fbc5a8d4b5b29f`.
This is not a compiler rebuilt from current sources.

Worktree: `/workspace/sounio-sprint-stats-viz`.
Original remote-main baseline `f5b2c5850be9f3d3995719ffcade44a6d851658a`.
Remote main reconfirmed `106f7a988d5b007472d0eead724f58ac50f92200`.
The canonical active checkout on another branch was not modified.
Changes remain uncommitted; no push, PR, merge or remote CI acceptance.

## Implemented

- `stats/bootstrap.sio`: borrowed bounded inputs; explicit size/confidence/data/arithmetic
  status and normalized seed metadata; no silent resample truncation.
- Integer MINSTD recurrence with signed-remainder seed normalization, including
  modulus multiples and i64 minimum, without negating i64 minimum.
- Rejection mapping for random indices, replacing the biased reduction.
- Compensated normalized point/resample means, normalized Welford sample variance,
  shared robust sqrt, and finite-result validation.
- Interpolated type-7 empirical percentile interval.
- In-place heapsort: O(n*b + b*log(b)) source work bound; one 1024-double buffer,
  no sample copy/heap allocation. No empirical speedup or RSS/stack claim.
- Real bootstrap results feed a two-row forest, exact lower/upper bounds and
  confidence metadata, a shared-domain labeled linear axis, Canvas pixels and SVG.
- Packaging helper now retains all 27 accepted fixtures plus the C++ reference.

## Independent reference

`tests/reference/bootstrap_reference.cpp`, compiled with
g++ 14.2.0, `-std=c++23 -O2 -Wall -Wextra -pedantic`.
Reference uses long-double accumulation, two-pass sample variance and std::sort,
rather than the Sounio Kahan/Welford/heapsort implementation. The same specified
integer random stream makes pathwise comparisons reproducible.

Ten datasets cover ordinary data; seed equal to the modulus; negative seed;
i64-min seed; values around 1e308; values around 1e-300; cancellation; constants;
two resamples; and n=256/b=1024. Every dataset checks mean, SE, lower, upper,
resample count and retained caller data. Normalized absolute tolerance 3e-12.
The separate status fixture checks invalid size/budget/confidence, NaN/inf,
zero arrays and seed normalization equivalence.

Reference source SHA256:
`6d6c850801c233ddf39207aed1b9408854819d48fe0278b5679fdae2bc13e10e`.
Reference executable SHA256:
`2f7e7b8c800c69b275323351610cec2c8dc55de18189a7da57306a45e6067a88`.
Generated fixture SHA256:
`3cc8cdc94c1bf5afeb1623e1a20bf6e7a77c46ace83a2136547750a0a7075d20`.

## Failing before / passing after

Controlled before run `stats-viz-bootstrap-20261017-before`, Slurm **14929**:
24 tests, 23 pass, one fail, no skips/errors.
Snapshot `6ed594014f30a3a52b6dae4df7fe4842096c1d2c19adff3cd725befc51a8a08d`.
Uses the untouched former `stats/bootstrap.sio` and the exact same
`stats_bootstrap_seed.sio` as the accepted run. Compilation succeeds, runtime
exits 1: seed=2147483647 collapses its f64 recurrence to zero and produces a
zero SE for the nonconstant sample [1,3]. The first assertion requires positive SE.
The accepted implementation passes this fixture, including its additional
finite-mean/SE check for [1e308,1.2e308].

Earlier job 14917 passed 25/25. Expanded job 14927 passed 26/27: the new
figure's declared ASCII label length truncated its last character, so its stdout
assertion failed. That fixture length was corrected before 14939.
Job 14931 subsequently passed 27/27, before the stricter RGB assertions were added.
14927 is not an accepted checkpoint; its packaged manifest also predates completion
of its regeneration. Final acceptance uses the completed final3 manifest; its archived content matches the source package.

## SVG style regression discovered during visual review

The accepted bootstrap example exposed a copied-node style projection problem
in the pinned backend: the IR and Canvas preserved blue=240/210, while the
SVG forest markers and axis emitted blue=1. Explicit typed scalar reads from
the original scene in vh_emit_forest and vh_emit_axis preserve all channels.

The strengthened figure fixture checks both original IR blue fields, an exact
packed Canvas marker pixel, SVG fill="rgb(80,180,240)" and
stroke="rgb(180,190,210)", as well as the label and confidence metadata.
Controlled renderer-before job **14940** compiles/runs but fails the SVG RGB
stdout assertion: 27 tests, 26 pass, one fail, zero skips/errors.
Snapshot:
`c22961bb107698575a82cd85aba3d4252863d6ca0329a70f5d3dcfbaacf291cf`.
Final job **14939** passes the identical strengthened fixture.
The candidate and style-control manifests differ only in viz/viz_html.sio.
This is a renderer compatibility repair for these paths, not a compiler repair
or certification of every other copied-style projection.

Standalone SVG custody extracts only the <svg>...</svg> element from the
renderer HTML stream; </body></html> is not part of the SVG artifact.
Quick Look inspection is an SVG preview, not a Canvas framebuffer export.

## Contract and remaining work

See `stdlib/stats/RESAMPLING.md`. This is an ordinary percentile interval
for iid empirical resampling, not BCa, studentized bootstrap or guaranteed
finite-sample coverage. Monte Carlo error remains; the bounded resample capacity
does not establish tail adequacy. MINSTD is a finite 31-bit linear generator,
not a claim of best available RNG quality. Scaled f64 arithmetic is not exact
or correctly rounded; tiny contributions may be lost and nonzero subnormal
results may round to zero.

The forest builder still requires the estimate inside the supplied interval;
a percentile interval can violate that in general. This consumer explicitly
checks containment and never adjusts the statistical bounds.

Other legacy resampling, BCa, correlation/difference bootstrap and permutation
modules are not certified by this checkpoint. The master sprint retains
statistics, visualization, mathematics, genomics, physics and execution/memory/
artifact optimization as separate implementation tracks.
