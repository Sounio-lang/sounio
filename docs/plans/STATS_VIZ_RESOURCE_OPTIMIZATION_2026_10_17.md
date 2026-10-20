<!-- docs:meta
topic_id: repo.docs.plans.stats-viz-resource-optimization-2026-10-17
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.plans.stats-viz-resource-optimization-2026-10-17
-->

# Execution, memory and artifact optimization — 2026-10-17

## Measured now

Disk usage is rounded output from du, not process memory:
- Isolated development worktree: 931 MB.
- Previous focused runtime package on t560: 104 MB.
- New focused runtime package in /workspace/stats-viz-runtime-20261017: 42 MB.
- Previous final snapshot: 22 MB compressed.
- New archive /workspace/stats-viz-runtime-20261017.tar.gz: 8.8 MB compressed.
- Existing stats-viz source snapshots on t560: 471 MB total.

New archive SHA256:
4bc2dde21353a60a2b24d5c4427c701300fbd87bec89563c0aaa91963da51b2a

## Implemented

scripts/dev/package_stats_viz_runtime.sh creates a new, runtime-only directory
from this worktree. It includes the full stdlib, pinned Madaros gzip/checksum,
existing launchers/resolvers, suite scripts, four selected fixtures and two C++23
references. Each fixture has a content-identical basic_math-prefixed alias for
the fixed Foundry corpus-smoke selector.

It excludes historical ELF backups, legacy engines, unrelated tests, compiler
source, docs and prior results. It refuses an existing destination or a target
within the source checkout. There are no deletion or compiler execution steps.
SOURCE.txt records baseline/scope; MANIFEST.sha256 identifies payload bytes;
PACKAGE_READY appears only after copying and manifest generation complete.

The compiler gzip was decompressed only into a hash stream. Its payload remains:
125c25703ac4062fc6befb7acf77e945abb13405f0adb5ab09fbc5a8d4b5b29f

No compiler build or test suite was run for the reduced package. Historical job
14679 validates the earlier package, not acceptance of this new dependency set.

## Memory: next bounded changes

The current source stores five fixed 1024-element f64 arrays per series, with
eight series slots in every VizScene. The explicit-interval constructor also
allocates rows and no_variance temporary arrays. Several render paths bind a
series by value. These are source-level candidates, not measured allocations:
compiler lowering may eliminate some copies.

Order:
1. Capture separate peak RSS and elapsed time for compilation and generated
   program execution on Slurm, with the pinned compiler and identical fixture.
2. Remove the interval constructor's temporary row/variance buffers by filling
   a validated destination slot directly. Preserve initialization and rejection
   atomicity, and compare Canvas pixels / SVG output before and after.
3. Inspect generated code before changing render bindings to borrowed access.
4. Consider a sparse interval pool only if measured savings justify the API,
   layout and capacity changes. Do not reduce public capacities speculatively.

The 8192 MiB requested for the earlier Slurm job is a reservation, not measured
application RSS. Earlier parallel compiler execution exhausted 4096 MiB; this
does not establish that visualization execution itself needs 8192 MiB.

## Artifact policy

Use Foundry's existing --reuse-snapshot for identical source bytes. For changed
payloads, use one package builder instead of ad hoc copying the checkout.
Keep successful acceptance snapshots and the relevant failing-before evidence.
A future retention operation must enumerate exact directories and preserve
logs, receipts and source identity before removing duplicates. Nothing removed.

## Source boundary

Development remains isolated at /workspace/sounio-sprint-stats-viz, baseline
f5b2c5850be9f3d3995719ffcade44a6d851658a. Remote main now resolves to
6861acfdf75c29e1174a9f668dea2877020f6012. No rebase or publication performed.
