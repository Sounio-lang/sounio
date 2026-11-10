<!-- docs:meta
topic_id: repo.docs.plans.statistics-visualization-execution-2026-10-16
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.plans.statistics-visualization-execution-2026-10-16
-->

# Statistics and scientific visualization: execution receipt, 2026-10-16

## Source and scope

Baseline: remote main at f5b2c5850be9f3d3995719ffcade44a6d851658a.
Worktree: /workspace/sounio-sprint-stats-viz (detached). Changes remain uncommitted.
The live /workspace/sounio checkout was not used as the validation source.

This checkpoint delivers a bounded statistics-to-forest-rendering slice and one
corrected descriptive statistic. It does not establish the full sprint or the
whole stdlib as SOTA, publication-ready, or fully validated.

## Implemented contracts

- VizSeriesData carries explicit lower and upper bounds, interval kind
  (confidence, prediction, credible), and level in (0,1).
- viz_add_forest_interval_plot rejects invalid count/capacity, kind, level,
  non-finite values, and lower > estimate or estimate > upper before mutation.
- Canvas and HTML/SVG consume the same supplied bounds. The existing variance
  forest API retains its one-sigma interpretation.
- Canvas estimate coordinates now explicitly bind f64 values and cast the
  normalized offset to i64 before adding the pixel origin. The preceding mixed
  expression failed the marker pixel checks on the tested committed compiler.
- stats::descriptive::mad computes the exact raw median of absolute deviations
  about the supplied median. It exposes a public API with explicit Mut/Div/Panic
  effects, a finite-input contract, and capacity 1..100. It uses bounded insertion
  sort and half-first averaging of the two central values to avoid sum overflow.
  This raw API differs from stats::robust::mad, which applies normal-consistency
  scaling; the existing robust API was not changed.
- The test harness saves the full captured output as result_<index>.log whenever
  SOUNIO_TEST_RESULTS_DIR is configured. Passing and failing runs remain auditable.

## Final executable evidence

Foundry run: stats-viz-forest-20261016-17
Slurm job: 14665
Accounting state: COMPLETED, ExitCode=0:0, Elapsed=00:00:12.
Resources: 2 CPUs, 8192 MiB, cpu-ops, UID/GID 1000.
JUnit: 3 tests, 0 failures, 0 skipped, 0 errors.

Selected tests (the package adds identical basic_math-prefix aliases for the
fixed Foundry corpus-smoke filter):

1. tests/run-pass/stats_descriptive_mad.sio
2. tests/run-pass/viz_chart_builders.sio
3. tests/run-pass/viz_statistical_interval_forest.sio

The statistical test checks six CI endpoints against independent C++23 results,
preservation of interval bounds/kind/level in the IR, invalid-input atomicity,
eight exact positive/negative pixel checks, and required SVG interval metadata
and asymmetric line coordinates. Exported confidence level 0.950000 and
prediction level 0.800000 were also read back from the standalone SVG files.

Source snapshot SHA-256:
793a279b7bbe77acb6688a8b3fcf6ee07a6315b6231b123e1e5223e4e6b95146

The final job reuses the identical snapshot from run -16; that earlier run hit
the 4 GiB memory limit while compiling two visual fixtures in parallel.
Increasing the allocation to 8 GiB completed all three tests without source
changes.

Compiler: committed Madaros v0.80.0 prebuilt, materialized and digest-checked
from the baseline repository's compressed artifact. No current-source compiler
rebuild was performed in this checkpoint.
Raw compiler SHA-256:
125c25703ac4062fc6befb7acf77e945abb13405f0adb5ab09fbc5a8d4b5b29f

The harness summary labels its legacy engine resolver as lean_single; the
actual bin/souc orientation and per-test logs identify the invoked Madaros
compiler above. The metadata label should be corrected in a separate harness
identity change; it is not evidence of a lean_single execution here.

The snapshot contains VALIDATION_SOURCE_MANIFEST.txt with baseline identity,
package scope, test alias mapping, and per-file hashes. It is a focused runtime
fixture package rather than a full checkout; Foundry's null Git fields should
not be interpreted as a clean versioned repository.

## Independent numerical reference

Source: tests/reference/statistical_interval_reference.cpp
Language/toolchain: C++23, GCC 14.2.0, remote t560 control host.
Command: g++ -std=c++23 -O2 -Wall -Wextra -pedantic <source> -o <reference>
The small reference was compiled and executed on t560 after a separate Slurm
C++ job found no g++ installed on its worker. Sounio tests ran on Slurm.

The reference integrates Student's density in long double using adaptive Simpson
quadrature and inverts the CDF by bisection. It therefore uses a numerical path
independent of Sounio's incomplete-beta inversion. Analytic Cauchy and df=2
identities check the reference itself. Sounio endpoint acceptance tolerance:
absolute error < 1e-6 for these three fixtures only.

| mean | sd | n | lower (95%) | upper (95%) |
| --- | --- | --- | --- | --- |
| 10 | 2 | 10 | 8.5692861880586702 | 11.4307138119413298 |
| 15 | 3 | 12 | 13.0938909380080802 | 16.9061090619919198 |
| 8 | 1.5 | 9 | 6.84699793239791666 | 9.15300206760208334 |

C++ MAD reference: odd/outlier=1, even=1, singleton=0, affine case=4,
capacity-100 case=1. Sounio also checks permutation invariance and input retention.

Reference source SHA-256:
eeb765dacc954a139d89eea6daceec7a5f795a41c8ca9de4d9b8f2eb556d8c9f
Reference executable SHA-256:
fa31ec3e275331c5b6290f89e5c16315ae444af219644573f8d8a7ced619b257
Reference output SHA-256:
8d941dd2c38ffbf51fa44aa810e169f860b37ac0ef3800d00890dbe6a955aa88

No Python, Rust, Julia, NumPy, or SciPy numerical oracle was used.

## Failing-before / passing-after evidence

- MAD: run stats-viz-forest-20261016-09-before-mad, job 14653, compiled the
  previous mean-of-deviations body with only public visibility/effects added;
  the new MAD regression exited 1. The corrected median algorithm passed in
  job 14655 and again in final job 14665.
- Canvas: jobs 14659/14660 confirmed correct primitive color controls but zero
  pixels of the required estimate-marker color. The explicit f64-to-i64
  coordinate path passed in job 14661 and in final job 14665.
- The first inherited run, job 14579, failed before compilation because the
  reduced package omitted scripts/lib/resolve_madaros.sh. Later packages contain
  the complete scripts/lib dependency set.

## Artifacts

Durable run root on t560:
/var/lib/cephfs-slurm/training/sounio-foundry/runs/stats-viz-forest-20261016-17
Worker-visible run root:
/orangefs/training/sounio-foundry/runs/stats-viz-forest-20261016-17
Reference root:
/var/lib/cephfs-slurm/training/sounio-foundry/runs/stats-viz-cpp-reference-20261016

Read summary.json, results.tsv, test-suite-junit.xml, runtime-receipt.json,
test-results/result_*.log, and source.tar.gz for full evidence.

Standalone SVG fixtures were extracted without redrawing from result_3.log:
- student_t_confidence.svg:
  a587f91587d379dfcb49a35c23aa7eacb4185a63641b081a01a945f34f2e9e84
- asymmetric_prediction.svg:
  06b083a35707f853bf6bc71a130abac7ffc35516a58ea0493af8d01fe5abc77c

These are verification fixtures, not a publication layout or a completed gallery.

## Open work

The broader sprint remains open: exact inferential tails, Shapiro-Wilk
coefficients and validity domain, DFFITS/linearity diagnostics, uncertainty
result contracts, axes/ticks/labels/legends/clipping, publication layouts,
missing-data policies, capacities, performance, and genomics/physics cases.

The existing viz_audit_hash summarizes interaction state; viz_audit_node_hash
summarizes node identity/layout; viz_app_frame_hash samples three pixels.
They are not full data/specification/content digests. A versioned scientific
receipt must cover input data, interval semantics, parameters, and output bytes
before general reproducibility claims rely on those helpers.

There was no full-corpus run, remote CI, commit, push, PR, merge, or release.
