<!-- docs:meta
topic_id: repo.docs.plans.stats-normality-execution-2026-10-18
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.plans.stats-normality-execution-2026-10-18
-->

# Checked moment normality — 2026-10-18

## Scope and implementation

Repaired the existing public stats::normality Jarque-Bera procedure.
Original function and five numerical fields remain; status, n and asymptotic
metadata are added. Invalid sizes, nonfinite values and constant samples no
longer masquerade as p=1. Callers must check status before numerical payload.

One fixed [f64;256] scratch array, anchor-preserving differences, overflow-safe
fallback normalization, centered rescaling and compensated sums replace raw
powers of unscaled observations. The exponential evaluates a negative argument
directly rather than overflowing a positive reciprocal. Storage is bounded;
no speed, allocation reduction or SOTA-superiority benchmark is claimed.

The chi-square(2) reference tail is asymptotic. This work does not calibrate
small-sample inference, assess independence, or map p to a Beta posterior.
The legacy epistemic-assumptions header is corrected to identify its private
pedagogical coefficients and bucketed p-values as uncertified heuristics.
Those routines are not certified by this batch.

## Independent scientific reference

tests/reference/normality_reference.cpp compiled on t560 using
g++ -std=c++23 -O2 -Wall -Wextra -pedantic.
Long-double raw central moments, standard-library pow/exp, no candidate
normalized-moment kernel reuse. An exponent-range static_assert prevents
silently using a double-width long double on an unsuitable host.

Twelve cases cover ordinary asymmetric data; positive/negative 1e-200 and 1e200
unit changes; 1e15 offsets; opposite-sign finite extremes whose direct
subtraction overflows; subnormal inputs; n=2 and n=3; symmetric n=256; and a
single outlier among 256 observations with expected p underflow to zero.
All available result fields are checked against generated reference literals,
with explicit absolute/relative tolerances. Native status checks include empty,
oversized, constant, infinity and NaN data, and retain original inline tests.
A separate unchanged-API fixture checks unit invariance.

## Native receipts

Final candidate run stats-viz-normality-20261018-v2, Slurm **15177**:
COMPLETED **0:0**, elapsed **00:01:56**.
JUnit **40 tests, 0 failures, 0 skips, 0 errors**, preserving all prior 37.
Snapshot:
6673f037e5df6e154768ab6afe7a1415014360fafa799b6553aab406822ada29.

Final untouched-body control stats-viz-normality-control-20261018-v2,
Slurm **15178**: FAILED **1:0**, elapsed **00:01:51**.
JUnit **38 tests: 37 pass, 1 runtime failure, 0 skips, 0 errors**.
The units fixture successfully typechecks and compiles, then exits 1 on the
first p-value unit-invariance assertion. The original 37 fixtures pass.
Snapshot:
8001a64c408230745c341c2495b6dccd1ffc3da60b32879d5cf6207696680196.
No effects, visibility or numerical-body repairs were made to the control.
Only the common-field units fixture is added; the new metadata/reference tests
cannot compile against the former result shape and are excluded from control.

Pinned committed Madaros v0.80 ELF SHA256:
125c25703ac4062fc6befb7acf77e945abb13405f0adb5ab09fbc5a8d4b5b29f.
2 CPUs, 8192 MiB, GPU off, five-minute limit, clean-v1, UID/GID 1000/1000.
JUnit test-suite counts are authoritative, not the three harness phases.

Initial candidate 15172: COMPLETED 0:0, 00:01:55, 40/40.
Initial untouched-body control 15173: 38 tests, 37 pass; new units fixture
compiled and timed out after 30s on the large-value path before assertions.
This is timeout evidence, not a completed numerical assertion failure.
For the final pair only the assertion order is changed: compare tiny-unit p
before calling the old huge-unit path. The old normality source is untouched.

## Source and main boundary

Isolated worktree: /workspace/sounio-sprint-stats-viz.
Original remote-main baseline: f5b2c5850be9f3d3995719ffcade44a6d851658a.
Remote main rechecked: e00a5d537cd0c6d01762c38fcd70cc9739062710.
Latest c4f1d3ce..e00a5d53 advance changes compiler/self-hosted/CI/website/docs
but no stdlib files, committed Madaros gzip or its resolver.
The prior intervening stdlib changes remain outside this focused closure.
This is not a compiler rebuilt from current main. Canonical active checkout
and its separate local compiler are untouched.

Archived package manifests match the submitted packages. Selected implementation,
reference, test and packaging files in local custody are checked against the
final candidate manifest. Contract/status/master and the legacy header warning
are documentation follow-ups outside the executable snapshot.
No commit, push, PR, merge or remote CI acceptance is claimed.

## Continued sprint

Next: checked Koenker/Breusch-Pagan heteroscedasticity with a direct
chi-square survival, scaled squared-residual auxiliary regression and explicit
unavailable states, then a real model-level diagnostic and scientific figure.
Shapiro-Wilk needs genuine coefficients and a defined calibration/domain;
it must not reuse this Jarque-Bera name or the old heuristic p-to-Beta mapping.
Broader genomics/physics numerical and data-I/O tracks, scientific layout,
nonlinear ticks, other-mark clipping and measured scalability remain in scope.
