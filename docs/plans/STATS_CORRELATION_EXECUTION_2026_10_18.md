<!-- docs:meta
topic_id: repo.docs.plans.stats-correlation-execution-2026-10-18
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.plans.stats-correlation-execution-2026-10-18
-->

# Checked correlation and scientific scatter execution — 2026-10-18

## Scope and source

Continuation of the same major statistics, scientific visualization, mathematics,
genomics and physics stdlib sprint. Worktree /workspace/sounio-sprint-stats-viz;
original isolated baseline f5b2c5850be9f3d3995719ffcade44a6d851658a.
Remote main refreshed at2b41c82671995278554214398b7caac01b89e524.
Canonical workspace observed on feat/two-readings at2b41c82671 with a local
compiler build; that artifact is excluded. The sprint worktree remains
unrebased, uncommitted and unpushed.

## Implemented

- Checked Pearson and average-rank Spearman, n2..256, explicit size/nonfinite/
  constant/arithmetic statuses, zero unavailable payloads and preserved n.
- Anchored differences preserve representable small spreads on large offsets.
  Raw-max scaling is used only when subtracting finite opposite extremes overflows.
  Compensated centered moments avoid raw covariance-product range failures.
- Explicit scaled projection-residual norm avoids subtracting rounded r² from1.
  A rounded endpoint coefficient can coexist with a nonzero residual and finite t.
- Deterministic index heapsort and averaged equal-value groups replace duplicate
  selection sorting. Index storage is reused; rank buffers are centered in place.
- Separate inference API requires caller opt-in to the Pearson normal model or
  Spearman t approximation. Coefficient validity, p availability, CI availability
  and calibration identity have distinct fields.
- Fisher CI uses stable atanh/tanh and the bounded accepted normal quantile helper.
  Perfect computed residual, unsupported quantile levels and invalid confidence
  are explicit; no cutoff fabricates huge t, p0 or a point CI.
- The public256 correlation APIs and fixed100 inferential Spearman surfaces
  delegate to checked kernels. Fixed100 guards run before copying/ranking.
  Unavailable compatibility p is -1; scalar rho sentinel is -2.
- Added metadata expands public result layouts. Argument signatures are retained,
  but external struct literals/foreign binaries require updating/rebuilding.
  No external constructors were found in the inspected repository.
- Scientific scatter uses actual coefficient outputs and real six raw pairs.
  SVG scatter now reads typed scalar fields directly from the original scene.

[Coefficient contract](../../stdlib/stats/CORRELATION.md);
[inference and compatibility contract](../../stdlib/stats/CORRELATION_INFERENCE.md).

## Native acceptance

Final **15353**, COMPLETED0:0, **3:45**: **77/77 pass**, zero failures/skips/errors.
Raw-centering control **15356**, FAILED1:0,3:45:76pass+1 numerical reference
runtime failure after successful compilation.
Prior-scatter control **15357**, FAILED1:0,3:45:76pass+1 missing scatter-count
stdout failure in the new figure. Both retain every prior73 fixture.

New selected fixtures: stats_correlation_reference, stats_correlation_status,
stats_correlation_legacy, viz_correlation_diagnostics.

| Run | Job | Source archive SHA256 |
|---|---:|---|
| stats-viz-correlation-20261018-v1 |15353|a3c51c76a49708d247d38760ad1e8903c2a87062ae096a9a49020a721039034e|
| stats-viz-correlation-control-scale-20261018-v1 |15356|a5ad40e99763027ff879acaa9be4836f146b1aed36ade9191f4ccc643019b478|
| stats-viz-correlation-control-scatter-20261018-v1 |15357|e9fb0ef66b50eeec07fbae83b79d4205c023631f29df5e139323b3d8950f7eb1|

The scale control replaces only anchored/scaled centering with raw sum/mean,
retaining finite/constant guards to avoid unsafe execution. It is a targeted
range-regression control, not an untouched historical correlation module.
The scatter control restores only the prior renderer function through its saved
full file; every other byte is identical. Each manifest differs at one file.

Pinned committed Madaros v0.80 ELF SHA256:
125c25703ac4062fc6befb7acf77e945abb13405f0adb5ab09fbc5a8d4b5b29f.
Gzip SHA256:
feaaf7fe181f1ee3efd30e7fe23b1709937ea7e93b52e812276fb093b6ccf52a.
Forge/Slurm only: GPUoff,2CPUs,8192MiB ceiling,5minute limit. This is not
a new compiler-source rebuild, full corpus, remote CI or merge acceptance.

## Independent C++23 evidence

g++14.2/libquadmath on t560, C++23. The numerical reference uses raw quad-precision
centered moments and explicit projection residuals, without replicating Sounio
input scaling. Ranks use independent less/equal counts rather than sorting.

64 report rows:29 datasets x2 methods plus6 inference cases.
Includes n2/max256, exact constants, all/partial ties, signed zeros, inactive poison,
active nonfinite, illegal sizes, subnormal/1e-300/1e300 scales, opposite extremes,
1e300 offsets with nextafter spreads, affine/permutation invariances and near fits.
A1e-7 perturbation yields residual_ratio2.82842712011724499e-08;
an ULP perturbation yields1.25607396694702003e-16 while r rounds to1.
The latter checks positivity/order, not an unjustified tight relative error.

Six inference cases independently evaluate df3/df1 Student closed forms,
normal quantile via96 erfcq bisections, then atanhq/tanhq Fisher95 endpoints.
These do not constitute a general Student-distribution oracle or coverage study.
Near-collinear p reference:9.60337399185908450e-24.

Root reran the C++ oracle with exit0 and byte-compared its fixture/report.
Numerical executable SHA256:
5d79e7732e7b30b2e1d19b54b891079b2e54d0837802a3e17d0385a967479c7b.
SVG geometry checker SHA256:
828a9c321492817712649eaa1e825f1e272f4db2c75c4b74b853844b684f974e.

## Native repair loop and real SVG defect

Focused15347 passed status/legacy. Its reference stopped at a generated df
comparison with integer type; the C++ emitter now emits f64 df, while its64-row
numerical report stayed byte-identical. Native reference15351 then passed.

The same first figure executed all native assertions but failed its stdout
scatter-count assertion. Independent readback found genuine corrupted SVG:
the count comment reported95 for6 input observations and extra circles were
emitted after the correct first six. The expectation was retained.

Only vh_emit_scatter was repaired: typed loads of count, layout, colors, domains
and coordinates replace copied-record projections. No other renderer functions,
leading imports, standard-uncertainty metadata or path bodies changed.
Focused15352 then passed all four new fixtures.

A separate C++23 SVG checker independently projects the six raw pairs and checks
exact circle count, coordinates, radius and RGB. It rejects the first SVG with
exit1 and accepts the corrected SVG with exit0. Canvas additionally checks an
axis pixel and the actual (2,4) marker. XML parsing and visual inspection complete
the artifact review. The two coincident (4,16) observations share a visible mark,
which is explicitly labeled; there are six SVG circles, five distinct positions.

Three subagents delivered core/contract, references and figure/renderer repair.
Root owned inference, compatibility integration, status/legacy fixtures, geometry
reference, controls and native/custody gates. Independent review confirmed the
legacy inline-test suffix and unrelated inferential t/ANOVA/effect-size bodies
are preserved. Public data layout expansion is disclosed above.

## Resources and scientific boundaries

Pearson uses two fixed scratch buffers; Spearman uses two rank buffers and one
reused index buffer, without heap allocation. Pearson is O(n); ranking O(nlogn).
Compatibility wrappers retain fixed100-to256 copies. No measured runtime/RSS
improvement or numerical superiority benchmark is claimed.

The descriptive coefficient does not imply a valid p-value or confidence level.
Normal-model assumptions are caller declarations, Spearman t inference is an
explicit approximation, and Fisher intervals are approximate. Exact rank
permutation inference, dependence/cluster calibration, broader resampling and
other private/legacy correlation-family routines remain outside this closure.

The SVG is960x520. QuickLook adds square padding/crops its right edge; it is a
preview, not a Canvas framebuffer capture. The original SVG retains its labels.

## Custody and next work

/Users/demetriosagourakis/Documents/Sounio-stdlib-sprint-2026-10-16/correlation-2026-10-18.
Selected sources/contracts, numerical report, native logs/JUnit/metadata/manifests,
SVG before/after, geometry checks, hashes and updated plan are retained.
No Git history/compiler-source/local-build copies are added to custody.

Next: correlation resampling/permutation calibration and remaining legacy
correlation consumers; then broader scientific layout/data consumers, genomics
count/I/O and physics covariance/invariants within the same major sprint.
No commit/push/PR/merge/publication or whole-sprint completion is claimed.
