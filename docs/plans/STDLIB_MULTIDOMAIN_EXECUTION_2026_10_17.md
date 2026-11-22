<!-- docs:meta
topic_id: repo.docs.plans.stdlib-multidomain-execution-2026-10-17
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.plans.stdlib-multidomain-execution-2026-10-17
-->

# Scientific stdlib multi-domain execution — 2026-10-17

## Mandate and source

This is an implementation checkpoint of the original large stdlib sprint:
statistics, scientific visualization, mathematics, genomics and physics.
Performance tooling supports that scope; it does not replace it.

Isolated worktree: /workspace/sounio-sprint-stats-viz.
Baseline remote main: f5b2c5850be9f3d3995719ffcade44a6d851658a.
Remote main rechecked at 6861acfdf75c29e1174a9f668dea2877020f6012.
No changes were made in the other agent's canonical live checkout.

## Implemented

### Statistics

Four formerly private legacy t-test entry points are now public with a public
TTestResult: one sample, pooled independent samples, Welch and paired samples.
Results include method identity, sample sizes, estimate, SE, df, p, effect
standardization, fixed confidence level and explicit status.

Welford moments replace naive summation in these public paths. Welch degrees
of freedom use normalized variance weights rather than a fourth-power expression
with an absolute epsilon. Small standard errors are not silently treated as zero.

95% confidence bounds now use the Student quantile at the actual degrees of
freedom, rather than critical=2. One-sample bounds concern mean(data)-mu0.
The paired effect is standardized by SD of differences; the Welch effect uses
sqrt((variance_x+variance_y)/2). The method field disambiguates these choices.
See stdlib/stats/T_TESTS.md for all status and capacity contracts.

### Mathematics

- sqrt reduces both small and large positive arguments into [1,4), then rescales.
  The prior routine reduced only small arguments and was inaccurate for large ones.
- sin uses its direct series, preserving relative accuracy near zero rather than
  evaluating cos(x-pi/2). sin and cos are public shared kernels.
- is_finite is a shared predicate including the pinned backend's NaN guard.
- Student df=1 quantiles use the analytic Cauchy cotangent relation, bypassing the
  known a=1/2 inverse-beta limitation. Existing df>=2 paths remain.

Nonpositive sqrt inputs retain the legacy zero sentinel. This checkpoint does
not establish all-f64 trigonometric range reduction or arbitrary extreme tails.

### Genomics

genomics::expression was a disabled one-line module. It now contains streaming,
two-pass TPM normalization from supplied counts and effective lengths, with
compensated summation, sample-wide invalidation and observable result status.
There is no fixed per-array capacity in the normalizer.

This is normalization, not a read-assignment, effective-length estimation,
alignment, or differential-expression engine. Both passes must describe the
same sample and lengths must use a common unit.

### Physics

physics::scattering::electron_compton_shift uses h/(m_e*c) and
2*sin(theta/2)^2, replacing the dimensional error and low-order cosine in the
legacy Compton path. Canonical physical constants supply the nominal values.
The returned uncertainty is the first-order angular contribution only.
Mass uncertainty and higher-order nonlinear propagation are separate work.

physics::sr::compton_shift delegates to this kernel and retains its tuple API.
Its units import now uses the existing units::lib surface; its epistemic bridge
uses the import-safe free-function API.

### Scientific visualization

A new executable consumer computes a public one-sample t-test from observations,
passes its CI and confidence level into the existing explicit-interval IR,
checks exact Canvas pixels, and emits SVG with confidence metadata and geometry.
This extends the previously validated MAD / interval / Canvas / SVG work.

## Independent C++23 reference

tests/reference/multidomain_reference.cpp contains the independent density
quadrature/CDF inversion from the earlier C++23 reference, standard-library
long-double trigonometry/square root, and count/length TPM reference arithmetic.

Build:
    g++ -std=c++23 -O2 -Wall -Wextra -Werror multidomain_reference.cpp -o reference

Regenerate five fixtures:
    ./reference stats_t_test_ci.sio genomics_tpm.sio physics_compton.sio \
      math_cauchy_small_angle.sio math_sqrt_range.sio > reference.tsv

Reference inputs include:
- one-sample, pooled, fractional-df Welch and paired contrasts;
- n=2 / df=1, translated inputs, scale 1e-12 and scale 1e50;
- Cauchy p={.01,.025,.25,.75,.975,.99};
- square-root arguments from 2.2250738585072014e-308 through 1e300;
- tiny tangent arguments +/-1e-12 and 1e-8;
- Compton angles 0,1e-12,1e-8,pi/6,pi/2,pi with angular u=.01;
- unequal transcript lengths, total TPM, count-scaling invariance, 5,000 streamed
  records and a Kahan stress case with one large plus 10,000 small rates.

Final per-value relative tolerances:
- t-test results: 2e-7;
- TPM, Cauchy/small-angle math, sqrt, Compton: 1e-12;
- relative comparisons include a 1e-290 absolute floor for zero-scale cases;
- angular uncertainty at pi uses an absolute bound because its derivative is zero.
These are fixture acceptance tolerances, not measured maximum errors or universal
accuracy guarantees.

## Failing-before evidence and integration history

- 14784 / stdlib-massive-20261017-01: initial nine-test multi-domain batch passed.
- 14786 / stdlib-massive-20261017-before: old t CI and Cauchy consumers reached
  their numerical assertions and exited 1. Only old API visibility prerequisites
  were adjusted, not the numerical implementations.
- 14787 / -02: ten of eleven tests passed; the legacy SR adapter exposed an
  unresolved old units import. Canonical import/free-function use repaired it.
- 14788 / -03: all eleven tests passed, including SR and inference-to-forest.
- 14791 / -04: an editing error truncated the math module while replacing sqrt;
  the expanded suite detected the missing declarations/runtime failures.
  This was an implementation mistake, not a compiler regression. The full saved
  module was restored and the replacement guarded by explicit marker/integrity
  checks; no partial module was accepted or published.
- 14796 / -05: all twelve tests passed after the full-module restoration.
- 14794 / stdlib-massive-20261017-before2: old sqrt and Compton consumers reached
  numerical assertions and exited 1. Units import and epistemic-call prerequisites
  were normalized; the Compton fixture added a diagnostic entry marker.
  These jobs complete successfully because they require the intended legacy
  numerical failures. They are not acceptance runs of the old implementations.

The baseline source archive was reused through hard links; receipts retain
the exact prerequisite transformations and fixture overlays.

## Final acceptance

Foundry run: stdlib-massive-20261017-final.
Slurm job: 14799, COMPLETED, exit 0:0, elapsed 36 seconds.
CPU 2, memory reservation 8192 MiB, no GPU.
JUnit: 12 tests, 0 failures, 0 skipped, 0 errors.

Snapshot SHA256:
2019af3723a16e4de7ddcafae98ee51e0a25a01d2693d9244bda02cf9582cc63

Compiler: committed Madaros v0.80.0 prebuilt, materialized from gzip.
SHA256:
125c25703ac4062fc6befb7acf77e945abb13405f0adb5ab09fbc5a8d4b5b29f

Accepted entries:
- genomics_tpm
- math_cauchy_small_angle
- math_sqrt_range
- physics_compton
- physics_sr_compton
- stats_descriptive_mad
- stats_inference_forest
- stats_inferential_tails
- stats_t_test_ci
- stats_t_test_status
- viz_chart_builders
- viz_statistical_interval_forest

This is a focused runtime package, not a complete Git checkout. Git fields in the
run packet do not establish identity. MANIFEST.sha256 and the source snapshot
bind the payload; content-identical basic_math aliases select the focused cases.
The updated package builder now includes these twelve entries and the new C++23
reference. No current-source compiler rebuild, full corpus, remote CI or merge.

## Open sprint work

Next: shared scientific scales, ticks, labels, legends and clipping in Canvas
and SVG; publication layouts and input/specification provenance; regression
DFFITS and statistically defined assumptions; Shapiro-Wilk/resampling domains;
genomics count/variant input paths; physics covariance/uncertainty and invariants.

The public t-test adapters still accept fixed arrays up to 100 observations.
No arbitrary missing-data policy, all-f64 numerical accuracy, complete RNA-seq
pipeline, general physics certification, or publication-quality figure system
is claimed by this method-level acceptance.

The master scope and continuation order remain in
STATISTICS_SCIENTIFIC_VISUALIZATION_SPRINT.md.

## Primary formula references

- NIST confidence limits for the mean:
  https://www.itl.nist.gov/div898/handbook/eda/section3/eda352.htm
- NIST difference-of-means / Welch confidence limits:
  https://itl.nist.gov/div898/software/dataplot/refman1/auxillar/diffmean.htm
- NIST Compton wavelength (nominal constant):
  https://physics.nist.gov/cgi-bin/cuu/Value?ecomwl
- Salmon output definitions for counts, effective lengths and TPM units:
  https://salmon.readthedocs.io/en/latest/file_formats.html

## Continuation checkpoint — signed bars and artifact offload, 2026-10-20

Main d7077ef1 is incorporated. Baseline Slurm15712 accepts111/111; final
Slurm15718 accepts114/114 with zero failures/skips/errors after complete HTML
recovery and a margin correction that preserves scientific minus signs.
Both renderers share checked bar rectangles, actual x positions, a true zero
baseline and whole-footprint clipping. Independent C++23 covers83 geometry/width
cases; the actual native SVG passes8-bar and15-explicit-tick checks. Previous-SVG
15713 and horizontal-intersection-removal15715 controls compile then fail their
intended expectations. See [bar execution receipt](BARS_EXECUTION_2026_10_20.md).

The authorized107-snapshot offload passed two-copy restoration and an A/B journal
audit; complete encrypted custody is on GitHub, keys held separately. Including
the earlier one-file step,1262940884 logical bytes left the hot quota. Quota stays
3TiB. Availability returned to96468992bytes, but validation snapshots reduced it
to4194304bytes: further reviewed retention is required for a massive ongoing
batch. See [offload receipt](ARTIFACT_OFFLOAD_EXECUTION_2026_10_20.md).

Statistics, mathematics, genomics and physics remain in scope. Conditioning and
convergence, inferential calibration, count/variant consumers, physical units and
invariants, and measured execution/memory/artifact work continue. This is a
bounded acceptance checkpoint; broader compiler CI and merge remain separate.
