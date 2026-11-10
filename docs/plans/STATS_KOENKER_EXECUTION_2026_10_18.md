<!-- docs:meta
topic_id: repo.docs.plans.stats-koenker-execution-2026-10-18
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.plans.stats-koenker-execution-2026-10-18
-->

# Checked Koenker heteroscedasticity and residual figure — 2026-10-18

## Final execution

Final candidate stats-viz-koenker-20261018-v5, Slurm **15196**:
COMPLETED **0:0**, elapsed **00:02:06**.
JUnit **44 tests, 0 failures, 0 skips, 0 errors**, retaining all previous 40.
Snapshot:
5228e6de8a5d52c711477be0f837188175dd97f4d05bf8d1708c61c4b7b047da.

Controlled old tail-policy stats-viz-koenker-control-20261018-v5,
Slurm **15197**: FAILED **1:0**, elapsed **00:02:06**.
JUnit **44 tests: 41 pass, 3 runtime failures, 0 skips, 0 errors**.
All previous 40 and new status/availability fixture pass.
The three p-value consumers (reference, legacy adapter, scientific figure)
successfully compile, then fail numerical assertions with the old buckets.
Snapshot:
8012887783484ce04e7f7ae3cadcf4eb458a2978af3f91c31217c4dd4e18bfb5.

Pinned committed Madaros v0.80 ELF SHA256:
125c25703ac4062fc6befb7acf77e945abb13405f0adb5ab09fbc5a8d4b5b29f.
2 CPUs, 8192 MiB, GPU off, five-minute limit, clean-v1, UID/GID 1000/1000.
JUnit test-suite counts are authoritative; phase-summary pass counts are not
the number of native tests. This is not a compiler rebuilt from main.

## Implementation and scientific contract

- stats/heteroscedasticity.sio: checked studentized Koenker variant with an
  intercept and one variance predictor, LM=n*auxiliary R-squared and df=1.
- Scalar O(n) passes, no squared-residual buffer or auxiliary OLS result arrays.
  Anchor-preserving x normalization and opposite-sign overflow fallback;
  factored scaled squared-magnitude differences and compensated accumulators.
  Correlation-squared avoids cancellation from 1-SSE/SST near zero.
- Direct upper-gamma reference survival, checked against independent erfc.
  Status distinguishes invalid size, nonfinite data, constant predictor,
  constant squared residuals and arithmetic failure. Metadata explicitly
  identifies asymptotic inference, not a posterior or finite-sample guarantee.
- stats/regression/assessments.sio: stable simple OLS once, followed by checked
  Jarque-Bera and Koenker diagnostics on that fit's residuals. Independent
  fit/normality/heteroscedasticity statuses are preserved. Exact OLS can have
  unavailable residual diagnostics; no aggregate assumption-pass flag.
- The former private epistemic breusch_pagan_test body is removed and replaced
  by a public by-value compatibility adapter returning KoenkerResult.
  Required effects, exclusive sort borrows and a pure abs helper allow the
  legacy file to import natively. Its other private pedagogical methods remain
  uncertified and are explicitly labeled. No p-to-Beta inference in this path.

Only 3<=n<=100 and the single-predictor case are supported.
The standalone caller supplies residuals from its intended conditional-mean
model; independence, finite moments, correct specification and asymptotic
calibration are not assessed. A high p is not probability of homoscedasticity.
See stdlib/stats/HETEROSCEDASTICITY.md.

Primary method: Koenker (1981), Journal of Econometrics 17(1),107-112,
https://doi.org/10.1016/0304-4076(81)90062-2.

## Independent C++23 reference

tests/reference/koenker_reference.cpp, compiled on t560:
g++ -std=c++23 -O2 -Wall -Wextra -pedantic.
Long-double raw squared-response auxiliary least squares with explicit fitted
SSE/SST is independent of candidate scaled correlation-squared.
std::erfc(sqrt(LM/2)) supplies the tail oracle.

Twelve datasets: positive/negative tiny/huge joint unit changes, 1e15 x offset,
opposite-sign extreme x, subnormal residuals, adjacent magnitudes differing by
binary64 ulps, zero correlation, n=3 and n=100 perfect auxiliary correlation.
Twelve tail points include both sides of the series/continued-fraction boundary,
LM=0, LM=100 and p approximately 1.5239706048321e-23.
One complete n=20 model case independently checks slope, residual JB and Koenker.
Native edge cases cover invalid sizes, NaN/infinity, constant inputs,
constant squared residuals, perfect original OLS and rank-deficient original OLS.
Legacy adapter is exercised with large residual/tiny predictor units.

Tolerances are explicit in generated Sounio assertions. The gamma a=1/2 domain
used here is checked, not its entire parameter space. No coverage study or
uniform error theorem is claimed.

## Scientific figure

tests/run-pass/viz_koenker.sio uses actual residuals from the n=20 synthetic OLS
fixture, plots squared normalized observations and their auxiliary regression,
and labels computed LM=17.84623301848... and p=2.39492898118e-5.
Title states synthetic data; method states chi-square(1) asymptotic/no posterior.
Connected observations are ordered by predictor; this is not a density or CI.
Same IR feeds Canvas and SVG, with exact shared domains, counts, Canvas tick RGB
assertion, dynamic numeric labels and method/axis/legend checks.
Final y domain uses binary-exact half steps and dyadic interpolation fractions
to avoid a near-zero floating tick label. SVG XML parses; QuickLook preview is visually inspected.
QuickLook square padding is a thumbnail artifact, not a Canvas framebuffer export.

## Controlled failure and preliminary runs

Control uses the same complete new implementation except for its final p-value
assignment, which is replaced by the exact four-bucket policy from the former
legacy BP function (.90/.60/.30/.05). This isolates the tail-policy defect.
It is **not** an untouched former implementation or full before/after memory
benchmark. Expected new reference, legacy-adapter and visual p assertions fail
after native compilation; original 40 and new availability fixture pass.

Initial job15186: 43 tests,42 pass, visual fixture runtime failure due to sampling
a pixel outside the leftward y-axis tick. Job15190 localizes that assertion.
Jobs15188/15189 additionally expose legacy missing effects and exclusive sort
borrow errors; those are repaired, not accepted as numerical-failure evidence.
Job15191: 44/44,0failures/skips/errors; old-tail control15192:41pass+3runtime
failures. Jobs15193/15195 repeat these counts. Visual inspection shows quarter-step
spacing alone still leaves a near-zero tick under weighted interpolation.
Final pair uses dyadic interpolation fractions, verifies the exact zero in a
C++23 static_assert and a native tick assertion, then repeats the complete batch.

## Source, custody and scope

Isolated worktree /workspace/sounio-sprint-stats-viz.
Original main baseline f5b2c5850be9f3d3995719ffcade44a6d851658a.
Remote main rechecked e00a5d537cd0c6d01762c38fcd70cc9739062710, unchanged from
previous checkpoint. Canonical active /workspace/sounio and its local compiler
are untouched. Prior remote-main stdlib changes remain outside this closure.

Archived manifests match final packages; selected code/reference/test/packaging
files in Mac custody match the final candidate manifest. Documentation follows
execution separately and is not part of the executable snapshot.
No commit, push, PR, remote CI or merge acceptance.

Scalar storage is established by source, not measured stack/RSS or speedup.
No new parallel library stack, no full compiler rebuild, no whole-stdlib
completion or SOTA-superiority claim follows. Statistics/scientific rendering,
math, genomics, physics and execution/memory/artifact objectives remain active.
Next method work: genuine Shapiro-Wilk coefficients/calibration and replacement
of remaining assumption heuristics; nonlinear ticks/layout, other-mark clipping,
domain I/O/consumers and measured scalability remain on the master sprint.
