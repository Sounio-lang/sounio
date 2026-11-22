<!-- docs:meta
topic_id: repo.docs.plans.stats-shapiro-execution-2026-10-18
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.plans.stats-shapiro-execution-2026-10-18
-->

# Checked Shapiro-Wilk and residual Q-Q figure — 2026-10-18

## Final native receipt

Final candidate stats-viz-shapiro-20261018-v2, Slurm **15211**:
COMPLETED **0:0**, elapsed **00:02:27**.
JUnit **48 tests,0 failures,0 skips,0 errors**, retaining all previous44.
Snapshot:
5fd6355403a77b5d208b3037c5109fbd059fced4f4ce9b61dd1e0aca04e3de40.

Old p-bucket policy control stats-viz-shapiro-control-20261018-v2,
Slurm **15212**: FAILED **1:0**, elapsed **00:02:26**.
JUnit **48 tests:45 pass,3 runtime failures,0 skips,0 errors**.
All previous44 plus new availability/status fixture pass.
The three new p consumers (reference,legacy adapter,Q-Q figure) successfully
compile then fail numerical assertions under the old p buckets.
Snapshot:
262a80765e7ed2104d9bbdccea77adbe0e8eb555f7ce5833ede68ad47306c4c5.

Pinned committed Madaros v0.80 ELF SHA256:
125c25703ac4062fc6befb7acf77e945abb13405f0adb5ab09fbc5a8d4b5b29f.
2 CPUs,8192 MiB,GPU off,five-minute limit,clean-v1,UID/GID1000/1000.
JUnit test-suite counts, not the three harness-phase passes, are authoritative.
Initial numerical/adapter/status run15209:47/47,COMPLETED0:0,00:02:17,
retaining all previous44. Final run adds the Q-Q figure.

## Implementation

stats/shapiro_wilk.sio replaces the former pedagogical coefficient path with
Royston AS R94 normal-order weights and separate n=3,n=4..11,n=12..100
calibrations. W and 1-W are computed using normalized data and compensated
projection residual norms. Direct normal survival uses shared upper gamma;
analytic n=3 uses pure atan. Source is independently organized around this
projection identity, not a verbatim translation of another implementation.

Finite prefix validation, size bounds and constant-sample handling precede
inference. Metadata reports ties, calibration branch and p approximation.
Unavailable arithmetic or calibration states never produce an invented p-value.
Heapsort uses one100-value data scratch array plus50 weights; no covariance
matrix or heap. O(n log n) sorting is a source property, not a measured speedup.
Normalized finite ranges preserve representable offset contrasts; opposite-sign
overflow uses scaled endpoints. A separately retained 1-W survives even when
W rounds to one.

stats/epistemic/assumptions.sio public by-value shapiro_wilk_test now delegates
to this kernel, returning ShapiroResult. Its former private Beta return,
pedagogical weights and p buckets are removed. No external former private
consumer was found. Remaining Levene/AD/KS private heuristics remain uncertified.
Regression assessments preserve existing fields and add shapiro, reusing the
residual buffer. Failed or exact original OLS retains typed unavailability.

## Scientific meaning

3<=n<=100; complete uncensored finite observations.
n=3 has analytic distribution calibration; n>=4 p is a Royston approximation.
Normal-order weights are the published approximation, not exact covariance
matrix coefficients for every sample size. Ties/quantization, censoring and
dependence are not corrected. A large p is not probability of normality.
No Beta posterior, aggregate assumption pass, empirical coverage, finite-sample
uniform accuracy or SOTA superiority is claimed.
For fitted OLS residuals the raw-sample calibration is a descriptive screen;
regression-specific type-I calibration is unassessed and the figure says so.

Primary method: Royston (1995), JRSS C44(4),547-551,
https://doi.org/10.2307/2986146.
The canonical R documentation also describes separate n=3,4..11,>=12 branches:
https://stat.ethz.ch/R-manual/R-patched/library/stats/html/shapiro.test.html.
No R/Python/Julia/Rust numerical execution was used.

## Independent C++23 concordance

tests/reference/shapiro_reference.cpp compiled on t560:
g++ -std=c++23 -O2 -Wall -Wextra -pedantic.
Long-double erfc-inverse bisection; full antisymmetric weight vector;
raw centered observations; standard-library elementary functions.
Direct squared correlation and projection residual W identity cross-check.
Royston coefficients and calibration formulas are shared mathematical input;
this is independent arithmetic and elementary-function evaluation, not an
independent validation or proof of those formulas.

24 datasets include analytic n=3 endpoints and asymmetric sample,4/5/6,
10/11/12 branch boundaries,20/50/99/100, positive/negative1e-200/1e200 units,
1e15 offset,opposite-sign finite extremes,subnormals,duplicates,n=100 outlier
tail p approximately3.37e-22,uniform sample and a near-perfect weight profile.
The near-perfect fixture requires a nonzero separately computed 1-W<1e-24,
rather than relying on subtraction from rounded W.
One independent long-double OLS fit checks the residual Shapiro consumer.
Endpoint p=0 has a finite arithmetic absolute tolerance; nonzero tiny tails
are compared relatively with an absolute floor1e-300.
Generator and submitted fixture are kept byte-identical.

## Figure and inspection

tests/run-pass/viz_shapiro_qq.sio uses the existing synthetic n=20 OLS dataset.
Ordered actual residuals are plotted against standard normal Blom positions.
The auxiliary Q-Q line is a visual least-squares guide, not the SW coefficients
or a confidence envelope. Shared x/y domains, real Canvas rendering with a
tick RGB assertion, SVG labels and dynamic W/p are checked.
Header identifies synthetic data; footer explicitly reports approximate p
and unassessed residual calibration. W approximately0.941178829811,
p approximately0.252379985586.
SVG XML parses and preview is visually inspected. QuickLook padding is a
thumbnail artifact, not an exported Canvas framebuffer.

## Controlled regression evidence

The final control restores only the old four p buckets as a function of the
new W, retaining the new weights, statistic, statuses and n=3 analytic branch.
It is not an untouched former implementation or a full old-algorithm benchmark.
Native reference, legacy-adapter and visual p assertions discriminate that
policy after successful compilation. Prior44 and new availability tests remain
separate from those three consumers. See final JUnit counts above.

## Main, custody and continuation

Isolated worktree /workspace/sounio-sprint-stats-viz.
Original remote-main baseline f5b2c5850be9f3d3995719ffcade44a6d851658a.
Current main rechecked e00a5d537cd0c6d01762c38fcd70cc9739062710.
Prior intervening stdlib changes remain outside this focused closure.
Canonical active checkout/local compiler untouched; committed pinned Madaros
is not a compiler rebuilt from the current main source.

Archived manifests match the final packages. Selected source/reference/tests/
packager in Mac custody match the accepted candidate manifest.
Contract/master/status/suite documentation is a separate follow-up outside the
executable snapshot. No commit,push,PR,remoteCI or merge acceptance.

Next: real median-centered Levene/Brown-Forsythe and remaining AD/KS heuristics;
shared scientific layout/nonlinear ticks/other-mark clipping and domain
I/O/consumers. Mathematics/genomics/physics and execution/memory/artifact
optimization remain active on the master sprint.
