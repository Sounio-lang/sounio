<!-- docs:meta
topic_id: repo.docs.plans.stats-brown-forsythe-execution-2026-10-18
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.plans.stats-brown-forsythe-execution-2026-10-18
-->

# Brown-Forsythe, shared ordering and F/beta tail repair — 2026-10-18

## Final native acceptance

Final clean-source candidate stats-viz-brown-20261018-v4,Slurm **15230**:
COMPLETED **0:0**,elapsed **00:02:42**.
JUnit **53 tests,0 failures,0 skips,0 errors**,retaining all prior48.
Snapshot:
4a51bef3fb808e7d563d1b03bbc515ee674f42ed51145d9e21ecbe8cd37fdce4.

Control stats-viz-brown-control-20261018-v4,Slurm **15231**:
FAILED **1:0**,elapsed **00:02:42**.
JUnit **53 tests:48 pass,5 numerical runtime failures,0 skips,0 errors**.
All prior48 pass; new reference,legacy adapter,status/F=0,tail and visual
fixtures typecheck and compile,then fail under the restored former policies.
Snapshot:
9b720c6511f07b5a2107808d2d6e3b8c2b6c5c07da45d43a301c2d9550ca64dc.

Pinned committed Madaros v0.80 ELF SHA256:
125c25703ac4062fc6befb7acf77e945abb13405f0adb5ab09fbc5a8d4b5b29f.
2 CPUs,8192 MiB,GPU off,five-minute limit,clean-v1,UID/GID1000/1000.
JUnit test-suite counts are authoritative; phase pass counts do not count tests.
This is not a compiler rebuilt from current main.

Initial run15224:52/52,COMPLETED0:0,00:02:36.
Run15226 adds the figure:53/53,COMPLETED0:0,00:02:43.
Run15228:53/53,COMPLETED0:0,00:02:43,includes six more F-tail references and
the positive subnormal. The final review then removed one accidental SSH-banner
comment line from each of two copied source files. Final15230 repeats all53 on
the exact cleaned source bytes; numerical bodies are unchanged. Control15231
uses equally cleaned old helper sources. git diff --check passes.
Use explicit source-start markers when capturing SSH output; do not slice at
the first // because a diagnostic URL can match. Final SVG is byte-identical
to the previously inspected numerical-body candidate.

## Implementation and resource structure

- stats/brown_forsythe.sio: two-group median-centered Levene/Brown-Forsythe,
  F=(N-2)*SS_between/SS_within and direct F(1,N-2) survival.
- Group-centered bounded coordinates preserve representable offset contrasts;
  opposite-sign range overflow uses maximum-absolute scaling. Even median
  midpoint is formed in bounded coordinates, not rounded raw units.
- Both groups then share physical deviation_scale, preserving relative spread.
  Compensated means/within sums and a two-group between contrast; no raw powers,
  no absolute-unit variance floor and no fabricated Beta confidence.
- Explicit size/nonfinite/zero-within/arithmetic statuses; F-reference p is
  marked approximate. Means are absolute deviations normalized by a common
  scale, not sample variance, median absolute deviation or CI endpoints.
- Canonical inputs borrowed; two100-value scratch arrays are reused for sorting,
  coordinates and deviations. The old body held separate sorted/deviation
  arrays for each group. No heap; no measured RSS, stack or wall-clock speedup.
- stats/order.sio centralizes bounded finite-prefix checked heapsort; Shapiro
  and legacy sorting delegate, preserving accepted Shapiro behavior.
- Legacy public by-value levene_test delegates to the checked result. Former
  private Beta return, bucket p and variance floor are removed from this path.
  Other private AD/KS paths remain uncertified.

## Shared F/beta extreme-tail defect

The old ff_exp and bt_exp formed exp(negative) via a positive reciprocal.
For a positive argument above709, that helper returned the sentinel1e300;
the reciprocal therefore produced an artificial1e-300-scale tail floor.
Both now delegate to the existing full-range math::pure::exp.

Regression distinguishes positive subnormal probability near7.239075918e-317
from actual binary64 underflow. Direct beta underflow is also checked.
No full certification of beta inverses, all F degrees of freedom, or unrelated
gamma domains follows. The numeric tail grid uses F numerator df1 and
denominator df2..198, plus the named beta case.

## Independent C++23 oracle

tests/reference/brown_forsythe_reference.cpp compiled on t560:
g++ -std=c++23 -O2 -Wall -Wextra -pedantic.
Raw long-double group medians/deviations and ANOVA do not reuse the candidate
normalization. Probability uses positive adaptive quadrature of a trigonometric
Student/F survival integral, with standard-library log-gamma normalization,
rather than the candidate incomplete-beta algorithm.
Six df2 closed-form cases cross-check quadrature through F=1e30.

15 datasets cover signed1e-200/1e200 units, independent group location shifts,
an even median whose raw midpoint is not representable, opposite-sign finite
extremes, subnormals, equal groups, one constant group,2-vs5/5-vs2,
100-vs100 strong spread contrasts and unequal-sized groups.
36 F-tail points cover denominator df2,4,9,10,98,198 and F up to1e30.
Normal nonzero probabilities are compared relatively; reconstructed subnormal
means and the subnormal p allow two minimum binary64 steps of quantization.
True underflow is required to return zero,not a positive arbitrary floor.
The reference generator and submitted fixtures match; no Python/Rust/Julia/R
numerical execution or independent oracle was used.

## Scientific figure

tests/run-pass/viz_brown_forsythe.sio uses the actual checked deviations of
synthetic groups A=5,B=6. It displays both absolute-deviation profiles and their
means, all in one common normalized unit/domain, with calculated F/p.
F approximately1.719104467740,p approximately0.222272183826.
Caption says approximate F reference; footer says these are mean absolute
deviations,not variance estimates or CIs. Profiles connect sorted observation
ranks, not paired observations, density or confidence envelopes.
Shared IR feeds real Canvas and SVG, with sample/node/series counts, mean-value
agreement, Canvas tick RGB, method/legend labels and dynamic numeric labels.
SVG XML parses; final preview is visually inspected. QuickLook thumbnail
padding is not an exported Canvas framebuffer.

## Control boundary

Final control uses the new statistic/status/storage but restores the exact
former four-bucket p policy and the two former F/beta exponential source files.
This is not an untouched former Levene implementation.
The separate tail fixture uses only shared F/beta routines and discriminates
the reciprocal-floor defect. Reference/legacy/figure and the F=0 status fixture
discriminate bucketed p. All prior48 tests remain a separate preserved set.
Control source differences are retained in manifests and receipt counts.

## Scientific scope and continuity

Two groups only;2..100 complete finite observations per group.
Median-centered Brown-Forsythe with approximate F reference; no independent
sampling/shape suitability check, finite-sample exactness, posterior probability,
automatic pooling decision, empirical power study or SOTA-superiority claim.
Constant absolute deviations give zero within sum and unavailable inference;
no substitute variance is inserted.

Method:
https://www.itl.nist.gov/div898/handbook/eda/section3/eda35a.htm

Isolated /workspace/sounio-sprint-stats-viz.
Original remote-main baseline f5b2c5850be9f3d3995719ffcade44a6d851658a.
Main rechecked e00a5d537cd0c6d01762c38fcd70cc9739062710,unchanged.
Prior intervening stdlib changes remain outside the focused closure.
Canonical active checkout and its local-build compiler are untouched.

Archived manifests match final packages. Selected source/reference/test/packager
files in Mac custody match the final candidate manifest. Contract/master/status/
suite are documentation follow-ups outside the executable snapshot.
No commit,push,PR,remoteCI or merge acceptance.

Next: replace AD/KS calibration heuristics and remaining p-to-Beta path;
scientific layout/nonlinear ticks/other-mark clipping; math/genomics/physics
kernels and real I/O/domain consumers. Execution,memory and artifact objectives
remain active across all tracks.
