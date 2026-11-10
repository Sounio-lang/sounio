<!-- docs:meta
topic_id: repo.docs.plans.statistics-inferential-tails-execution-2026-10-16
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.plans.statistics-inferential-tails-execution-2026-10-16
-->

# Inferential tail execution receipt — 2026-10-16

## Source and scope

Worktree: /workspace/sounio-sprint-stats-viz.
Baseline remote main: f5b2c5850be9f3d3995719ffcade44a6d851658a.
This extends the MAD / interval-forest checkpoint documented in
STATISTICS_VISUALIZATION_EXECUTION_2026_10_16.md.

Implementation:
- student_t::t_sf evaluates the symmetric tail directly, preserving small
  probabilities instead of subtracting a rounded CDF from one.
- inferential t-test and Spearman p-value callers use student_t::t_two_tail;
  the legacy helper ignored df and used an incorrectly scaled erf expression.
- The private ANOVA caller now delegates to fisher_f::f_sf instead of a rough
  algebraic approximation. Its wrapper and callers declare their effects.
- SpearmanResult and its fields are public, matching the existing public
  spearman_test entry point.
- The import-blocking Mann-Whitney internal check now compares result.u_stat,
  rather than comparing a struct with a floating point number.

## Independent numerical reference

tests/reference/inferential_tail_reference.cpp is C++23, built with:
g++ -std=c++23 -O2 -Wall -Wextra -Werror.

Student tails use long-double adaptive Simpson integration of a transformed
Student density, not an incomplete-beta implementation. Normalized integration
on [0,1] and log coefficients preserve tiny tails. Cauchy and df=2 analytic
identities are checked at four abscissae each; convergence exhaustion throws.
F references use the analytic d1=2 identity
SF(x) = exp(-(d2/2)*log1p(2*x/d2)).

The executable writes both a TSV reference and the Sounio fixture, using
18-digit constants and explicit floating literals. Regenerate with:
./inferential-tail-reference tests/run-pass/stats_inferential_tails.sio

Verified fixture:
- 46 positive Student survival / two-tail pairs:
  df in {1,2,5.5,10,30,100,500}, t in {0,1,2,5,10,100,100000};
  combinations below 1e-290 are excluded, yielding 46 cases.
- 21 analytic F survival cases: d1=2, d2 in {1,2,5,10,30,100,500},
  x in {1,5,100}.
- Each comparison requires a positive probability <=1 and relative error
  strictly below 1e-7. This is a fixture tolerance, not a measured maximum error.
- Negative-t symmetry and preservation of the legacy df<=0 survival sentinel.
- Public Spearman: x={1,2,3,4,5}, y={1,3,2,5,4}, rho=0.8,
  df=3, t=2.30940107675850306, two-sided p=0.104088038661827859.

Spearman uses a t approximation to rank-test significance. This checkpoint does
not implement exact permutation probabilities.

## Failing-before evidence

- stats-viz-tails-20261016-before / job 14672: fixture preparation/import
  diagnostics, including integer literal emission, private result fields, and
  the pre-existing Mann-Whitney struct/scalar comparison. Not a numerical failure.
- stats-viz-tails-20261016-before2 / job 14674: existing Student survival failed
  at case 20 (df=5.5, t=100); the fixture isolated Student tails and emitted
  corrected floating constants.
- stats-viz-tails-20261016-after1 / job 14675: effect declarations identified
  and subsequently propagated through private callers and internal checks.
- stats-viz-tails-20261016-after2 / job 14676: initial 46-case Student /
  Spearman fixture passed with the preceding three sprint regressions.
- stats-viz-tails-20261016-spearman-before / job 14678: restored baseline
  inferential source with ONLY public-result / import-check prerequisites fixed.
  Corrected Student survival and all 21 F references remained present.
  The fixture failed after its tail rows, before its final success marker.
  No baseline source was restored into the development worktree.

## Final execution

Foundry run: stats-viz-tails-20261016-final.
Slurm job: 14679, COMPLETED, ExitCode 0:0, elapsed 15 seconds.
Resources: CPU 2, memory 8192 MiB, no GPU.
Snapshot SHA256:
978996685e0bcd8e4cbbf5c63e5172366ef6a827d313016c97dadd8725766ab3

Madaros v0.80.0 prebuilt, materialized from committed gzip; no compiler rebuild.
Compiler SHA256:
125c25703ac4062fc6befb7acf77e945abb13405f0adb5ab09fbc5a8d4b5b29f

JUnit: tests=4, failures=0, skipped=0, errors=0:
- stats_descriptive_mad
- stats_inferential_tails
- viz_chart_builders
- viz_statistical_interval_forest

The focused runtime package is not a complete Git checkout. Its manifest
records baseline, source hashes, and content-identical basic_math-prefixed
aliases used to satisfy the fixed corpus-smoke selector. Run packet Git fields
are not source-identity evidence. The Foundry phase summary pass count describes
three phases; the test-suite JUnit describes the four tests.

Artifacts on t560:
/var/lib/cephfs-slurm/training/sounio-foundry/runs/stats-viz-tails-20261016-final

## Remaining limits

This is a sampled-domain result. It does not certify arbitrary degrees of
freedom, NaN/infinity behavior, subnormal tails, enormous t whose square
overflows, or the df>10000 normal approximation. Existing exponential helpers
and incomplete-beta extremes require further work.

The ANOVA wiring is typechecked and its canonical F kernel is exercised for
d1=2, matching three-group ANOVA. Private ANOVA/ttest result construction is
not separately executed by this fixture. Legacy t-test CIs still use a fixed
critical value 2.0 and remain unfinished.

Axes, ticks, legends, clipping, layout, scientific content provenance,
Shapiro-Wilk, regression diagnostics, genomics and physics remain sprint work.
No full corpus, remote CI, commit, push, PR, merge, or release is claimed.
