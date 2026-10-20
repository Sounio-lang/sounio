<!-- docs:meta
topic_id: repo.docs.plans.stats-regression-influence-execution-2026-10-17
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.plans.stats-regression-influence-execution-2026-10-17
-->

# Stable simple OLS, real influence and scientific consumer — 2026-10-17

## Accepted native scope

Final run: `stdlib-ols-20261017-final2`, Slurm **14870**.
COMPLETED, exit **0:0**, elapsed **00:01:01**.
JUnit: **18 tests, 0 failures, 0 skips, 0 errors**.
Snapshot SHA-256:
`e23fe684bc9aa181c0575c3c67e34b62d60bd9e27d835ebb5848530b362648c2`.
Committed Madaros v0.80 decompressed ELF SHA-256 (checked by the launcher):
`125c25703ac4062fc6befb7acf77e945abb13405f0adb5ab09fbc5a8d4b5b29f`.
This uses the pinned compiler, not a compiler rebuilt from current source.

Allocation: 2 CPUs, 8192 MiB, GPU off, time limit five minutes.
Runtime UID/GID 1000/1000, clean-v1 environment.
Canonical evidence:
`/var/lib/cephfs-slurm/training/sounio-foundry/runs/stdlib-ols-20261017-final2`.
Job-side mirror: same suffix beneath `/orangefs/training/sounio-foundry/runs`.

The isolated worktree remains `/workspace/sounio-sprint-stats-viz`, baseline
`f5b2c5850be9f3d3995719ffcade44a6d851658a`, originally taken from remote main.
Remote main rechecked as `67e14779dd2b708767a064d15fc696dc5eebd734`.
The stdlib differences between those committed heads are optimize/nelder_mead.sio
and polynomial/lib.sio, outside this focused dependency closure.
No commit, push, PR, merge, repository-wide test or complete-stdlib certification.

## Implementation

- New borrowed core `stats/regression/ols.sio`: centered/scaled OLS,
  compensated centered sums, scaled residual norm, no absolute determinant cutoff.
- Real case-deletion refits supply deleted sigma, externally studentized residuals
  and DFFITS. Cook distance has independent validity metadata. Invalid/deleted
  singular fits, unit leverage and zero deleted variance are explicit statuses.
- Public `stats/regression/linear.sio` delegates to that core. Its 95% slope
  interval uses Student t with n-2 degrees of freedom. Significance requires
  valid inference. R-squared availability is explicit.
- Removed the heuristic normality/variance/serial-independence confidence claims,
  zero DFFITS placeholder and fixed 1.96 slope interval.
  Beta(1,1) compatibility fields mean unassessed, not calibrated confidence.
- Added coefficient variance status for overflow/underflow while retaining the
  representable standard error. Epistemic adaptation reuses the full influence
  fit rather than fitting the whole sample again.
- Public contract: `stdlib/stats/regression/REGRESSION.md`.
  Added record fields change layouts; downstream literals and ABI need review.

## Independent numerical evidence

`tests/reference/ols_influence_reference.cpp` builds with:

```sh
g++ -std=c++23 -O2 -Wall -Wextra -pedantic ols_influence_reference.cpp -o ols-reference
./ols-reference stats_ols_influence.sio
```

C++23 uses long-double two-column QR with reorthogonalization, 11 datasets and
88 actual case-deletion QR fits. Prediction differences independently supply
DFFITS and Cook distance, rather than sharing the Sounio closed formula.
Generated checks cover coefficients, standard errors, residual scale/residuals,
hat diagonal, DFFITS and Cook distance. Input transformations include scales
1e-200 and 1e200, coupled x/y scales 1e-150 and 1e150, translated coordinates,
high leverage and response outliers. Normalized tolerance is 2e-8 against each
declared characteristic scale; this does not certify every finite f64 input.

The public adapter fixture separately exercises df=6 Student interval width,
unassessed provenance, nonzero DFFITS, invalid sample sizes, rank deficiency,
constant response, exact fit, deleted exact fit, unit leverage, NaN/infinity
and coefficient variance overflow/underflow.

## Scientific figure

`viz_regression_influence.sio` passes actual DFFITS from the regression into
three shared-domain line series, shared axes, clipping and style-bound legends.
The symmetric screening lines are labelled `screening +/-2 sqrt(2/n)`;
they are not confidence bounds, p-values or automatic deletion decisions.
Canvas executes and the exact native SVG is exported. The final figure asserts
both axis creation and scene/series counts and includes observation labels.
The first figure was numerically correct but its horizontal axis was rejected
for insufficient label spacing and its largest point was clipped. Visual
inspection exposed those gaps; final2 fixes both and exports the full outlier.

Local custody:
`/Users/demetriosagourakis/Documents/Sounio-stdlib-sprint-2026-10-16/regression-influence-2026-10-17/`.
SVG is 800x320. Quick Look PNG is a thumbnail with white padding, not a renderer
export; the SVG is the canonical scientific artifact.

## Execution history and counterfactual

- 14864: previous 15 passed; new fixture skipped because its run-pass directive
  was missing. This was not regression acceptance.
- 14866: 15 passed, 3 failed; identified missing Panic effects and legacy
  unsupported &self receiver syntax.
- 14867: first 18/18 passing regression integration.
- 14868: 18/18 after QR standard-error checks, variance statuses and fit reuse.
- 14870: final 18/18, including corrected horizontal axis and full point range.
- Controlled legacy-policy restoration: see below when receipt is appended.
  This retains all new APIs and compilation and changes only the core fit:
  a raw normal-equation determinant below 1e-10 returns zero coefficients,
  reproducing the former scale-dependent singularity policy. It is an isolated
  counterfactual, not an unmodified baseline or an ABI comparison.

## Resource limits and continuation

Core fit is O(n), influence is O(n^2) by real deletion, n capped at 100.
Kernels allocate no heap; fixed array capacity and explicit statuses remain.
The legacy adapter retains value-array arguments; new callers can borrow arrays
through the core. This batch does not claim a measured RSS, runtime or binary-size
improvement. Compact SVG paths and renderer profiling remain open.
The packaging helper now includes all 18 fixtures; the earlier helper omitted
the three accepted axis/clip/legend tests even though their manually populated
native packages did execute them.

Next: assessed regression/resampling procedures and robust data contracts;
automatic ticks, log/symlog policies, shared domain binding and compact SVG;
continue genomics count-matrix and physics covariance tracks in the master plan.

### Controlled numerical regression result

Run `stdlib-ols-20261017-before2`, Slurm **14871**, FAILED **1:0**, elapsed
**00:01:01**. JUnit: **17 passes, 1 failure, zero skips/errors**.
Snapshot `03a722ce9271b712d672cab50420cdaf6bb858552cc71306f2e64eab29790443`.
All package file hashes match final2 except `stdlib/stats/regression/ols.sio`.
The QR fixture compiles and executes case 0, then fails the numerical assertions
in case 1 (x scale 1e-200). Thus the acceptance is sensitive to the old absolute
determinant singularity behavior, with APIs, compiler and other sources held fixed.
This control does not validate every removed placeholder or every possible input.
Compressed compiler artifact SHA-256:
`feaaf7fe181f1ee3efd30e7fe23b1709937ea7e93b52e812276fb093b6ccf52a`.
