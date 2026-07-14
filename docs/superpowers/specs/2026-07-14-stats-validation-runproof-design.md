<!-- docs:meta
topic_id: repo.docs.superpowers.specs.2026-07-14-stats-validation-runproof-design
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.superpowers.specs.2026-07-14-stats-validation-runproof-design
-->

# Design — Independent run-proof for stats::validation (coordinated, disjoint lane)

**Status:** approved design, pre-implementation
**Date:** 2026-07-14
**Constraint:** No compiler changes. **Coordination: the `stats` module is under active development by
another lane (PR #905 `feat/stats-suite-11`, the `stats-suite-*` series). This lane is DISJOINT — it
reads `stdlib/stats/validation.sio` and edits NO stats source and NO existing test/harness; write-set is
only new, uniquely-named files.** EN-UK.

## 1. Why
Sixth application of the playbook (GUM #860, units #873, linalg #892, neg-fix #900, prob #902), adapted
for coordination: instead of hardening a module the other lane owns, add **independent published-value
verification** for a stable, unclaimed stats module — complementing the active lane's breadth with
depth, without an ownership conflict.

## 2. Verified starting state
- `stdlib/stats/validation.sio` (self-contained, green, ~360 lines) exposes a `&[f64]`-slice descriptive +
  regression API: `mean/variance/std_dev/variance_population/standard_error/min/max/range/correlation/
  r_squared/linear_regression/residuals/mse/rmse/mae/t_statistic/confidence_interval_95/validate`.
- **Disjoint**: last touched by an unrelated commit (`e8b7e2d65`, GPU); NOT in PR #905's files; no test
  file references `stats::validation`.
- **Runs correctly** under `lean_single` (verified): mean([2,4,6,8,10])=6, std=√10=3.162278; the `&x`
  slice idiom sidesteps the cross-module `[f64;N]`-by-value corruption that blocks the array-by-value
  regression API.
- Build caveat: like other multi-module programs, native compile is via `SOUNIO_SOUC_ENGINE=lean_single`
  (+ `chmod +x`); default Madaros hits visibility-preflight. (See the 2026-07-14 audit dispatches.)

## 3. Goal
An independent run-proof asserts `stats::validation`'s descriptive + regression outputs against
published/hand-computed values, gated — proving correctness without touching the active lane's source.

## 4. Scope
### In
1. **Run-proof driver** (`tests/stdlib/stats/test_validation_runproof.sio`) — asserts mean, variance,
   std_dev, standard_error, max, range, correlation, r_squared, and OLS slope/intercept.
2. **Gate** (`scripts/stats_validation_gate.sh`, lean_single).
### Out
- **No edit to any `stdlib/stats/` source or existing test/harness** (coordination — the other lane owns
  those). No display helper. No new stats module.
- No compiler edits.
- Math-review of the statistical identities is run and logged.

## 5. Design — assertions (published / hand-computed)
- data = [2,4,6,8,10]: mean=6; sample variance=10; std=√10≈3.162278; SE=std/√5≈1.414214; max=10; range=8.
- x=[1,2,3,4,5], y=[2,4,5,4,5]: Pearson r = Sxy/√(Sxx·Syy) = 6/√60 ≈ 0.774597; r²=0.6;
  OLS slope = Sxy/Sxx = 0.6; intercept = ȳ − slope·x̄ = 2.2.
All inline in `main` (multi-module importing-program constraint); assert on returned f64 fields.

## 6. Module layout
```
tests/stdlib/stats/test_validation_runproof.sio   (new — uniquely named)
scripts/stats_validation_gate.sh                   (new)
```

## 7. Verification
- `validation.sio` compiles under lean_single (it uses `.len()` etc. the default Madaros engine rejects).
- `SOUNIO_SOUC_ENGINE=lean_single souc compile … && chmod +x && ./elf` → `STATS_VALIDATION_OK`.
- `scripts/stats_validation_gate.sh` → `STATS_VALIDATION_GATE_OK`.
- Math-review logged.

## 8. Success criteria
1. Run-proof asserts published values for descriptive + regression stats and passes under lean_single.
2. Gate green.
3. No stats source or active-lane file modified (disjoint); no compiler files touched.

## 9. Risks
| Risk | Mitigation |
|---|---|
| Active stats lane drift causes a merge conflict | Unique filenames; no source edits; rebase-on-conflict (as done for #892/#902). |
| Reader assumes default Madaros builds it | Gate + header comment state the lean_single requirement. |
