<!-- docs:meta
topic_id: repo.docs.audit.stats-ols-diag-e2e-vertical-2026-07-18
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.stats-ols-diag-e2e-vertical-2026-07-18
-->

# OLS diagnostics E2E — 2026-07-18

## Scope

| | |
|---|---|
| Driver | `tests/stdlib/stats/test_ols_diag_e2e.sio` |
| Gate | `bash scripts/stats_ols_diag_e2e_gate.sh` → `STATS_OLS_DIAG_E2E_GATE_OK` |
| Engine | `lean_single` |
| Surfaces | `stats::cooks_distance`, `stats::validation`, `stats::shapiro_wilk` |

## Proven quantities

| Design | Metrics |
|---|---|
| x=1..5, y={1,2,3,4,10} | slope=2, intercept=−2, Cook’s D₅=2.25, D₁=0.5625, max_d=2.25, n_influential≥1 |
| x=1..5, y={2,4,5,4,5} | slope=0.6, intercept=2.2, r²=0.6 |
| residuals of influence fit | Shapiro W on e={1,0,−1,−2,2} (assumption signal) |

## Claims not made

- Full statsmodels diagnostic suite
- Madaros multi-module path
