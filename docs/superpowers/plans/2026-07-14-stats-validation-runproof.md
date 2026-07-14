<!-- docs:meta
topic_id: repo.docs.superpowers.plans.2026-07-14-stats-validation-runproof
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.superpowers.plans.2026-07-14-stats-validation-runproof
-->

# Stats::validation Run-Proof — Implementation Plan (coordinated, disjoint)

> Compile-and-run is the gate. Build with `SOUNIO_SOUC_ENGINE=lean_single` + `chmod +x`. **No edit to any stats source or existing test/harness** (the stats module is under active development by another lane — PR #905 / stats-suite-*).

Spec: `docs/superpowers/specs/2026-07-14-stats-validation-runproof-design.md`.

## Task 1 — Run-proof driver
- [ ] `tests/stdlib/stats/test_validation_runproof.sio` (inline in main, `use stats::validation::*`). Assert published values: mean=6, variance=10, std=√10, SE≈1.414214, max=10, range=8 (data [2,4,6,8,10]); correlation≈0.774597, r²=0.6, OLS slope=0.6, intercept=2.2 (x=[1..5], y=[2,4,5,4,5]). Then `STATS_VALIDATION_OK`.
- [ ] `SOUNIO_SOUC_ENGINE=lean_single souc compile … && chmod +x && ./elf` → `STATS_VALIDATION_OK`. No tolerance-retrofit. Commit.

## Task 2 — Gate
- [ ] `scripts/stats_validation_gate.sh` — check validation.sio; lean_single compile+chmod+run driver (grep `STATS_VALIDATION_OK`); end `STATS_VALIDATION_GATE_OK`. Run it. Commit.

## Task 3 — Math-review + PR
- [ ] `bin/llm-offload -t math-review -p xai` on the descriptive/regression identities; log it.
- [ ] `node scripts/docs/sync_governance_metadata.mjs`; commit governance + docs.
- [ ] Push; PR to `main`; ensure `Contracts`/`CI Decision` green; merge (rebase-on-conflict if the stats lane drifted).
