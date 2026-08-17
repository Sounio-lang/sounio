<!-- docs:meta
topic_id: repo.docs.handoff.blk-20260804-p0a-d3-validation-e019
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.handoff.blk-20260804-p0a-d3-validation-e019
-->

# Blocker: BLK-20260804-p0a-d3-validation-e019

```text
Blocker-ID: BLK-20260804-p0a-d3-validation-e019
Status: closed
Severity: B1
Class: compiler-semantics
Owner: cursor--madaros-d3-d6-effects-20260806
Lane: madaros-d3-d6-effects-20260806
Closed-utc: 2026-08-06
Closeout: stats::validation migrated to fixed [f64;256]+n (no open-slice .len());
  Madaros import green via scripts/ci/madaros_validation_import_gate.sh and
  positive control in scripts/ci/madaros_ols_fixed_e2e_gate.sh.
  Checker also accepts .len() on TyArray/TySlice via len_method_supported when
  present; science path no longer depends on open-slice methods.
Acceptance-Gate: MADAROS_VALIDATION_IMPORT_GATE_OK + MADAROS_OLS_FIXED_E2E_GATE_OK
Evidence-Level: E3
Legacy-Kept: yes (ols_fixed remains the thinner OLS surface)
LLM-Offload: not-required
Residual: open-slice &[f64] + imported .len() lowering is NOT claimed closed
  (KNOWN_LIMITATIONS D3 remaining surface).
```

## Context

Attention P0=A closed the science-usable OLS path under Madaros via fixed arrays.
Shepherd-merge `1e74b97610` migrated `stats::validation` to the same fixed-buffer
contract. The historical E019 negative control is retired; gates pin
`SOUNIO_STDLIB_PATH=$ROOT/stdlib` so a foreign stdlib cannot revive the false red.
