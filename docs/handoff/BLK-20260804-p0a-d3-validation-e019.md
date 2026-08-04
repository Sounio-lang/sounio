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
Status: classified
Severity: B1
Class: compiler-semantics
Owner: unassigned (needs check.sio / specializer lane; do not steal from issue901 claim)
Lane: p0a-d3-residual-20260804 (residual after fixed-array OLS closeout)
Worktree: /tmp/sounio-p0a-d3-20260804
Branch: research/p0a-d3-residual-20260804
Files-Owned: (open — likely self-hosted/check/** when claimed)
Do-Not-Touch: self-hosted/check/check.sio while issue901 claim is ACTIVE
Repro: ./bin/souc compile /tmp/neg.sio
  with `use stats::validation::{linear_regression}` + fixed [f64;5] args
Observed: error[E019] method calls are not supported for this type (during multi-module check)
Expected: slice/array `.len()` on `&[f64]` resolves under Madaros multi-module check
Acceptance-Gate: validation import compiles+runs textbook OLS under default Madaros;
  negative control in scripts/ci/madaros_ols_fixed_e2e_gate.sh must flip from E019 to green
  only when this blocker closes (update gate accordingly)
Evidence-Level: E3
Evidence: scripts/ci/madaros_ols_fixed_e2e_gate.sh negative control; /tmp/p0a_val.err
Fallback-Path: use stats::ols_fixed::{linear_regression_n,r_squared_n} + cooks_distance + shapiro_wilk
Legacy-Kept: yes (stats::validation unchanged)
LLM-Offload: not-required
Next-Action: When issue901 / check.sio write window is free, claim check surfaces and
  teach Madaros multi-module check to resolve `.len()` on slice/array types (or rewrite
  validation to fixed arrays). Do not widen ols_fixed claims beyond fixed buffers.
```

## Context

Attention P0=A closed the **science-usable** OLS path under Madaros via fixed arrays.
The historical "OLS multi-mod still red with E019" note is narrowed: only
`stats::validation` slice methods remain red; cooks/shapiro/ols_fixed are green.
