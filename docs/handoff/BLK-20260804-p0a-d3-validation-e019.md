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
Status: closed (2026-08-04)
Severity: B1 (was)
Class: compiler-semantics / stdlib-reshape
Owner: cursor--p0a3-e019-validation-20260804
Lane: p0a3-e019-validation-20260804
Worktree: /tmp/sounio-p0a3-e019-20260804
Branch: research/p0a3-e019-validation-20260804
Closed-By: rewrite stats::validation to fixed &[f64; 256] + explicit n (no .len()/.push())
Root-Cause (two layers):
  1) growable [f64].push() → multi-module check E019 (poisoned whole module import)
  2) any imported .len() method (slice or [T;N]) → Madaros lower segfault
Acceptance-Gate: scripts/ci/madaros_validation_import_gate.sh
  + scripts/ci/madaros_ols_fixed_e2e_gate.sh (positive validation control)
Evidence-Level: E3
LLM-Offload: not-required (API reshape; textbook OLS oracles unchanged)
Residual: open-slice &[f64] + .len() still unsupported under Madaros import/lower
  (compiler residual; not claimed closed — use fixed buffers + n)
Next-Action: none for validation science path; optional future check/lower for .len()
```

## Context

Attention P0=A closed fixed-array OLS via `ols_fixed`. This closeout brings
`stats::validation` itself onto the same Madaros-safe contract so textbook
descriptives + OLS import/compile/run under default Madaros. Call sites must
pass `n` and use `[f64; 256]` buffers (breaking change vs open slices).
