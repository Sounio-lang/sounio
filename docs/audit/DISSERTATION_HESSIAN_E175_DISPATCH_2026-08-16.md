<!-- docs:meta
topic_id: repo.docs.audit.dissertation-hessian-e175-dispatch-2026-08-16
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.dissertation-hessian-e175-dispatch-2026-08-16
-->

# DISSERTATION_HESSIAN_E175_DISPATCH_2026-08-16.md

**Lane**: grok-cli4 / diss-hessian-e175 (WS-F adjacent visibility class)  
**Date**: 2026-08-16  
**Claim**: `bin/sounio-coord claim --agent grok-cli4 --lane diss-hessian-e175 --intent 'root-cause the E175 on an intra-module call in the PBPK14 Hessian test' --files docs/audit/DISSERTATION_HESSIAN_E175_DISPATCH_2026-08-16.md`  
**Evidence source**: Slurm job 9908 (A/B: checked-in `bin/souc` vs. Madaros built-from-source on compute node, HEAD 6f2c4e2461, staged at `/orangefs/training/diss-gates-ab-6f2c4e2461-20260816T124730Z-job9908`). Reproducible on both engines.  
**Your gate**: `scripts/ci/dissertation_pbpk_hessian_gate.sh` (reports 3 failures; CSV mismatch is cascade).

## Root Cause (Compiler Bug, Not Dissertation Code)

The failing test is `tests/run-pass/dissertation_pbpk14_hessian.sio`.

**Exact error** (both engines):
```
error[E175] in run-pass/dissertation_pbpk14_hessian::synthetic_residual_test at 0..2674: function is private in its defining module.
```

- `fn synthetic_residual_test() -> bool with IO, Mut, Div, Panic { … }` declared at **line 62**.
- Called at **line 139** in the **same file** (`let ok = synthetic_residual_test()` in `main()`).
- No `pub` modifier (correct for intra-module helper; the file is a self-contained run-pass test).
- `use darwin_pbpk::epistemic_pbpk14_hessian::*` (line 24) brings in `hessian_pbpk14_auc`, `hessian_emit_csv_header`, `hessian_emit_budget_csv` — none named `synthetic_residual_test`. No homonym or duplicate definition anywhere in the tree (confirmed via `grep -r synthetic_residual_test` — only this file and a training JSONL example).

**This is a compiler defect in the visibility resolver / preflight** (E175 class, adjacent to the E137/E175 multi-module wave closed in July 2026 — see `docs/audit/MADAROS_MULTIMODULE_PRINT_IMPORT_BUGS_2026-07-13.md`, `docs/audit/MADAROS_DUAL_GUM_KNOWLEDGE_IMPORT_2026-07-19.md`, `docs/compiler/KNOWN_LIMITATIONS.md` D4/D6 residual). Same-file non-`pub` function should **never** trigger "private in its defining module". The resolver is incorrectly treating the local binding as cross-module or picking up a stale private symbol from an imported module's namespace collision (known failure mode from prior visibility preflight bugs).

**Gate behaviour**:
- `souc check` and `souc compile` both surface E175 but exit **rc=0** (documented diagnostic-muting in this project — gate parses output to catch it).
- No CSV output → "rapamycin Hessian CSV differs" (11-line golden vs. 0-line runtime) is pure cascade. The program never reached `emit_rapamycin_budget()`.

**Fault attribution**: **Compiler at fault**. Dissertation code is correct (intra-module call to private helper in a run-pass test; `use *` is standard and does not shadow the local fn). This is not numerical drift, not a dissertation regression, and not a PBPK model bug. It is a residual visibility resolution bug exposed by the Hessian test's import pattern.

**Repro command** (Slurm-direct, matches job 9908):
```bash
env SLURM_CONF=/tmp/slurm-direct.conf srun --partition=cpu-ops --chdir=/tmp \
  bash -lc '
    cd /orangefs/training/diss-gates-ab-6f2c4e2461-20260816T124730Z-job9908
    ./bin/souc check tests/run-pass/dissertation_pbpk14_hessian.sio
    # Expect E175 + rc=0 (gate catches via output parse)
  '
```
(Local repro: `./bin/souc check tests/run-pass/dissertation_pbpk14_hessian.sio` reproduces identically.)

**Next action for close** (37 days before defense):
- Fix visibility resolver / preflight for intra-module non-`pub` fns in presence of `use *` (likely in `self-hosted/check/mod.sio` or `visibility_preflight`).
- Re-run `scripts/ci/dissertation_pbpk_hessian_gate.sh` (should go 6/6 green).
- Update June qualification reports + `docs/dissertation/` artefacts with new evidence.
- No changes to dissertation code required.

**Receipt**: The A/B measurement on job 9908 (identical failure on checked-in vs. freshly-built Madaros) + same-file declaration/call + no homonym = definitive compiler bug. Dissertation CI gate is correctly failing on a real diagnostic.

**Status**: Open compiler blocker (E175 residual, visibility class). Dissertation code innocent.

*This document is the canonical root-cause record. Last revised 2026-08-16 by grok-cli4 (this lane). See also `scripts/ci/dissertation_pbpk_hessian_gate.sh`, `KNOWN_LIMITATIONS.md` (D4 visibility), and prior E175 dispatches.*
