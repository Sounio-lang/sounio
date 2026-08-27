<!-- docs:meta
topic_id: repo.docs.audit.madaros-propagate-monte-carlo-fnptr-2026-08-23
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-propagate-monte-carlo-fnptr-2026-08-23
-->

# Madaros `monte_carlo` fn-ptr promotion (2026-08-23)

## Symptom (2026-08-06 → 2026-08-14)

`epistemic::propagate::monte_carlo(x, f, n)` with a named `fn(f64)->f64` under
default Madaros either returned invalid statistics or SEGV'd at runtime while
lean_single passed (`MONTE_CARLO_FNPTR PASS`).

## Root cause

Two independent compiler bugs:

1. **AST DCE (`spec_dce_scan_expr`)** — first-class function values (`square` in
   `call_it(square, x)`) are `ExprIdent`, not `ExprCall` callees. The
   reachability mark set never included the target function, so `square` was
   deleted before lowering. `LoadFnRef` then bound to a stale/wrong `fn_id`
   (often `0` / null), producing SIGSEGV on `IrCallIndirect`.

2. **`IrCallIndirect` f64 returns** — codegen stored the SysV return in rax but
   did not apply `IR_FLOAT_REG_MARKER_FLAG` / `nc_core_mark_float_reg`, so
   subsequent f64 arithmetic misinterpreted IEEE bit patterns (same class as
   Wave14e builtin float returns).

Streaming IR reachability (`scg_scan_function`, `reach_mark_one_pass`,
`reach_patch_calls`) was also extended to follow `IrLoadFnRef` edges so the
wide-compile driver cannot re-delete fn-ref targets.

## Fix

| Area | File | Change |
|---|---|---|
| DCE reachability | `self-hosted/check/specializer.sio` | Mark `ExprIdent` / single-segment `ExprPath` in `spec_dce_scan_expr` |
| IR reachability | `self-hosted/ir/stream_reach.sio`, `reachability.sio` | Follow + remap `IrLoadFnRef` |
| Lowering | `self-hosted/ir/lower.sio` | `fn_ptr_ret_float` local metadata; float markers on `ir_call_indirect` |
| Codegen | `self-hosted/native/codegen_x86_linux.sio` | Mirror `IrCall` float marking on `IrCallIndirect` |

## Evidence

```bash
export SOUNIO_STDLIB_PATH=$PWD/stdlib
./bin/souc run tests/epistemic_trust/madaros_propagate_monte_carlo_fnptr_probe.sio
bash scripts/ci/madaros_propagate_monte_carlo_fnptr_gate.sh
./bin/souc run examples/native/hof_fn_ref_call.sio
```

Promotion gate: `scripts/ci/madaros_propagate_monte_carlo_fnptr_gate.sh`.

Probe: `tests/epistemic_trust/madaros_propagate_monte_carlo_fnptr_probe.sio`.

Historical fail-closed gate is now a thin alias to the promotion gate.
