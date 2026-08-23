<!-- docs:meta
topic_id: repo.docs.audit.madaros-propagate-monte-carlo-fnptr-2026-08-06
authority: repo_only
audience: users
last_validated: 2026-08-23
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-propagate-monte-carlo-fnptr-2026-08-06
-->

# Madaros — `propagate::monte_carlo` fn-ptr (2026-08-06 → promoted 2026-08-23)

**Status:** GREEN under default Madaros (imported multi-module native path)  
**Gate:** `scripts/ci/madaros_propagate_monte_carlo_fnptr_gate.sh`  
**Probe:** `tests/known_failures/madaros_propagate_monte_carlo_fnptr_probe.sio`

## Historical finding (2026-08-06)

Under stock Madaros, `monte_carlo(x, square, N)` with a named `fn(f64)->f64`:

- `souc check` OK; native compile+run exited without SEGV;
- mean/variance were invalid (sentinel `MONTE_CARLO_FNPTR FAIL`);
- `lean_single` oracle printed `MONTE_CARLO_FNPTR PASS` (≈4.01 / 0.16).

## Root causes (2026-08-23 closeout)

1. **AST DCE (`spec_dce_scan_expr`)** dropped functions referenced only as bare
   idents (`let p = square`, `monte_carlo(x, square, n)`), so `IrLoadFnRef` targeted
   address 0 → SEGV or wrong indirect call.
2. **`IrCallIndirect` codegen** did not mark f64 return registers (integer compare
   path on IEEE bits).
3. **`propagate::monte_carlo`** used untyped `let y = f(sample)`; annotated
   `let y: f64 = f(sample)` plus `ir_mark_float_reg` on annotated `let` bindings
   restores float semantics.

## Fixes

| Layer | File | Change |
|---|---|---|
| DCE | `self-hosted/check/specializer.sio` | Mark `ExprIdent` in `spec_dce_scan_expr` |
| DCE (bootstrap path) | `self-hosted/ir/stream_reach.sio`, `self-hosted/ir/reachability.sio` | Follow `IrLoadFnRef` edges |
| Lower | `self-hosted/ir/lower.sio` | `ir_mark_float_reg(bind_reg)` on annotated `let y: f64` |
| Codegen | `self-hosted/native/codegen_x86_linux.sio` | Float/int mark after `IrCallIndirect` |
| Stdlib | `stdlib/epistemic/propagate.sio` | `let y: f64 = f(sample)` |

## Evidence

```bash
export SOUNIO_STDLIB_PATH="$PWD/stdlib"
./bin/souc run tests/known_failures/madaros_propagate_monte_carlo_fnptr_probe.sio
# → MONTE_CARLO_FNPTR PASS

bash scripts/ci/madaros_propagate_monte_carlo_fnptr_gate.sh
# → MADAROS_PROPAGATE_MONTE_CARLO_FNPTR_GATE_OK
```

Value-style LCG kernels (`monte_carlo_identity`, `monte_carlo_square`) remain
green via `scripts/madaros_propagate_native_gate.sh`.

## Residual

Untyped indirect fn-ptr results (`let r = p(2.0)` without `: f64`) may still
mis-compare until a dedicated `IrCallIndirect` float marker is wired from fn-ptr
types; `monte_carlo` is covered by the stdlib annotation.
