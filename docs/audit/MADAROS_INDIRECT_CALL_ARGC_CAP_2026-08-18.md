<!-- docs:meta
topic_id: repo.docs.audit.madaros-indirect-call-argc-cap-2026-08-18
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-indirect-call-argc-cap-2026-08-18
-->

# Madaros — `IrCallIndirect` codegen capped indirect calls at 2 arguments — **RESOLVED**

- **Date filed / fixed:** 2026-08-18
- **Reporter / fixer:** fable-1 (CEI WS-A P2 verification)
- **Severity:** correctness/limitation — silent hard-abort of native codegen
- **Status:** **FIXED** on branch `lane/fable-1/indirect-call-argc-fix` (off `main` `c7bc1ecf23`), commit `7e5fa7efdc`.
- **Reproduced on:** `main` `200b53419b` / `c7bc1ecf23`, built from source.
- **Family:** compiler backend, same dispatch shape as #1799 / #1800.

## Symptom (before fix)

Any **indirect call** (a by-value / non-capturing closure called through a bound
name, or a function pointer) with **3 or more arguments** aborted native-v2
codegen with the generic bridge error, exit status 12:

```
error: native-v2 bridge compilation failed
```

Type-independent: 3× i64 failed identically to 3× f64. Named-fn direct `IrCall`
had no such cap.

## Root cause

`self-hosted/native/codegen_x86_linux.sio`, the `IrCallIndirect` arm:

```sounio
let argc = IR_A_ARG_COUNT[...]
if argc > 2 { return false }          // hard cap -> rc=12
... // only arg0 -> rdi(7), arg1 -> rsi(6) were ever materialised
```

The arm never adopted the SysV marshaling the direct `IrCall` arm already uses
(`nc_core_push_call_stack_args` / `nc_core_load_call_arg_into` into
rdi,rsi,rdx,rcx,r8,r9 / `nc_core_stack_call_cleanup_bytes`), so it could place at
most 2 arguments and bailed on the rest.

A **second, independent** defect masked the win for f64: f64 closure *params*
were never marked float in `lower_closure_expr_ref` (`self-hosted/ir/lower.sio`) —
named-fn params get `ir_mark_float_reg`, closures did not — so f64 args were read
from integer registers and computed garbage even at arity 2.

## Fix (commit `7e5fa7efdc`)

1. `codegen_x86_linux.sio` `IrCallIndirect`: replace the 2-arg cap with the full
   SysV GP marshaling, mirroring the direct `IrCall` arm — args 0–5 into
   rdi,rsi,rdx,rcx,r8,r9, args 6+ pushed on the stack, cleanup after the call.
   The fnref (call target) is loaded into `rax` **last** so arg loads cannot
   clobber it (rax is not a SysV arg register).
2. `lower.sio` `lower_closure_expr_ref` param loop: mark f64 params float
   (`ir_mark_float_reg` + scalar-kind 2), mirroring the named-fn path.

## Verification (from source; self-compile fixed-point held)

Correct values across arity 2–7, both types:

| Case | Result |
|---|---|
| closure 3 f64 | `0.600000` |
| closure 6 f64 (all GP regs) | `2.100000` |
| closure 7 f64 (**stack arg**) | `2.800000` |
| closure 2 f64 | `0.700000` |
| closure 3 i64 | `12` |
| named-fn 3 i64 / 6 f64 / 7 f64 (controls) | unchanged / correct |

**Zero regressions** vs a *same-source* clean-`c7bc1ecf23` baseline (built from
source — the committed `bin/madaros-linux-x86_64` is 100 commits stale and is
NOT a valid baseline): closure slice 12 pass / 14 fail identical; vec slice 10
pass / 11 fail identical. The 14 closure-slice fails are pre-existing
type-check / kind-0-`println` / escape-analysis baselines, unrelated to this arm.

## Impact unblocked

- Indirect dispatch (by-value closures, function pointers) now works at full
  arity, not just 2 args.
- CEI (WS-A): non-capturing handler clauses `let op = |a,b,c,..|` dispatched via
  `ir_call_indirect` can now take ≥3 args, including f64 measurements threaded
  source→handler — the constraint that had bounded
  `examples/effect_uncertainty_gum_vs_mc.sio` to 2 f64 args.

## Note (separate, not fixed here)

`let r = w(3,4,5)` with **no** result annotation still SEGVs on `println(r)` — the
pre-existing kind-0 `println(call-result)` bug (#1799 family). `let r: i64 =`
annotated form works. Out of scope for this dispatch.
