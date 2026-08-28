# M1.2 step 4 — blocker: array-init RHS handler hangs self-compiled stage1

Date: 2026-04-30
Worktree branch: claude/s-ssm-zero-divisor-gating-KbKQe (this worktree's HEAD)
Author: Claude Opus 4.7 (1M context)

## Symptom

Adding handling for `var name: [T; N] = [v0, v1, ...]` / `[val; N]`
initializer RHS in `parse_stmt_ir` (TK_VAR/TK_LET array-decl branch,
~line 2424 of `self-hosted/compiler/native_compile_driver.sio`) makes
the self-compiled stage1 driver spin (100% CPU, hits 30s timeout) on
inputs like `tests/run-pass/symbolic_test.sio` — even though:

- `souc check` of the driver passes.
- The JIT-driven `souc run native_compile_driver.sio -- target.sio` works
  correctly (e.g. `[10,20,30,40]` initializer produces an ELF that prints
  `100`).
- `scripts/ci/native_v2_driver_self_compile_gate.sh` PASSES (stage1 ==
  stage2 == stage3 fixed-point) — i.e. the codegen is deterministic, but
  the produced binary spins.

Removing **only** the array-init handler (keeping cluster #6 + cluster
#5 if-expr fixes) restores fast/clean behaviour on the same inputs.

## Implication for the gate

`native_v2_driver_self_compile_gate.sh` proves *fixed-point* (stage1
binary is byte-identical to its self-compile output), not that the
binary actually compiles arbitrary inputs. A deterministic codegen bug
in stage1 reproduces identically in stages 2/3 and the gate stays green.

A useful follow-up would be a "stage1 smoke" step in the gate that runs
a small cohort of `tests/run-pass/` cases through the freshly-built
stage1 driver and asserts each completes within (say) 5 seconds.

## What landed instead

Two of the four sub-clusters in M1.2 step 4 closed cleanly:

- **Cluster #6 (operators in nested expr position)**: `parse_stmt_ir`
  speculative call with rollback so a tail-position call followed by a
  binop / cmp / `&&` / `||` / `as` / `.` / `[` falls through to
  `parse_expr_ir`. This is what fixes the original `fib(n-1) +
  fib(n-2)` regression. Validated: `kind=132` 5 → 1 in run-pass+broader,
  `kind=138` 17 → 0.

- **Cluster #5 (`if` as expression)**: new `parse_if_expr_ir` invoked
  from `parse_atom_ir` on `TK_IF`, supporting the `let x = if cond {
  THEN } else { ELSE }` idiom for single tail-expression arms. Validated:
  `kind=20 text=if` 17 → 2 in run-pass+broader.

Cluster #4 (array-init) is **deferred** until the stage1-hang root
cause is understood — possibly a label-count, register-count, or
nested-loop limit in lean_single's codegen of the new branch.

## Next-step suggestions for whoever picks this up

1. Bisect the array-init branch by reducing it to its simplest possible
   form (just skip past `= [...]`, no `parse_expr_ir` recursion, no
   per-element copies). If a no-op skip still hangs, the trigger isn't
   the new `parse_expr_ir` calls.
2. Add stage1 smoke to the self-compile gate so this trap is caught at
   the gate level.
3. Inspect `V2_NEXT_LABEL` / `V2_NEXT_REG` per-fn limits — the array-init
   handler uses extra labels/regs and might trip a hard cap.

## 2026-04-30 follow-up — bisect outcome (step C, M1.2 A+B+C sprint)

The recommended bisect (steps 1 → 2 → 3 above) was executed against
`scripts/ci/native_v2_driver_self_compile_gate.sh` with a new
**stage1-smoke phase** (`examples/native/array_init_tail.sio` as
activator) so any stage1 hang would hard-fail the gate instead of
being masked by byte-identical stage2/3 determinism.

Outcome: **all three bisect steps are gate-green**, including the
full `parse_expr_ir`-recursion form (step 3). The previously-reported
spin did not reproduce. Most plausible explanations, in order:

1. The M1.1 self-compile gate hardening (`6b37a722`) plus the cluster
   #5/#6 speculative-rollback work (`c91127fd`) closed the interaction
   that produced the spin — the handler's new `parse_expr_ir` calls no
   longer hit the path that used to diverge when stage1 tried to
   compile itself.
2. The original handler that triggered the hang had a state-corruption
   bug (e.g. `V2_LAST_STRUCT_IDX` left stale across per-element parses)
   that the reconstructed bisect-3 handler does not have because it
   resets the flag around every element parse.

Either way, the handler landed is **still only local-scope**. Top-level
globals declared as `var NAME: [T; N] = [...]` (as in
`tests/run-pass/symbolic_test.sio`) still fall through the non-driver
global-registration path and are reported as `unsupported_frontend`.
Unblocking those 100+-element literal globals is a separate M1.2 task —
register the global slot + emit rodata/bss init — and is out of scope
for step 4 closure.

**Post-bisect run-pass inventory**: ok 74 → 75, nv2_compile 285 → 279
(−6), nv2_run 26 → 31 (+5). Five files moved from "does not compile"
to "compiles but runtime diverges" — mostly cases that use element
expressions the bisect-3 handler accepts but whose subsequent codegen
is still incomplete (e.g. negative literals, f64 element path, very
large local arrays). These are follow-up work, not step-4 regressions.
