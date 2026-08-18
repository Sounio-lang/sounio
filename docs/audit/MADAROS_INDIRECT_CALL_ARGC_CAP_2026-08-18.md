# Madaros dispatch — `IrCallIndirect` codegen caps indirect calls at 2 arguments

- **Date:** 2026-08-18
- **Reporter:** fable-1 (CEI WS-A P2 verification)
- **Severity:** correctness/limitation — silent hard-abort of native codegen
- **Reproduces on:** current `main` `200b53419b`, built from source
  (`scripts/ci/build_modular_madaros.sh`), fresh detached worktree.
- **Family:** compiler backend (`self-hosted/native/codegen_x86_linux.sio`),
  same dispatch shape as #1799 (println kind-0) / #1800 (handle-table 182).

## Symptom

Any **indirect call** (a by-value / non-capturing closure called through a bound
name, or a function pointer) with **3 or more arguments** aborts native-v2
codegen with the generic bridge error and exit status 12:

```
Error: Failed to write native binary to /tmp/madaros-run.XXXX/main.elf rc=12
  error: native-v2 bridge compilation failed
```

The abort is unconditional on argument count and independent of argument type.

## Minimal reproductions (all on from-source `main` 200b53419b)

| Probe | Shape | Result |
|---|---|---|
| `let w = \|a,b\| {..}; w(3,4)` | non-capturing closure, **2** i64 | codegens OK¹ |
| `let w = \|a,b,c\| {..}; w(3,4,5)` | non-capturing closure, **3** i64 | **rc=12 abort** |
| `let w = \|a:f64,b:f64,c:f64\| {..}; w(..)` | non-capturing closure, **3** f64 | **rc=12 abort** |
| `let w = \|a:f64,b:f64,c:i64\| {..}; w(..)` | mixed, **3** total | **rc=12 abort** |
| `fn addf(a,b,c) {..}; addf(3,4,5)` | **named fn**, 3 args (direct `IrCall`) | correct `1.200000` |
| `let k=..; let w=\|a,b,c\|{..+k}; w(..)` | **capturing** closure, 3 args (direct `IrCall`) | correct on main² |

¹ The 2-arg non-capturing case codegens; at runtime it hits the *separate*
known kind-0 `println(closure-result)` SEGV (#1799 family) unless the result is
annotated — that is not this bug.
² The capturing (direct-`IrCall`) path handles ≥3 args correctly on `main`. (On
the stale branch `lane/fable-1/handle-table-182-dispatch`, off `dca2775061`, the
capturing path additionally *zeroed* its args at arity ≥3 — a branch-local
artifact already fixed on main; **not** part of this dispatch.)

The type-independence (3×i64 fails identically to 3×f64) proves this is an
**arity** cap, not a float-ABI issue.

## Root cause — exact site

`self-hosted/native/codegen_x86_linux.sio`, the `IrCallIndirect` arm (≈ line 8326):

```sounio
} else if instr_op == IrOpcode::IrCallIndirect {
    let argc = IR_A_ARG_COUNT[(ir_region_slot_r((*func).region, (i))) as usize]
    if argc > 2 {
        return false                     // <-- hard cap; becomes rc=12
    }
    let arg_pair = (ir_arena_arg_at(.., 0), ir_arena_arg_at(.., 1))
    if argc > 0 { .. nc_emit_load_rbp_disp32_reg(nc, 7, ..) }   // arg0 -> rdi
    if argc > 1 { .. nc_emit_load_rbp_disp32_reg(nc, 6, ..) }   // arg1 -> rsi
    let fnref_reg = IR_A_SRC1[..]
    nc_core_load_temp_to_rax(nc, fnref_reg)
    nc_emit_call_rax(nc)
    ..
}
```

`return false` from the per-instruction emitter is what the driver reports as
"native-v2 bridge compilation failed". The arm only ever materialises `arg0`
and `arg1`, into integer registers **rdi (7)** and **rsi (6)**; there is no
slot for `arg2..arg5`. The direct `IrCall` arm (named fns, capturing closures)
has no equivalent cap — hence the asymmetry.

## Impact

- Caps **all** indirect dispatch in the language at 2 arguments: by-value
  closures stored in locals, function-pointer tables, and any future
  first-class-function feature that lowers to `ir_call_indirect`.
- Directly blocks CEI (Certified Effect Interpretation, WS-A): non-capturing
  handler clauses `let op = |a,b,c,..|` are dispatched via `ir_call_indirect`
  (`self-hosted/ir/lower.sio` perform hook), so a `perform E.op(x,y,z)` with ≥3
  arguments cannot reach a non-capturing clause. The P2 demonstrator was kept to
  2 f64 args (`examples/effect_uncertainty_gum_vs_mc.sio`, GUM 0.831558 vs MC
  0.851582, rc=0) for exactly this reason.

## Secondary observation for the fixer

Even within the ≤2-arg envelope, the arm loads args into **GP registers only**
(rdi/rsi), with no SysV float/XMM classification. Empirically, 2 f64 args
through a non-capturing closure produce the correct result (verified: `0.700000`),
so Madaros closures evidently pass all args — including f64 — through GP
registers and the closure prologue reads them from GP. Any fix should preserve
that all-GP calling convention (or fix both sides together), not introduce XMM
routing on the caller side alone.

## Proposed fix (next dispatch — NOT done here)

Extend the `IrCallIndirect` arm to marshal up to 6 arguments into the SysV GP
argument registers, mirroring the direct `IrCall` arm's arg-loading:

- arg0→rdi(7), arg1→rsi(6), arg2→rdx(2), arg3→rcx(1), arg4→r8(8), arg5→r9(9).
- Replace the fixed `arg_pair` 2-tuple with a loop over
  `ir_arena_arg_at(slot, k)` for `k in 0..argc`.
- Keep the `fnref` load into `rax` and the `call rax` last (after args are in
  place), and guard `argc <= 6` (report a clean diagnostic beyond 6 rather than
  a bare `return false`).

Falsifier for the fix: the named-fn control (`addf/3`) must stay correct, and
the 2-arg non-capturing control must stay byte-identical.

## Evidence provenance

- Probes: `$CLAUDE_JOB_DIR/tmp/{direct3,i64_3,mix,capwit,nc2,named3}.sio`.
- Main build: fresh `git worktree add --detach origin/main`, HEAD `200b53419b`,
  `scripts/ci/build_modular_madaros.sh artifacts/self-hosted/madaros`
  ("Madaros ready … 100026487 bytes"), run via `MADAROS_RAW_BIN=<artifact>`.
