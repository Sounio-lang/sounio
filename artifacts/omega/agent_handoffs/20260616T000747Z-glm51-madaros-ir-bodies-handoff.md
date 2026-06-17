# GLM 5.1 Madaros IR Bodies Handoff

Purpose: hand off a reproducible Madaros front-half failure so the GLM 5.1
lane does not have to reverse-engineer the trigger from the full Erdos lane.

## Blocker ID

`BLK-20260615-erdos-madaros-ir-bodies`

## Symptom

Madaros (default engine of `bin/souc`) fails with:

```text
native_v2_compile: front-half failed: ir_bodies_failed
```

when compiling `examples/erdos/cube_sieve_skeleton.sio` or the minimal
reproducer below.

## Minimal Reproducer

File:

```text
examples/erdos/reproducer_madaros_ir_bodies_2026-06-15.sio
```

Command:

```bash
cd /workspace/sounio
./bin/souc compile examples/erdos/reproducer_madaros_ir_bodies_2026-06-15.sio -o /tmp/out.elf
```

The reproducer contains only two functions: a recursive `pri` and `main`.

## Root Cause Found by Erdos Lane

The failure is **not** a semantic bug in the Erdos source. It is a hard per-function
IR instruction budget in the Madaros compiler:

- `self-hosted/ir/ir.sio:13`: `pub let IR_MAX_INSTRS: i64 = 128`
- `self-hosted/ir/ir.sio:704`: `pub instrs: [IrInstr; 128]` inside `IrFunction`
- `self-hosted/ir/lower.sio:3091`: `Lowerer.emit()` calls `report_error()` when
  `instr_count >= IR_MAX_INSTRS`.
- `self-hosted/compiler/module_frontend.sio:3122`: the error is surfaced as
  `ir_bodies_failed`.

A recursive function followed by 8 or more sequential `if` statements generates
enough IR instructions to exceed the 128-instruction budget. Bisection results:

| # sequential `if` after recursion | Result |
|-----------------------------------|--------|
| ≤ 7                               | ✅ compiles |
| ≥ 8                               | ❌ `ir_bodies_failed` |
| replaced with `else if` chain     | ❌ segmentation fault in `bin/madaros` |

The `else if` segfault is a separate bug in the same area.

## Files in Scope

Likely touched by a fix:

- `self-hosted/ir/lower.sio`
- `self-hosted/ir/*.sio`
- `self-hosted/native/codegen_*.sio`
- possibly `self-hosted/native/elf_bulk.sio`

## Erdos Lane State

- `cube_sieve_skeleton.sio` and `souc_sat.sio` are intentionally kept on the
  `lean_single` engine via `SOUNIO_SOUC_ENGINE=lean_single` in the Erdos scripts.
- This is a documented fallback, not a permanent solution.
- 10/10 lightweight Erdos gates pass with the fallback.

## Suggested GLM 5.1 Next Steps

1. Confirm the reproducer fails in your isolated Madaros worktree.
2. Decide between the two main fix strategies:
   - **Strategy A (preferred long-term):** increase `IR_MAX_INSTRS` and update
     every `[IrInstr; 128]` hardcoded array in `self-hosted/ir/*.sio` to the new
     size. Suggested target: 512 or 1024. Be careful with `self-hosted/ir/loop_opt.sio`,
     `self-hosted/ir/optimize.sio`, `self-hosted/ir/const_prop.sio`, and any other
     module that copies `IrFunction` by value or allocates temporary `[IrInstr; 128]`
     buffers. Also update `self-hosted/bootstrap/bootstrap_v0.sio:10014` which
     redeclares `IR_MAX_INSTRS`.
   - **Strategy B (quick workaround):** split very large functions in
     `cube_sieve_skeleton.sio` and `souc_sat.sio` into smaller helpers. This is an
     Erdos-side workaround and does not fix the underlying compiler limitation.
3. Fix the `else if` segmentation fault, since it is a related crash in the same
   lowering path.
4. Add small witness tests to the compiler test suite:
   - recursive function + 7 sequential `if`s (baseline)
   - recursive function + 8 sequential `if`s (must pass after fix)
   - recursive function + `else if` chain with 10 cases (must not segfault)
5. Verify the fix against the full `examples/erdos/cube_sieve_skeleton.sio`.
6. Once `cube_sieve_skeleton.sio` compiles, move on to `souc_sat.sio`, which has
   additional Madaros incompatibilities (missing `Mut` effects, `i32`/`i64`
   type mismatches, undeclared variables).

## Acceptance Gate

```bash
cd /workspace/sounio
./bin/souc compile examples/erdos/reproducer_madaros_ir_bodies_2026-06-15.sio -o /tmp/out.elf
test -x /tmp/out.elf
```

## Coordination Note

This handoff intentionally does not edit compiler source files. The shared
checkout has active WIP in `self-hosted/gpu`, `self-hosted/ir`, and
`self-hosted/native` from other agents, so compiler changes should happen in the
GLM 5.1 isolated worktree/lane.

Erdos lane will continue using the `lean_single` fallback until this blocker is
closed.
