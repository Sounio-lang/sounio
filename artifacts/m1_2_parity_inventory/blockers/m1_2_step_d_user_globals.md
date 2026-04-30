# M1.2 step D — Top-level user-global array access (BLOCKED)

**Status (2026-04-30):** Infrastructure is written but blocked on a
stage1-hang trap. WIP preserved on branch
`m1_2-step-d-user-globals-WIP` (commit `b9816786`) — DO NOT merge.

## Goal

Close cluster #1 of the M1.2 punch-list: programs referencing
top-level `var NAME: [T; N]` globals (RNG_A, MULT, Span, CovMat,
EState, …) currently fail with `kind=123 text=NAME` because the
driver's hardcoded global registry doesn't see them.

## What landed (in the WIP)

- `UFN_USER_GLOBAL_LOAD/STORE` opcodes (slots 21-22 in the UFN
  encoding, mirrored in the self-resolver)
- `USER_GLOBAL_*` driver registries (slots 74-79 in
  `driver_global_id_tok`)
- `drv_driver_data_len` bumped from 21,233,664 → 22,806,528
- `scan_user_globals` — top-level pre-pass that walks `var NAME:
  [T; N]` declarations and registers them
- `user_global_id_tok` — resolver that maps an identifier token to
  a user-global slot
- `drv_emit_user_global_load` — rip-relative lea + indexed load
  into the user-global data region

## Blocker

After wiring everything correctly, the driver:
- Self-compiles to stage1 successfully ✅
- Stage1 then spins at 99% CPU compiling the same source to
  stage2 — never terminates ❌

This is the **stage1-hang trap** Agent 4 documented for the array-init
case in `m1_2_step4_array_init.md`. The lean_single fixed-point gate
proves *determinism*, not *termination*. The newer stage1-smoke gate
(`f34c54c5`) catches simpler hangs but not this one (the smoke program
is too small to trigger whatever path is non-terminating).

## Diagnosis recommended (next session)

Likely candidates:
1. `scan_user_globals` recursing into the driver's own large
   `var X: [i64; N] = [0; N]` declarations and not advancing past
   the `[0; N]` initializer correctly — would explain a non-terminating
   outer loop.
2. `ufn_record_user_global_*` overflow check missing → infinite loop on
   exhaustion.
3. Relocation collision: driver-global relocations vs user-global
   relocations both writing to overlapping `.data` regions, causing
   the linker pass to never converge.

Recommended approach:
- Add `print_int(i)` instrumentation to `scan_user_globals` and run
  stage1 with `timeout 10s` — see how far it gets before the timeout.
- Once the offending loop is identified, fix in the WIP branch and
  re-run the self-compile gate.

## How to resume

```bash
git checkout m1_2-step-d-user-globals-WIP
# work, commit
# do NOT merge until stage2 actually terminates
# verify with:
bash scripts/ci/native_v2_driver_self_compile_gate.sh
bash scripts/ci/lean_single_fixed_point_gate.sh
```
