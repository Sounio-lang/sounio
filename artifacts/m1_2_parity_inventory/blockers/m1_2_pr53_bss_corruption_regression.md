# PR #53 stage1 BSS-corruption regression (REVERTED)

**Status (2026-04-30 evening):** PR #53 (`fa903deb`) reverted by
`860c4626`. WIP code preserved on the original branch
`m1_2-step-d-debug-iter-cap` (commit `fa903deb` and earlier
`bd9dc5ba` / `f441ae04` / `88ecfaa0`).

## What PR #53 did

Three things bundled into one 260-line driver-edit commit:

1. Bumped `DRV_LABEL_OFFSETS` / `DRV_LABEL_PATCH_OFFSETS` /
   `DRV_LABEL_PATCH_IDS` from `[i64; 256]` to `[i64; 1024]`. Closes
   the `21+21=78` hello.sio bug (label table overflow that silently
   dropped patches).
2. Added `V2_GLOBAL_USER_GLOBAL_*` slots 81-86 in
   `driver_global_id_tok` + `driver_const_value_tok`, bumping
   `drv_driver_data_len` from 21,233,664 → 22,806,528.
3. Re-introduced the `b9816786` user-globals scaffold:
   `USER_GLOBAL_*` registries, `scan_user_globals`,
   `user_global_id_tok`, `ufn_record_user_global_load/store`,
   `drv_emit_user_global_load`, the `UFN_USER_GLOBAL_LOAD/STORE`
   opcodes (21/22), plus the `parse_block_ir` no-progress guard from
   `f441ae04`.

## Why reverted

PR #53 broke `scripts/ci/native_v2_driver_self_compile_gate.sh` at the
stage2 step. The gate's stage1-smoke phase still passes (the small
cohort programs — fib, while_loop, logical_ops, array_basics,
struct_basic — don't trigger the bug). But when stage1 is invoked on
the **driver source itself** (the stage2 build), it exits rc=3 with
corrupted output: thousands of bytes of memory garbage in the
`fn=...` field of the first `unsupported_frontend` line.

### Repro

```bash
git checkout fa903deb
DRV=/tmp/r53_stage1
./bin/souc run self-hosted/compiler/native_compile_driver.sio -- \
  self-hosted/compiler/native_compile_driver.sio -o "$DRV"
chmod +x "$DRV"

# This produces 14k+ lines of garbage and exits rc=3:
"$DRV" self-hosted/compiler/native_compile_driver.sio /tmp/stage2.elf 2>&1 | head -3
```

Output starts with:

```
native_compile: unsupported_frontend fn=  [thousands of \0/garbage bytes]
```

The `fn=` field is reading past the IDENT token's actual end into
uninitialized memory.

### Why CI didn't catch this on PR

The PR's original CI run had **6 failing checks**: Contracts, macOS
arm64, Sounio Lint, Lean Proofs, Website, Full Test Suite. Only
"Native Self-Host (Linux x86_64)" passed. The PR was merged anyway.
The Linux self-host job may be running an older / leaner subset of
the gate that doesn't include the stage2 step where the bug
surfaces.

### Why the wall-clock timeout (`b845c522`) didn't catch it

The new wall-clock guard fires only on rc=124 (timeout). PR #53's
bug exits rc=3 quickly — not a hang. Different bug class than the
step-4 / step-D hangs the timeout was designed to catch.

## Likely root cause

Hypothesis: the BSS layout shift introduced by combining (1) the
`DRV_LABEL_*` bump (additional 6 KB), (2) the user-global slots
(slots 81-86 → +6 × 262,144 = +1.5 MB), and (3) the new
`USER_GLOBAL_*` driver registries pushes some array's byte address
past where another part of the codegen expects it. Since the
hello.sio test passed (per the PR title), the bug must surface only
at large input sizes or specific patterns — the driver source itself
being the trigger.

Specifically: the corrupted `fn=...` field suggests the IDENT-text
reader is using a pointer or offset that's now pointing at a
different BSS region. The first `kind=123` (TK_IDENT) token in the
driver's input is being read with a length field from the wrong
location.

## Re-land plan

The D.0 fix (label-table bump for hello.sio) is genuinely valuable.
To re-land safely:

1. **Bisect into separate commits.** Land the three pieces of PR #53
   independently:
   - Just the `DRV_LABEL_*` bump (256 → 1024).
   - Just the V2_GLOBAL_USER_GLOBAL_* slot allocation + data_len bump.
   - Just the scan_user_globals + ufn_record_user_global_* +
     drv_emit_user_global_load scaffold.
   - Run the full gate after each.
2. **Run all 6 currently-failing CI checks locally before merge.**
   Don't merge past failing CI on the convergence-critical lane.
3. **Add a stage2-smoke variant that catches BSS-shift bugs.** The
   current stage1-smoke runs stage1 against small user programs; we
   need a variant that runs stage1 against the driver source itself
   (which is what the existing stage2 step does) but with a sharper
   diagnostic — print the first non-prefix-`unsupported_frontend`
   error and validate the fn name field is sensible (printable ASCII,
   reasonable length).

## How to resume

```bash
# The reverted PR's full content:
git show fa903deb
# Or check out the WIP branch:
git checkout m1_2-step-d-debug-iter-cap
```

Pieces of the PR to bisect (commits on `m1_2-step-d-debug-iter-cap`
in the user's working chain):
- `bd9dc5ba` hoist USER_GLOBAL_* to driver-global slots 81-86
- `f441ae04` no-progress guard in parse_block_ir
- `88ecfaa0` canonical user-global array test
- (the DRV_LABEL bump came in fa903deb itself per its diff)
