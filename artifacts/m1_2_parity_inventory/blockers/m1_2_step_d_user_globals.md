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

## 2026-04-30 second-pass diagnosis

A follow-up session on top of `b9816786` added the missing piece the first
pass lacked: **hard-coding `USER_GLOBAL_*` as real driver-global slots**
(81–86) in `driver_global_id_tok` + `driver_const_value_tok` + bumping
`drv_driver_data_len` 21,233,664 → 22,806,528. Without that, stage1 fails
with `unsupported_frontend fn=user_global_id_tok token=…i` — because the
driver's own references to `USER_GLOBAL_COUNT` / `USER_GLOBAL_NAME_TOK`
inside `scan_user_globals` and `user_global_id_tok` couldn't resolve
through the hard-coded global path.

With those slots hoisted, stage1 **builds** (ok, 270800 bytes) but stage1
still **hangs** when it tries to compile itself to stage2 — and not in
`scan_user_globals` as hypothesis 1 guessed. Instrumented debug prints
narrowed the hang precisely:

```
dbg:scan_struct_begin        ← stage1 prints
dbg:scan_struct_end
dbg:scan_enum_end
dbg:scan_user_globals_end count=0     ← confirms scan_user_globals finished;
                                        USER_GLOBAL_COUNT=0 after, so
                                        user_global_id_tok in the
                                        parse-path hot loop is an O(1) no-op
dbg:scan_user_fns_end
dbg:user_fn_count=285
dbg:fn[0]                    ← first user fn (id=32 = V2_FIRST_USER_FN_ID)
dbg:cuf_begin id=32
dbg:cuf_name name_pos=1889   ← fn name token found
dbg:cuf_open=1896            ← `{` found
dbg:cuf_close=1931           ← matching `}` found
dbg:cuf_before_parse_block   ← entering parse_block_ir
  <hang — never prints dbg:cuf_after_parse_block>
```

So:

- The hang is inside **`parse_block_ir`** for the **first user fn**, which
  is `v2_driver_context_new` — a trivial function whose body is a single
  struct-literal tail expression.
- It is **not** `scan_user_globals` (hypothesis 1 from the first pass).
- It is **not** `ufn_record_user_global_*` overflow (hypothesis 2).
- It is **not** relocation collision (hypothesis 3): the hang is in
  stage1's parse phase, before any codegen emits relocations.
- It is **not** `user_global_id_tok` being slow: `USER_GLOBAL_COUNT=0`
  across the entire self-compile, so the resolver returns -1 immediately
  on every call.

The recurring pattern in stage1 failures like this ("empty fn name" in
error output, print streams that emit raw source chars) is `print_token_text`
being called with a `V2_USER_FN_TOKEN_IDX[k]` whose `PT_END[tok]` appears
to cover the whole file. Diagnostic: when stage1 printed
`dbg:fn[0]=1889 id=32 name=…` with `print_token_text(1889)` included, it
dumped ~200k characters that looked like a full lexical dump of the
file (all single-char punctuation tokens spaced out). This points at
**PT_END being out-of-range for user-fn name tokens in stage1**, which is
orthogonal to the hang but is part of the same diagnostic surface.

Most plausible remaining hypothesis for the hang itself: one of the
speculative `parse_expr_ir` rollbacks inside `parse_stmt_ir` that my
step-D additions made reachable for `TK_IDENT`-leading statements is
now entering a state where the rollback counter (`V2_UFN_COUNT`,
`V2_NEXT_REG`, or `V2_LAST_STRUCT_IDX`) is not being reset correctly on
one of the IDENT paths, causing the same tokens to re-parse forever.
This is speculative — the stash saved on the WIP branch
(`stash@{0}: m1.2-stepD-diagnosis-2026-04-30`) contains the debug-print
instrumentation for the next session to continue from.

**Recommended next action:** add a hard iteration cap in
`parse_block_ir`'s outer loop (e.g. `if iter > token_count * 4 { break }`)
which would turn the hang into a detectable infinite loop with a fn name,
identifying the exact statement that's cycling. Then bisect with
`user_global_id_tok` calls commented out one parse-site at a time.

---

## 2026-05-01 — No-progress guard landed; self-compile gate clean

Commit `21d19869` added the no-progress guard to `parse_block_ir`. The guard
detects when `parse_stmt_ir` returns without advancing (`p <= prev_p`) and
converts the infinite loop into a `native_compile: parse_block_no_progress`
diagnostic + `unsupported_frontend` error.

The full gate now passes:
```
[native-v2-driver-self] stage1-smoke OK ran=7 checked=7
[native-v2-driver-self] fixed-point md5=45165e0a04cd57fd1bec63bfa104f4e0
[native-v2-driver-self] PASS: baseline, stage1 driver, stage2 driver,
  fixed-point (stage2==stage3), hello parity across all stages,
  epistemic-fixed-point verified
```

No `parse_block_no_progress` messages were emitted during the full gate run,
confirming that the original hang (stage1 → stage2) was caused by PR #53's
user-global infrastructure (since reverted). The no-progress guard remains as
a permanent defensive measure.

**Next step:** Re-implement step D user-global infrastructure (USER_GLOBAL_*)
without the memory corruption that caused PR #53's revert. The `examples/native/user_global_basic.sio` test file is ready but NOT yet in the smoke cohort (baseline driver can't handle it yet — that's expected until step D lands).

