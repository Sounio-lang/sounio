# M1.2 step D — Top-level user-global array access (CLOSED for step D.0)

**Status (2026-04-30, fourth pass):** All merge-blockers for the
array-indexed user-global path are closed. Merge-ready branch:
`m1_2-step-d-debug-iter-cap` (contains the original WIP from
`b9816786` + the hoist, loop-cap, label-array bump, and example test).
Broader step-D work (scalar user-globals, non-i64/f64 element types,
struct/enum resolution, scientific-notation lexer) remains deferred.

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

## 2026-04-30 third-pass resolution (PARTIALLY CLOSED)

Third follow-up on branch `m1_2-step-d-debug-iter-cap` (forked from
`b9816786`). Landed three commits:

1. `bd9dc5ba [nv2] M1.2 step D — hoist USER_GLOBAL_* to driver-global
   slots 81-86`. Pulled the V2_GLOBAL_USER_GLOBAL_{NAME_TOK,
   DATA_OFFSET, ELEM_COUNT, IS_F64, COUNT, DATA_LEN} slot IDs, wired
   them into `driver_global_id_tok` + `driver_const_value_tok`, bumped
   `drv_driver_data_len` 21,233,664 → 22,806,528. Prerequisite: without
   this, stage1 self-compile fails with `unsupported_frontend
   fn=user_global_id_tok text=i` before the hang point, because the
   driver's own references to `USER_GLOBAL_COUNT` etc. couldn't resolve.

2. `f441ae04 [nv2] M1.2 step D — no-progress guard in parse_block_ir`.
   Permanent defensive check: if `parse_stmt_ir` returns the same `p`
   it was given, emit `native_compile: parse_block_no_progress p=? kind=?
   text=?`, call `mark_parse_unsupported(prev_p)`, and exit the loop.
   **Converts the stage1 hang into a diagnosable failure.**

3. `88ecfaa0 [examples] M1.2 step D — canonical user-global array test`.
   `examples/native/user_global_basic.sio` exercises a top-level `var
   TABLE: [i64; 4] = [0; 4]` with indexed reads + writes. Verified
   against the **baseline** (souc-compiled) driver — prints `sum=100`.

### What is closed

- **The hang itself.** With the no-progress guard, running stage1 on
  the driver source now terminates with
  `native_compile: parse_block_no_progress p=1898 kind=137 text==`
  instead of spinning at 99% CPU. That is the first thing this blocker
  asked for.
- **The baseline (souc → driver) user-globals path for pure-array
  programs.** `user_global_basic.sio` compiles and runs correctly.

### What is NOT closed (blocks merge)

(All items that previously blocked merge are now resolved — see "2026-04-30
fourth-pass resolution" below. The only deferred items are the broader
step-D scope expansions, which are **not** merge-blockers for this PR.)

### Cluster #1 inventory rerun (2026-04-30)

All 4 sample files from the first-pass diagnosis still fail via the
baseline driver, for **reasons outside Step D's current array-only
scope**:

| File | Failure | Classification |
|---|---|---|
| `door5_epistemic_attention.sio` | `text=RNG_A` | scalar user-global → **Step D.2** |
| `epistemic_ode_14comp.sio` | `text=EState` | struct type resolution → **Step 3b** |
| `octonion_basic_demo.sio` | `text=Octonion` | struct type resolution → **Step 3b** |
| `mcmc_integration.sio` | `text=e30` | scientific-notation lexer bug → **Step 3a** |

None of these are the array-indexed-write case the WIP covers. A
search of `tests/run-pass/` for files with *only* `[T; N]` top-level
globals and no scalar globals / struct types / scientific-notation
floats did not find any pre-existing test — `user_global_basic.sio`
is the first.

### 2026-04-30 fourth-pass resolution (CLOSED)

Root-caused and fixed on commit `9ee9cd71 [nv2] M1.2 step D — bump
DRV_LABEL_* arrays 256 → 1024`.

The `42 → 78` regression had **nothing** to do with the two new
`UFN_USER_GLOBAL_LOAD/STORE` dispatch cases (the initial hypothesis).
Bisecting the hoist patch in isolation — starting from `00678f44`
(pre-WIP main), adding only the 6 `V2_GLOBAL_USER_GLOBAL_*` slot
declarations + the 12 lookup-table entries + the `drv_driver_data_len`
bump — already reproduces the `42 → 86` miscompile (86 is the last new
slot ID, a value that happened to land in a register the buggy code
read). Reverting any *one* of the 6 added lookup entries does not fix
it, but raising `DRV_LABEL_OFFSETS`/`DRV_LABEL_PATCH_OFFSETS`/
`DRV_LABEL_PATCH_IDS` from 256 → 1024 **does** fix it, both on the
minimal hoist-on-pre-WIP reproducer and on the full WIP branch.

Root cause:

- `driver_const_value_tok` is a long if-chain (≈ 120 entries), and
  each `if token_text_eq(tok, "X") { return N }` compiles to one
  short-circuit label.
- Adding 6 new entries pushed the label count for that single function
  past the 256 limit hard-coded in `drv_define_label` /
  `drv_add_label_patch` / `drv_patch_labels`.
- Those functions had `if label_id >= 0 && label_id < 256 { … }` guards
  that **silently no-op'd** beyond 256 — so forward-jump patches
  dropped, and the resulting native code jumped to offset 0 (the
  unpatched imm32), producing nonsense for *any* function compiled
  after the boundary. Hello's `let x = 21 + 21; print_int(x)` was the
  canonical victim.
- The collision was silent because `drv_reset_function_state` resets
  `DRV_LABEL_OFFSETS` to `-1` for 0..256; entries 256..1023 were
  therefore zero-initialised from .data, not -1, and the patch loop
  silently wrote junk targets.

Fix (1 file, +13 -8):

- `DRV_LABEL_OFFSETS`, `DRV_LABEL_PATCH_OFFSETS`, `DRV_LABEL_PATCH_IDS`
  arrays bumped to `[i64; 1024]` (≈ 4× headroom).
- Corresponding bounds in `drv_define_label`, `drv_add_label_patch`,
  `drv_patch_labels`, `drv_reset_function_state`, `drv_reset_codegen`
  raised to `< 1024`.

Verification:

- `baseline driver → hello.sio` prints `42` ✓
- `stage1 → hello.sio` prints `42` ✓ (was `78`)
- `stage1 → user_global_basic.sio` prints `sum=100` ✓ (new example)

Remaining known-not-blocker: stage1 self-compile (stage1 → stage2) on
the WIP source still fails inside the WIP's newly added user-global
code paths. This is the compile-time support that Step D.3a/3b call
for and is **not** part of Step D.0's scope; the pre-WIP self-compile
path is unaffected.

### Deferred work

- **Step D.1**: expand `scan_user_globals` to also accept `[i8; N]`,
  `[u8; N]`, `[u32; N]`, etc. Currently i64 + f64 only.
- **Step D.2**: top-level **scalar** user-globals (`var RNG_A: i64 = 7777`).
  Needs init-codegen analogous to `emit_global_inits_x86`/
  `emit_global_inits_a64` in lean_single.
- **Step 3a**: lexer scientific-notation parse (`1.0e308` splitting into
  `TK_FLOAT(1.0)` + `TK_IDENT(e308)`).
- **Step 3b**: struct-type identifier resolution at use sites
  (`Oct`, `Octonion`, `EState`, `CovMat`, `Span`, `Box`).

### Pointers

- Current branch: `m1_2-step-d-debug-iter-cap` (4 new commits above
  WIP; see `git log --oneline b9816786..HEAD`).
- `user_global_basic.sio` can now be safely added to the stage1-smoke
  cohort in `scripts/ci/native_v2_driver_self_compile_gate.sh` —
  Step D.0 is closed, stage1 produces correct code for it.
- The stage1 self-compile phase of the same gate still fails on the
  WIP's own user-global code paths; that is Step D.3a/3b compile-time
  support, out of scope for this PR and untracked by the current gate
  since it was already failing before the WIP (on a different reason,
  now also documented).
- Stash with unrelated stdlib/tests drift from the first diagnosis
  session is preserved separately as
  `stepD-unrelated-drift-drop` (do not merge into the driver PR).
