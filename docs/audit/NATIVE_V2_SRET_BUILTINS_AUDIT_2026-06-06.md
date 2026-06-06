# Native-v2 SRET / large-aggregate return ABI — enumeration + audit

Date: 2026-06-06
Branch / worktree: `claude/sret-builtins` (worktree `../sounio-sret`, base `feat/exact-orc-machinery` @ `8fcf23d18`)
Scope: dispatch "② builtin SRET smashes" — enumerate native-v2 builtins returning
aggregates > 16 bytes and make their return marshalling ABI-correct.

---

## HONESTY ADDENDUM (2026-06-06, branch `honest/sret-builtins`, base `integration/native-v2-onto-exact-orc` @ `7718c86ca`)

This addendum supersedes §4's "main.sio won't build" blocker and corrects the
verification posture for the modular self-hosted compiler.

**1. main.sio DOES build on the integration base.** The §4 blocker was measured
against `feat/exact-orc-machinery` @ `8fcf23d18` (1024-local overflow). On the
integration base `7718c86ca` the bootstrap `./bin/souc` builds
`self-hosted/compiler/main.sio` to an 85 MB native ELF cleanly, and the honest
modular gate `scripts/ci/native_v2_e2e_codegen_suite_gate.sh` (which builds
main.sio from `./bin/souc`) PASSES 9/9 (scalar1/42/200/255, call, multicall,
control, control-ft, arith) — IR → ELF → exit-code, verified here.

**2. The forbidden driver gate was NOT landed.** The source lane's
`scripts/ci/native_v2_sret_builtins_gate.sh` drove `native_compile_driver.sio`
(a side-driver), not `main.sio`; it is excluded from the integration build rule
("the only acceptable verification builds main.sio"). It was never committed to
integration, so there is nothing to remove there. The honest e2e suite gate is
the only SRET-relevant gate and it builds main.sio.

**3. No SRET witness can be added to main.sio's compiled-in path today — and that
is the corrected, evidence-backed conclusion of this lane.** The compiled-in
native-v2 backend models structs as **heap handles**: `IrAlloc` allocates a
header+payload object on the runtime heap and `IrReturn` returns the *handle*
(a single pointer in `rax`). The `IrOpcode` enum (`self-hosted/ir/ir.sio:60`) has
exactly one return opcode, `IrReturn`, lowered at `codegen_x86_linux.sio:6152` as
an unconditional single-register `rax` return. There is **no** by-value
multi-eightbyte / hidden-pointer (`sret`) return opcode. The genuine >16-byte
by-value emitter `native_v2_core_emit_return_struct_into`
(`codegen_x86_linux.sio:6398`) is **dead code** in the compiled-in path: its only
reference is the import at `native_compile_driver.sio:50` (the forbidden side-
driver). Therefore no `compiler_main_make_native_v2_*_module` that main.sio can
build and run exercises the >16-byte by-value SRET ABI; a heap-handle "struct
roundtrip" witness would return a scalar pointer and must NOT be labelled "sret".

**Honest verdict for this lane:** the source lane made zero ABI changes
(`abi_lower.sio` / `frame.sio` / `codegen_x86_linux.sio` unchanged — its own §
"No-regression bar" confirms this), so there is nothing to port. main.sio builds
and the honest modular gate passes, but it contains no SRET witness and one
cannot be added without first implementing the missing feature:
- a struct-return IR opcode (e.g. `IrReturnStruct`) in `IrOpcode`,
- a dispatch arm in `compile_ir_function_v2_core_ir_into` that calls
  `native_v2_core_emit_return_struct_into` (+ `native_v2_core_emit_rdx_to_temp_into`
  on the caller for the rax:rdx read-back, and a hidden-pointer path for >2
  eightbytes), and
- an `IrModule` builder + `--native-v2-emit-sret*` dispatch + `emit_and_check`
  line in the e2e suite gate.

That is a backend feature, out of scope for "make the lane honest." This lane's
durable, honest deliverable is the corrected premise (above) plus the enumeration
table (§1) showing 0/19 builtins return a >16-byte aggregate.

## TL;DR

1. **The dispatch premise does not hold: NO native-v2 builtin returns an aggregate
   larger than 16 bytes.** Strings are bare `*u8` (NUL-terminated C pointers), not
   `(ptr,len)` structs, so every one of the 19 builtins returns a single-register
   scalar (`i64` / `bool` / `f64` / pointer). There is therefore no builtin whose
   `sret` marshalling can be smashed.
2. **The genuine `> 16`-byte return path is USER functions returning structs**, not
   builtins. That path already has dedicated SRET machinery in
   `native_compile_driver.sio` (register multi-field return for ≤ 6 flat fields,
   hidden-pointer `sret` for `> 6`). On source inspection it is internally
   self-consistent between caller and callee.
3. **No ABI change was made.** Per repo operating-principle #6/#7 and the dispatch
   guardrail ("do NOT paper over… fix the marshalling"), changing load-bearing ABI
   code requires an empirical reproduction of a smash first. None could be produced
   (see "Toolchain blocker") and inspection found no inconsistency. Inventing a fix
   to match the assumed premise would be drift.
4. **Deliverables that are real and durable:** the enumeration table below; an audit
   of the real `> 16`-byte struct-return ABI; 7 SRET witness programs
   (`tests/native_v2_sret_builtins/*.sio`, all `souc check`-clean); and a
   regression gate `scripts/ci/native_v2_sret_builtins_gate.sh` that locks the
   large-aggregate return behavior SOURCE → ELF → exit-code.
5. **Empirical SOURCE → ELF verification is blocked in this environment** — no
   available compiler can emit native ELF from real `.sio` source (the documented
   "native-v2 wall"), and the pinned reference release (`v1.0.0-beta.5`) is offline
   (HTTP 404). This blocks both the new gate and the 11 pre-existing
   `souc run …driver… -- src` gates equally; it is an environment limitation, not a
   property of the code under test.

### No-regression bar (the strict part of the dispatch)

The changeset is **purely additive** — `git status` shows only untracked files
(`docs/audit/NATIVE_V2_SRET_BUILTINS_AUDIT_2026-06-06.md`,
`scripts/ci/native_v2_sret_builtins_gate.sh`, `tests/native_v2_sret_builtins/`) and
**zero modified tracked files**. No source any existing gate executes was touched
(in particular no ABI file: `abi_lower.sio`, `frame.sio`, `codegen_x86_linux.sio`,
`native_compile_driver.sio` are unchanged). Therefore the change **cannot** alter any
pre-existing gate's exit code, independent of whether the suite can be run offline.
The no-regression *intent* is met by construction; a literal "whole suite green"
demonstration is the part blocked by the offline toolchain.

## 1. Enumeration — every native-v2 builtin and its return ABI

Dispatch table: `native_v2_builtin_id_for_func_ref` in
`self-hosted/native/codegen_x86_linux.sio:1045`. Emit bodies: `emit_builtin_*`
(by-value) and `emit_builtin_*_into` (`&!`) in the same file.

| id | builtin | signature | return type | return size | return reg | aggregate > 16B? |
|---:|---|---|---|---:|---|---|
| 1 | `print_int` | `(i64) -> unit` | unit | 0 | — | no |
| 2 | `print_char` | `(i64) -> unit` | unit | 0 | — | no |
| 3 | `print` | `(*u8) -> unit` | unit | 0 | — | no |
| 4 | `get_arg_count` | `() -> i64` | i64 | 8 | rax | no |
| 5 | `get_arg` | `(i64) -> *u8` | pointer | 8 | rax | no |
| 6 | `str_len` | `(*u8) -> i64` | i64 | 8 | rax | no |
| 7 | `str_eq` | `(*u8,*u8) -> bool` | bool | 1 | rax | no |
| 8 | `str_slice` | `(*u8,i64) -> *u8` | pointer | 8 | rax | no |
| 9 | `starts_with` | `(*u8,*u8) -> i64` | i64 | 8 | rax | no |
| 10 | `str_concat` | `(*u8,*u8) -> *u8` | pointer | 8 | rax | no |
| 11 | `read_file` | `(*u8) -> *u8` | pointer | 8 | rax | no |
| 12 | `write_file` | `(*u8,*u8,i64) -> i64` | i64 | 8 | rax | no |
| 13 | `file_size` | `(*u8) -> i64` | i64 | 8 | rax | no |
| 14 | `sqrt` | `(f64) -> f64` | f64 | 8 | xmm0 | no |
| 15 | `print_f64` | `(f64) -> unit` | unit | 0 | — | no |
| 16 | `exp` | `(f64) -> f64` | f64 | 8 | xmm0 | no |
| 17 | `log` | `(f64) -> f64` | f64 | 8 | xmm0 | no |
| 18 | `sin` | `(f64) -> f64` | f64 | 8 | xmm0 | no |
| 19 | `cos` | `(f64) -> f64` | f64 | 8 | xmm0 | no |

Key evidence that the string model is a bare pointer (so "string-returning"
builtins are NOT 16-byte `(ptr,len)` structs):

- `emit_builtin_str_slice` (`:2908`) computes `rax = rdi + rsi` and returns it — a
  single pointer, no length.
- `emit_builtin_str_concat` (`:2958`) bump-allocates `len_a+len_b+1`, copies, NUL-
  terminates, returns the buffer pointer in `rax`.
- `emit_builtin_str_len` (`:2832`) walks to the NUL terminator — there is no length
  field to read.
- `get_arg` (`:2672`) returns `argv[n]` (a `char*`) in `rax`.

Conclusion: **0 / 19 builtins return an aggregate > 16 bytes.** The dispatch's
"several builtins return large structs and smash memory" does not apply to this
backend.

Completeness: the table covers the `native_v2_builtin_id_for_func_ref` dispatch
(`codegen_x86_linux.sio:1045`). The driver also exposes its own `V2_BUILTIN_*`
intrinsics (`pt_read_kind_code`/`pt_read_start`/`pt_read_end`/`pt_read_int_value`,
`str_char_at`, `arg_count`, `driver_read_file_to_globals`, `indirect_call`); each is
scalar by inspection (they return an `i64`/`bool`/pointer into a global or a single
register — e.g. `native_v2_core_emit_str_char_at_into:2754`,
`…_driver_read_file_to_globals_into:2782`), so they do not change the conclusion.

## 2. The real `> 16`-byte return path — user struct returns

Large-aggregate returns happen for **user functions returning structs**, lowered in
`self-hosted/compiler/native_compile_driver.sio`. The convention (caller and callee
must agree) is an *internal* ABI, not the SysV-C one:

- **Threshold:** `STRUCT_FLAT_SIZE[…]` flat-field count. `> 6` flat fields ⇒ `sret`;
  otherwise in-register multi-field return. The threshold is applied identically on
  both sides:
  - callee return: `drv_emit_return_struct` (`:7220`) — `if fcount > 6 …`
  - callee entry sret-pointer save: `:6775` / setup at `:7796`–`:7811`
    (`if fn_ret_flat > 6 { V2_CURRENT_FN_SRET = 1; … }`)
  - caller call-site: `:7636` (`if rf > 6 { drv_emit_call_argc_sret … }`)
  - caller dst allocation: `:7931` (`if flat_size > 6`)
- **In-register path (≤ 6 fields):** callee loads field *k* into
  `rax,rdx,rcx,r8,r9,r10` (`drv_emit_return_struct:7225`–`7230`); caller reads the
  same registers back into `dst+k` (`:7641`–`7646`). Internally consistent.
- **SRET path (> 6 fields):** caller `lea rdi, [rbp+slot(dst)]` and passes real args
  shifted to `rsi…` (`drv_emit_call_argc_sret:7156`); callee saves `rdi` to a vreg on
  entry (`:6775`) and writes field *k* to `[rcx - k*8]` through that pointer
  (`drv_emit_return_struct_sret:7200`). The descending `-k*8` displacement matches
  the descending stack-slot layout the caller uses for `dst+k`, so field *k* lands at
  `buffer - k*8`. Internally consistent.

This is **not** SysV-compatible (SysV would put any aggregate > 16 bytes in MEMORY
class / `sret`, and a 24–48-byte struct would never be returned in 6 GPRs). It is
nonetheless **self-consistent within driver-emitted code**, and no return value
crosses to the C ABI (the only external calls are syscalls and the scalar-only
builtins), so within the native-v2 world it round-trips. Source inspection did not
reveal a caller/callee mismatch or a one-past-end walk on this path.

Scope of this inspection: it covers the **return** emission (caller dst alloc +
field read-back, callee field write) and the sret-pointer save/threading. It does
**not** independently verify the *callee's parameter reception* when an sret return
collides with a real argument (rdi taken by the hidden pointer ⇒ params shifted to
rsi…); that interaction is what `ret7_arg.sio` is designed to catch at run time.
"Self-consistent on inspection" is therefore a hypothesis the gate is built to
falsify, not a proof of correctness.

## 3. Witnesses (ready; locked by the new gate)

`tests/native_v2_sret_builtins/` — each `fn main() -> i64` returns the sum of all
fields of a struct returned by value from another function; the process exit code is
that sum, so a dropped/garbled field changes the exit code.

| witness | struct size | path exercised | oracle exit |
|---|---:|---|---:|
| `ret2_16b.sio` | 16 B | rax:rdx boundary (≤ 2 eightbytes) | 33 |
| `ret3_24b.sio` | 24 B | in-register multi-field (first size > 16 B) | 60 |
| `ret6_48b.sio` | 48 B | in-register multi-field upper edge (6 fields) | 100 |
| `ret7_56b.sio` | 56 B | SRET path lower edge (`> 6` ⇒ hidden pointer) | 28 |
| `ret8_64b.sio` | 64 B | SRET path | 108 |
| `ret7_forward.sio` | 56 B | SRET **return-forwarding** (relay returns a call's result) | 84 |
| `ret7_param.sio` | 56 B | SRET return **+** large-struct by-value parameter | 56 |
| `ret7_arg.sio` | 56 B | SRET return from a builder **taking an arg** (rdi=sret ptr ⇒ arg shifted to rsi) | 98 |

Oracle values avoid the shell-reserved exit codes {126,127,128,134,139} so a
non-program failure (exec-fail / SIGABRT / SIGSEGV) can never be mistaken for a PASS.

All 8 are `souc check`-clean (verified with
`/workspace/sounio-nv2-consolidate/bin/souc.elf check`). **Caveat:** `check` runs the
*general modular* frontend; the gate compiles through `native_compile_driver.sio`'s
*restricted scalar* frontend, which parses/lowers independently. No pre-existing
native example returns a struct with more than 2 flat fields, so there is no prior
evidence the driver frontend supports 3–8-field struct construction/return at all.
Consequently the **first** gate run must distinguish two failure modes:
"driver frontend does not support this construct" (frontend gap, surfaces as a
parse/lower error or empty output) vs. "ABI smash" (binary emitted, wrong exit code).
Only the latter is an ABI bug; the former is a separate frontend task.

Gate: `scripts/ci/native_v2_sret_builtins_gate.sh` (executable). It raises the stack
(`ulimit -s 1048576`, as the driver frontend has multi-MB frames), type-checks the
driver, then for each witness compiles SOURCE → ELF via
`souc run native_compile_driver.sio -- <src> -o <bin>`, runs `<bin>`, and asserts the
exit code equals the oracle. It fails closed if the driver emits nothing.

## 4. Toolchain blocker — empirical SOURCE → ELF is not reproducible here

The dispatch's acceptance bar ("source → ELF + matching exit code or it didn't
happen") cannot be met in this environment because **no available compiler can emit
native ELF from real `.sio` source.** Evidence (all commands re-runnable):

- `bin/souc`, `bin/souc-linux-x86_64*`, `artifacts/self-hosted/souc-self-hosted-*`
  are all `mini_native` (positional `<src> <out>` CLI, no `run`/`check`
  subcommands). `mini_native` is **not** import-tolerant: building the modular
  `main.sio` aborts with `error: too many local variables in function (max 1024)`;
  building `native_compile_driver.sio` aborts on tuple-field type errors in
  `refinement.sio` / `opt_cleanup.sio`. So the current branch's flag-CLI `souc`
  cannot be built locally.
- The pinned reference release `souc-linux-x86_64` `v1.0.0-beta.5`
  (the version `scripts/lib/resolve_souc.sh` expects) returns **HTTP 404** — offline.
- The local `artifacts/omega/souc-bin/souc-linux-x86_64-gpu` is `v1.0.0-beta.4` and
  **SIGABRTs** on `check` of the current driver (too old).
- A flag-CLI `souc` built fresh from the known-buildable `531403d84`
  (`/tmp/host_souc.elf`, `v0.80.0`) `check`s simple programs but **SIGSEGVs (139)**
  when running/native-compiling the driver — even with `ulimit -s` at 1 GiB, so it
  is not a stack-size issue.
- Every sibling-agent modular `souc` (`/workspace/mc-build/mc.elf`,
  `/workspace/sounio-nv2-consolidate/bin/souc.elf`, `…/sounio-fnptr-integ/…`, etc.,
  all `v0.80.0`) either SIGSEGVs on the driver or runs it to a no-op: it prints
  `Native dispatch: … fallback=unresolved_default_x86_64_linux` then `Compilation
  successful! Output: <path>` **without writing the file**. The native source → ELF
  backend resolves to an `unresolved_default` stub in every available build.

This is the documented "native-v2 wall": all available `souc` binaries were
bootstrapped by a `mini_native` that carries the very large-struct-return bug at
issue, so none can compile real source to a correct native ELF; the only reference
that can (`beta.5`) is unreachable offline.

Note on the working pattern: the `souc run native_compile_driver.sio -- <src>` path
is non-functional on **every** available `souc`, including the canonical native-v2
build base `integration/native-v2-onto-exact-orc`
(`/workspace/sounio-nv2-consolidate`). The pattern that *does* run in this
environment is a **hardcoded in-process witness mode** compiled into `souc` — e.g.
`--native-v2-emit13` (verified here) and the prior fn-pointer work's
`--native-v2-emit-fnptr`. A maximally-rigorous follow-up that does not depend on
`beta.5` would therefore be: from the `…-onto-exact-orc` base, add a hardcoded
`--native-v2-emit-sret*` witness mirroring these `tests/native_v2_sret_builtins`
programs, rebuild `souc`, and assert the exit codes — which additionally exercises
the *compiled-in* `native_v2_core_emit_return_struct_into` path rather than only the
driver's `drv_emit_return_struct*`. That is a separate, larger change on a different
base than this dispatch's `feat/exact-orc-machinery` scope.

What this means for the no-regression bar: the 11 pre-existing gates that use
`souc run …native_compile_driver.sio… -- <src>` (`native_v2_struct_return_gate.sh`,
`…_struct_gate`, `…_array_gate`, `…_nested_field_gate`, `…_logical_gate`,
`…_enum_match_gate`, `…_struct_param_gate`, `…_struct_mutation_gate`,
`…_out_param_boundary_gate`, `…_prebundle_gate`, `…_serious_track_gate`) are blocked
**identically** to the new gate. Gates that use the hardcoded in-process witness
modes still pass — e.g. `native_v2_e2e_exit_code_gate.sh` PASSES here via
`--native-v2-emit13` (verified, with `SOUNIO_MODULAR_SOUC=/tmp/host_souc.elf`).

## 5. Recommendation

- Treat the dispatch premise as corrected: there is no builtin-level SRET smash; the
  large-aggregate locus is user struct returns, whose machinery is already present
  and self-consistent on inspection.
- Land the witnesses + gate as a regression lock. When a working flag-CLI `souc`
  is available (pinned `beta.5` online, or a freshly bootstrapped current-branch
  `souc`), run `scripts/ci/native_v2_sret_builtins_gate.sh`:
  - all green ⇒ the `> 16`-byte struct-return ABI is correct; the lock stands and
    the premise correction is confirmed.
  - any mismatch ⇒ the failing witness names the exact size/shape that smashes, and
    the fix lands in `drv_emit_return_struct` / `drv_emit_return_struct_sret` /
    `drv_emit_call_argc_sret` (and, for the compiled-in path, the parallel
    `native_v2_core_emit_return_struct_into` in `codegen_x86_linux.sio`).
- Do not modify the ABI marshalling until such a reproduction exists; it is
  load-bearing for every other native-v2 consumer.
