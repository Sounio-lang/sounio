# Task C bundle — error-tail diagnostic (2026-05-23)

Branch `compiler/task-c-bundle-codegen` @ `d2cd36c1c`, 17 commits ahead of
merge-base (PR #161 `2d4bf086d`), **84 behind origin/main**. Working tree clean.
Companion to [[project_task_c_blocker]].

## How to reproduce
```
cd /workspace/.sounio-lane3
bash scripts/bootstrap/bootstrap_concat.sh   # errors print to STDOUT, not stderr
```
Capture: `bash scripts/bootstrap/bootstrap_concat.sh >/tmp/bc.log 2>&1`

## Current result
`RESULT: FAIL (exit code 139 = SIGSEGV)`. 124 files concatenated, 222,858 lines.

## Error tally (TRUNCATED — see segfault note)
~363 visible error lines: E200 ×92, unknown-field ×66, assignment-type-mismatch
×52, E001/Type-mismatch ×44, field-initializer ×26, unknown-identifier ×20,
exhaustiveness ×3, effect-not-declared ×2. (unknown-field down from 779 at lane
start — real progress.)

## ⚠️ The segfault truncates the count
Highest error line seen = **128,974** (egraph.sio), but the bundle is **222,858**
lines. The checker crashes (139) ~line 129K and **never checks the last ~94K
lines**. The 363 is a floor, not the true count. **Localize the crash before
grinding source errors** — fixing visible errors won't reveal what's past it.

## Rebase urgency: LOW
Branch∩main overlap = **2 files only**: `self-hosted/compiler/lean_single.sio`
(+ trivially `bin/souc`). `check.sio` (the 536-line workhorse of this lane) does
**not** overlap. So root-cause work won't be heavily redone; rebase can wait.
(Main did land #181 A64 parity, #176/#178 Windows PE, #183 graphics.)

## Line→file map technique (bundle strips `module`/`use` lines)
The concat manifest in the log gives post-strip line counts per file:
```
awk '/^[[:space:]]*self-hosted\/.*[0-9]+ lines/ {
  path=$1; for(i=1;i<=NF;i++) if($i=="lines") n=$(i-1);
  s=cum+1; cum+=n; print s, cum, path }' /tmp/bc.log > /tmp/filemap.txt
# then: awk -v L=<errline> '$1<=L&&$2>=L{print $3}' /tmp/filemap.txt
```
Note: local line ≠ source line (module/use stripped); read `build/bootstrap_stage1.sio`
directly at the error line to see what the checker saw.

## First error cluster (NOT a one-line fix)
- bundle 2241/2245 = `read_byte()` → **no `fn read_byte` anywhere**; not in
  resolve/effects/check. It's an IO intrinsic the checker's resolver doesn't know.
  Fix path: register it where other IO intrinsics live (cf. `lean_single.sio`
  special-cases `read_line` at `fn_find < 0`).
- bundle 2247 = `lex(buf, len, 0)` (3 args) → matches only
  `bootstrap/bootstrap_v0.sio:1904 pub fn lex(source, source_len, file_id)`.
  This is a **concat order / file-inclusion** problem (Sounio has no forward
  refs), not a type error. Check whether bootstrap_v0 is in the `core` profile
  and precedes `lexer/mod.sio`.
- E200 string is emitted from the shared resolver, not `check/*.sio`.

## Recommended next-session order
1. Localize the SIGSEGV (bisect bundle: check lines 1–129K vs full; or find the
   construct near bundle ~129K in egraph.sio). The crash is the real wall.
2. Register `read_byte` intrinsic + fix `lex/3` inclusion-order → kills a chunk
   of the 92 E200s.
3. Re-run, re-tally, then batch the type-mismatch / field-initializer families.
4. Rebase onto main only after the segfault is gone (overlap is just lean_single).

---

## SEGFAULT LOCALIZED (2026-05-23, follow-up)

Bisected the crash by truncating `build/bootstrap_stage1.sio` at increasing line
counts and watching the exit code flip 1 → 139:

| head -N | exit |
|--------:|:----:|
| 100000  | 1 (clean) |
| 106000  | 1 (clean) |
| **106100** | **139** |
| 107000  | 139 |
| 222858 (full) | 139 |

**Crash trigger: bundle lines 106001–106100** = the opening body of
`fn deserialize_ir_epistemic_section` in `self-hosted/ir/serialize.sio`
(451 lines; 36 `while` + 25 `if`; sibling `serialize_ir_epistemic_section` 403
lines). The pattern is array-field stores into the ~30-array-field giant struct
`IrEpistemicSection` (`ep.model_families[i] = pair.0`, `ep.policies[i] = …`, …).

**Ruled out empirically:**
- **Stack overflow** — re-ran the crashing input with `ulimit -s 1048576` (1 GB,
  80× default). Still exit 139. Not recursion depth.
- **TypeEnv table overflow** — `check/env.sio` is bounds-guarded: `push_scope`
  checks `< 64`, `bind` checks `< 128`. Silently drops past cap; cannot SIGSEGV.

**Prime suspect — unguarded fixed field/index tables in `check/check.sio`:**
the checker has small caps `fields: [i64; 8]` (l.8540), `[i64; 6]`, `[i64; 16]`
(l.8741) with index writes like `seen[model_idx as usize] = 1` in
`collect_audit_posterior_weight_fields` — **no bounds check on `model_idx`**. A
struct/section with more entries than the cap → out-of-bounds write → SIGSEGV.
This is the family to audit first.

**Next-session fix path:**
1. Add bounds guards (`if idx < CAP`) to the `seen[...]`/`fields[...]` writes
   around `check/check.sio:8540–8760`, or raise the caps to cover
   `IrEpistemicSection`'s field count. Re-run truncated `/tmp/crash.sio`
   (`head -106100 build/bootstrap_stage1.sio`) — fast crash repro, no full build.
2. Once exit ≠ 139, re-run full `bootstrap_concat.sh` to get the **true**
   (un-truncated) error count past line 129K.
3. Then resume the source-error tail (read_byte intrinsic, lex/3 order, etc.).

Repro one-liner: `head -106100 build/bootstrap_stage1.sio > /tmp/crash.sio && bin/souc check /tmp/crash.sio; echo $?`

---

## Fix attempt (2026-05-23) — target corrected, 5 hypotheses ruled out

**CRITICAL REDIRECTION: the crashing checker is `self-hosted/compiler/lean_single.sio`,
NOT `check.sio`.** `bin/souc` is a wrapper → self-hosted native compiler built from
`lean_single.sio` (per `scripts/ci/build_native_souc.sh:99`; the binary has 0 Rust
marker strings). `check.sio` is the *checkee* (the modular bundle being checked),
not the running checker. **All crash-fix edits must go in `lean_single.sio`
(32,320 lines), and verifying them requires the fragile boot4→lean_single rebuild.**

Trigger struct `IrEpistemicSection` (ir/ir.sio:1737): ~25 array-of-struct fields
`[IrXInfo; 32|64]` + counts (~50 fields), megabytes of layout.

**Hypotheses tested empirically — ALL NEGATIVE (don't re-test these):**
1. Stack overflow — re-ran crash input with `ulimit -s 1048576` (1 GB). Still 139. ✗
2. TypeEnv overflow — `check/env.sio` bounds-guarded (`<64`, `<128`). ✗ (also wrong file)
3. lean_single local tables — `VAR_*: [i64; 1024]`; 75 locals << 1024. ✗
4. Struct field-count — synthetic struct w/ 81 fields + `[Sub;32]` arrays: exit 0. ✗
5. Total struct byte-size — synthetic `[Sub(200×i64);32]×25` (~MB): exit 1 (clean
   reject), not 139. ✗

**Conclusion:** the SIGSEGV is specific to the *real* `deserialize_ir_epistemic_section`
in context (nested real `IrXInfo` types + the `ep.arr[i] = readfn(...).0`
tuple-store pattern), not any simple guessable cap. **Pinpointing needs a debugger
backtrace — `gdb` is NOT installed in this env** — or printf-instrumentation of
`lean_single.sio` across rebuild cycles. A blind cap-bump is unjustified (no
identified table).

**Recommended next-session entry:** install `gdb`, then
`gdb -batch -ex run -ex bt --args <resolved-souc> check /tmp/crash.sio` to name the
exact crashing routine in the checker. That single backtrace replaces all the
guesswork above.

---

## ROOT CAUSE FOUND via gdb (2026-05-23, breakthrough)

`gdb` IS installable here — `sudo apt-get install -y gdb` (passwordless sudo).
Resolved binary the wrapper runs: `bin/souc-linux-x86_64` (invoked `<src> <out>`,
i.e. **compile mode**, not pure check). The crash is in **code emission**, not
type-checking.

**Faulting instruction** (RIP `0x51eaeb`): `mov %dl,(%rax,%rcx,1)` with
`rax=0x14849268` (= base of `var CD: [i8; 134217728]`, the 128 MB code buffer)
and **`rcx=0xfffffffc`**.

**Faulting function: `patch32(off, target)`** (lean_single.sio:~16585):
```
fn patch32(off: i64, target: i64) with Mut, Panic {
    let rel = target - off - 4
    CD[off as usize] = (rel & 0xFF) as i8      // ← off = 0xfffffffc → CD[4294967292] → OOB
    ...
}
```
gdb breakpoint at entry: **`off = target = 0xfffffffc = 4294967292`**. This is a
32-bit `-4` zero-extended into an i64 — a **32-bit truncation bug**: a patch
offset that should be a small position (or a `-1`/`-4` "unset" sentinel) was
passed through a 32-bit-narrowing op, becoming ~4 GB. `CD` is only 128 MB →
out-of-bounds store → SIGSEGV. (So it is an **OOB overflow via truncation**, not
a literal negative index — that's why no `[..;N]` cap nor negative guard explains it.)

**Call chain (return addrs, stripped binary, no symbols):**
`0x60db1f → 0x604e03 → 0x5803a4 → 0x53ec16 → patch32(0x51ea6d)`.

**Sibling latent bug (same class, NOT this crash — A64 path):**
lean_single.sio:25668 `let rc_ok_patch_a = CL - 4` in the AArch64
`require_confidence` codegen — captures a patch offset *after* emitting, so if
`CL < 4` it goes negative. Worth fixing alongside.

**Hypotheses ruled out empirically (do NOT re-test):** stack overflow (1 GB),
TypeEnv `[128]`/scope `[64]`, local tables `[1024]`, struct field-count (81 clean),
total struct size (clean reject), LOOP_TOP/LOOP_END_PATCH `[16]` (sequential loops
to 36 = clean; depth-indexed, not cumulative), isolated array-field-store/`&&`/
nested-if patterns (all clean — needs the real ~36-block scale).

**Next-session fix path (advisor-gated to ONE rebuild cycle):**
1. Find the x86 `patch32` caller that yields `off=0xfffffffc`. Fastest: gdb
   `break *0x51ea6d` `condition 1 ($edi==0xfffffffc)` then `bt`/disassemble the
   caller at `0x53ec16` to its prologue; map its shape to a `patch32(...)` callsite
   (candidates: 12284/12547/12927 slice/array-bounds patches — deserialize is
   array-store-heavy). The truncation is likely an `as i32`/`em32`/`rd32_at`
   round-trip on a stored patch offset.
2. Fix the truncation at the producer (keep the offset i64), or add a
   **panic crash-boundary** in `patch32` (`if off < 0 || off as i64 >= 134217728
   { panic }`) — a guard ONLY as a loud boundary, never silent skip (silent =
   miscompiled self-host binary, strictly worse).
3. Rebuild via `scripts/ci/build_native_souc.sh` (boot4→lean_single — fragile, can
   time out; budget ONE attempt). Re-run `/tmp/crash.sio`; if exit≠139, run full
   `bootstrap_concat.sh` for the true error count past line 129K.

## RESOLVED (2026-05-23) — root cause + fix landed + verified

**Real root cause: code-buffer (`CD`) overflow, not a truncation/tuple bug.**
Instrumenting `patch32` and rebuilding (`SOUNIO_FORCE_SOURCE_BOOTSTRAP=1
build_native_souc.sh` — succeeds in <1 cycle) showed the bad call is
`patch32(off=134217728, ...)` where `134217728 = 0x8000000 = the exact capacity of
var CD: [i8; 134217728]` (the 128 MB code buffer). Compiling the bundle emits
>128 MB of machine code (the giant `IrEpistemicSection` struct → a 10 MB stack
frame in fn#5472 and massive by-value copy code — see `warning: stack frame too
large (10242384 bytes)`). `em()` already saturates + sets `CD_OVF` at the cap, but
**`patch32` had no such guard** → it wrote `CD[CL]` one past the end → SIGSEGV.

**Fix (lean_single.sio:16584 `patch32`):** guard the buffer bound exactly like
`em()` — `if off < 0 || off + 3 >= 134217728 { set CD_OVF; return }`. Minimal,
matches existing overflow machinery, no silent miscompile (the run is already
doomed once `CD_OVF` is set; this just lets it fail cleanly instead of crashing).

**Verified:** rebuilt compiler on the full bundle → **exit 1 (was 139), no
segfault**, and the true error count is unmasked: **5303 errors** (was truncated
to ~363 by the crash), checker now reaches **line 178690** (was stuck at 128974).

**Remaining for Task C (now unblocked & measurable):**
- 5303 type errors to work through (the real tail — E200/unknown-field/type-mismatch).
- The deeper perf issue: `IrEpistemicSection` by-value codegen is pathological
  (10 MB frame, >128 MB code). Long-term, that struct should be heap/pointer-based,
  or enlarge `CD` — but the crash itself is fixed.
- **Binary propagation:** the checked-in `bin/souc-linux-x86_64` / pinned artifact
  still lack the fix; rebuild + commit the binary to make gates see it.

---

**Caller pinned to TUPLE-BINDING codegen.** (Superseded by the RESOLVED section
above — the caller is irrelevant; the real cause is the CD cap.) Disassembling the immediate caller
(ret `0x53ec16`) shows it zeroes the globals at `0x250098e8`/`0x250098f0`
(= `TUP_PAT_NEXT`/`TUP_RHS_NEXT`, declared right after `patch32` at ~16586) then
`mov 0x8(%rsp),%rdi; mov 0x0(%rsp),%rsi; call patch32`. So the crashing `patch32`
is inside the x86 **tuple destructure/bind** path (`tuple_bind_pattern_x86`
l.16830 / `tuple_destructure_from_ptr_x86` l.16685 and neighbours). This matches
deserialize's `let pair = read_X(buf,p)` + `ep.field[i] = pair.0` /
`p = cN.1` tuple-extraction pattern at scale. **Start the next session here:** find
the `patch32(off, …)` reached from the tuple-bind path where `off` is built from a
32-bit-narrowed value (look for `em32`/`as i32` round-trips on a saved offset in
16595–16900), confirm with `break *0x51ea6d` + `bt`.

---

## Error-tail attack (2026-05-23 cont.) — 5304 → 423

**Root cluster fixed: top-level `let` constants (E200 555 → 20).** lean_single's
global-decl scanner registered `const`/`pub const` but not `let`; the modular
dialect declares 741 module-scope scalar constants as `let NAME: T = VALUE`.
Added a `let` branch (lean_single.sio ~5463, mirrors bare-const + `A - B` literal
fold) and raised the const guard 511 → 4095. **Total errors 5304 → 423.**
Commits `3ac470dbc` (source) + `f1694540d` (binary, fixed point `e254d3da`).

**unknown-field cluster (62) — NO single root (verified, do not re-investigate):**
split 34 `(*ptr).field` (mostly `(*list).head`) + 25 direct `var.field`. Match-arm
binding already propagates inner type (lean_single.sio:20130, handles
`Option<Box<T>>`). Five isolated repros with the current binary ALL pass:
`Option<Box<Node>>`+`(*list).head`, non-boxed `Option<Node>`, direct `Box<Node>`,
recursive struct w/ struct-typed head, impl-method-with-self. Struct table is fine
(808 structs < 1024 cap; max 83 fields < 128 stride). So these are case-specific —
likely `st_find` hash collisions among 808 structs or type edge-cases with real
types, NOT a systematic root. This is the genuine per-case Task C tail.

**Remaining 423 by category:** unknown-field 62, assignment-mismatch 54,
type-mismatch/E001 44, field-init 27, E200 20, arity 7, exhaustiveness 5,
effects 4. Heterogeneous — incremental Cat-D work, ~1 rebuild cycle per batch.

**Minor latent:** a 4th `CONST_COUNT >= 511` guard remains on the `pub const`
path (lean_single.sio ~5504); bump to 4095 if pub-const overflow ever bites
(E200 already at 20, so not currently a problem).

**Warning runaway (flag, separate bug):** `nested field store requires struct
base` at bundle line 119783 loops ~63M times → 126M-line output, times out any
full compile. Warning, not error, but blocks a clean end-to-end run.

---

## E001/type-mismatch cluster (44) — dominant sub-root: stale ownership API

Investigated the 33 E001 sites. The dominant sub-cluster (check.sio:10043–10552,
~6 calls + matching arity errors) is a **stale ownership API**: `check.sio`
threads ownership by value and by name, but `ownership.sio` was rewritten to be
`&! OwnContext`-mutating and **id-based**:

| check.sio (old) | ownership.sio (new) | migration |
|---|---|---|
| `c.ownership = own_enter_scope(c.ownership)` | `own_enter_scope(&! ctx, is_loop)` → unit | mechanical |
| `c.ownership = own_exit_scope(c.ownership)` | `own_exit_scope(&! ctx)` → unit | mechanical |
| `own_check_linear_at_end(c.ownership).error_count` | `own_check_linear_at_end(&! ctx) -> bool` | struct→bool |
| `own_declare_var(c.ownership, s.name, kind)` | `own_declare_var(&! ctx, kind, line) -> i64` | **name→id** |
| `own_transfer(c2.ownership, e.name)` | `own_transfer(&! ctx, var_id, line) -> bool` | **name→id** |

The last two require check.sio to track declared vars by the **id** returned from
`own_declare_var` (the new model), not by name. That's a scoped semantic refactor
of check.sio's ownership integration, NOT a mechanical fix — a half-migration
would silently break linear-type checking. **Source-only (check.sio); no
lean_single rebuild needed** — cheap to iterate against the current binary.

Other E001 sites (48476–48503 `checker_collect_hyper_alg_*` with `&!` ref args;
95805–95848 `ir_fast_summary_*` with `&!`/`&`) are per-case ref/type mismatches.

**Verdict on the tail:** the one big mechanical root (top-level `let` consts) is
done (5304→423). The remaining clusters (unknown-field, E001) are genuine per-case
Cat-D source bugs in the bundle (stale APIs, ref mismatches), each needing
understanding of the specific API/type intent. No further single-root shortcut.
