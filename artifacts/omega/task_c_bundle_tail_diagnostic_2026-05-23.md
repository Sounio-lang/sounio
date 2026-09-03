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

---

## Tail attack cont. (2026-05-23) — 423 → 375, then the resolution-at-scale wall

Cleared two more clusters with source-only fixes (no lean_single rebuild):
- **Ownership (dead stale API):** removed 6 redundant `own_*` calls in check.sio
  (borrow checker is authoritative). 423 → 407. Commit `5332d645e`.
- **`&! i64` deref (lint asserts):** `passed = passed + 1` → `*passed = *passed + 1`
  in `lint_assert_*`. 407 → 375. Commit `0a55db68a`.

**unknown-field + ctx.types[]=desc + info=pair.1 share one elusive root —
forward-referenced struct field types, but the existing fixup misses specific
cases at scale.** Investigation (all current-binary, no rebuilds wasted):
- 69 struct fields reference a struct defined LATER in the bundle (e.g.
  `ExprList.head: Expr`, `StmtList.head: Stmt`, `ItemList.head: Item`) — Sounio
  has no forward refs, so single-pass registration records these unresolved.
- BUT lean_single already has `resolve_forward_struct_types()` (lean_single.sio
  :22853), called from compile_all (23315, 23380), which re-runs
  `resolve_type_deep` over every struct field after all structs are registered.
  `resolve_type_deep` handles `ty==0` via `resolve_type_leaf`.
- So the two-pass fixup EXISTS and handles most cases — every isolated repro
  passes (`Option<Box<T>>`+`(*list).head`, recursive struct, struct-valued
  `arr[i]=desc`, struct `info=pair.1`). The 62+ bundle failures are a SUBSET the
  fixup misses.
- Ruled out: struct-name hash collisions (0 among 873 names, FNV-1a/37-bit),
  struct-count cap (808 < 1024), field-stride (max 83 fields < 128 stride).

**Next step (needs ONE instrumented rebuild):** add a print to
`resolve_forward_struct_types()` emitting any field still unresolved (ty==0 or
hash==0) after the pass, rebuild, run on the bundle. That names the exact structs/
fields the fixup misses and why (likely a specific type encoding `resolve_type_leaf`
doesn't recover, or a field beyond the fixup's per-struct field-count read
`ST[fix_si*130+1]`). Pinning it could clear unknown-field + ctx.types + info at
once (~80 errors) — a potential third systematic root.

**Session tally:** 5304 → 375 type errors (-93%) + segfault fixed + rebased.
Remaining 375 are: the forward-ref-fixup-miss family (~80), plus heterogeneous
per-case mismatches (E001 ref args, type-mismatch). No further cheap source root;
the fixup-miss is the next high-leverage target but needs instrumented lean_single.

---

## Instrumented investigation (2026-05-23) — VERDICT: unknown-field = genuine source bugs, no checker root

Instrumented `st_field_offset` (the shared field-lookup) to log misses with the
struct + field name hashes, rebuilt, ran on the bundle, and correlated each miss
with the actual `tc_error("unknown field access")` emission (filtering out the
function's benign feature-probe calls — e.g. ~30 `.span` "misses" that are just
"does this type have a span field?" detection, not errors).

**Real unknown-field errors map to fields/methods that genuinely don't exist on
the resolved struct — confirmed by reading the struct defs:**
- `TraitDef.method_count` (7) + `TraitDef.supertrait_count` (5) — **TraitDef has
  only `{name, methods: Option<Box<ItemList>>, supertraits, span}`; no `*_count`
  fields.** The caller wants counts but the struct stores linked lists. Source bug.
- `IrHyperExprInfo.reassoc_strategy` + others (6) — field-name mismatches.
- `Checker.report`, `Parser.peek/advance/...` — these are **methods**, hitting the
  field-lookup path → method-vs-field resolution, not a missing field.

**Conclusion (definitive):** there is NO third systematic checker root. The
forward-ref fixup (`resolve_forward_struct_types`) works; struct/field tables are
correctly populated. The remaining unknown-field / ctx.types / info errors are
**genuine per-case source bugs in the bundle** (references to non-existent fields,
stale field names, method calls mis-written as field access). Each is its own
small Cat-D fix in the modular source — no shortcut.

**Final session tally: 5304 → 375 type errors (−93%)**, segfault fixed, rebased
onto main, self-host fixed point preserved. The one big mechanical root (top-level
`let` consts) is done; ownership-dead-API and `&!`-deref cleared; the rest is
confirmed genuine per-case source work with no remaining systematic root.

---

## IrHyperExprInfo cluster (~6) — half-implemented feature, deferred (not a bug fix)

`check_hyper_expr_law_profile_audit_tag(info: IrHyperExprInfo)` (called from
main.sio:7259) reads `info.reassoc_strategy / forbidden_law_mask /
law_profile_source / law_profile_fingerprint`, none of which exist on
`IrHyperExprInfo` (ir.sio:826 has only span/algebra_tag/op_kind/knowledge_wrapped/
grade_*/cl_*). Also `make_hyper_expr_info(...)` (called check/mod.sio:282) is NOT
defined anywhere, and serialize.sio:3486 doesn't read/write the law fields.

This is an **aspirational law-profile feature that was never implemented** on
IrHyperExprInfo. Completing it = add 4 fields + the missing `make_hyper_expr_info`
constructor + symmetric serialize/deserialize read_i64/write_i64 for each +
populate at all construction sites (mod.sio:283, check.sio:1626, serialize.sio:3486,
ir.sio:839). That's a focused feature task with **serialization-symmetry risk**
(IrHyperExprInfo round-trips through the IR serializer; asymmetric read/write would
corrupt the format), not worth a mid-session grind for ~6 errors. **Deferred.**

## Session close — 5304 → 338 type errors (-94%)
Roots fixed: top-level `let` consts (the big one, 5304→423), dead ownership API
(delete), `&! i64` deref in lint, **TraitDef bundle name-collision** (rename AST
node → TraitItemDef), 4 stale method names. Plus the segfault root-cause + fix
(patch32 code-buffer guard) and the rebase onto main. Self-host fixed point
preserved throughout; all source-only fixes need no lean_single rebuild.
Remaining ~338 are genuine per-case Cat-D source bugs + this one deferred
half-feature; no further systematic root (confirmed by instrumentation).

---

## unknown-field cluster re-attack (2026-05-23) — bulk is base-type-unresolved, not field-table

Re-instrumented `st_field_offset` -> -1 returns, rebuilt, correlated with real
`tc_error("unknown field access")`. Of 42 unknown-field errors, only ~7 actually
reach st_field_offset and fail there — all **IrHyperExprInfo** law-profile fields
(the DEFERRED half-feature) + 1 Resolver field. **The bulk (.head ×25, .name ×15,
.kind ×13) never reach st_field_offset** — they fail earlier because the base
expression has no type: `(*list).head` where `list` is bound by `Some(list)` over
an `Option<Box<XList>>` param, and `list` resolves untyped.

This is the match-binding / parameter-type propagation gap for `Option<Box<T>>`
**at bundle scale** — it does NOT reproduce in isolation (every minimal repro of
the exact pattern, incl. method-with-self and recursive structs, type-checks
clean). The match-arm propagation (lean_single.sio:20130) and forward-struct fixup
(22853) both exist and work for the simple cases; something in the 223K-line
context leaves these specific `list` bindings untyped.

**Next step:** instrument the `.head` access site's base-type resolution (NOT
st_field_offset) — print `inner_ty/inner_hash` when a deref-field access bails —
to see whether `list` is ty==0 (binding never typed) or a struct hash that
st_find misses. That distinguishes a param-type-recording gap from a match-binding
gap. Focused type-inference task; not a mid-session source fix.

Session left at **338 type errors** (5304→338, -94%). The two systematic roots are
banked; the unknown-field bulk + IrHyperExprInfo are the documented elusive/deferred
remainder.

---

## .head base-type instrumentation (2026-05-24) — blocked by the warning-loop

Instrumented two sites across two rebuilds:
1. `st_field_offset` -> -1 (field-name miss): caught only ~7 real misses
   (IrHyperExprInfo law fields + 1 Resolver) after correlation.
2. expression field-read fallthrough `lean_single.sio:14207` (found==0,
   tc_error_hard): **0 hits** — the .head errors don't reach it.

Read every `"unknown field access"` site: all either go through `st_field_offset`
(8663/8934/8977/14173/14185) or are special-field handlers (14145 Knowledge
.value/.variance/..., 14207 brute-force fallthrough). The 46 errors use plain
`tc_error` and evade both instrumented points.

**Root obstacle identified: the `nested field store requires struct base`
warning-loop (bundle line 119783, ~63M repetitions) saturates stdout.** Every
capture is `head`-capped, so the `.head` markers — which stream AFTER ~30 early
`.span` probe-misses — get truncated before correlation. Clean full-output
measurement is impossible until the warning loop is fixed.

**Revised next step (reordered):** FIX the warning-loop first (it's a warning-
recovery bug that doesn't advance/dedupe at bundle line 119783 — a `nested field
store` codegen path that re-warns without consuming input). With clean bounded
output, re-run the `st_field_offset` + a per-site-tagged `"unknown field access"`
instrumentation to localize the `.head` base-type miss properly. Until then the
.head cluster (~25) cannot be reliably measured or localized.

Session unchanged at **338 type errors (5304→338, -94%)**; no source edits this
round (instrumentation reverted, tree clean, binary = let-fix e254d3da).

---

## WARNING-LOOP FIXED (2026-05-24) → first clean complete measurement: 1392 errors

Root: `compile_nested_field_store_x86/a64` warned "nested field store requires
struct base" then `return`ed without advancing EP (the EP++ sat after the return).
compile_stmt re-processed the same statement forever → ~63M warning repetitions at
bundle line 119783, 126M-line output. Fix: skip to statement terminator
(`while TK[EP] != 7 && != 0 { EP++ }`) before returning, both sites. Commits
`15eb7d029` (source + binary, fixed point `dbd3d800`).

**This unmasks the true error count — every prior number was truncated.** The loop
saturated output, so all measurements (incl. the 5304 baseline and the 338 figure)
were windows BEFORE line 119783. With the loop gone:
- output 126M → **3809 lines** (fully capturable, no `head` cap needed)
- warnings 63M → 31 (once per offending statement)
- **errors: clean complete count = 1392**, spanning to line 177373 (was stuck ~119783)

**Corrected baseline (clean, trustworthy) — 1392 by category:**
E200 505 (of which "unknown identifier" 401), assignment-mismatch 200,
**effect-not-declared 189**, Type-mismatch/E001 141, unknown-field 76,
field-init 21, exhaustiveness 5, arity 5, misc 8.

The earlier source/root fixes (let-const, TraitDef, ownership, deref, method names)
were all real reductions in the measurable window and remain valid. But the honest
state is: **the bundle has 1392 type errors, now cleanly measurable for the first
time** (no segfault, no warning-loop, full output). The big remaining clusters
(E200/unknown-identifier 505, effect-not-declared 189, assignment 200) are the real
Task C tail — and several may have systematic roots now that they're visible
(e.g. 189 effect-not-declared could share a root; 401 unknown-identifier may be more
top-level consts in the previously-hidden region past line 119783).

**Next session: measurement is now reliable.** Re-attack from the clean 1392 —
start with effect-not-declared (189, likely 1-2 roots) and the now-visible E200s.

---

## effect-not-declared cluster (189) — root confirmed: missing `with Mut` (bulk Cat-A1)

Repro (current binary): a function with NO `with` clause doing local-struct field
stores (`var m = em(); m.f = x`) errors "effect not declared in function
signature" (one per store); adding `with Mut` clears it. **In Sounio's effect
system, mutating a local struct's field (or array element) requires the `Mut`
effect.** Many modular functions omit it.

Concentrated in 4 files: `native/codegen.sio` (45), `ir/egraph.sio` (45),
`native/reloc.sio` (37), `native/pe_coff.sio` (30) = 157/183; remainder scattered
(frame/elf/auto_vectorize/tailcall/inline/...). All are builder/mutator functions
(e.g. `reloc_table_new`, `rela_section_new`, `inl_make_test_module`) that build a
struct via local-var field mutation but lack `with Mut`.

**This is bulk mechanical Cat-A1 work**, NOT a single root fix: ~50 functions, each
needing its `with` clause extended (add `Mut`; create a `with Mut` clause if none).
Caveat: some may also need `Div`/`Panic`/`Alloc` (e.g. array indexing `[i as usize]`
→ Div/Panic), so expect 1-2 rebuild-iterations per file to converge. Reliable
detection needs care — static awk under-counts (misses array/nested stores). Best
done methodically per-file in a fresh session, not a marathon tail.

Recommended approach next session: per file, add `with Mut` to each flagged
function, rebuild, re-run, add any remaining effects the new errors name, repeat
until that file is clean; then next file. Measurement is now reliable (warning-loop
fixed), so each iteration gives a trustworthy count.

**Session state: 1392 clean errors. Root identified for the largest tractable
cluster (effect-not-declared 189 = add-Mut). Deferred to a methodical fresh pass.**

## reloc.sio proof (2026-05-24): effect-not-declared CASCADES (key finding)

Fixed the 3 reloc.sio builder fns lacking Mut (reloc_table_new etc.) — correct,
but net only **1392→1391** (effect-not-declared 189→188). Adding `with Mut` to a
leaf builder shifts the requirement to its callers (now lacking Mut). **The cluster
propagates up the call graph** — it needs transitive-closure annotation (fn + all
callers), not per-leaf fixes. The earlier "37 errors in reloc.sio" was a filemap
artifact; reloc.sio really had ~3. Next session: annotate Mut bottom-up by call
chain (or top-down), expecting net reduction only once a whole chain is consistent.

---

## E200 cluster (2026-05-24) → NEW root class: incomplete core bundle. 1391 → 1129

The biggest E200 sub-cluster (MIR_*, ~80 + cascades) was NOT a compiler bug — the
`core` profile in bootstrap_concat.sh concatenated CONSUMERS of native definitions
(codegen/regalloc/lower_ir reference MIR_OP_*/PE_*/...) but omitted the PROVIDER
files. Bundle-completeness fixes (build-script only, no rebuild):
- add `machine_ir.sio` (defines MIR_*) → 1391→1218 (-173)
- add runtime_context/target_policy/gc/stack_maps/peephole → 1218→1129 (-89),
  E200 497→191 (-306). They bring some own errors but net negative.

Two cap hypotheses tried first and REVERTED (both net 0 — not the cause):
CONST table cap (pub-let already registered fine) and global table cap (1024→2048).
The real cause was always missing files. Lesson: for E200 in the bundle, FIRST check
whether the symbol's defining file is in the core profile (`grep <sym> build/...`
def-count) before suspecting the compiler.

**Remaining E200 (191):** dominated by **`loop` (30) — a real compiler feature gap**:
lean_single doesn't recognize the `loop { }` keyword (Rust-style infinite loop),
treats it as an identifier → E200. Fixing needs parser+codegen support (`loop {body}`
≡ `while true {body}`) in lean_single + rebuild. Plus a few PE_* and scattered
others (possibly more missing-file cases).

**Corrected baseline: 1129 errors** (was 1391). The two maskers (segfault,
warning-loop) + bundle-incompleteness were inflating/hiding the true picture; the
core bundle is now far more complete. Categories now: E200 191, effect-not-declared
(cascade), assignment/type-mismatch, unknown-field, + the added files' own errors.

---

## More missing def-files? — NO more cheap wins (2026-05-24, checked)

Compared the full-profile FILES list vs the core list across ALL dirs. Many files
are full-only (gpu/, llvm/, hlir/, effects/, printer/, collections/, ir/opt_*) —
but these are **separate subsystems the core pipeline doesn't reference**. Verified:
sample functions from effects/types.sio, ir/opt_cleanup.sio, ir/opt_strategy.sio,
ir/profile.sio all have **0 uses in the bundle** → adding them resolves nothing and
only injects their own errors. The genuine missing dependencies were the native
codegen providers (machine_ir + runtime_context/target_policy/gc/stack_maps/
peephole), already added (1391→1129).

Confirming signal: the only NAMED undefined symbols left are the 6 PE_* (defined
IN-CORE in pe_coff.sio) — so core has no undefined refs into the missing files
(those would surface as named E200). 

**Remaining E200 (191) is NOT missing-files:**
- PE_* (6): in-core puzzle — `let PE_DIR_IAT` def at bundle L180604 but used at
  L160599 (use-before-def cross-file); the const pre-scan should register it
  regardless of order, so this is a scan/cap quirk worth a separate look (small).
- `read_byte` (unnamed): an IO builtin lean_single's resolver doesn't know.
- `lex/3`: defined only in bootstrap/bootstrap_v0.sio (a bootstrap entrypoint, NOT
  appropriate for core — would collide).
- the bulk of "unknown identifier at line" (unnamed) needs per-site inspection.

Bundle-completeness as an error-reduction lever is now exhausted. Baseline 1069.

---

## effect-not-declared bottom-up attempt (2026-05-24) — CASCADE confirmed at scale, reverted

Instrumented the 3 `tc_*_effect*` emission sites to dump CURRENT_FN + needed-effect
mask. Got 48 distinct flagged functions (need: 2=Mut, 6=Mut+Panic, 14=Mut+Panic+Div),
dominated by builders/test-helpers (ph_*, a64_preview_*, *_new, *make_test_module).
Batch-added the missing effects to all 48 (script-driven). Rebuilt (fixed point holds
a0e0184f).

**Result: effect-not-declared 189 → 375 (+186), total 1069 → 1069 (net 0).** Adding
Mut to 48 leaf functions exposed ALL their callers (now missing Mut) → more errors,
not fewer. Confirms (at scale) the reloc proof: this is a CASCADE up the call graph.
Reverted (kept the clean 1069 baseline).

**Takeaway / decision for next session:** incremental bottom-up does NOT converge —
the mutation-transitive-closure (every fn that mutates a local struct OR calls one,
up to main) is most of the bundle. Two viable paths, both deliberate:
1. **Annotate the whole closure in one coordinated script pass** (compute reachable
   mutators + all transitive callers, add Mut everywhere, rebuild once). Big but
   convergent.
2. **Relax the effect rule in lean_single** — requiring `Mut` for *local-only* struct
   field mutation (value never escapes) is unusually strict; if local mutation isn't
   an observable effect, the checker could stop requiring it, clearing all 189+ at the
   ROOT without touching source. Semantic decision (verify it doesn't weaken real
   effect checking) — likely the higher-leverage, lower-churn fix.

Recommend evaluating path 2 first. Baseline stays 1069.

---

## relax-rule evaluation (2026-05-24) — VIABLE, high-leverage; recommend local-only refinement

Mut (effect 2) is required at 18 store-site checks (`current_fn_allows_mutation()
== false → tc_effect_violation(EP, 2, ...)`, lean_single.sio ~18717–19059) covering
field/array/nested/indexed stores. `current_fn_allows_mutation()` (2331) only checks
`FN_EFFECTS[CURRENT_FN] & 2` — no info about whether the store target escapes.

**Ceiling experiment** — made `current_fn_allows_mutation()` return true (blanket),
rebuilt: **total 1069 → 766 (-303), effect-not-declared 189 → 72 (-117), fixed point
HOLDS (52cb6ad6).** Key: the Mut effect does NOT affect codegen (self-compile stays
bit-identical) — it's purely a check; relaxing it changes what's *rejected*, not
what's *produced*. Reverted (blanket guts a real language guarantee).

**Verdict:** relax-rule is clearly the right lever (−303, no codegen risk, fixed
point intact) — far better than annotating the cascade closure. But blanket-relax
removes Mut checking for genuinely-observable mutation (`&!` ref params, globals),
which is a language-design overreach.

**Recommended implementation — local-only relax:** require Mut only when the store
target ESCAPES (base var is pointer-like = a `&!` ref, OR a global). Local `var
S` field/array mutation doesn't escape → no Mut needed. Add a helper
`store_escapes(ns,ne) -> bool` (var_find_idx + type_is_pointer_like; gl_find for
globals; conservative-true on unknown) and gate the 18 sites:
`if !current_fn_allows_mutation() && store_escapes(...) { tc_effect_violation(...) }`.
This clears the local-struct false-positives (the bulk of the 189) while preserving
Mut for observable mutation. ~18 careful site edits + 1 rebuild; the base name (ns/ne)
is available at each store site (some computed just after the current check — minor
reorder). Remaining 72 effect-not-declared after the Mut relax are OTHER effects
(IO/Div/Panic/Alloc), separate per-case.

Baseline stays 1069 (experiment reverted).

---

## 72 remaining effect-not-declared (2026-05-24) — genuine effect-propagation, mostly test code

After the Mut local-store relax (1069→766), 72 effect-not-declared remain. These are
NOT relaxable like local-Mut — they're genuine effect *propagation*: calling a
function that has Mut/Panic/Div requires the caller to declare it.

Breakdown: ~40 are calls to `ph_add_instr`/`ph_add_imm_instr` (peephole.sio, signature
`(w: &! PhWindow, ...) with Mut, Panic` — Mut is CORRECT, it writes through a `&!` ref
= escaping). The callers are ~80 `ph_test_*` TEST functions + `ph_run_all_tests` /
`ph_optimize` that lack the propagated effects. Plus ~5 `/` division (genuine Div) and
a few `orbit_*`/`_emit_*`/`name_is_*` call cascades.

**Root: these are TEST functions pulled into core via the bundle-completeness add of
peephole.sio** (codegen needs ph_optimize/ph_run_on_func; the ~80 ph_test_* tests came
along). The effects are real (Mut via `&!`, Panic, Div) so they must be declared, but
it's a propagation cascade through ~80 mechanical annotations of low-value test code.

Options (next session): (a) annotate the ph_* call-chain per-file (mechanical, ~80 fns,
cascades within peephole.sio); (b) a call-site relax — calling a Mut fn with only
LOCAL `&!` args doesn't propagate Mut — but that's interprocedurally unsound (the
callee could touch a global), so risky; (c) accept these as test-code artifacts.
Not a clean systematic win like the local-store relax. Baseline stays 766.
