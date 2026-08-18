<!-- docs:meta
topic_id: repo.docs.audit.ffi-builtin-parameter-types-dispatch-2026-08-16
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.ffi-builtin-parameter-types-dispatch-2026-08-16
-->

# Give `ffi_` builtins real parameter types — P0-F residual dispatch

**Date:** 2026-08-16
**Engine:** Madaros v0.80.0 (default `bin/souc`), modular pipeline
**Parent:** P0-F close-out on `lane/fable-1/p0f-ffi-takeover` (`e2d20025c5` residual note; full Track A close-out in `docs/audit/MADAROS_EXTERN_C_BUILTIN_PORT_DISPATCH_2026-08-16.md`). Ancestor: `docs/audit/EXTERN_C_FFI_SILENT_NOOP_DISPATCH_2026-08-13.md`.
**Status:** OPEN — docs-only dispatch. **No `self-hosted/` edits in this tranche.** Implementation is a follow-on lane after this document is accepted.
**Owner:** unassigned (implementation lane). Dispatch author: `grok-cli2` / lane `ffi-builtin-typing-dispatch`.
**Known-failure pin:** `tests/run-pass/ffi_system_array_arg.sio` (`//@ known-failure`)

---

## Why this dispatch

P0-F made `extern "C"` genuine under default Madaros: multi-decl blocks parse, wrappers fail closed (E137 when the intrinsic is missing), and `system()` actually `fork`/`execve`/`wait4`s. The idiomatic C binding

```sounio
extern "C" { fn system(cmd: string) -> i64 }
```

works end-to-end (side-effect file created; `/bin/true` = 0 and `/bin/false` ≠ 0 anti-fabrication pair). One binding shape was deliberately left open rather than forced into the same PR:

```sounio
extern "C" { fn system(cmd: &[i8; 1024]) -> i32 }
```

That shape is the historical Track B / LEMON-bridge style (`&[i8;N]` buffer filled from a Sounio string). It compiles, type-checks, and *runs*, but the pointer that reaches the `ffi_system` emitter is empty — so `execve` never sees the intended C string.

fable-1 proved the scope carefully. This dispatch preserves that precision: it is **not** a regression of the P0-F close, and **not** a general ABI bug. The same argument shape works when the callee is typed; string (scalar-pointer) arguments already work through the `ffi_` path. The defect is specific to **untyped-builtin lowering**: `ffi_*` names are bound as signatureless runtime builtins, so an aggregate-by-reference argument has nothing to lower against.

---

## Defect

### Observable symptom

`tests/run-pass/ffi_system_array_arg.sio` (checked in as known-failure on the P0-F branch):

```sounio
extern "C" {
    fn system(cmd: &[i8; 1024]) -> i32
}

fn main() -> i32 with IO, Mut, Panic, Div {
    var cmd: [i8; 1024] = [0; 1024]
    // fill cmd from "/usr/bin/touch /tmp/sounio_ffi_system_array_probe"
    let rc = system(&cmd)
    // expected: rc == 0 and the side-effect file exists
    // observed: empty pointer reaches ffi_system → no real exec of the command
}
```

The user-facing wrapper keeps the declared signature (`cmd: &[i8; 1024]`). The wrapper body forwards to `ffi_system(...)`. At the **inner** call — the one resolved through the runtime-builtin table — the aggregate reference is not lowered to the heap/object base pointer the emitter expects in `rdi`. The `ffi_system` body still runs (fork/execve/wait4); it just receives a null/empty `char*`.

### Mechanism (why "signatureless")

P0-F's Track A route rewrites every `extern "C"` declaration into an ordinary Sounio wrapper whose body calls `ffi_<name>`. Resolution of that name is **not** a typed function definition:

| Layer | What happens today for `ffi_system` |
|---|---|
| **Checker** (`checker_collect_runtime_builtins_inplace`) | `checker_bind_import_unknown_inplace(c, make_name("ffi_system"))` — name exists so the wrapper body does not E137; **no parameter types, no return type, no arity** |
| **IR ensure-builtins** (`ir_module_ensure_builtin_call_targets`) | Matches `IrCall` by callee name → empty stub fn (`instr_count == 0`) rebound as the call target |
| **Native registry** (`native_v2_builtin_id_for_name` / `native_v2_emit_builtin_by_id_into`) | Name → id 33 → hand-coded emitter that assumes **SysV `rdi` already holds a NUL-terminated `char*`** |

Nothing in that chain carries "parameter 0 is `&[i8; N]` (or `string` / `*i8`)". For a **scalar pointer-like** argument (`string`), the ordinary call-arg path already places a usable address in `rdi`, so the emitter's assumption holds. For a **reference to a fixed array** (`&[i8; N]`), the typed path would materialise "address of aggregate / object base"; the untyped builtin path has no parameter type against which to apply that rule, and the forwarded value collapses to an empty pointer.

This is the same class of failure the emitter comments already document for other builtins that *do* special-case aggregate handles (e.g. `str_from_bytes`: `rdi` is a **handle**, not a raw pointer — that knowledge lives only in the hand-coded emitter, not in a type). `ffi_system` cannot special-case what it never receives.

### What this is *not*

| Claim | Status | Evidence |
|---|---|---|
| Regression of P0-F string `system()` | **False** | `tests/run-pass/ffi_system_exec.sio` green under default Madaros after `e2d20025c5`; anti-fabrication pair holds |
| General SysV / aggregate-ref ABI bug | **False** | Control: same `&[i8;1024]` shape through a typed wrapper → typed callee works (below) |
| Parser multi-decl / silent-noop residual | **False** | Closed by `7a871288ec` + `637dbf751c` + `e2d20025c5`; this residual is a *typed* path that still mis-lowers one arg class |
| "All `ffi_*` pointer args broken" | **False** | String binding works; only reference-to-aggregate is pinned |

---

## Scope boundary and controls

### In scope

1. **Call shape:** `extern "C"` wrapper → `ffi_<name>(arg)` where `arg` has type `&[T; N]` (reference to fixed array / aggregate).
2. **Primary receipt:** promote `tests/run-pass/ffi_system_array_arg.sio` from known-failure to run-pass under **default** Madaros (side-effect file must exist; fabricated-0 no-op must not pass).
3. **Root cause class:** missing parameter types on `ffi_*` runtime builtins (checker bind + whatever lowering consults for call-arg materialisation).
4. **Fix direction:** give `ffi_` builtins real parameter types (and, as needed, enough return/arity info that call-arg lowering can choose "pass pointer to aggregate" vs "pass scalar" vs "pass handle").

### Explicitly out of scope

| Item | Why |
|---|---|
| Re-opening P0-F silent-noop / multi-decl parser work | Already closed; do not re-litigate |
| Dynamic linking / `extern_relocs` PLT path | Rejected in Track A with evidence; still rejected |
| Gen-seed large-aggregate-in-global miscompile | Separate owed forensic (fable-1); repros share the `p0f_repros/` directory only by chronology |
| Full C ABI for arbitrary structs by value | Not required for this residual; `&[i8;N]` is by-ref |
| MLI implementation / MIR port | Shares the *diagnosis*, does not implement this fix (see § Connection to MLI) |
| Changing the idiomatic public binding away from `string` | `string` remains correct for `const char*`; this dispatch restores the *also-valid* array-ref binding |

### Controls (committed under `docs/audit/p0f_repros/`)

Proven by fable-1 on `lane/fable-1/p0f-ffi-takeover` (`e2d20025c5`). Paths:

| Control | Path | What it proves |
|---|---|---|
| **Typed ref-array forward works** | `docs/audit/p0f_repros/refarray_forward_typed_works.sio` | `fn outer(cmd: &[i8;1024]) -> i64 { inner(cmd) }` with `inner` also typed `&[i8;1024]` correctly delivers the buffer (reads byte 0 = 65). **Same argument shape, typed callee → OK.** |
| **String through `ffi_` works** | `docs/audit/p0f_repros/system_string_binding_works.sio` | `extern "C" { fn system(cmd: string) -> i64 }` + `system("/usr/bin/touch …")` exercises the real `ffi_system` emitter with a scalar string pointer. **Same `ffi_` path, non-aggregate arg → OK.** |
| **Product receipt (known-fail)** | `tests/run-pass/ffi_system_array_arg.sio` | The failing conjunction: aggregate-ref + `ffi_` builtin. |
| **Product receipt (pass)** | `tests/run-pass/ffi_system_exec.sio` | Canonical green: string + `ffi_system`. |

Minimal shapes (for implementers who do not want to open the committed files):

```sounio
// Control A — typed path (must keep passing)
fn inner(p: &[i8; 1024]) -> i64 { (*p)[0] as i64 }
fn outer(cmd: &[i8; 1024]) -> i64 { inner(cmd) }
// outer(&buf) with buf[0] = 65 → 65

// Control B — string via ffi_ (must keep passing)
extern "C" { fn system(cmd: string) -> i64 }
// system("/usr/bin/touch …") creates the file
```

Any proposed fix that breaks Control A or Control B is out of bounds for this residual.

---

## Why the untyped path fails where the typed path succeeds

### Typed path (Control A)

1. Callee parameters are real `ParamList` entries with `TypeExpr` (`&[i8; N]`).
2. Lowering / native call prep sees "parameter is a reference to aggregate" and materialises the address the callee expects (object base / stack address of the array), not a bare handle index and not a zeroed reg.
3. Forwarding `outer → inner` by bare ident works because both sides share that typed rule.

### Untyped `ffi_` path (known-failure)

1. Wrapper is typed at the **user** boundary (`system(cmd: &[i8; N])`) — that is why check succeeds.
2. Wrapper body calls `ffi_system(cmd)`. `ffi_system` is only an **unknown import name** in the checker; the empty stub created for the builtin has **no params**.
3. Call-arg lowering therefore cannot ask "what does the callee declare?" It falls through a default that is adequate for scalars and for `string` (already a pointer-sized value) but **not** for reference-to-aggregate.
4. Emitter `emit_builtin_ffi_system_into` correctly treats `rdi` as `char*` (holds `cmd` across the child path in `rdx`, builds argv, `execve`). Garbage-in / empty-in is not an emitter bug.

```text
  source:  system(&cmd)          // cmd: [i8;1024]
      │
      ▼
  wrapper(fn system, typed &[i8;N])     ← types present
      │  body: ffi_system(cmd)
      ▼
  IrCall  callee=ffi_system             ← name only
      │  ensure_builtin → empty stub
      ▼
  call-arg lower: no param types        ← FAILURE LOCUS
      │  aggregate-ref not materialised as base pointer
      ▼
  emit_builtin_ffi_system: rdi = empty  ← emitter honest, input wrong
```

**One-line root cause:** aggregate-by-reference argument lowering is *type-directed*; `ffi_*` builtins currently have no types.

---

## Proposed design — type the `ffi_` builtins

### Goal

Make the **inner** `ffi_*` call as type-directed as an ordinary typed callee for the argument classes P0-F actually needs, without building a general C ABI layer.

### Design options

#### Option 1 — Typed synthetic `FnDef`s for each `ffi_*` (recommended)

When the checker (or a thin post-parse bind pass) installs runtime builtins, bind each `ffi_*` as a **real function symbol** with an explicit parameter list and return type, not via `checker_bind_import_unknown_inplace`.

Concrete for the residual:

| Builtin | Proposed signature (Sounio-level) | Notes |
|---|---|---|
| `ffi_system` | `(cmd: string) -> i64` **or** `(cmd: &[i8; N]) -> i64` | See arity/overloading note below |
| `ffi_getpid` / `ffi_getppid` | `() -> i32` | Zero-arg; typing is cheap consistency |
| `ffi_exit` | `(code: i32) -> !` / `-> i32` | Scalar; already works untyped |
| `ffi_abort` | `() -> !` | |
| `ffi_malloc` | `(size: i64) -> i64` | |
| `ffi_free` | `(ptr: i64) -> ()` | |

**Overloading / dual shape for `system`:** C has one type (`const char *`). Sounio users write either `string` or `&[i8;N]`. Two workable policies:

1. **Single canonical inner type `string` (or `*i8`), convert at the wrapper.** Prefer if the parser wrapper can insert an explicit `str_from_bytes`-style conversion when the user declared `&[i8;N]`. Keeps one emitter ABI (`rdi = char*`).
2. **Two inner names** (`ffi_system` for string, `ffi_system_buf` for array-ref) selected by the wrapper rewrite from the declared param type. Slightly more registry surface; clearest lowering.

Recommend **(1)** if conversion is local and cheap; **(2)** if inserting conversion in the parser wrapper is riskier than a second builtin id. Implementer measures; this dispatch does not freeze the choice beyond "the known-failure must turn green without breaking the string control."

**Where to attach types (implementation locus, ordered):**

1. **Checker bind** — replace unknown bind for `ffi_*` with a typed declaration table (name → param types + return). This is the minimum that makes type-directed call-arg lowering *possible*.
2. **IR stub / ensure_builtin** — empty stubs should carry param count and "param *i* is ref-to-aggregate / string / i64" flags (or full type ids) so native call prep does not re-invent checker knowledge.
3. **Native call-arg path** — when callee is a builtin stub, consult those param types before placing args in SysV regs. For ref-to-aggregate: emit the same address materialisation used for typed callees (Control A path).

Do **not** special-case only `ffi_system` in the emitter by reading the *caller's* local type ad hoc if a typed-param table is achievable — that recreates the signatureless pattern under another name.

#### Option 2 — Emitter-only special case (rejected as primary)

Teach `emit_builtin_ffi_system_into` (or the call site just before it) to detect that the arg came from an array local and resolve the handle/base. Rejected as the *primary* design because:

- it does not generalise to the next `extern "C"` that takes `*mut T` / `&[T;N]`;
- it leaves the checker/IR still signatureless;
- it fights the P0-F discipline that made missing intrinsics fail at **check** time.

Emitter knowledge of "rdi is char*" stays; **how rdi is filled** must be type-directed above the emitter.

#### Option 3 — Ban `&[i8;N]` in extern wrappers (rejected)

Force all `system` bindings to `string` and delete the known-failure. Rejected: Track B / LEMON-style code and the deliberate coverage pin exist so a future fix has a live receipt. Banning papers over erasure.

### Minimal implementation sketch (for the executing lane)

1. Add a small table next to `checker_collect_runtime_builtins_inplace`:

   ```text
   FfiBuiltinSig { name, params: [TypeExpr], ret: TypeExpr }
   ```

   Seed with the P0-F allowlist (`ffi_getpid` … `ffi_system`). Prefer structured types over comments.

2. Bind those symbols so call checking sees arity and argument types (mismatch → E001, not silent bad pointer).

3. Thread param-kind into the empty builtin stub (or skip empty stubs and attach a typed `IrFunction` skeleton with `instr_count == 0` still recognised by `native_v2_builtin_id_for_func_ref`).

4. In call-arg lowering, if callee param *i* is ref-to-array/aggregate, materialise base pointer as for typed callees.

5. Flip `//@ known-failure` on `ffi_system_array_arg.sio`; keep Controls A/B green; re-run `ffi_system_exec.sio` and `make madaros-full-gate` (or the lane's scoped subset + full gate before merge).

### Acceptance gate

| # | Gate | Pass criterion |
|---|---|---|
| 1 | `tests/run-pass/ffi_system_array_arg.sio` | Remove `//@ known-failure`; default `bin/souc run` creates `/tmp/sounio_ffi_system_array_probe` (or the path the test uses) and prints PASS |
| 2 | Control A | `docs/audit/p0f_repros/refarray_forward_typed_works.sio` still PASS |
| 3 | Control B / product string path | `docs/audit/p0f_repros/system_string_binding_works.sio` and `tests/run-pass/ffi_system_exec.sio` still PASS (anti-fabrication pair intact) |
| 4 | No silent no-op regression | Unregistered `extern "C" { fn nosuch(...) }` still E137 at check (fail-closed) |
| 5 | Suite honesty | Zero *new* attributable hard-fail vs parent; known-failure count drops by the one promoted test |

### Non-goals for the first landing PR

- Typing every historical runtime builtin (`print`, `sqrt`, …) — only the `ffi_*` allowlist is required for this residual; broader cleanup is welcome but not blocking.
- Full C struct-by-value / sret for extern returns.
- Changing `emit_builtin_ffi_system_into` syscall sequence (already verified).

---

## Connection to MLI design (diagnosis only — MLI does not fix this)

This residual is an instance of the **erasure / "type was a comment"** problem that
[`docs/architecture/MLI_DESIGN.md`](../architecture/MLI_DESIGN.md) exists to stop
**one layer lower** (machine-level IR). The connection is real but limited:

| MLI claim (read carefully) | How it bears here |
|---|---|
| §4.1 — **"Kinds are part of the type, not comments."** | `ffi_*` names today are comments-with-a-name: the checker knows they exist; call-arg lowering does not know what they take. Same structural failure mode: behaviour depends on lore in emitters rather than on typed structure. |
| Preflight **D2** kind-model exclusions (kinds frozen in S1; no silent aliasing of distinct machine meanings) | D2's force is "if it affects lowering, it must be in the type." Parameter types on builtins are the *frontend/mid-end* analogue of kinds on MLI operands. |
| MLI non-goal / risk: renaming a layer without a kind model fakes completion | Shipping more `ffi_*` emitters without signatures would fake "extern C complete" the same way. |

**What this connection is not:**

- MLI does **not** implement or schedule this fix. This residual is checker / IR-stub / native call-arg work on the **existing** Madaros path.
- Closing this residual is **not** an MLI milestone and must not be claimed as S1–S3 progress.
- Epistemic / Cayley–Dickson kinds are not implicated; the parallel is the *discipline* (types drive lowering), not the domain.

A reader should leave this section with: *same disease family (erasure of distinctions lowering needs), different organ (builtin signatures vs MLI operand kinds), different surgery.*

---

## Blocker contract (for the executing lane)

| Field | Value |
|---|---|
| **Blocker-ID** | `BLK-20260816-ffi-builtin-parameter-types` |
| **Severity** | P1 residual of closed P0-F (not reopening active_p0 F; does not block WS-C PR1 by founder discipline, but blocks honest "all extern C array-ref bindings work" claims) |
| **Class** | compiler / native-call-arg / type-directed lowering |
| **Evidence level** | E2 — known-failure pin + two positive controls + mechanism traced through checker unknown-bind → empty stub → emitter |
| **Owner** | unassigned (claim before `self-hosted/` edits) |
| **Worktree** | dedicated lane worktree; do not write on shared control checkout |
| **Acceptance gate** | table above |
| **Next action** | Assign implementer; Option 1 typed synthetic signatures; measure Option 1.1 vs 1.2 for dual `system` shapes; promote known-failure |

---

## References (repo-local)

- `docs/audit/EXTERN_C_FFI_SILENT_NOOP_DISPATCH_2026-08-13.md` — original silent-noop parent
- `docs/audit/MADAROS_EXTERN_C_BUILTIN_PORT_DISPATCH_2026-08-16.md` — Track A + fable-1 close-out residual paragraph
- `docs/audit/p0f_repros/refarray_forward_typed_works.sio` — Control A
- `docs/audit/p0f_repros/system_string_binding_works.sio` — Control B
- `tests/run-pass/ffi_system_array_arg.sio` — known-failure receipt
- `tests/run-pass/ffi_system_exec.sio` — green string receipt
- `docs/architecture/MLI_DESIGN.md` §4.1, D2 note — shared diagnosis only
- `.claude/attention_p0.v1.json` slot F residual text (fable-1 wording preserved)

---

## Document control

| Date | Change |
|---|---|
| 2026-08-16 | Initial dispatch: residual scope, controls, untyped-vs-typed analysis, Option-1 design, MLI diagnosis link (no overclaim), acceptance gate. Docs only. |
