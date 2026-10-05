<!-- docs:meta
topic_id: repo.docs.audit.madaros-println-bool-scalarkind-segv-2026-08-17
authority: repo_only
audience: users
last_validated: 2026-08-17
validated_by: fable-1
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-println-bool-scalarkind-segv-2026-08-17
-->

# Madaros `println(bool)` / unclassified-scalar → char\* printer SIGSEGV

**Date:** 2026-08-17
**Scope:** `bin/souc` default Madaros engine — **native run** SIGSEGV (rc=139) at
`println`/`print` of a scalar whose lowering-time scalar-kind is not positively
resolved. **Compile/typecheck pass; the generated ELF faults at runtime.**
**Status:** mechanism CONFIRMED with a minimal import-free witness. **Fixes 1
and 2 LANDED and verified from source** (commits on lane/fable-1/println-scalarkind-segv-20260817);
fix 3 deferred as a follow-up recommendation (§5). One residual (if-expr) named
in §5.

## 0a. Fix status (2026-08-17)

Fixes 1 and 2 are implemented in `self-hosted/ir/lower.sio` and **built from
source and verified control-first** (never on the prebuilt wrapper):

- **Control** (parent, no fix, source-built): `biomaterial_release` and the
  minimal witness both SIGSEGV (rc=139) — the bug is real on a from-source
  binary, not a prebuilt artefact.
- **Fix** (source-built): `biomaterial_release` runs clean (rc=0, completes
  through TEST 8); the whole §3 witness set for bool call/field/param and
  int/bool ident-copy passes. String / int / f64 / int-param **positive controls
  unchanged** (a `println("string")` still prints TEXT — the fix never reroutes
  strings to print_int). A 70-test struct/bool/print/enum/generic control-vs-fix
  sweep shows **zero differences**; three generic-struct fails
  (`generic_struct_instantiate`/`_nested` = 139, `_return_structf` = 1) are
  **pre-existing** — identical on control. Full-suite regression still pends
  CI/Slurm. Merge owner: codex-2 (compiler).
**Engine split:** Madaros only (the dissertation triage records lean_single PASS
on the same programs).
**Related:** `project_scalar_kind_param_bug_2026-07-01` (the f64-param twin of
this family); `DISSERTATION_PBPK_SUITE_TRIAGE_2026-08-16.md` (tests #6
`biomaterial_release`, #7 `rapamycin_clinical` labelled "lower_array SIGSEGV 139").

## 1. What actually crashes (the "lower_array" label is misleading)

The triage tagged these fails "lower_array … SIGSEGV 139". The string
`lower_array:` in `souc run` output is a **compiler status line**
(`lower_array: arena_reset_totals …`), **not** the crash locus. The crash is at
**runtime**, in the emitted program, on a `println`/`print` of a scalar.

Live example — `stdlib/darwin_pbpk/release/biomaterial_release.sio` (test #6),
current main `dca2775061`, built-from-node triage confirms both arms:

```
TEST 5: Higuchi + 14-comp PBPK simulation completes
  success =                         <- string literal printed OK
<SEGV>                              <- println(m_hig.success) : bool FIELD load
```

`m_hig = simulate_14_release(...)` returns a `ReleasePBPKMetrics` whose last
field is `success: bool`. `println(m_hig.success)` faults. Reading `m_hig.auc`
(an `f64` field) instead runs clean.

## 2. Minimal witness (import-free, single module, no PBPK)

```sounio
fn getb() -> bool { true }
fn main() -> i32 with IO {
    let v = getb()
    print("V=")
    println(v)          // SIGSEGV (rc=139)
    println("read ok")
    0
}
```

`check` = OK; `run` = 139. The value never prints; the process faults inside the
`println(v)` call.

## 3. Trigger boundary (measured, current main)

`fn getb() -> bool`, `fn geti() -> i64`, `fn getf() -> f64`; structs as noted.

| # | Program shape | rc | Note |
|---|---|---:|---|
| C1 | `let b = true; println(b)` | 0 | prints `1` — bool **literal** works |
| Q  | `let v = getb(); println(v)` | **139** | bool from fn return |
| T  | `println(getb())` | **139** | bool, direct call |
| U  | `fn pb(b: bool){ println(b) }; pb(true)` | **139** | bool param |
| N  | `let v = if s.ok {true} else {false}; println(v)` | **139** | bool via if-expr |
| M  | `println(s.ok)` (struct bool field) | **139** | bool field load |
| G  | `if s.ok { … }` (branch, no println) | 0 | field **load** is fine |
| V  | `let a=true; let b=a; println(b)` | **139** | **copy** of a bool literal |
| z_int | `let v = geti(); println(v)` | 0 | int from fn return — classified |
| z_cint | `let a=7; let b=a; println(b)` | **139** | **copy** of an int literal |
| z_f64 | `let v = getf(); println(v)` | 0 | f64 from fn return — classified |
| W  | `var v = true; v=false; println(v)` | 0 | mutated var path classifies |
| X  | `let v = 1>0; println(v)` | 0 | comparison → kind 1 |

Two independent gaps produce the same SIGSEGV:
- **bool is never classified** (fn return / field / param / if-expr / copy all
  fault). Only a bool *literal* binding works — and it prints as `1`/`0`.
- **ident-to-ident copy `let b = a` drops the kind for ANY scalar** (int copy
  `z_cint` faults too, not just bool).

## 4. Mechanism (root cause)

`println`/`print` pick their backend builtin at **lowering** time, in
`Lowerer::println_dispatch_name` (`self-hosted/ir/lower.sio:9526`):

```
let sk = self.expr_result_scalar_kind_ref(&(*list).head)
if sk == 1 { print_int } else if sk == 2 { print_f64 } else { print }
```

`print` is the **char\* printer** (strlen + write). Routing a scalar there makes
the backend treat the scalar *value* as a pointer: a bool `1` becomes address
`0x1`, strlen dereferences it → **SIGSEGV**. The `else` (kind 0) default is
therefore *unsafe* — "unclassified" is assumed to mean "string".

`expr_result_scalar_kind_ref` (`lower.sio:10572`) returns:
- `IntLit`→1, `FloatLit`→2, `StringLit`→3;
- `Ident`→ `lookup_local_scalar_kind` else `bss_global_scalar_kind`;
- `Call`/`MethodCall`→ 1 **iff** the callee's `return_struct_name` satisfies
  `lower_name_is_integer_type`;
- `FieldAccess`→ 1 **iff** `field_is_int_for_base_ref` / `field_is_int_by_name_simple`;
- `Index`→ float?2:1; `Binary`→ compare?1 : float?2 : …; `Cast`→ f64?2 : int?1 : 0.

`lower_name_is_integer_type` (`lower.sio:4670`) is the gate, and its own comment
is explicit: *"Does NOT include f32/f64 (kind 2), string, **bool**, or char —
… so print_int is selected instead of char\* strlen → SEGV."* So a bool return /
bool field is never kind 1 → falls to the `else` → char\* → SEGV. The bool
**literal** works only because `true`/`false` lower as `IntLit` (kind 1), which
is also why literal `println(true)` prints `1`.

The copy gap is orthogonal: `let b = a` does not record `b`'s
`lookup_local_scalar_kind` from `a`'s, so the `Ident` path returns 0 even for a
provably-int `a`.

## 5. Fix (positive classification; safe default deferred)

Scoped to `self-hosted/ir/lower.sio`; behaviour-preserving (bool already prints
`1`/`0` on the literal path).

**Fix 1 — classify bool as print_int kind (1) everywhere int is classified
(LANDED).** Added `lower_type_expr_is_bool` / `lower_name_is_bool_type` (mirror
the i64 pair) and returned kind 1 for bool at every site int is:
- `register_struct_fields` — **all six** field-registration paths mark a bool
  field with the scalar-int marker `3` (consumed only by `field_is_int_*`, inert
  for float/array/store). *Only patching one path is a trap:* the layout consumed
  at the println site comes from a different registration path, so the field case
  stays broken until all six carry the marker.
- `expr_result_scalar_kind_ref` Call/MethodCall — a bool-returning callee → 1.
- `lower_fn_params` **path-A** (`lowerer_mark_local_scalar_kind_mut`) — a bool
  param → 1. Path-A is the load-bearing param path; the `_ref` param table alone
  does not reach println dispatch.

**Fix 2 — propagate scalar-kind through ident copies (LANDED).** The let-binding
classifier had no Ident case, so `let b = a` left `b` at kind 0. Propagate `a`'s
recorded kind, guarded `!= 0`. Covers the int-copy `z_cint` axis too (not just
bool).

**Residual (NOT fixed here) — if-expr / other unclassified expression kinds.**
`let v = if c { … } else { … }; println(v)` still SIGSEGVs, and it does so for an
**int** if-expr too (`ife` = 139), so this is not a bool gap — it is the same
class as the char\*-default problem, one more expression-kind
(`expr_result_scalar_kind_ref` has no `ExprIf` case). Left for the follow-up
below rather than special-cased here.

**Fix 3 — follow-up recommendation: make the kind-0 default safe (SEPARATE PR).**
The durable closure is to invert the `println_dispatch_name` default: an
unresolved scalar should route to `print_int`, not the char\* printer. Only a
**positively-resolved** string (`StringLit`, or a type proven `str`/`&str`)
should reach `print` (strlen+write). Today the default assumes "unclassified ==
string", so every expression-kind the classifier does not yet recognise
(if-expr, and whatever is added next) is a latent SIGSEGV that must be closed one
at a time — fixes 1 and 2 are two such patches, and the residual above is the
next. Inverting the default converts that open-ended SIGSEGV surface into an
at-worst-wrong-but-safe number and makes the classifier's job "detect strings"
(a closed, small set) instead of "detect every scalar" (open-ended). It is
deliberately **not** bundled here: it is a semantics change to the default
branch and deserves its own PR and its own argument, with a positive control
proving a genuine `println(string_variable)` — not a literal — still prints text,
since that is the one case the inversion must not break.

## 6. Verification obligation before landing

`bin/souc` is prebuilt; this mechanism was measured on the checked-in Madaros
artifact (built at HEAD, zero `self-hosted/{ir,native,check}` commits since the
triage). Any fix MUST be built from source (Slurm `partition=all`, or the global
build lock) and re-run against §3's witness table **and** the dissertation suite
gate, with the positive control (a genuine `println("string")`) shown still
printing text — a regression here would silently reroute strings to `print_int`.

## 7. Non-goals

- Do not relabel this a "lower_array" bug; the lowering pass completes and emits
  a valid ELF. The defect is in the `println` builtin **selection**, realised at
  runtime.
- Do not claim the general scalar-kind classifier is complete after fix 1/2;
  §5.3 (safe default) is the durable closure and is a separate decision for the
  compiler owner (codex-2).
