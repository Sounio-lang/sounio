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
**Status:** mechanism CONFIRMED with a minimal import-free witness; compiler root
cause OPEN. Not yet fixed.
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

## 5. Proposed fix (positive classification; safe default as backstop)

Scoped to `self-hosted/ir/lower.sio`; behaviour-preserving (bool already prints
`1`/`0` on the literal path):

1. **Classify bool as print_int kind (1)** everywhere int is classified — add a
   `lower_name_is_bool_type` companion and return 1 for a bool callee return, a
   bool field, and a bool-typed local/param ident. This matches the existing
   literal output and removes the char\* misroute for every bool shape.
2. **Propagate scalar-kind through ident copies** — when lowering `let b = a`,
   seed `b`'s local scalar-kind from `a`'s (covers the int-copy `z_cint` axis).
3. **(Defensive, optional) make the kind-0 default safe** — routing an
   unresolved scalar to `print_int` instead of `print` turns a guaranteed
   SIGSEGV into an at-worst-wrong-but-safe number. Only genuine `str`/`&str`
   (`StringLit`→3, or a positively-resolved string type) should reach the char\*
   printer. This is the architectural inversion the whole "println i64-segfault
   fix" comment series has been chasing one expression-kind at a time.

Fix 1 alone unblocks the dissertation crashers whose fault is a bool
field/return (`biomaterial_release`); fixes 1+2 close the measured witness set.

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
