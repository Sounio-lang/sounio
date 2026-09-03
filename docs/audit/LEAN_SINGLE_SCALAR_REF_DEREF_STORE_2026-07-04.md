<!-- docs:meta
topic_id: repo.docs.audit.lean-single-scalar-ref-deref-store-2026-07-04
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.lean-single-scalar-ref-deref-store-2026-07-04
-->

# lean_single forensic dispatch — `*n = v` for a mutable scalar reference silently emits no store

Date: 2026-07-04
Branch: `main` @ `c4d81097e`
Class: **codegen gap** (valid code compiles clean but the write never happens at
runtime — worse than a compile error, since nothing signals the defect)
Status: root-caused, fixed, verified (full test suite 1311/1311, zero regressions)

## Symptom

A dereference-assignment through a mutable reference to a **scalar** (not a struct, not
an array) compiles without a fatal error but writes nothing at runtime:

```sio
fn write_only(n: &!i64) with Mut { *n = 999 }
fn main() -> i64 with IO {
    var x: i64 = 5
    write_only(&!x)
    println(x)   // prints 5, not 999
    0
}
```

Reproduces even as a function's *first statement with no preceding call at all* —
unrelated to and independent from issue #601's "Bug H" (PR #619), which this was
originally found while validating. `bin/souc check` on this snippet emits only a
**non-fatal warning**:

```
warning: dereference assignment requires raw pointer binding at <main>:2
```

## Root cause

`self-hosted/compiler/lean_single.sio`'s `*PTR = expr` statement handler (two
near-identical copies, one per backend: x86-64 and aarch64) only emits the actual store
instruction when the left-hand variable's type is `VAR_TY == 11` (raw pointer):

```sio
if lvi >= 0 && VAR_TY[lvi as usize] == 11 && type_is_pointer_like(...) {
    ...
    emit_store_to_pointer_offset_x86(lslot, 0, inner_ty, inner_hash)
} else {
    tc_error(name_tok, "dereference assignment requires raw pointer binding")
}
```

For `n: &!i64`, `VAR_TY[lvi]` is `10` (the "ref" type code), so execution always fell
into the `else` branch. `tc_error` (as opposed to `tc_error_hard`) only prints a
non-fatal warning — compilation proceeds to completion, but **no store instruction is
ever emitted for the statement**. This is not a wrong value; the write does not exist in
the generated code at all.

This gap predates and is unrelated to issue #601/"Bug H" — bisected by testing the
statement as a function's *only* statement with zero preceding calls, still reproduces
identically.

**Refs (`VAR_TY == 10`) and raw pointers (`VAR_TY == 11`) share the same inner-type hash
encoding and runtime representation** — confirmed directly in the source:

```sio
fn ptr_hash_inner_ty(h: i64) -> i64 { return ref_hash_inner_ty(h) }
fn ptr_hash_inner_hash(h: i64) -> i64 { return ref_hash_inner_hash(h) }
```

`ptr_hash_*` is a pure delegate to `ref_hash_*` — there was never a second, distinct
encoding to support; the `emit_store_to_pointer_offset_{x86,a64}` codegen function is
itself fully generic (handles both scalar and aggregate targets identically regardless
of which "kind" of pointer-like value populated the variable's slot). The only missing
piece was the `VAR_TY == 11`-only gate in the statement dispatcher.

## Fix

Extend the gate to also accept a *mutable* reference (`VAR_TY == 10` **and**
`ref_hash_mut(hash) != 0`, using the codebase's existing mutability-check idiom already
used elsewhere, e.g. `self-hosted/compiler/lean_single.sio:9711`). A **shared** reference
(`&T`, `ref_hash_mut == 0`) must continue to be rejected — writing through a shared
reference is a real safety violation, not just a missing feature:

```sio
let is_raw_ptr = lvar_ty == 11 && type_is_pointer_like(lvar_ty, lvar_hash)
let is_mut_ref = lvar_ty == 10 && ref_hash_mut(lvar_hash) != 0
if lvi >= 0 && (is_raw_ptr || is_mut_ref) {
    ... emit_store_to_pointer_offset_x86(...) ...
} else {
    tc_error(name_tok, "dereference assignment requires a raw pointer or a mutable (&!T) reference binding")
}
```

Applied to both the x86-64 and aarch64 copies of the handler. Verified the safety
property is preserved: `*n = 999` for `n: &i64` (shared reference) still produces the
warning and emits no store.

## Confirmed real-world impact

`stdlib/epistemic/ode.sio` has 4 sites using exactly this pattern
(`compute_jacobian_state`, `compute_jacobian_params`, `bump_n_evals`,
`propagate_variance_gum` — lines 688, 744, 834, 1012), all taking a
`n_evals: &!i64` parameter and incrementing it via `*n_evals = *n_evals + N`. Direct
before/after test:

```sio
use epistemic::ode::*
fn main() -> i64 with IO {
    var n: i64 = 0
    bump_n_evals(&!n); bump_n_evals(&!n); bump_n_evals(&!n)
    println(n)
    0
}
```

Before this fix: prints `0` (4 warnings, one per silently-no-op'd site touched
transitively). After: prints `3`, zero warnings. This means the ODE solver's function-
evaluation counters (`ODESolution.n_rhs_evals`, populated via `rk4_step`/`rk45_step`
calling `bump_n_evals`) have been silently stuck since `bump_n_evals` was introduced in
PR #599 (the issue #580 fix) — no existing test asserts on this field's value, so this
was not caught by the regression suite. No stdlib fix needed on top of this compiler
change; the existing `bump_n_evals` helper now works correctly as originally intended.

## Verification

```bash
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
scripts/dev/souc-build-lock.sh ./bin/souc-lean-single-x86_64 self-hosted/compiler/lean_single.sio /tmp/lean_fixed.elf
bash scripts/run_sio_test_suite.sh --format junit --jobs 8
# Pass: 1311  Fail: 0  Known failures: 127  Skip: 689  Total: 2127
```

Also confirmed: shared-reference rejection preserved; raw-pointer path unaffected;
`ode.sio`'s `n_evals` counters now increment correctly (shown above). Madaros
(`self-hosted/compiler/main.sio` + module frontend, a fully separate source tree) is
untouched by this fix — not independently verified whether it shares this gap.

## Cross-references

- `docs/audit/LEAN_SINGLE_MULTIPLICATIVE_LINE_BOUNDARY_2026-07-04.md` — the "Bug H"
  parser fix (PR #619) this issue was discovered while validating; independent bugs,
  same discovery session.
- GitHub issue #620 — tracks this finding; closed by this fix.
- `stdlib/epistemic/ode.sio` `bump_n_evals` — the workaround from issue #580's fix
  (PR #599) that this compiler fix makes *actually effective* rather than merely
  "compiles without error."
