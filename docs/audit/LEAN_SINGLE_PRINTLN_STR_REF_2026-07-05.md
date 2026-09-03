<!-- docs:meta
topic_id: repo.docs.audit.lean-single-println-str-ref-2026-07-05
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.lean-single-println-str-ref-2026-07-05
-->

# lean_single forensic dispatch — `print`/`println` don't unwrap `&str`

Date: 2026-07-05
Branch: `main` (post-PR #634, issue #632)
Class: **type-dispatch gap** (a value's static type wasn't unwrapped before
choosing how to print it) — closes issue #633
Status: root-caused, fixed, verified (full test suite 1314 pass / 0 fail /
124 known failures / 689 skip / 2127 total — exact match to the current
baseline, zero regressions)

## Symptom

`println(s)` (and `print(s)`) for an `s: &str` value prints a raw pointer
value instead of the string's contents. A bare `string`-typed value prints
correctly.

```sio
fn takes_str(s: &str) -> i64 with IO {
    println(s)   // prints e.g. "4198732" instead of "hello"
    0
}
fn main() -> i64 with IO { takes_str("hello") }
```

## Root cause

`&str` and bare `string` are ABI-identical — both a bare byte pointer,
already established and relied upon by issue #601's Bug E
(`docs/audit/LEAN_SINGLE_LITERAL_REF_ARG_2026-07-05.md`), which unified them
for call-argument type-compatibility and `==`/`!=` comparisons. `print()`'s
and `println()`'s own runtime-type dispatch was never updated to match:

```sio
if EXPR_IS_F64 == 1 || EXPR_TY == 2 {
    emit_print_f64()
} else if EXPR_TY == 3 {
    emit_print_cstr()
    emit_print_nl()
} else {
    emit_print_int()
}
```

`EXPR_TY == 3` is the bare `string` type. A `&str` value has `EXPR_TY == 10`
(shared reference) with `ref_hash_inner_ty(EXPR_TY_HASH) == 3` — a case this
dispatch never checked — so it fell into the final `else`, and the pointer
value already sitting in `rax` was printed as a plain integer via
`emit_print_int()`.

This exact same gap exists in all four call sites that duplicate this
dispatch: `print()` and `println()`, each with an x86-64 and an aarch64
version.

## Fix

Since `&str` and `string` are ABI-identical, the pointer in `rax` at the
point of dispatch is already exactly what `emit_print_cstr()` expects —
no codegen change is needed, only the type-check gate:

```sio
} else if EXPR_TY == 3 || (EXPR_TY == 10 && ref_hash_inner_ty(EXPR_TY_HASH) == 3) {
    emit_print_cstr()
    ...
```

Applied identically to all four sites (`print`/`println` × x86-64/aarch64).

## Verification

```bash
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
scripts/dev/souc-build-lock.sh ./bin/souc-lean-single-x86_64 self-hosted/compiler/lean_single.sio /tmp/lean_fixed.elf
bash scripts/run_sio_test_suite.sh --format junit --jobs 8
# Pass: 1314  Fail: 0  Known failures: 124  Skip: 689  Total: 2127
```

Exact match to the current baseline — zero regressions.

Directly confirmed by output:
- Issue #633's exact repro: `println(s)` for `s: &str` now prints `hello`
  (was a raw pointer value).
- `print(s)` for the same `&str` parameter: also fixed (same gap, same fix).
- Regression check — all pre-existing print paths unaffected: string
  literal (`println("literal string")`), a bound `string` local
  (`println(s)` where `s: string`), `i64` (`println(42)`), and `f64`
  (`println(3.14)`) all print correctly, unchanged.

## Cross-references

- GitHub issue #633 — closed by this fix.
- `docs/audit/LEAN_SINGLE_LITERAL_REF_ARG_2026-07-05.md` — issue #601's
  Bug E, which established `&str`/`string` ABI-identity and unified them for
  call arguments and comparisons; this dispatch is the same unification
  applied to the one remaining place (`print`/`println`'s runtime dispatch)
  that Bug E's own writeup explicitly flagged as a known, not-yet-fixed
  follow-up.
