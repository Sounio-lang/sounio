<!-- docs:meta
topic_id: repo.docs.audit.lean-single-literal-ref-arg-2026-07-05
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.lean-single-literal-ref-arg-2026-07-05
-->

# lean_single forensic dispatch — literals don't unify with reference-typed parameters/comparisons

Date: 2026-07-05
Branch: `main` (post-PR #627, Bug D verification)
Class: **parser gap + two type-compatibility gaps** (three mechanically
distinct defects the original issue grouped as one) — root-causes and closes
issue #601's "Bug E"
Status: root-caused, fixed, verified (full test suite 1314 pass / 0 fail /
124 known failures / 689 skip — unchanged from the current baseline, zero
regressions; no tracked test exercised these exact patterns before, since the
existing stdlib workarounds remain in place and unaffected)

## Summary — three distinct mechanisms, not one

Issue #601 catalogued "Bug E" as a single defect with three symptomatic
variants, all attributed to "a bare literal's inferred type... doesn't unify
with a `&T` reference parameter type without an intermediate named binding."
Investigation shows this is actually **three independent defects** sharing a
theme (literal ↔ reference-type friction) but living in three different
functions:

1. `&[1.0, 2.0, 3.0]` as a call argument — a genuine **parser gap**:
   `compile_borrow_primary_x86()`/its aarch64 twin unconditionally assumed
   the token after `&`/`&!` is a variable identifier and never checked for a
   literal.
2. `takes_str("hello")` where `takes_str(s: &str)` — **not** an `&`-prefix
   issue at all (the call site has no `&`!). This is a pure
   **type-compatibility gap** in `call_arg_type_compatible()`: a bare
   `string` value was never accepted where a `&str` parameter is declared,
   even though the two are ABI-identical (both a bare byte pointer).
3. `s == "hello"` where `s: &str` — same theme, different function: a
   **type-compatibility gap** in `compile_comparison()`'s operand-visibility
   helpers (`comparison_visible_ty`/`comparison_visible_hash`), which already
   unwrap `Unobserved<T>` for comparison purposes but had no equivalent rule
   for `&str` vs. bare `string`.

## Reproduction

```sio
fn takes_arr(a: &[f64; 3]) -> f64 { a[0] + a[1] + a[2] }
fn main() -> f64 { takes_arr(&[1.0, 2.0, 3.0]) }
// pre-fix: error: unknown identifier `[` + error: arity mismatch

fn takes_str(s: &str) -> i64 { 0 }
fn main2() -> i64 { takes_str("hello") }
// pre-fix: error[E001]: Type mismatch in call argument

fn eq_str(s: &str) -> bool { s == "hello" }
// pre-fix: error: comparison operands must have the same type
```

## Root causes and fixes

### 1. `&<literal>` as a call argument (parser gap)

`compile_borrow_primary_x86()` (`self-hosted/compiler/lean_single.sio`,
originally line 9470) and its aarch64 inline twin, after consuming `&`/`&!`,
unconditionally read the current token's span as a variable name:

```sio
let rns = TS[EP as usize]
let rne = TE[EP as usize]
EP = EP + 1
if TK[EP as usize] == 41 { /* &var[a..b] slice borrow */ ... }
...  // more identifier-only postfix handling
```

For `&[1.0, 2.0, 3.0]`, the token right after `&` is `[` (41), not an
identifier — `rns`/`rne` capture the `[` token's own span, and the
subsequent `var_find_idx(rns, rne)` lookup fails with "unknown identifier
`[`"; the array literal's own `(1.0, 2.0, 3.0]` tokens are left unconsumed,
producing the follow-on arity mismatch on whatever call this appeared in.

**Fix**: check `TK[EP as usize] != 3` immediately after the (unaffected,
pre-existing) `&(*ptr)` special case. Sounio's array-literal codegen already
materializes an array literal to a fresh stack slot and returns *its address*
via `lea` (the same representation a plain array variable's address-of
already uses); a string literal's codegen already returns its own address
via `lea rax, [rip+disp]`. Both are therefore already exactly the value a
reference to that literal needs — no new codegen, just: compile the literal
via `compile_primary()`/`compile_primary_a64()`, then tag the result as a
reference to whatever type it produced:

```sio
if TK[EP as usize] != 3 {
    compile_primary()
    let lit_inner_ty = EXPR_TY
    let lit_inner_hash = EXPR_TY_HASH
    EXPR_IS_F64 = 0
    EXPR_TY = 10
    EXPR_TY_HASH = ref_hash_make(lit_inner_ty, lit_inner_hash, want_mut)
    return
}
```

### 2. Bare `string` literal accepted where `&str` parameter is declared

`call_arg_type_compatible()` (line 3013) already has an "auto-ref coercion"
section for `expected_ty == 10` (shared reference) covering struct/array
field-access coercion and the `&!T`→`&T` safe downgrade, but no rule for a
bare `string`(`ty==3`) value against a `&str`(`ty==10` wrapping `ty==3`)
parameter — even though they share an identical runtime representation (a
bare byte pointer). Added:

```sio
if exp_inner_ty == 3 && actual_ty == 3 { return true }
```

inside the existing `expected_ty == 10 && ref_hash_mut(expected_hash) == 0`
block.

### 3. `&str` vs. bare `string` in `==`/`!=` comparisons

`comparison_visible_ty()`/`comparison_visible_hash()` (lines 3072/3105)
already unwrap `Unobserved<T>` before `compile_comparison()`'s type-equality
check runs, but had no equivalent unwrap for `&str` → `string`. Added the
same unwrap:

```sio
// comparison_visible_ty
if ty == 10 && ref_hash_inner_ty(ty_hash) == 3 { return 3 }
// comparison_visible_hash
if ty == 10 && ref_hash_inner_ty(ty_hash) == 3 { return ref_hash_inner_hash(ty_hash) }
```

`compile_comparison()`'s existing `left_ty == 3 && right_ty == 3` branch
(string equality via `emit_str_eq_regs_x86()`) then fires unchanged — the
actual register values for a `&str` and a bare `string` were always
identical, so no codegen change was needed, only the type-check gate.

## Discovered but out of scope: `println` does not unwrap `&str`

While verifying fix #2/#3, `println(s)` for an `s: &str` **parameter**
prints a raw pointer value instead of the string's contents — confirmed to
reproduce identically via the pre-existing, already-working workaround
(`let s: string = "hello"; takes_str(&s)`, then `println(s)` inside
`takes_str`), so this is a separate, pre-existing gap in `println`'s runtime
type dispatch, not something this fix introduces or is scoped to address.
Not yet issue-tracked.

## Verification

```bash
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
scripts/dev/souc-build-lock.sh ./bin/souc-lean-single-x86_64 self-hosted/compiler/lean_single.sio /tmp/lean_fixed.elf
bash scripts/run_sio_test_suite.sh --format junit --jobs 8
# Pass: 1314  Fail: 0  Known failures: 124  Skip: 689  Total: 2127
```

Identical to the pre-fix baseline — zero regressions. No tracked test
exercises these exact patterns (the stdlib files that motivated Bug E,
`kinetics.sio`/`ontology.sio`, already use the documented workarounds and are
unaffected either way), so this fix's value is preventing the need for those
workarounds going forward and closing a real, previously-uncaught class of
literal/reference friction, not moving the pass count.

Directly confirmed, by return value (not `println`, given the discovered gap
above):
- `takes_arr(&[1.0, 2.0, 3.0])` → `6.0` (sum of elements), was a compile
  error.
- `takes_str("hello")` → `5` (function's own return value), was `error[E001]`.
- `eq_str("hello")` → `true`; `eq_str("world")` → `false`, was a hard compile
  error for either.

Also confirmed unaffected: `&!x` passed where `&x` is expected (existing
safe-downgrade rule), bare `string == string` comparisons, `&arr` where `arr`
is a real array *variable* (not a literal) as a call argument — all still
use their original, unchanged code paths, since the new checks only fire for
a non-identifier token after `&`/`&!` (fix 1) or specifically `ty==10`
wrapping `ty==3` (fixes 2/3).

## Cross-references

- GitHub issue #601 — tracks Bug E (closed by this fix). Bugs F–G remain
  open, plus the `use ... as alias` variant
  (`docs/audit/LEAN_SINGLE_NAMED_USE_IMPORT_2026-07-05.md`) and the
  `println(&str)` gap noted above, neither yet issue-tracked.
