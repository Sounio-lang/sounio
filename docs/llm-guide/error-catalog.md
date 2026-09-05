<!-- docs:meta
topic_id: repo.docs.llm-guide.error-catalog
authority: repo_only
audience: users
last_validated: 2026-08-26
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.llm-guide.error-catalog
-->

# Sounio Error Catalog

Every error an LLM is likely to produce, with diagnosis and fix.
Organized by the mistake pattern, not the error message text.

---

## E01 — Missing effect declaration

**Symptom:** Code compiles in your head but the checker rejects it.

**Error pattern:**
```
error: function `foo` uses IO but does not declare `IO` in its effects
error: effect `Div` required but not declared
```

**Cause:** You called a function (or used `/`, `%`, `println`, array index) without declaring the
required effect in `with`.

**Fix:**
```sio
// ✗ WRONG
fn compute(a: f64, b: f64) -> f64 {
    a / b
}

// ✓ CORRECT
fn compute(a: f64, b: f64) -> f64 with Div, Panic {
    a / b
}
```

**Rule:** Effects propagate upward. If `inner` needs `Div`, then every caller of `inner` also needs `Div`.
When in doubt, give `main` the full set: `with IO, Mut, Div, Panic`.

**Effect reference:**

| What you wrote | Effect needed |
|----------------|---------------|
| `a / b` or `a % b` | `Div` |
| `arr[i]` | `Panic` |
| `assert(x)` | `Panic` |
| `x as u8` | `Panic` |
| `println(...)` | `IO` |
| `*x = v` or `self.field = v` | `Mut` |
| `var x = 0; x = 1` | `Mut` |
| `if obs > threshold` on `Unobserved<T>` | `Observe` |

---

## E02 — Semicolon in expression

**Symptom:** Function returns wrong type; unexpected `()` return.

**Error pattern:**
```
error: expected i32, found ()
type mismatch: statement returns () but function declares -> i32
```

**Cause:** Added `;` after an expression, turning it into a statement that returns `()`.

**Fix:**
```sio
// ✗ WRONG
fn get_val() -> i32 {
    let x = 5;
    x + 1;           // ← semicolon makes this return ()
}

// ✓ CORRECT
fn get_val() -> i32 {
    let x = 5
    x + 1            // no semicolons, expression is the return value
}
```

**Global rule:** No semicolons. Ever. Not on `let`, not on expressions, not anywhere.

---

## E03 — `&mut` instead of `&!`

**Symptom:** Parse error on function signature.

**Error pattern:**
```
error: unexpected token `mut`
expected type, found `mut`
```

**Cause:** Used Rust's `&mut T` instead of Sounio's `&!T`.

**Fix:**
```sio
// ✗ WRONG
fn fill(arr: &mut [i64; 8]) { ... }
fill(&mut buf)

// ✓ CORRECT
fn fill(arr: &![i64; 8]) with Mut, Panic { ... }
fill(&!buf)
```

---

## E04 — `let mut` instead of `var`

**Symptom:** Parse error, or variable treated as immutable.

**Fix:**
```sio
// ✗ WRONG
let mut counter = 0
counter = counter + 1

// ✓ CORRECT
var counter = 0
counter = counter + 1
```

---

## E05 — Rust macro syntax

**Symptom:** Parse error on `!` after function name.

**Error pattern:**
```
error: unexpected `!` after identifier
```

**Fix:**
```sio
// ✗ WRONG
println!("hello")
assert!(x == 5)
vec![1, 2, 3]

// ✓ CORRECT
println("hello")
assert(x == 5)
// No vec! — use [T; N] arrays
```

---

## E06 — Unary minus

**Symptom:** Parse error on negative literal.

**Fix:**
```sio
// ✗ WRONG
let x = -42
let y = -3.14
let z = 0 - -x   // double negative

// ✓ CORRECT
let x = 0 - 42
let y = 0.0 - 3.14
let z = x         // 0 - (0 - x) = x
```

---

## E07 — Closure literal

**Symptom:** Parse error on `|`.

**Error pattern:**
```
error: unexpected `|`
```

**Cause:** Used closure syntax `|x| expr` which is not supported.

**Fix:**
```sio
// ✗ WRONG
let doubled = map(data, |x| x * 2)
let f = |x: i64| -> i64 { x + 1 }

// ✓ CORRECT — define a named function, pass its reference
fn double(x: i64) -> i64 { x * 2 }
fn add_one(x: i64) -> i64 { x + 1 }

let doubled = map(data, double)
let f = add_one
```

---

## E08 — Generic type parameter

**Symptom:** Parse error on `<T>` in function signature.

**Fix:**
```sio
// ✗ WRONG
fn identity<T>(x: T) -> T { x }
fn process<T: Display>(x: T) { }

// ✓ CORRECT — write monomorphic functions, one per type
fn identity_i64(x: i64) -> i64 { x }
fn identity_f64(x: f64) -> f64 { x }
```

The only exception: `Knowledge<T>` is a built-in generic — use it as-is.

---

## E09 — Bit shift without `u8` operand

**Symptom:** Type error on shift expression.

**Error pattern:**
```
error: shift amount must be u8
```

**Fix:**
```sio
// ✗ WRONG
let high = byte >> 4
let low = byte & 15

// ✓ CORRECT
let high = byte >> 4u8
let low = byte & 15u8
```

---

## E10 — Array mutation doesn't propagate (interpreter)

**Symptom:** Values written to `&![T; N]` inside a function are invisible to the caller.

**Cause:** Known interpreter limitation — bare array `&!` mutation doesn't propagate.
(Native compiler is unaffected.)

**Fix — wrap in struct:**
```sio
// ✗ Broken in JIT interpreter
fn fill(arr: &![f64; 16]) with Mut, Panic {
    (*arr)[0] = 99.0    // caller doesn't see this
}

// ✓ CORRECT — wrap in struct
struct Buf16 { data: [f64; 16] }

fn fill(buf: &!Buf16) with Mut, Panic {
    buf.data[0] = 99.0  // struct field mutation works
}

var b = Buf16 { data: [0.0; 16] }
fill(&!b)
// b.data[0] is now 99.0
```

---

## E11 — Top-level `let` (global variable)

**Symptom:** Compiles with JIT, crashes or links incorrectly with native.

**Cause:** The native compiler does not support global `let` bindings (BSS limitation).

**Fix:**
```sio
// ✗ WRONG — not portable
let BUFFER_SIZE: i64 = 256
let PI: f64 = 3.14159

// ✓ CORRECT — use fn constants
fn BUFFER_SIZE() -> i64 { 256 }
fn PI() -> f64 { 3.14159265358979323846 }
```

---

## E12 — Missing explicit `self` in impl

**Symptom:** Method compiles but doesn't mutate anything, or type error.

**Cause:** Forgot `self: &Type` or `self: &!Type`.

**Fix:**
```sio
// ✗ WRONG — no self parameter
impl Counter {
    fn increment() with Mut { count = count + 1 }  // count is undefined
}

// ✓ CORRECT
impl Counter {
    fn increment(self: &!Counter) with Mut {
        self.count = self.count + 1
    }
    fn value(self: &Counter) -> i64 {
        self.count
    }
}
```

---

## E13 — Float comparison with `==`

**Symptom:** Test passes or fails non-deterministically; assertion on computed float fails.

**Cause:** `f64 == f64` is exact equality — rounding error causes mismatch.

**Fix:**
```sio
// ✗ WRONG
assert(result == 3.14159)
assert(computed == expected)

// ✓ CORRECT
use test::helpers::{check_near}
assert(check_near(result, 3.14159, 1e-5))
assert(check_near(computed, expected, 1e-9))
```

---

## E14 — Array index type

**Symptom:** Type error when indexing with `i64` or `i32`.

**Error pattern:**
```
error: array index must be usize
```

**Fix:**
```sio
// ✗ WRONG
let x = arr[i]         // i is i64

// ✓ CORRECT
let x = arr[i as usize]   // needs Panic effect
```

Or declare the loop variable as `usize` if indexing only:
```sio
var i: usize = 0
while i < n { arr[i]; i = i + 1 }
```

---

## E15 — Returning from inside `if` branch

**Symptom:** Return value is ignored; function returns wrong value.

**Fix:**
```sio
// This is correct — early return with explicit return keyword
fn clamp(x: f64, lo: f64, hi: f64) -> f64 {
    if x < lo { return lo }
    if x > hi { return hi }
    x
}

// This is also correct — if as expression
fn abs(x: f64) -> f64 {
    if x < 0.0 { 0.0 - x } else { x }
}
```

Both forms work. Use `return` for early exit, if-expression for terminal value.

---

## Quick Diagnosis Flowchart

```
Got an error?
│
├─ Parse error → Check for semicolons, &mut, let mut, |x|, #[, <T>, -n literals
│
├─ Effect error → Add missing effects to the function signature (and all callers)
│
├─ Type mismatch → Check array index type (usize), check f64 == comparison
│
├─ Mutation not working → Are you on JIT? Wrap array in struct
│
└─ Undefined variable → Top-level let? Convert to fn constant
```

---

## Machine-Readable Error Code Reference

Codes the compiler *can* emit in `error[Exxxx]:` format. Note: there is **no** `souc check --json` flag and **no** `souc explain <CODE>` subcommand in Madaros v0.80.0 (both were removed / never shipped — verify with `souc --help`). Read the per-code files under `explanations/` directly.

> **⚠️ Enforcement reality — verified 2026-07-11 against the default `bin/souc` (Madaros).** The default compiler is **more permissive** than this table implies; several listed codes do **not** currently fire under `souc check` (the "wrong" example compiles clean). Verified non-firing: **E035** (missing IO/Div/Observe effect — effects are not enforced under `check`), **E040/E041/E042/E043** (Rust `let mut` / `&mut` / `#[...]` / `ident!()` — these surface as a bare `parse error` or `check: OK`, not a coded compat error), **E201–E207** (the `ZD` capability family — unenforced), **E208/E209** (refinement predicates — `Pos`/`Prob` treated nominally; the predicate is not evaluated), **E213** (tuple-destructure arity), **E216** (recursive struct type), **E224** (unreadable/dead import — silently ignored). Wrong code numbers: an arity mismatch surfaces as **E010** (not E006); a tail/return-type mismatch as **E008** (not E218). Confirmed firing: E001, E010, E170, E171. `check` stops before codegen, so codegen/linker codes (E007, E217–E223) are not reachable via `check`. Treat this table as the code *namespace*, not a guarantee that every guard is wired. (Note: the `lean_single` seed engine is stricter and rejects some of the above — but agents use the default Madaros.)

> **⚠️ One number, two meanings — measured 2026-08-26 (#2180).** This namespace has
> **two owners**. `lean_single` (the CI oracle and the seed) and `check.sio` (the
> Madaros checker) both emit `error[E<N>]`, and for the fifteen codes below they
> emit it for **different diagnostics**. The table above documents the
> `lean_single` identity in every case; `explanations/E<N>.md` explains that one.
> If you saw the code come out of the default `bin/souc`, the right-hand column is
> what you actually hit, and the explanation file will not describe it.
>
> This happened mechanically, not carelessly: until #2180 `lean_single` printed 42
> of its diagnostics as a bare `error: <text>` with no number, so a survey taken
> from inside `check.sio` saw those numbers as free. It now prints 41 of them,
> and `scripts/ci/diagnostic_identity_gate.sh` surveys both engines, so the next
> re-allocation collides visibly instead of silently. Which identity keeps each
> number is a separate decision, per code, and is not yet made.
>
| Code | `lean_single` emits (documented above) | `check.sio` / Madaros emits |
|------|----------------------------------------|------------------------------|
| E006 | arity mismatch | conditions must be of type `bool` |
| E007 | too many local variables in function | branches have incompatible types |
| E008 | too many globals | return value does not match the declared return type |
| E036 | `Unobserved<T>` crosses the observation boundary without `with Observe` | confidence bound is not tight enough |
| E040 | Rust `let mut` — use `var` | linear value must be used |
| E041 | Rust `&mut` — use `&!` | ontology subsumption could not be verified |
| E042 | Rust attribute `#[...]` not valid in Sounio | value does not satisfy the refinement predicate |
| E043 | Rust macro `ident!(...)` not valid in Sounio | incompatible unit dimensions in conversion |
| E067 | potential confounding | causal query is non-identifiable (no valid adjustment set) |
| E080 | `Deterministic` function declares an IO effect | contest metadata is incomplete for witness projection |
| E208 | refinement type violation (integer) | ZD locus is not a well-formed sedenion pair `eIeJ` |
| E217 | invalid function body span | f128/f256 value conversion is not implemented; wide-float casts fail closed |
| E218 | tail type mismatch | f128/f256 is reserved for compiler-owned format identity |
| E219 | function pass mismatch | call to an `extern "C"` function |

> **E219 is the one exception, and it is instructive.** `lean_single` does NOT
> print its tag for `function pass mismatch`, because
> `scripts/ci/e219_engine_oracle_gate.sh` asserts the seed must not spell E219 at
> all: that oracle scores an engine SPLIT in which E219 is a Madaros judgment
> about an unimplemented `extern "C"` builtin, and it says outright that a
> cosmetic E219 on the seed "is a fail, not a close. A real seed refuse is also a
> fail — update the theorem, do not silence the gate." So the catalogue and that
> gate disagree about who owns E219. Tagging the seat forced the disagreement
> into the open; resolving it is a decision about the theorem, not an edit.
| E221 | no main | this math function is bound for typechecking but the native backend cannot emit it |

| Code | Component | Severity | Gloss | Explanation |
|------|-----------|----------|-------|-------------|
| E000 | legacy | error | Unclassified error (legacy — no code assigned) | — |
| E001 | type-checker | error | Type mismatch / linear constraint violation | [E001.md](explanations/E001.md) |
| E006 | type-checker | error | Arity mismatch — wrong number of arguments | [E006.md](explanations/E006.md) |
| E007 | codegen | error | Too many local variables in function | [E007.md](explanations/E007.md) |
| E008 | codegen | error | Too many globals | [E008.md](explanations/E008.md) |
| E035 | type-checker/effects | error | Effect not declared in function signature | [E035.md](explanations/E035.md) |
| E036 | type-checker/effects | error | Unobserved\<T\> crosses observation boundary without Observe | [E036.md](explanations/E036.md) |
| E040 | parser/compat | error | Rust `let mut` — use `var` instead | [E040.md](explanations/E040.md) |
| E041 | parser/compat | error | Rust `&mut` — use `&!` instead | [E041.md](explanations/E041.md) |
| E042 | parser/compat | error | Rust attribute `#[...]` not valid in Sounio | [E042.md](explanations/E042.md) |
| E043 | parser/compat | error | Rust macro `ident!(...)` not valid in Sounio | [E043.md](explanations/E043.md) |
| E067 | type-checker/epistemic | warning | Potential confounding in epistemic model | [E067.md](explanations/E067.md) |
| E070 | type-checker/kernel | error | IO or Mut effect in kernel function | [E070.md](explanations/E070.md) |
| E072 | type-checker/kernel | error | Kernel function must return unit `()` | [E072.md](explanations/E072.md) |
| E170 | type-checker/epistemic | error | `.value` on `Knowledge<T>` requires `with Epistemic` | [E170.md](explanations/E170.md) |
| E171 | type-checker/epistemic | error | Cannot cast epistemic type to its inner type | [E171.md](explanations/E171.md) |
| E200 | lean_single/resolve | error | Undefined identifier | — |
| E178 | type-checker/epistemic | error | noise-source capacity exceeded: more independent measurement sources than the noise-symbol domain can represent | — |
| E179 | type-checker/epistemic | error | noise-set interning table exhausted: too many distinct source-sets in one checker run | — |
| E201 | type-checker/zero-divisor | error | `ExactlyPrivate<T>` requires `with ZD` | [E201.md](explanations/E201.md) |
| E202 | type-checker/zero-divisor | error | `Editable<T>` requires `with ZD` | [E202.md](explanations/E202.md) |
| E203 | type-checker/zero-divisor | error | `CapabilityGated<T>` requires `with ZD` | [E203.md](explanations/E203.md) |
| E204 | type-checker/zero-divisor | error | `Composable<T>` requires `with ZD` | [E204.md](explanations/E204.md) |
| E205 | type-checker/zero-divisor | error | `Audited<T>` requires `with ZD, Witness` | [E205.md](explanations/E205.md) |
| E206 | type-checker/zero-divisor | error | `Revivable<T>` requires `with ZD, Temporal` | [E206.md](explanations/E206.md) |
| E207 | type-checker/zero-divisor | error | `Interpretable<T>` requires `with ZD` | [E207.md](explanations/E207.md) |
| E208 | type-checker/refinement | error | refinement type violation — integer value violates predicate | [E208.md](explanations/E208.md) |
| E209 | type-checker/refinement | error | refinement type violation — f64 value violates predicate | [E209.md](explanations/E209.md) |
| E210 | type-checker/algebra | error | algebra property violation | [E210.md](explanations/E210.md) |
| E211 | type-checker/study | error | study block requires at least one hypothesis | [E211.md](explanations/E211.md) |
| E212 | type-checker/algebra | error | Hessian AD over a non-associative algebra | [E212.md](explanations/E212.md) |
| E213 | type-checker/destructure | error | tuple destructure arity mismatch | [E213.md](explanations/E213.md) |
| E214 | type-checker/epistemic | error | confidence gate violation | [E214.md](explanations/E214.md) |
| E215 | type-checker/epistemic | error | EpistemicComplete violation | [E215.md](explanations/E215.md) |
| E216 | type-checker/recursive-type | error | infinite recursive type | [E216.md](explanations/E216.md) |
| E217 | codegen | error | invalid function body span | [E217.md](explanations/E217.md) |
| E218 | codegen | error | tail type mismatch | [E218.md](explanations/E218.md) |
| E219 | codegen | error | function pass mismatch | [E219.md](explanations/E219.md) |
| E220 | codegen/linker | error | unresolved function body for call target | [E220.md](explanations/E220.md) |
| E221 | codegen/linker | error | no main function | [E221.md](explanations/E221.md) |
| E222 | codegen | error | code buffer overflow | [E222.md](explanations/E222.md) |
| E223 | codegen/pe | error | too many ExitProcess call sites | [E223.md](explanations/E223.md) |
| E224 | import | error | unreadable import | [E224.md](explanations/E224.md) |
| E225 | import | error | import dedup table full | [E225.md](explanations/E225.md) |
| E226 | import | error | import path table full | [E226.md](explanations/E226.md) |
| E227 | import | error | import too large for SRC buffer | [E227.md](explanations/E227.md) |
| E228 | import | error | import copy truncated | [E228.md](explanations/E228.md) |
| E229 | lexer | error | source exceeds lexer byte buffer, or token table full (fail-closed) | [E229.md](explanations/E229.md) |
| E230 | type-checker/epistemic | error | independence-assuming operation over correlated uncertainty | — |
| E232 | type-checker/struct | error | the same array local initialises two fields | — |
| E236 | type-checker/epistemic | error | binary operation on Knowledge values reuses a correlated binding; independence is not assumed (add `with Correlated`) | — |
| E244 | type-checker/epistemic | error | Knowledge provenance boundary mismatch: the argument's provenance cannot satisfy the parameter's requirement (trusted classes: Source, Literature, Measured) | — |
| E245 | type-checker/epistemic | error | arithmetic between two Knowledge<T> values is not supported | — |
| E246 | type-checker/effects | error | unknown effect, or effect dropped as a ninth slot | — |
| E247 | type-checker/zero-divisor | error | ZD locus is not a well-formed sedenion pair | [E247.md](explanations/E247.md) |
| E248 | type-checker/numeric | error | f128/f256 value conversion is not implemented | [E248.md](explanations/E248.md) |
| E249 | parser/type-checker | error | f128/f256 is reserved for compiler-owned format identity | [E249.md](explanations/E249.md) |
| E250 | type-checker/extern | error | call to an extern C function the native backend does not implement | [E250.md](explanations/E250.md) |
| E251 | type-checker/epistemic | error | product variance shortcut is unsound beyond the octonions (norm not multiplicative on Sedenion/Clifford) | [E251.md](explanations/E251.md) |
| E252 | type-checker/algebra | error | algebra declares a reassociation strategy its multiplication law cannot certify | [E252.md](explanations/E252.md) |
| E253 | type-checker/gpu | error | kernel function must return unit type | — |
| E254 | type-checker/causal | error | is not a node of the declared causal graph | — |
| E255 | type-checker/causal | error | is not d-separated from the other variable given the stated conditioning set | — |
| E256 | type-checker/causal | error | `d_separated` needs a declared causal graph and exactly two variable names, with an optional conditioning set | — |
| E257 | type-checker/causal | error | `indep_dsep`'s argument must be a `d_separated(...)` call the compiler itself discharged, not a hand-written value | — |
| E258 | parser | error | string literal(s) exceed Name capacity (384 bytes incl. quotes); refuses rather than silently truncating | [E258.md](explanations/E258.md) |
