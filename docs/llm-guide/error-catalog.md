<!-- docs:meta
topic_id: repo.docs.llm-guide.error-catalog
authority: repo_only
audience: users
last_validated: 2026-03-07
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

Stable codes emitted by the compiler in `error[Exxxx]:` format. Parseable by `souc check --json` into `sounio.diagnostic.v1` JSON.

Use `souc explain <CODE>` for a one-paragraph explanation + minimal example + canonical fix.

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
| E201 | type-checker/zero-divisor | error | `ExactlyPrivate<T>` requires `with ZD` | [E201.md](explanations/E201.md) |
| E202 | type-checker/zero-divisor | error | `Editable<T>` requires `with ZD` | [E202.md](explanations/E202.md) |
| E203 | type-checker/zero-divisor | error | `CapabilityGated<T>` requires `with ZD` | [E203.md](explanations/E203.md) |
| E204 | type-checker/zero-divisor | error | `Composable<T>` requires `with ZD` | [E204.md](explanations/E204.md) |
| E205 | type-checker/zero-divisor | error | `Audited<T>` requires `with ZD, Witness` | [E205.md](explanations/E205.md) |
| E206 | type-checker/zero-divisor | error | `Revivable<T>` requires `with ZD, Temporal` | [E206.md](explanations/E206.md) |
| E207 | type-checker/zero-divisor | error | `Interpretable<T>` requires `with ZD` | [E207.md](explanations/E207.md) |
