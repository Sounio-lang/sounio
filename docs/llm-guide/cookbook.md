<!-- docs:meta
topic_id: repo.docs.llm-guide.cookbook
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.llm-guide.cookbook
-->

# Sounio Cookbook

Idiomatic patterns for common tasks. Each recipe is grounded in real stdlib files or
passing tests. If you're about to write something from scratch, check here first.

---

## 1. Test File Structure

This is the canonical test pattern used throughout `tests/stdlib/`.

```sio
//@ run-pass
//! tests/stdlib/mymodule/test_foo_e2e.sio
//!
//! Brief description of what is tested.

use mymodule::{FooType, foo_new, foo_compute}
use test::helpers::{check_near, check_eq_i64}

// ============================================================================
// Individual test functions — each tests one thing
// ============================================================================

fn test_foo_basic() with Mut, Div, Panic {
    let result = foo_compute(1.0, 2.0)
    assert(check_near(result, 3.0, 1e-9))
}

fn test_foo_edge_zero() with Mut, Div, Panic {
    let result = foo_compute(0.0, 0.0)
    assert(check_near(result, 0.0, 1e-12))
}

// ============================================================================
// Main: run all tests, report pass/fail count
// ============================================================================

fn main() -> i32 with IO, Mut, Div, Panic {
    var passed = 0
    var failed = 0

    if { test_foo_basic(); true } { passed = passed + 1 } else { failed = failed + 1 }
    if { test_foo_edge_zero(); true } { passed = passed + 1 } else { failed = failed + 1 }

    print("passed: ")
    print_i64(passed)
    println("")
    print("failed: ")
    print_i64(failed)
    println("")

    if failed > 0 { 1 } else { 0 }
}
```

**Rules for test files:**
- Header annotation `//@ run-pass` is required
- Each test function: one assertion, name = `test_<what>_<scenario>`
- Effects on test functions: at minimum `with Mut, Div, Panic` (safe default)
- `main` returns `0` on all pass, `1` on any failure
- Use `check_near` for floats — never compare `f64` with `==`

---

## 2. Numeric Computation: ODE Solver

Pattern for calling `ode::rk4` for a scalar ODE `du/dt = f(u, t)`.

```sio
//@ run-pass
use ode::rk4::{rk4_integrate, RK4Solution}
use test::helpers::{check_near}

// Exponential decay: du/dt = -k*u, exact solution u(t) = u0 * exp(-k*t)
fn decay_rhs(u: f64, t: f64) -> f64 with Div {
    let k = 0.5
    0.0 - k * u
}

fn main() -> i32 with IO, Mut, Div, Panic {
    let u0 = 1.0
    let t_end = 4.0
    let n_steps = 400

    let sol = rk4_integrate(u0, 0.0, t_end, n_steps)

    // sol.values[sol.n - 1] is the final value
    let u_final = sol.values[sol.n - 1]
    let exact = 0.13533528         // exp(-0.5 * 4) ≈ 0.1353

    assert(check_near(u_final, exact, 1e-4))
    0
}
```

**Key constraint:** `rk4_integrate` takes a fixed `fn(f64, f64) -> f64` RHS.
For multi-variable ODEs, use the struct-based API with `PKState` structs (see `ode/rk4.sio`).

---

## 3. Epistemic Computation: GUM Uncertainty

Pattern for propagating measurement uncertainty through a calculation.

```sio
//@ run-pass
use epistemic::gum::{
    type_a_uncertainty, type_b_uncertainty,
    gum_combine, gum_expanded,
    GUMUncertainty,
}
use test::helpers::{check_near}

fn main() -> i32 with IO, Mut, Div, Panic {
    // Measurement A: 5 repeated readings, std dev = 0.05 mg
    let u_a = type_a_uncertainty(0.05, 5)

    // Measurement B: instrument spec = 0.02 mg (rectangular, divide by sqrt(3))
    let u_b = type_b_uncertainty(0.02)

    // Combined standard uncertainty (law of propagation)
    var components: [GUMUncertainty; 2] = [u_a, u_b]
    let u_c = gum_combine(components, 2)

    // Expanded uncertainty at 95% confidence
    let u_exp = gum_expanded(u_c, 0.95)

    // u_exp.expanded_uncertainty ≈ 2 * u_c (k≈2 for large dof)
    assert(u_exp.expanded_uncertainty > 0.0)
    assert(u_exp.coverage_factor >= 1.9)

    0
}
```

---

## 4. Statistical Pattern: Struct + impl

How to define a computation struct with methods.

```sio
struct RunningMean {
    sum: f64,
    count: i64,
}

impl RunningMean {
    fn new() -> RunningMean {
        RunningMean { sum: 0.0, count: 0 }
    }

    fn update(self: &!RunningMean, x: f64) with Mut {
        self.sum = self.sum + x
        self.count = self.count + 1
    }

    fn mean(self: &RunningMean) -> f64 with Div, Panic {
        if self.count == 0 { return 0.0 }
        self.sum / (self.count as f64)
    }
}

fn main() -> i32 with IO, Mut, Div, Panic {
    var rm = RunningMean::new()
    rm.update(1.0)
    rm.update(2.0)
    rm.update(3.0)
    let m = rm.mean()    // 2.0
    0
}
```

**Key patterns:**
- Constructor is a plain `fn new() -> T { T { ... } }` — no `Self`
- Mutable methods: `self: &!Type` + `with Mut` on both method and caller
- Immutable methods: `self: &Type`, no Mut required

---

## 5. Array Processing

Fixed-size arrays are the only collection with compile-time guarantees.

```sio
fn sum_array(data: [f64; 16], n: i64) -> f64 with Mut {
    var total = 0.0
    var i: i64 = 0
    while i < n {
        total = total + data[i]
        i = i + 1
    }
    total
}

fn scale_array(data: &![f64; 16], factor: f64) with Mut, Panic {
    var i: i64 = 0
    while i < 16 {
        (*data)[i] = (*data)[i] * factor
        i = i + 1
    }
}

// Fill pattern
fn zeros_16() -> [f64; 16] {
    [0.0; 16]
}

// Two-array operations (no dynamic allocation needed)
fn dot_product(a: [f64; 8], b: [f64; 8]) -> f64 with Mut {
    var sum = 0.0
    var i: i64 = 0
    while i < 8 {
        sum = sum + a[i] * b[i]
        i = i + 1
    }
    sum
}
```

**Rules:**
- Size must be a literal: `[f64; 16]` not `[f64; n]`
- Indexing requires `Panic` effect (bounds check can panic)
- Mutating through `&!` reference requires explicit deref: `(*arr)[i] = x`
- When mutation through bare `&![T; N]` fails (interpreter bug): wrap in struct

---

## 6. Higher-Order Functions

No closures. Pass named function references.

```sio
fn square(x: f64) -> f64 { x * x }
fn negate(x: f64) -> f64 { 0.0 - x }
fn identity(x: f64) -> f64 { x }

// Map over fixed-size array
fn map8(arr: [f64; 8], f: fn(f64) -> f64) -> [f64; 8] with Mut, Panic {
    var out: [f64; 8] = [0.0; 8]
    var i: i64 = 0
    while i < 8 {
        out[i] = f(arr[i])
        i = i + 1
    }
    out
}

// Reduce
fn fold8(arr: [f64; 8], init: f64, f: fn(f64, f64) -> f64) -> f64 with Mut {
    var acc = init
    var i: i64 = 0
    while i < 8 {
        acc = f(acc, arr[i])
        i = i + 1
    }
    acc
}

// Usage: pass named function, not closure
let squares = map8(data, square)
let total = fold8(data, 0.0, add_f64)
```

---

## 7. Error Handling Pattern

```sio
struct ParseResult {
    value: f64,
    error: i32,    // 0 = success, non-zero = error code
}

fn parse_positive(x: f64) -> ParseResult with Div {
    if x < 0.0 {
        return ParseResult { value: 0.0, error: 1 }
    }
    ParseResult { value: x, error: 0 }
}

fn compute(x: f64) -> (f64, i32) with Div, Panic {
    let r = parse_positive(x)
    if r.error != 0 { return (0.0, r.error) }
    // ... continue with r.value
    (r.value * 2.0, 0)
}
```

---

## 8. Sedenion / Hypercomplex Pattern

```sio
//@ run-pass
use math::sedenion64::{Sedenion64, sed64_basis, sed64_mul,
                       sed64_norm_sq, sed64_to_array}
use test::helpers::{check_near}

fn main() -> i32 with IO, Mut, Div, Panic {
    // Known zero-divisor pair in S (sedenions)
    let e3 = sed64_basis(3)
    let e10 = sed64_basis(10)
    let a = sed64_add(e3, e10)    // e3 + e10

    let e6 = sed64_basis(6)
    let e15 = sed64_basis(15)
    let b = sed64_sub(e6, e15)    // e6 - e15

    // a * b = 0 (zero divisor)
    let c = sed64_mul(a, b)
    let norm = sed64_norm_sq(c)

    assert(check_near(norm, 0.0, 1e-10))
    0
}
```

**Note:** Sedenion multiplication is non-associative and non-alternative in the full algebra.
Use the `math::g2_variety` module for G₂ variety computations.

---

## 9. Scientific Constant Pattern

The stdlib has no global constants (BSS restriction). Use `fn` constants instead:

```sio
// CORRECT — fn returning constant
fn PI() -> f64 { 3.14159265358979323846 }
fn E()  -> f64 { 2.71828182845904523536 }
fn SPEED_OF_LIGHT() -> f64 { 299792458.0 }

// Usage
let circumference = 2.0 * PI() * r
```

**Never use `let` at module level** — top-level `let` is not supported in the native compiler.

---

## 10. Effects Propagation: The Superset Rule

A function must declare all effects it *or any callee* requires.

```sio
fn inner() -> f64 with Div, Panic { 1.0 / 2.0 }

// WRONG: caller doesn't declare Div, Panic
fn outer() -> f64 {
    inner()    // ERROR: missing Div, Panic
}

// CORRECT: outer declares superset of inner's effects
fn outer() -> f64 with Div, Panic {
    inner()
}

// Entry point fn main() must declare everything used anywhere in the call tree
fn main() -> i32 with IO, Mut, Div, Panic {
    // ...
    0
}
```

**Rule of thumb for `main`:** always add `with IO, Mut, Div, Panic`. Add `Observe` if you
use `Unobserved<T>`.

---

## 11. Negative Numbers

```sio
// WRONG — no unary minus operator
let neg = -42
let neg_f = -3.14

// CORRECT
let neg = 0 - 42
let neg_f = 0.0 - 3.14

// In expressions
let diff = 0.0 - x * 2.0    // -(x * 2.0)
let negated = 0.0 - val
```

---

## 12. Module Layout Template

```sio
// stdlib/mymodule/lib.sio

// ============================================================================
// Types
// ============================================================================

pub struct MyType {
    value: f64,
    count: i64,
}

pub struct MyResult {
    out: f64,
    error: i32,
}

// ============================================================================
// Constants (fn pattern — no top-level let)
// ============================================================================

fn MY_CONST() -> f64 { 1.234 }

// ============================================================================
// Internal helpers (no pub)
// ============================================================================

fn helper_abs(x: f64) -> f64 {
    if x < 0.0 { 0.0 - x } else { x }
}

// ============================================================================
// Public API
// ============================================================================

pub fn my_new(v: f64) -> MyType {
    MyType { value: v, count: 0 }
}

pub fn my_compute(t: MyType) -> MyResult with Div, Panic {
    if t.count == 0 { return MyResult { out: 0.0, error: 1 } }
    MyResult { out: t.value / (t.count as f64), error: 0 }
}

impl MyType {
    fn update(self: &!MyType, x: f64) with Mut {
        self.value = self.value + x
        self.count = self.count + 1
    }
}
```

---

## Anti-Patterns to Avoid

```sio
// ✗ Global state — not supported
let GLOBAL_BUF: [f64; 256] = [0.0; 256]

// ✗ Generic functions — only Knowledge<T> is built-in
fn foo<T>(x: T) -> T { x }

// ✗ Trait bounds
fn bar(x: impl Display) { ... }

// ✗ Dynamic allocation for computation
let v = Vec::new()    // Vec exists but don't use for math

// ✗ Negative literal
let x = -1.0          // use 0.0 - 1.0

// ✗ Missing effects on nested calls
fn outer() { inner_that_divides() }   // must add with Div, Panic

// ✗ Mutation without Mut effect
fn bad(x: &!f64) { *x = 1.0 }        // must add with Mut

// ✗ Semicolons
let x = 5;    // parse error or wrong type
```
