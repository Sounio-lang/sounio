<!-- docs:meta
topic_id: repo.docs.guide.llm-programming-guide
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A5
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.guide.llm-programming-guide
-->

# Sounio LLM Programming Guide

Definitive syntax reference for LLMs writing Sounio code. Every example is verified from `tests/run-pass/` or working stdlib files.

**Sounio is NOT Rust.** The syntax looks similar but semantics differ. When in doubt, check real `.sio` files.

## 1. Hello World

```sio
// Source: tests/run-pass/hello.sio
fn main() with IO {
    println("Hello, Sounio!")
}
```

## 2. Variables

```sio
let x = 5                    // immutable
var y: i32 = 10              // mutable (can reassign)
y = y + 1                    // OK — var allows reassignment

// NO semicolons
// let x = 5;   <-- WRONG
```

**No `let mut`** — use `var`.

## 3. Types

### Primitives
`i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`, `f32`, `f64`, `bool`, `char`

### Fixed-Size Arrays [Production]
```sio
var buffer: [u8; 256] = [0; 256]
let data: [i64; 4] = [1, 2, 3, 4]
let matrix: [f64; 9] = [0.0; 9]
```

### Vec [Production]
```sio
// Source: tests/run-pass/for_in_loops.sio:29
let vec: Vec<i32> = [1, 2, 3, 4]
for x in vec {
    // iterate
}
```
Stdlib also provides monomorphic `IntVec`, `FloatVec` via `stdlib/collections/vec.sio` with `impl` blocks and push/pop/len.

### Tuples
```sio
let pair = (1, 2)

// Destructuring works:
// Source: tests/run-pass/tuple_destructure_let.sio
let (a, b) = (1, 2)
let (x, (y, z)) = (10, (20, 30))
let (first, _) = (5, 10)       // wildcard
```

### Structs [Production]
```sio
struct Point { x: f64, y: f64 }
let p = Point { x: 1.0, y: 2.0 }

linear struct Handle { fd: i32 }   // linear types
```

### Enums [Beta]
```sio
// Source: tests/run-pass/native_enum_basic.sio
enum Color { Red, Green, Blue }

let r = Color::Red
let g = Color::Green
```

Note: Enum definition and variant access work. Passing enum values to functions expecting `i64` may require casting — the type checker distinguishes enum types from integers.

### Refinement Types [Beta]
```sio
type Probability = { p: f64 | p >= 0 }
fn divide(num: i32, denom: { d: i32 | d != 0 }) -> i32 with Panic {
    num / denom
}
```

### Units of Measure [Production]
```sio
unit kg;
unit mg = 0.001 * kg;
let dose: mg = 500.0
```

### Epistemic Types [Production]
```sio
let m: Knowledge<mg> = measure(500.0, uncertainty: 2.5)
```

### Algebra Declarations [Beta]
```sio
algebra Octonion over f64 {
    add: commutative, associative
    mul: alternative, non_commutative
    reassociate: fano_selective
}
```

Supported properties:
- `add`: `commutative`, `associative`
- `mul`: `commutative`, `associative`, `alternative`, `non_commutative`
- `reassociate`: `free`, `blocked`, `fano_selective`

`mul: alternative` derives `NonAssoc` requirements for functions that multiply the type.

### Observation Types [Beta]
```sio
fn sense() -> Unobserved<f64> with Observe {
    37.2
}

fn above_threshold(x: Unobserved<f64>) -> bool with Observe {
    x > 36.0
}
```

`Unobserved<T>` carries a value before observation. Comparisons and other observation boundaries require `with Observe`; pure functions can pass `Unobserved<T>` through unchanged.

## 4. Functions

```sio
fn add(a: i32, b: i32) -> i32 {
    a + b
}

// With effects (REQUIRED for certain operations)
fn divide(a: f64, b: f64) -> f64 with Div, Panic {
    a / b
}

// Explicit return
fn abs(x: f64) -> f64 {
    if x < 0.0 { return 0.0 - x }
    x
}
```

### Function References [Production]
```sio
// Source: tests/run-pass/closure_fn_ref.sio
fn square(x: i64) -> i64 { x * x }
fn apply(f: fn(i64) -> i64, x: i64) -> i64 { f(x) }

// Store in variable
let f = square
let r = f(7)          // 49

// Return from function
fn select_op(which: i64) -> fn(i64) -> i64 with Mut, Panic, Div {
    if which == 0 { add_one }
    else { negate }
}
```

### Higher-Order Patterns [Production]
```sio
// Source: tests/run-pass/closure_higher_order.sio
fn map4(arr: [i64; 4], f: fn(i64) -> i64) -> [i64; 4] with Mut, Panic, Div {
    var out: [i64; 4] = [0; 4]
    var i: i64 = 0
    while i < 4 { out[i] = f(arr[i]); i = i + 1 }
    out
}

fn fold4(arr: [i64; 4], init: i64, f: fn(i64, i64) -> i64) -> i64 with Mut, Panic, Div {
    var acc = init
    var i: i64 = 0
    while i < 4 { acc = f(acc, arr[i]); i = i + 1 }
    acc
}

// Usage:
let doubled = map4(data, dbl)
let sum = fold4(data, 0, add)
let sum_sq = fold4(map4(data, sq), 0, add)   // chained
```

**Note:** Closure literals (`|x| x + 1`) are BLOCKED. Only named function references work.

### impl Blocks [Production]
```sio
// Source: stdlib/collections/vec.sio
impl IntVec {
    fn new() -> IntVec {
        IntVec { data: [0; 4096], len: 0 }
    }

    fn push(self: &! IntVec, val: i64) {
        if self.len < VEC_CAP {
            self.data[self.len] = val
            self.len = self.len + 1
        }
    }

    fn len(self: &IntVec) -> i64 {
        self.len
    }
}
```

Methods use explicit `self: &Type` or `self: &! Type` — no implicit `self`.

## 5. Effects System [Production]

Effects track what a function can do. Missing effects = compile error.

| Effect | Required when | Example |
|--------|---------------|---------|
| `IO` | Printing, file ops, env | `println("text")` |
| `Mut` | Mutating `&!` refs or arrays — **not** a plain local `var` | `arr[i] = 42`, `*x = 10` |
| `Div` | Division `/` or modulo `%` | `a / b` (always pair with `Panic`) |
| `Panic` | Array bounds, asserts, `as` casts | `arr[i]`, `assert(cond)` |
| `Alloc` | Heap allocation | Rare |
| `Observe` | Collapsing `Unobserved<T>` at an observation boundary | `if reading > 36.0 { ... }` |
| `Async` | Async operations | Rare |
| `GPU` | GPU kernels | Rare |
| `Prob` | Probabilistic operations | Rare |

> **`Mut` and the two engines (measured 2026-07-27).** The rule above — `Mut`
> for mutation the caller can observe, nothing for a function-local `var` — is
> the intended semantics, specified in `docs/spec/LANGUAGE_SPECIFICATION.md`
> §7.2.1. Neither shipped engine enforces exactly it: the default compiler
> (Madaros) currently requires `Mut` for **neither** case, and the frozen
> `lean_single` seed requires it for **both** (so a pure integer helper that
> mutates a local is rejected under `SOUNIO_SOUC_ENGINE=lean_single`). Writing
> the annotation as the table describes is correct and future-proof; omitting
> it on a local `var` will not currently be caught by the default compiler.
> Scoped in `docs/audit/MUT_EFFECT_ENFORCEMENT_DISPATCH_2026-07-27.md`.

```sio
fn pure_add(a: i64, b: i64) -> i64 { a + b }                   // no effects = pure
fn bump(n: i64) -> i64 { var y = 1  y = y + n  y }             // local var only: no Mut
fn mutate(x: &!i32) with Mut { *x = 42 }                        // mutation
fn divide(a: f64, b: f64) -> f64 with Div, Panic { a / b }      // division
fn observe(x: Unobserved<f64>) -> bool with Observe { x > 0.0 } // observation
fn process() with IO, Mut, Panic, Div { /* all four */ }         // multiple
```

Effect subsetting: a caller with `with IO, Mut, Div` can call a callee with no effects or fewer effects. Effects propagate upward.

## 6. Control Flow

### if/else [Production]
```sio
if x > 0 {
    println("positive")
} else if x < 0 {
    println("negative")
} else {
    println("zero")
}

// As expression
let result = if condition { value1 } else { value2 }
```

### while [Production]
```sio
var i = 0
while i < 10 {
    process(i)
    i = i + 1
}
```

### for-in [Production]
```sio
// Source: tests/run-pass/for_in_loops.sio

// Range (exclusive)
for i in 0..5 { /* 0,1,2,3,4 */ }

// Range (inclusive)
for i in 0..=5 { /* 0,1,2,3,4,5 */ }

// Range with variable bound
let n = 10
for i in 0..n { /* 0..9 */ }

// Array iteration
let arr = [10, 20, 30]
for x in arr { sum = sum + x }

// Vec iteration
let vec: Vec<i32> = [1, 2, 3, 4]
for x in vec { sum = sum + x }

// Nested
for i in 0..3 {
    for j in 0..3 { /* ... */ }
}

// break and continue
for i in 0..100 {
    if i >= 5 { break }
}
for i in 0..10 {
    if i % 2 == 0 { continue }
    odd_sum = odd_sum + i
}
```

### match [Production]
```sio
// Source: tests/run-pass/native_enum_basic.sio
fn color_to_int(c: i64) -> i64 {
    match c {
        Color::Red => 10
        Color::Green => 20
        Color::Blue => 30
        _ => 0
    }
}
```

## 7. References

### Shared Reference `&T` [Production]
```sio
fn read_ref(r: &i64) -> i64 { *r }
let val = read_ref(&x)
```

### Exclusive Reference `&!T` [Production]
```sio
// Source: tests/run-pass/array_mut_ref.sio
fn fill(arr: &![i64; 8]) with Mut, Panic {
    (*arr)[0] = 99       // EXPLICIT DEREF for bare arrays
    (*arr)[1] = 42
}

fn main() -> i64 with IO, Mut, Panic {
    var buf: [i64; 8] = [0, 0, 0, 0, 0, 0, 0, 0]
    fill(&! buf)         // note: space between & and ! is OK
    buf[0]               // reads 99
}
```

**Known Bug:** Bare `&![T; N]` mutations may not propagate in the interpreter. Workaround: wrap in a struct.

```sio
// WORKAROUND — struct wrapper pattern
struct SortBuf { data: [i64; 10000] }
fn sort(b: &! SortBuf) with Mut { b.data[0] = 99 }   // works correctly
```

## 8. Operators

### Arithmetic
`+`, `-`, `*`, `/` (needs `Div`), `%` (needs `Div`)

### Comparison
`==`, `!=`, `<`, `<=`, `>`, `>=`

### Logical
`&&`, `||`, `!` (short-circuit)

### Bitwise
`&`, `|`, `^`, `>>`, `<<`

**Bit shift operand must be `u8`:**
```sio
let high = byte >> 4u8
let low = byte & 15u8
```

### No Unary Minus
```sio
let neg = 0 - 42       // correct
// let neg = -42        // WRONG — no unary minus
```

### Concatenation
```sio
let combined = a ++ b   // array concatenation
```

### Type Casting
```sio
let u: u8 = i as u8
let idx = n as usize    // required for array indexing
```

## 9. Modules & Imports [Production]

```sio
// Source: tests/run-pass/import_basic_main.sio
use import_basic_a::{imported_add}

fn main() -> i64 {
    let result = imported_add(3, 4)
    result
}
```

Visibility: `pub fn`, `pub struct` export items across module boundaries.

```sio
// stdlib/encoding/hex.sio
pub fn hex_encode(data: &[u8; 256], data_len: i32, out: &![u8; 512]) -> i32
    with Mut, Div, Panic { /* ... */ }
```

## 10. Strings

String literals work directly:
```sio
println("Hello, World!")
print("value = ")
```

For mutable string data, use fixed-size byte arrays:
```sio
var name: [i8; 64] = [0; 64]
name[0] = 72i8    // 'H'
name[1] = 101i8   // 'e'
```

## 11. Error Handling

No exceptions. Use error codes or result structs:

```sio
// Simple: tuple with error code
fn safe_divide(a: f64, b: f64) -> (f64, i32) with Div, Panic {
    if b == 0.0 { (0.0, 1) }    // error code 1
    else { (a / b, 0) }          // success
}

// Stdlib provides monomorphic result types:
// stdlib/core/result.sio — IntResult, FloatResult
// stdlib/core/option.sio — IntOption, FloatOption
```

## 12. Testing

```sio
// Inline assertions
fn test_addition() {
    let result = add(2, 3)
    assert(result == 5)
}

// Test helpers from stdlib/test/helpers.sio
pub fn check_near(a: f64, b: f64, tol: f64) -> bool {
    let d = a - b
    let ad = if d < 0.0 { 0.0 - d } else { d }
    ad < tol
}
```

Test annotations in file headers:
- `//@ run-pass` — should compile and run
- `//@ compile-fail` — should fail to compile
- `//@ error-pattern: <text>` — expected error
- `//@ ignore` — skip this test

## 13. FFI [Production, with limits]

```sio
extern "C" {
    fn sqrt(x: f64) -> f64
    fn pow(x: f64, y: f64) -> f64
}
```

**Supported FFI functions** (JIT only):
- Single-arg `f64→f64`: `sqrt`, `sin`, `cos`, `tan`, `exp`, `log`, `floor`, `ceil`, `atan`, `sinh`, `cosh`, `tanh`, `asin`, `acos`, `cbrt`, `round`, `log2`, `log10`
- Two-arg `f64,f64→f64`: `pow`, `atan2`
- Integer FFI (`malloc`, `getpid`, etc.) **silently terminates** — do not use.

## 14. Async Concurrency [Production — native binary only]

All async primitives use the OS fork model (not green threads or state machines). Requires `with Async` effect.

```sio
// Source: tests/run-pass/async_spawn.sio
fn main() with IO, Async {
    let h1 = spawn { 10 + 5 }     // fork — runs concurrently
    let h2 = spawn { 20 + 1 }
    let r1 = h1.await              // wait4(pid), read mmap slot
    let r2 = h2.await
    print("r1="); print_i64(r1)   // 15
    print(" r2="); print_i64(r2)  // 21
}
```

### Channels (pipe-backed)

```sio
// Source: tests/run-pass/async_channels.sio
fn main() with IO, Async {
    let (tx, rx) = channel::<i64>()
    let h = spawn { tx.send(42).await }
    let v = rx.recv().await
    h.await
    print_i64(v)   // 42
}
```

### sleep(ms).await

```sio
// Source: tests/run-pass/async_sleep.sio
fn main() with IO, Async {
    sleep(10).await               // nanosleep — 10ms
    let t1 = spawn { sleep(5).await; 1 }
    let t2 = spawn { sleep(5).await; 2 }
    let r1 = t1.await
    let r2 = t2.await             // both ran in parallel
}
```

### join(h1, h2)

```sio
// Source: tests/run-pass/async_join.sio
fn main() with IO, Async {
    let h1 = spawn { 10 }
    let h2 = spawn { 20 }
    let (r1, r2) = join(h1, h2)  // returns (i64, i64) tuple
}
```

**Async rules:**
- `spawn { expr }` requires `with Async`; the block body runs in a forked child
- Child cannot write back to parent variables (fork COW isolation — expected)
- `join` supports exactly 2 handles; for more handles, use sequential `.await`
- `sleep`/`join` are soft keywords — identifiers named `sleep`/`join` are fine in other scopes

## 16. Ontology Declarations [Production]

```sio
// Source: tests/run-pass/ontology_roles_basic.sio
ontology Pharma {
    class Drug
    class Disease
    class Rapamycin subclass_of Drug
    role treats domain Drug range Disease
    role treated_by inverse_of treats
    role has_part transitive
    disjoint Drug, Disease

    class StrongDrug subclass_of Drug {
        property potency: f64 where potency >= 10.0
    }
}
```

Classes become types usable in function signatures. Disjointness and subsumption are enforced at compile time. OWL 2 axiom semantics: SubClassOf, EquivalentClasses, DisjointClasses, object property domain/range, inverse properties.

## 17. Study Blocks (PPCR / Clinical Research) [Beta]

```sio
// Source: tests/run-pass/study_block_basic.sio
study MyTrial {
    title: "Rapamycin Dosing Study"
    design: parallel_rct
    participants { sample_size: 120, power: 0.80 }
    outcomes { primary: blood_concentration }
    analysis {
        hypothesis H1 { outcome: blood_concentration, direction: greater, effect_size: 0.5 }
        alpha: 0.05
        correction: bonferroni
    }
}
```

CONSORT-aligned study declarations with pre-registered hypotheses, multiple testing correction, and audit trails.

### PPCR Effects

| Effect | ID | Required when |
|--------|----|---------------|
| `Audit` | 15 | Provenance tracking (PROV-DM) |
| `Hypothesis` | 16 | Statistical tests on registered endpoints |
| `MultiTest` | 17 | Multiple hypothesis tests requiring correction |

```sio
// Registered analysis — no warning
fn analyze() -> i32 with Hypothesis {
    t_test_one_sample(endpoint_a, 30, 0)
}

// Unregistered analysis — W041 warning
fn exploratory() -> i32 with Hypothesis {
    t_test_one_sample(bmi, 50, 0)  // W041: testing unregistered variable
}
```

## 18. What Does NOT Work (Verified)


| Feature | Status | Use Instead |
|---------|--------|-------------|
| Semicolons `let x = 5;` | Never | `let x = 5` |
| `&mut` | Never | `&!` |
| `let mut` | Never | `var` |
| Rust macros `assert!()` `println!()` | Never | `assert()` `println()` |
| Closure literals `\|x\| x+1` | Blocked | Named fn refs: `let f = square` |
| Attributes `#[test]` `#[derive]` | Never | Inline tests |
| Unary minus `-42` | Never | `0 - 42` |
| Integer FFI (`malloc`, etc.) | Broken in JIT | Fixed-size arrays; native binary has workaround via syscall stubs |
| Bare `&![T;N]` mutation (interpreter) | JIT only | Struct wrapper in JIT; works in native binary |
| Async / `spawn` / `channel` | JIT: not supported | Use native binary (`./bin/souc run`) |

## 19. Quick Checklist

Before submitting Sounio code:

1. No semicolons in expressions
2. `&!` for mutable refs, `var` for mutable bindings
3. Effects declared: `with IO, Mut, Div, Panic` as needed
4. No Rust macros — use `assert()`, `println()`, `print()`
5. Bit shifts use `u8`: `x >> 4u8`
6. Negative numbers: `0 - x`
7. Array index cast: `arr[i as usize]`
8. Named fn refs for higher-order, not closures
9. Bare array `&!` mutation: wrap in struct if interpreter

## 20. Real Code to Study

| File | What it demonstrates |
|------|---------------------|
| `tests/run-pass/hello.sio` | Hello world, `println`, effects |
| `tests/run-pass/for_in_loops.sio` | All for-in variants, break, continue |
| `tests/run-pass/closure_fn_ref.sio` | Function references, `apply`, `select_op` |
| `tests/run-pass/closure_higher_order.sio` | map, fold, any, all patterns |
| `tests/run-pass/native_enum_basic.sio` | Enums, match expressions |
| `tests/run-pass/array_mut_ref.sio` | `&!` array mutation, explicit deref |
| `tests/run-pass/tuple_destructure_let.sio` | Tuple destructuring |
| `stdlib/encoding/hex.sio` | Real stdlib: hex encode/decode |
| `stdlib/collections/vec.sio` | impl blocks, IntVec, push/pop |
| `stdlib/math/approx.sio` | Function pointers, Chebyshev approximation |
| `stdlib/prob/normal.sio` | Structs, math, scientific computing |
| `tests/run-pass/ontology_roles_basic.sio` | Ontology: classes, roles, disjoint |
| `tests/run-pass/study_block_basic.sio` | PPCR: study block with hypotheses |
| `tests/run-pass/hypothesis_registered.sio` | Hypothesis effect, registered analysis |
| `tests/run-pass/algebra_g2_invariants.sio` | Algebra declarations, octonions |
| `tests/run-pass/async_spawn.sio` | spawn/await, concurrent tasks |
| `tests/run-pass/async_channels.sio` | channel::<T>(), send, recv |
| `tests/run-pass/async_sleep.sio` | sleep(ms).await, parallel sleep |
| `tests/run-pass/async_join.sio` | join(h1, h2), tuple destructuring |
