<!-- docs:meta
topic_id: repo.docs.archived.getting-started-duplicates.sounio-style-guide
authority: archived
audience: maintainers
last_validated: 2026-03-07
validated_by: A7
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.archived.getting-started-duplicates.sounio-style-guide
-->


<!-- docs:status-note:start -->
> Docs status: `archived`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Sounio Style Guide for LLMs

Sounio looks like Rust but has DIFFERENT syntax and semantics. This guide covers style, patterns, and conventions.

**Full syntax ref**: [docs/LLM_PROGRAMMING_GUIDE.md](../LLM_PROGRAMMING_GUIDE.md)

## 1. Basic Syntax

```sio
// Variables (NO semicolons)
let x = 42
var y: i32 = 10

// Functions with effects
fn divide(a: f64, b: f64) -> f64 with Div, Panic { a / b }

// Structs
struct Point { x: f64, y: f64 }

// Arrays (fixed size)
var buffer: [u8; 256] = [0; 256]
```

### What NOT to Write (Rust-isms)
```
let x = 42;              // ❌ semicolons
let mut x = 5;            // ❌ use 'var'
fn foo(x: &mut i32) {}    // ❌ use '&!'
assert!(x == 5);          // ❌ use 'assert()'
println!("hi");           // ❌ use 'println()'
let neg = -42;            // ❌ use '0 - 42'
```

## 2. Effects System

Effects track what a function can do. Missing = compile error.

```sio
fn pure_add(a: i64, b: i64) -> i64 { a + b }                    // pure
fn mutate(x: &!i32) with Mut { *x = 42 }                         // mutation
fn divide(a: f64, b: f64) -> f64 with Div, Panic { a / b }       // division
fn hello() with IO { println("hi") }                              // I/O
fn process(arr: &![u8; 256]) with IO, Mut, Panic, Div { /* */ }  // multiple
```

| Effect | When required |
|--------|---------------|
| `IO` | `print()`, `println()`, file/env |
| `Mut` | `&!` mutation, array assignment |
| `Div, Panic` | Division `/`, modulo `%` |
| `Panic` | Array access, `assert()`, `as` casts |
| `Alloc` | Heap allocation (rare) |

## 3. Type System

### Primitives
`i8` `i16` `i32` `i64` `u8` `u16` `u32` `u64` `f32` `f64` `bool` `char`

### Fixed-Size Arrays
```sio
var data: [u8; 256] = [0; 256]
let matrix: [f64; 9] = [0.0; 9]
let arr = [10, 20, 30]
```

### Vec (stdlib)
```sio
// Source: tests/run-pass/for_in_loops.sio
let vec: Vec<i32> = [1, 2, 3, 4]

// Monomorphic stdlib vecs: IntVec, FloatVec (stdlib/collections/vec.sio)
```

### Tuples
```sio
let pair = (1, 2)
let (a, b) = (1, 2)                   // destructuring works
let (x, (y, z)) = (10, (20, 30))      // nested
```

### Structs
```sio
struct Point { x: f64, y: f64 }
let p = Point { x: 1.0, y: 2.0 }

linear struct Handle { fd: i32 }       // linear types
```

### Enums
```sio
// Source: tests/run-pass/native_enum_basic.sio
enum Color { Red, Green, Blue }
let r = Color::Red
```

### Refinement Types
```sio
type Probability = { p: f64 | p >= 0 }
```

### Units
```sio
unit kg;
unit mg = 0.001 * kg;
let dose: mg = 500.0
```

### Strings
```sio
println("Hello")                     // string literals for output
var name: [i8; 64] = [0; 64]        // mutable: fixed byte array
```

## 4. Control Flow

### if/else
```sio
if x > 0 { println("positive") }
else if x < 0 { println("negative") }
else { println("zero") }

let result = if flag { 1 } else { 0 }   // expression
```

### while
```sio
var i = 0
while i < 10 { process(i); i = i + 1 }
```

### for-in
```sio
// Source: tests/run-pass/for_in_loops.sio
for i in 0..5 { /* 0..4 */ }
for i in 0..=5 { /* 0..5 inclusive */ }
for x in arr { sum = sum + x }
for x in vec { sum = sum + x }
// break and continue work
```

### match
```sio
// Source: tests/run-pass/native_enum_basic.sio
match c {
    Color::Red => 10
    Color::Green => 20
    _ => 0
}
```

## 5. Functions

### Signatures
```sio
fn add(a: i32, b: i32) -> i32 { a + b }
pub fn process(data: &[u8; 256], len: i32) -> i32 with Mut, Div, Panic { /* */ }
```

### Function References
```sio
// Source: tests/run-pass/closure_fn_ref.sio
fn square(x: i64) -> i64 { x * x }
fn apply(f: fn(i64) -> i64, x: i64) -> i64 { f(x) }

let f = square
let r = apply(square, 5)    // 25

fn select_op(which: i64) -> fn(i64) -> i64 with Mut, Panic, Div {
    if which == 0 { add_one } else { negate }
}
```

**No closure literals** (`|x| x+1` is blocked). Only named fn refs.

### impl Blocks
```sio
// Source: stdlib/collections/vec.sio
impl IntVec {
    fn new() -> IntVec { IntVec { data: [0; 4096], len: 0 } }
    fn push(self: &! IntVec, val: i64) { /* ... */ }
    fn len(self: &IntVec) -> i64 { self.len }
}
```

Methods use explicit `self: &Type` or `self: &! Type`.

## 6. References

```sio
// Shared
fn read(r: &i64) -> i64 { *r }

// Exclusive (&! not &mut)
fn write(x: &!i32) with Mut { *x = 42 }

// Bare array mutation — use explicit deref
fn fill(arr: &![i64; 8]) with Mut, Panic {
    (*arr)[0] = 99     // (*arr)[i] NOT arr[i]
}
```

## 7. Modules & Imports

```sio
use encoding::hex::{hex_encode}
pub fn my_function() -> i32 { /* ... */ }
```

## 8. Common Patterns

### Higher-Order Functions
```sio
// Source: tests/run-pass/closure_higher_order.sio
fn map4(arr: [i64; 4], f: fn(i64) -> i64) -> [i64; 4] with Mut, Panic, Div {
    var out: [i64; 4] = [0; 4]
    var i: i64 = 0
    while i < 4 { out[i] = f(arr[i]); i = i + 1 }
    out
}
let doubled = map4(data, dbl)
```

### Error Handling
```sio
fn safe_divide(a: f64, b: f64) -> (f64, i32) with Div, Panic {
    if b == 0.0 { (0.0, 1) } else { (a / b, 0) }
}
let (result, err) = safe_divide(10.0, 0.0)
```

### Struct By-Value Update (JIT workaround)
```sio
struct Counter { value: i32 }
fn increment(c: Counter) -> Counter {
    Counter { value: c.value + 1 }
}
var c = Counter { value: 0 }
c = increment(c)
```

### Bit Operations
```sio
let high = (b >> 4u8) & 15u8    // shift amount MUST be u8
let low = b & 15u8
```

## 9. FFI

```sio
extern "C" {
    fn sqrt(x: f64) -> f64
    fn pow(x: f64, y: f64) -> f64
}
```

Only math functions work. Integer FFI (`malloc`, `getpid`) silently terminates.

## 10. Testing

```sio
fn test_addition() {
    let result = add(2, 3)
    assert(result == 5)
}

fn check_near(a: f64, b: f64, tol: f64) -> bool {
    let d = a - b
    let ad = if d < 0.0 { 0.0 - d } else { d }
    ad < tol
}
```

Annotations: `//@ run-pass`, `//@ compile-fail`, `//@ error-pattern: <text>`, `//@ ignore`

## 11. Checklist

1. No semicolons
2. `&!` not `&mut`, `var` not `let mut`
3. Effects: `with IO, Mut, Div, Panic` as needed
4. No Rust macros: `assert()` not `assert!()`, `println()` not `println!()`
5. No unary minus: `0 - x`
6. Bit shifts: `u8` operand (`x >> 4u8`)
7. Array index cast: `arr[i as usize]`
8. Named fn refs, not closure literals
9. Bare `&![T;N]`: use `(*arr)[i]` or struct wrapper

## 12. Real Code to Study

| File | Demonstrates |
|------|-------------|
| `tests/run-pass/hello.sio` | Hello world |
| `tests/run-pass/for_in_loops.sio` | All for-in variants |
| `tests/run-pass/closure_fn_ref.sio` | Function references |
| `tests/run-pass/closure_higher_order.sio` | map, fold, any, all |
| `tests/run-pass/native_enum_basic.sio` | Enums + match |
| `stdlib/encoding/hex.sio` | Real stdlib code |
| `stdlib/collections/vec.sio` | impl blocks |

**Sounio ≠ Rust. When in doubt, check `tests/run-pass/` for verified examples.**
