<!-- docs:meta
topic_id: repo.docs.archived.getting-started-duplicates.sounio-quick-start
authority: archived
audience: maintainers
last_validated: 2026-03-07
validated_by: A7
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.archived.getting-started-duplicates.sounio-quick-start
-->


<!-- docs:status-note:start -->
> Docs status: `archived`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Sounio Quick Start for LLMs

> **Other guides**: [Scientists' Quick Start](../QUICK_START_GUIDE.md) | [General Getting Started](getting-started.md) | [Conservative contract](MINIMUM_VIABLE_SOUNIO.md)

**TL;DR**: Sounio is NOT Rust. No semicolons. Effects required. `&!` not `&mut`. Study real `.sio` files.

**Full syntax ref**: [docs/LLM_PROGRAMMING_GUIDE.md](../LLM_PROGRAMMING_GUIDE.md)

## 30-Second Comparison

| Thing | Rust | Sounio |
|-------|------|--------|
| Function | `fn add(a: i32) -> i32 { }` | `fn add(a: i32) -> i32 { }` |
| Effects | N/A | `fn div(a: f64, b: f64) -> f64 with Div, Panic { a / b }` |
| Variable | `let x = 5;` | `let x = 5` (no semicolon) |
| Mutable var | `let mut x = 5;` | `var x = 5` |
| Arrays | `Vec<u8>` or `[u8; N]` | `[u8; N]` (fixed) + `Vec<i32>` (stdlib) |
| Mutable ref | `&mut x` | `&!x` |
| Closures | `\|x\| x * 2` | Named fn refs: `let f = double` |
| Error | `Result<T, E>` | Error codes `(T, i32)` or `IntResult` |
| Print | `println!("text")` | `println("text")` (no `!`) |

## Must-Know Rules

### 1. NO Semicolons
```sio
let x = 5        // ✅ correct
// let x = 5;    // ❌ WRONG
```

### 2. Effects Are Required
```sio
fn divide(a: f64, b: f64) -> f64 with Div, Panic { a / b }
fn mutate(x: &!i32) with Mut { *x = 42 }
fn hello() with IO { println("hi") }
```

### 3. Mutable References Use `&!`
```sio
fn increment(x: &!i32) with Mut { *x = *x + 1 }
var counter: i32 = 0
increment(&!counter)
```

### 4. No Unary Minus
```sio
let neg = 0 - 42       // ✅ correct
// let neg = -42        // ❌ WRONG
```

### 5. Bit Shifts Use u8
```sio
let shifted = byte >> 4u8     // ✅ correct
// let shifted = byte >> 4    // ❌ WRONG
```

### 6. No Rust Macros
```sio
assert(x == 5)        // ✅ Sounio
println("hello")       // ✅ Sounio
// assert!(x == 5)    // ❌ Rust macro
// println!("hello")  // ❌ Rust macro
```

## What Works (verified from tests/run-pass/)

### For-In Loops
```sio
// Source: tests/run-pass/for_in_loops.sio
for i in 0..5 { /* 0,1,2,3,4 */ }
for i in 0..=5 { /* 0,1,2,3,4,5 (inclusive) */ }
for x in arr { sum = sum + x }
for x in vec { sum = sum + x }
// break and continue work
```

### Function References (not closures)
```sio
// Source: tests/run-pass/closure_fn_ref.sio
fn square(x: i64) -> i64 { x * x }
fn apply(f: fn(i64) -> i64, x: i64) -> i64 { f(x) }

let f = square
let r = f(7)                    // 49
let r2 = apply(square, 5)      // 25

// ❌ Closure literals are BLOCKED: |x| x * 2
```

### Enums & Match
```sio
// Source: tests/run-pass/native_enum_basic.sio
enum Color { Red, Green, Blue }
let r = Color::Red

match c {
    Color::Red => 10
    Color::Green => 20
    _ => 0
}
```

### Tuple Destructuring
```sio
// Source: tests/run-pass/tuple_destructure_let.sio
let (a, b) = (1, 2)
let (x, (y, z)) = (10, (20, 30))
```

### Imports
```sio
// Source: tests/run-pass/import_basic_main.sio
use import_basic_a::{imported_add}
```

### impl Blocks
```sio
// Source: stdlib/collections/vec.sio
impl IntVec {
    fn new() -> IntVec { IntVec { data: [0; 4096], len: 0 } }
    fn push(self: &! IntVec, val: i64) { /* ... */ }
    fn len(self: &IntVec) -> i64 { self.len }
}
```

## Common Patterns (Copy-Paste Ready)

### Pattern 1: Array Processing
```sio
fn process_bytes(data: &[u8; 256], len: i32) with Panic {
    var i: i32 = 0
    while i < len {
        let byte = data[i as usize]
        i = i + 1
    }
}
```

### Pattern 2: Higher-Order Functions
```sio
// Source: tests/run-pass/closure_higher_order.sio
fn map4(arr: [i64; 4], f: fn(i64) -> i64) -> [i64; 4] with Mut, Panic, Div {
    var out: [i64; 4] = [0; 4]
    var i: i64 = 0
    while i < 4 { out[i] = f(arr[i]); i = i + 1 }
    out
}

let doubled = map4(data, dbl)
let sum_sq = fold4(map4(data, sq), 0, add)   // chained
```

### Pattern 3: Error Codes
```sio
fn safe_divide(a: f64, b: f64) -> (f64, i32) with Div, Panic {
    if b == 0.0 { (0.0, 1) }     // error
    else { (a / b, 0) }           // success
}
let (result, err) = safe_divide(10.0, 0.0)
```

### Pattern 4: Struct By-Value Update (JIT workaround)
```sio
struct Counter { value: i32 }
fn increment(c: Counter) -> Counter {
    Counter { value: c.value + 1 }
}
var c = Counter { value: 0 }
c = increment(c)    // reassign to propagate
```

### Pattern 5: Bare Array &! Mutation (explicit deref)
```sio
// Source: tests/run-pass/array_mut_ref.sio
fn fill(arr: &![i64; 8]) with Mut, Panic {
    (*arr)[0] = 99    // MUST use (*arr)[i] for bare arrays
}
var buf: [i64; 8] = [0, 0, 0, 0, 0, 0, 0, 0]
fill(&! buf)
```

## Effect Reference

| Effect | When | Example |
|--------|------|---------|
| `IO` | Print, file, env | `println("text")` |
| `Mut` | Mutate `&!` refs or arrays | `*x = 42`, `arr[i] = v` |
| `Div, Panic` | Division `/` or modulo `%` | `a / b` |
| `Panic` | Array bounds, asserts, casts | `arr[i]`, `assert()`, `as` |
| `Alloc` | Heap (rare) | Avoid |

## Checklist

1. No semicolons
2. `&!` not `&mut`, `var` not `let mut`
3. Effects declared (`with IO, Mut, Div, Panic`)
4. No Rust macros — `assert()` not `assert!()`, `println()` not `println!()`
5. No unary minus — `0 - x`
6. Bit shifts use `u8` — `x >> 4u8`
7. Array index cast — `arr[i as usize]`
8. Named fn refs, not closure literals

## Real Sounio to Study

- `tests/run-pass/hello.sio` — Hello world
- `tests/run-pass/for_in_loops.sio` — All for-in variants
- `tests/run-pass/closure_fn_ref.sio` — Function references
- `tests/run-pass/closure_higher_order.sio` — map, fold, any, all
- `tests/run-pass/native_enum_basic.sio` — Enums + match
- `tests/run-pass/array_mut_ref.sio` — Mutable array refs
- `stdlib/encoding/hex.sio` — Real stdlib code
- `stdlib/collections/vec.sio` — impl blocks, IntVec

---

**Remember**: Sounio ≠ Rust. When in doubt, check real `.sio` files.
