# Sounio Quick Start for LLMs

**TL;DR**: Sounio is NOT Rust. Fixed arrays. No semicolons. Effects required. Study real `.sio` files.

## 30-Second Comparison

| Thing | Rust | Sounio |
|-------|------|--------|
| Function | `fn add(a: i32) -> i32 { }` | `fn add(a: i32) -> i32 { }` |
| Same? | ✓ | ✓ |
| Effects | N/A | `fn div(a: f64, b: f64) -> f64 with Div { a / b }` |
| Variable | `let x = 5;` | `let x = 5` |
| Semicolon? | ✓ (required) | ✗ (forbidden in expressions) |
| Arrays | `Vec<u8>` (heap) or `[u8; N]` (fixed) | `[u8; N]` (ONLY fixed) |
| Mutable ref | `&mut x` | `&!x` |
| Strings | `String` (owned) or `&str` (slice) | `[i8; N]` (fixed arrays) |
| Error | `Result<T, E>` | Return error codes: `(T, i32)` |

## Must-Know Rules

### 1. NO Semicolons (except in array init)
```sio
// ✅ CORRECT
let x = 5
let y = x + 3

// ❌ WRONG
let x = 5;  // ERROR!
```

### 2. Arrays ONLY Fixed-Size
```sio
// ✅ CORRECT - fixed size
var buffer: [u8; 256] = [0; 256]
var data: [f64; 100] = [0.0; 100]

// ❌ WRONG - no dynamic arrays
// var vec: Vec<u8> = vec![0, 1, 2]  // Doesn't exist!
// let slice: &[u8] = &[1, 2, 3]     // No slices!
```

### 3. Mutable References Use `&!` (two tokens)
```sio
// ✅ CORRECT
fn increment(x: &!i32) with Mut {
    *x = *x + 1
}

var counter: i32 = 0
increment(&!counter)

// ❌ WRONG
// fn increment(x: &mut i32) { }  // Rust syntax!
```

### 4. Effects Are Required and Explicit
```sio
// ✅ CORRECT - effects listed
fn divide(a: f64, b: f64) -> f64 with Div, Panic {
    a / b
}

fn mutate(x: &!i32) with Mut {
    *x = 42
}

// ❌ WRONG - missing effects
// fn divide(a: f64, b: f64) -> f64 {  // Missing 'with Div'
//     a / b
// }
```

### 5. No Rust Methods
```sio
// ❌ WRONG - Rust methods don't exist
let len = my_string.len()
let item = array.first()
let doubled = array.iter().map(|x| x * 2).collect()

// ✅ CORRECT - manual loops + parameter passing
var i = 0
while i < len {
    process(array[i])
    i = i + 1
}
```

### 6. Type Casting with `as`
```sio
// ✅ CORRECT
let b = 42u8
let i = b as i32
let idx = 5i32 as usize

// ❌ WRONG
// let idx = (5i32) as usize  // Extra parens unneeded (usually)
```

## Common Patterns (Copy-Paste Ready)

### Pattern 1: Byte Array Loop
```sio
fn process_bytes(data: &[u8; 256], len: i32) -> () {
    var i: i32 = 0
    while i < len {
        let byte = data[i as usize]
        // ... do something with byte
        i = i + 1
    }
}
```

### Pattern 2: Fixed-Size Output
```sio
fn hex_encode(input: &[u8; 256], in_len: i32, out: &![u8; 512]) -> i32 with Mut, Div, Panic {
    if in_len < 0 || in_len > 256 { return 0 - 1 }

    var i: i32 = 0
    while i < in_len {
        let byte = input[i as usize]
        out[(i * 2) as usize] = hex_nibble(byte >> 4u8)
        out[(i * 2 + 1) as usize] = hex_nibble(byte & 15u8)
        i = i + 1
    }

    in_len * 2
}
```

### Pattern 3: Error Code (Not Exception)
```sio
fn safe_divide(a: f64, b: f64) -> (f64, i32) with Div, Panic {
    if b == 0.0 {
        (0.0, 1)  // Error: return (bad_value, error_code)
    } else {
        (a / b, 0)  // Success: (result, 0)
    }
}

// Caller:
let (result, err) = safe_divide(10.0, 0.0)
if err != 0 {
    // Handle error
}
```

### Pattern 4: Inline Test (Required for Stdlib)
```sio
fn test_hex_encode() {
    var out: [u8; 512] = [0; 512]
    var data: [u8; 256] = [0; 256]
    data[0] = 255u8  // 0xFF

    let len = hex_encode(&data, 1, &!out)

    assert(len == 2)
    assert(out[0] == 102u8)  // 'f'
    assert(out[1] == 102u8)  // 'f'
}
```

### Pattern 5: Struct with By-Value Update (Avoid JIT &! Bug)
```sio
struct TestResult {
    pass: i32,
    fail: i32,
}

fn test_pass(r: TestResult) -> TestResult {
    TestResult { pass: r.pass + 1, fail: r.fail }
}

fn test_fail(r: TestResult) -> TestResult {
    TestResult { pass: r.pass, fail: r.fail + 1 }
}

// Usage:
var t = TestResult { pass: 0, fail: 0 }
t = test_pass(t)
t = test_fail(t)
```

## Effect Reference

When do you need `with`?

| Effect | When | Example |
|--------|------|---------|
| `with Mut` | Mutate `&!` refs or arrays | `arr[i] = 42` |
| `with Div, Panic` | Division `/` or modulo `%` | `a / b` (always needs Panic too) |
| `with Panic` | Array bounds or asserts | `arr[i]`, `assert(x == y)` |
| `with IO` | Print, file, env | `print(x)` |
| `with Alloc` | Heap (rare) | Avoid |

## Gotchas

### Gotcha 1: Negative Numbers
```sio
// ✅ CORRECT - use 0 - x
let neg = 0 - 42

// ❌ WRONG - no unary minus
// let neg = -42  // Parse error!
```

### Gotcha 2: Bit Shift Requires u8 Operand
```sio
// ✅ CORRECT
let shifted = byte >> 4u8
let masked = byte & 15u8

// ❌ WRONG
// let shifted = byte >> 4   // Type mismatch! Must be u8
```

### Gotcha 3: Array Size Must Match
```sio
// ✅ CORRECT
var buf: [u8; 256] = [0; 256]

// ❌ WRONG - size mismatch
// var buf: [u8; 10] = [0; 256]  // ERROR!
```

### Gotcha 4: String Literals Are Tricky
```sio
// Works in some contexts (compile-time known size)
let msg = "Hello"

// For variable strings, use fixed array
var name: [i8; 64] = [0; 64]
name[0] = 65u8  // 'A'
```

### Gotcha 5: No Closures or Callbacks
```sio
// ✅ CORRECT - pass function pointers (Sounio uses different approach)
fn map_operation(x: i32) -> i32 { x * 2 }

// ❌ WRONG - closures don't exist
// let f = |x| x * 2  // Syntax error!
```

## How to Check Your Code

Before running Sounio:

1. **No semicolons** in expressions? (`let x = 5` not `let x = 5;`)
2. **All arrays fixed-size?** (`[u8; 256]` not `Vec<u8>`)
3. **Effects listed** for Mut/Div/Panic/IO? (`fn foo() with Mut { }`)
4. **&! for mutable refs?** (not `&mut`)
5. **No Rust methods?** (no `.len()`, `.push()`, `.iter()`, etc.)
6. **Error codes not exceptions?** (return `(result, error_code)`)
7. **Inline tests present?** (for stdlib modules)

## Real Sounio to Study

- [stdlib/encoding/hex.sio](stdlib/encoding/hex.sio) — Hex encoding (real production code)
- [stdlib/test/helpers.sio](stdlib/test/helpers.sio) — Test utilities
- [stdlib/compiler/lexer/tokens.sio](stdlib/compiler/lexer/tokens.sio) — Token definitions
- [stdlib/compiler/parser/expr.sio](stdlib/compiler/parser/expr.sio) — Parser implementation

## When You're Stuck

1. **Look at the error message** — Sounio is explicit
2. **Search existing `.sio` files** for patterns
3. **Check SOUNIO_STYLE_GUIDE.md** for detailed rules
4. **Read CONVENTIONS.md** for stdlib patterns

---

**Remember**: Sounio ≠ Rust. The syntax looks similar, but the semantics are different.
**Always check real `.sio` files in the repo before guessing.**
