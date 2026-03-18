# Sounio Style Guide for LLMs

## 🚨 CRITICAL: Sounio is NOT Rust

**Sounio looks like Rust but has DIFFERENT syntax and semantics.**

## 1. Basic Syntax Rules

### ✅ CORRECT Sounio
```sio
// Function with effects
fn read_file(path: &str) -> string with IO, Panic {
    // ...
}

// Variables
let x = 42
var y: i32 = 10

// Arrays (FIXED SIZE ONLY)
var buffer: [u8; 256] = [0; 256]

// Structs
struct Point { x: f64, y: f64 }
```

### ❌ WRONG (Rust-isms)
```rust
// Rust - WRONG for Sounio
fn read_file(path: &str) -> String { // No 'with' effects
    // ...
}

let x = 42;  // Semicolon!
let y: Vec<u8> = vec![];  // No heap allocation!
```

## 2. Unique Sounio Features

### Effects System
```sio
// Effects are REQUIRED for certain operations
fn divide(a: f64, b: f64) -> f64 with Div, Panic {
    a / b  // Division requires Div effect
}

fn mutate(arr: &![u8; 256]) -> () with Mut {
    arr[0] = 42u8
}

fn print_hello() -> () with IO {
    print("Hello")
}
```

### Mutable References
```sio
// &! means mutable reference (TWO tokens: & then !)
fn increment(x: &!i32) with Mut {
    *x = *x + 1
}

// Usage
var counter: i32 = 0
increment(&!counter)
```

### Unit Literals
```sio
// Built-in unit support
let dose = 500_mg
let volume = 10.5_mL
let time = 2.3_sec
```

## 3. Type System

### Primitive Types
- Integers: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`
- Floats: `f32`, `f64`
- Boolean: `bool`
- Character: `char` (actually `i8` in arrays)

### Arrays (FIXED SIZE ONLY)
```sio
// Correct
var data: [u8; 256] = [0; 256]
let matrix: [f64; 16] = [0.0; 16]

// WRONG - No dynamic arrays
// var vec = Vec::new()  // Doesn't exist!
// let slice: &[u8]      // No slices!
```

### Strings
```sio
// Strings are fixed-size char arrays
var name: [i8; 64] = [0; 64]
name[0] = 'A' as i8
name[1] = 'B' as i8

// Or use string literals (compile-time)
let greeting = "Hello"  // Type depends on context
```

## 4. Control Flow

### No Semicolons
```sio
// Correct
if x > 0 {
    let y = x * 2
    print(y)
}

// Wrong
if x > 0; {  // NO semicolon!
    let y = x * 2;
}
```

### Loops
```sio
// While loop
var i = 0
while i < 10 {
    print(i)
    i = i + 1
}

// For loop with range
for i in 0..10 {
    print(i)
}
```

## 5. Functions

### Signature
```sio
// Full signature
pub fn process(data: &[u8; 256], len: i32) -> i32 with Mut, Div, Panic {
    // ...
}
```

### Effects Required For:
- `with Mut`: Mutating references (`&!`), array assignment
- `with Div`: Division (`/`), modulo (`%`)
- `with Panic`: Array bounds checks, overflow
- `with IO`: Printing, file operations
- `with Alloc`: Heap allocation (rare)

## 6. Common Patterns

### Array Processing
```sio
fn sum_array(arr: &[i32; 100]) -> i32 {
    var total = 0
    var i = 0
    while i < 100 {
        total = total + arr[i]
        i = i + 1
    }
    total
}
```

### Error Handling
```sio
// Return error codes, not exceptions
fn safe_divide(a: f64, b: f64) -> (f64, i32) with Div {
    if b == 0.0 {
        (0.0, 1)  // Error code 1
    } else {
        (a / b, 0)  // Success
    }
}
```

### Bit Operations
```sio
// Common in encoding/decoding
fn byte_to_hex(b: u8) -> (u8, u8) {
    let high = (b >> 4u8) & 15u8
    let low = b & 15u8
    (high, low)
}
```

## 7. FFI Integration

```sio
extern "C" {
    fn time(tloc: &!i64) -> i64
    fn print(s: &str) -> ()
}

fn get_timestamp() -> i64 with IO {
    var ts: i64 = 0
    time(&!ts)
    ts
}
```

## 8. Testing

```sio
// Inline tests (per CONVENTIONS.md)
fn test_addition() {
    let result = add(2, 3)
    assert(result == 5)
}

// Test helpers
fn check_near(a: f64, b: f64, tol: f64) -> bool {
    let d = a - b
    let ad = if d < 0.0 { 0.0 - d } else { d }
    ad < tol
}
```

## 9. Common Mistakes

### ❌ Array size mismatch
```sio
// Wrong
var small: [u8; 10] = [0; 256]  // Size mismatch!

// Correct
var correct: [u8; 256] = [0; 256]
```

### ❌ Missing effects
```sio
// Wrong - missing Mut effect
fn set_value(x: &!i32) {  // ERROR: Needs 'with Mut'
    *x = 42
}

// Correct
fn set_value(x: &!i32) with Mut {
    *x = 42
}
```

### ❌ Rust string methods
```sio
// Wrong - Rust methods
let s = "hello"
let len = s.len()  // No .len() method!

// Correct - fixed array
var s: [i8; 64] = [0; 64]
// Manually track length
```

## 10. Quick Reference

| Feature | Sounio Syntax | Notes |
|---------|--------------|-------|
| Function | `fn name() -> Type with Effects { }` | Effects required |
| Variable | `let x = val` or `var x: Type = val` | No semicolon |
| Array | `[Type; Size] = [init; Size]` | Fixed size only |
| Mutable ref | `&!x` | Two tokens: `&` then `!` |
| Division | `a / b` with Div effect | |
| Print | `print(x)` with IO effect | |
| Assert | `assert(condition)` | Panics if false |
| Loop | `while cond { }` or `for i in 0..n { }` | |

## 11. Verification Checklist

Before submitting Sounio code, check:
1. [ ] No semicolons (except array initializers)
2. [ ] Fixed-size arrays only
3. [ ] Effects specified (`with Mut, Div, Panic, IO`)
4. [ ] `&!` for mutable references
5. [ ] No Rust methods (`.len()`, `.push()`, etc.)
6. [ ] Unit literals supported (`500_mg`)
7. [ ] Error codes instead of exceptions
8. [ ] Inline tests present
9. [ ] Documentation per `CONVENTIONS.md`

## 12. Examples from Stdlib

See real Sounio in:
- `stdlib/encoding/hex.sio` - Hex encoding
- `stdlib/compiler/lexer/tokens.sio` - Token definitions
- `stdlib/test/helpers.sio` - Test utilities

## Remember: Sounio ≠ Rust

**When in doubt, look at existing Sounio code in the repository.**
The compiler is self-hosted, so all `.sio` files are REAL Sounio.
