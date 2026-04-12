<!-- docs:meta
topic_id: repo.docs.guide.sounio-gotchas
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A5
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.guide.sounio-gotchas
-->

> **Status**: Production | **Last validated**: 2026-03-07 | **Source**: tests/run-pass/

# Sounio Gotchas & Common Mistakes

These are the mistakes LLMs (and humans) make writing Sounio. Learn them.

**Full syntax ref**: [docs/LLM_PROGRAMMING_GUIDE.md](../LLM_PROGRAMMING_GUIDE.md)

## 1. SEMICOLONS - The #1 Mistake

### The Mistake
```sio
// ❌ COMPLETELY WRONG
let x = 5;
let y = 10;
fn foo() -> i32 {
    let result = x + y;
    result;
}
```

### Why It's Wrong
- Sounio expressions don't end with semicolons
- The parser treats `;` as statement separator, not terminator
- Result type becomes `()` if you add `;`

### The Fix
```sio
// ✅ CORRECT
let x = 5
let y = 10
fn foo() -> i32 {
    let result = x + y
    result
}
```

---

## 2. `&mut` vs `&!`

### The Mistake
```sio
// ❌ WRONG - Rust syntax
fn increment(x: &mut i32) with Mut {
    *x = *x + 1
}
var counter: i32 = 0
increment(&mut counter)
```

### The Fix
```sio
// ✅ CORRECT - Sounio uses &! (two tokens: & then !)
fn increment(x: &!i32) with Mut {
    *x = *x + 1
}
var counter: i32 = 0
increment(&!counter)
```

---

## 3. MISSING EFFECTS

### The Mistake
```sio
// ❌ WRONG - missing Mut effect
fn set_value(x: &!i32) {
    *x = 42  // ERROR: mutation without 'with Mut'
}

// ❌ WRONG - missing Div effect
fn divide(a: f64, b: f64) -> f64 {
    a / b  // ERROR: division without 'with Div, Panic'
}

// ❌ WRONG - missing IO effect
fn say_hello() {
    println("Hello")  // ERROR: IO without 'with IO'
}
```

### The Fix
```sio
// ✅ CORRECT
fn set_value(x: &!i32) with Mut { *x = 42 }
fn divide(a: f64, b: f64) -> f64 with Div, Panic { a / b }
fn say_hello() with IO { println("Hello") }
```

### Effect Reference
| Effect | Required for |
|--------|-------------|
| `Mut` | `&!` mutation, array assignment |
| `Div, Panic` | Division `/`, modulo `%` |
| `Panic` | Array access, `assert()`, `as` casts |
| `IO` | `print()`, `println()`, file ops |

---

## 4. RUST MACROS DON'T EXIST

### The Mistake
```sio
// ❌ WRONG - Rust macros
assert!(x == 5)
println!("hello {}", name)
vec![1, 2, 3]
```

### The Fix
```sio
// ✅ CORRECT - Sounio functions (no !)
assert(x == 5)
println("hello")
print(name)
```

---

## 5. NEGATIVE NUMBERS & UNARY MINUS

### The Mistake
```sio
// ❌ WRONG - unary minus doesn't exist
let neg = -42
let result = -x
```

### The Fix
```sio
// ✅ CORRECT
let neg = 0 - 42
let result = 0 - x
let value = a - (0 - b)  // = a + b
```

---

## 6. BIT SHIFT REQUIRES u8 OPERAND

### The Mistake
```sio
// ❌ WRONG - shift amount must be u8
let shifted = byte >> 4       // ERROR: 4 is i32!
```

### The Fix
```sio
// ✅ CORRECT
let shifted = byte >> 4u8
let masked = byte & 15u8
let high = (byte >> 4u8) & 15u8
```

---

## 7. ARRAY SIZE MISMATCHES

### The Mistake
```sio
// ❌ WRONG - initialization size doesn't match type
var small_buffer: [u8; 10] = [0; 256]  // ERROR: 256 != 10!
```

### The Fix
```sio
// ✅ CORRECT
var buffer: [u8; 256] = [0; 256]
```

---

## 8. CLOSURE LITERALS vs FUNCTION REFERENCES

### The Mistake
```sio
// ❌ WRONG - closure literals are BLOCKED
let doubled = numbers.iter().map(|x| x * 2).collect()
let callback = |x| { x + 1 }
```

### Why It's Wrong
- Sounio has NO closure literals (`|x| expr`)
- But **named function references DO work** as first-class values

### The Fix
```sio
// ✅ CORRECT - named function references (verified: closure_fn_ref.sio)
fn double(x: i64) -> i64 { x * 2 }
fn apply(f: fn(i64) -> i64, x: i64) -> i64 { f(x) }

let f = double            // store fn ref in variable
let r = f(7)              // call through variable: 14
let r2 = apply(double, 5) // pass as argument: 10

// Higher-order patterns work (verified: closure_higher_order.sio)
fn map4(arr: [i64; 4], f: fn(i64) -> i64) -> [i64; 4] with Mut, Panic, Div {
    var out: [i64; 4] = [0; 4]
    var i: i64 = 0
    while i < 4 { out[i] = f(arr[i]); i = i + 1 }
    out
}
let doubled = map4(data, double)
```

---

## 9. METHODS ON CORE TYPES

### The Mistake
```sio
// ❌ WRONG - core types have no methods
let text = "hello"
let len = text.len()
let upper = text.to_uppercase()
let first = arr.first()
```

### Why It's Wrong
- Core types (`i32`, `[T;N]`, string literals) have no methods
- But **stdlib types DO have methods** via `impl` blocks

### The Fix
```sio
// For core types: manual loops and functions
var i = 0
while i < len {
    process(array[i as usize])
    i = i + 1
}

// For stdlib types: impl methods work (verified: stdlib/collections/vec.sio)
impl IntVec {
    fn len(self: &IntVec) -> i64 { self.len }
    fn push(self: &! IntVec, val: i64) { /* ... */ }
}
```

---

## 10. TYPE CASTING WITH `as`

### The Mistake
```sio
// ❌ WRONG - missing casts
let i: i32 = 5
let arr: [u8; 256] = [0; 256]
let val = arr[i]     // ERROR: needs [usize] not [i32]
```

### The Fix
```sio
// ✅ CORRECT
let val = arr[i as usize]
let u: u8 = i as u8
```

---

## 11. EXCEPTION HANDLING DOESN'T EXIST

### The Mistake
```sio
// ❌ WRONG
try {
    let x = risky_operation()
} catch (error) {
    print("Error!")
}
```

### The Fix
```sio
// ✅ CORRECT - return error codes
fn divide(a: f64, b: f64) -> (f64, i32) with Div, Panic {
    if b == 0.0 { (0.0, 1) }
    else { (a / b, 0) }
}
let (result, err) = divide(10.0, 0.0)
if err != 0 { println("division error") }
```

---

## 12. BARE ARRAY `&!` MUTATION BUG

### The Mistake
```sio
// ❌ BUG - interpreter doesn't propagate bare array mutations
fn sort_broken(arr: &![i64; 10000]) with Mut {
    arr[0] = 99  // mutation invisible to caller!
}
```

### Why It's Wrong
- Known interpreter bug: bare `&![T; N]` mutations don't propagate
- Struct wrapper pattern propagates correctly

### The Fix
```sio
// ✅ CORRECT - wrap in struct
struct SortBuf { data: [i64; 10000] }
fn sort(b: &! SortBuf) with Mut {
    b.data[0] = 99  // works correctly
}

// Also correct - explicit deref (works in JIT)
// Source: tests/run-pass/array_mut_ref.sio
fn fill(arr: &![i64; 8]) with Mut, Panic {
    (*arr)[0] = 99   // explicit deref
}
```

---

## 13. STRING MANIPULATION

### The Mistake
```sio
// ❌ WRONG - strings are not dynamic objects
let greeting = "Hello"
greeting.push_str(" World")
let upper = greeting.to_uppercase()
```

### The Fix
```sio
// ✅ CORRECT - string literals for output
println("Hello, World!")

// For mutable string data: fixed-size byte arrays
var greeting: [i8; 64] = [0; 64]
greeting[0] = 72i8   // 'H'
greeting[1] = 101i8  // 'e'
```

---

## 14. FFI LIMITED TO MATH FUNCTIONS

### The Mistake
```sio
// ❌ WRONG - integer FFI silently terminates
extern "C" { fn malloc(size: i64) -> i64 }
extern "C" { fn getpid() -> i32 }
```

### Why It's Wrong
- Only `f64→f64` and `(f64,f64)→f64` FFI works in JIT
- Integer FFI (`malloc`, `getpid`, etc.) **silently terminates** the program

### The Fix
```sio
// ✅ CORRECT - supported FFI functions
extern "C" {
    fn sqrt(x: f64) -> f64
    fn sin(x: f64) -> f64
    fn pow(x: f64, y: f64) -> f64
}
// Full list: sqrt sin cos tan exp log floor ceil atan sinh cosh tanh
//            asin acos cbrt round log2 log10 pow atan2
```

---

## Checklist Before Submitting

- [ ] No semicolons in function bodies
- [ ] `&!` not `&mut`, `var` not `let mut`
- [ ] All effects declared (`with Mut, Div, Panic, IO`)
- [ ] No Rust macros — `assert()` not `assert!()`, `println()` not `println!()`
- [ ] No unary minus — use `0 - x`
- [ ] Bit shifts use `u8` — `x >> 4u8`
- [ ] Array sizes match — `[u8; 256] = [0; 256]`
- [ ] Type casts explicit — `i as usize`
- [ ] Named fn refs, not closure literals
- [ ] Error codes not exceptions
- [ ] Bare array `&!` mutations: use struct wrapper or explicit `(*arr)[i]`
- [ ] FFI: only math functions (`sqrt`, `sin`, `pow`, etc.)

---

**TL;DR**: Sounio ≠ Rust. Study `tests/run-pass/` for verified examples. Read [docs/LLM_PROGRAMMING_GUIDE.md](../LLM_PROGRAMMING_GUIDE.md) for full syntax reference.
