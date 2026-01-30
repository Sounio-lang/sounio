# Minimum Viable Sounio (MVS) — What Works Today

This guide tells you **exactly what you can use in Sounio right now** without running into compiler errors or unimplemented features.

Read this first. It supersedes aspiration in the other docs.

---

## TL;DR — What You Can Do NOW

✅ **Single-file programs** with pure functions, structs, and effects (`with IO`)
✅ **Basic math**: `sin()`, `cos()`, `sqrt()`, `exp()`, `log()`, `abs()`, `max()`, `min()` + all standard libm functions
✅ **FFI bindings** to C libraries via `@extern()`
✅ **Quaternion operations** (quat_mul, quat_conjugate, etc.)
✅ **Epistemic types** (Knowledge<T>) with confidence tracking
✅ **Effect system** (IO, Async, Alloc, Panic — with type-level tracking)
✅ **GPU kernels** (PTX/Metal backends)
✅ **Units of measure** (mg, mL, h, etc.)
✅ **Refinement types** (basic)

✅ **Module system** (`use` statements work for multi-file projects)
✅ **Forward references** (functions can be defined after use)
✅ **Tuple destructuring** (`let (x, y) = tuple` now supported)
✅ **Visibility modifiers** (`pub` keyword is enforced)

---

## What Changed in Phase 1

### 1. Math Functions Now Available

**Before**: Had to write manual FFI for every math function.

```sio
// OLD (required manual FFI):
@extern("sqrt")
fn sqrt(x: f64) -> f64 { ... }

let y = sqrt(4.0)
```

**After**: Math functions are in `stdlib/math/core.sio` and re-exported by default.

```sio
// NEW (just use it):
let y = sqrt(4.0)           // f64 version
let z = sqrt_f32(2.0)       // f32 version
let pi = PI                 // 3.14159...
let angle = deg_to_rad(90)  // Convert 90° to radians
```

**Available Functions**:
- Trigonometric: `sin()`, `cos()`, `tan()`, `asin()`, `acos()`, `atan()`, `atan2()`
- Exponential: `exp()`, `log()`, `log10()`, `log2()`, `pow()`
- Roots: `sqrt()`, `cbrt()`
- Rounding: `floor()`, `ceil()`, `round()`, `trunc()`
- Utilities: `abs()`, `max()`, `min()`, `deg_to_rad()`, `rad_to_deg()`
- Constants: `PI` (f64 and f32), `E` (f64 and f32)

All functions available in both **f64** (default) and **f32** variants (suffixed `_f32`).

### 2. Issue #16 Resolved

**The Problem**: Importing multiple stdlib modules caused "duplicate main()" errors.

**The Status**: ✅ **Already fixed** — No `fn main()` declarations in stdlib files. Module imports don't conflict.

---

## Complete Example: Single-File Program

Here's what a real, working Sounio program looks like today:

```sio
// File: my_program.sio
// This compiles and runs today

use qnn::mnist::*    // WAIT: Doesn't work yet (modules not implemented)

fn square(x: f64) -> f64 {
    return x * x
}

fn main() with IO {
    let x = 3.0
    let result = square(x)
    let sqrt_result = sqrt(result)

    println("x = {}")
    println(x)

    println("x² = {}")
    println(result)

    println("√(x²) = {}")
    println(sqrt_result)

    // Math available now
    let angle_rad = deg_to_rad(45.0)
    println("45° in radians = {}")
    println(angle_rad)

    println("sin(45°) = {}")
    println(sin(angle_rad))
}
```

**BUT WAIT**: The `use qnn::mnist::*` doesn't work. Module system isn't implemented.

---

## Known Limitations — What Doesn't Work

### 1. Module System (Not Implemented)

**Status**: Planned for v0.70.0, not available now.

❌ **These don't work**:
```sio
use std::qnn          // ❌ Module system not implemented
use math::*           // ❌ Use statements don't resolve
import data_loader    // ❌ No import keyword
```

**Workaround**: Everything in one file.

```sio
// ✅ Put all code in a single .sio file
// If it gets too large, split into multiple files but compile them together

// One file can be ~1000+ lines and still be fine
```

**When to expect**: Planned for v0.70.0 (Q1 2026 estimate based on roadmap)

### 2. Forward References (NOW WORKING ✅)

**Status**: Forward references within a single file are now fully supported.

✅ **This now works**:
```sio
fn main() with IO {
    helper()     // ✅ OK: forward reference resolved
}

fn helper() with IO {
    println("hello")
}
```

The resolver performs a two-pass analysis:

1. First pass: Collect all definitions
2. Second pass: Resolve references (including forward references)

### 3. Tuple Destructuring (NOW WORKING ✅)

**Problem**: Can't unpack tuples with `let (x, y) = ...` syntax.

✅ **This now works**:
```sio
let pair = (1, 2.0)
let (a, b) = pair    // ✅ Tuple destructuring supported!
```

✅ **Nested tuples also work**:
```sio
fn get_coords() -> (f64, f64) { (3.0, 4.0) }
let (x, y) = get_coords()    // ✅ Works

// Nested destructuring
let ((a, b), (c, d)) = ((1, 2), (3, 4))

// With wildcards
let (first, _) = (42, "ignored")
```

📝 **Note**: You can still use tuple field access if you prefer:

```sio
let pair = (1, 2.0)
let a = pair.0    // Access first element
let b = pair.1    // Access second element
```

### 4. Visibility Modifiers (Not Implemented)

**Status**: `pub` keyword is parsed but ignored.

```sio
pub fn my_function() { ... }  // 'pub' does nothing right now
fn my_function() { ... }       // Same as above
```

This matters when modules are implemented. For now, all declarations are "public" (no encapsulation possible).

### 5. Scientific Notation (NOW WORKING ✅)

**Status**: Fully supported.

✅ **These all work**:
```sio
let large = 1e10       // ✅ Integer with exponent
let small = 1.5e-3     // ✅ Float with negative exponent
let precise = 2.5E+3   // ✅ Uppercase E, positive exponent
let big = 6.022e23     // ✅ Avogadro's number
```

---

## What Actually Works: Complete Feature List

### ✅ Core Language (All Working)

- ✅ `let` and `var` bindings
- ✅ `fn` function definitions with `-> ReturnType`
- ✅ Effects: `fn f() -> T with IO { ... }`, `with Async`, `with Mut`, `with Alloc`, `with Panic`, `with Div`, `with Prob`, `with GPU`
- ✅ `struct` definitions with fields
- ✅ `impl` blocks with methods
- ✅ `if`, `if-else`, `else-if`, `else` control flow
- ✅ `for`, `while`, `loop` loops with `break` and `continue`
- ✅ `match` expressions with pattern matching
- ✅ Generics `<T>`, `<T: Trait>`
- ✅ References `&T` (shared) and `&!T` (exclusive/mutable)
- ✅ Arrays `[T; N]` and slices `[T]`
- ✅ Tuples `(T1, T2, ...)` with `.0`, `.1` field access
- ✅ Basic operators: `+`, `-`, `*`, `/`, `%`, `==`, `!=`, `<`, `>`, `<=`, `>=`, `&&`, `||`
- ✅ Comments: `//` and `/* ... */`

### ✅ Type System (All Working)

- ✅ Primitives: `i32`, `i64`, `f32`, `f64`, `bool`, `string`, `char`
- ✅ User-defined types: `struct`, `enum`
- ✅ Generic types: `Option<T>`, `Result<T, E>`
- ✅ First-class functions (closures)

### ✅ Effects System (All Working)

All effects track at the type level:

- ✅ `with IO` — File/console I/O
- ✅ `with Async` — Asynchronous operations
- ✅ `with Mut` — Mutable state
- ✅ `with Alloc` — Memory allocation
- ✅ `with Panic` — Exception handling
- ✅ `with GPU` — GPU computation
- ✅ `with Prob` — Probabilistic computation
- ✅ `with Div` — May divide by zero

Effect system enforces: if you call a function `with IO`, you must mark your function `with IO` too.

### ✅ Standard Library: Math (`stdlib/math/core.sio`)

All functions available in f64 and f32 variants:

**Trigonometric**:
- `sin()`, `cos()`, `tan()`, `asin()`, `acos()`, `atan()`, `atan2()`

**Exponential/Logarithmic**:
- `exp()`, `log()`, `log10()`, `log2()`, `pow()`, `sqrt()`, `cbrt()`

**Rounding**:
- `floor()`, `ceil()`, `round()`, `trunc()`

**Utilities**:
- `abs()`, `max()`, `min()`
- `deg_to_rad()`, `rad_to_deg()`

**Constants**:
- `PI`, `E` (both f64 and f32)

### ✅ Standard Library: I/O (`stdlib/io`)

- ✅ `println()` — print to stdout
- ✅ File I/O (basic)
- ✅ String formatting

### ✅ Standard Library: String (`stdlib/str`)

- ✅ String type
- ✅ String concatenation
- ✅ Basic string operations

### ✅ Quaternion Operations (`stdlib/math/quaternion` via `qnn`)

```sio
// Quaternion type and operations
let q1 = quat_new(w, x, y, z)
let q2 = quat_mul(q1, q1)        // Hamilton product
let conj = quat_conjugate(q1)
let magnitude = quat_magnitude(q1)
let normalized = quat_normalize(q1)
```

### ✅ Epistemic Types (`stdlib/epistemic`)

```sio
// Knowledge<T> represents uncertain data
let measurement: Knowledge<f64> = ...
let confidence = get_confidence(measurement)
let value = extract_value(measurement)

// Uncertainty propagates through operations
let result = measurement * 2.0  // Still Knowledge<f64>
```

### ✅ Units of Measure (`stdlib/units`)

```sio
let dose: mg = 500.0           // Milligrams
let volume: mL = 100.0         // Milliliters
let conc: mg/mL = dose / volume // Type-checked unit arithmetic
```

### ✅ GPU Programming (`stdlib/gpu`)

PTX (NVIDIA) and Metal (Apple) kernels compile when feature enabled.

### ✅ FFI (`stdlib/ffi`)

Bind to C functions:

```sio
@extern("strlen")
fn strlen(s: &string) -> i64 { ... }

let len = strlen("hello")
```

---

## What's Aspirational (Don't Use Yet)

These are documented in the guide but **don't actually work**:

### ❌ Module Organization

```sio
// Examples show this:
use std::qnn::mnist
use std::math::algebra

// But this doesn't work today!
```

### ❌ Complex Pattern Matching with Tuples

```sio
match value {
    (x, y) if x > y => ...    // Tuple destructuring in match not supported
    _ => ...
}
```

### ❌ Attribute Macros

```sio
#[test]
fn my_test() { ... }           // Attribute macros not implemented

#[inline]
fn small_function() { ... }    // Will ignore this
```

---

## FAQ: "Why Can't I...?"

### "Why can't I import from multiple modules?"

**Answer**: Module system not implemented. Everything must be in one file for now.

**When fixed**: v0.70.0 (planned Q1 2026)

### "Why must I define functions before I use them?"

**Answer**: ✅ **This is no longer a limitation!** Forward references now work.

```sio
fn main() with IO {
    helper()     // ✅ Works - forward reference resolved
}

fn helper() with IO {
    println("hello")
}
```

### "How do I use tuple destructuring?"

**Answer**: Tuple destructuring in `let` statements now works!

```sio
// Basic destructuring
let (x, y) = (1, 2)

// Nested tuples
let ((a, b), (c, d)) = ((1, 2), (3, 4))

// With wildcards
let (first, _) = (42, "ignored")
```

**Alternative**: If you prefer, you can still use tuple field access:

```sio
let pair = (1, 2)
let x = pair.0
let y = pair.1
```

**Note**: Tuple destructuring in `match` patterns is not yet supported.

---

## How to Write Code That Works Today

### Rule 1: Single File

Put everything in one `.sio` file. It can be up to ~1000+ lines without issues.

### Rule 2: Helper Functions First

```sio
// Put helpers at the top
fn sqrt_int_2(x: i32) -> f64 {
    return sqrt(x as f64)
}

fn process_data(values: &[f64]) -> f64 {
    let sum = 0.0
    // ... computation
    return sum
}

// Main logic at the bottom
fn main() with IO {
    let data = [1.0, 2.0, 3.0]
    let result = process_data(&data)
    println(result)
}
```

### Rule 3: Use Tuple Field Access (Not Destructuring)

```sio
// ❌ Don't do this:
let (x, y) = get_point()

// ✅ Do this:
let point = get_point()
let x = point.0
let y = point.1
```

### Rule 4: Manual Math FFI Only If Needed

```sio
// ✅ Standard math works:
let y = sqrt(4.0)

// If you need something exotic, use FFI:
@extern("hypot")
fn hypot(x: f64, y: f64) -> f64 { ... }

let distance = hypot(3.0, 4.0)
```

### Rule 5: Effect Annotations Are Required

```sio
// ❌ This won't compile:
fn read_file(path: string) -> string {
    // I/O here... but forgot the effect annotation
}

// ✅ This works:
fn read_file(path: string) -> string with IO {
    // I/O here is now legal
}

// And if you call it:
fn main() with IO {
    let content = read_file("data.txt")  // ✅ OK
}
```

---

## Next Steps

### For Users

1. **Learn what works**: Use this guide + `MV_CORE_CHECKLIST.md`
2. **Check examples**: Look in `tests/run-pass/` for compilable patterns
3. **Write single-file programs**: Focus on logic, not organization
4. **Use the compiler actively**: `cargo run -- check yourfile.sio` tells you what's wrong

### For the Project

**Phase 1 Complete** ✅
- Math stdlib created
- Issue #16 verified resolved
- Documentation updated (this file)

**Phase 2 Complete** ✅

- ✅ Tuple destructuring (completed)
- ✅ Forward references (two-pass name resolution — completed)
- ✅ Cross-module imports (path parsing fix — completed)

**Phase 3 Complete** ✅

- ✅ Scientific notation (already implemented in lexer)
- ✅ Type aliases (`type Name = Type;` syntax)
- ✅ Doc comments (`///` outer, `//!` inner)

**Phase 4 Complete** ✅

- ✅ Module system (`use` statements for multi-file projects)
- ✅ Nested module paths (`use utils::strings::concat;`)
- ✅ Visibility modifiers (`pub` enforcement)

**All known limitations from the original roadmap have been resolved!**

---

## See Also

- `docs/MV_CORE_CHECKLIST.md` — Formal definition of minimum viable Sounio
- `compiler/docs/KNOWN_LIMITATIONS.md` — Technical details on what doesn't work
- `docs/LLM_PROGRAMMING_GUIDE.md` — Comprehensive reference (mix of working + aspirational)
- `tests/run-pass/` — Real, working examples
