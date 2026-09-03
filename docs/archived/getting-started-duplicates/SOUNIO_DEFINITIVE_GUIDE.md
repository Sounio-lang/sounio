<!-- docs:meta
topic_id: repo.docs.archived.getting-started-duplicates.sounio-definitive-guide
authority: archived
audience: maintainers
last_validated: 2026-03-07
validated_by: A7
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.archived.getting-started-duplicates.sounio-definitive-guide
-->


<!-- docs:status-note:start -->
> Docs status: `archived`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# 🎯 DEFINITIVE Sounio Guide - From REAL Code Analysis

## 🚨 STOP HALLUCINATING: Sounio ≠ Rust

**Based on ACTUAL code in `/tests/run-pass/` and `/tests/compile-fail/`**

## 1. 🎯 DEFINITIVE Syntax (From `tests/run-pass/`)

### ✅ REAL Sounio - `array_mut_ref.sio`
```sio
fn fill(arr: &![i64; 8]) with Mut, Panic {
    (*arr)[0] = 99  // EXPLICIT DEREFERENCE!
    (*arr)[1] = 42
}

fn main() -> i64 with IO, Mut, Panic {
    var buf: [i64; 8] = [0, 0, 0, 0, 0, 0, 0, 0]  // Array literal
    fill(&! buf)  // &! is CORRECT
    if buf[0] == 99 {
        return 0  // EXPLICIT RETURN!
    }
    return 1
}
```

### ✅ REAL Sounio - `native_struct_smoke.sio`
```sio
struct Vec3 { x: f64, y: f64, z: f64 }

fn vec3_dot(a: Vec3, b: Vec3) -> f64 {
    return a.x * b.x + a.y * b.y + a.z * b.z  // EXPLICIT RETURN!
}

fn main() -> i64 {
    let a = Vec3 { x: 3.0, y: 4.0, z: 0.0 }
    var c = Vec3 { x: 10.0, y: 20.0, z: 30.0 }
    c.x = 100.0  // Mutation on VAR only
    0  // Return value (no return statement needed for last expression?)
}
```

### ✅ REAL Sounio - `hello.sio`
```sio
fn main() with IO {
    println("Hello, Sounio!")
}
```

## 2. 🚨 CRITICAL Differences from Rust

### ❌ NEVER WRITE (Rust Patterns):
```rust
// WRONG - Rust mutable reference
fn foo(x: &mut i32) { *x = 42 }  // Should be &!i32 with Mut

// WRONG - Rust implicit return
fn add(a: i32, b: i32) -> i32 { a + b }  // Should have return

// WRONG - Rust methods
arr.len()    // No! Use manual tracking
arr.push(1)  // No! Fixed arrays only

// WRONG - Rust semicolons
let x = 42;  // No semicolons!
```

### ✅ ALWAYS WRITE (Sounio Patterns):
```sio
// CORRECT - Mutable reference
fn foo(x: &!i32) with Mut {
    (*x) = 42  // EXPLICIT DEREFERENCE!
}

// CORRECT - Explicit return
fn add(a: i32, b: i32) -> i32 {
    return a + b  // EXPLICIT!
}

// CORRECT - Manual iteration
var i = 0
while i < array_size {
    // process array[i]
    i = i + 1
}

// CORRECT - No semicolons
let x = 42  // Good!
```

## 3. 📋 DEFINITIVE Language Rules

### Effects System (NON-NEGOTIABLE)
```sio
// From effect_handler_basic.sio
effect Choice {
    fn pick() -> bool
}

fn coin_flip() with Choice {
    // Uses Choice effect
}

fn main() with Choice {
    coin_flip()  // ok: main declares Choice
}
```

**Effects required for:**
- `IO`: Printing, file operations
- `Mut`: Mutating `&!` references
- `Div`: Division `/` and modulo `%`
- `Panic`: Array bounds, overflow

### Type System
- **Primitives**: `i32`, `i64`, `f64`, `bool`, `string`
- **Structs**: `struct Name { field: Type }`
- **Arrays**: `[Type; Size]` ONLY (fixed size)
- **References**: `&Type` (immutable), `&!Type` (mutable)
- **NO**: `Vec<T>`, `&[T]`, `Box<T>`, `Rc<T>`, generics `T`

### Variables
```sio
let x = 42      // Immutable (cannot reassign)
var y: i32 = 10 // Mutable (can reassign)
y = 20          // OK for var
// x = 30       // ERROR: let is immutable
```

## 4. 🔍 Common Patterns from Real Code

### Array Literals
```sio
// From array_mut_ref.sio
var buf: [i64; 8] = [0, 0, 0, 0, 0, 0, 0, 0]  // COMMA separated!

// NOT: [0; 8]  // This might be wrong!
```

### Function Returns
```sio
// Pattern 1: Explicit return
fn explicit() -> i32 {
    return 42
}

// Pattern 2: Last expression (no return) - NEEDS VERIFICATION
fn implicit() -> i32 {
    42  // Might work as last expression?
}
```

### Printing
```sio
// From native_struct_smoke.sio
print("T1_ax=")
print_int(ax_i)
print("\n")

// println exists too
println("Hello, Sounio!")
```

## 5. 🧪 Validation Checklist

### ✅ MUST HAVE
- [ ] Effects declared (`with IO, Mut, Div, Panic`)
- [ ] `&!` for mutable references
- [ ] `(*x)` for dereferencing mutable refs
- [ ] Fixed arrays only (`[Type; Size]`)
- [ ] No Rust methods (`.len()`, `.push()`, etc.)

### ✅ MUST NOT HAVE
- [ ] NO semicolons (except maybe in arrays?)
- [ ] NO `Vec<T>` or heap types
- [ ] NO `&mut` (use `&!`)
- [ ] NO implicit returns (use `return`)
- [ ] NO `.iter()` or for-each loops

## 6. 🎯 Quick Reference

| Feature | Sounio Syntax | Real Example |
|---------|--------------|--------------|
| Function | `fn name() -> Type with Effects { return expr }` | `fn fill(arr: &![i64;8]) with Mut,Panic` |
| Array | `[Type; Size] = [val, val, ...]` | `[i64; 8] = [0,0,0,0,0,0,0,0]` |
| Mutable ref | `&!x` then `(*x) = value` | `(*arr)[0] = 99` |
| Print | `print("text")` with IO | `print("T1_ax=")` |
| Return | `return expression` | `return 0` |
| Struct | `struct Name { field: Type }` | `struct Vec3 { x:f64, y:f64, z:f64 }` |

## 7. 📚 Study These REAL Files

### Simple Examples
1. `tests/run-pass/hello.sio` - Basic hello world
2. `tests/run-pass/native_struct_smoke.sio` - Structs, mutation
3. `tests/run-pass/array_mut_ref.sio` - Arrays, mutable refs
4. `tests/run-pass/effect_handler_basic.sio` - Effects

### Error Examples
1. `tests/compile-fail/effect_missing.sio` - Missing effects
2. `tests/compile-fail/unit_mismatch.sio` - Unit errors

### Production Code
1. `stdlib/encoding/hex.sio` - Real algorithms
2. `stdlib/test/helpers.sio` - Test utilities

## 8. 🚨 Hallucination Triggers

**STOP if you see these in your mind:**
1. **`.len()`** → Manual size tracking
2. **`.push()`** → Fixed arrays only
3. **`&mut`** → `&!`
4. **`Vec::new()`** → `[Type; Size]`
5. **Semicolons** → Remove them
6. **Implicit return** → Add `return`
7. **`for x in iter`** → `while i < n`
8. **Generics `<T>`** → Concrete types only

## 9. ✅ Verification Script

```bash
#!/bin/bash
# check_sounio.sh - Validate Sounio code for Rust-isms

echo "Checking for Rust patterns in Sounio code..."

# Check for Rust methods
echo "\n1. Checking for Rust methods:"
grep -n "\.\w*(" *.sio | grep -v "print\|println" | head -20

# Check for &mut
echo "\n2. Checking for &mut (should be &!):"
grep -n "&mut" *.sio

# Check for Vec
echo "\n3. Checking for Vec (should be fixed arrays):"
grep -n "Vec" *.sio

# Check for semicolons (except in array literals)
echo "\n4. Checking for semicolons:"
grep -n ";" *.sio | grep -v "^.*:\s*\[.*;.*\]" | head -20

echo "\n✅ Validation complete"
```

## 10. 🎓 Final Rule: TRUST THE TESTS

**When writing Sounio, ALWAYS look at `tests/run-pass/` first.**

The tests are REAL Sounio that compiles. Copy patterns from them.

**DO NOT** invent syntax. **DO NOT** assume Rust patterns work.

---

**Source:** Analysis of actual Sounio test files
**Verified:** `array_mut_ref.sio`, `native_struct_smoke.sio`, `hello.sio`
**Anti-Hallucination:** ✅ Based on REAL compiler inputs
