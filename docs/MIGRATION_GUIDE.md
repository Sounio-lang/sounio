<!-- docs:meta
topic_id: repo.docs.migration-guide
authority: repo_only
audience: users
last_validated: 2026-04-12
validated_by: human
source_of_truth: CHANGELOG.md
-->

> **Status**: Production | **Last validated**: 2026-04-12 | **Source**: `CHANGELOG.md`

# Sounio Migration Guide

How to update your code when Sounio changes between versions. Based on the [CHANGELOG](../CHANGELOG.md).

---

## 1.0.0-beta.4 → 1.0.0-beta.6 (2026-03-21)

### Enums and Match

Enums are now available. You can replace integer-based dispatch:

**Before:**
```sio
let kind = 0  // 0=circle, 1=square
if kind == 0 { area = pi * r * r }
else { area = s * s }
```

**After:**
```sio
enum Shape { Circle, Square }
match shape {
    Shape::Circle => pi * r * r
    Shape::Square => s * s
}
```

### Function References

Named function references now work. You can store and pass functions:

**Before:**
```sio
// Had to inline or use integer dispatch
```

**After:**
```sio
fn square(x: i64) -> i64 { x * x }
let f = square
let r = f(7)  // 49

fn apply(f: fn(i64) -> i64, x: i64) -> i64 { f(x) }
```

### Observe Effect

A new `Observe` effect is available for observer-inclusion patterns:

```sio
fn observe_value(x: f64) -> f64 with Observe {
    x
}
```

### `&expr` and `*expr` in Bootstrap

Reference (`&`) and dereference (`*`) operators are now available in bootstrap-compiled code.

---

## 1.0.0-beta.5 → 1.0.0-beta.6

No breaking language changes. Additive release with new stdlib modules (cybernetics, psychiatry, connectomics).

---

## 0.x → 1.0.0-beta.4

### Variables: `let mut` → `var`

**Before (pre-beta):**
```sio
let mut counter = 0
```

**After:**
```sio
var counter = 0
```

### References: `&mut` → `&!`

**Before (pre-beta):**
```sio
fn increment(x: &mut i32) { *x = *x + 1 }
increment(&mut counter)
```

**After:**
```sio
fn increment(x: &!i32) with Mut { *x = *x + 1 }
increment(&!counter)
```

### Semicolons Removed

All semicolons were removed from the language:

**Before:**
```sio
let x = 5;
fn add(a: i32, b: i32) -> i32 {
    return a + b;
}
```

**After:**
```sio
let x = 5
fn add(a: i32, b: i32) -> i32 {
    return a + b
}
```

Exception: semicolons inside array type syntax `[T; N]` are retained.

### Effects Are Now Mandatory

Functions with side effects must declare them:

**Before (pre-beta):**
```sio
fn hello() {
    println("hi")
}
```

**After:**
```sio
fn hello() with IO {
    println("hi")
}
```

### Assert: `assert!()` → `assert()`

**Before:**
```sio
assert!(x > 0)
```

**After:**
```sio
assert(x > 0)
```

---

## General Migration Tips

1. **Always check with `souc check`** after upgrading:
   ```bash
   "$SOUC_BIN" check my_code.sio
   ```

2. **Run the test suite** to catch regressions:
   ```bash
   make test
   ```

3. **Read `docs/compiler/KNOWN_LIMITATIONS.md`** before assuming a bug is new.

4. **Check `souc info`** to confirm which features your artifact supports.

5. **Use the lint tool** to catch Rust-isms:
   ```bash
   python3 scripts/dev/sounio-lint.py my_code.sio
   ```
