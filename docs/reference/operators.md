# Sounio Operators Reference

This document provides a complete reference for all operators in the Sounio programming language, including their precedence, associativity, and usage.

## Operator Precedence Table

Operators are listed from highest precedence (binds tightest) to lowest precedence (binds loosest).

| Precedence | Operator(s) | Description | Associativity |
|------------|-------------|-------------|---------------|
| 15 | `()` `[]` `.` `::` | Grouping, indexing, field access, path | Left-to-right |
| 14 | `!` `~` `-` `&` `&!` `*` | Unary operators | Right-to-left |
| 13 | `as` | Type cast | Left-to-right |
| 12 | `*` `/` `%` | Multiplication, division, remainder | Left-to-right |
| 11 | `+` `-` | Addition, subtraction | Left-to-right |
| 10 | `<<` `>>` | Bit shifts | Left-to-right |
| 9 | `&` | Bitwise AND | Left-to-right |
| 8 | `^` | Bitwise XOR | Left-to-right |
| 7 | `\|` | Bitwise OR | Left-to-right |
| 6 | `==` `!=` `<` `>` `<=` `>=` | Comparisons | Left-to-right |
| 5 | `&&` | Logical AND | Left-to-right |
| 4 | `\|\|` | Logical OR | Left-to-right |
| 3 | `..` `..=` | Range operators | Non-associative |
| 2 | `++` `+-` | Concatenation, uncertainty | Left-to-right |
| 1 | `=` `+=` `-=` `*=` `/=` `%=` `&=` `\|=` `^=` `<<=` `>>=` | Assignment | Right-to-left |

---

## Arithmetic Operators

### Binary Arithmetic

| Operator | Name | Description | Example |
|----------|------|-------------|---------|
| `+` | Addition | Add two numbers | `5 + 3` = `8` |
| `-` | Subtraction | Subtract two numbers | `5 - 3` = `2` |
| `*` | Multiplication | Multiply two numbers | `5 * 3` = `15` |
| `/` | Division | Divide two numbers | `15 / 3` = `5` |
| `%` | Remainder | Remainder after division | `17 % 5` = `2` |

### Unary Arithmetic

| Operator | Name | Description | Example |
|----------|------|-------------|---------|
| `-` | Negation | Negate a number | `-5` |
| `+` | Positive | Explicit positive (no-op) | `+5` |

### Usage Notes

```sio
// Integer division truncates toward zero
let result = 7 / 3     // 2, not 2.333...

// Remainder preserves sign of dividend
let r1 = 17 % 5        // 2
let r2 = -17 % 5       // -2
let r3 = 17 % -5       // 2

// Float division produces float result
let f = 7.0 / 3.0      // 2.333...
```

---

## Comparison Operators

| Operator | Name | Description | Example |
|----------|------|-------------|---------|
| `==` | Equal | True if operands are equal | `5 == 5` is `true` |
| `!=` | Not equal | True if operands are not equal | `5 != 3` is `true` |
| `<` | Less than | True if left is less than right | `3 < 5` is `true` |
| `>` | Greater than | True if left is greater than right | `5 > 3` is `true` |
| `<=` | Less or equal | True if left is at most right | `5 <= 5` is `true` |
| `>=` | Greater or equal | True if left is at least right | `5 >= 5` is `true` |

### Chained Comparisons

Unlike some languages, Sounio does NOT support chained comparisons:

```sio
// This does NOT work as you might expect
// let valid = 0 < x < 10  // ERROR

// Use logical AND instead
let valid = 0 < x && x < 10  // Correct
```

---

## Logical Operators

| Operator | Name | Description | Example |
|----------|------|-------------|---------|
| `&&` | Logical AND | True if both operands are true | `true && false` is `false` |
| `\|\|` | Logical OR | True if either operand is true | `true \|\| false` is `true` |
| `!` | Logical NOT | Negates the boolean | `!true` is `false` |

### Short-Circuit Evaluation

Logical operators use short-circuit evaluation:

```sio
// Right side not evaluated if left side determines result
let result = false && expensive_computation()  // expensive_computation not called
let result = true || expensive_computation()   // expensive_computation not called
```

---

## Bitwise Operators

### Binary Bitwise

| Operator | Name | Description | Example |
|----------|------|-------------|---------|
| `&` | Bitwise AND | AND each bit | `0b1100 & 0b1010` = `0b1000` |
| `\|` | Bitwise OR | OR each bit | `0b1100 \| 0b1010` = `0b1110` |
| `^` | Bitwise XOR | XOR each bit | `0b1100 ^ 0b1010` = `0b0110` |
| `<<` | Left shift | Shift bits left | `0b0001 << 2` = `0b0100` |
| `>>` | Right shift | Shift bits right | `0b1000 >> 2` = `0b0010` |

### Unary Bitwise

| Operator | Name | Description | Example |
|----------|------|-------------|---------|
| `~` | Bitwise NOT | Invert all bits | `~0b1100` = `0b...0011` |

### Bitwise Examples

```sio
// Setting a bit
let flags = flags | (1 << bit_position)

// Clearing a bit
let flags = flags & ~(1 << bit_position)

// Toggling a bit
let flags = flags ^ (1 << bit_position)

// Checking a bit
let is_set = (flags & (1 << bit_position)) != 0
```

---

## Assignment Operators

### Simple Assignment

| Operator | Description | Example |
|----------|-------------|---------|
| `=` | Assign value | `x = 5` |

### Compound Assignment

| Operator | Equivalent | Example |
|----------|------------|---------|
| `+=` | `a = a + b` | `x += 5` |
| `-=` | `a = a - b` | `x -= 5` |
| `*=` | `a = a * b` | `x *= 5` |
| `/=` | `a = a / b` | `x /= 5` |
| `%=` | `a = a % b` | `x %= 5` |
| `&=` | `a = a & b` | `x &= 0xFF` |
| `\|=` | `a = a \| b` | `x \|= 0x01` |
| `^=` | `a = a ^ b` | `x ^= 0xFF` |
| `<<=` | `a = a << b` | `x <<= 2` |
| `>>=` | `a = a >> b` | `x >>= 2` |

---

## Reference and Pointer Operators

| Operator | Name | Description | Example |
|----------|------|-------------|---------|
| `&` | Shared reference | Create shared reference | `let r = &value` |
| `&!` | Exclusive reference | Create mutable reference | `let r = &!value` |
| `*` | Dereference | Access referenced value | `*r = 10` |

### Reference Syntax

**Important**: Sounio uses `&!` for mutable references, NOT `&mut`:

```sio
// Shared (read-only) reference
fn read(x: &i32) -> i32 {
    return *x
}

// Exclusive (mutable) reference
fn modify(x: &!i32) {
    *x = *x + 1
}
```

---

## Range Operators

| Operator | Name | Description | Example |
|----------|------|-------------|---------|
| `..` | Exclusive range | Range excluding end | `0..10` (0 to 9) |
| `..=` | Inclusive range | Range including end | `0..=10` (0 to 10) |
| `...` | Rest pattern | Match remaining elements | `[first, ...]` |

### Range Usage

```sio
// Exclusive range: 0, 1, 2, ..., 9
for i in 0..10 {
    println(i)
}

// Inclusive range: 0, 1, 2, ..., 10
for i in 0..=10 {
    println(i)
}

// Array slicing
let first_three = arr[0..3]    // Elements 0, 1, 2
let last_three = arr[len-3..]  // Last three elements
let middle = arr[1..=3]        // Elements 1, 2, 3
```

---

## Special Operators

### Concatenation Operator

| Operator | Name | Description | Example |
|----------|------|-------------|---------|
| `++` | Concatenation | Concatenate arrays/strings | `[1, 2] ++ [3, 4]` |

```sio
// Array concatenation
let combined = [1, 2, 3] ++ [4, 5, 6]  // [1, 2, 3, 4, 5, 6]

// String concatenation
let greeting = "Hello, " ++ name ++ "!"
```

### Uncertainty Operator

| Operator | Name | Description | Example |
|----------|------|-------------|---------|
| `+-` | Plus-minus | Value with uncertainty | `100.0 +- 2.5` |

```sio
// Create uncertain value
let measurement = 98.6 +- 0.5

// Access components
let value = measurement.mean       // 98.6
let uncertainty = measurement.std  // 0.5
```

### Path Operators

| Operator | Name | Description | Example |
|----------|------|-------------|---------|
| `::` | Path separator | Namespace/module path | `std::io::println` |
| `.` | Field/method access | Access field or method | `point.x` |

```sio
// Module path
import std::collections::HashMap

// Field access
let x = point.x

// Method call
let length = vec.len()

// Chained access
let result = obj.field.method()
```

---

## Arrow Operators

| Operator | Name | Description | Example |
|----------|------|-------------|---------|
| `->` | Return type arrow | Function return type | `fn add(a: i32) -> i32` |
| `=>` | Fat arrow | Match arm / closure | `0 => "zero"` |
| `<-` | Left arrow | Monadic bind (future) | `x <- computation` |

### Usage Examples

```sio
// Function return type
fn double(x: i32) -> i32 {
    return x * 2
}

// Match arms
match value {
    0 => "zero",
    1 => "one",
    _ => "other",
}

// Closure (alternative syntax)
let add = |a, b| => a + b
```

---

## Indexing and Access

| Operator | Name | Description | Example |
|----------|------|-------------|---------|
| `[]` | Index | Access element by index | `arr[0]` |
| `.` | Field | Access struct field | `point.x` |
| `()` | Call | Function/method call | `func(arg)` |

### Indexing Syntax

```sio
// Single element access
let first = arr[0]
let last = arr[len - 1]

// Slice access
let slice = arr[1..4]      // Elements 1, 2, 3
let head = arr[..3]        // First 3 elements
let tail = arr[3..]        // From element 3 to end
let all = arr[..]          // All elements
```

---

## Partial Derivative Operator

| Operator | Name | Description | Example |
|----------|------|-------------|---------|
| `\partial` or `∂` | Partial derivative | PDE notation | `∂u/∂t` |

```sio
// In PDE blocks
pde HeatEquation {
    equations {
        ∂u/∂t = k * (∂²u/∂x² + ∂²u/∂y²)
    }
}
```

---

## Type Cast Operator

| Operator | Name | Description | Example |
|----------|------|-------------|---------|
| `as` | Type cast | Convert between types | `x as f64` |

### Valid Casts

```sio
// Numeric conversions
let i: i32 = 42
let f = i as f64       // 42.0
let u = i as u32       // 42 (if positive)

// Integer size conversions
let big: i64 = 1000
let small = big as i32  // May truncate

// Character to integer
let c = 'A'
let code = c as u32    // 65
```

---

## Operator Overloading

Sounio supports operator overloading through traits:

```sio
trait Add<Rhs = Self> {
    type Output;
    fn add(self, rhs: Rhs) -> Self::Output;
}

struct Point { x: f64, y: f64 }

impl Add for Point {
    type Output = Point;

    fn add(self, other: Point) -> Point {
        return Point {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

// Now + works with Point
let p3 = p1 + p2
```

### Operator Traits

| Operator | Trait |
|----------|-------|
| `+` | `Add` |
| `-` | `Sub` |
| `*` | `Mul` |
| `/` | `Div` |
| `%` | `Rem` |
| `-` (unary) | `Neg` |
| `!` | `Not` |
| `&` | `BitAnd` |
| `\|` | `BitOr` |
| `^` | `BitXor` |
| `<<` | `Shl` |
| `>>` | `Shr` |
| `==` | `Eq` |
| `<`, `<=`, `>`, `>=` | `Ord` |
| `[]` | `Index`, `IndexMut` |

---

## Common Operator Patterns

### Checked Arithmetic

```sio
// Overflow-checked operations
let result = a.checked_add(b)  // Option<i32>
match result {
    Some(sum) => use(sum),
    None => handle_overflow(),
}

// Saturating arithmetic
let sat = a.saturating_add(b)  // Clamps at max/min

// Wrapping arithmetic
let wrap = a.wrapping_add(b)   // Wraps around on overflow
```

### Chaining Comparisons Pattern

```sio
// Range check pattern
fn is_in_range(x: i32, min: i32, max: i32) -> bool {
    return min <= x && x <= max
}

// Equality chain
fn all_equal(a: i32, b: i32, c: i32) -> bool {
    return a == b && b == c
}
```

### Null-Coalescing Pattern

```sio
// Default value for Option
let value = opt.unwrap_or(default)

// Lazy default
let value = opt.unwrap_or_else(|| compute_default())
```

---

## Operator Precedence Examples

Understanding precedence avoids common mistakes:

```sio
// Multiplication before addition
let result = 2 + 3 * 4     // 14, not 20

// Comparison before logical
let check = x > 0 && y > 0  // Compares first, then AND

// Bitwise before comparison (use parentheses!)
let flag = (x & mask) != 0  // Parentheses needed!
// let flag = x & mask != 0  // Wrong: compares mask != 0 first

// Assignment is lowest precedence
let x = a + b * c           // Computes right side first
```

### When in Doubt, Use Parentheses

```sio
// Clear intent with parentheses
let result = (a + b) * c
let check = (x > 0) && (y > 0)
let flag = (x & mask) == expected
```
