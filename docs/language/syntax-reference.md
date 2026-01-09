---
title: Sounio Syntax Reference
description: Comprehensive syntax reference for the Sounio programming language
prerequisites: []
reading_time: 25 minutes
---

# Sounio Syntax Reference

This document provides a complete reference for Sounio's core syntax. Sounio is a novel L0 systems + scientific programming language for epistemic computing. It is NOT a dialect of Rust, Julia, or any other language.

## Comments

Sounio supports three types of comments.

### Line Comments

```sio
// This is a line comment
let x = 5  // Comment at end of line
```

### Block Comments

```sio
/* This is a block comment
   that spans multiple lines */
let x = /* inline comment */ 5
```

### Documentation Comments

```sio
/// Documentation comment for the following item
/// Supports markdown formatting
fn documented_function() -> i32 {
    return 42
}
```

## Variables

Sounio provides three ways to declare variables: immutable bindings, mutable bindings, and compile-time constants.

### Immutable Bindings (let)

Use `let` for values that should not change after initialization.

```sio
let x = 5              // Type inferred as i32
let y: i32 = 10        // Explicit type annotation
let name = "Sounio"    // Type inferred as string
```

### Mutable Bindings (var)

Use `var` for values that need to be modified.

```sio
var count = 0
count = count + 1      // OK: count is mutable

var total: f64 = 0.0
total = total + 3.14   // OK: total is mutable
```

### Constants (const)

Use `const` for compile-time constant values.

```sio
const PI: f64 = 3.14159265359
const MAX_SIZE: usize = 1024
const GREETING: string = "Hello"
```

**Key differences:**
- `let`: Immutable at runtime, value computed at runtime
- `var`: Mutable at runtime
- `const`: Immutable, value must be known at compile time

## Primitive Types

### Integer Types

**Signed integers:**
| Type | Size | Range |
|------|------|-------|
| `i8` | 8 bits | -128 to 127 |
| `i16` | 16 bits | -32,768 to 32,767 |
| `i32` | 32 bits | -2^31 to 2^31 - 1 |
| `i64` | 64 bits | -2^63 to 2^63 - 1 |
| `i128` | 128 bits | -2^127 to 2^127 - 1 |
| `isize` | Platform | Pointer-sized signed |

**Unsigned integers:**
| Type | Size | Range |
|------|------|-------|
| `u8` | 8 bits | 0 to 255 |
| `u16` | 16 bits | 0 to 65,535 |
| `u32` | 32 bits | 0 to 2^32 - 1 |
| `u64` | 64 bits | 0 to 2^64 - 1 |
| `u128` | 128 bits | 0 to 2^128 - 1 |
| `usize` | Platform | Pointer-sized unsigned |

```sio
let byte: u8 = 255
let count: i32 = -42
let big: i64 = 9_223_372_036_854_775_807
let size: usize = 1024
```

### Floating Point Types

| Type | Size | Precision |
|------|------|-----------|
| `f32` | 32 bits | ~7 decimal digits |
| `f64` | 64 bits | ~15 decimal digits |

```sio
let pi: f64 = 3.14159265359
let temp: f32 = 98.6
let scientific = 1.23e-4  // Scientific notation
```

### Boolean Type

```sio
let active: bool = true
let done: bool = false
```

### Character Type

```sio
let letter: char = 'A'
let emoji: char = '\u{1F600}'  // Unicode scalar
```

### String Type

```sio
let greeting: string = "Hello, Sounio!"
let multiline = "Line 1\nLine 2"
let escaped = "Quote: \"text\""
```

### Unit Type

The unit type `()` represents the absence of a value.

```sio
fn print_message(msg: string) -> () {
    println(msg)
}
```

## Composite Types

### Arrays

Fixed-size, homogeneous collections.

```sio
// Fixed-size array with explicit type
let arr: [i32; 5] = [1, 2, 3, 4, 5]

// Type and size inferred
let numbers = [10, 20, 30]

// Array with repeated value
let zeros: [i32; 100] = [0; 100]

// Access elements (0-indexed)
let first = arr[0]
let last = arr[4]
```

### Slices

References to a contiguous sequence of elements.

```sio
let arr = [1, 2, 3, 4, 5]

// Slice reference
let slice: &[i32] = &arr[1..4]  // [2, 3, 4]

// Slice operations
let head = arr[..3]      // First 3 elements: [1, 2, 3]
let tail = arr[2..]      // From index 2: [3, 4, 5]
let mid = arr[1..4]      // Indices 1-3: [2, 3, 4]
let all = arr[..]        // All elements
```

### Dynamic Arrays (Vec)

Growable, heap-allocated arrays.

```sio
let vec: Vec<i32> = [1, 2, 3]

var numbers: Vec<i32> = Vec::new()
numbers.push(10)
numbers.push(20)
let len = numbers.len()
```

### Array Concatenation

```sio
let a = [1, 2, 3]
let b = [4, 5, 6]
let combined = a ++ b  // [1, 2, 3, 4, 5, 6]
```

### Tuples

Fixed-size, heterogeneous collections.

```sio
let pair: (i32, string) = (42, "answer")
let triple = (1.0, true, "hello")

// Access by index
let x = pair.0    // 42
let y = pair.1    // "answer"
```

**Note:** Tuple destructuring (`let (a, b) = tuple`) is NOT supported. Use index access instead.

### Structs

Named product types with fields.

```sio
struct Point {
    x: f64,
    y: f64,
}

// Create instance
let p = Point { x: 1.0, y: 2.0 }

// Access fields
let x_coord = p.x
let y_coord = p.y
```

### Enums

Sum types with variants.

```sio
// Simple enum
enum Direction {
    North,
    South,
    East,
    West,
}

// Enum with data
enum Option<T> {
    Some(T),
    None,
}

// Enum with struct variants
enum Message {
    Quit,
    Move { x: i32, y: i32 },
    Write(string),
}

let dir = Direction::North
let maybe: Option<i32> = Option::Some(42)
let msg = Message::Move { x: 10, y: 20 }
```

## Functions

### Basic Function Syntax

```sio
fn function_name(param1: Type1, param2: Type2) -> ReturnType {
    // function body
    return value
}
```

### Examples

```sio
// Simple function
fn add(a: i32, b: i32) -> i32 {
    return a + b
}

// No return value
fn greet(name: string) {
    println("Hello, " + name)
}

// Multiple parameters
fn calculate(x: f64, y: f64, z: f64) -> f64 {
    return x * y + z
}
```

### Functions with Effects

Sounio tracks computational effects in function signatures.

```sio
// Function with IO effect
fn read_config(path: string) -> string with IO {
    return read_file(path)
}

// Function with multiple effects
fn process_data(input: string) -> Result<Data, Error> with IO, Alloc {
    let file = read_file(input)
    return parse(file)
}

// Pure function (no effects)
fn pure_add(a: i32, b: i32) -> i32 {
    return a + b
}
```

### Method Syntax

Methods are defined inside `impl` blocks.

```sio
impl Point {
    // Associated function (constructor)
    fn new(x: f64, y: f64) -> Self {
        return Point { x: x, y: y }
    }

    // Method with &self (immutable borrow)
    fn distance(&self, other: &Point) -> f64 {
        let dx = self.x - other.x
        let dy = self.y - other.y
        return sqrt(dx * dx + dy * dy)
    }

    // Method with &!self (mutable borrow)
    fn translate(&!self, dx: f64, dy: f64) {
        self.x = self.x + dx
        self.y = self.y + dy
    }
}

// Usage
let p1 = Point::new(0.0, 0.0)
let p2 = Point::new(3.0, 4.0)
let dist = p1.distance(&p2)
```

## Closures

Anonymous functions that can capture their environment.

### Basic Closure Syntax

```sio
// With explicit types
let add_one = |x: i32| -> i32 { x + 1 }

// With inferred types
let double = |x| x * 2

// Multi-line closure
let process = |data: &[f64]| -> f64 {
    var sum = 0.0
    for x in data {
        sum = sum + x
    }
    return sum
}
```

### Using Closures

```sio
// As function arguments
let numbers = [1, 2, 3, 4, 5]
let doubled = numbers.map(|x| x * 2)
let sum = numbers.fold(0, |acc, x| acc + x)
let evens = numbers.filter(|x| x % 2 == 0)
```

**Important:** Tuple destructuring in closure parameters is NOT supported.

```sio
// WRONG - will not work
arr.map(|(x, y)| x + y)

// CORRECT - use index access
arr.map(|pair| pair.0 + pair.1)
```

## Control Flow

### If Expressions

```sio
// Basic if-else
if condition {
    // code
} else {
    // code
}

// Chained conditions
if x > 10 {
    println("large")
} else if x > 5 {
    println("medium")
} else {
    println("small")
}

// If as expression
let max = if a > b { a } else { b }
```

### Match Expressions

Pattern matching for control flow.

```sio
// Basic match
match value {
    0 => println("zero"),
    1 | 2 => println("one or two"),
    n if n > 10 => println("big"),
    _ => println("other"),
}

// Match as expression
let result = match option {
    Some(x) => x * 2,
    None => 0,
}

// Match on enums
match direction {
    Direction::North => move_up(),
    Direction::South => move_down(),
    Direction::East => move_right(),
    Direction::West => move_left(),
}
```

### While Loops

```sio
var count = 0
while count < 10 {
    println(count)
    count = count + 1
}
```

### For Loops

```sio
// Range iteration (exclusive)
for i in 0..10 {
    println(i)  // 0, 1, 2, ..., 9
}

// Range iteration (inclusive)
for i in 0..=10 {
    println(i)  // 0, 1, 2, ..., 10
}

// Collection iteration
let arr = [10, 20, 30]
for x in arr {
    println(x)
}

// Reference iteration
for x in &arr {
    println(x)
}
```

### Loop (Infinite)

```sio
loop {
    // Infinite loop
    if done {
        break
    }
}

// Loop with value
let result = loop {
    if found {
        break value
    }
}
```

### Loop Control

```sio
// Break exits the loop
for i in 0..100 {
    if i >= 10 {
        break
    }
}

// Continue skips to next iteration
for i in 0..10 {
    if i % 2 == 0 {
        continue
    }
    println(i)  // Prints odd numbers only
}
```

## Operators

### Arithmetic Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `+` | Addition | `a + b` |
| `-` | Subtraction | `a - b` |
| `*` | Multiplication | `a * b` |
| `/` | Division | `a / b` |
| `%` | Remainder | `a % b` |
| `^` | Exponentiation | `a ^ b` |

### Comparison Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `==` | Equal | `a == b` |
| `!=` | Not equal | `a != b` |
| `<` | Less than | `a < b` |
| `>` | Greater than | `a > b` |
| `<=` | Less or equal | `a <= b` |
| `>=` | Greater or equal | `a >= b` |

### Logical Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `&&` | Logical AND | `a && b` |
| `\|\|` | Logical OR | `a \|\| b` |
| `!` | Logical NOT | `!a` |

### Bitwise Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `&` | Bitwise AND | `a & b` |
| `\|` | Bitwise OR | `a \| b` |
| `~` | Bitwise NOT | `~a` |
| `<<` | Left shift | `a << n` |
| `>>` | Right shift | `a >> n` |

### Special Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `++` | Concatenation | `arr1 ++ arr2` |
| `+-` | Plus-minus (uncertainty) | `100.0 +- 2.5` |
| `..` | Exclusive range | `0..10` |
| `..=` | Inclusive range | `0..=10` |

### Assignment Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `=` | Assign | `x = 5` |
| `+=` | Add and assign | `x += 1` |
| `-=` | Subtract and assign | `x -= 1` |
| `*=` | Multiply and assign | `x *= 2` |
| `/=` | Divide and assign | `x /= 2` |
| `%=` | Remainder and assign | `x %= 3` |

### Operator Precedence

From highest to lowest:

1. Field/method access: `.`, `::`
2. Function call: `f()`
3. Unary: `-`, `!`, `&`, `&!`, `*`
4. Exponentiation: `^`
5. Multiplicative: `*`, `/`, `%`
6. Additive: `+`, `-`
7. Shift: `<<`, `>>`
8. Bitwise AND: `&`
9. Bitwise XOR: `^`
10. Bitwise OR: `|`
11. Comparison: `==`, `!=`, `<`, `>`, `<=`, `>=`
12. Logical AND: `&&`
13. Logical OR: `||`
14. Range: `..`, `..=`
15. Assignment: `=`, `+=`, `-=`, etc.

## Modules and Imports

### Module Declaration

```sio
// At the top of a file
module mymodule
```

### Imports

```sio
// Import a module
import std::io
use std::math        // Alias for import

// Both :: and . work as path separators
import std::collections::HashMap
use std.collections.HashMap    // Equivalent

// Selective import
import std::io::{read, write}

// Import with alias
import std::collections::HashMap as Map
```

### Visibility

```sio
// Public function (accessible from other modules)
pub fn exported_function() -> i32 {
    return 42
}

// Private function (default, module-internal)
fn internal_function() -> i32 {
    return 0
}

// Public struct
pub struct PublicPoint {
    pub x: f64,    // Public field
    y: f64,        // Private field
}
```

## Attributes

Attributes provide metadata for items.

### Common Attributes

```sio
// Compatibility level
#[compat(0.15)]
fn compatible_function() { }

#[compat(strict)]
fn strict_function() { }

// Inline hint
#[inline]
fn small_function() { }

// Repr for FFI
#[repr(C)]
struct CCompatible {
    x: i32,
    y: i32,
}

// Derive traits
#[derive(Copy)]
struct Point { x: f64, y: f64 }
```

## Keywords Reference

### Declaration Keywords
- `fn` - Function declaration
- `let` - Immutable binding
- `var` - Mutable binding
- `const` - Compile-time constant
- `struct` - Struct declaration
- `enum` - Enum declaration
- `trait` - Trait declaration
- `impl` - Implementation block
- `type` - Type alias
- `module` - Module declaration

### Control Flow Keywords
- `if` - Conditional
- `else` - Alternative branch
- `match` - Pattern matching
- `for` - For loop
- `while` - While loop
- `loop` - Infinite loop
- `break` - Exit loop
- `continue` - Next iteration
- `return` - Return from function

### Type Keywords
- `linear` - Linear type modifier
- `affine` - Affine type modifier
- `Self` - Self type in impl blocks

### Effect Keywords
- `with` - Effect annotation
- `effect` - Effect declaration
- `handler` - Effect handler
- `handle` - Handle effects
- `perform` - Perform effect operation
- `resume` - Resume from effect

### Visibility Keywords
- `pub` - Public visibility
- `use` - Import (alias for import)
- `import` - Import module
- `export` - Export items

### Async Keywords
- `async` - Async function/block
- `await` - Await future
- `spawn` - Spawn task

### Scientific Keywords
- `kernel` - GPU kernel
- `sample` - Sample from distribution
- `observe` - Observe data
- `Knowledge` - Epistemic type
- `Quantity` - Unit-bearing type

### Other Keywords
- `in` - For loop iteration
- `as` - Type cast / rename
- `where` - Generic constraints
- `self` - Current instance
- `true` - Boolean true
- `false` - Boolean false

## See Also

- [Type System](./type-system.md) - Detailed type system reference
- [Ownership and Borrowing](./ownership-borrowing.md) - Memory safety model
- [Generics](./generics.md) - Generic programming
