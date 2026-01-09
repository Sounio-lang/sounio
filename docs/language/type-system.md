---
title: Sounio Type System
description: Complete reference for Sounio's static type system with inference
prerequisites:
  - ./syntax-reference.md
reading_time: 20 minutes
---

# Sounio Type System

Sounio features a powerful static type system with type inference. Types are checked at compile time, catching errors before your code runs. This document covers all aspects of the type system.

## Static Typing with Inference

Sounio is statically typed, meaning all types are known at compile time. However, you don't always need to write type annotations explicitly.

### Type Inference

The compiler can infer types from context:

```sio
let x = 5              // Inferred as i32
let pi = 3.14159       // Inferred as f64
let name = "Sounio"    // Inferred as string
let active = true      // Inferred as bool

// Inference from function return types
fn get_number() -> i32 { return 42 }
let n = get_number()   // Inferred as i32

// Inference in collections
let numbers = [1, 2, 3, 4, 5]  // Inferred as [i32; 5]
```

### Explicit Type Annotations

You can always be explicit about types:

```sio
let x: i32 = 5
let pi: f64 = 3.14159
let name: string = "Sounio"
let numbers: [i32; 5] = [1, 2, 3, 4, 5]
```

Type annotations are required in:
- Function parameters
- Function return types (unless returning unit)
- Struct and enum field definitions
- When inference is ambiguous

## Primitive Types

### Integer Types

Sounio provides a complete set of integer types.

**Signed integers** store positive and negative values:

```sio
let a: i8 = -128                    // 8-bit signed
let b: i16 = -32768                 // 16-bit signed
let c: i32 = -2147483648            // 32-bit signed (default)
let d: i64 = -9223372036854775808   // 64-bit signed
let e: i128 = 0                     // 128-bit signed
let f: isize = 0                    // Platform pointer-size signed
```

**Unsigned integers** store only non-negative values:

```sio
let a: u8 = 255                     // 8-bit unsigned
let b: u16 = 65535                  // 16-bit unsigned
let c: u32 = 4294967295             // 32-bit unsigned
let d: u64 = 18446744073709551615   // 64-bit unsigned
let e: u128 = 0                     // 128-bit unsigned
let f: usize = 0                    // Platform pointer-size unsigned
```

**Integer literals:**

```sio
let decimal = 1_000_000      // Underscores for readability
let hex = 0xFF               // Hexadecimal
let octal = 0o77             // Octal
let binary = 0b1111_0000     // Binary
```

### Floating Point Types

```sio
let single: f32 = 3.14       // 32-bit IEEE 754
let double: f64 = 3.14159    // 64-bit IEEE 754 (default)

// Scientific notation
let avogadro: f64 = 6.022e23
let planck: f64 = 6.626e-34
```

### Boolean Type

```sio
let yes: bool = true
let no: bool = false

// Boolean operations
let and_result = yes && no   // false
let or_result = yes || no    // true
let not_result = !yes        // false
```

### Character and String Types

```sio
// Character (Unicode scalar value)
let letter: char = 'A'
let greek: char = '\u{03B1}'   // alpha
let emoji: char = '\u{1F600}'  // grinning face

// String (UTF-8)
let greeting: string = "Hello, World!"
let escaped: string = "Line 1\nLine 2"
let unicode: string = "Hello, \u{4E16}\u{754C}"
```

### Unit Type

The unit type `()` represents the absence of a meaningful value.

```sio
// Function returning unit
fn log_message(msg: string) -> () {
    println(msg)
}

// Unit value
let nothing: () = ()
```

## Composite Types

### Structs

Named product types with fields.

```sio
// Basic struct
struct Point {
    x: f64,
    y: f64,
}

// Struct with various field types
struct Person {
    name: string,
    age: u32,
    active: bool,
}

// Creating instances
let origin = Point { x: 0.0, y: 0.0 }
let user = Person {
    name: "Alice",
    age: 30,
    active: true,
}

// Field access
let x_coord = origin.x
let user_name = user.name
```

### Enums

Sum types with variants.

```sio
// Simple enum (unit variants)
enum Color {
    Red,
    Green,
    Blue,
}

// Enum with tuple variants
enum Result<T, E> {
    Ok(T),
    Err(E),
}

// Enum with struct variants
enum Event {
    Click { x: i32, y: i32 },
    KeyPress { key: char },
    Resize { width: u32, height: u32 },
}

// Usage
let color = Color::Red
let result: Result<i32, string> = Result::Ok(42)
let event = Event::Click { x: 100, y: 200 }
```

### Tuples

Fixed-size, ordered, heterogeneous collections.

```sio
// Tuple types
let pair: (i32, string) = (42, "answer")
let triple: (f64, bool, char) = (3.14, true, 'x')

// Nested tuples
let nested: ((i32, i32), string) = ((1, 2), "point")

// Access by index
let first = pair.0     // 42
let second = pair.1    // "answer"
```

### Arrays

Fixed-size, homogeneous sequences.

```sio
// Array with explicit type
let nums: [i32; 5] = [1, 2, 3, 4, 5]

// Array with repeated value
let zeros: [i32; 10] = [0; 10]

// Type inferred from elements
let values = [1.0, 2.0, 3.0]  // [f64; 3]

// Access
let first = nums[0]
let last = nums[4]
```

### Slices

Dynamically-sized views into arrays.

```sio
// Slice reference type
let arr = [1, 2, 3, 4, 5]
let slice: &[i32] = &arr[1..4]  // Reference to [2, 3, 4]

// Mutable slice
var data = [1, 2, 3, 4, 5]
let mutable_slice: &![i32] = &!data[..]
```

### Vec (Dynamic Array)

Growable, heap-allocated arrays.

```sio
let vec: Vec<i32> = [1, 2, 3]
let empty: Vec<string> = Vec::new()

var numbers: Vec<f64> = Vec::new()
numbers.push(1.0)
numbers.push(2.0)
```

## Generic Types

Types can be parameterized over other types.

```sio
// Generic struct
struct Container<T> {
    value: T,
}

// Generic enum
enum Option<T> {
    Some(T),
    None,
}

// Multiple type parameters
struct Pair<A, B> {
    first: A,
    second: B,
}

// Usage
let int_container: Container<i32> = Container { value: 42 }
let maybe_string: Option<string> = Option::Some("hello")
let pair: Pair<i32, string> = Pair { first: 1, second: "one" }
```

## Type Aliases

Create alternative names for types.

```sio
// Simple alias
type Sequence = [u8]
type Matrix = [[f64]]

// Generic alias
type StringResult = Result<string, Error>
type Callback<T> = fn(T) -> bool

// Usage
let data: Sequence = [1, 2, 3, 4]
let process: Callback<i32> = |x| x > 0
```

## The Never Type

The never type `!` represents computations that never complete normally.

```sio
// Function that never returns
fn infinite_loop() -> ! {
    loop {
        // Never terminates
    }
}

// Function that always panics
fn fail(msg: string) -> ! {
    panic(msg)
}

// Used in match arms
fn get_value(opt: Option<i32>) -> i32 {
    match opt {
        Some(x) => x,
        None => panic("no value"),  // panic returns !
    }
}
```

## Function Types

Functions have types based on their parameters and return type.

```sio
// Function type syntax: fn(params) -> return_type
type BinaryOp = fn(i32, i32) -> i32

// Function with effects
type Reader = fn(string) -> string with IO

// Using function types
fn apply(f: fn(i32) -> i32, x: i32) -> i32 {
    return f(x)
}

let double: fn(i32) -> i32 = |x| x * 2
let result = apply(double, 5)  // 10
```

## Reference Types

References provide borrowed access to values.

### Shared References (`&T`)

Immutable, shared access.

```sio
fn read_value(x: &i32) -> i32 {
    return *x
}

let n = 42
let reference: &i32 = &n
let value = read_value(&n)
```

### Exclusive References (`&!T`)

Mutable, exclusive access. **Note: Sounio uses `&!T`, NOT `&mut T`.**

```sio
fn increment(x: &!i32) {
    *x = *x + 1
}

var n = 42
increment(&!n)  // n is now 43
```

## Raw Pointer Types (FFI)

For interfacing with C code.

```sio
// Const pointer (read-only)
let ptr: *const i32 = null_ptr()

// Mutable pointer
let mut_ptr: *mut i32 = null_mut()
```

## Type Coercion

Sounio performs automatic type coercions in specific situations.

### Reference Coercions

```sio
// &!T coerces to &T
fn read(x: &i32) -> i32 { return *x }

var n = 42
let ptr: &!i32 = &!n
read(ptr)  // &!i32 automatically coerces to &i32
```

### Deref Coercions

Types implementing `Deref` are automatically dereferenced.

```sio
let s: string = "hello"
let len = s.len()  // string derefs to &str
```

### Numeric Coercions

Explicit casting is required for numeric conversions.

```sio
let x: i32 = 42
let y: i64 = x as i64    // Explicit cast required
let z: f64 = x as f64    // Integer to float
```

## Refinement Types

Types constrained by predicates, verified at compile time via SMT solving.

```sio
// Basic refinement
type Positive = { x: i32 | x > 0 }
type NonEmpty<T> = { arr: [T] | len(arr) > 0 }
type Percentage = { p: f64 | p >= 0.0 && p <= 100.0 }

// Usage
let pos: Positive = 42              // OK: 42 > 0
// let bad: Positive = -1           // COMPILE ERROR

// Function with refined parameter
fn sqrt_safe(x: { n: f64 | n >= 0.0 }) -> f64 {
    return sqrt(x)
}

// Struct invariants
struct BoundedCounter {
    value: i32,
    invariant value >= 0 && value <= 100
}
```

## Epistemic Types

Sounio's distinctive feature: types that track uncertainty and provenance.

### Knowledge Type

```sio
let measurement: Knowledge[
    content = f64,
    confidence = 0.95,
    provenance = Measured("sensor_001"),
    valid_until = "2024-12-31"
] = 98.6

// Accessing knowledge
let value = measurement.value()
let conf = measurement.confidence()
```

### Uncertain Values

```sio
// Value with uncertainty using +- operator
let measured = 100.0 +- 2.5  // 100 with uncertainty 2.5

// Uncertainty propagates through operations
let doubled = measured * 2.0  // Uncertainty doubles too
```

### Quantity Types (Units of Measure)

```sio
// Values with physical units
let mass: mg = 500.0_mg
let volume: L = 0.5_L
let concentration: mg/L = mass / volume  // Type-safe!

// Unit mismatch is a compile error
// let wrong = mass + volume  // ERROR: mg + L not allowed
```

## Type Inference Algorithm

Sounio uses bidirectional type inference:

1. **Synthesis**: Infer type from expression
2. **Checking**: Verify expression against expected type

```sio
// Synthesis: infer x is i32 from literal
let x = 42

// Checking: verify [1, 2, 3] matches Vec<i32>
let v: Vec<i32> = [1, 2, 3]

// Bidirectional: use context to resolve ambiguity
fn process(data: Vec<i32>) { }
process([1, 2, 3])  // [1, 2, 3] checked against Vec<i32>
```

## Linear Algebra Types

Built-in types for mathematical computing.

```sio
// Vectors
let v2: vec2 = vec2(1.0, 2.0)
let v3: vec3 = vec3(1.0, 2.0, 3.0)
let v4: vec4 = vec4(1.0, 2.0, 3.0, 4.0)

// Matrices
let m2: mat2 = mat2::identity()
let m3: mat3 = mat3::identity()
let m4: mat4 = mat4::rotation_x(angle)

// Quaternion
let q: quat = quat::from_axis_angle(axis, angle)

// Operations
let dot = v3.dot(other_v3)
let cross = v3.cross(other_v3)
let transformed = m4 * v4
```

## Automatic Differentiation Types

```sio
// Dual numbers for forward-mode autodiff
let x: dual = dual(3.0, 1.0)  // value = 3, derivative seed = 1
let y = x * x                  // y.value = 9, y.deriv = 6
```

## Common Type Errors

### Mismatched Types

```sio
let x: i32 = "hello"  // ERROR: expected i32, found string
```

### Missing Type Annotation

```sio
let empty = Vec::new()  // ERROR: cannot infer type, add annotation
let empty: Vec<i32> = Vec::new()  // OK
```

### Numeric Overflow

```sio
let x: u8 = 256  // ERROR: literal out of range for u8
```

### Reference Mutability

```sio
fn modify(x: &i32) {
    *x = 10  // ERROR: cannot assign to immutable reference
}

fn modify(x: &!i32) {
    *x = 10  // OK: exclusive reference allows mutation
}
```

## See Also

- [Syntax Reference](./syntax-reference.md) - Core language syntax
- [Ownership and Borrowing](./ownership-borrowing.md) - Memory safety model
- [Generics](./generics.md) - Generic programming
