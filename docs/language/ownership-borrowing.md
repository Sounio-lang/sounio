---
title: Ownership and Borrowing in Sounio
description: Memory safety through ownership, references, and linear types
prerequisites:
  - ./syntax-reference.md
  - ./type-system.md
reading_time: 20 minutes
---

# Ownership and Borrowing

Sounio provides memory safety without garbage collection through its ownership system. This document explains how ownership, borrowing, and linear types work together to prevent memory errors at compile time.

## Core Concepts

### Ownership Rules

1. Each value has exactly one owner
2. When the owner goes out of scope, the value is dropped
3. Values can be moved to new owners or borrowed

## Move Semantics

By default, values are moved when assigned or passed to functions.

### Basic Moves

```sio
struct Data {
    buffer: Vec<u8>,
}

fn main() {
    let a = Data { buffer: [1, 2, 3] }
    let b = a           // a is moved to b

    // println(a.buffer)  // ERROR: a has been moved
    println(b.buffer)     // OK: b owns the data
}
```

### Function Argument Moves

```sio
fn consume(data: Data) {
    // data is owned here
    println(data.buffer)
}  // data is dropped here

fn main() {
    let d = Data { buffer: [1, 2, 3] }
    consume(d)        // d is moved into consume

    // consume(d)     // ERROR: d has been moved
}
```

### Move vs Copy

Some types implement `Copy` and are automatically copied instead of moved:

```sio
// Copy types: primitives
let x: i32 = 42
let y = x          // x is copied, not moved
println(x)         // OK: x is still valid
println(y)         // OK: y has a copy

// Move types: heap-allocated
let v1: Vec<i32> = [1, 2, 3]
let v2 = v1        // v1 is moved, not copied
// println(v1)     // ERROR: v1 has been moved
```

**Copy types include:**
- All integer types (`i8`, `i16`, `i32`, `i64`, `i128`, `u8`, `u16`, `u32`, `u64`, `u128`)
- All floating point types (`f32`, `f64`)
- `bool`, `char`
- Tuples of Copy types
- Fixed-size arrays of Copy types
- References (`&T` and `&!T`)

## References and Borrowing

Instead of moving values, you can borrow them through references.

### Shared References (`&T`)

Shared references provide read-only access. Multiple shared references can exist simultaneously.

```sio
fn calculate_length(s: &string) -> usize {
    return s.len()
}

fn main() {
    let text = "Hello, Sounio"

    let len1 = calculate_length(&text)  // Borrow text
    let len2 = calculate_length(&text)  // Borrow again

    println(text)  // text is still valid
}
```

### Exclusive References (`&!T`)

**CRITICAL: Sounio uses `&!T` for mutable references, NOT `&mut T`.**

Exclusive references provide read-write access. Only one exclusive reference can exist at a time.

```sio
fn increment(x: &!i32) {
    *x = *x + 1
}

fn main() {
    var n = 42
    increment(&!n)     // Exclusive borrow of n
    println(n)         // 43
}
```

### Borrowing Rules

These rules are enforced at compile time:

1. **You can have either:**
   - Any number of shared references (`&T`), OR
   - Exactly one exclusive reference (`&!T`)

2. **References must be valid for their entire lifetime**

```sio
// OK: Multiple shared references
let data = [1, 2, 3, 4, 5]
let r1: &[i32] = &data
let r2: &[i32] = &data
let r3: &[i32] = &data
// All three can read simultaneously

// OK: Single exclusive reference
var numbers = [1, 2, 3]
let excl: &![i32] = &!numbers
// excl can read and write

// ERROR: Cannot mix shared and exclusive
var value = 42
let shared: &i32 = &value
let excl: &!i32 = &!value  // ERROR: already borrowed as shared
```

### Reference Lifetimes

References cannot outlive the data they point to.

```sio
fn dangling_reference() -> &i32 {
    let x = 42
    return &x        // ERROR: x is dropped, reference would dangle
}

fn valid_reference(data: &i32) -> &i32 {
    return data      // OK: lifetime comes from caller
}
```

## Linear Types

Linear types extend ownership with usage tracking. A linear value must be used exactly once.

### Linear Structs

```sio
linear struct FileHandle {
    fd: i32,
}

impl FileHandle {
    fn open(path: string) -> FileHandle with IO {
        // Open file and return handle
    }

    fn close(self) with IO {
        // Close the file descriptor
        // self is consumed here
    }
}

fn process_file(path: string) with IO {
    let handle = FileHandle::open(path)

    // ... use the file ...

    handle.close()  // MUST be called - linear type requires use
}

fn bad_example(path: string) with IO {
    let handle = FileHandle::open(path)
    // ERROR at compile time: handle is never used/closed
}
```

### Why Linear Types?

Linear types guarantee resource cleanup:

- **Files** must be closed
- **Locks** must be released
- **Network connections** must be terminated
- **Memory allocations** must be freed

The compiler enforces that these resources are properly handled.

## Affine Types

Affine types can be used at most once (zero or one times).

```sio
affine struct TempBuffer {
    ptr: *u8,
    len: usize,
}

impl TempBuffer {
    fn allocate(size: usize) -> TempBuffer with Alloc {
        // Allocate temporary buffer
    }

    fn free(self) with Alloc {
        // Free the buffer
    }
}

fn example() with Alloc {
    let buf = TempBuffer::allocate(1024)

    // Option 1: Use and free
    buf.free()

    // Option 2: Just let it go out of scope (OK for affine)
    // The compiler may insert implicit cleanup
}
```

### Linear vs Affine

| Property | Linear | Affine |
|----------|--------|--------|
| Must use | Yes, exactly once | No |
| Can discard | No | Yes (drop is implicit use) |
| Use case | Critical resources | Optional cleanup |

## Relevant Types

Relevant types must be used at least once.

```sio
// Relevant knowledge must be acknowledged
fn mandatory_evidence_check(evidence: relevant Knowledge<f64>) {
    // Must use evidence at least once
    let value = evidence.value()
    // Can use again
    let conf = evidence.confidence()
}
```

## Type Modalities Summary

| Modality | Use Count | Example |
|----------|-----------|---------|
| **Unrestricted** | 0+ times | Normal types, `Copy` types |
| **Linear** | Exactly 1 | `linear struct`, file handles |
| **Affine** | At most 1 | `affine struct`, temp buffers |
| **Relevant** | At least 1 | Mandatory evidence |

## Reference Syntax Comparison

**CRITICAL: Sounio is NOT Rust. Use the correct syntax.**

| Purpose | Sounio | Rust (WRONG in Sounio) |
|---------|--------|------------------------|
| Shared reference | `&T` | `&T` |
| Exclusive/mutable reference | `&!T` | `&mut T` |
| Take shared reference | `&value` | `&value` |
| Take exclusive reference | `&!value` | `&mut value` |

### Examples

```sio
// CORRECT Sounio syntax
fn process(data: &[f64], output: &![f64]) {
    for i in 0..len(data) {
        output[i] = data[i] * 2.0
    }
}

var results = [0.0; 10]
let inputs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
process(&inputs, &!results)
```

```sio
// WRONG - This is Rust syntax, NOT Sounio
fn wrong(data: &[f64], output: &mut [f64]) {  // ERROR: &mut doesn't exist
    // ...
}
```

## Method Receivers

Methods can take `self` in different forms:

```sio
struct Point {
    x: f64,
    y: f64,
}

impl Point {
    // Consumes self (takes ownership)
    fn into_tuple(self) -> (f64, f64) {
        return (self.x, self.y)
    }

    // Borrows self (read-only)
    fn magnitude(&self) -> f64 {
        return sqrt(self.x * self.x + self.y * self.y)
    }

    // Mutably borrows self (read-write)
    fn scale(&!self, factor: f64) {
        self.x = self.x * factor
        self.y = self.y * factor
    }
}

fn main() {
    var p = Point { x: 3.0, y: 4.0 }

    let mag = p.magnitude()   // Borrows p
    p.scale(2.0)              // Mutably borrows p

    let tuple = p.into_tuple()  // Consumes p
    // p is no longer valid here
}
```

## Common Patterns

### Returning Borrowed Data

```sio
// Return reference with same lifetime as input
fn first_element<T>(arr: &[T]) -> &T {
    return &arr[0]
}

// Return mutable reference
fn first_element_mut<T>(arr: &![T]) -> &!T {
    return &!arr[0]
}
```

### Borrowing in Structs

```sio
// Structs can hold references (advanced)
struct Parser<'a> {
    input: &'a string,
    position: usize,
}

// More common: own the data
struct Parser {
    input: string,
    position: usize,
}
```

### Interior Mutability

For cases where you need mutation behind a shared reference:

```sio
use std::cell::RefCell

struct Counter {
    value: RefCell<i32>,
}

impl Counter {
    fn increment(&self) {
        var val = self.value.borrow_mut()
        *val = *val + 1
    }
}
```

## Common Errors and Solutions

### Error: Cannot Borrow as Mutable

```sio
let data = [1, 2, 3]
data[0] = 10  // ERROR: data is immutable

// Solution: Use var
var data = [1, 2, 3]
data[0] = 10  // OK
```

### Error: Value Moved

```sio
let v = vec![1, 2, 3]
let v2 = v
println(v)  // ERROR: v has been moved

// Solution 1: Clone
let v = vec![1, 2, 3]
let v2 = v.clone()
println(v)  // OK

// Solution 2: Borrow
let v = vec![1, 2, 3]
let v2 = &v
println(v)  // OK
```

### Error: Cannot Borrow Mutably More Than Once

```sio
var data = [1, 2, 3]
let r1 = &!data
let r2 = &!data  // ERROR: already borrowed

// Solution: Limit scope of first borrow
var data = [1, 2, 3]
{
    let r1 = &!data
    // use r1
}
let r2 = &!data  // OK: r1's borrow ended
```

### Error: Linear Value Not Used

```sio
linear struct Resource { handle: i32 }

fn bad() {
    let r = Resource { handle: 1 }
    // ERROR: linear value 'r' not used
}

// Solution: Use or explicitly consume
fn good() {
    let r = Resource { handle: 1 }
    consume_resource(r)  // r is used
}
```

## Best Practices

1. **Prefer borrowing over moving** when you don't need ownership
2. **Use `&T` by default**, only use `&!T` when mutation is needed
3. **Keep borrows short-lived** to avoid conflicts
4. **Use linear types for resources** that require cleanup
5. **Remember: `&!T` NOT `&mut T`** - Sounio is not Rust

## See Also

- [Syntax Reference](./syntax-reference.md) - Core language syntax
- [Type System](./type-system.md) - Type system reference
- [Generics](./generics.md) - Generic programming
