---
title: "Language Guide"
description: "Complete syntax and semantics reference for the Sounio programming language"
weight: 2
---

# Sounio Language Guide

This guide covers the complete syntax and semantics of the Sounio programming language.

## Basics

### Variables

Sounio distinguishes between immutable and mutable bindings:

```sio
let x = 5           // Immutable binding
var y = 10          // Mutable binding

y = 20              // OK: y is mutable
// x = 10           // Error: x is immutable
```

### Types

Sounio is statically typed with bidirectional type inference:

```sio
let x: i32 = 42                    // Explicit type annotation
let y = 3.14                       // Inferred as f64
let name: string = "Sounio"        // String type
let flag: bool = true              // Boolean type
```

**Primitive Types:**

| Type | Description | Size |
|------|-------------|------|
| `i8`, `i16`, `i32`, `i64` | Signed integers | 1-8 bytes |
| `u8`, `u16`, `u32`, `u64` | Unsigned integers | 1-8 bytes |
| `f32`, `f64` | Floating point | 4-8 bytes |
| `bool` | Boolean | 1 byte |
| `char` | Unicode scalar | 4 bytes |
| `string` | UTF-8 string | varies |

### References

Sounio uses `&` for shared references and `&!` for exclusive (mutable) references:

```sio
let x = 42
let r: &i32 = &x              // Shared reference
var y = 10
let mr: &!i32 = &!y           // Exclusive/mutable reference
*mr = 20                      // Dereference and assign
```

**Note:** Sounio uses `&!` instead of `&mut` for mutable references.

### Functions

Functions are declared with `fn` and can have effect annotations:

```sio
fn add(a: i32, b: i32) -> i32 {
    a + b
}

fn greet(name: string) -> () with IO {
    print("Hello, ", name)
}

fn modify(arr: &![i32]) with Mut {
    arr[0] = 42
}
```

---

## Control Flow

### Conditionals

```sio
if condition {
    // then branch
} else if other_condition {
    // else-if branch
} else {
    // else branch
}

// Expression form
let max = if a > b { a } else { b }
```

### Loops

```sio
// While loop
while condition {
    // body
}

// For loop
for i in 0..10 {
    print(i)
}

// For-each
for item in collection {
    process(item)
}

// Loop with break
loop {
    if done { break }
}
```

### Pattern Matching

```sio
match value {
    0 => print("zero"),
    1..=9 => print("single digit"),
    n if n < 0 => print("negative"),
    _ => print("other")
}
```

---

## Data Types

### Structs

```sio
struct Point {
    x: f64,
    y: f64,
}

let p = Point { x: 1.0, y: 2.0 }
let distance = sqrt(p.x * p.x + p.y * p.y)
```

### Enums

```sio
enum Option<T> {
    Some(T),
    None,
}

enum Result<T, E> {
    Ok(T),
    Err(E),
}

let maybe: Option<i32> = Option::Some(42)
```

### Arrays and Slices

```sio
let arr: [i32; 5] = [1, 2, 3, 4, 5]    // Fixed-size array
let slice: &[i32] = &arr[1..4]         // Slice (borrowed view)

// Darwin Atlas operations
let head = arr[..3]      // First 3 elements
let tail = arr[2..]      // From index 2 to end
let concat = a ++ b      // Concatenation
```

---

## Effects System

Every side effect is tracked in the type system:

### Effect Annotations

```sio
fn read_file(path: string) -> string with IO { ... }
fn mutate(arr: &![i32]) with Mut { ... }
fn allocate<T>() -> Box<T> with Alloc { ... }
kernel fn compute(data: &[f32]) with GPU { ... }
fn sample() -> f64 with Prob { ... }
```

### Available Effects

| Effect | Description |
|--------|-------------|
| `IO` | File, network, console I/O |
| `Mut` | Mutation through references |
| `Alloc` | Heap allocation |
| `Panic` | Can panic/abort |
| `Async` | Asynchronous operations |
| `GPU` | GPU computation |
| `Prob` | Probabilistic/random operations |
| `Div` | Can diverge (infinite loop) |

### Effect Handlers

```sio
effect Logger {
    fn log(msg: string) -> ()
}

handler ConsoleLogger for Logger {
    fn log(msg: string) -> () {
        print("[LOG] ", msg)
    }
}

fn main() with IO {
    with ConsoleLogger {
        do_work()
    }
}
```

---

## Linear Types

Linear types ensure resources are used exactly once:

```sio
linear struct FileHandle {
    fd: i32
}

fn open(path: string) -> FileHandle with IO {
    FileHandle { fd: sys_open(path) }
}

fn close(handle: FileHandle) with IO {
    sys_close(handle.fd)
    // handle is consumed, cannot be used again
}

fn main() with IO {
    let file = open("data.txt")
    // Must consume file exactly once
    close(file)
    // close(file)  // Error: file already consumed
}
```

---

## Units of Measure

Type-safe dimensional analysis:

```sio
let mass: kg = 1.5
let distance: m = 100.0
let time: s = 9.58

let velocity: m/s = distance / time
let energy: J = 0.5 * mass * velocity * velocity

// Compile-time error:
// let invalid: kg = distance    // Error: m cannot be assigned to kg
```

### SI Units

- Base: `kg`, `m`, `s`, `A`, `K`, `mol`, `cd`
- Derived: `N`, `J`, `W`, `Pa`, `Hz`, `V`, `Ω`
- Pharma: `mg`, `μg`, `mL`, `L`, `h`

---

## Epistemic Types

The `Knowledge<T>` type tracks uncertainty:

```sio
let measurement: Knowledge<mg> = Knowledge::new(
    value: 500.0,
    std_uncertainty: 2.5,
    confidence: 0.95
)

let result = measurement * 2.0
// Uncertainty propagates automatically (GUM-compliant)

print(result.value)           // 1000.0
print(result.std_uncertainty) // 5.0 (linear propagation)
print(result.ci_95())         // (990.2, 1009.8)
```

---

## GPU Kernels

GPU computation with the `kernel` keyword:

```sio
kernel fn vector_add(a: &[f32], b: &[f32], c: &![f32]) with GPU {
    let i = gpu.thread_id.x
    if i < a.len() {
        c[i] = a[i] + b[i]
    }
}

fn main() with IO, GPU {
    let a = gpu.alloc([1.0; 1000])
    let b = gpu.alloc([2.0; 1000])
    var c = gpu.alloc([0.0; 1000])

    vector_add<<<4, 256>>>(a, b, &!c)
}
```

---

## Refinement Types

Compile-time constraints via Z3:

```sio
type Positive = { x: i32 | x > 0 }
type NonEmpty<T> = { arr: [T] | arr.len() > 0 }
type Percentage = { p: f64 | 0.0 <= p && p <= 100.0 }

fn divide(a: i32, b: Positive) -> i32 {
    a / b  // Safe: b cannot be zero
}
```

---

## See Also

- **[Standard Library](/docs/stdlib/)** — Module reference
- **[API Reference](/docs/api/)** — Compiler APIs
- **[Examples](/examples/)** — Runnable code samples
- **[Getting Started](/docs/getting-started/)** — Installation and first program
