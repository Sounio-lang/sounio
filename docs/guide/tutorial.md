<!-- docs:meta
topic_id: repo.docs.guide.tutorial
authority: repo_only
audience: users
last_validated: 2026-04-12
validated_by: human
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.guide.tutorial
-->

> **Status**: Production | **Last validated**: 2026-04-12 | **Source**: `tests/run-pass/`

# Sounio Tutorial

A step-by-step guide to learning Sounio, the language for epistemic computing.

**Implementation key** (used throughout this tutorial):
- **Production** — implemented, tested, gate-backed
- **Beta** — works for common patterns, edge cases may exist
- **Planned** — specified but not yet implemented; examples show intended syntax

## Table of Contents

1. [Getting Started](#1-getting-started)
2. [Basic Syntax](#2-basic-syntax)
3. [Effect System](#3-effect-system)
4. [Epistemic Types](#4-epistemic-types)
5. [Units of Measure](#5-units-of-measure)
6. [Data Structures](#6-data-structures)
7. [Scientific Computing](#7-scientific-computing)
8. [Advanced Features](#8-advanced-features)

---

## 1. Getting Started

### Installation

```bash
git clone https://github.com/sounio-lang/sounio.git
cd sounio

export SOUC_BIN="$(pwd)/bin/souc"
"$SOUC_BIN" info
```

### Your First Program

Create a file `hello.sio`:

```sio
fn main() with IO {
    println("Hello, Sounio!")
}
```

Run it:

```bash
"$SOUC_BIN" run hello.sio
```

---

## 2. Basic Syntax

### Variables — **Production**

```sio
// Immutable by default
let x = 42
let name = "Sounio"

// Mutable with 'var'
var counter = 0
counter = counter + 1

// Type annotations
let age: i32 = 25
let pi: f64 = 3.14159
```

**Key Difference from Rust**: Sounio uses `var` for mutable variables, not `let mut`. No semicolons.

### Functions — **Production**

```sio
// Simple function with implicit return
fn add(a: i32, b: i32) -> i32 {
    a + b
}

// Functions with side effects MUST declare effects
fn greet(name: &str) with IO {
    println(name)
}

// Error code pattern
fn safe_divide(a: f64, b: f64) -> (f64, i32) with Div, Panic {
    if b == 0.0 { (0.0, 1) }
    else { (a / b, 0) }
}
```

### Control Flow — **Production**

```sio
// If expressions
let max = if x > y { x } else { y }

// While loops
var i = 0
while i < 10 {
    println(i)
    i = i + 1
}

// For-in loops (source: tests/run-pass/for_in_loops.sio)
for i in 0..5 { /* 0, 1, 2, 3, 4 */ }
for i in 0..=5 { /* 0, 1, 2, 3, 4, 5 (inclusive) */ }
for x in arr { /* iterate over array */ }

// Pattern matching (source: tests/run-pass/native_enum_basic.sio)
match c {
    Color::Red => 10
    Color::Green => 20
    _ => 0
}
```

---

## 3. Effect System — **Production**

Effects track what a function can do. Missing effects cause compile errors.

### Required Effects

| Effect | When required | Example |
|---|---|---|
| `IO` | `print()`, `println()`, file/env | `println("text")` |
| `Mut` | `&!` mutation, array assignment | `*x = 42` |
| `Div, Panic` | Division `/`, modulo `%` | `a / b` |
| `Panic` | Array access, `assert()`, `as` casts | `arr[i]` |

### Examples

```sio
fn mutate(x: &!i32) with Mut { *x = 42 }
fn divide(a: f64, b: f64) -> f64 with Div, Panic { a / b }
fn hello() with IO { println("hi") }
fn process(arr: &![u8; 256]) with IO, Mut, Panic, Div { /* ... */ }
```

### Custom Effect Handlers — **Beta**

```sio
// source: tests/run-pass/effect_handler_basic.sio
effect Choice {
    fn pick() -> bool
}

fn coin_flip() with Choice {
    // uses Choice effect
}

fn main() with Choice {
    coin_flip()
}
```

---

## 4. Epistemic Types — **Production** (core)

Sounio's signature feature: values that carry explicit uncertainty metadata.

### Knowledge Values

```sio
// Struct literal construction
let risky = Knowledge { value: 15.0, epsilon: 0.4 }
let safe = Knowledge { value: 15.0, epsilon: 0.9 }
```

### Epistemic Arithmetic

Uncertainty propagates automatically through arithmetic operations:

```sio
let x = epistemic_std(10.0, 0.5, 0.95)
let y = epistemic_std(5.0, 0.2, 0.95)

let sum = add_epistemic(x, y)
let product = mul_epistemic(x, y)
```

### Automatic Propagation with Operators — **Beta**

Direct operator overloading for `Knowledge` values (works in interpreter, limited in JIT):

```sio
let x = Knowledge::new(10.0, uncertainty: 0.5)
let y = Knowledge::new(5.0, uncertainty: 0.2)
let area = x * y  // uncertainty propagated via GUM
```

### Confidence-Based Execution — **Production**

```sio
fn administer_drug(dose: Knowledge<f64>) with IO {
    if dose.confidence > 0.95 {
        inject(dose)
    } else {
        println("Dose confidence too low")
    }
}
```

---

## 5. Units of Measure — **Production**

Sounio has first-class support for physical units, preventing dimensional errors.

### Basic Units

```sio
// Declare quantities with units
let distance: f64<m> = 100.0
let time: f64<s> = 10.0
let velocity = distance / time  // type: f64<m/s>

// Compile-time unit checking
// let invalid = distance + time  // ERROR: can't add meters to seconds
```

### Unit Literals

```sio
let dose: f64<mg> = 500.0
let volume: f64<mL> = 250.0
let concentration = dose / volume  // type: f64<mg/mL>
```

### Unit Conversions — **Planned**

```sio
// Planned syntax — not yet gate-backed
let distance_m: f64<m> = 1000.0
let distance_km: f64<km> = convert(distance_m)
```

---

## 6. Data Structures

### Structs — **Production**

```sio
struct Point { x: f64, y: f64 }

let p = Point { x: 1.0, y: 2.0 }
let px = p.x

// Linear types for single-ownership resources
linear struct FileHandle { fd: i32 }
```

### Enums — **Beta**

```sio
// source: tests/run-pass/native_enum_basic.sio
enum Color { Red, Green, Blue }

let c = Color::Red
match c {
    Color::Red => 10
    Color::Green => 20
    _ => 0
}
```

### Fixed-Size Arrays — **Production**

```sio
var buffer: [u8; 256] = [0; 256]
let data: [i64; 4] = [1, 2, 3, 4]
let first = data[0]
```

### Mutable Array References — **Production**

```sio
// source: tests/run-pass/array_mut_ref.sio
fn fill(arr: &![i64; 8]) with Mut, Panic {
    (*arr)[0] = 99    // explicit dereference required for bare arrays
}

var buf: [i64; 8] = [0, 0, 0, 0, 0, 0, 0, 0]
fill(&!buf)
```

### impl Blocks — **Production**

```sio
// source: stdlib/collections/vec.sio
impl IntVec {
    fn new() -> IntVec { IntVec { data: [0; 4096], len: 0 } }
    fn push(self: &!IntVec, val: i64) with Mut, Panic { /* ... */ }
    fn len(self: &IntVec) -> i64 { self.len }
}
```

### Tuples — **Production**

```sio
// source: tests/run-pass/tuple_destructure_let.sio
let (a, b) = (1, 2)
let (x, (y, z)) = (10, (20, 30))
let (first, _) = (5, 10)
```

---

## 7. Scientific Computing

### Epistemic Arithmetic — **Production**

```sio
import stdlib.epistemic::*

let x = epistemic_std(10.0, 0.5, 0.95)
let y = epistemic_std(5.0, 0.2, 0.95)
let sum = add_epistemic(x, y)
let product = mul_epistemic(x, y)
```

### ODE Solvers — **Production**

```sio
import stdlib.ode::*

// RK4 solver — verified in tests/run-pass/
let y_end = rk4_step(dydt_fn_id, 0.0, 100.0, 0.1)
```

Higher-level ODE DSL with named parameters — **Planned**:

```sio
// Planned — not yet gate-backed
let solution = solve_ode(
    f: exponential_decay,
    y0: 100.0,
    t_span: (0.0, 10.0),
    method: RK45
)
```

### Linear Algebra — **Production**

```sio
import stdlib.linalg::*

// Matrix operations with fixed-size arrays
var A: [f64; 4] = [1.0, 2.0, 3.0, 4.0]  // 2x2 row-major
let det = mat2_det(A)
```

### Signal Processing — **Planned**

```sio
// Planned — not yet gate-backed
let signal = [1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0]
let spectrum = fft(signal)
```

---

## 8. Advanced Features

### Refinement Types — **Beta**

Parsing works; SMT verification requires Z3 and falls back to runtime assertions.

```sio
type Positive = { x: i32 | x > 0 }

fn sqrt(x: Positive) -> f64 {
    math.sqrt(x as f64)
}
```

### Linear Types — **Production**

```sio
linear struct FileHandle { fd: i32 }

fn close(handle: FileHandle) {
    os.close(handle.fd)
}

let file = open("data.txt")
close(file)
// close(file)  // ERROR: file has been consumed
```

### GPU Computing — **Production** (GPU artifact only)

```sio
// Use the GPU artifact: souc-linux-x86_64-gpu
// souc build file.sio --backend gpu -o kernel.ptx

kernel vec_add(a: &[f32], b: &[f32], out: &[f32], n: i32) {
    let idx = gpu_thread_id()
    if idx < n {
        out[idx] = a[idx] + b[idx]
    }
}
```

### Generic Functions — **Production**

Function-level generics work:

```sio
// source: tests/run-pass/closure_higher_order.sio
fn map4(arr: [i64; 4], f: fn(i64) -> i64) -> [i64; 4] with Mut, Panic, Div {
    var out: [i64; 4] = [0; 4]
    var i: i64 = 0
    while i < 4 { out[i] = f(arr[i]); i = i + 1 }
    out
}
```

### Generic Structs — **Planned**

No struct generics yet. `Knowledge<T>` is monomorphic (f64 only):

```sio
// Planned — not yet implemented
struct Pair<T, U> { first: T, second: U }
```

### Closures — **Planned**

No closure literals. Use named function references:

```sio
fn square(x: i64) -> i64 { x * x }
let f = square
let r = f(7)  // 49
```

---

## Next Steps

### Continue Learning
- [Getting Started](getting-started.md) — canonical entry point
- [Cookbook](../COOKBOOK.md) — task-oriented recipes
- [LLM Programming Guide](LLM_PROGRAMMING_GUIDE.md) — definitive syntax reference
- [Gotchas](SOUNIO_GOTCHAS.md) — common mistakes
- [Standard Library](../reference/STDLIB_REFERENCE.md) — API docs

### Get Help
- [FAQ](../FAQ.md) — common questions
- [Glossary](../GLOSSARY.md) — term definitions
- [Minimum Viable Sounio](MINIMUM_VIABLE_SOUNIO.md) — what works today
