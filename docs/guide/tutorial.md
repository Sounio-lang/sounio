<!-- docs:meta
topic_id: repo.docs.guide.tutorial
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A5
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.guide.tutorial
-->

# Sounio Tutorial

A step-by-step guide to learning Sounio, the language for epistemic computing.

> **Canonical epistemic API.** The checked public surface for epistemic values
> is `epistemic::knowledge` (free-fn form: `ep_measured`, `ep_val`, `ep_std`,
> `ep_add`, `ep_mul`, `ep_div`, `ep_merge`, `ep_is_credible`, `ep_*_cov` for
> covariance-aware GUM), anchored by
> `tests/stdlib/epistemic/test_knowledge_madaros_import_e2e.sio` and
> `tests/run-pass/ep_gum_covariance.sio`. The legacy
> `stdlib::epistemic::lib` surface (`epistemic_std`, `add_epistemic`,
> `mul_epistemic`, `fuse_measurements`) is not exercised by `tests/run-pass/`
> and is not part of the checked artifact. `with_confidence` operators and
> units-as-type-parameters are aspirational in source and absent from the
> checked public surface — see `docs/compiler/KNOWN_LIMITATIONS.md`.

## Table of Contents

1. [Getting Started](#1-getting-started)
2. [Basic Syntax](#2-basic-syntax)
3. [Epistemic Types](#3-epistemic-types)
4. [Effect System](#4-effect-system)
5. [Units of Measure](#5-units-of-measure)
6. [Scientific Computing](#6-scientific-computing)
7. [Advanced Features](#7-advanced-features)

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
fn main() -> i32 {
    print("Hello, Sounio!")
    0
}
```

Run it:

```bash
"$SOUC_BIN" run hello.sio
```

---

## 2. Basic Syntax

### Variables

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

**Key Difference from Rust**: Sounio uses `var` for mutable variables, not `let mut`.

### Functions

```sio
// Simple function
fn add(a: i32, b: i32) -> i32 {
    a + b  // Implicit return
}

// With effects
fn read_file(path: string) -> string with IO {
    // IO effect tracks side effects
    let content = fs.read_to_string(path)
    content
}

// Multiple return values
fn divmod(a: i32, b: i32) -> (i32, i32) {
    (a / b, a % b)
}
```

### Control Flow

```sio
// If expressions
let max = if x > y { x } else { y }

// While loops
var i = 0
while i < 10 {
    print(i)
    i = i + 1
}

// For loops
for item in array {
    print(item)
}

// Pattern matching
match result {
    Ok(value) => print("Success: ", value),
    Err(e) => print("Error: ", e),
}
```

### Data Structures

```sio
// Structs
struct Point {
    x: f64,
    y: f64,
}

let p = Point { x: 1.0, y: 2.0 }
print(p.x, p.y)

// Enums
enum Result<T, E> {
    Ok(T),
    Err(E),
}

// Arrays
let numbers = [1, 2, 3, 4, 5]
let first = numbers[0]
```

---

## 3. Epistemic Types

This is where Sounio shines. Every measurement in science has uncertainty—Sounio makes it explicit.

### Basic Epistemic Type

```sio
use epistemic::knowledge::{Epistemic, ep_measured, ep_val, ep_std, ep_confidence}

// Create a measurement with uncertainty.
// ep_measured stores variance = std_dev^2 and confidence = 900 (medium).
let mass = ep_measured(10.5, 0.2)   // val=10.5, std_dev=0.2, confidence=900/1000

print("Mass: ", ep_val(&mass), " ± ", ep_std(&mass), " kg")
print("Confidence: ", ep_confidence(&mass), "/1000  (", ep_confidence(&mass) / 10, "%)")

// Note: the source/instrument lives in stdlib/epistemic/provenance.sio,
// not as a field of Epistemic.
```

### Automatic Propagation

Uncertainty propagates automatically through calculations:

```sio
use epistemic::knowledge::{Epistemic, ep_mul, ep_val, ep_std}

// Struct-literal form: variance = std_dev^2, confidence integer 0..1000.
let length = Epistemic { val: 5.0, variance: 0.01,  confidence: 900 }
let width  = Epistemic { val: 3.0, variance: 0.0025, confidence: 900 }

// Area = length * width (GUM delta method via ep_mul, uncorrelated).
let area = ep_mul(&length, &width)
// Var(area) = w^2 * Var(L) + l^2 * Var(W) + Var(L)*Var(W)
//         = 9 * 0.01    + 25 * 0.0025   + 0.01 * 0.0025
//         ≈ 0.1525,  sigma ≈ 0.39
print("Area: ", ep_val(&area), " ± ", ep_std(&area))
```

### Confidence-Based Execution

```sio
use epistemic::knowledge::{Epistemic, ep_is_credible, ep_confidence}

fn administer_drug(dose: Epistemic) with IO {
    if ep_is_credible(&dose, 950) {
        // High confidence - proceed automatically
        inject(dose)
    } else if ep_is_credible(&dose, 800) {
        // Medium confidence - require confirmation
        if confirm("Confidence is ", ep_confidence(&dose), "/1000. Proceed?") {
            inject(dose)
        }
    } else {
        // Low confidence - reject
        error("Dose confidence too low: ", ep_confidence(&dose))
    }
}
```

### Provenance Tracking

```sio
use epistemic::knowledge::{Epistemic, ep_mul, ep_val}

let measurement1 = Epistemic { val: 100.0, variance: 25.0, confidence: 900 }

// Scalar arithmetic on an Epistemic takes the bare value (use epistemic fns for variance propagation).
let result = measurement1 * 2.0
print("Result: ", ep_val(measurement1) * 2.0)
```

> Note: the `Source` struct with `instrument` / `calibration_date` / `operator`
> fields, and `result.provenance.instrument`, are **not** in the checked
> surface. Provenance is tracked separately in `stdlib/epistemic/provenance.sio`;
> `Epistemic` itself carries only `val`, `variance`, and `confidence`.

---

## 4. Effect System

Sounio uses algebraic effects to track side effects in the type system.

### Common Effects

```sio
// IO - Input/output operations
fn write_log(msg: string) -> () with IO {
    fs.write("log.txt", msg)
}

// Mut - Mutable state
fn increment(x: &! i32) -> () with Mut {
    *x = *x + 1
}

// Async - Asynchronous operations
fn fetch_data(url: string) -> string with Async {
    http.get(url).await
}

// Panic - Can panic/error
fn divide(a: i32, b: i32) -> i32 with Panic {
    if b == 0 {
        panic("Division by zero")
    }
    a / b
}
```

### Effect Combinations

```sio
// Multiple effects
fn process_file(path: string) -> Result<Data> with IO, Panic {
    let content = fs.read_to_string(path)  // IO
    parse_data(content)  // Panic if invalid
}
```

### Effect Handlers

```sio
// Custom effect handlers (advanced)
effect Log {
    fn log(msg: string) -> ()
}

fn compute() -> i32 with Log {
    do Log.log("Starting computation")
    let result = 42
    do Log.log("Computation complete")
    result
}

// Handle the effect
let result = handle compute() {
    Log.log(msg) => {
        print("[LOG] ", msg)
        resume(())
    }
}
```

---

## 5. Units of Measure

Sounio has first-class support for physical units, preventing dimensional errors at compile time.

### Basic Units (Quantity form)

> Units-as-type-spellets (`let distance: m = 100.0`, `fn f(x: kg) -> N`) are
> **aspirational**. The checked surface is `Quantity` + `dim_*()` constructors
> from `stdlib/units/lib.sio`; see `tests/stdlib/units/test_units_stdlib.sio`.

```sio
use units::lib::*   // dim_mass, dim_length, dim_time, quantity_new, quantity_div, ...

let distance = quantity_new(100.0, 0.0, dim_length())  // meters
let time     = quantity_new(10.0,  0.0, dim_time())     // seconds
let velocity = quantity_div(distance, time)

// Compile-time unit checking (via dim_eq / quantity_is_compatible).
let mass  = quantity_new(5.0, 0.0, dim_mass())
let force = quantity_mul(mass, quantity_new(9.8, 0.0, dim_acceleration()))

// Dimension mismatch:
let invalid = quantity_add(distance, time)
// quantity_is_compatible(distance, time) == false
```

### Custom Units (Quantity form)

> `mg`, `mL`, `mg*h/L`, `L/h` as type-spellets are **aspirational**. The
> checked surface is `Quantity` with `dim_*()` dimensions
> (`stdlib/units/lib.sio`).

```sio
use units::lib::*;

let dose         = quantity_new(500.0, 0.0, dim_mass())
let volume       = quantity_new(250.0, 0.0, dim_volume())
let concentration = quantity_div(dose, volume)

// Units in function signatures — pass Quantity directly.
fn calculate_clearance(dose: Quantity, auc: Quantity) -> Quantity
    with Mut, Div, Panic
{
    quantity_div(dose, auc)
}
```

### Unit Conversions

```sio
use units::lib::*   // exports both dim_* and convert_*_to_* free functions.

let distance_m     = 1000.0                     // bare f64 (number space)
let distance_cm_equiv = convert_m_to_cm(distance_m)   // 100000.0

let temp_c = 25.0
let temp_k = convert_celsius_to_kelvin(temp_c)        // 298.15
let temp_f_equiv = convert_kelvin_to_celsius(temp_k)  // 25.0
```

> `convert(value)` is not in the checked surface. The shipped conversions are
> named free functions: `convert_m_to_cm`, `convert_kg_to_g`,
> `convert_celsius_to_kelvin`, etc. See `stdlib/units/lib.sio`.

---

## 6. Scientific Computing

### Epistemic Arithmetic

```sio
use epistemic::knowledge::{
    Epistemic, ep_measured, ep_add, ep_mul, ep_sqrt_ep, ep_val, ep_std
}

// Measurements with uncertainty (canonical free-fn form).
let x = ep_measured(10.0, 0.5)
let y = ep_measured(5.0,  0.2)

// All operations propagate uncertainty via GUM delta method.
let sum      = ep_add(&x, &y)                  // Var(sum) = Var(x)+Var(y)  (uncorrelated)
let product  = ep_mul(&x, &y)                  // Var(prod) = y^2 Var(x) + x^2 Var(y) + ...
let sqrt_x   = ep_sqrt_ep(&x)                 // Var(sqrt(x)) = Var(x)/(4 x)
let exp_val  = /* std::lib/epistemic/propagate.sio::ep_exp */ f64::exp(ep_val(&x))

print("sqrt(x) = ", ep_val(&sqrt_x), " ± ", ep_std(&sqrt_x))
```

> Note: `Knowledge::new(...)` is aspirational; the canonical constructor is
> `Epistemic { val, variance, confidence }` (`tests/run-pass/ep_gum_covariance.sio`)
> or `ep_measured(val, std_dev)`. `sqrt`/`exp` are provided as `ep_sqrt_ep` and
> `exp` on bare `f64`; `sqrt` is not a free fn on `Epistemic`.

### ODE Solvers

> **Aspirational surface — not in the checked artifact.** `stdlib::ode` is not
> present in `git ls-files stdlib/ode*` and is tracked in
> `docs/compiler/KNOWN_LIMITATIONS.md`. The shipped source-tracked propagation
> lives in `stdlib/epistemic/affine` (anchor: `tests/run-pass/affine_shared_source_add.sio`,
> `affine_product_delta.sio`). For closed-form physics, see
> `stdlib/physics/mechanics` (`kinetic_energy_q`, `hookean_force_q`, etc.,
> exercised in `tests/stdlib/physics/test_mechanics_e2e.sio`).

### Linear Algebra

> **Aspirational surface — not in the checked artifact.** `stdlib::linalg` is
> not present in `git ls-files stdlib/linalg*`. The shipped GPU/back-end matrix
> primitives are in `stdlib/gpu/clifford_kernel.sio` and the clifford-kernel
> helpers (`cl_gpu_mul_batch`, `sed_f3_batch`, `cd_gpu_count_tk`); see
> `examples/gpu/vec_add.sio` and `examples/gpu.sio` for the canonical kernel
> surface.

### Signal Processing

> **Aspirational surface — not in the checked artifact.** `stdlib::signal`
> is not present in `git ls-files stdlib/signal*`. The shipped GPU FFT lives in
> `stdlib/gpu/fft.sio`. For host-side numeric transforms of epistemic values,
> use `ep_sqrt_ep`, `ep_square`, `ep_merge`, and `ep_*_cov` from
> `epistemic::knowledge` (anchor: `tests/run-pass/ep_gum_covariance.sio`).

---

## 7. Advanced Features

### Refinement Types

```sio
// Refinement types add logical predicates
type Positive = { x: i32 | x > 0 }
type Even = { x: i32 | x % 2 == 0 }

fn sqrt(x: Positive) -> f64 {
    // Compiler ensures x > 0
    math.sqrt(x as f64)
}

// ERROR at compile time
// sqrt(-5)  // Type error: -5 is not Positive
```

### Linear Types

> **Aspirational syntax.** `linear struct FileHandle { … }` is not in the
> checked source surface. The shipped source-tracked ownership semantics live in
> `stdlib/epistemic/affine` (anchor: `tests/run-pass/affine_shared_source_add.sio`).
> The `linear` keyword on a `struct` is not part of the checked artifact and is
> tracked in `docs/compiler/KNOWN_LIMITATIONS.md`. For file handle ownership in
> source, see `stdlib/coordination/fleet_transaction.sio` and the `linear_ad`
> module of `stdlib/autodiff/linear_ad.sio`.

```sio
struct FileHandle {
    fd: i32
}

fn close(handle: FileHandle) {
    os.close(handle.fd)
}

let file = open("data.txt")
close(file)
// Source-tracked ownership is unenforced in source; the compiler emits E249
// for access-after-move at the IR level.
```

### GPU Computing

```sio
use gpu::*   // exports kernel fn marker, gpu_thread_id_x, perform GPU.{launch,sync}

// Mark the function as a GPU kernel; effect `GPU` declares GPU execution.
kernel fn vector_add(n: i64, a: &[f64], b: &[f64], c: &![f64])
    with GPU, Div, Panic
{
    // Each thread covers one element (runnable anchor: examples/gpu/vec_add.sio).
    let i = gpu_thread_id_x()
    if i < n {
        c[i as usize] = a[i as usize] + b[i as usize]
    }
}

// Host-side dispatch (canonical pattern):
perform GPU.launch(vector_add)
    on (grid: [n], block: [64])
    with (args: (n, a, b, c))
perform GPU.sync()
```

### Generic Programming

```sio
// Generic functions
fn map<T, U>(array: [T], f: fn(T) -> U) -> [U] {
    let result = []
    for item in array {
        result.push(f(item))
    }
    result
}

// Generic structs
struct Pair<T, U> {
    first: T,
    second: U,
}

// Trait constraints
trait Numeric {
    fn add(self, other: Self) -> Self
    fn mul(self, other: Self) -> Self
}

fn dot_product<T: Numeric>(a: [T], b: [T]) -> T {
    let sum = T::zero()
    for i in 0..a.len() {
        sum = sum + a[i] * b[i]
    }
    sum
}
```

---

## Next Steps

### Continue Learning
- **[Programming Guide](programming.md)** - Complete reference
- **[Standard Library](../reference/STDLIB_REFERENCE.md)** - API documentation
- **[Examples](../../examples/)** - Real-world code

### Start Building
- Try the [medical examples](../../examples/medlang/) for PK/PD modeling
- Explore [GPU examples](../../examples/gpu/) for high-performance computing
- Check [fMRI examples](../../examples/fmri/) for neuroimaging

### Get Help
- **[FAQ](../FAQ.md)** - Common questions
- **[Glossary](../GLOSSARY.md)** - Term definitions
- **[GitHub Issues](https://github.com/sounio-lang/sounio/issues)** - Bug reports & questions

---

*Welcome to epistemic computing. Now go build something that knows its own uncertainty.*
