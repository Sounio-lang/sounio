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
> `epistemic::lib` surface (`epistemic_std`, `add_epistemic`) is exercised by
> `tests/stdlib/epistemic/test_core_e2e.sio` (a `//@ run-pass` test collected by
> `scripts/dev/run_sio_test_suite.sh`) but is a legacy surface separate from the
> canonical `epistemic::knowledge` API. `mul_epistemic` and `fuse_measurements`
> are part of `epistemic::lib` but are not exercised by that test
> (`fuse_measurements` is private; `mul_epistemic` is public but untested).
> `with_confidence` operators and
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

// Note: provenance is a separate, private model in `stdlib/epistemic/prov.sio`
// (it declares no `pub` symbols), not a field of `Epistemic`.
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
// Var(area) = w^2 * Var(L) + l^2 * Var(W)   (GUM delta, uncorrelated: no Var(L)·Var(W) term)
//         = 9 * 0.01    + 25 * 0.0025
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

### Scaling Epistemic Values

```sio
use epistemic::knowledge::{Epistemic, ep_mul, ep_val, ep_scale}

let measurement1 = Epistemic { val: 100.0, variance: 25.0, confidence: 900 }

// Scale an Epistemic by a scalar with ep_scale (propagates variance via GUM δ-method).
let result = ep_scale(&measurement1, 2.0)
print("Result: ", ep_val(&result))
```

> Note: the `Source` struct with `instrument` / `calibration_date` / `operator`
> fields, and `result.provenance.instrument`, are **not** in the checked
> surface. A standalone provenance model exists in `stdlib/epistemic/prov.sio`
> (private — it declares no `pub` symbols — so it cannot be imported as a
> user-facing module); `Epistemic` itself carries only `val`, `variance`, and
> `confidence`.

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

> Units-as-type spellings (`let distance: m = 100.0`, `unit N; fn f(x: kg) -> N`) and
> derived units (`f64<m/s>`) ARE part of the checked surface and are exercised by
> current-source tests (`tests/frontend/unit_derived_velocity_decl_current_source.sio`,
> `tests/frontend/unit_f64_unit_expr_velocity_current_source.sio`). The runtime
> `Quantity` + `dim_*()` form is the complementary representation.

```sio
use units::lib::*   // dim_mass, dim_length, dim_time, quantity_new, quantity_div, ...

let distance = quantity_new(100.0, 0.0, dim_length())  // meters
let time     = quantity_new(10.0,  0.0, dim_time())     // seconds
let velocity = quantity_div(distance, time)

// Runtime unit compatibility (checked by quantity_add via assert(dim_eq) /
// quantity_is_compatible).
let mass  = quantity_new(5.0, 0.0, dim_mass())
let force = quantity_mul(mass, quantity_new(9.8, 0.0, dim_acceleration()))

// Dimension mismatch panics at runtime inside quantity_add (assert dim_eq):
let invalid = quantity_add(distance, time)  // runtime panic, not a compile-time error
```

### Custom Units (Quantity form)

> `mg` is supported as a unit spelling and tested in `tests/run-pass/unit_same_add.sio`
> (which declares and uses only `mg`); `mL` is declared in `stdlib/units/pharmacological.sio`.
> The `unit Clearance = L/h`
> and `mg*h/L` derived-declaration syntax exists but is not yet covered by a
> run-pass fixture. The runtime `Quantity` with `dim_*()` dimensions
> (`stdlib/units/lib.sio`) is the complementary representation.

```sio
use units::lib::*;

let dose    = quantity_new(0.0005, 0.0, dim_mass())              // 500 mg = 5e-4 kg
let volume  = quantity_new(2.5e-4, 0.0, UnitDim {
    mass: 0, length: 3, time: 0, temperature: 0, amount: 0, current: 0, luminosity: 0,
})                                                                 // 250 mL = 2.5e-4 m^3
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
let temp_c_equiv = convert_kelvin_to_celsius(temp_k)  // 25.0 (round-trip back to °C)
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
use epistemic::propagate::{exp}

// Measurements with uncertainty (canonical free-fn form).
let x = ep_measured(10.0, 0.5)
let y = ep_measured(5.0,  0.2)

// All operations propagate uncertainty via GUM delta method.
let sum      = ep_add(&x, &y)                  // Var(sum) = Var(x)+Var(y)  (uncorrelated)
let product  = ep_mul(&x, &y)                  // Var(prod) = y^2 Var(x) + x^2 Var(y)  (uncorrelated)
let sqrt_x   = ep_sqrt_ep(&x)                 // Var(sqrt(x)) = Var(x)/(4 x)
let exp_val  = exp(x)                          // Epistemic -> Epistemic via GUM delta propagation

print("sqrt(x) = ", ep_val(&sqrt_x), " ± ", ep_std(&sqrt_x))
```

> Note: `Knowledge::new(...)` is aspirational; the canonical constructor is
> `Epistemic { val, variance, confidence }` (`tests/run-pass/ep_gum_covariance.sio`)
> or `ep_measured(val, std_dev)`. Epistemic-aware `sqrt`/`exp` are provided as
> `ep_sqrt_ep` (in `epistemic::knowledge`) and `exp` (in `epistemic::propagate`);
> there is no `f64::exp` builtin and no scalar `*` operator on `Epistemic`.

### ODE Solvers

> **Source-tracked surface.** `stdlib::ode` is present in the repository as a
> set of submodules — `rk4.sio`, `tsit5.sio`, `bdf.sio`, `epistemic.sio`,
> `solver.sio`, `solvers.sio`, and `pbpk3_stable.sio` — that provide RK4, RK45,
> Tsit5, BDF, epistemic integration, and PBPK sources. Note: `stdlib/ode/lib.sio`
> itself only re-exports the epistemic PK-fit functions (`epistemic_pk_fit`,
> `epistemic_pkpd_fit`); import the solvers from their defining submodules, not
> from `lib.sio` as a solver entry point: the RK4 solver lives in `ode::rk4`
> (`rk4_step`, `rk4_integrate`), and the adaptive (RK45) and BDF wrappers live in
> `ode::solver` (`solve_rk45_exp_decay`, `solve_bdf1_exp_decay`,
> `default_options`). Note that `ode::tsit5` only exports its test `main` and
> `ode::bdf` exports no public functions, so do not import solvers from
> `ode::tsit5` or `ode::bdf`. The shipped
> source-tracked uncertainty propagation
> lives in `stdlib/epistemic/affine` (anchor: `tests/run-pass/affine_shared_source_add.sio`,
> `affine_product_delta.sio`).

### Linear Algebra

> **Source-tracked surface.** `stdlib::linalg` is present (`matrix.sio`,
> `vector.sio`, `eigen.sio`, `factorize.sio`) with host-side matrix/vector and
> eigendecomposition APIs. GPU-accelerated matrix primitives also live in
> `stdlib/gpu/clifford_kernel.sio` (`cl_gpu_mul_batch`, `cd_gpu_count_tk`) and
> `stdlib/gpu/sedenion_kernels.sio` (`sed_f3_batch`); see
> `examples/gpu/vec_add.sio` and `examples/gpu.sio` for the canonical kernel
> surface.

### Signal Processing

> **Source-tracked surface.** `stdlib::signal` is present and
> `stdlib/signal/lib.sio` publicly re-exports FFT construction, forward/inverse
> transforms, magnitude, phase, power spectrum, and epistemic FFT APIs. The
> shipped GPU FFT also lives in `stdlib/gpu/fft.sio`. For general numeric
> transforms of epistemic values (e.g. `ep_sqrt_ep`, `ep_square`, `ep_merge`,
> `ep_*_cov` from `epistemic::knowledge`), see
> `tests/run-pass/ep_gum_covariance.sio` — these are scalar epistemic arithmetic
> helpers, not signal transforms; the canonical FFT/signal API is the
> `signal::lib` re-export shown above.

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

> **Checked surface.** `linear struct FileHandle { … }` is part of the checked
> source surface — `tests/run-pass/linear_balanced_branches.sio` uses it
> directly, and shipped runtime code uses linear structs extensively. The
> `linear` keyword enforces single-use ownership at check time (the checker emits
> `E039` for use-after-consumption and `E040` for an unconsumed linear value).
> For file handle ownership in source, see `stdlib/coordination/fleet_transaction.sio`
> (which uses `linear struct`); for a minimal linear-struct field-access example, see
> `tests/run-pass/linear_struct_field_access.sio`.

```sio
linear struct FileHandle {
    fd: i32
}

fn close(handle: FileHandle) {
    os.close(handle.fd)
}

let file = open("data.txt")
close(file)
// Source-tracked ownership is enforced at check time; the compiler emits E039
// for use-after-consumption (and E040 if a linear value is left unconsumed).
```

### GPU Computing

```sio
use gpu::lib::*   // re-exports Clifford/sedenion GPU helpers (e.g. cl_gpu_mul_batch, sed_f3_batch)
// kernel fn, gpu_thread_id_x, and perform GPU.{launch,sync} are compiler surfaces, not gpu::lib::* exports

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

// Host-side dispatch (canonical pattern, tests/run-pass/gpu_launch_vec_slices.sio):
// reference params are launched as &a, &b, and &!c (exclusive output).
fn main() -> i32 with IO, Mut, Panic, GPU, Div {
    let n: i64 = 64
    let grid = (1, 1, 1)
    let block = (64, 1, 1)

    var a: [f64; 64] = [0.0; 64]
    var b: [f64; 64] = [0.0; 64]
    var c: [f64; 64] = [0.0; 64]

    perform GPU.launch(vector_add, grid, block)(n, &a, &b, &!c)
    perform GPU.sync()
    0
}
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
