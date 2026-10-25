<!-- docs:meta
topic_id: repo.docs.guide.tutorial
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
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
> `with_confidence` operators are not the checked public surface.
> Derived unit annotations (`f64<m/s>`, `mol/cm3`) are closed on the ledger
> (KL-13) and have named tests. That is not the same as `let d: m = 100.0`,
> which default Madaros still rejects (#2752). See section 5 and
> `docs/compiler/KNOWN_LIMITATIONS.md`.

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

The committed `bin/souc` prebuilt is a Linux x86-64 ELF. Apple Silicon has a separate current-source native ARM64 route; build and install its artifact as described in the [macOS ARM64 guide](../compiler/MADAROS_MACOS_ARM64.md) before using the launcher there. The commands below use the committed Linux prebuilt unless stated otherwise. `check` does not execute the generated program.

### Installation

```bash
git clone https://github.com/sounio-lang/sounio.git
cd sounio

export SOUC_BIN="$(pwd)/bin/souc"
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
"$SOUC_BIN" --version
"$SOUC_BIN" check examples/hello.sio
```

### Your First Program

The canonical program is `examples/hello.sio`:

```sio
fn main() -> i32 with IO {
    println("Hello, Sounio")
    return 0
}
```

On Linux x86-64:

```bash
"$SOUC_BIN" run examples/hello.sio
```

---

## 2. Basic Syntax

These forms are the small-program surface the claim registry marks stable (`core.syntax`, `core.execution`, `core.structs`) and that `CLAUDE.md` §7 records as enforced or accepted. Payload enum variants are not in that surface (KL-20). Closures are not a safe user surface.

### Variables

```sio
// Immutable by default
let x = 42
let name = "Sounio"

// Mutable with 'var', not `let mut`
var counter = 0
counter = counter + 1

// Type annotations
let age: i32 = 25
let pi: f64 = 3.14159
```

**Key difference from Rust**: Sounio uses `var` for mutable variables. Exclusive references are `&!T`, not `&mut T`. There are no Rust macros: write `println("hi")` and `assert(cond)`, not `println!` or `assert!`.

### Functions

```sio
fn add(a: i32, b: i32) -> i32 {
    a + b
}

fn greet() -> i32 with IO {
    println("Hello, Sounio")
    return 0
}
```

`fs.read_to_string` and a `string` type are not the checked surface. Do not copy them from older drafts of this tutorial.

### Control Flow

```sio
let max = if x > y { x } else { y }

var i = 0
while i < 10 {
    i = i + 1
}

for i in 0..10 { }
for i in 0..=10 { }
```

### Data Structures

```sio
struct Point {
    x: f64,
    y: f64,
}

let p = Point { x: 1.0, y: 2.0 }

impl Point {
    fn get_x(self: &Point) -> f64 { self.x }
}

let numbers: [i64; 5] = [1, 2, 3, 4, 5]
let first = numbers[0]
```

Payload-free enums are usable. `enum Result<T, E> { Ok(T), Err(E) }` is not: variants with payloads have no runtime representation on either engine (KL-20, `TOUR.md` §6). Do not match on `Ok(value)`.

---

## 3. Epistemic Types

This is where Sounio shines. Every measurement in science has uncertainty—Sounio makes it explicit.

### Basic Epistemic Type

`print` and `println` take one argument. A call like `print("Mass: ", ep_val(&mass))` is `error[E010]`. `ep_std` requires the caller to declare `Mut, Div, Panic` (`error[E035]` if they are missing). Provenance is not a field of `Epistemic`; `stdlib/epistemic/prov.sio` declares no `pub` symbols.

```sio
use epistemic::knowledge::{ep_measured, ep_std, ep_confidence}

fn main() -> i32 with IO, Mut, Div, Panic {
    // ep_measured stores variance = std_dev^2 and confidence = 900.
    let mass = ep_measured(10.5, 0.2)
    print("mass std = ")
    print_f64(ep_std(&mass))
    println("")
    // ep_confidence(&mass) is 900. Do not return it from main:
    // souc run uses that i32 as the process status (900 % 256 = 132).
    let _ = ep_confidence(&mass)
    return 0
}
```

### Automatic Propagation

Inside one expression, and through a *pure helper*, first-order uncertainty propagates by the GUM delta method. A pure helper is an `f64` or struct return whose body is `let`s plus a tail, with no loop, `match`, or closure (KL-11, narrowed 2026-10-13; witnesses `gum_fo_inline_helper_1706.sio` and `gum_fo_struct_helpers_1706.sio`). A general user call still drops the channel: `tests/run-pass/gum_fo_across_call.sio` keeps a `//@ known-failure`. Do not write code that assumes every function call preserves variance.

The intro demo that the tour measured on Madaros is `demos/intro/demo_incerteza.sio` (`measure`, `variance_of`). The library form below is the checked `epistemic::knowledge` surface.

```sio
use epistemic::knowledge::{Epistemic, ep_mul, ep_val, ep_std}

fn main() -> i32 with IO, Mut, Div, Panic {
    // Struct-literal form: variance = std_dev^2, confidence integer 0..1000.
    let length = Epistemic { val: 5.0, variance: 0.01, confidence: 900 }
    let width = Epistemic { val: 3.0, variance: 0.0025, confidence: 900 }
    // Uncorrelated GUM delta. Var(area) = w^2*Var(L) + l^2*Var(W) = 0.1525.
    let area = ep_mul(&length, &width)
    print("area = ")
    print_f64(ep_val(&area))
    print(" std = ")
    print_f64(ep_std(&area))
    println("")
    return 0
}
```

### Confidence-Based Execution

```sio
use epistemic::knowledge::{Epistemic, ep_is_credible, ep_confidence}

fn administer_drug(dose: Epistemic) with IO {
    if ep_is_credible(&dose, 950) {
        println("high")
    } else if ep_is_credible(&dose, 800) {
        println("medium")
    } else {
        println("low")
    }
}
```

`ep_is_credible` and `ep_confidence` are the checked names. `inject`, `confirm`, and `error` are not.

### Scaling Epistemic Values

```sio
use epistemic::knowledge::{Epistemic, ep_scale, ep_val}

fn main() -> i32 with IO {
    let measurement1 = Epistemic { val: 100.0, variance: 25.0, confidence: 900 }
    let result = ep_scale(&measurement1, 2.0)
    print_f64(ep_val(&result))
    println("")
    return 0
}
```

> Note: the `Source` struct with `instrument` / `calibration_date` / `operator`
> fields, and `result.provenance.instrument`, are **not** in the checked
> surface. A standalone provenance model exists in `stdlib/epistemic/prov.sio`
> (private — it declares no `pub` symbols — so it cannot be imported as a
> user-facing module); `Epistemic` itself carries only `val`, `variance`, and
> `confidence`.

---

## 4. Effect System

Sounio tracks side effects in the type. The checked cases are `IO`, `Mut`, `Div`, and `Panic` (`effects.subtyping` and `effects.diagnostics` in the claim registry). Declare them with `with`. There is no checked `fs` module, no `http.get(url).await`, and no user-facing `effect` / `handle` block in this tutorial. Those snippets were removed because they are not backed by a `run-pass` test.

```sio
fn greet() -> i32 with IO {
    println("Hello, Sounio")
    return 0
}

fn increment(x: &!i32) with Mut {
    *x = *x + 1
}

fn divide(a: i32, b: i32) -> i32 with Div, Panic {
    if b == 0 {
        panic("Division by zero")
    }
    a / b
}
```

A function that calls another must declare at least the effects it uses. `println` needs `IO`.

---

## 5. Units of Measure

Units are a fixture-backed prototype (`units.measure` in the claim registry). They are not a finished dimensional type system, and the two engines do not agree.

- **Default Madaros** does not accept the intro demo. Measured 2026-10-15: `souc check demos/intro/demo_unidades.sio` exits 1 with `error[E001]` "this binding expects a different type" (expected `f64`, found `f64`) on `let distance: m = 100.0` (#2752). The diagnostic does not name the unit.
- **lean_single** runs that demo: `unit velocity = m / s` type-checks, and `m + s` is rejected. A multiply assigned to a derived unit is still accepted and prints a wrong-dimension number (#2751). Do not treat lean_single units as sound.
- **KL-13 is closed** for derived declarations and derived annotations (`f64<m/s>`, `mol/cm3`), with pins `tests/run-pass/unit_derived_annotation_mol_per_cm3.sio`, `tests/compile-fail/unit_derived_annotation_refuse_add.sio`, `tests/frontend/unit_derived_velocity_decl_current_source.sio`, and `tests/frontend/unit_f64_unit_expr_velocity_current_source.sio`. Closing KL-13 did not close #2752.

The runtime `Quantity` + `dim_*()` form below is the complementary representation. It checks compatibility at runtime inside `quantity_add`, not as the static `m` / `s` spelling.

### Basic Units (Quantity form)

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

These notes name source files. They were not re-run on 2026-10-15. Do not treat a directory listing as a passing gate.

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

// print takes one argument. ep_std needs Mut, Div, Panic on the caller.
print_f64(ep_val(&sqrt_x))
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

The blocks below are not the checked onboarding surface. Refinements fall back to runtime checks, and `souc run` does not enforce `where` (#2753). Closures and trait bounds are not a safe user surface. `TOUR.md` §6 measured GPU emission as an empty-body skeleton, not the kernel below. Read the status table in [programming.md](programming.md) before copying any of this section.

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

Do not copy a `kernel fn` from this tutorial. Two records disagree, and this pass did not re-run either:

- `TOUR.md` §6, measured 2026-10-04: `souc build --backend gpu` emits PTX only for kernels with empty bodies.
- `CLAUDE.md` §13, older note: `examples/kernel_vec_add.sio` was reported as valid PTX under Madaros.
- Root `README.md`, reconciled 2026-10-11: scoped Spark evidence for GPU/Pireus; the generic GPU CLI needs its own status.

### Generic Programming

Generics are prototype in the claim registry. The block below is not the stable surface. Payload-bearing return types are KL-20.

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
- Check [connectivity examples](../../examples/connectivity/) for brain-network analysis

### Get Help
- **[FAQ](../FAQ.md)** - Common questions
- **[Glossary](../GLOSSARY.md)** - Term definitions
- **[GitHub Issues](https://github.com/sounio-lang/sounio/issues)** - Bug reports & questions

---

*Welcome to epistemic computing. Now go build something that knows its own uncertainty.*
