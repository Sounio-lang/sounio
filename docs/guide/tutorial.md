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

export SOUC_BIN="$(pwd)/artifacts/omega/souc-bin/souc-linux-x86_64-jit"
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

### Basic Knowledge Types

```sio
import stdlib.epistemic::*

// Create a measurement with uncertainty
let mass = Knowledge::new(
    value: 10.5,           // kg
    uncertainty: 0.2,      // ± 0.2 kg
    confidence: 0.95,      // 95% confidence interval
    source: "scale_lab_1"
)

print("Mass: ", mass.value, " ± ", mass.uncertainty, " kg")
print("Confidence: ", mass.confidence * 100.0, "%")
```

### Automatic Propagation

Uncertainty propagates automatically through calculations:

```sio
let length = Knowledge::new(5.0, uncertainty: 0.1)
let width = Knowledge::new(3.0, uncertainty: 0.05)

// Area calculation with automatic uncertainty propagation
let area = length * width

// Uncertainty is calculated using GUM (Guide to Uncertainty in Measurement)
print("Area: ", area.value, " ± ", area.uncertainty)
// Output: Area: 15.0 ± 0.35
```

### Confidence-Based Execution

```sio
fn administer_drug(dose: Knowledge<mg>) with IO {
    if dose.confidence > 0.95 {
        // High confidence - proceed automatically
        inject(dose)
    } else if dose.confidence > 0.80 {
        // Medium confidence - require confirmation
        if confirm("Confidence is ", dose.confidence, ". Proceed?") {
            inject(dose)
        }
    } else {
        // Low confidence - reject
        error("Dose confidence too low: ", dose.confidence)
    }
}
```

### Provenance Tracking

```sio
let measurement1 = Knowledge::new(
    value: 100.0,
    uncertainty: 5.0,
    source: Source {
        instrument: "Spectrometer-A",
        calibration_date: "2025-01-15",
        operator: "Dr. Smith",
    }
)

// Provenance is preserved through calculations
let result = measurement1 * 2.0
print("Result source: ", result.provenance.instrument)
```

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

### Basic Units

```sio
import stdlib.units::*

// Declare quantities with units
let distance: m = 100.0    // meters
let time: s = 10.0         // seconds
let velocity = distance / time  // Type: m/s

// Compile-time unit checking
let mass: kg = 5.0
let force: N = mass * 9.8  // N = kg⋅m/s²

// ERROR: Type mismatch
// let invalid = distance + time  // Can't add meters to seconds!
```

### Custom Units

```sio
// Pharmacology example
let dose: mg = 500.0
let volume: mL = 250.0
let concentration = dose / volume  // Type: mg/mL

// Units in function signatures
fn calculate_clearance(dose: mg, auc: mg*h/L) -> L/h {
    dose / auc
}
```

### Unit Conversions

```sio
import stdlib.units::conversions::*

let distance_m: m = 1000.0
let distance_km: km = convert(distance_m)  // 1.0 km

let temp_c: celsius = 25.0
let temp_f: fahrenheit = convert(temp_c)  // 77.0°F
```

---

## 6. Scientific Computing

### Epistemic Arithmetic

```sio
import stdlib.epistemic::*
import stdlib.math::*

// Measurements with uncertainty
let x = Knowledge::new(10.0, uncertainty: 0.5)
let y = Knowledge::new(5.0, uncertainty: 0.2)

// All operations propagate uncertainty
let sum = x + y
let product = x * y
let sqrt_x = sqrt(x)
let exp_x = exp(x)

print("sqrt(x) = ", sqrt_x.value, " ± ", sqrt_x.uncertainty)
```

### ODE Solvers

```sio
import stdlib.ode::*

// Define a differential equation: dy/dt = -k*y
fn exponential_decay(t: f64, y: f64, k: f64) -> f64 {
    -k * y
}

// Solve from t=0 to t=10
let solution = solve_ode(
    f: exponential_decay,
    y0: 100.0,
    t_span: (0.0, 10.0),
    params: (k: 0.1),
    method: RK45
)

for point in solution {
    print("t=", point.t, " y=", point.y)
}
```

### Linear Algebra

```sio
import stdlib.linalg::*

// Matrix operations
let A = matrix([
    [1.0, 2.0],
    [3.0, 4.0]
])

let b = vector([5.0, 6.0])

// Solve Ax = b
let x = solve(A, b)

// Eigenvalues and eigenvectors
let (eigenvalues, eigenvectors) = eig(A)
```

### Signal Processing

```sio
import stdlib.signal::*

// FFT
let signal = [1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0]
let spectrum = fft(signal)

// Filtering
let filtered = lowpass_filter(signal, cutoff: 0.5)

// Convolution
let kernel = [0.25, 0.5, 0.25]
let smoothed = convolve(signal, kernel)
```

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

```sio
// Linear types ensure single ownership
linear struct FileHandle {
    fd: i32
}

fn close(handle: FileHandle) {
    // Consumes handle - can't be used again
    os.close(handle.fd)
}

let file = open("data.txt")
close(file)
// ERROR: file has been moved
// close(file)
```

### GPU Computing

```sio
import stdlib.gpu::*

// Mark function for GPU execution
@gpu
fn matrix_multiply(a: [f32], b: [f32], n: i32) -> [f32] {
    // Executes on GPU
    let result = [0.0; n * n]
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                result[i*n + j] += a[i*n + k] * b[k*n + j]
            }
        }
    }
    result
}

// Automatic kernel generation for CUDA/Metal
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
