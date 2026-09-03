<!-- docs:meta
topic_id: repo.docs.guide.programming
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A5
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.guide.programming
-->

# Sounio Programming Guide for LLMs

This comprehensive guide enables LLMs to correctly generate Sounio code. Sounio is a novel L0 systems + scientific programming language - it is NOT a dialect of Rust, Julia, Python, or any other language.

## Implementation Status (Read First)

> **WARNING**: This document mixes **implemented** and **aspirational** features.
> Sections marked with ⚠️ are NOT YET IMPLEMENTED.

### What Actually Works (Madaros v0.80.0)

| Feature | Status |
|---------|--------|
| Core syntax (variables, functions, control flow) | ✅ Implemented |
| Type system (generics, structs, enums, traits) | ✅ Implemented |
| Effects system (type-level tracking) | ✅ Implemented |
| Units of measure | ✅ Implemented |
| Epistemic types (`Knowledge<T>`) | ✅ Interpreter only |
| GPU kernels (codegen) | ✅ Implemented in the checked GPU artifact |
| GPU kernel launch surface | ✅ Checkable syntax; runtime proof is gate-backed |
| Basic async (spawn, await, channels) | ✅ Implemented |
| Async join/select | ⚠️ Not implemented |
| Refinement types (parsing) | ✅ Implemented |
| Refinement types (SMT verification) | ⚠️ Runtime checks only |
| ODE/PDE DSL | ⚠️ Not implemented |
| Probabilistic programming | ⚠️ Effect stub only |
| Causal models | ⚠️ Not implemented |
| AD jacobian/hessian (CPU) | ⚠️ Stub only |
| Package manager | ⚠️ Feature flag stub |

### Canonical References

For what the compiler accepts today:
- `docs/guide/MINIMUM_VIABLE_SOUNIO.md` — What actually works
- `compiler/docs/KNOWN_LIMITATIONS.md` — What is not implemented
- `tests/run-pass/` — Compilable syntax patterns

If you are generating `.sio` code, prefer the working subset above.

## Table of Contents

1. [Core Syntax](#core-syntax)
2. [Type System](#type-system)
3. [Effects System](#effects-system)
4. [Async Programming](#async-programming)
5. [Units of Measure](#units-of-measure)
6. [GPU Programming](#gpu-programming)
7. [Scientific Computing](#scientific-computing)
8. [Epistemic Types](#epistemic-types)
9. [FFI (Foreign Function Interface)](#ffi-foreign-function-interface)
10. [Standard Library](#standard-library)
    - [I/O Module](#io-module-stdio)
    - [JSON Module](#json-module-stdjson)
    - [String Module](#string-module-stdstr)
    - [Collections](#collections)
    - [Octonion Neural Networks](#octonion-neural-networks-stdonn)
11. [Package Manager](#package-manager)
12. [What is NOT Supported](#what-is-not-supported)
13. [Idiomatic Patterns](#idiomatic-patterns)
14. [Common Mistakes](#common-mistakes)

---

## Core Syntax

### Hello World

```d
fn main() -> i32 {
    return 0
}
```

### Variables

```d
// Immutable binding (preferred)
let x = 5
let y: i32 = 10

// Mutable binding
var count = 0
count = count + 1

// Constants (compile-time)
const PI: f64 = 3.14159265359
```

### Functions

```d
// Basic function
fn add(a: i32, b: i32) -> i32 {
    return a + b
}

// Function with effects (see Effects section)
fn read_config(path: string) -> string with IO {
    // IO operations here
}

// Generic function
fn identity<T>(x: T) -> T {
    return x
}

// Method syntax (on structs)
impl Point {
    fn distance(&self, other: &Point) -> f64 {
        let dx = self.x - other.x
        let dy = self.y - other.y
        return sqrt(dx*dx + dy*dy)
    }
}
```

### Control Flow

```d
// If-else
if condition {
    // code
} else if other_condition {
    // code
} else {
    // code
}

// Match (pattern matching)
match value {
    0 => println("zero"),
    1 | 2 => println("one or two"),
    n if n > 10 => println("big"),
    _ => println("other"),
}

// For-in loops (range iteration)
for i in 0..10 {        // exclusive: 0, 1, ..., 9
    println(i)
}

for i in 0..=10 {       // inclusive: 0, 1, ..., 10
    println(i)
}

// For-in loops (collection iteration)
for item in array {     // iterate over array elements
    process(item)
}

for item in vec {       // iterate over Vec elements
    process(item)
}

// While loops
while condition {
    // code
}

// Loop (infinite, use break to exit)
loop {
    if done {
        break
    }
}
```

### Data Structures

```d
// Structs
struct Point {
    x: f64,
    y: f64,
}

// Create struct instance
let p = Point { x: 1.0, y: 2.0 }

// Linear structs (must be used exactly once)
linear struct FileHandle {
    fd: i32,
}

// Enums
enum Color {
    Red,
    Green,
    Blue,
    Rgb(u8, u8, u8),
}

// Type aliases
type Sequence = [u8]
type Matrix = [[f64]]
```

### References

**IMPORTANT: Sounio uses `&!` for exclusive/mutable references, NOT `&mut`**

```d
// Shared reference (read-only)
fn read_value(x: &i32) -> i32 {
    return *x
}

// Exclusive reference (mutable) - uses &! NOT &mut
fn increment(x: &!i32) {
    *x = *x + 1
}

// In function signatures
fn process(data: &[f64], output: &![f64]) {
    // data is read-only, output is mutable
}
```

### Arrays and Slices

```d
// Fixed-size array
let arr: [i32; 5] = [1, 2, 3, 4, 5]

// Dynamic array (Vec)
let vec: Vec<i32> = [1, 2, 3]

// Slice reference
let slice: &[i32] = &arr[1..4]

// Slice operations (Darwin Atlas syntax)
let head = arr[..k]      // first k elements
let tail = arr[k..]      // from k to end
let mid = arr[a..b]      // elements a to b-1
let all = arr[..]        // all elements

// Array concatenation
let combined = arr1 ++ arr2
```

### Closures

```d
// Basic closure
let add_one = |x: i32| -> i32 { x + 1 }

// Closure with inferred types
let double = |x| x * 2

// Closure in higher-order function
let result = numbers.map(|x| x * 2)

// Multi-line closure
let process = |data: &[f64]| -> f64 {
    var sum = 0.0
    for x in data {
        sum = sum + x
    }
    return sum
}
```

### Modules and Imports

```d
// Module declaration (file-level, optional):
// module mymodule

// Import (both syntaxes work)
import std::io
use std::math       // alias for import

// Path separators: both :: and . work
import std::collections::HashMap
use std.collections.HashMap    // equivalent

// Selective import
import std::io::{read, write}

// Public exports
pub fn exported_function() -> i32 {
    return 42
}
```

---

## Type System

### Primitive Types

```d
// Integers (signed)
i8, i16, i32, i64, i128

// Integers (unsigned)  
u8, u16, u32, u64, u128

// Floating point
f32, f64

// Boolean
bool    // true or false

// Character and string
char    // Unicode scalar
string  // UTF-8 string

// Unit type (no value)
()
```

### Linear and Affine Types

```d
// Linear type: must be used exactly once
linear struct Connection {
    socket: Socket,
}

// Affine type: can be used at most once
affine struct TempBuffer {
    data: &[u8],
}

// Move semantics
fn consume(conn: Connection) {
    // conn is consumed here
}

// Copy vs Move
// Copy types are implicit for primitives; no derive attribute syntax.
struct Point { x: f64, y: f64 }
```

### Generics

```d
// Generic struct
struct Container<T> {
    value: T,
}

// Generic function
fn swap<T>(a: &!T, b: &!T) {
    let temp = *a
    *a = *b
    *b = temp
}

// Trait bounds
fn print<T: Display>(x: T) {
    println(x.to_string())
}

// Where clauses
fn complex<T, U>(x: T, y: U) -> T
where
    T: Clone + Debug,
    U: Into<T>,
{
    // implementation
}
```

### Refinement Types

> ⚠️ **PARTIAL**: Refinement types parse and typecheck, but SMT verification is not
> implemented. Non-trivial predicates become runtime checks, not compile-time proofs.

Refinement types encode constraints directly in the type system, verified at compile-time via SMT solving.

```sio
// Basic refinement type with predicate
type Positive = { x: i32 | x > 0 }
type NonEmpty<T> = { arr: [T] | len(arr) > 0 }
type Percentage = { p: f64 | p >= 0.0 && p <= 100.0 }

// Scientific domain refinements
type OrbitRatio = { r: f64 | 0.25 <= r && r <= 1.0 }
type Probability = { p: f64 | 0.0 <= p && p <= 1.0 }
type Concentration = { c: f64 | c >= 0.0 }  // Non-negative

// Function with refined parameter
fn sqrt_safe(x: { n: f64 | n >= 0.0 }) -> f64 {
    return sqrt(x)
}

// Invariants in structs
struct BoundedCounter {
    value: i32,

    invariant value >= 0 && value <= 100
}
```

#### Best Practices for Refinement Types

**1. Extracting Values**

Refinement types coerce to their base type when needed:

```sio
type OrbitRatio = { r: f64 | 0.25 <= r && r <= 1.0 }

fn compute_interval(ratio: OrbitRatio, margin: f64) -> (f64, f64) {
    // ratio coerces to f64 in arithmetic
    let lower = ratio - margin
    let upper = ratio + margin
    return (lower, upper)
}
```

**2. Creating Refined Values**

Use explicit type annotation - the compiler verifies the constraint:

```sio
// Literal values - verified at compile time
let ratio: OrbitRatio = 0.75  // OK: 0.25 <= 0.75 <= 1.0
let bad: OrbitRatio = 0.1     // COMPILE ERROR: violates constraint

// From computed values - verified via SMT
fn normalize(x: f64) -> Probability {
    let clamped = clamp(x, 0.0, 1.0)
    return clamped  // OK: clamp guarantees 0.0 <= result <= 1.0
}
```

**3. Preserving Refinements Through Calculations**

The type checker tracks constraints through operations:

```sio
type Positive = { x: f64 | x > 0.0 }

fn scale(p: Positive, factor: Positive) -> Positive {
    // Compiler proves: positive * positive = positive
    return p * factor
}

fn half(p: Positive) -> Positive {
    // Compiler proves: positive / 2 = positive
    return p / 2.0
}
```

**4. Fallible Conversions**

When a value might not satisfy the constraint, use Option:

```sio
fn try_as_probability(x: f64) -> Option<Probability> {
    if x >= 0.0 && x <= 1.0 {
        return Some(x)
    }
    return None
}

// Usage
let maybe_prob = try_as_probability(user_input)
match maybe_prob {
    Some(p) => use_probability(p),
    None => handle_error(),
}
```

**5. Domain-Specific Refinements**

Create meaningful refinements for your domain:

```sio
// Pharmacokinetics
type Volume = { v: f64 | v > 0.0 }            // L
type Clearance = { cl: f64 | cl > 0.0 }       // L/h
type HalfLife = { t: f64 | t > 0.0 }          // h

// The type system ensures physical validity
fn elimination_rate(cl: Clearance, v: Volume) -> { k: f64 | k > 0.0 } {
    return cl / v  // Compiler proves: positive / positive = positive
}
```

**6. Combining with Epistemic Types**

Refinement types can be wrapped in EpistemicValue for uncertainty tracking:

```sio
use epistemic::{EpistemicValue, from_measurement}

type OrbitRatio = { r: f64 | 0.25 <= r && r <= 1.0 }

// Track uncertainty while preserving domain constraints
let ratio: EpistemicValue<OrbitRatio> = from_measurement(
    0.75,           // measured value (must satisfy OrbitRatio)
    0.02,           // uncertainty (± 2%)
    0.95            // 95% confidence
)

// Uncertainty bounds are automatically clamped to refinement bounds
```

---

## Effects System

Sounio has first-class algebraic effects for tracking computational side effects.

### Built-in Effects

```d
// IO effect - file/network/console operations
fn read_file(path: string) -> string with IO {
    // file operations
}

// Mut effect - mutable state
fn update_state() with Mut {
    // state mutations
}

// Alloc effect - heap allocation
fn create_buffer(size: usize) -> Vec<u8> with Alloc {
    // allocation
}

// Panic effect - may panic/abort
fn divide(a: i32, b: i32) -> i32 with Panic {
    if b == 0 {
        panic("division by zero")
    }
    return a / b
}

// GPU effect - GPU operations
fn gpu_compute() with GPU {
    // GPU operations
}

// Prob effect - probabilistic operations
fn sample_distribution() -> f64 with Prob {
    return sample(Normal(0.0, 1.0))
}

// Async effect - async operations
async fn fetch_data() -> Data with Async {
    // async operations
}
```

### Custom Effects

```d
// Define a custom effect
effect State<T> {
    fn get() -> T;
    fn put(value: T);
}

// Use the effect
fn counter() -> i32 with State<i32> {
    let current = perform State.get()
    perform State.put(current + 1)
    return current
}

// Handle the effect
handler IntState for State<i32> {
    get() => resume(self.value),
    put(v) => {
        self.value = v
        resume(())
    }
}

// Apply handler
fn main() {
    let result = handle {
        counter() + counter() + counter()
    } with IntState { value: 0 }
    // result = 0 + 1 + 2 = 3
}
```

### Effect Handlers

```d
// Exception-like effect
effect Exn {
    fn throw(message: String) -> !;
}

// Handler that converts to Option
fn safe_divide(a: i32, b: i32) -> Option<i32> {
    handle {
        if b == 0 {
            perform Exn.throw("division by zero")
        }
        a / b
    } with {
        throw(msg) => None,
        return(v) => Some(v),
    }
}
```

---

## Async Programming

Sounio has first-class async/await support with structured concurrency.

### Async Functions

```sio
// Async function declaration
async fn fetch_data(url: string) -> Data with Async, IO {
    // Async operations here
}

// Call and await
fn main() with Async, IO {
    let data = fetch_data("https://api.example.com").await
    process(data)
}
```

### Async Blocks

```sio
// Create a future from a block
let future = async {
    // Computation runs when awaited
    expensive_computation()
}

// Await the result
let result = future.await
```

### Spawn (Background Tasks)

```sio
// Spawn a background task
let handle = spawn {
    long_running_task()
}

// Continue with other work...
do_other_work()

// Wait for the spawned task to complete
let result = handle.await
```

### Channels

```sio
// Create a channel for communication
let (sender, receiver) = channel::<Message>()

// Producer task
spawn {
    for item in items {
        sender.send(item).await
    }
}

// Consumer
while let Some(msg) = receiver.recv().await {
    process(msg)
}
```

### Join (Wait for All)

> ⚠️ **ASPIRATIONAL**: `join` is not yet implemented. Use sequential awaits for now.

```sio
// Wait for multiple futures concurrently
let (result1, result2, result3) = join(
    fetch_users(),
    fetch_posts(),
    fetch_comments()
).await

// All three complete before continuing
```

### Select (First Ready)

> ⚠️ **ASPIRATIONAL**: `select` is not yet implemented. Basic async with channels works.

```sio
// Wait for the first future to complete
select {
    msg = receiver.recv() => {
        handle_message(msg)
    }
    result = timeout(5.seconds) => {
        handle_timeout()
    }
    _ = shutdown_signal() => {
        cleanup_and_exit()
    }
}
```

### Async Closures

```sio
// Async closure
let fetch = async |id: i32| -> Result<User, Error> {
    let response = http_get(format("/users/{}", id)).await
    parse_user(response)
}

// Use in higher-order functions
let users = ids.map(fetch).collect().await
```

### Timeout and Cancellation

```sio
// With timeout
let result = timeout(Duration::seconds(30), async {
    slow_operation()
}).await

match result {
    Ok(value) => use_value(value),
    Err(TimeoutError) => handle_timeout(),
}

// Cancellation token
let token = CancellationToken::new()

spawn {
    loop {
        if token.is_cancelled() {
            break
        }
        do_work()
    }
}

// Cancel after some condition
token.cancel()
```

### Effect Tracking

Async operations must declare the `Async` effect:

```sio
// Must declare Async effect
async fn process() -> i32 with Async {
    compute().await
}

// IO operations need both effects
async fn fetch() -> string with Async, IO {
    http_get(url).await
}

// Pure async computation
async fn pure_async() -> i32 with Async {
    42  // No side effects other than async
}
```

---

## Units of Measure

Sounio has built-in dimensional analysis for scientific computing.

### Defining Units

```d
// Literals with units (underscore prefix)
let mass = 500_mg
let volume = 10_mL
let time = 24_h

// Angle brackets syntax (also works)
let dose: mg = 500.0<mg>
let conc: mg/L = 50.0<mg/L>
```

### Unit Arithmetic

```d
// Units are tracked through computations
let dose: mg = 500.0_mg
let volume: L = 0.5_L
let concentration: mg/L = dose / volume  // Type-safe!

// Dimensional mismatch is a compile error
// let wrong = dose + volume  // ERROR: mg + L not allowed

// Unit conversion happens automatically when compatible
let mass_kg: kg = 2.5_kg
let mass_g: g = mass_kg  // Automatic conversion
```

### Common Unit Types

```d
// Mass
mg, g, kg

// Volume
mL, L

// Time
s, min, h, day

// Concentration
mg/L, g/L, mol/L

// Rate
1/h, 1/min, L/h

// Compound units
mL/min, mg/kg, L/h/kg
```

### Units in Structs

```d
struct PKParams {
    cl: L/h,       // Clearance
    v: L,          // Volume of distribution
    ka: 1/h,       // Absorption rate
}

struct Patient {
    weight: kg,
    age: years,
    crcl: mL/min,  // Creatinine clearance
}
```

---

## GPU Programming

### Kernel Definition

```d
kernel fn vector_add(n: i64) with GPU {
}

kernel fn scale_vector(factor: f64, n: i64) with GPU, Div {
}
```

### Checked Public Surface

```d
perform GPU.launch(vector_add, grid, block)(n)
perform GPU.launch(scale_vector, grid, block)(2.0, n)
perform GPU.sync()
```

### Launching Kernels

Use the checked GPU profile when validating GPU syntax:

```d
fn main() with GPU, IO {
    let n: i64 = 1024
    let grid = (16, 1, 1)
    let block = (64, 1, 1)

    perform GPU.launch(vector_add, grid, block)(n)
    perform GPU.sync()
}
```

Validation commands:

```d
"$SOUC_GPU_BIN" check examples/gpu.sio
"$SOUC_GPU_BIN" check tests/run-pass/gpu_launch_surface.sio
"$SOUC_GPU_BIN" build examples/kernel_matmul.sio --backend gpu -o /tmp/kernel_matmul.ptx
```

### Important Limitation

```d
// Older sketches used `gpu.thread_id`, `gpu.block_id`, `gpu.block_dim`,
// and `gpu.alloc`. Those names are not yet resolved by the checked public
// GPU artifact, so do not present them as the default public syntax.
```

---

## Scientific Computing

### ODE Systems

> ⚠️ **ASPIRATIONAL**: ODE DSL syntax is not implemented. Use stdlib/ode functions directly.

```d
ode ExponentialDecay {
    state {
        y: f64,
    }
    
    params {
        k: 1/s,    // decay constant
    }
    
    equations {
        dy/dt = -k * y
    }
    
    initial {
        y = 100.0,
    }
}
```

### Probabilistic Programming

> ⚠️ **ASPIRATIONAL**: Probabilistic effect is type-level only. No inference runtime.

```d
fn bayesian_inference() with Prob {
    // Prior distribution
    let theta = sample(Beta(1.0, 1.0))
    
    // Likelihood
    for observation in data {
        observe(Bernoulli(theta), observation)
    }
    
    return theta
}

// Run inference
let posterior = infer(bayesian_inference, method: MCMC(samples: 10000))
```

### Causal Models

> ⚠️ **ASPIRATIONAL**: Causal DSL syntax is not implemented. Dead code in compiler.

```d
causal DrugEffect {
    nodes {
        dose: f64,
        concentration: f64,
        effect: f64,
    }
    
    edges {
        dose -> concentration,
        concentration -> effect,
    }
    
    equations {
        concentration = dose / volume,
        effect = sigmoid(concentration - ec50),
    }
}

// Interventional query
let result = do(DrugEffect, dose = 100.0).query(effect)

// Counterfactual query
let cf = counterfactual(DrugEffect, 
    factual: { dose: 50.0, effect: 0.3 },
    intervention: { dose: 100.0 }
).query(effect)
```

### Automatic Differentiation

> ⚠️ **PARTIAL**: `grad()` works on GPU. CPU `jacobian`/`hessian` are stubs.

```d
// Dual numbers for forward-mode AD
let x = dual(3.0, 1.0)  // value = 3, derivative seed = 1
let y = x * x           // y.value = 9, y.deriv = 6

// Gradient computation
fn f(x: f64) -> f64 {
    return x * x * x - 2.0 * x
}

let gradient = grad(f, at: 2.0)  // df/dx at x=2 = 3*4 - 2 = 10

// Jacobian (GPU only)
let jacobian_matrix = jacobian(vector_function, at: point)

// Hessian (GPU only)
let hessian_matrix = hessian(scalar_function, at: point)
```

### Linear Algebra Types

```d
// Vectors
let v2: vec2 = vec2(1.0, 2.0)
let v3: vec3 = vec3(1.0, 2.0, 3.0)
let v4: vec4 = vec4(1.0, 2.0, 3.0, 4.0)

// Matrices
let m2: mat2 = mat2(
    1.0, 0.0,
    0.0, 1.0
)
let m3: mat3 = mat3::identity()
let m4: mat4 = mat4::rotation_x(angle)

// Quaternion
let q: quat = quat::from_axis_angle(axis, angle)

// Operations
let dot_product = v3.dot(other_v3)
let cross = v3.cross(other_v3)
let transformed = m4 * v4
```

---

## Epistemic Types

Sounio tracks knowledge, confidence, and provenance at the type level.

### Knowledge Type

```d
// Knowledge with metadata
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

### Uncertainty Propagation

```d
struct Uncertain {
    mean: f64,
    std: f64,
}

fn multiply_uncertain(a: Uncertain, b: Uncertain) -> Uncertain {
    let mean_product = a.mean * b.mean
    let std_product = sqrt(
        (b.mean * a.std) * (b.mean * a.std) +
        (a.mean * b.std) * (a.mean * b.std)
    )
    return Uncertain { mean: mean_product, std: std_product }
}

// Using +- operator
let value = 100.0 +- 2.5  // 100 with uncertainty 2.5
```

### Provenance Tracking

```d
// Provenance sources
let observed_data = Source::Measured("lab_id_123")
let derived_value = Source::Computed(formula, inputs)
let literature_value = Source::Literature("DOI:10.1234/paper")

// Attach provenance
let result: Knowledge[f64] = compute_result()
    .with_provenance(Derived {
        from: [input1, input2],
        method: "linear_regression"
    })
```

---

## FFI (Foreign Function Interface)

Sounio supports C-compatible FFI through raw pointer types and extern blocks.

### Raw Pointer Types

```d
// Const raw pointer (read-only)
let ptr: *const i32 = null_ptr()

// Mutable raw pointer (read-write)
let mut_ptr: *mut i32 = null_mut()

// Type inference works too
let ptr = null_ptr()
let mut_ptr = null_mut()
```

### Extern Blocks (C FFI)

```d
// Declare external C functions
extern "C" {
    fn malloc(size: i64) -> *mut i8;
    fn free(ptr: *mut i8);
    fn strlen(s: *const i8) -> i64;
    fn memcpy(dest: *mut i8, src: *const i8, n: i64) -> *mut i8;
}

// System calls
extern "C" {
    fn open(path: *const i8, flags: i32) -> i32;
    fn read(fd: i32, buf: *mut i8, count: i64) -> i64;
    fn write(fd: i32, buf: *const i8, count: i64) -> i64;
    fn close(fd: i32) -> i32;
}
```

### Pointer Operations

```d
// Create null pointers
let const_ptr = null_ptr()      // *const ()
let mut_ptr = null_mut()        // *mut ()

// Check for null
if is_null(ptr) {
    print("Pointer is null\n")
}

// Compare pointers
if ptr_eq(ptr1, ptr2) {
    print("Pointers are equal\n")
}

// Get address as integer
let addr: i64 = ptr_addr(ptr)

// Create pointer from address
let ptr = ptr_from_addr(1024)        // *const ()
let mut_ptr = ptr_from_addr_mut(1024) // *mut ()

// Pointer arithmetic
let next = ptr_offset(ptr, 8)    // Offset by 8 bytes
let elem = ptr_add(ptr, 1)       // Add 1 element
let prev = ptr_sub(ptr, 1)       // Subtract 1 element

// Pointer difference
let diff: i64 = ptr_diff(ptr2, ptr1)

// Pointer casting
let const_ptr = as_const(mut_ptr)    // *mut T -> *const T
let mut_ptr = as_mut(const_ptr)      // *const T -> *mut T (unsafe!)

// Type information
let size = size_of()     // Size of type in bytes
let align = align_of()   // Alignment of type
```

### FFI Best Practices

```d
// Always check for null before using pointers
fn safe_strlen(s: *const i8) -> i64 {
    if is_null(s) {
        return 0
    }
    return strlen(s)
}

// Pair allocations with deallocations
fn with_buffer() {
    let buf = malloc(1024)
    if is_null(buf) {
        panic("allocation failed")
    }
    // ... use buffer ...
    free(buf)
}
```

---

## Standard Library

Sounio includes a standard library with modules for I/O, JSON, strings, and more.

### I/O Module (`std.io`)

```d
import io::*;

// Read entire file as string
let content = read_file("data.txt")?;

// Write string to file
write_file("output.txt", "Hello, world!")?;

// Append to file
append_file("log.txt", "New entry\n")?;

// Check if file exists
if file_exists("config.json") {
    // ...
}

// Exit process
exit(0);  // Success
exit(1);  // Error

// Environment variables
let args = env::args();           // Command line arguments
let home = env::var("HOME");      // Get env var
env::set_var("MY_VAR", "value");  // Set env var
let cwd = env::current_dir()?;    // Current directory

// Standard streams
print("Hello");
println("Hello with newline");
eprint("Error message");
eprintln("Error with newline");
let line = read_line()?;

// Path utilities
let full = path::join("dir", "file.txt");  // "dir/file.txt"
let name = path::file_name("/home/user/file.txt");  // Some("file.txt")
let parent = path::parent("/home/user/file.txt");   // Some("/home/user")
let ext = path::extension("file.txt");              // Some("txt")
```

### JSON Module (`std.json`)

```d
import json::*;

// Parse JSON string
match parse_json(r#"{"name": "test", "value": 42}"#) {
    Ok(json) => {
        // Type checking
        if json.is_object() {
            // Access by key
            let name = json["name"].as_str().unwrap_or("default");
            let value = json["value"].as_i64().unwrap_or(0);
            
            // Check key existence
            if json.has("optional") {
                let opt = json["optional"];
            }
        }
    },
    Err(e) => println("Parse error: " ++ e.message()),
}

// JSON value types
let null_val = JsonValue::null();
let bool_val = JsonValue::bool(true);
let num_val = JsonValue::number(3.14);
let str_val = JsonValue::string("hello");
let arr_val = JsonValue::array();
let obj_val = JsonValue::object();

// Value extraction methods
json.as_bool()   // -> Option<bool>
json.as_f64()    // -> Option<f64>
json.as_i64()    // -> Option<i64>
json.as_str()    // -> Option<&str>
json.as_array()  // -> Option<&Vec<JsonValue>>
json.as_object() // -> Option<&Map<String, JsonValue>>

// Array access
let first = json[0];
let len = json.len();
json.push(JsonValue::number(42));

// Object manipulation
json.set("key", JsonValue::string("value"));
json.remove("key");

// Path-based access
let nested = json.path("users[0].name");

// Serialize back to string
let s = json.to_json_string();
let pretty = json.to_json_string_pretty();

// Parse JSON Lines format
let records = parse_jsonl(content)?;
```

### String Module (`std.str`)

```d
import str::*;

// String checks
if s.is_empty() { }
let length = s.len();

// Trimming
let trimmed = "  hello  ".trim();        // "hello"
let left = "  hello".trim_start();       // "hello"
let right = "hello  ".trim_end();        // "hello"

// Splitting
let parts: Vec<&str> = "a,b,c".split(',').collect();
let lines: Vec<&str> = content.lines().collect();

// Searching
if s.contains("needle") { }
if s.starts_with("prefix") { }
if s.ends_with("suffix") { }
let pos = s.find("pattern");      // Option<usize>
let last = s.rfind("pattern");

// Replacement
let new_s = s.replace("old", "new");

// Case conversion
let upper = s.to_uppercase();
let lower = s.to_lowercase();

// Repetition
let repeated = "ab".repeat(3);  // "ababab"

// Parsing
let n: i32 = "42".parse().unwrap();
let f: f64 = "3.14".parse().unwrap();

// Iteration
for c in s.chars() { }
for (i, c) in s.char_indices() { }
for b in s.bytes() { }

// String building
var builder = String::new();
builder.push('c');
builder.push_str("string");

// Concatenation (++ operator)
let combined = "hello" ++ " " ++ "world";
```

### Comparison Module (`std.cmp`)

```d
import cmp::*;

// Min and max
let smaller = min(10, 20);   // 10
let larger = max(10, 20);    // 20

// Clamp to range
let clamped = clamp(15, 0, 10);  // 10
let clamped = clamp(-5, 0, 10);  // 0
let clamped = clamp(5, 0, 10);   // 5

// Partial ordering for floats (handles NaN)
let result = min_partial(1.0, 2.0);  // Some(1.0)
let nan = 0.0 / 0.0;
let result = min_partial(nan, 1.0);  // None

// Ordering enum
match a.cmp(&b) {
    Ordering::Less => println("a < b"),
    Ordering::Equal => println("a == b"),
    Ordering::Greater => println("a > b"),
}
```

### Collections Module (`std.collections`)

```d
import collections::*;

// Vec (dynamic array)
var vec: Vec<i32> = Vec::new();
vec.push(1);
vec.push(2);
vec.extend([3, 4, 5].iter());
let first = vec[0];
let len = vec.len();

// HashMap
var map: HashMap<String, i32> = HashMap::new();
map.insert("key", 42);
let value = map.get("key");

// HashSet
var set: HashSet<i32> = HashSet::new();
set.insert(1);
if set.contains(&1) { }

// Deque (double-ended queue)
var deque: Deque<i32> = Deque::new();
deque.push_back(1);
deque.push_front(0);
let front = deque.pop_front();
```

### Octonion Neural Networks (`std.onn`)

**Sounio is the first language with a complete Octonion Neural Network (ONN) framework.** Octonions are 8-dimensional hypercomplex numbers that enable neural networks with **8x parameter compression** while preserving expressiveness through non-associative algebra.

#### Core Concepts

**Octonion Type:**
```sio
struct Octonion {
    w: f32,  // scalar (real) part
    x: f32,  // i component
    y: f32,  // j component
    z: f32,  // k component
    t: f32,  // l component
    u: f32,  // il component
    v: f32,  // jl component
    s: f32,  // kl component
}
```

**Use ONN when:**
- Building rotation-heavy networks (3D, robotics, computer vision)
- Memory/parameter budget is critical (mobile, edge devices)
- You need group-theoretic structure (SO(3) preservation)
- Model compression is more important than raw speed

#### Basic Operations

```sio
use std::onn

fn main() {
    // Create octonions
    let o1 = Octonion { w: 1.0, x: 0.0, y: 0.0, z: 0.0, t: 0.0, u: 0.0, v: 0.0, s: 0.0 }

    // Core operations (via GPU intrinsics)
    let conj = oct_conj(o1)           // Conjugate
    let norm_sq = oct_norm_sq(o1)     // Squared norm
    let normalized = oct_normalize(o1) // Unit normalization
    let inv = oct_inv(o1)             // Multiplicative inverse

    // Composition (Cayley-Dickson, 120 FLOPs)
    let o2 = Octonion { w: 0.5, x: 0.5, y: 0.5, z: 0.5, t: 0.5, u: 0.5, v: 0.5, s: 0.5 }
    let product = oct_mul(o1, o2)

    // Dot product (inner product of 8D vectors)
    let dot = oct_dot(o1, o2)
}
```

#### Linear Layers

```sio
use std::onn::{Octonion, OctLinearLayer}
use std::onn::linear::{oct_linear_forward, oct_xavier_init}

fn main() {
    // Create layer (8D → 16D semantic, = 8 octonions → 2 octonions)
    let layer = OctLinearLayer { input_features: 8, output_features: 2 }

    // Xavier initialization
    let weights = oct_xavier_init(8, 2, seed: 42)  // 16 octonion parameters
    let bias = oct_zero_init(2)                     // 2 octonions

    // Forward pass
    let input: [Octonion] = [o1, o2, o3, o4, o5, o6, o7, o8]
    let output = oct_linear_forward(layer, weights, input, bias)  // [Octonion; 2]
}
```

#### Convolutional Layers

```sio
use std::onn::conv::{oct_conv2d_forward, oct_conv2d_backward, oct_conv_xavier_init}

fn main() {
    // 3×3 Conv2d: 3 octonion channels → 16 octonion channels
    let layer = OctConv2dLayer {
        in_channels: 3,
        out_channels: 16,
        kernel_h: 3,
        kernel_w: 3,
        stride: 1,
        padding: 1,
    }

    // Initialize kernels with Xavier
    let kernels = oct_conv_xavier_init(3, 16, 3, 3, seed: 42)
    let bias = oct_zero_init(16)

    // Forward pass
    let output = oct_conv2d_forward(layer, kernels, input, bias)

    // Backward pass (returns grad_input, grad_kernel, grad_bias)
    let (grad_input, grad_kernel, grad_bias) = oct_conv2d_backward(
        layer, kernels, input, grad_output
    )
}
```

#### Activations

```sio
use std::onn::activation::{oct_relu, oct_gelu, oct_sigmoid, oct_tanh}
use std::onn::g2_activation::{oct_gelu_g2, oct_leaky_relu_g2}

fn forward_pass(x: [Octonion]) -> [Octonion] {
    // Component-wise activations (apply to each of 8 components)
    let after_relu = oct_relu(x)
    let after_gelu = oct_gelu(x)       // GELU: x * sigmoid(1.702 * x)
    let after_sigmoid = oct_sigmoid(x)

    // G2-equivariant activations (preserve rotation structure)
    // Only apply to norm, preserves direction
    let g2_gelu = oct_gelu_g2(x)
    let g2_leaky = oct_leaky_relu_g2(x, alpha: 0.1)

    after_gelu
}
```

#### Loss Functions

```sio
use std::onn::loss::{oct_mse_loss, oct_mae_loss, oct_cross_entropy_loss}

fn compute_loss(predictions: [Octonion], targets: [Octonion]) -> f32 {
    // MSE loss: (1/n) * sum |pred - target|^2
    let mse = oct_mse_loss(predictions, targets)

    // MAE loss: (1/n) * sum ||pred - target||
    let mae = oct_mae_loss(predictions, targets)

    // Cross-entropy (for classification with octonion logits)
    let ce = oct_cross_entropy_loss(predictions, targets, num_classes: 10)

    mse
}
```

#### Optimizers

```sio
use std::onn::optimizer::{
    SgdOptimizer, AdamOptimizer, RmspropOptimizer,
    create_sgd_optimizer, create_adam_optimizer, create_rmsprop_optimizer,
    sgd_step, adam_step, rmsprop_step
}

fn train_step(parameters: &![Octonion], gradients: &[Octonion]) {
    // SGD with momentum
    var sgd_opt = create_sgd_optimizer(
        num_params: 16,
        learning_rate: 0.01,
        momentum: 0.9,
        weight_decay: 0.0001
    )
    sgd_step(sgd_opt, parameters, gradients)

    // Adam optimizer
    var adam_opt = create_adam_optimizer(
        num_params: 16,
        learning_rate: 0.001,
        beta1: 0.9,
        beta2: 0.999,
        epsilon: 1e-8,
        weight_decay: 0.0
    )
    adam_step(adam_opt, parameters, gradients)

    // RMSprop
    var rmsprop_opt = create_rmsprop_optimizer(
        num_params: 16,
        learning_rate: 0.01,
        alpha: 0.99,
        epsilon: 1e-8,
        weight_decay: 0.0
    )
    rmsprop_step(rmsprop_opt, parameters, gradients)
}
```

#### Normalization Layers

```sio
use std::onn::normalization::{
    oct_bn_fwd, oct_bn_bwd,
    oct_layer_norm_fwd, oct_layer_norm_bwd,
    oct_group_norm_fwd, oct_group_norm_bwd,
    oct_instance_norm_fwd, oct_instance_norm_bwd
}

fn normalization_forward(x: [Octonion], gamma: [Octonion], beta: [Octonion]) {
    // Batch normalization (forward pass)
    let (normalized, mean, rstd) = oct_bn_fwd(x, batch_size: 32, num_features: 16)

    // Batch normalization backward
    let (grad_x, grad_gamma, grad_beta) = oct_bn_bwd(
        x, mean, rstd, grad_output, batch_size: 32, num_features: 16
    )

    // Layer normalization
    let ln_out = oct_layer_norm_fwd(x, gamma, beta, normalized_shape: 16, epsilon: 1e-5)

    // Group normalization (8 octonions × 4 groups)
    let gn_out = oct_group_norm_fwd(x, gamma, beta, num_groups: 4, epsilon: 1e-5)

    // Instance normalization (useful for style transfer)
    let in_out = oct_instance_norm_fwd(x, gamma, beta, epsilon: 1e-5)
}
```

#### Multi-Head Attention

```sio
use std::onn::attention::{OctAttention, create_attention, oct_attention_forward}

fn attention_layer() {
    let attn = create_attention(
        dim: 64,          // 8 octonions (8 components each = 64 F32 dims)
        num_heads: 4,
        dropout_p: 0.1,
        seed: 42
    )

    // Input: batch of 8 octonion sequences
    let query: [Octonion] = [...]
    let key: [Octonion] = [...]
    let value: [Octonion] = [...]

    // Causal mask for autoregressive models
    let mask: [f32] = [...]  // 0 for masked positions, 1 for valid

    let output = oct_attention_forward(
        attention_state, attn,
        query, key, value,
        mask
    )
}
```

#### Training Loop

```sio
use std::onn::training::{
    TrainingState, LrScheduler, DataLoader,
    create_training_state, create_lr_scheduler, create_dataloader,
    update_training_metrics, reset_epoch_metrics, get_learning_rate, step_scheduler
}

fn training_loop() {
    let mut state = create_training_state()
    let mut scheduler = create_lr_scheduler(
        initial_lr: 0.001,
        schedule_type: 1,  // 0=const, 1=linear decay, 2=exp, 3=cosine
        total_steps: 1000,
        decay_rate: 0.1
    )

    for epoch in 0..100 {
        reset_epoch_metrics(state)

        let loader = create_dataloader(num_samples: 10000, batch_size: 32, shuffle: true)

        for batch in 0..get_num_batches(loader) {
            let indices = get_batch_indices(loader)

            // Get current learning rate
            let lr = get_learning_rate(scheduler)

            // Forward pass
            let predictions = forward(batch_data)
            let loss = compute_loss(predictions, batch_labels)

            // Backward pass
            let gradients = backward(loss)

            // Optimizer step
            adam_step(optimizer, parameters, gradients)

            // Update metrics
            update_training_metrics(state, loss)
            step_scheduler(scheduler)
        }

        let epoch_loss = get_epoch_loss(state)
        print("Epoch " ++ epoch ++ ": " ++ epoch_loss)
    }
}
```

#### GPU Acceleration

The repo contains substantial GPU backend work, but contributor docs should tie
execution claims back to the checked GPU artifact and Omega GPU artifacts:

```sio
"$SOUC_GPU_BIN" build examples/kernel_matmul.sio --backend gpu -o /tmp/kernel_matmul.ptx
```

Use `docs/features/GPU_RUNTIME.md` for the current artifact-backed GPU contract
before claiming automatic runtime execution behavior.

#### Important Notes

1. **Non-Associativity**: Octonion multiplication is not associative: `(a*b)*c ≠ a*(b*c)`. All operations preserve left-to-right evaluation order.

2. **Moufang Identities**: While not fully associative, octonions satisfy Moufang identities: `a*(b*(a*c)) = (a*b*a)*c`. This ensures algebraic stability.

3. **Component-wise vs G2-equivariant**:
   - Component-wise: Apply activation to each of 8 components independently
   - G2-equivariant: Apply to norm only, preserve direction (rotation-aware)

4. **Parameter Counting**: A single octonion has semantic expressiveness of ~8 float32 parameters through 8D structure.

5. **Expected Performance**:
   - CPU: 2.5-3.5x slower than float32 (Cayley-Dickson overhead)
   - GPU: Parity to 1.2x faster (with fused kernels)
   - Model size: 8-10x smaller (8 components per parameter)

#### Example: 3D Rotation Prediction Network

```sio
use std::onn

fn rotation_network(input_poses: [Octonion]) -> [Octonion] {
    // Input: batch of 3D rotations as octonions

    // Layer 1: 8 octonions → 16 octonions
    let w1 = oct_xavier_init(8, 16, seed: 1)
    let b1 = oct_zero_init(16)
    let h1 = oct_linear_forward(layer1, w1, input_poses, b1)
    let h1_act = oct_gelu_g2(h1)  // G2-equivariant (preserves rotations)

    // Layer 2: 16 octonions → 4 octonions (output = rotation)
    let w2 = oct_xavier_init(16, 4, seed: 2)
    let b2 = oct_zero_init(4)
    let output = oct_linear_forward(layer2, w2, h1_act, b2)
    let output_norm = oct_normalize(output)  // Ensure unit rotation

    output_norm
}
```

For comprehensive benchmarks and performance analysis, see [docs/research/ONN_BENCHMARK_ANALYSIS.md](../research/ONN_BENCHMARK_ANALYSIS.md).

---

## Package Manager

> Current release contract: local package support only. There is no launched
> public package registry and no supported publish/login/search workflow.

Local package support is available through the checked `tools/sounio-pkg/sounio-pkg` wrapper and the compiler's local package-import resolver. The supported surface is:

- create a local package scaffold
- type-check package source files
- run local package tests
- import checked local packages from `packages/*`

### Project Setup

Create a new project with a `sounio.toml` manifest:

```toml
[package]
name = "my-project"
version = "0.1.0"
description = "A Sounio project"

[dependencies]
local-lib = { path = "../local-lib" }
```

### CLI Commands

```bash
# Create a local package
tools/sounio-pkg/sounio-pkg new my-project

# Build/check/test the local package
cd my-project
../tools/sounio-pkg/sounio-pkg build
../tools/sounio-pkg/sounio-pkg check
../tools/sounio-pkg/sounio-pkg test
```

### Workspaces

Workspace metadata is draft-only. Keep multi-package projects as explicit local packages until a workspace gate is added:

```toml
# Root Sounio.toml
[workspace]
members = [
    "core",
    "cli",
    "lib/*"
]

[workspace.dependencies]
serde = "1.0"
```

```toml
# core/Sounio.toml
[package]
name = "my-core"
version = "0.1.0"

[dependencies]
serde.workspace = true  # Inherit from workspace
```

### Lock File

Dependencies are locked in `Sounio.lock`:

```toml
# This file is auto-generated by Sounio
# Do not edit manually

[[package]]
name = "serde"
version = "1.0.193"
checksum = "sha256:abc123..."

[[package]]
name = "tokio"
version = "1.35.0"
checksum = "sha256:def456..."
dependencies = ["mio", "bytes"]
```

### Private Registries

Configure custom registries in `~/.sounio/config.toml`:

```toml
[registries]
my-company = { url = "https://registry.company.com" }

[registries.my-company]
token = "env:COMPANY_REGISTRY_TOKEN"
```

Then use in `Sounio.toml`:

```toml
[dependencies]
internal-lib = { version = "1.0", registry = "my-company" }
```

### Credentials

Credentials are stored in `~/.sounio/credentials.toml`:

```toml
[registries.default]
token = "your-api-token"

[registries.my-company]
token = "company-token"
username = "your-username"
```

---

## What is NOT Supported

**CRITICAL: The following syntax/features do NOT exist in Sounio L0:**

### Rust Syntax That Does NOT Work

```d
// WRONG: &mut does not exist
fn wrong(x: &mut i32) { }     // ERROR
// CORRECT: use &!
fn correct(x: &!i32) { }      // OK

// WRONG: impl blocks without struct name
impl SomeTrait for Type { }   // Currently limited support

// WRONG: ? operator for error handling (not implemented)
let result = might_fail()?    // ERROR
// CORRECT: use match or handle
let result = match might_fail() {
    Ok(v) => v,
    Err(e) => return Err(e)
}

// WRONG: derive macros (limited)
// derive(Debug, Clone)        // NOT supported

// WRONG: async/await in all contexts
async fn foo() { }            // Only with Async effect

// WRONG: lifetime annotations
fn foo<'a>(x: &'a str) { }    // ERROR - no lifetimes
```

### Features Not Implemented

```d
// NO: Macros (limited support)
macro_rules! my_macro { }     // NOT supported
assert!(condition)            // NOT supported
assert_eq!(a, b)              // NOT supported

// NO: Tuple destructuring in closures
arr.map(|(x, y)| x + y)       // ERROR
// WORKAROUND: use explicit indexing
arr.map(|pair| pair.0 + pair.1)

// NO: Pattern matching in let
let (a, b) = tuple            // ERROR
// WORKAROUND:
let a = tuple.0
let b = tuple.1

// NO: Test attributes
// test                        // NOT supported
// cfg(test)                   // NOT supported

// NO: u2 or other non-standard integer sizes
let x: u2 = 3                 // ERROR - u2 doesn't exist
// WORKAROUND: use u8 and mask
let x: u8 = 3 & 0b11

// NO: Trait objects (limited)
let x: Box<dyn Trait>         // Limited support

// NO: Associated types (limited)
type Item = i32               // In trait context - limited
```

### Syntax Differences from Rust

```d
// Semicolons: optional at end of expressions
let x = 5      // OK (no semicolon)
let x = 5;     // OK (with semicolon)

// Return: explicit return preferred
fn foo() -> i32 {
    return 42   // Preferred
    // 42       // Also works (implicit return)
}

// Type annotations: use same syntax
let x: i32 = 5         // OK
let x = 5<i32>         // OK (type suffix syntax)
let x = 5.0<f64>       // OK for floats with units

// Effect annotation: use 'with' keyword
fn foo() -> i32 with IO { }   // Sounio style
// NOT: fn foo() -> Result<i32, Error>  // No Result-based effects
```

---

## Idiomatic Patterns

### Error Handling

```d
// Option for optional values
fn find(arr: &[i32], target: i32) -> Option<i32> {
    for i in 0..len(arr) {
        if arr[i] == target {
            return Some(i)
        }
    }
    return None
}

// Result for fallible operations
fn parse_int(s: string) -> Result<i32, ParseError> {
    // ...
}

// Effect-based error handling
effect Fail {
    fn fail(msg: string) -> !;
}

fn safe_operation() -> i32 with Fail {
    if bad_condition {
        perform Fail.fail("something went wrong")
    }
    return 42
}
```

### Resource Management

```d
// Linear types ensure cleanup
linear struct Connection {
    handle: *mut c_void,
}

impl Connection {
    fn close(self) {
        // self is consumed, cleanup happens
        unsafe { ffi_close(self.handle) }
    }
}

// Cannot forget to close:
fn use_connection() {
    let conn = Connection::open()
    // ... use conn ...
    conn.close()  // MUST be called
}
```

### Iteration Patterns (For-In Loops)

Sounio supports `for-in` loops for both range iteration and collection iteration.

#### Range Iteration

```d
// Exclusive range: 0, 1, 2, ..., n-1
for i in 0..n {
    println(i)
}

// Inclusive range: 0, 1, 2, ..., n
for i in 0..=n {
    println(i)
}

// Range with arbitrary start
for i in 5..10 {
    // i = 5, 6, 7, 8, 9
}

// Unbounded range (use with break)
for i in 0.. {
    if i >= limit {
        break
    }
    process(i)
}
```

#### Collection Iteration

```d
// Array iteration
let arr = [10, 20, 30]
for x in arr {
    println(x)  // prints 10, 20, 30
}

// Vec iteration
let vec: Vec<i32> = [1, 2, 3, 4]
for x in vec {
    println(x)
}

// Reference iteration (borrows the collection)
for x in &arr {
    println(x)
}
```

#### Loop Control

```d
// Break exits the loop early
for i in 0..100 {
    if i >= 10 {
        break
    }
}

// Continue skips to next iteration
for i in 0..10 {
    if i % 2 == 0 {
        continue  // skip even numbers
    }
    println(i)  // prints 1, 3, 5, 7, 9
}
```

#### Index-Based Iteration

```d
// When you need the index
for i in 0..len(arr) {
    arr[i] = arr[i] * 2
}

// Nested loops
for i in 0..rows {
    for j in 0..cols {
        matrix[i][j] = i * cols + j
    }
}
```

#### Functional Style

```d
let doubled = numbers.map(|x| x * 2)
let sum = numbers.fold(0, |acc, x| acc + x)
let filtered = numbers.filter(|x| x > 0)
```

### Struct Patterns

```d
// Constructor pattern
struct Point {
    x: f64,
    y: f64,
}

impl Point {
    fn new(x: f64, y: f64) -> Self {
        return Point { x: x, y: y }
    }
    
    fn origin() -> Self {
        return Point { x: 0.0, y: 0.0 }
    }
}

// Builder pattern for complex structs
struct Config {
    timeout: i32,
    retries: i32,
    verbose: bool,
}

impl Config {
    fn builder() -> ConfigBuilder {
        return ConfigBuilder::new()
    }
}

struct ConfigBuilder {
    timeout: i32,
    retries: i32,
    verbose: bool,
}

impl ConfigBuilder {
    fn new() -> Self {
        return ConfigBuilder {
            timeout: 30,
            retries: 3,
            verbose: false,
        }
    }
    
    fn timeout(&!self, t: i32) -> &!Self {
        self.timeout = t
        return self
    }
    
    fn build(&self) -> Config {
        return Config {
            timeout: self.timeout,
            retries: self.retries,
            verbose: self.verbose,
        }
    }
}
```

### Scientific Computing Patterns

```d
// Uncertainty-aware calculations
fn calculate_with_uncertainty(
    measured: Uncertain,
    reference: Uncertain
) -> Uncertain {
    // Propagate uncertainty through calculation
    let ratio = divide_uncertain(measured, reference)
    return scale_uncertain(ratio, 100.0)  // Percentage
}

// Unit-safe calculations
fn drug_concentration(
    dose: mg,
    volume: L,
    bioavailability: f64
) -> mg/L {
    return (dose * bioavailability) / volume
}

// GPU launch syntax accepted by the checked artifact
fn parallel_process(n: i64) with GPU, IO {
    let grid = ((n + 255) / 256, 1, 1)
    let block = (256, 1, 1)

    perform GPU.launch(process_kernel, grid, block)(n)
    perform GPU.sync()
}
```

---

## Common Mistakes

### 1. Using &mut Instead of &!

```d
// WRONG
fn increment(x: &mut i32) { }

// CORRECT
fn increment(x: &!i32) { }
```

### 2. Forgetting Effect Annotations

```d
// WRONG - missing IO effect
fn print_value(x: i32) {
    println(x)  // ERROR: IO effect not declared
}

// CORRECT
fn print_value(x: i32) with IO {
    println(x)
}
```

### 3. Using Rust Macros

```d
// WRONG
assert!(x > 0)
println!("value: {}", x)

// CORRECT
if x <= 0 {
    panic("x must be positive")
}
println("value: " + x.to_string())
```

### 4. Assuming Implicit Semicolons Work Like Rust

```d
// Both work in Sounio, but be consistent:
fn foo() -> i32 {
    return 42    // OK
}

fn bar() -> i32 {
    42           // Also OK (implicit return)
}
```

### 5. Tuple Destructuring

```d
// WRONG
let (x, y) = point

// CORRECT
let x = point.0
let y = point.1
// OR use struct:
let x = point.x
let y = point.y
```

### 6. String Formatting

```d
// WRONG
println!("x = {}, y = {}", x, y)

// CORRECT
println("x = " + x.to_string() + ", y = " + y.to_string())
// OR
println(format("x = {}, y = {}", x, y))  // if format() is available
```

### 7. Generic Bounds Syntax

```d
// WRONG (Rust-style)
fn foo<T: Clone + Debug>(x: T) { }

// CORRECT (Sounio-style with where)
fn foo<T>(x: T) where T: Clone + Debug { }
// OR inline:
fn foo<T: Clone>(x: T) { }  // Simple bounds work
```

---

## Quick Reference

### Keywords

```
module import use export fn let var mut const type struct enum
trait impl if else match for while loop break continue return
in as where pub self Self effect handler handle with perform
resume linear affine move copy drop kernel tile device shared
gpu async await spawn join select channel sample observe infer
proof ode pde causal timeout cancel
Knowledge Quantity Tensor vec2 vec3 vec4 mat2 mat3 mat4 quat
dual grad jacobian hessian do counterfactual query invariant
requires ensures assert assume unsafe extern static true false
```

### Operators

```
+  -  *  /  %  ^              // Arithmetic
&  |  ~  !                    // Bitwise/logical
== != < > <= >=               // Comparison
&& ||                         // Logical
<< >>                         // Shift
++ (concatenation)            // Array concat
+- (plus-minus)               // Uncertainty
-> => <-                      // Arrows
.. ..= ...                    // Ranges
:: .                          // Path/field
```

### Built-in Effects

```
IO      // Input/output operations
Mut     // Mutable state
Alloc   // Heap allocation
Panic   // May panic
GPU     // GPU operations
Prob    // Probabilistic operations
Async   // Async operations
Div     // May diverge
```

### Type Syntax

```
i8 i16 i32 i64 i128           // Signed integers
u8 u16 u32 u64 u128           // Unsigned integers
f32 f64                        // Floats
bool char string ()            // Other primitives
[T; N]                         // Array
[T]                            // Slice
Vec<T>                         // Vector
Option<T>                      // Optional
Result<T, E>                   // Result
&T                             // Shared reference
&!T                            // Exclusive reference
*const T                       // Raw const pointer (FFI)
*mut T                         // Raw mutable pointer (FFI)
{ x: T | predicate }           // Refinement type
```

---

## Version Information

This guide is for **Sounio L0** (low-level systems syntax).

For questions about specific features, check the compiler source at `compiler/src/` or run:

```bash
# Check a Sounio file
souc check your_file.sio --show-ast --show-types

# Run a Sounio program
souc run your_file.sio

# Start the REPL
souc repl

# Package manager
sou pkg --help
```
