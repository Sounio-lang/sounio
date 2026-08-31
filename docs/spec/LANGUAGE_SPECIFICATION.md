<!-- docs:meta
topic_id: repo.docs.spec.language-specification
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.spec.language-specification
-->

# The Sounio Programming Language

## Language Specification v1.0.0-beta.6

**Author**: Demetrios Chiuratto Agourakis
**Status**: Release Candidate
**Last Updated**: 2026-03-21

> **Note (2026-05-27, updated):** Lambda/closure literals (Section 4.7.2) are **normative and implemented** — 16/17 `tests/run-pass/closure_*.sio` pass (the exception is `closure_linear.sio` which tests linear-resource closures, marked `//@ ignore`). The authoritative tiering for every feature in this spec is `docs/serious-language/public-claim-registry.v1.tsv` (see `closures.lambdas = validated_research`); for compiler limits and active gaps see `docs/compiler/KNOWN_LIMITATIONS.md`.

---

## Abstract

Sounio is a novel systems programming language designed for scientific computing, medical modeling, and high-performance applications. The language combines strong static typing with algebraic effects, linear types, units of measure, and first-class GPU support. This specification defines the syntax, semantics, and type system of the Sounio language.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Lexical Structure](#2-lexical-structure)
3. [Types](#3-types)
   - 3.9 [Epistemic Types](#39-epistemic-types)
4. [Expressions](#4-expressions)
5. [Statements](#5-statements)
6. [Declarations](#6-declarations)
7. [Effect System](#7-effect-system)
8. [Ownership and Borrowing](#8-ownership-and-borrowing)
9. [Units of Measure](#9-units-of-measure)
10. [Refinement Types](#10-refinement-types)
11. [GPU Programming](#11-gpu-programming)
12. [Standard Library](#12-standard-library)
13. [Appendix: Grammar](#appendix-a-grammar)
14. [Appendix: Built-in Types](#appendix-b-built-in-types)
15. [Appendix: Built-in Effects](#appendix-c-built-in-effects)

---

## 1. Introduction

### 1.1 Design Goals

Sounio is designed with the following goals:

1. **Safety**: Prevent common programming errors through the type system
2. **Expressiveness**: Enable concise expression of complex scientific computations
3. **Performance**: Generate efficient native code for CPU and GPU
4. **Clarity**: Make side effects and resource usage explicit and trackable

### 1.2 Influences

Sounio draws inspiration from several languages and type systems:

- **Rust**: Ownership and borrowing (with different syntax)
- **Koka/Eff**: Algebraic effect system
- **F#**: Units of measure
- **Liquid Haskell**: Refinement types
- **CUDA/SYCL**: GPU programming model

### 1.3 Notation

Throughout this specification:

- `monospace` denotes code or syntax
- *italics* denote meta-variables
- **bold** denotes keywords
- `[x]` denotes optional element x
- `{x}` denotes zero or more repetitions of x
- `x | y` denotes choice between x and y

---

## 2. Lexical Structure

### 2.1 Character Set

Sounio source files are UTF-8 encoded. Identifiers may contain Unicode letters and digits.

### 2.2 Keywords

The following identifiers are reserved keywords:

**Declaration keywords:**
```
fn        kernel    struct    enum      type      trait
impl      const     let       var       module    import
export    from      pub       effect
```

**Control flow keywords:**
```
if        else      match     for       while     loop
break     continue  return
```

**Type keywords:**
```
linear    affine    own       where     as        in
mut       unsafe
```

**Effect keywords:**
```
with      handle    on        resume    perform
```

> **Measured 2026-08-19 — two of these tables are wrong, and one entry is
> reserved against itself.** Each word was tested as an identifier
> (`let <word> = 1`); a reserved word cannot be one. Both engines agree.
>
> | word | listed as | measured |
> |---|---|---|
> | `own` | type keyword | **not reserved** — compiles as an identifier |
> | `handle` | effect keyword | **not reserved** — compiles as an identifier; it is *contextual*, recognised in the `handle<IO> { … }` position §7.3 measures |
> | `mut` | type keyword | reserved, but refused **by design**: `error[E040]: Sounio uses \`var\` for mutable bindings, not \`let mut\`` |
>
> `mut` is reserved to teach against, not to use, so listing it beside `linear`
> and `where` reads as an endorsement of a form the compiler exists to refuse.
> The remaining ten refuse as parse failures and are reserved as documented.
> `perform` is reserved on Madaros but resolves as an ordinary identifier under
> `lean_single`, which reports `error[E200]: undefined identifier` — see
> §7.3.4 in `S07_EFFECT_HANDLERS.md`.


**Literal keywords:**
```
true      false
```

### 2.3 Built-in Type Names

The following are built-in type names (not keywords, but reserved in type namespace):

```
int       i8        i16       i32       i64       i128
uint      u8        u16       u32       u64       u128
f32       f64       bool      char      string
```

> **Measured 2026-08-19 — `int` and `uint` are listed and do not work.**
> `let x: i64 = 0` checks; `let x: int = 0` gives
> `error[E001]: this binding expects a different type`, as does `uint`. The two
> widths this document might have been expected to be aspirational about,
> `i128` and `u128`, **do** check — that guess was tested and was wrong.
>
> **A test in parameter position proves nothing here.** `fn f(x: T) -> i32 { 0 }`
> checks under Madaros for *every* `T`, including `naoexiste_zorble`. An
> undeclared parameter type is refused only once the function has a caller
> (`error[E009]`), so the declaration is never interrogated on its own — the use
> is. lean_single emits an ELF **either way**, caller or not.
>
> This is `SOUNIO-TYPE-INTERROGATION` at the level of the type namespace: a name
> in a declaration is not asked to exist until something forces the question.

### 2.4 Built-in Effect Names

```
IO        Mut       Alloc     Panic     Async     GPU       Prob      Div
```

> **Measured 2026-08-19 — this list is a quarter of the table.** The compiler's
> closed effect-name list holds **30** names
> (`scripts/ci/effect_name_closed_list_gate.sh`, frozen at 2,845 sites). All
> eight above are in it, so nothing here is wrong; **22 are missing**:
>
> `Approx  Audit  Causal  Chaotic  Confidence  Deterministic  Epistemic
> Hypothesis  Learn  MultiTest  NarrowWidthApproximation  NaturalityG2  Network
> NonAssoc  NonUnitary  Observe  Perturbative  Render  Sensor  Temporal  Witness
> ZD`
>
> The omissions are not marginal: `ZD`, `Witness`, `Learn`, `Temporal` and
> `Epistemic` carry the surgical-unlearning and epistemic-typing claims, and
> `Observe` is named as a core effect in `CLAUDE.md` §7.
>
> **The list cannot be checked by writing one.** `fn g() -> i32 with Zorblex { 0 }`
> gives `check: OK` under Madaros and an ELF under lean_single. An unknown effect
> name is dropped in silence on both engines — §6.1 measures the same on the
> declaration side — so membership of this table is unfalsifiable from outside
> the compiler, and the closed-list gate exists because of that.

### 2.5 Operators

**Arithmetic:**
```
+    -    *    /    %    ^
```

**Comparison:**
```
==   !=   <    <=   >    >=
```

**Logical:**
```
&&   ||   !
```

**Bitwise:**
```
&    |    ~    <<   >>
```

**Reference:**
```
&    &!
```

**Other:**
```
=    ->   =>   ::   :    ;    ,    .    ..   ...
```

### 2.6 Literals

#### 2.6.1 Integer Literals

```
42              // Decimal
0xFF            // Hexadecimal
0b1010          // Binary
0o77            // Octal
1_000_000       // With separators
```

#### 2.6.2 Floating-Point Literals

```
3.14            // Basic
1e10            // Scientific
2.5e-3          // Scientific with sign
```

#### 2.6.3 Unit-Annotated Literals

```
500.0_mg        // 500 milligrams
10.0_mL         // 10 milliliters
1.5_hours       // 1.5 hours
```

#### 2.6.4 String Literals

```
"hello"         // Basic string
"line1\nline2"  // With escape sequences
```

Escape sequences: `\n`, `\r`, `\t`, `\\`, `\"`, `\0`, `\xHH`, `\u{HHHH}`

#### 2.6.5 Character Literals

```
'a'             // Character
'\n'            // Escaped character
```

#### 2.6.6 Boolean Literals

```
true
false
```

### 2.7 Comments

```sio
// Single-line comment

/* Multi-line
   comment */

/// Documentation comment (for items)

//! Module-level documentation
```

---

## 3. Types

### 3.1 Primitive Types

| Type | Description | Size |
|------|-------------|------|
| `bool` | Boolean | 1 byte |
| `char` | Unicode scalar value | 4 bytes |
| `i8`, `i16`, `i32`, `i64`, `i128` | Signed integers | 1, 2, 4, 8, 16 bytes |
| `u8`, `u16`, `u32`, `u64`, `u128` | Unsigned integers | 1, 2, 4, 8, 16 bytes |
| `int` | Platform-sized signed | Platform-dependent |
| `uint` | Platform-sized unsigned | Platform-dependent |
| `f32`, `f64` | IEEE 754 floating-point | 4, 8 bytes |
| `string` | UTF-8 string | Dynamic |

### 3.2 Compound Types

#### 3.2.1 Tuples

```sio
let point: (f64, f64) = (1.0, 2.0)
let (x, y) = point              // Destructuring
```

#### 3.2.2 Arrays

```sio
let arr: [int; 5] = [1, 2, 3, 4, 5]
let first = arr[0]
```

#### 3.2.3 Slices

```sio
let slice: &[int] = &arr[1..4]
```

### 3.3 User-Defined Types

#### 3.3.1 Structures

```sio
struct Point {
    x: f64,
    y: f64,
}

let p = Point { x: 1.0, y: 2.0 }
```

#### 3.3.2 Enumerations

```sio
enum Option<T> {
    Some(T),
    None,
}

enum Result<T, E> {
    Ok(T),
    Err(E),
}
```

#### 3.3.3 Type Aliases

```sio
type Coordinate = (f64, f64)
type Matrix = [[f64; 4]; 4]
```

### 3.4 Reference Types

#### 3.4.1 Shared References

```sio
let r: &int = &x        // Shared (immutable) borrow
```

#### 3.4.2 Exclusive References

```sio
let r: &!int = &!x      // Exclusive (mutable) borrow
```

Note: Sounio uses `&!` instead of `&mut` for exclusive references.

#### 3.4.3 Owned Values

```sio
fn consume(data: own Buffer)    // Takes ownership
```

### 3.5 Substructural Types

#### 3.5.1 Linear Types

Linear types must be used exactly once:

```sio
linear struct FileHandle {
    fd: int,
}

fn open(path: string) -> FileHandle with IO { ... }
fn close(handle: FileHandle) with IO { ... }

// Usage
let f = open("file.txt")
close(f)                // Must be consumed
// f cannot be used again
```

#### 3.5.2 Affine Types

Affine types may be used at most once:

```sio
affine struct TempBuffer {
    ptr: *mut u8,
    size: usize,
}

// Can be dropped without explicit consumption
```

### 3.6 Function Types

```sio
type BinaryOp = fn(int, int) -> int
type Effectful = fn(string) -> Result<int, Error> with IO
```

### 3.7 Generic Types

```sio
struct Vec<T> {
    data: *mut T,
    len: usize,
    cap: usize,
}

fn identity<T>(x: T) -> T {
    return x
}
```

### 3.8 Type Inference

Sounio uses bidirectional type inference. Types can often be omitted:

```sio
let x = 42              // Inferred as int
let y = 3.14            // Inferred as f64
let z = [1, 2, 3]       // Inferred as [int; 3]
```

### 3.9 Epistemic Types

Epistemic types enable rigorous uncertainty quantification in scientific computing, following the Guide to the Expression of Uncertainty in Measurement (GUM, JCGM 100:2008).

#### 3.9.1 The Knowledge\<T\> Type

The `Knowledge<T>` type represents a value with associated uncertainty:

```sio
struct Knowledge<T> {
    value: T,                    // Point estimate
    variance: f64,               // Uncertainty (squared)
    confidence: BetaConfidence,  // Bayesian credibility
    provenance: Provenance,      // Audit trail
}
```

**Construction:**

```sio
// From direct measurement
let temp = Knowledge::new(
    value: 98.6,
    variance: 0.04,  // ±0.2°F standard uncertainty
    confidence: BetaConfidence::new(alpha: 100.0, beta: 5.0),
    provenance: Provenance::measurement("thermometer_A")
)

// Exact (zero uncertainty)
let g = Knowledge::exact(9.81)  // Gravitational constant
```

#### 3.9.2 Uncertainty Propagation

Arithmetic operations on `Knowledge<T>` values automatically propagate uncertainty according to GUM first-order Taylor series approximation:

**Addition/Subtraction:**

```
u²(x + y) = u²(x) + u²(y)
```

**Multiplication:**

```
u²(x * y) = y² u²(x) + x² u²(y)
```

**Division:**

```
u²(x / y) = u²(x)/y² + x² u²(y)/y⁴
```

**Example:**

```sio
let dose_mg = Knowledge::measured(500.0, 5.0, "scale")     // 500 ± 5 mg
let volume_mL = Knowledge::measured(250.0, 2.0, "pipette") // 250 ± 2 mL
let conc = dose_mg / volume_mL  // Automatically: 2.0 ± 0.024 mg/mL
```

#### 3.9.3 BetaConfidence

Confidence levels are represented using Beta distributions, supporting Bayesian updating:

```sio
struct BetaConfidence {
    alpha: f64,  // Successes + prior
    beta: f64,   // Failures + prior
}

fn beta_mean(c: &BetaConfidence) -> f64 {
    c.alpha / (c.alpha + c.beta)
}

fn beta_update(c: &BetaConfidence, successes: i64, failures: i64)
    -> BetaConfidence {
    BetaConfidence {
        alpha: c.alpha + successes,
        beta: c.beta + failures
    }
}
```

#### 3.9.4 Provenance Tracking

Every `Knowledge` value carries an immutable audit trail:

```sio
enum Source {
    Measurement { instrument: string, timestamp: i64 },
    Derived { op: string, inputs: Vec<ProvenanceId> },
    Literature { doi: string, table: string },
    Simulation { model: string, run_id: i64 },
}

struct Provenance {
    id: ProvenanceId,      // Merkle hash
    source: Source,
    metadata: HashMap<string, string>,
}
```

**Querying provenance:**

```sio
fn audit_calculation(result: Knowledge<f64>) {
    let trace = result.provenance.trace()
    for source in trace {
        match source.source {
            Source::Measurement { instrument, .. } =>
                println("Used instrument: {}", instrument),
            Source::Literature { doi, .. } =>
                println("Cited: {}", doi),
            _ => {}
        }
    }
}
```

#### 3.9.5 Uncertainty Loss Warnings

The type system prevents **silent uncertainty loss** through compile-time warnings:

```sio
fn compute_exact(x: f64) -> f64 { x * 2.0 }

fn main() with IO {
    let k = Knowledge::measured(5.0, 0.01, "scale")

    // ⚠ Warning: uncertainty discarded
    let bad = compute_exact(k.value())

    // ✓ OK: uncertainty preserved
    let good = k.scale(2.0)
}
```

**Type safety theorem:** `Knowledge<T>` is not a subtype of `T`. Explicit extraction generates compiler warnings.

#### 3.9.6 Operational Semantics

Epistemic evaluation follows these reduction rules:

**E-ADD:**

```
(v₁, σ₁², β₁, π₁) + (v₂, σ₂², β₂, π₂)
  → (v₁+v₂, σ₁²+σ₂², comb(β₁,β₂), merge(π₁,π₂))
```

**E-MUL:**

```
(v₁, σ₁², β₁, π₁) × (v₂, σ₂², β₂, π₂)
  → (v₁v₂, v₂²σ₁² + v₁²σ₂², comb(β₁,β₂), merge(π₁,π₂))
```

**E-DIV:**

```
(v₁, σ₁², β₁, π₁) / (v₂, σ₂², β₂, π₂)
  → (v₁/v₂, σ₁²/v₂² + v₁²σ₂²/v₂⁴, comb(β₁,β₂), merge(π₁,π₂))
```

Where:

- `comb(β₁,β₂)` combines confidence levels (minimum)
- `merge(π₁,π₂)` creates derived provenance

#### 3.9.7 GUM Compliance

**Theorem (GUM Compliance):** For any expression `e`, if `eval(e) = (v, σ², β, π)`, then `σ²` equals the combined standard uncertainty `u_c²(e)` as defined by GUM Equation (10).

**Proof sketch:** By structural induction on `e`. Base case: literals have exact variance. Inductive step: arithmetic operations use GUM propagation formulas.

#### 3.9.8 Usage Examples

**Pharmaceutical dosing:**

```sio
fn cockcroft_gault(
    age: Knowledge<years>,
    weight: Knowledge<kg>,
    serum_cr: Knowledge<mg/dL>
) -> Knowledge<mL/min> with IO {
    let k = Knowledge::exact(140.0)
    let numerator = (k - age) * weight
    let denominator = serum_cr * Knowledge::exact(72.0)
    numerator / denominator  // Uncertainty propagated through formula
}
```

**Climate ensemble analysis:**

```sio
fn sea_level_rise_projection(
    models: &[Knowledge<meters>],
    scenario: EmissionScenario
) -> Knowledge<meters> {
    // Within-model uncertainty
    var within_model = 0.0
    for m in models { within_model = within_model + m.variance }
    within_model = within_model / models.len()

    // Between-model uncertainty (ensemble spread)
    let mean_val = Knowledge::mean_value(models)
    var between_model = 0.0
    for m in models {
        let diff = m.value - mean_val
        between_model = between_model + diff * diff
    }
    between_model = between_model / (models.len() - 1)

    Knowledge::new(
        value: mean_val,
        variance: within_model + between_model,
        confidence: BetaConfidence::from_ensemble_size(models.len()),
        provenance: Provenance::cmip6(scenario)
    )
}
```

#### 3.9.9 Implementation Notes

- **v1.0 Status**: Full native code generation with GUM-compliant propagation
- **SIR Pass**: `epistemic_insertion.rs` transforms `BinOp` on `Knowledge<T>`
- **Runtime**: 593 lines in `backend/native/epistemic_runtime.rs`
- **Memory layout**: 40 bytes stack-allocated (value: 8B, variance: 8B, confidence: 16B, provenance: 8B pointer)
- **SIMD**: AVX2 (x86-64), NEON (ARM) for parallel value/variance computation
- **Optimizations**: Exact elision, variance fusion, provenance elision

---

## 4. Expressions

### 4.1 Literal Expressions

See Section 2.6 for literal syntax.

### 4.2 Path Expressions

```sio
x                       // Variable
std::io::read           // Qualified path
Option::Some(42)        // Variant constructor
```

### 4.3 Operators

#### 4.3.1 Precedence (highest to lowest)

| Level | Operators | Associativity |
|-------|-----------|---------------|
| 1 | `()`, `[]`, `.`, `::` | Left |
| 2 | `!`, `-` (unary), `&`, `&!` | Right |
| 3 | `^` | Right |
| 4 | `*`, `/`, `%` | Left |
| 5 | `+`, `-` | Left |
| 6 | `<<`, `>>` | Left |
| 7 | `&` (bitwise) | Left |
| 8 | `\|` (bitwise) | Left |
| 9 | `<`, `<=`, `>`, `>=` | Left |
| 10 | `==`, `!=` | Left |
| 11 | `&&` | Left |
| 12 | `\|\|` | Left |
| 13 | `=` | Right |

### 4.4 Block Expressions

```sio
let result = {
    let a = 1
    let b = 2
    a + b               // Last expression is the value
}
```

### 4.5 Control Flow Expressions

#### 4.5.1 If Expression

```sio
let max = if a > b { a } else { b }
```

#### 4.5.2 Match Expression

```sio
let description = match value {
    0 => "zero",
    1 => "one",
    n if n < 0 => "negative",
    _ => "other",
}
```

### 4.6 Function Calls

```sio
let result = add(1, 2)
let chained = obj.method().another()
```

### 4.7 Function References and Lambda Literals (both normative)

Sounio has first-class **function references** and **lambda literals** (`|x| x + 1`). Both are normative, implemented, and gate-tested. The public-claim registry reflects this at `closures.lambdas = validated_research` (`docs/serious-language/public-claim-registry.v1.tsv`).

#### 4.7.1 Named Function References (normative — implemented)

```sio
fn square(x: i64) -> i64 { x * x }

fn apply(f: fn(i64) -> i64, val: i64) -> i64 {
    f(val)
}

fn main() -> i64 {
    let f = square         // f is a function reference
    apply(f, 5)            // returns 25
}
```

Function references support higher-order programming: functions can be stored in variables, passed as arguments, and called indirectly.

#### 4.7.2 Lambda Literals (normative — implemented)

Lambda literals (`|params| body`) create anonymous functions. They may capture variables from the enclosing scope (closures), be passed to higher-order functions, and escape their definition scope.

```sio
fn apply(f: fn(i64) -> i64, val: i64) -> i64 with Mut, Panic, Div {
    f(val)
}

fn main() -> i64 with IO, Mut, Panic, Div {
    // Zero-capture lambda
    let double = |x: i64| x * 2
    assert(double(5) == 10)

    // Capturing closure
    let offset = 100
    let shifted = |x: i64| x + offset
    assert(shifted(5) == 105)

    // Passed to HOF
    let r = apply(|x: i64| x * x, 7)
    assert(r == 49)

    // Escaping closure (multi-capture, heap env)
    // See closure_returned.sio / closure_capture.sio for full gate tests.
    0
}
```

**One open feature:** *linear closures* (capturing linear/affine resources) are not yet implemented. `closure_linear.sio` is `//@ ignore`. All other closure forms are gate-tested in `tests/run-pass/closure_*.sio`.

### 4.8 Effect Expressions

#### 4.8.1 With Expression

```sio
with handler {
    // Code runs with handler installed
    let x = perform SomeEffect::operation()
    x + 1
}
```

#### 4.8.2 Resume Expression

```sio
handle my_handler for MyEffect {
    on operation(arg) -> {
        let result = process(arg)
        resume(result)          // Continue with value
    }
}
```

---

## 5. Statements

### 5.1 Let Bindings

```sio
let x: int = 42         // Immutable
let y = compute()       // Type inferred
```

### 5.2 Var Bindings

```sio
var counter: int = 0    // Mutable
counter = counter + 1
```

### 5.3 Const Bindings

```sio
const MAX_SIZE: int = 1024      // Compile-time constant
```

### 5.4 Return Statement

```sio
fn add(a: int, b: int) -> int {
    return a + b
}
```

### 5.5 Control Flow Statements

```sio
// While loop
while condition {
    // ...
}

// For loop
for item in collection {
    // ...
}

// Loop with break
loop {
    if done {
        break
    }
}

// Continue
for i in 0..10 {
    if i % 2 == 0 {
        continue
    }
    process(i)
}
```

---

## 6. Declarations

### 6.1 Function Declarations

```sio
fn function_name(param1: Type1, param2: Type2) -> ReturnType {
    // body
}

// With effects
fn read_all(path: string) -> string with IO, Panic {
    // body
}

// Generic function
fn swap<T>(a: &!T, b: &!T) {
    let temp = *a
    *a = *b
    *b = temp
}
```

### 6.2 Struct Declarations

```sio
struct Name {
    field1: Type1,
    field2: Type2,
}

// With modifiers
linear struct Resource { ... }
affine struct Temporary { ... }

// Generic struct
struct Container<T> {
    value: T,
}
```

### 6.3 Enum Declarations

```sio
enum Status {
    Pending,
    Active,
    Complete,
}

enum Tree<T> {
    Leaf(T),
    Node(Box<Tree<T>>, Box<Tree<T>>),
}
```

### 6.4 Trait Declarations

```sio
trait Printable {
    fn print(&self) with IO
}

trait Numeric {
    fn zero() -> Self
    fn add(self, other: Self) -> Self
}
```

### 6.5 Impl Blocks

```sio
impl Point {
    fn new(x: f64, y: f64) -> Point {
        Point { x, y }
    }
    
    fn distance(&self, other: &Point) -> f64 {
        let dx = self.x - other.x
        let dy = self.y - other.y
        (dx * dx + dy * dy).sqrt()
    }
}

impl Printable for Point {
    fn print(&self) with IO {
        println("({}, {})", self.x, self.y)
    }
}
```

### 6.6 Effect Declarations

```sio
effect State<S> {
    fn get() -> S
    fn put(s: S)
}

effect Exception<E> {
    fn raise(e: E) -> !     // Never returns normally
}
```

### 6.7 Module Declarations

```sio
module math

import std::io
import external::lib::{foo, bar}
from ./local import helper

pub fn public_function() { ... }
fn private_function() { ... }
```

---

## 7. Effect System

### 7.1 Overview

D features a complete algebraic effect system. Effects describe computational side effects and are tracked in function signatures.

### 7.2 Built-in Effects

| Effect | Description | Operations |
|--------|-------------|------------|
| `IO` | Input/output | File, network, console operations |
| `Mut` | Mutable state that escapes the callee | Writing through an exclusive reference (`&!T`), a raw `*mut T`, or an array/struct reached through one. See §7.2.1. |
| `Alloc` | Memory allocation | Heap allocation |
| `Panic` | Recoverable failure | Panic and recovery |
| `Async` | Asynchronous computation | Await, spawn |
| `GPU` | GPU operations | Kernel launch, device memory |
| `Prob` | Probabilistic computation | Sample, observe |
| `Div` | Divergence | Potential non-termination |

#### 7.2.1 `Mut` and local mutation — the escape rule

`Mut` describes mutation that is **observable by the caller**. Mutating a
function-local `var` is not: the binding dies with the frame, so it is invisible
outside and does **not** require `Mut` in the signature.

```sio
// No `with Mut` needed: `y` is local, the mutation cannot escape.
// (Division is omitted deliberately -- it would additionally require
//  `with Div, Panic`, which is a separate obligation from `Mut`.)
fn accumulate(n: i64) -> i64 {
    var y = 1
    y = y + n
    y
}

// `with Mut` IS needed: the write is through an exclusive reference,
// so the caller observes it.
fn zero_first(buf: &![i64; 4]) with Mut {
    (*buf)[0] = 0
}
```

A mutation escapes — and therefore requires `Mut` — when it is written through
an exclusive reference (`&!T`), a raw `*mut T`, an array or struct reached
through either, or when it is captured by a closure that outlives the frame.
The checker is intended to **infer** this from the function body rather than
require the author to reason about it; the annotation is a declaration of the
inferred fact, not an independent obligation.

> **Implementation status (measured 2026-07-27, not a statement of intent).**
> Neither shipped engine implements the rule above. Under the default compiler
> (Madaros, `self-hosted/check/`) `Mut` is **not required for either case** —
> `check_callee_effects` (`self-hosted/check/check.sio:11688`) enforces effects
> only at *call sites*, propagating a callee's declared effects to its caller;
> nothing demands `Mut` at an assignment. Under the frozen `lean_single`
> bootstrap seed, `Mut` is required for **both** cases, including the local
> `var` above. A live consequence: `self-hosted/ir/egraph.sio` fails
> `SOUNIO_SOUC_ENGINE=lean_single souc check` at line 1549 — inside `eg_isqrt`
> (`egraph.sio:1511`), on the statement `y = (y + x / y) / 2`, a mutation of a
> function-local `var` in an otherwise pure integer helper. Verified by
> isolation that the missing effect there is `Mut` and not `Div`: the seed
> rejects a local `var` mutation with no division at all, and accepts a bare
> division with no mutation. Closing this gap in the default compiler is
> scoped in
> `docs/audit/MUT_EFFECT_ENFORCEMENT_DISPATCH_2026-07-27.md`; the seed is
> deliberately left frozen (see `docs/compiler/KNOWN_LIMITATIONS.md`).

### 7.3 Effect Annotations

```sio
// Pure function (no effects)
fn add(a: int, b: int) -> int {
    return a + b
}

// Function with effects
fn read_file(path: string) -> string with IO, Panic {
    // ...
}

// Multiple effects
fn simulate() -> f64 with Prob, Alloc, IO {
    // ...
}
```

### 7.4 Effect Polymorphism

```sio
fn map<A, B, E>(f: fn(A) -> B with E, xs: List<A>) -> List<B> with E {
    // Effects of f propagate to map
}

fn compose<A, B, C, E1, E2>(
    f: fn(A) -> B with E1,
    g: fn(B) -> C with E2
) -> fn(A) -> C with E1, E2 {
    |a| g(f(a))
}
```

### 7.5 Effect Handlers

```sio
// Define a handler
handle state_handler<S>(initial: S) for State<S> {
    var current = initial
    
    on get() -> resume(current)
    
    on put(s) -> {
        current = s
        resume(())
    }
    
    return(x) -> (x, current)
}

// Use the handler
let (result, final_state) = with state_handler(0) {
    let x = State::get()
    State::put(x + 1)
    State::get()
}
```

### 7.6 Effect Subtyping

Pure functions can be used where effectful functions are expected:

```
Pure <: E for any effect set E
```

### 7.7 Effect Inference

Effects are inferred from function bodies when not explicitly annotated:

```sio
fn example() {           // Effects inferred
    let f = open("x")    // IO, Panic
    let s = read(f)      // IO
    close(f)             // IO
}
// Inferred: with IO, Panic
```

---

## 8. Ownership and Borrowing

### 8.1 Ownership Rules

1. Each value has exactly one owner
2. When the owner goes out of scope, the value is dropped
3. Ownership can be transferred (moved)

### 8.2 Moving Values

```sio
let s1 = String::from("hello")
let s2 = s1                     // s1 is moved to s2
// s1 is no longer valid
```

### 8.3 Borrowing

#### 8.3.1 Shared Borrows

```sio
let s = String::from("hello")
let r1 = &s                     // Shared borrow
let r2 = &s                     // Multiple shared borrows OK
// s is still valid
```

#### 8.3.2 Exclusive Borrows

```sio
let s = String::from("hello")
let r = &!s                     // Exclusive borrow
// No other borrows allowed while r exists
r.push('!')                     // Can mutate through r
```

### 8.4 Borrow Rules

1. At any time, you can have either:
   - Any number of shared borrows (`&T`), OR
   - Exactly one exclusive borrow (`&!T`)
2. Borrows must not outlive the borrowed value

### 8.5 Lifetime Annotations

```sio
fn longest<'a>(x: &'a string, y: &'a string) -> &'a string {
    if x.len() > y.len() { x } else { y }
}
```

### 8.6 Linear and Affine Types

Linear types provide stronger guarantees:

```sio
linear struct Connection {
    handle: *mut c_void,
}

// Must be explicitly closed
fn close(conn: Connection) with IO {
    // Connection is consumed
}

// Compiler error: linear value not consumed
fn bad_example() {
    let conn = connect()
    // Error: conn must be used
}
```

---

## 9. Units of Measure

### 9.1 Overview

Sounio supports compile-time dimensional analysis through units of measure.

### 9.2 Unit Literals

```sio
let mass: kg = 70.0_kg
let distance: m = 100.0_m
let time: s = 9.58_s
let speed: m/s = distance / time
```

### 9.3 Unit Arithmetic

| Operation | Result Unit |
|-----------|-------------|
| `a: U + b: U` | `U` |
| `a: U - b: U` | `U` |
| `a: U * b: V` | `U·V` |
| `a: U / b: V` | `U/V` |
| `a: U ^ n` | `Uⁿ` |

### 9.4 Predefined Units

**SI Base Units:**
```sio
m       // meter (length)
kg      // kilogram (mass)
s       // second (time)
A       // ampere (current)
K       // kelvin (temperature)
mol     // mole (amount)
cd      // candela (luminosity)
```

**Medical Units:**
```sio
mg      // milligram
mcg     // microgram
mL      // milliliter
L       // liter
h       // hour
min     // minute
```

### 9.5 Custom Units

```sio
unit mmHg = 133.322_Pa          // Derived unit
unit BPM = 1/min                // Beats per minute

let bp: mmHg = 120.0_mmHg
let hr: BPM = 72.0_BPM
```

### 9.6 Unit Conversions

```sio
fn to_mg(g: g) -> mg {
    g * 1000.0
}

let dose: mg = to_mg(0.5_g)     // 500 mg
```

### 9.7 Dimensionless Values

```sio
let ratio: 1 = mass1 / mass2    // Dimensionless
let percentage = ratio * 100.0
```

---

## 10. Refinement Types

### 10.1 Overview

Refinement types add predicate constraints to types, verified at compile time using SMT solvers.

### 10.2 Syntax

```sio
type Positive = { x: int | x > 0 }
type Percentage = { p: f64 | p >= 0.0 && p <= 100.0 }
type NonEmpty<T> = { v: Vec<T> | v.len() > 0 }
```

### 10.3 Usage

```sio
fn sqrt(x: { n: f64 | n >= 0.0 }) -> f64 {
    // x is guaranteed non-negative
}

fn divide(a: int, b: { d: int | d != 0 }) -> int {
    a / b   // Division by zero impossible
}
```

### 10.4 Medical Refinements

```sio
type SafeDose = { dose: mg | dose > 0.0 && dose <= max_therapeutic_dose }
type ValidCrCl = { crcl: mL/min | crcl > 0.0 && crcl < 200.0 }
type ValidBMI = { bmi: kg/m^2 | bmi > 10.0 && bmi < 100.0 }

fn calculate_dose(
    weight: { w: kg | w > 0.0 },
    crcl: ValidCrCl
) -> SafeDose {
    // Compiler verifies the result satisfies SafeDose
}
```

### 10.5 Verification

Refinement predicates are verified by:
1. **Local inference** for simple cases
2. **SMT solving** (Z3) for complex predicates
3. **Runtime checks** when compile-time verification is impossible

---

## 11. GPU Programming

### 11.1 Overview

Sounio provides first-class GPU support through the `GPU` effect and `kernel` functions.

### 11.2 Kernel Functions

```sio
kernel fn vector_add(a: &[f32], b: &[f32], c: &![f32], n: int) {
    let i = gpu.thread_id.x
    if i < n {
        c[i] = a[i] + b[i]
    }
}
```

### 11.3 GPU Memory

```sio
fn compute() with GPU, Alloc {
    // Allocate device memory (linear type)
    let d_a = gpu.alloc<f32>(1024)
    let d_b = gpu.alloc<f32>(1024)
    let d_c = gpu.alloc<f32>(1024)
    
    // Copy to device
    gpu.copy_to_device(host_a, d_a)
    gpu.copy_to_device(host_b, d_b)
    
    // Launch kernel
    gpu.launch(
        vector_add,
        grid: (32,),
        block: (32,),
        args: (d_a, d_b, d_c, 1024)
    )
    
    // Copy back
    gpu.copy_to_host(d_c, host_c)
    
    // Free device memory (required for linear types)
    gpu.free(d_a)
    gpu.free(d_b)
    gpu.free(d_c)
}
```

### 11.4 GPU Intrinsics

```sio
gpu.thread_id.x         // Thread index in block (x dimension)
gpu.thread_id.y
gpu.thread_id.z
gpu.block_id.x          // Block index in grid
gpu.block_dim.x         // Block dimensions
gpu.grid_dim.x          // Grid dimensions
gpu.sync_threads()      // Synchronize threads in block
```

### 11.5 Shared Memory

```sio
kernel fn reduce(input: &[f32], output: &![f32], n: int) {
    shared let cache: [f32; 256]
    
    let tid = gpu.thread_id.x
    let i = gpu.block_id.x * gpu.block_dim.x + tid
    
    cache[tid] = if i < n { input[i] } else { 0.0 }
    gpu.sync_threads()
    
    // Reduction in shared memory
    var stride = gpu.block_dim.x / 2
    while stride > 0 {
        if tid < stride {
            cache[tid] = cache[tid] + cache[tid + stride]
        }
        gpu.sync_threads()
        stride = stride / 2
    }
    
    if tid == 0 {
        output[gpu.block_id.x] = cache[0]
    }
}
```

---

## 12. Standard Library

### 12.1 Core Module

```sio
import std::core::{Option, Result, Vec, String, Box}
```

### 12.2 IO Module

```sio
import std::io::{read_file, write_file, stdin, stdout, println}
```

### 12.3 Math Module

```sio
import std::math::{sin, cos, sqrt, exp, log, abs, min, max}
```

### 12.4 Collections Module

```sio
import std::collections::{HashMap, HashSet, BTreeMap, LinkedList}
```

### 12.5 Probability Module

```sio
import std::prob::{Normal, Uniform, Bernoulli, sample, observe}
```

---

## Appendix A: Grammar

```ebnf
program         = { item } ;

item            = function_def
                | struct_def
                | enum_def
                | type_alias
                | const_def
                | effect_def
                | trait_def
                | impl_block
                | module_def
                | import_stmt
                ;

function_def    = "fn" IDENT generics? "(" params? ")" 
                  [ "->" type ] [ "with" effect_list ] 
                  block ;

kernel_def      = "kernel" "fn" IDENT "(" params? ")" 
                  [ "->" type ] block ;

struct_def      = [ "linear" | "affine" ] "struct" IDENT generics?
                  "{" { field "," } "}" ;

enum_def        = "enum" IDENT generics? "{" { variant "," } "}" ;

type_alias      = "type" IDENT generics? "=" type ;

effect_def      = "effect" IDENT generics? "{" { effect_op } "}" ;

trait_def       = "trait" IDENT generics? "{" { trait_item } "}" ;

impl_block      = "impl" generics? type [ "for" type ] "{" { impl_item } "}" ;

type            = primitive_type
                | IDENT [ "<" type_args ">" ]
                | "&" type
                | "&!" type
                | "own" type
                | "linear" type
                | "affine" type
                | "(" [ type { "," type } ] ")"
                | "[" type ";" expr "]"
                | "fn" "(" [ type { "," type } ] ")" "->" type [ "with" effect_list ]
                | "{" IDENT ":" type "|" expr "}"
                ;

effect_list     = effect { "," effect } ;

effect          = "IO" | "Mut" | "Alloc" | "Panic" | "Async" | "GPU" | "Prob" | "Div"
                | IDENT
                ;

stmt            = let_stmt
                | var_stmt
                | return_stmt
                | if_stmt
                | while_stmt
                | for_stmt
                | loop_stmt
                | match_stmt
                | expr_stmt
                ;

let_stmt        = "let" pattern [ ":" type ] [ "=" expr ] ;
var_stmt        = "var" IDENT [ ":" type ] [ "=" expr ] ;
return_stmt     = "return" [ expr ] ;

expr            = literal
                | path_expr
                | unary_expr
                | binary_expr
                | call_expr
                | index_expr
                | field_expr
                | if_expr
                | match_expr
                | block
                | lambda_expr
                | with_expr
                ;

literal         = INT_LIT | FLOAT_LIT | STRING_LIT | CHAR_LIT | BOOL_LIT | UNIT_LIT ;

block           = "{" { stmt } [ expr ] "}" ;
```

---

## Appendix B: Built-in Types

| Type | Description | Default Value |
|------|-------------|---------------|
| `bool` | Boolean | `false` |
| `i8` | 8-bit signed integer | `0` |
| `i16` | 16-bit signed integer | `0` |
| `i32` | 32-bit signed integer | `0` |
| `i64` | 64-bit signed integer | `0` |
| `i128` | 128-bit signed integer | `0` |
| `int` | Platform-sized signed integer | `0` |
| `u8` | 8-bit unsigned integer | `0` |
| `u16` | 16-bit unsigned integer | `0` |
| `u32` | 32-bit unsigned integer | `0` |
| `u64` | 64-bit unsigned integer | `0` |
| `u128` | 128-bit unsigned integer | `0` |
| `uint` | Platform-sized unsigned integer | `0` |
| `f32` | 32-bit IEEE 754 float | `0.0` |
| `f64` | 64-bit IEEE 754 float | `0.0` |
| `char` | Unicode scalar value | `'\0'` |
| `string` | UTF-8 string | `""` |
| `()` | Unit type | `()` |
| `!` | Never type | (no values) |

---

## Appendix C: Built-in Effects

| Effect | Operations | Description |
|--------|------------|-------------|
| `IO` | read, write, print | Input/output operations |
| `Mut` | get, set | Mutable state access |
| `Alloc` | alloc, free | Heap memory allocation |
| `Panic` | panic, catch | Recoverable errors |
| `Async` | await, spawn | Asynchronous execution |
| `GPU` | launch, sync, alloc | GPU operations |
| `Prob` | sample, observe | Probabilistic operations |
| `Div` | (implicit) | Potential non-termination |

---

## References

1. Plotkin, G., & Pretnar, M. (2009). Handlers of algebraic effects.
2. Biernacki, D., et al. (2019). Abstracting algebraic effects.
3. Kennedy, A. (2010). Types for units-of-measure.
4. Rondon, P., et al. (2008). Liquid types.
5. Wadler, P. (1990). Linear types can change the world!

---

*End of Sounio Language Specification v0.1.0*
