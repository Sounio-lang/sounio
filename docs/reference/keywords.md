# Sounio Keywords Reference

This document provides a complete list of all keywords in the Sounio programming language, organized by category.

## Overview

Sounio has approximately 100 reserved keywords. These fall into several categories:

- **Core Language Keywords**: Basic control flow and declarations
- **Type System Keywords**: Types, traits, and generics
- **Effect System Keywords**: Algebraic effects and handlers
- **Ownership Keywords**: Linear types and memory management
- **GPU/Parallel Keywords**: GPU programming and parallelism
- **Scientific Keywords**: ODE/PDE, causal models, and scientific computing
- **Epistemic Keywords**: Knowledge types and provenance tracking

---

## Core Language Keywords

### Declaration Keywords

| Keyword | Description | Example |
|---------|-------------|---------|
| `fn` | Function declaration | `fn add(a: i32, b: i32) -> i32` |
| `let` | Immutable binding | `let x = 5` |
| `var` | Mutable binding | `var count = 0` |
| `const` | Compile-time constant | `const PI = 3.14159` |
| `static` | Static variable | `static COUNTER: i32 = 0` |
| `type` | Type alias | `type Meters = f64` |
| `struct` | Structure definition | `struct Point { x: f64, y: f64 }` |
| `enum` | Enumeration | `enum Color { Red, Green, Blue }` |
| `trait` | Trait definition | `trait Display { fn display(&self) }` |
| `impl` | Implementation block | `impl Point { ... }` |

### Control Flow Keywords

| Keyword | Description | Example |
|---------|-------------|---------|
| `if` | Conditional branch | `if x > 0 { ... }` |
| `else` | Alternative branch | `if x > 0 { ... } else { ... }` |
| `match` | Pattern matching | `match value { 0 => "zero", _ => "other" }` |
| `for` | For loop | `for i in 0..10 { ... }` |
| `while` | While loop | `while condition { ... }` |
| `loop` | Infinite loop | `loop { if done { break } }` |
| `break` | Exit loop | `break` |
| `continue` | Next iteration | `continue` |
| `return` | Return from function | `return value` |
| `in` | Iterator/range keyword | `for x in collection` |

### Module Keywords

| Keyword | Description | Example |
|---------|-------------|---------|
| `module` | Module declaration | `module mylib` |
| `import` | Import module | `import std::io` |
| `use` | Import alias | `use std::collections::HashMap` |
| `export` | Export symbol | `export fn public_func()` |
| `pub` | Public visibility | `pub fn visible()` |
| `from` | Import source | `from "package" import func` |

### Other Core Keywords

| Keyword | Description | Example |
|---------|-------------|---------|
| `as` | Type coercion | `x as f64` |
| `where` | Generic constraints | `fn f<T>() where T: Clone` |
| `self` | Instance reference | `fn method(&self)` |
| `Self` | Type of self | `fn new() -> Self` |
| `true` | Boolean true | `let flag = true` |
| `false` | Boolean false | `let flag = false` |
| `mut` | Mutable modifier | `let mut x = 5` (legacy) |

---

## Type System Keywords

### Primitive Type Keywords

| Keyword | Description |
|---------|-------------|
| `i8`, `i16`, `i32`, `i64`, `i128` | Signed integers |
| `u8`, `u16`, `u32`, `u64`, `u128` | Unsigned integers |
| `f32`, `f64` | Floating point numbers |
| `bool` | Boolean type |
| `char` | Unicode character |
| `string` | UTF-8 string |

### Linear Algebra Types

| Keyword | Description | Example |
|---------|-------------|---------|
| `vec2` | 2D vector | `let v: vec2 = vec2(1.0, 2.0)` |
| `vec3` | 3D vector | `let v: vec3 = vec3(1.0, 2.0, 3.0)` |
| `vec4` | 4D vector | `let v: vec4 = vec4(1.0, 2.0, 3.0, 4.0)` |
| `mat2` | 2x2 matrix | `let m: mat2 = mat2::identity()` |
| `mat3` | 3x3 matrix | `let m: mat3 = mat3::identity()` |
| `mat4` | 4x4 matrix | `let m: mat4 = mat4::identity()` |
| `quat` | Quaternion | `let q: quat = quat::identity()` |

### Automatic Differentiation Types

| Keyword | Description | Example |
|---------|-------------|---------|
| `dual` | Dual number (forward AD) | `let x = dual(3.0, 1.0)` |
| `grad` | Gradient computation | `let g = grad(f, at: x)` |
| `jacobian` | Jacobian matrix | `let J = jacobian(f, at: x)` |
| `hessian` | Hessian matrix | `let H = hessian(f, at: x)` |

### Contract Keywords

| Keyword | Description | Example |
|---------|-------------|---------|
| `invariant` | Loop/type invariant | `invariant x >= 0` |
| `requires` | Precondition | `requires x > 0` |
| `ensures` | Postcondition | `ensures result >= 0` |
| `assert` | Runtime assertion | `assert(x > 0, "must be positive")` |
| `assume` | Assume for verification | `assume(x < 100)` |

### Unsafe/External Keywords

| Keyword | Description | Example |
|---------|-------------|---------|
| `unsafe` | Unsafe block | `unsafe { raw_ptr_op() }` |
| `extern` | External linkage | `extern "C" { fn malloc(size: i64) }` |

---

## Effect System Keywords

| Keyword | Description | Example |
|---------|-------------|---------|
| `effect` | Effect definition | `effect State<T> { fn get() -> T }` |
| `handler` | Effect handler | `handler IntState for State<i32> { ... }` |
| `handle` | Handle effects | `handle { code } with Handler { }` |
| `with` | Effect annotation | `fn read() -> string with IO` |
| `perform` | Perform effect op | `perform State.get()` |
| `resume` | Resume from effect | `resume(value)` |

### Built-in Effect Names

These are not keywords but are commonly used effect identifiers:

| Effect | Description |
|--------|-------------|
| `IO` | Input/output operations |
| `Mut` | Mutable state |
| `Alloc` | Heap allocation |
| `Panic` | May panic |
| `GPU` | GPU operations |
| `Prob` | Probabilistic operations |
| `Async` | Async operations |
| `Div` | May diverge |

---

## Ownership Keywords

| Keyword | Description | Example |
|---------|-------------|---------|
| `linear` | Linear type marker | `linear struct FileHandle { }` |
| `affine` | Affine type marker | `affine struct Buffer { }` |
| `move` | Move semantics | `move \|\| captured_value` |
| `copy` | Copy trait/semantics | `#[derive(Copy)]` |
| `drop` | Drop/destructor | `fn drop(self)` |

---

## Async/Parallel Keywords

| Keyword | Description | Example |
|---------|-------------|---------|
| `async` | Async function | `async fn fetch() with Async` |
| `await` | Await future | `let result = future.await` |
| `spawn` | Spawn task | `spawn { long_task() }` |

---

## GPU Keywords

| Keyword | Description | Example |
|---------|-------------|---------|
| `kernel` | GPU kernel function | `kernel fn add_vec(a: &[f32], b: &[f32], c: &![f32])` |
| `tile` | Tiled execution | `tile(32, 32) { ... }` |
| `device` | Device memory | `device var buffer: [f32]` |
| `shared` | Shared memory | `shared sdata: [f32; 256]` |
| `gpu` | GPU intrinsics | `gpu.thread_id.x` |

---

## Scientific Computing Keywords

### ODE/PDE Keywords

| Keyword | Description | Example |
|---------|-------------|---------|
| `ode` | ODE system | `ode Decay { state { y: f64 } ... }` |
| `pde` | PDE system | `pde Heat { domain { x: 0..1 } ... }` |
| `state` | State variables | `state { y: f64, v: f64 }` |
| `params` | Parameters | `params { k: f64 }` |
| `equations` | Differential equations | `equations { dy/dt = -k * y }` |
| `domain` | Spatial domain | `domain { x: 0..1, y: 0..1 }` |
| `boundary` | Boundary conditions | `boundary { u(0, t) = 0 }` |
| `initial` | Initial conditions | `initial { y = 100.0 }` |

### Causal Modeling Keywords

| Keyword | Description | Example |
|---------|-------------|---------|
| `causal` | Causal model | `causal DrugEffect { ... }` |
| `nodes` | Causal nodes | `nodes { dose: f64, effect: f64 }` |
| `edges` | Causal edges | `edges { dose -> effect }` |
| `do` | Intervention | `do(Model, dose = 100.0)` |
| `counterfactual` | Counterfactual query | `counterfactual(Model, ...)` |
| `query` | Query operator | `query(effect)` |

### Probabilistic Keywords

| Keyword | Description | Example |
|---------|-------------|---------|
| `sample` | Sample from distribution | `sample(Normal(0.0, 1.0))` |
| `observe` | Observe data point | `observe(Bernoulli(p), true)` |
| `infer` | Run inference | `infer(model, method: MCMC)` |
| `proof` | Proof/verification | `proof { ... }` |

---

## Epistemic Keywords

### Knowledge Types

| Keyword | Description | Example |
|---------|-------------|---------|
| `Knowledge` | Knowledge type | `Knowledge<f64>` |
| `Quantity` | Physical quantity | `Quantity<Mass>` |
| `Tensor` | Tensor type | `Tensor<f64, [N, M]>` |

### Provenance Keywords

| Keyword | Description | Example |
|---------|-------------|---------|
| `Source` | Provenance source | `Source::Measured("sensor")` |
| `Computed` | Computed provenance | `Computed { from: [...] }` |
| `Literature` | Literature source | `Literature("DOI:...")` |
| `Measured` | Measurement source | `Measured("lab_id")` |
| `Input` | User input source | `Input("form_field")` |
| `Derived` | Derived provenance | `Derived { from: [...] }` |

### Validity Keywords

| Keyword | Description | Example |
|---------|-------------|---------|
| `Valid` | Always valid | `Valid` |
| `ValidUntil` | Valid until date | `ValidUntil("2024-12-31")` |
| `ValidWhile` | Valid while condition | `ValidWhile(temp < 100)` |

---

## Ontology Keywords

| Keyword | Description | Example |
|---------|-------------|---------|
| `ontology` | Ontology declaration | `ontology "ChEBI" from "chebi.owl"` |
| `OntologyTerm` | Ontology term type | `let drug: OntologyTerm<ChEBI>` |
| `align` | Ontology alignment | `align ChEBI with DrugBank` |
| `distance` | Semantic distance | `distance(term1, term2)` |
| `threshold` | Distance threshold | `threshold = 0.15` |
| `compat` | Compatibility annotation | `#[compat(0.35)]` |

---

## Reserved but Unused Keywords

The following keywords are reserved for future use:

| Keyword | Planned Use |
|---------|-------------|
| `macro` | Macro definitions |
| `yield` | Generator yield |
| `try` | Try expression |
| `catch` | Exception handling |
| `throw` | Exception throwing |
| `await` | Currently implemented |
| `abstract` | Abstract types |
| `final` | Final/sealed |
| `override` | Method override |
| `virtual` | Virtual dispatch |

---

## Keyword Categories Summary

| Category | Count | Examples |
|----------|-------|----------|
| Core Language | ~25 | `fn`, `let`, `if`, `for` |
| Type System | ~20 | `struct`, `trait`, `vec3` |
| Effects | ~6 | `effect`, `handler`, `with` |
| Ownership | ~5 | `linear`, `affine`, `move` |
| Async/Parallel | ~3 | `async`, `await`, `spawn` |
| GPU | ~5 | `kernel`, `shared`, `tile` |
| Scientific | ~15 | `ode`, `causal`, `sample` |
| Epistemic | ~15 | `Knowledge`, `Source`, `Valid` |
| Ontology | ~6 | `ontology`, `align`, `distance` |

---

## Identifier Rules

### Valid Identifiers

- Start with a letter or underscore
- Contain letters, digits, or underscores
- Are not keywords

```sio
let valid_name = 1
let _private = 2
let camelCase = 3
let PascalCase = 4
let name123 = 5
```

### Invalid Identifiers

```sio
let 123name = 1     // Cannot start with digit
let fn = 2          // Reserved keyword
let my-name = 3     // Hyphens not allowed
let my.name = 4     // Dots not allowed
```

### Contextual Keywords

Some keywords are only reserved in specific contexts:

```sio
// `state` is a keyword in ODE blocks
ode Model {
    state { x: f64 }  // keyword here
}

// But can be used as identifier elsewhere
let state = 42  // OK: identifier
```
