---
title: "Effect System"
description: "Sounio's algebraic effect system: 10 built-in effects with row polymorphism, handler capabilities, and epistemic impact tracking."
---

## Algebraic Effect System

Sounio tracks computational side effects at the type level using an **algebraic effect system** with row polymorphism. Every function declares its effects, and the compiler verifies that all effects are properly handled.

### Built-in Effects

Sounio defines **10 first-class effects** (`compiler/src/types/core.rs:850-945`):

| Effect | Domain | Operations |
|--------|--------|-----------|
| **IO** | File system, console | `read_file`, `write_file`, `print`, `read_line` |
| **Mut** | Mutable state | `get`, `set`, `modify`, `clear` |
| **Alloc** | Memory management | `alloc`, `dealloc`, `realloc`, `alloc_array` |
| **Prob** | Probabilistic computing | `sample`, `observe`, `condition` |
| **GPU** | GPU compute | `launch`, `sync`, `alloc_device` |
| **Div** | Division (may throw) | Division operations |
| **Async** | Asynchronous operations | `spawn`, `await`, `join`, `select` |
| **Exn** | Typed exceptions | `throw`, `try_catch`, `rethrow` |
| **Epistemic** | Confidence tracking | `degrade`, `assert_confidence`, `firewall` |
| **FFI** | Foreign function interface | `extern "C"` calls |

### Effect Sets and Row Polymorphism

**File**: `compiler/src/types/core.rs:981-1029`

Effects are tracked as sets with support for effect variables (row polymorphism):

```rust
pub struct EffectSet {
    effects: HashSet<String>,      // Concrete effects
    effect_vars: HashSet<EffectVar>, // Effect type variables
}
```

This enables polymorphic functions that work with any additional effects:

```sio
fn map<A, B, E>(list: List<A>, f: fn(A) -> B with E) -> List<B> with E {
    // E is a row variable: any effects from f propagate to map
}
```

### Effect Checking

**File**: `compiler/src/effects/inference.rs:101-125`

The `EffectChecker` verifies that:
1. All effects used in a function body are declared in its signature
2. All effects are handled before reaching `main`
3. Higher-order function effects propagate correctly

```rust
pub struct EffectChecker<'a> {
    symbols: &'a SymbolTable,
    fn_effects: HashMap<DefId, EffectSet>,
    method_effects: HashMap<(String, String), EffectSet>,
    hof_effects: HashMap<String, EffectSet>,
    declared: EffectSet,        // Current function's declared effects
    inferred: EffectSet,        // Inferred effects from body
}
```

**Error types** (`inference.rs:144-170`):
- `UndeclaredEffect`: Effect used but not in function signature
- `UnhandledEffect`: Effect not handled by any enclosing handler
- `EffectInPureContext`: Effectful operation in a pure function
- `EffectfulClosureArg`: Closure with effects passed to non-HOF function

### Handler Capability System

**File**: `compiler/src/effects/handler_capability.rs`

Effect handlers implement the `HandlerCapability` trait:

```rust
pub trait HandlerCapability {
    fn operations(&self) -> &[OperationSpec];
    fn handle(&self, op: &str, args: &[Value], cont: ContinuationId) -> HandlerResult;
}
```

Each operation specification includes:
- Parameter types and return type
- Whether it uses a continuation (for `resume`)
- **Confidence factor**: How the operation affects epistemic confidence

Handler results can:
- **Return**: Provide a value immediately
- **Resume**: Continue with the suspended computation
- **Suspend**: Park the computation (for async effects)
- **Abort**: Cancel with an error

### 12 Built-in Handlers

**File**: `compiler/src/effects/handlers/mod.rs:54-80`

| Handler | Effect | Purpose |
|---------|--------|---------|
| `IOHandler` | IO | File and console I/O |
| `MutHandler` | Mut | Mutable state cells |
| `AllocHandler` | Alloc | Memory allocation |
| `ProbHandler` | Prob | Probability distributions |
| `GpuHandler` | GPU | GPU kernel dispatch |
| `DivHandler` | Div | Division with zero-check |
| `AsyncHandler` | Async | Task scheduling |
| `ExnHandler` | Exn | Exception handling |
| `EpistemicHandler` | Epistemic | Confidence tracking |
| `PanicHandler` | Panic | Unrecoverable errors |
| `NetworkHandler` | Network | HTTP/TCP operations |
| `SensorHandler` | Sensor | Hardware sensor access |

### Epistemic Impact

**File**: `compiler/src/effects/epistemic_effects.rs`

Every effect operation has an associated **epistemic impact** that modifies confidence:

```rust
pub struct EpistemicImpact {
    confidence_factor: f64,     // Multiplicative factor (0.0-1.0)
    provenance_tag: String,     // Added to provenance chain
    crosses_firewall: bool,     // Whether it crosses a confidence boundary
}
```

For example:
- `IO.read_file` might degrade confidence by 0.95 (external data source)
- `Prob.sample` degrades by 0.8-0.99 depending on distribution
- `GPU.launch` preserves confidence (deterministic computation)
- `Epistemic.firewall` resets confidence to a boundary value

### Cranelift JIT Effect Dispatch

**File**: `compiler/src/codegen/cranelift.rs:112-145`

In the JIT backend, effects are dispatched via handler IDs:

| ID Range | Effect | Handlers |
|----------|--------|----------|
| 10-19 | Prob | Deterministic (10), Importance Sampling (11), Enumeration (12) |
| 20-29 | Causal | SCM (20), Backdoor (21), Front-door (22) |
| 30-39 | Grad | Forward (30), Reverse (31), Numeric (32) |
| 40-49 | Epistemic | Full tracking (40), Simple (41), Audit (42) |

### Type System Integration

The type checker (`check/mod.rs:43-97`) maintains effect state alongside type state:

```rust
struct TypeChecker {
    effects: EffectInference,
    next_effect_var: u32,
    handler_effects: HashMap<String, String>,
    masked_effects: EffectSet,   // Effects handled in current scope
}
```

When a handler is entered, its effect is added to `masked_effects`. Functions inside the handler can use the effect without declaring it, because the handler provides the implementation.
