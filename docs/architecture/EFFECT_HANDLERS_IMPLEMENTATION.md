<!-- docs:meta
topic_id: repo.docs.architecture.effect-handlers-implementation
authority: historical
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.architecture.effect-handlers-implementation
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Effect Handlers Implementation Guide

**Status**: ✅ Phases 1-4 Complete (Interpreter Runtime)
**Date**: January 2026
**Research Foundation**: Affect (POPL 2025), Soundly Handling Linearity (POPL 2024), Retrofitting OCaml (PLDI 2021)

## Overview

Sounio's effect handler system provides algebraic effects with real continuation capture and resumption. The implementation uses **selective CPS via closure capture** for the interpreter, making it efficient and maintainable.

## Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Effect Handler System                     │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐    ┌─────────────────┐    ┌────────────┐ │
│  │   Linearity  │───▶│  Continuation   │───▶│  Handler   │ │
│  │    Types     │    │    Capture      │    │  Dispatch  │ │
│  └──────────────┘    └─────────────────┘    └────────────┘ │
│         │                     │                     │        │
│         │                     │                     │        │
│         ▼                     ▼                     ▼        │
│  ┌──────────────────────────────────────────────────────┐  │
│  │           12 Concrete Effect Handlers                 │  │
│  │  IO, Mut, Alloc, Panic, Async, Prob, GPU, Network,  │  │
│  │        Sensor, Epistemic, Exn, Div                   │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Key Files

| File | Purpose | LOC |
|------|---------|-----|
| `effects/linearity.rs` | Linearity type system | 224 |
| `effects/continuation.rs` | Continuation capture/resume | 1,216 |
| `effects/handler_capability.rs` | Handler trait | 742 |
| `effects/handlers/` | 12 concrete handlers | ~17,974 |
| `interp/effect_dispatch.rs` | Effect dispatch logic | ~800 |
| `interp/eval.rs` | Interpreter integration | 2,698 |

## Linearity Type System

Based on **Affect (POPL 2025)**, we track three linearity constraints:

### Linearity Variants

```rust
pub enum Linearity {
    /// Affine: may be resumed zero or one times
    OnceOrNever,

    /// Linear: must be resumed exactly once (default)
    ExactlyOnce,

    /// Unrestricted: may be resumed multiple times
    MultiShot,
}
```

### Compatibility Rules

- **Multi-shot → One-shot**: ✅ (can use multi-shot where one-shot expected)
- **Linear → Affine**: ✅ (stronger guarantee)
- **Affine → Linear**: ❌ (missing guarantee)

### Performance Impact

- **One-shot continuations**: 3-5x faster (no deep copy overhead)
- **Default**: ExactlyOnce for safety
- **95%+ use cases**: One-shot sufficient

## Continuation Capture

### Closure-Based Approach

Instead of full CPS transformation, we use Rust's closure capture:

```rust
// At perform site
HirExprKind::Perform { effect, op, args } => {
    let cont = CapturedContinuation::new_one_shot(
        ResumePoint::interpreter_closure(
            move |resume_value| Ok(resume_value),
            Some(&format!("{}.{}", effect, op))
        )
    );

    self.effect_ctx.dispatch_with_continuation(effect, op, args, cont)
}
```

### Resume Point Types

```rust
pub enum ResumePoint {
    /// One-shot: FnOnce closure
    InterpreterClosure {
        resume_fn: Box<dyn FnOnce(Value) -> Result<Value, ContinuationError>>,
        description: Option<String>,
    },

    /// Multi-shot: Arc<dyn Fn> for cloning
    InterpreterMultiShot {
        resume_fn: Arc<dyn Fn(Value) -> Result<Value, ContinuationError> + Send + Sync>,
        description: Option<String>,
    },

    /// JIT: Machine state for compiled code
    Jit {
        return_address: usize,
        saved_registers: Vec<u64>,
        stack_snapshot: Vec<u8>,
    },

    /// Stub: Placeholder
    Stub,
}
```

## Effect Dispatch

### Handler Stack

Handlers are resolved via stack-based scoping:

1. Walk handler stack from top to bottom
2. Match by effect name
3. Fall back to registry if not on stack
4. Error if no handler found

```rust
impl EffectContext {
    fn find_handler(&self, effect: &str) -> Result<&dyn HandlerCapability> {
        // Check stack first (most recent handler wins)
        for handler in self.handler_stack.iter().rev() {
            if handler.effect_name() == effect {
                return Ok(handler.as_ref());
            }
        }

        // Fall back to registry
        if let Some(registry) = &self.registry {
            if let Some(handler) = registry.get(effect) {
                return Ok(handler.as_ref());
            }
        }

        Err(EffectError::UnhandledEffect { effect, operation: "" })
    }
}
```

### Dispatch Flow

```
1. perform effect.operation(args)
   ↓
2. Capture continuation (closure)
   ↓
3. Find handler (stack → registry)
   ↓
4. handler.handle(operation, args, continuation, state)
   ↓
5. Resume or Suspend
   ↓
6. Cleanup continuation
```

## Handler Implementation

### Handler Capability Trait

```rust
pub trait HandlerCapability: Debug {
    /// Name of the effect this handler handles
    fn effect_name(&self) -> &str;

    /// Handle an effect operation
    fn handle(
        &self,
        operation: &str,
        args: &[Value],
        continuation: Continuation,
        state: &mut CapabilityHandlerState,
    ) -> CapabilityResult;

    /// Get linearity constraint for an operation
    fn operation_linearity(&self, operation: &str) -> Linearity {
        Linearity::ExactlyOnce  // Default
    }
}
```

### Available Handlers

| Handler | Purpose | Status |
|---------|---------|--------|
| **IO** | I/O operations (print, read_file, write_file) | ✅ Working |
| **Mut** | Mutable state (get, set, modify) | ✅ Working |
| **Alloc** | Memory allocation | ✅ Stub |
| **Panic** | Runtime panics (assert, unwrap) | ✅ Stub |
| **Async** | Async/await (spawn, await) | ✅ Stub |
| **Prob** | Probabilistic effects (sample, observe) | ✅ Stub |
| **GPU** | GPU kernels | ✅ Stub |
| **Network** | Network I/O | ✅ Stub |
| **Sensor** | Sensor data | ✅ Stub |
| **Epistemic** | Uncertainty tracking | ✅ Working |
| **Exn** | Exceptions (throw, catch) | ✅ Stub |
| **Div** | Division checking | ✅ Stub |

## One-Shot Enforcement

### Runtime Checks

```rust
impl CapturedContinuation {
    pub fn resume(&mut self, value: Value) -> Result<Value, ContinuationError> {
        // Check one-shot enforcement
        if !self.is_multi_shot && self.resume_count > 0 {
            return Err(ContinuationError::AlreadyResumed {
                id: self.id,
                label: self.label.clone(),
            });
        }

        self.resume_count += 1;

        // Resume the continuation
        // ...
    }
}
```

### Error Reporting

```rust
pub enum ContinuationError {
    AlreadyResumed { id: ContinuationId, label: Option<String> },
    InvalidState { message: String },
    ResumeFailure { source: Box<dyn Error> },
}
```

Labels are preserved for debugging:
```
Error: Continuation already resumed
  Continuation: IO.print (id: 0x1234)
  Previous resume: line 42
```

## Testing

### Test Coverage

**34 integration tests** across 4 test files:

1. **effect_handler_continuations.rs** (6 tests)
   - Basic continuation capture
   - Cleanup on abort
   - Store tracking
   - Registry integration
   - Multiple sequential dispatches

2. **effect_oneshot_enforcement.rs** (6 tests)
   - Single resume (success)
   - Double resume (failure)
   - Multi-shot multiple resumes
   - Linearity types
   - Compatibility checking

3. **effect_complex_scenarios.rs** (10 tests)
   - Multiple sequential effects
   - State persistence
   - IO operations
   - Div operations
   - Handler registry completeness

4. **effect_individual_handlers.rs** (12 tests)
   - One test per handler
   - Verifies dispatch and cleanup

### End-to-End Test

`examples/effects/comprehensive_effects.sio`:
- 6 test scenarios
- Multiple effects (IO, Mut, Div)
- Nested handlers
- State persistence
- All passing with JIT interpreter

### Running Tests

```bash
# All effect tests
cargo test effect --quiet

# Specific test files
cargo test --test effect_handler_continuations
cargo test --test effect_oneshot_enforcement
cargo test --test effect_complex_scenarios
cargo test --test effect_individual_handlers

# End-to-end Sounio program
cargo run --bin souc --features jit -- run examples/effects/comprehensive_effects.sio
```

## Usage Examples

### Simple Effect

```sounio
fn greet(name: String) with IO {
    println("Hello, " ++ name ++ "!")
}

fn main() with IO {
    greet("World")
}
```

### Stateful Effect

```sounio
fn counter() with Mut, IO {
    let count = get("counter")
    set("counter", count + 1)
    println("Count: " ++ count)
}

fn main() with Mut, IO {
    set("counter", 0)
    counter()  // Prints: Count: 1
    counter()  // Prints: Count: 2
}
```

### Nested Handlers

```sounio
fn inner() with IO {
    println("  Inner")
}

fn outer() with IO {
    println("Outer start")
    inner()
    println("Outer end")
}

fn main() with IO {
    outer()
}
```

## Performance Characteristics

### One-Shot vs Multi-Shot

| Metric | One-Shot | Multi-Shot |
|--------|----------|------------|
| Resume cost | O(1) | O(n) copy |
| Memory | Single allocation | Clone per resume |
| Speedup | Baseline | 3-5x slower |
| Use cases | 95%+ | Backtracking, Amb |

### Optimization Strategy

1. **Default to one-shot**: ExactlyOnce linearity
2. **Opt-in to multi-shot**: Explicit annotation
3. **Runtime enforcement**: Prevent double-resume
4. **Label tracking**: Better error messages

## Implementation Status

### ✅ Complete (Phases 1-4)

- [x] Continuation capture (closure-based)
- [x] Effect dispatch with handler stack
- [x] Handler capability trait
- [x] 12 concrete handlers (stubs + IO/Mut/Epistemic working)
- [x] Linearity type system
- [x] One-shot enforcement
- [x] Label tracking for debugging
- [x] Comprehensive test suite (34 tests)
- [x] End-to-end Sounio programs

### 🚧 Future Work (Optional)

- [ ] CPS transformation for compiled backend
- [ ] Row polymorphic effects (type system)
- [ ] Gradual effect typing
- [ ] Session-typed effects
- [ ] Effect handler optimizations (regions, static analysis)
- [ ] Complete remaining handler implementations

## Research Foundations

### Affect (POPL 2025)
- **One-shot vs multi-shot tracking**: Default to one-shot for 3-5x speedup
- **Linearity inference**: Automatically detect one-shot opportunities
- **Runtime enforcement**: Prevent double-resume errors

### Soundly Handling Linearity (POPL 2024)
- **Linear types + effects**: Track resources through continuations
- **Affine semantics**: OnceOrNever for optional cleanup
- **Type safety**: Prevent resource duplication via multi-shot

### Retrofitting Effect Handlers onto OCaml (PLDI 2021)
- **Incremental approach**: Interpreter first, compiled later
- **Closure capture**: Avoid full CPS transformation initially
- **Backward compatibility**: Gradual rollout strategy

## Migration Path

### For Existing Code

1. **Add effect annotations**: `fn foo() with IO, Mut`
2. **Run type checker**: Effect inference suggests annotations
3. **Test incrementally**: Effect tests verify behavior
4. **Performance check**: Benchmark one-shot vs multi-shot

### For New Code

1. **Design with effects**: Think about effect signatures upfront
2. **Use linearity**: Default ExactlyOnce, opt into MultiShot when needed
3. **Test handlers**: Verify continuation cleanup
4. **Document effects**: Explain which effects functions use

## Debugging

### Common Issues

**Double Resume Error**
```
Error: Continuation already resumed
  Continuation: IO.print (id: 0x5f3a)
```
→ Check for accidental resume calls in handler

**Unhandled Effect**
```
Error: Unhandled effect: CustomEffect
  Operation: my_operation
```
→ Add handler to registry or push onto handler stack

**Continuation Leak**
```
Warning: 5 active continuations after cleanup
```
→ Ensure handlers call resume() or return

## Contributors

- Implementation: Phases 1-4 (January 2026)
- Research: Q1 2025 literature review
- Testing: 34 integration tests

## License

Same as Sounio language (see root LICENSE file)
