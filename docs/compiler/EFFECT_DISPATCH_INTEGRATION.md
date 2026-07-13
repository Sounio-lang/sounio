<!-- docs:meta
topic_id: repo.docs.compiler.effect-dispatch-integration
authority: historical
audience: contributors
last_validated: 2026-03-07
validated_by: A4
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.compiler.effect-dispatch-integration
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Effect Dispatch System Integration

This document describes the unified effect dispatch system and how it's integrated across all backends.

This is an implementation architecture document. It does not mean every backend
path described here is exposed by the checked public compiler artifacts. Public
docs still need to distinguish source-tree integration from artifact-backed CLI
support.

## Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                     User Sounio Code                        │
│                  (with effect operations)                   │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   HLIR (High-Level IR)                      │
│            Op::PerformEffect / Op::DispatchEffect           │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ▼              ▼              ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│   Native    │ │  Cranelift  │ │    LLVM     │
│   Backend   │ │     JIT     │ │   Backend   │
└──────┬──────┘ └──────┬──────┘ └──────┬──────┘
       │               │               │
       ▼               ▼               ▼
┌──────────────────────────────────────────────┐
│       UnifiedEffectRuntime (new)             │
│         EffectDispatch trait                 │
└──────────────────┬───────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────┐
│    runtime::handler_stack                    │
│    (C-callable __sounio_* functions)         │
└──────────────────────────────────────────────┘
```

## Components

### 1. EffectDispatch Trait (`backend/effect_dispatch.rs`)

The core abstraction that all backends implement:

```rust
pub trait EffectDispatch {
    fn dispatch_effect(&mut self, effect: &str, op: &str, args: &[f64])
        -> Result<f64, EffectError>;
    fn push_handler(&mut self, effect: &str, handler_id: u32)
        -> Result<(), EffectError>;
    fn pop_handler(&mut self) -> Result<(), EffectError>;
    fn handler_depth(&self) -> usize;
}
```

### 2. UnifiedEffectRuntime (`backend/effect_dispatch.rs`)

Wraps `runtime::handler_stack` functions into a safe, high-level interface. Implements `EffectDispatch`.

### 3. EffectOp Enum (`backend/effect_dispatch.rs`)

Type-safe enumeration of all 13 effects and their operations:
- IO (print, println, read, write, flush)
- Mut (get, set, modify)
- Alloc (alloc, dealloc, realloc)
- Panic (panic)
- Async (spawn, yield, await)
- GPU (launch, sync, memcpy)
- Div (div, rem)
- Prob (sample, observe, factor, score)
- Grad (forward, reverse, jvp, vjp)
- Causal (do, observe, counterfactual, query, ate, cate)
- Network (ping, send, recv)
- Sensor (read, calibrate)
- Exn (throw, catch)

## Backend Integration

### Native Backend (`backend/native/effects.rs`)

**Strategy**: Generate AArch64/x86-64 machine code that calls `__sounio_dispatch_*` functions.

**Implementation**:
- `NativeEffectDispatcher` wraps `UnifiedEffectRuntime`
- `emit_effect_dispatch()` generates BL (branch-link) instructions
- Arguments moved to D0-D7 (f64 calling convention)
- Result returned in D0

**Example**:
```rust
let dispatcher = NativeEffectDispatcher::new();
dispatcher.emit_effect_dispatch(
    &mut emitter,
    "IO",
    "print",
    &[AArch64Reg::V2],  // arg in V2
    AArch64Reg::V0       // result in V0
)?;
// Generates:
// FMOV V0, V2          # Move arg to calling convention register
// BL __sounio_dispatch_io_print
// # Result already in V0
```

### Cranelift Backend (`codegen/cranelift.rs`)

**Strategy**: Keep existing `JitEffectState` for specialized handlers, route through unified system for all 13 effects.

**Current State**:
- Has custom `JitEffectState` with Prob/Causal/Grad optimizations
- Only supports 3 effects explicitly (Prob, Causal, Grad)
- Uses custom `runtime_effect_dispatch` C function

**Integration Plan** (Phase 2.2 - future work):
1. Make `JitEffectState` use `UnifiedEffectRuntime` as fallback
2. Keep specialized Prob/Causal/Grad handlers for performance
3. Route all other effects through unified runtime
4. This gives us best of both worlds: performance + completeness

**Pseudocode**:
```rust
impl JitEffectState {
    fn dispatch_effect(&self, effect: &str, op: &str, args: &[f64]) -> Option<f64> {
        // First try specialized handlers (Prob, Causal, Grad)
        if let Some(result) = self.dispatch_specialized(effect, op, args) {
            return Some(result);
        }

        // Fall back to unified runtime for all other effects
        UnifiedEffectRuntime::new().dispatch_effect(effect, op, args).ok()
    }
}
```

### LLVM Backend (`codegen/llvm/codegen.rs`)

**Strategy**: Generate LLVM IR calls to `__sounio_dispatch_generic` and specific dispatch functions.

**Current State**:
- Uses `compile_effect_dispatch()` to generate LLVM IR
- Calls `__sounio_dispatch_generic` with string arguments
- Already compatible with unified system

**No changes needed** - LLVM backend already works correctly with the unified runtime.

## Runtime Layer (`runtime/handler_stack.rs`)

The C-callable layer that all backends ultimately call into:

**Exported Functions**:
```c
// Handler stack management
extern "C" fn __sounio_push_handler_io()
extern "C" fn __sounio_push_handler_mut()
// ... (13 total, one per effect)
extern "C" fn __sounio_pop_handler()

// Effect dispatch (specific)
extern "C" fn __sounio_dispatch_io_print(value: f64) -> f64
extern "C" fn __sounio_dispatch_mut_get(key: f64) -> f64
extern "C" fn __sounio_dispatch_div_div(a: f64, b: f64) -> f64
// ... (40+ specific dispatch functions)

// Generic dispatch
extern "C" fn __sounio_dispatch_generic(
    effect_ptr: *const u8,
    op_ptr: *const u8,
    args_ptr: *const f64,
    args_len: usize
) -> f64
```

## Effect Coverage

| Effect   | Native | Cranelift | LLVM | Operations |
|----------|--------|-----------|------|------------|
| IO       | ✅     | ✅        | ✅   | print, println, read, write, flush |
| Mut      | ✅     | ✅        | ✅   | get, set, modify |
| Alloc    | ✅     | ✅        | ✅   | alloc, dealloc, realloc |
| Panic    | ✅     | ✅        | ✅   | panic |
| Async    | ✅     | ✅        | ✅   | spawn, yield, await |
| GPU      | ✅     | ✅        | ✅   | launch, sync, memcpy |
| Div      | ✅     | ✅        | ✅   | div, rem |
| Prob     | ✅     | ✅ (opt)  | ✅   | sample, observe, factor, score |
| Grad     | ✅     | ✅ (opt)  | ✅   | forward, reverse, jvp, vjp |
| Causal   | ✅     | ✅ (opt)  | ✅   | do, observe, counterfactual, query, ate, cate |
| Network  | ✅     | ✅        | ✅   | ping, send, recv |
| Sensor   | ✅     | ✅        | ✅   | read, calibrate |
| Exn      | ✅     | ✅        | ✅   | throw, catch |

**Legend**: ✅ = supported, (opt) = has optimized implementation

## Testing

Each layer has comprehensive tests:

1. **UnifiedEffectRuntime tests** (`backend/effect_dispatch.rs`)
   - Handler push/pop
   - Effect dispatch for all 13 effects
   - Error handling (invalid args, no handler)

2. **NativeEffectDispatcher tests** (`backend/native/effects.rs`)
   - Code generation verification
   - Function name resolution
   - Calling convention correctness

3. **Integration tests** (existing)
   - `tests/jit_effects.rs` - JIT effect execution
   - `tests/native_effects.rs` - Native backend effect execution (TODO)

## Future Work (Phase 2.2)

1. **Cranelift Integration**: Update `JitEffectState` to use `UnifiedEffectRuntime` as fallback
2. **Native x86-64**: Extend `NativeEffectDispatcher` to support x86-64 (currently AArch64 only)
3. **Performance**: Add benchmarks comparing specialized vs unified dispatch
4. **Handler Composition**: Support custom handler registration in native code
5. **GPU Effects**: Implement actual GPU kernel launch (currently stub)

## Design Principles

1. **Correctness First**: All backends must support all 13 effects
2. **Unification**: Common infrastructure reduces duplication
3. **Performance**: Specialized paths for hot effects (Prob, Grad, Causal)
4. **Safety**: Type-safe Rust wrappers over C ABI
5. **Testability**: Each layer independently testable

## Implementation Status

- ✅ Phase 2.0: Handler ID propagation through SIR pipeline
- ✅ Phase 2.1: Unified effect dispatch trait and runtime
- ⏳ Phase 2.2: Cranelift backend integration (future)
- ⏳ Phase 2.3: Performance benchmarks (future)
