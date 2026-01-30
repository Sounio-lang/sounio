# Known Language Limitations

This document tracks limitations in the Sounio language implementation.

## v1.0 Release Status

As of **v1.0.0**, all core features are implemented. Advanced features require
external dependencies (LLVM, Z3, CUDA) for full functionality.

### Native Epistemic Backend
- **Status**: ✅ Implemented (v1.0.0)
- `Knowledge<T>` works in interpreter with full GUM uncertainty propagation
- SIR epistemic insertion pass transforms `BinOp` on epistemic values
- Native codegen emits uncertainty propagation calls
- **Files**: `sir/passes/epistemic_insertion.rs`, `backend/native/epistemic_runtime.rs`

### SMT Refinement Verification
- **Status**: ✅ Infrastructure complete (v1.0.0)
- Refinement types parse and typecheck correctly
- Z3 solver integration in `smt/z3_solver.rs`
- `refine_assert.rs` pass for compile-time verification
- **Requires**: Z3 library + `--features smt` for compile-time proofs
- **Fallback**: Runtime assertions when Z3 unavailable

### GPU Kernel Launch
- **Status**: ✅ Implemented (v1.0.0)
- PTX, Metal, and SPIR-V code generation complete
- `GpuRuntimeBridge` singleton for kernel launch dispatch
- Thread-safe design with `OnceLock<Mutex<...>>`
- **Files**: `runtime/gpu_bridge.rs`, `codegen/gpu/runtime.rs`
- **Requires**: CUDA toolkit for actual kernel execution

### Package Manager
- **Status**: ✅ Implemented (v1.0.0)
- Full `HttpRegistry` with REST API (fetch, search, publish, yank)
- Manifest parsing with Git dependency support (`git`, `branch`, `tag`, `rev`)
- Package resolution and caching infrastructure
- **Files**: `pkg/registry.rs`, `pkg/manifest.rs`, `pkg/build.rs`

### Async Join/Select
- **Status**: ✅ Implemented (v1.0.0)
- `TaskScheduler` with ready queue, suspension tracking, dependency graphs
- `JoinFuture` waits for ALL tasks, `SelectFuture` waits for ANY
- Bounded/unbounded channels with `Sender<T>`/`Receiver<T>`
- Full test coverage (37 tests)
- **Files**: `runtime/async_runtime.rs`, `runtime/handler_stack.rs`

### Optional External Dependencies

| Feature | Dependency | Effect if Missing |
|---------|------------|-------------------|
| `--features llvm` | LLVM 15+ | Use Cranelift JIT instead |
| `--features smt` | Z3 + cmake | Refinement types fall back to runtime checks |
| `--features gpu` | CUDA toolkit | GPU codegen works, execution requires runtime |

---

## Syntax Limitations - All Resolved

This section documents previously-resolved limitations for historical context.

## Syntax - All Resolved

### Module System
- **Status**: Resolved (v0.99.0)
- **Resolution**: Full `module`/`use` support with file-based module loading and hierarchical namespace resolution.

### Visibility Modifiers
- **Status**: Resolved (v0.99.0)
- **Resolution**: `pub` visibility supported and enforced across module boundaries.

### Logical Operators
- **Status**: Resolved (v0.66.0)
- **Resolution**: `&&` and `||` implemented with short-circuit evaluation and boolean type checking.
```sio
if a > 0 && b > 0 { ... }
if is_empty || is_null { ... }
```

### Documentation Comments
- **Status**: Resolved (v0.99.0)
- **Resolution**: `///` outer docs and `//!` inner docs are parsed and preserved through AST → HIR.

### Numeric Literals
- **Status**: Resolved (v0.99.0)
- **Resolution**: Scientific notation supported in the lexer (e.g., `1e10`, `1.5e-3`).

## Type System - All Resolved

### Type Aliases
- **Status**: Resolved (v0.99.0)
- **Resolution**: `type` aliases are supported, including generic aliases; aliases expand transparently during type checking.
```sio
type Vec2 = (f64, f64)
```

### Unit Definitions
- **Status**: Resolved (v0.99.0)
- **Resolution**: User-defined units are supported and integrate with unit checking.
```sio
unit kg;
unit mg = 0.001 * kg;
unit velocity = m / s;
```

## Reserved Keywords

The following identifiers are reserved and used by the language:
- `var` - mutable binding
- `effect` - effect declaration
- `type` - type alias definition
- `module` - module declaration
- `use` - module import
- `pub` - public visibility modifier
- `unit` - unit definition

## Scoping Behavior - All Resolved

### Variable Shadowing
- **Status**: Resolved (v0.99.0)
- **Resolution**: Shadowing works correctly across nested scopes.

### Forward Declarations
- **Status**: Resolved (v0.99.0)
- **Resolution**: 2-pass resolver enables forward references and mutual recursion.

## Feature Resolution Summary

All previously planned features are implemented as of v0.99.0:

| Feature | Resolved In | Resolution |
|---------|------------|------------|
| Module system | v0.99.0 | File-based module loading with `module`/`use` |
| `&&` / `\|\|` operators | v0.66.0 | Short-circuit logical operators |
| `pub` visibility | v0.99.0 | Visibility enforcement across modules |
| Scientific notation | v0.99.0 | Lexer supports `1e10`, `1.5e-3` |
| Type aliases | v0.99.0 | `type Name = Type;` with generics |
| Doc comments | v0.99.0 | `///` + `//!` parsed and preserved |
| Variable shadowing | v0.99.0 | Correct scoping rules |
| Forward declarations | v0.99.0 | 2-pass resolver |
| Unit definitions | v0.99.0 | User-defined units + checking |

## Reporting Issues

If you encounter any new issues, please report them at:
https://github.com/Chiuratto-AI/sounio/issues
