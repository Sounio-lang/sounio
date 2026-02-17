# Known Language Limitations

This document tracks limitations in the Sounio language implementation.
Updated February 2026 after full-project audit.

## Maturity Tiers

### Production (ship with confidence)

| Component | Status | Notes |
|-----------|--------|-------|
| Lexer/Parser/AST | Production | logos-based, error recovery, comprehensive |
| Type Checker (core) | Production | Bidirectional inference, generics, unification |
| Epistemic Types | Production | GUM uncertainty, confidence propagation, provenance |
| Effects System | Production | 8 effects (IO, Mut, Alloc, Panic, Async, GPU, Prob, Div) |
| HIR + HLIR | Production | SSA generation, async transform |
| SIR | Production | Domain-specific IR, epistemic passes |
| Native Backend | Production | ELF/Mach-O, epistemic runtime, continuations |
| Cranelift Codegen | Production | Full implementation, effect handlers |
| Interpreter | Production | Full eval, 100+ builtins |
| Module System | Production | 2-pass resolver, imports, hierarchical namespaces |
| CLI | Production | check/build/run/repl/format/doc |
| Formatter | Production | AST-based, all constructs, diff mode |
| snn/ (sedenion NN) | Production | Training, backward, similarity, 8 scoring functions |

### Beta (works for common patterns, edge cases exist)

| Component | Status | Limitations |
|-----------|--------|-------------|
| Ownership/Borrowing | Beta | Method receiver inference uses heuristic string matching; `infer_method_receiver_use` should look up actual method signatures |
| LLVM Codegen | Beta | Requires LLVM 15+, feature-gated |
| Refinement Types + SMT | Beta | Requires Z3, falls back to runtime assertions |
| LSP | Beta | Cross-file navigation uses legacy symbol index |
| REPL | Beta | 17 commands, JIT, epistemic badges |
| Self-hosted Compiler | Beta | Phase 1.2 complete, 30K lines .sio |
| Ontology | Beta | 10K terms, subsumption, distance |
| Package Manager | Beta | Manifest parsing works, no public registry exists |

### Known Bugs

**Implicit `var` return with `i32` type**: When a function's return type is `i32` and its last expression is a `var` variable (implicit return), the type checker may report `expected I32, found I64`. Workaround: use explicit `return x` instead of trailing `x`. Does not affect `f64` returns or explicit `return` statements.

**Effect checker `Div` propagation**: The effect checker requires `Div` for any operation involving division. It also requires `Panic` alongside `Div` (divide-by-zero potential), and for array access (out-of-bounds potential) and `as` casts. These are strict but correct.

### Pruned/Experimental Modules

The following stdlib modules exist but are stubs or incomplete. They are not part of the default experience:

- `stdlib/gpu/` - requires CUDA runtime (behind `--features gpu`)
- `stdlib/crypto/` - stub
- `stdlib/ffi/` - stub
- `stdlib/compress/` - stub
- `stdlib/autodiff/` - framework only
- `stdlib/interop/` - stub

### Optional External Dependencies

| Feature | Dependency | Effect if Missing |
|---------|------------|-------------------|
| `--features llvm` | LLVM 15+ | Use Cranelift JIT instead |
| `--features smt` | Z3 + cmake | Refinement types fall back to runtime checks |
| `--features gpu` | CUDA toolkit | GPU codegen works, execution requires runtime |

### Platform Support

- **Linux x86-64**: Primary supported platform
- **macOS**: Mach-O backend available, not regularly tested
- **Windows**: Not yet supported

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
