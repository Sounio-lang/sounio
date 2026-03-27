<!-- docs:meta
topic_id: repo.docs.compiler.known-limitations
authority: repo_only
audience: contributors
last_validated: 2026-03-07
validated_by: A4
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.compiler.known-limitations
-->

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
| Effects System | Production | 9 effects (IO, Mut, Alloc, Panic, Async, GPU, Prob, Div, Observe) |
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

**`&![T; N]` mutable ref mutation in interpreter** (JIT only): When passing a bare array variable by `&!` reference to a function (`fn f(arr: &![i64; 10000], ...) with Mut`), mutations inside the function are not visible to the caller in the Cranelift JIT runtime. Workaround: wrap the array in a struct and pass `&! StructWithArray`, or use explicit deref `(*arr)[i] = v`. Native compilation is unaffected.

```sio
// Works — struct wrapper pattern
struct SortBuf { data: [i64; 10000] }
fn sort(b: &! SortBuf) with Mut { b.data[0] = 99 }

// Works — explicit deref
fn sort_deref(arr: &![i64; 10000]) with Mut { (*arr)[0] = 99 }

// Broken (JIT only) — bare array index
fn sort_broken(arr: &![i64; 10000]) with Mut { arr[0] = 99 }
```

**`extern "C"` FFI limited to math functions** (JIT only): Only f64-typed extern functions are supported in `$SOUC run`. Integer FFI (`malloc`, `getpid`, etc.) silently terminates due to Cranelift JIT's return register handling (reads XMM0 instead of RAX for integer returns). Native compilation handles integer FFI correctly.

**Observation boundary coverage differs by frontend**: The multi-file checker enforces `with Observe` across comparison, pattern-match, IO, and FFI observation boundaries. `self-hosted/compiler/lean_single.sio` currently enforces `Observe` only for comparison-triggered observation.

**Mixed-Hyper optimizer metadata is conservative**: Registry-driven reassociation activates only when lowering can stamp one unambiguous `hyper_algebra_kind` onto a function. If a function mixes Hyper algebras or the tag is unknown, the optimizer leaves the small e-graph at default settings instead of guessing a registry entry.

### Fixed in Self-Hosted Compiler (activate on $SOUC rebuild)

The following bugs have been fixed in the self-hosted source but require a rebuilt `$SOUC` binary to take effect:

**Implicit `var`/`let` with `i32` type** (fixed): Integer literal narrowing now allows `var x: i32 = 5` without "expected I32, found I64" errors. Literals are compatible with annotated smaller integer types (i32, i8).

**`Option::None` type inference** (fixed): Bidirectional type inference now propagates the expected type for enum variant paths. `let x: Option<i32> = Option::None` correctly infers `Option<i32>`.

**Unit type declarations** (fixed): The resolver now registers `unit` declarations as `SymUnit` (was incorrectly using `SymTypeAlias`).

**String methods** (fixed): `.as_bytes()` and `.len()` are now supported on `string` types in the type checker.

**Turbofish syntax** (added): `func::<T, U>(args)` explicit generic type arguments are now parsed and propagated to call expressions.

**Trait definitions** (added): `trait Name { fn method(); ... }` syntax is now parsed and trait definitions are collected into the `TraitRegistry`. Builtin trait implementations (Copy, Drop, Eq, Ord, Hash, Add, Sub, Mul, Div, Display, Debug) are pre-registered for primitive types.

**Borrow release at call boundaries** (fixed): Borrows taken for function call arguments are now unconditionally released after the call returns, fixing false positive errors on consecutive calls borrowing the same variable.

**Ownership state machine** (wired): The `OwnContext` ownership tracker (2836 lines, 72+ functions) is now integrated into the `Checker` — linear variable registration, ownership transfer on use, and linear-at-end checking at function exit.

**Effect propagation** (verified): Call-site effect checking (`check_callee_effects`) validates that callee effects are a subset of the caller's declared effects, reporting E035 on violations.

### Pruned/Experimental Modules

The following stdlib modules are stubs or incomplete:

- `stdlib/gpu/` - requires CUDA runtime (behind `--features gpu`)
- `stdlib/crypto/` - requires integer FFI (JIT limitation)
- `stdlib/compress/` - requires integer FFI (JIT limitation)
- `stdlib/ffi/` - stub
- `stdlib/autodiff/` - framework only
- `stdlib/interop/` - stub
- `stdlib/text/`, `stdlib/time/`, `stdlib/os/` - disabled (require type conversion from Rust unsigned types)

### Recently Activated Modules

- `stdlib/prob/` - Beta, Normal, MCMC, random distributions (4 modules activated)
- `stdlib/onn/` - Octonion neural network: activation, attention, conv, linear, loss, normalization, optimizer, training (8 modules)
- `stdlib/ontology/` - LOINC, biomedical module, namespaces (3 modules)
- `stdlib/heliobiology/units.sio` - space weather units
- `stdlib/ode/tsit5_multicomp.sio` - multi-compartment adaptive Tsit5 solver
- `stdlib/medlang/` - full MedLang DSL (lexer, parser, AST, codegen, PK models, population, dosing) — all active

### Optional External Dependencies

| Feature | Dependency | Effect if Missing |
|---------|------------|-------------------|
| `--features llvm` | LLVM 15+ | Use Cranelift JIT instead |
| `--features smt` | Z3 + cmake | Refinement types fall back to runtime checks |
| `--features gpu` | CUDA toolkit | GPU codegen works, execution requires runtime |

### Platform Support

- **Linux x86-64**: Primary supported platform
- **macOS**: Mach-O backend implemented (2,512 lines, x86-64 + ARM64), not yet wired into codegen driver
- **Windows**: PE/COFF backend implemented (3,508 lines, x86-64 + ARM64), not yet wired into codegen driver

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
https://github.com/sounio-lang/sounio/issues
