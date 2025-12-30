# Sounio Compiler Audit & Roadmap (December 2025)

This document captures the comprehensive audit findings and prioritized roadmap for Sounio language development.

---

## Executive Summary

| Metric | Value |
|--------|-------|
| **Compiler LOC (Rust)** | 353,428 |
| **Stdlib LOC (Sounio)** | 207,062 |
| **Tests Passing** | 2,817 |
| **Stdlib Modules** | 49 |
| **Compiler Modules** | 70+ |
| **Overall Maturity** | 85% |

**Verdict:** Production-grade core with 3 critical blockers preventing real-world usage.

---

## Critical Blockers (P0)

### BLOCKER 1: Expression Type Checking Incomplete

**Location:** `compiler/src/check/mod.rs:3074-3078`

**Problem:** 9 expression types fall through to a wildcard match returning `Unit`:

```rust
// Line 3074-3078: These silently compile as Unit
_ => {
    (HirExprKind::Literal(HirLiteral::Unit), HirType::Unit)
}
```

**Affected Expressions:**
- `Expr::Perform { effect, op, args }` - Effect operation invocation
- `Expr::Handle { expr, handler }` - Effect handler application
- `Expr::Sample { distribution }` - Probabilistic sampling
- `Expr::Await { expr }` - Async/await
- `Expr::AsyncBlock { block }` - Async block expressions
- `Expr::AsyncClosure { ... }` - Async closures
- `Expr::Spawn { expr }` - Task spawning
- `Expr::Select { arms }` - Multi-future selection
- `Expr::Join { futures }` - Concurrent join

**Impact:** Programs using effects, async, or probabilistic constructs will type-check but produce wrong results at runtime.

**Fix Effort:** ~2 days

**Fix Strategy:**
1. Add match arms for each expression type
2. Implement proper type inference for effect operations
3. Connect to effect inference system
4. Add integration tests for each expression type

---

### BLOCKER 2: Refinement Types Missing SMT Integration

**Location:** `compiler/src/types/refinement.rs:480-494`

**Problem:** The `RefinementChecker` type exists but has no implementation. Panics indicate incomplete code:

```rust
// Lines 480, 493, 494
_ => panic!("Expected And predicate")
_ => panic!("Expected Compare predicate")
_ => panic!("Expected InsufficientConfidence")
```

**Impact:**
- Array bounds checking not verified at compile-time
- Null safety not enforced
- Domain-specific constraints (e.g., `type Positive = { x: i32 | x > 0 }`) not checked

**Fix Effort:** ~3 days

**Fix Strategy:**
1. Implement `RefinementChecker::check()` method
2. Connect to Z3 solver via `compiler/src/smt/`
3. Add predicate evaluation for Compare, Arith, And, Or, Not
4. Implement constraint propagation through expressions
5. Add timeout handling for Z3 queries

---

### BLOCKER 3: Effect Handler Runtime Incomplete

**Location:** `compiler/src/effects/inference.rs`

**Problem:** Effect handler dispatch is skeleton only. The type system tracks effects but runtime doesn't execute handlers.

**Impact:**
- Compiled probabilistic programs don't work
- Effect operations return undefined values
- Only interpreter mode works for effect-using programs

**Fix Effort:** ~3 days

**Fix Strategy:**
1. Implement handler dispatch table in runtime
2. Add continuation-based handler invocation
3. Connect effect inference to codegen
4. Implement handler frames for nested handlers
5. Add tests for Prob, IO, Mut effects

---

## High Priority Gaps (P1)

### GAP 1: GPU SPIR-V Incomplete

**Location:** `compiler/src/codegen/gpu/portable.rs:331-342`

**Problem:** 3 operations not lowered:
```rust
todo!("Abs requires lowering")  // Line 331
todo!("Min requires lowering")  // Line 337
todo!("Max requires lowering")  // Line 342
```

**Fix Effort:** ~1 day

---

### GAP 2: Module Context Missing in Types

**Location:** `compiler/src/check/mod.rs:775, 802, 809`

**Problem:** Types don't track their defining module:
```rust
source_module: None, // TODO: extract from struct's module context
source_module: None, // TODO: extract from enum's module context
```

**Fix Effort:** ~1 day

---

### GAP 3: Unwrap Cleanup

**Problem:** 3,408 `unwrap()` calls in compiler, many in library code

**High-Risk Modules:**
- `ontology/` (~50 unwraps) - SQLite operations
- `distributed/` (~40 unwraps) - Network code
- `interp/` (~60 unwraps) - Interpreter edge cases

**Fix Effort:** ~5 days (can be incremental)

---

## Codegen Backend Status

| Backend | LOC | Completeness | Production Ready |
|---------|-----|--------------|------------------|
| **LLVM** | 47,000+ | 95% | Yes |
| **Cranelift JIT** | 44,367 | 90% | Yes (dev) |
| **GPU PTX (CUDA)** | 5,538 | 90% | Yes |
| **GPU Metal** | 2,157 | 85% | Partial |
| **GPU SPIR-V** | 3,437 | 70% | No |

### LLVM Backend Details
- Full SSA translation from HLIR to LLVM IR
- Optimization levels: O0, O1, O2, O3, Os, Oz
- Cross-compilation: x86_64, AArch64, NVPTX64, SPIR-V
- DWARF debug info generation
- **Known Issue:** Array concatenation stub (line 914)

### Cranelift JIT Details
- Fast compile times for development
- Native print function bindings
- **Best for:** Interactive development, testing

### GPU Backend Details
- PTX: SM_60 through SM_120 (Blackwell Ultra)
- Epistemic GPU computing with shadow registers
- Counterfactual execution on GPU
- **Gaps:** SPIR-V instruction lowering incomplete

---

## Type System Status

| Feature | Status | Notes |
|---------|--------|-------|
| Basic types | 100% | Primitives, structs, enums, arrays |
| Generics | 100% | TypeVar, TypeScheme, substitution |
| Linear/Affine | 100% | Ownership tracking complete |
| Effect types | 100% | IO, Mut, Alloc, Prob, GPU, Epistemic |
| **Refinement** | 35% | **SMT integration missing** |
| Epistemic | 100% | Knowledge[T], confidence, provenance |
| Units | 100% | Dimensional analysis complete |
| Semantic/Ontology | 100% | 15M+ terms, threshold compatibility |
| Multiplicities (QTT) | 100% | Zero/One/Many semiring |
| Erasure | 100% | Ontology types erased at runtime |
| **Expression checking** | 85% | **9 expressions unhandled** |

---

## Stdlib Status

### Production-Ready (Tier 1)

| Module | LOC | Description |
|--------|-----|-------------|
| `core` | 886 | Option<T>, Result<T, E> |
| `collections` | 2,735 | Vec, HashMap, HashSet, Deque |
| `io` | 770+ | File I/O, directories, process |
| `iter` | 1,702 | Iterator trait, combinators |
| `time` | 1,322 | Duration, DateTime, Instant |
| `json` | 1,165 | RFC 7159 compliant |
| `test` | 500+ | Assertions, mocking, property-based |
| `str/string` | 500+ | String utilities |
| `sync` | 300+ | Mutex, RWLock, Channel |

### Domain-Specific (Tier 2)

| Module | LOC | Description |
|--------|-----|-------------|
| `epistemic` | 20,000+ | Knowledge[T], uncertainty, provenance |
| `nn` | 12,700+ | Neural networks, autograd |
| `ode` | 1,500+ | Tsit5, RK4, BDF, Radau5 |
| `linalg` | 1,000+ | BLAS, LAPACK, fixed-size vectors |
| `quantum` | 1,337 | VQE with epistemic bounds |
| `autodiff` | 500+ | Dual numbers, reverse-mode |
| `bayes` | 1,570+ | MCMC, variational inference |
| `causal` | 250+ | Pearl's do-calculus |
| `darwin_pbpk` | 2,000+ | PBPK simulation engines |

### Incomplete/Stubs (Tier 3)

| Module | Status | Priority |
|--------|--------|----------|
| `mem` | Stub | High |
| `ml` | Stub | High |
| `signal` | 13 lines | Medium |
| `stats` | 13 lines | Medium |
| `search` | Stub | Medium |
| `fmri` | 15 lines | Low |
| `connectivity` | 7 lines | Low |
| `fusion` | 7 lines | Low |

---

## Prioritized Roadmap

### Phase 1: Critical Blockers (Week 1-2)

| Task | Effort | Owner | Status |
|------|--------|-------|--------|
| Implement expression type checking for Perform, Handle, Sample | 2 days | - | Not Started |
| Implement expression type checking for Await, Spawn, Select, Join | 2 days | - | Not Started |
| Connect SMT solver to RefinementChecker | 3 days | - | Not Started |
| Implement effect handler runtime dispatch | 3 days | - | Not Started |

### Phase 2: High-Priority Gaps (Week 3-4)

| Task | Effort | Owner | Status |
|------|--------|-------|--------|
| Fix GPU portable ops (Abs, Min, Max) | 1 day | - | Not Started |
| Complete SPIR-V instruction lowering | 2 days | - | Not Started |
| Add module context to type definitions | 1 day | - | Not Started |
| Convert critical unwraps to proper error handling | 3 days | - | Not Started |

### Phase 3: Stdlib Expansion (Week 5-8)

| Task | Effort | Owner | Status |
|------|--------|-------|--------|
| Implement stdlib/mem (allocators) | 3 days | - | Not Started |
| Implement stdlib/ml (beyond nn) | 5 days | - | Not Started |
| Implement stdlib/signal | 3 days | - | Not Started |
| Implement stdlib/stats | 2 days | - | Not Started |
| Implement stdlib/search | 2 days | - | Not Started |

### Phase 4: Polish (Week 9+)

| Task | Effort | Owner | Status |
|------|--------|-------|--------|
| Comprehensive benchmarking vs Julia/NumPy | 3 days | - | Not Started |
| Documentation site | 5 days | - | Not Started |
| Package manager integration | 5 days | - | Not Started |
| IDE plugins (VS Code, IntelliJ) | 5 days | - | Not Started |

---

## Test Coverage

| Category | Count | Status |
|----------|-------|--------|
| Unit tests (embedded) | 100+ | Good |
| Integration tests | 28 files | Comprehensive |
| E2E tests | 6 categories | Working |
| GPU tests | 7 dedicated | Functional |
| Epistemic tests | 3+ files | Thorough |

**Total:** 2,817 tests passing

**Gaps:**
- No tests for unhandled expression types
- Refinement type tests missing
- Effect handler integration tests needed

---

## Architecture Reference

```
compiler/src/
├── lexer/          # Logos-based tokenization
├── parser/         # Recursive descent (4,996 LOC)
├── ast/            # Abstract syntax tree (1,804 LOC)
├── check/          # Bidirectional type inference (4,882 LOC)
├── types/          # Type representations
│   ├── core.rs         # Base types, effects
│   ├── ownership.rs    # Linear/affine tracking
│   ├── refinement.rs   # Refinement predicates (INCOMPLETE)
│   ├── epistemic.rs    # Knowledge types
│   ├── units.rs        # Dimensional analysis
│   ├── semantic.rs     # Ontology types
│   └── erasure.rs      # QTT erasure analysis
├── effects/        # Algebraic effect system
├── hir/            # Typed high-level IR (2,061 LOC)
├── hlir/           # SSA-based low-level IR (2,345 LOC)
├── codegen/
│   ├── llvm/           # LLVM backend (47K LOC)
│   ├── cranelift.rs    # JIT backend (44K LOC)
│   └── gpu/            # GPU backends (42K LOC)
├── epistemic/      # Epistemic inference (14K LOC)
├── ontology/       # Scientific ontologies (230K LOC)
├── interp/         # Interpreter
└── lsp/            # Language Server
```

---

## Key Files for Each Blocker

### Blocker 1 (Expression Type Checking)
- `compiler/src/check/mod.rs` - Add match arms at line 3074
- `compiler/src/ast/mod.rs` - Expression definitions (lines 1234-1580)
- `compiler/src/hir/mod.rs` - HIR expression types

### Blocker 2 (Refinement Types)
- `compiler/src/types/refinement.rs` - RefinementChecker
- `compiler/src/smt/mod.rs` - Z3 integration
- `compiler/src/check/mod.rs` - Connect checker to type inference

### Blocker 3 (Effect Handlers)
- `compiler/src/effects/inference.rs` - Effect inference
- `compiler/src/effects/mod.rs` - Effect definitions
- `compiler/src/interp/` - Reference implementation
- `compiler/src/codegen/cranelift.rs` - JIT handler dispatch

---

## Success Metrics

### Phase 1 Complete When:
- [ ] All 9 expression types properly type-checked
- [ ] Refinement types validated against SMT solver
- [ ] Effect handlers execute in compiled code
- [ ] New integration tests pass for each fixed feature

### Production Ready When:
- [ ] All critical blockers resolved
- [ ] Zero panics in library code paths
- [ ] Benchmark parity with Julia for numerical workloads
- [ ] Documentation complete for all public APIs

---

## Appendix: Audit Commands

```bash
# Run all tests
cd compiler && cargo test

# Check specific feature
cargo run -- check examples/hello.sio --show-types

# JIT execution
cargo run --features jit -- run examples/hello.sio

# Build with all features
cargo build --features full

# Find TODOs
grep -r "TODO\|FIXME" compiler/src --include="*.rs" | wc -l

# Count LOC
find compiler/src -name "*.rs" | xargs wc -l | tail -1
```

---

*Generated: December 2025*
*Sounio Version: 0.97.0*
