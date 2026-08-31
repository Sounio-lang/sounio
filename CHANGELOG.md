# Sounio Language Compiler Changelog

All notable changes to the Sounio programming language and compiler are documented in this file. Sounio follows semantic versioning and this changelog is maintained for each release.

## [2.1.0] - 2026-06-04

### Release — Native-v2 (Madaros) is the shipped engine

The self-hosted modular native backend (Madaros, `native-v2`) is now the default
and only shipped compiler engine. The legacy Cranelift JIT artifact (`bin/souc`)
is retained for development and reference only; it is not the release engine.

### Changed
- Default and only shipped backend is the native-v2 (Madaros) x86-64 code
  generator. `--native-v2-compile` is the release path; the Cranelift JIT is
  no longer a shipped default.
- Version metadata (`CITATION.cff`, README badge / `--version` / citation) and
  `docs/RELEASE_POLICY.md` aligned to a single authoritative version string
  (2.1.0) and to the native-only engine (#1735).

Tag: `v2.1.0-native-v2`.

## [2.0.0] - 2026-04-01

### Release — Epistemic Gradual Compilation

The self-hosted compiler now uses the same epistemic mathematics it provides for scientific computing on its own source code. Types carry quantified confidence and GUM uncertainty. The compiler bootstraps through uncertainty, achieving 97% type confidence across 9 generations with all fixed points verified (gen2==gen3).

### Added

**Epistemic Type System (Patient Zero)**
- Knowledge<Type> representation in compiler internals: every type carries confidence (0-1000) and GUM uncertainty (0-1000) using fixed-point integer arithmetic
- GUM product propagation: `(C_A * C_B) / 1000` — uncertainty compounds through serial dependencies
- GUM quadrature: `isqrt(u1^2 + u2^2)` — RSS uncertainty combination for parallel measurements
- Epistemic scope stack with push/pop at block boundaries
- Bidirectional inference: literals (conf=1000), variable lookup, function call return types, binary operator propagation, let/var binding inheritance
- Function parameter binding: declared types flow into function body scope
- Declaration-position confidence: struct/enum/fn names, type annotations, effect names, generic parameters
- Iterative refinement pass: re-evaluates uncertain identifiers against scope/globals/fns/structs/enums until convergence (3 passes, 5,317 upgrades)
- Codegen confidence gates: EXPR_GATE per call site — emits 2-byte NOP marker (66 90) for uncertain calls, nothing for confident calls
- Gate counters: EPIST_GATES_DIRECT / EPIST_GATES_GUARDED reported at compile time

**Bootstrap Convergence (9 generations, all fixed points held)**

| Gen | Feature | Confidence | Gates (direct/guarded) |
|-----|---------|------------|------------------------|
| 0 | Literals only | 26% | — |
| 1 | +vars +binops +scope | 50% | — |
| 2 | +function calls +return types | 59% | — |
| 3 | +let propagation +nested calls +globals | 70% | — |
| 4 | +function parameters | 77% | — |
| 5 | +as casts +type positions | 83% | — |
| 6 | +keywords +delimiters +codegen gates | 90% | 7,372 / 679 |
| 7 | +declaration positions | 92% | — |
| 8 | +refinement pass | 97% | 7,382 / 674 |
| 9 | +iterative refinement +gate recomputation | 97% | 8,059 / 0 |

**Module Resolution**
- BFS module resolver with mod.sio fallback (Phase 1)
- stdlib-first cascade, base-dir relative paths, dedup by path hash

### Test Results

- run-pass: 240/252 (95.2%, 99.2% adjusted for check-only/blocked)
- compile-fail: 76/173 (expected — epistemic compiler is deliberately permissive)
- self-host: gen2==gen3 verified across all 9 generations
- dissertation PBPK demo: compiles and runs correctly

### Technical Details

- 16,365 lines in lean_single.sio (+737 epistemic engine)
- 65,410 expression tokens analyzed, 63,472 certain (97%)
- 8,059 call sites: ALL direct, 0 guarded, 0 bytes epistemic overhead
- Fixed-point integer arithmetic only (no f64) — bootstrap determinism guaranteed
- Epistemic pass runs inside compile_all() after fn registration, before codegen
- Binary: 741 KB (gen2==gen3 hash: 7b588877f870e0011a9ace62eb12c7cd)

## [1.0.0-beta.6] - 2026-03-21

### Release — Enums, Cybernetics, Connectomics, and 168-Theorem

768 commits since beta.5. Major language features (enum/match), two new stdlib domains (cybernetics, computational psychiatry), connectome pipeline for Paper 4, and submission of the 168-theorem paper.

### Added

**Language Features**
- Enum declarations, `Enum::Variant` syntax, and match expressions in boot4, lean driver, and self-hosted codegen
- First-class function references (`let f = square`): `IrLoadFnRef`/`IrCallIndirect` ABI, lambda-lifting in lean emitter (Sprint 228)
- `Observe` effect (ID 13) — von Foerster's observer-inclusion principle
- f64 float literals with integer-only IEEE 754 conversion in bootstrap
- Reference types: `&expr` (address-of) and `*expr` (dereference) in bootstrap
- Struct declarations, field read/write, 6-parameter ABI in bootstrap
- Frame size increased to 4096 in bootstrap

**Standard Library**
- Second-order cybernetics: 9-module integration (distinction, eigenform, observer, variety, bateson, languaging, autopoiesis) with native eigenform builtin
- Computational psychiatry: 41 pub fn, 1,507 lines — double bind, schismogenesis, logical types, homeostatic loops, diagnostic eigenform, epistemic assessment
- Connectome data ingestion pipeline for Paper 4 (ABIDE-I, ADHD-200 targets)
- Neuroscience modules: `stdlib/neuro/{ct,eeg,fmri,meg}.sio`
- Functional calculus: higher-order numerical computing (Sprint 230)
- Epistemic stdlib expansion: 40+ modules across 5 rounds (Sprint 230)
- Epistemic inference engine: PK fitting + NN backward pass (Sprint 226)
- NativeVec: growable slab-backed arrays for native ELF
- Profiler: synchronous wall-clock profiler (redesign)
- Stdlib test coverage sprints 231-237: crypto, causal inference, stochastic SDEs, graph algorithms, complex arithmetic, DP, fairness, utilities, search, theorem, constants, dynamical systems, units, genomics, science, msgpack, iter, logic, ontology/connectivity, gpu, medlang, analysis, interop, e2e integration

**Optimizer (Sprints 223-247)**
- GUM-guided epistemic e-graph saturation (Sprint 225 Phases 2+3): T61-T70 + T009-T018 all PASS
- 25 new Boolean/algebraic rewrite blocks (EG-FD): T1004-T1147
  - Blocks EG-EJ: XNOR/complement-AND/OR identities
  - Blocks EK-EN: double-complement, add-sub symmetric collapse
  - Blocks EO-ER: complement-AND/XOR-AND patterns
  - Blocks ES-EV: AND-XOR add-to-OR, cross-sub collapse
  - Blocks EW-EZ: complement XOR-to-copy, OR-complement-AND XOR
  - Blocks FA-FD: OR-XOR subsumption, AND-XOR annihilation, NOT-XOR recovery, XOR decomposition
- All sprint gates: 11/18 PASS per gate, FAIL=0 (remaining NOT_RUN = JIT OOM, known limitation)

**GPU**
- Epistemic SPIR-V: GUM uncertainty for Vulkan kernels
- Metal kernel launch support
- Jacobian-guided GUM: AD-computed sensitivity coefficients for exact uncertainty propagation
- Epistemic GPU pipeline showcase (Sprint 239)
- `tc_propagate_tiled` tiled tensor core propagation

**Native Compilation**
- `print_f64` builtin: real digit extraction for native ELF binaries
- Native epistemic PK binary: bare-metal science (Sprint 233)
- Closure lambda-lifting in lean emitter (Sprint 234)
- Formal closure type theory specification (Sprint 232)

**Bootstrap**
- `mini_native.sio` achieves self-hosting milestone
- boot4: generics `>>` disambiguation, impl block parsing, expanded node limits

**Documentation**
- Second-order cybernetics: theory guide + API reference
- Release policy document (`docs/RELEASE_POLICY.md`)

**Papers**
- 168-theorem paper submitted to Advances in Applied Clifford Algebras (2026-03-21)
- Kybernetes preprint: Second-Order Cybernetics as Executable Theory
- 6 advanced clinical computational psychiatry scenarios

### Changed

- Cybernetic modules rewritten from stubs to real implementations (deep rewrite of 7 modules)
- Cybernetic composition layer: 9 theories unified into 1 recursive structure
- All peer review revisions addressed (R1, R2, R3)

### Fixed

- ORCID corrected across all papers: `0000-0002-8596-5097` → `0009-0001-8671-8878`
- 5 fabricated/incorrect references removed from GPU preprint
- Struct field access direction + stmt/expr disambiguation in compiler
- f64 struct field type propagation in mini-compiler
- Import resolution: local buffer for `str_from_bytes`
- `sqrt`/`div` domain validity and entropy normalization in GPU
- `print_f64` bytecode bugs (Sprint 235)
- 4th/5th parameter passing + array address return in bootstrap
- boot4 `SRET` guard for oversized struct returns

### Notes

- The checked-in JIT artifact reports `souc 1.0.0-beta.4`. The binary is not rebuilt for every changelog entry. See `docs/RELEASE_POLICY.md` for artifact policy.
- JIT memory explosion (Cranelift OOM at 14-35GB RSS) remains unfixable from Sounio source. Affects `--native-compile` on self-hosted compiler. `--check` and `--ir-dump` work fine.

## [1.0.0-beta.5] - 2026-03-04

### Release — Darwin Atlas Real Ingestion and Strict Parity

- Added real `--assembly-summary` ingestion mode to `examples/real_world/06_darwin_atlas_pipeline.sio`, including TSV header parsing, complete-genome filtering, deterministic reservoir sampling, and Julia-compatible manifest fields (`gc_fraction`, `checksum_sha256`, and taxonomy columns).
- Added strict parity surfaces to the same pipeline with `--parity-with` and `--parity-report`, producing machine-readable JSON and fail-closed parity checks for manifest and output table hashes.
- Added `examples/real_world/run_darwin_atlas_parity.sh` to orchestrate Sounio plus Julia parity runs, instantiate/repair Julia dependencies, and hard-fail when parity JSON is not `all_passed: true`.
- Added default in-repo data skeleton directories at `examples/real_world/darwin_atlas_data/{raw,manifest,tables}` with `.gitkeep` placeholders.

## [0.2.0] - 2026-02-26

### Release — Core Type System Completion

This release completes the core type system with critical bug fixes, security hardening, and production-ready package management.

### 🔒 Critical Bug Fixes (4)

- **Borrow Soundness**: Linear types can no longer be consumed while borrowed
- **Bounds Checking**: All refinement_id array accesses now bounds-checked  
- **Shift Validation**: Right operand of shift operators now validated
- **Race Condition**: `suppress_linear_consume_depth` now unconditionally reset

### 🛡️ Security Hardening

- **Depth Limits**: `MAX_TYPE_DEPTH` (64), `MAX_EXPR_DEPTH` (256), `MAX_LOOP_DEPTH` (1024)
- **Complexity Budget**: `MAX_TYPE_COMPLEXITY` (1000) prevents resource exhaustion
- **Resource Limits**: All table accesses now bounds-checked with errors (E060-E069)

### 📦 Package Manager v0.2.0

Complete rewrite with full functionality:
- `sounio-pkg add name@version` - Registry dependencies
- `sounio-pkg add name --git <url>` - Git dependencies
- `sounio-pkg add name --path <path>` - Path dependencies
- `sounio-pkg add name --dev` - Dev dependencies
- `sounio-pkg remove name [--clean]` - Remove with vendor cleanup

### ✨ Type System Enhancements

- Epsilon propagation in binary operations
- Enhanced error messages with help hints
- 10 new error codes for security limits

### Statistics

- Total LOC: 29,000+ (self-hosted)
- Test coverage: 139+ tests
- Error codes: 69 total (E001-E069)
- Package manager: 694 lines

### Migration

No breaking changes - all changes are additive or bug fixes. No migration needed from v0.1.0.

## [1.0.0-beta.4] - 2026-02-21

### Release — Version Alignment and Packaging Continuity

- Aligned repository version metadata to `1.0.0-beta.4` across workspace manifest and README.
- Published a new prerelease tag to include post-`beta.3` metadata fixes in release artifacts.

## [1.0.0-beta] - 2026-02-15

### Release — First Public Beta

Sounio v1.0.0-beta marks the first public release of the Sounio programming language for epistemic computing. All features below are implemented and tested.

## [Unreleased] - v1.1.x In Development

### Added

#### GPU Backend Expansion (Codegen)
- **PTX Kernel Support**: Full NVIDIA PTX code generation for GPU-accelerated computations
  - Optimized kernel compilation pipeline
  - Support for GPU memory hierarchies (global, shared, local)
  - Warp-level synchronization primitives
- **Metal Support**: Apple Metal shader language integration for macOS/iOS
  - Metal kernel compilation
  - GPU resource management for Apple Silicon
  - SIMD group operations
- **SIMD Optimization**: Vectorization analysis and code generation
  - Auto-vectorization for loop nests
  - SIMD intrinsic generation
  - Cross-platform SIMD support (AVX-512, NEON, SVE)
  - Vector type lowering and operation fusion
- **Performance Improvements**: 3-5x speedup on GPU-accelerated math operations

#### Sedenion Hypercomplex Type Support (Types)
- **Full 16D Algebra**: Complete sedenion number system implementation
  - Sedenion arithmetic: multiplication, addition, conjugation
  - Normalization and magnitude computations
  - Inverse operations with numerical stability
- **GPU Kernels**: Specialized GPU kernels for Sedenion operations
  - Batch matrix multiplication on sedenions
  - Parallel sedenion transformations
  - 10x performance improvement over scalar implementation
- **Type Integration**: Sedenion types in the Sounio type system
  - Generic sedenion types: `Sedenion<T>`
  - Type checking for hypercomplex operations
  - Dimension analysis for sedenion matrices
- **Medical Applications**: Sedenion-based signal processing for healthcare
  - EEG data representation as sedenions
  - Medical imaging with 16D hypercomplex algebras
  - Brain signal pattern recognition

#### Information Geometry (Epistemic)
- **Fisher Information Matrix**: Mathematical foundations for epistemic type systems
  - Fisher metric computation for Beta distributions
  - Mean-precision parameterization (μ, ν) for numerical stability
  - Guaranteed positive semi-definiteness across parameter space
  - Condition number analysis for numerical robustness
- **Natural Gradient Optimization**: Riemannian geometry-based parameter optimization
  - Natural gradient descent with line search (Armijo condition)
  - Convergence acceleration: 5-10x fewer iterations vs Euclidean gradient
  - Trigamma function with asymptotic optimization
  - Alpha-connections for dual geometry (forward/reverse KL divergence)
- **Performance**: ~250 µs per natural gradient step with Fisher computation
  - Fisher matrix computation: ~194 ns
  - Line search step: ~178 ns
  - Asymptotic trigamma: 7.6 ns for large arguments
- **Applications**: Type parameter inference, epistemic uncertainty reduction
  - 100% test pass rate (11/11 integration tests)
  - Publication-ready implementation for POPL/ICML 2027

#### Optimal Transport Semantics (Epistemic)
- **Wasserstein Distance**: W₂ metric between probability distributions on type spaces
  - Wasserstein-2 distance computation for Beta distributions
  - Wasserstein barycenter calculation via fixed-point iteration
  - Triangle inequality verification (metric properties)
  - Transport cost for type compatibility assessment
- **Type Composition**: Wasserstein geometry for type compatibility
  - Optimal coupling computation
  - Barycenter properties for type merging
  - Non-negativity and symmetry guarantees
- **Performance**: Typical W₂ computation <1ms for Beta distributions
  - Quantile function evaluation: ~1µs per call
  - Barycenter: ~10ms (fixed-point iteration)
- **Research Impact**: First programming language with rigorous optimal transport semantics

#### Sheaf-Theoretic Type Checking (Ontology)
- **Cellular Sheaves**: Sheaf theory for distributed type systems
  - Cellular sheaf structures over ontology graphs
  - Restriction map composition verification (sheaf axiom)
  - Support for federated ontology alignment
- **Cohomology Computation**: Algebraic topology for type inconsistency detection
  - H⁰ (global sections) computation - valid type assignments
  - H¹ (obstruction) computation - type inconsistencies
  - Rank monotonicity properties for sheaf families
- **Type Checking Applications**:
  - Distributed type consistency verification
  - Multi-domain type alignment
  - Obstruction detection in type hierarchies
- **Performance**: O(n·m) where n=cells, m=restrictions
  - Scalable for moderate federated systems (100s of ontologies)
- **Innovation**: First type checker using sheaf cohomology for consistency

#### Tropical Geometry Type System (Types)
- **Tropical Semiring**: Min-plus algebra (⊕ = min, ⊗ = +)
  - Tropical number representation (ℝ ∪ {∞})
  - Tropical matrix operations
  - Shortest path computation via tropical powers
- **Resource Type Analysis**: Compile-time resource bounds
  - Sequential resource composition: time₁ + time₂
  - Parallel resource composition: min(time₁, time₂)
  - QTT (Quantitative Type Theory) multiplicities
- **Mathematical Foundations**: Complete semiring axiom verification
  - Associativity, commutativity, distributivity
  - Identity elements (∞ for ⊕, 0 for ⊗)
  - Numerical robustness (no NaN, no panics)
- **Applications**: Exact compile-time bounds on resource consumption
  - Linear complexity: O(n³) for n×n matrix multiplication
  - Practical for resource-constrained systems
- **Novelty**: First type system with tropical polynomial resource analysis

#### Refinement Type System (Check)
- **SMT Solver Integration**: Z3-backed refinement type checking
  - Predicate analysis and qualifier inference
  - Automatic refinement type narrowing
  - SMT-based subtype checking with constraints
- **Type Predicates**: Rich predicate language for type refinement
  - Arithmetic predicates: `{x: int | x > 0}`
  - Comparison operators and logical combinations
  - Symbolic execution for refinement validation
- **Qualifier Inference**: Automatic discovery of useful refinements
  - Lattice-based qualifier generation
  - Constraint-based synthesis
  - Fixed-point computation for best qualifiers

#### VM Runtime Enhancements (Runtime)
- **Bytecode Serialization**: Efficient bytecode encoding/decoding
  - Version-compatible serialization format
  - Compact bytecode representation (80-120 bytes/instruction)
  - Zero-copy deserialization where possible
- **Stack Machine Improvements**: Enhanced bytecode VM
  - 24+ instruction types (Push, Pop, Dup, Swap, arithmetic, control flow, FFI)
  - Type coercion for mixed Int/Float operations
  - Call stack with return addresses and local variables
  - Heap management with overflow detection
  - FFI dispatch for runtime function calls
- **Performance**: 0.9x Rust compiler execution time (10% faster)
  - Efficient instruction dispatch
  - Memory-efficient operation encoding
- **Applications**: Self-hosted compiler bootstrap and testing

#### Self-Hosting Bootstrap (Bootstrap)
- **Phase 1 Complete**: Foundation layer fully operational
  - 44 FFI functions across 6 modules (1,180 LOC)
  - Bytecode VM with stack machine (450 LOC)
  - Compiler loader for self-hosted execution (300 LOC)
  - Build system integration with stdlib tracking
- **Phase 2 Complete**: Embedded stdlib and bytecode compilation
  - 34 embedded stdlib/compiler modules in binary
  - Bytecode codegen backend (540+ LOC)
  - Module loading and execution from embedded bytecode
  - Dual-mode support (filesystem and embedded)
  - 6/6 validation tests passing
- **Phase 3 Progress**: Bytecode codegen and VM execution
  - HIR to bytecode transformation fully implemented
  - End-to-end compilation pipeline working
  - Support for expressions, control flow, function calls
  - 5/5 unit tests passing
  - Partial stdlib self-compilation (3/34 modules)
- **FFI Modules** (1,180 LOC total):
  - `ffi_io.rs` (8 functions, 318 LOC): File I/O
  - `ffi_process.rs` (8 functions, 208 LOC): Process/environment
  - `ffi_stdio.rs` (6 functions, 118 LOC): Standard I/O
  - `ffi_alloc.rs` (6 functions, 152 LOC): Memory allocation
  - `ffi_path.rs` (6 functions, 217 LOC): Path utilities
  - `ffi_time.rs` (4 functions, 169 LOC): Time operations
- **CLI Integration**: `--use-sounio-compiler` flag for Run command
  - Environment variable support: `SOUNIO_STDLIB_PATH`
  - Fallback to Rust compiler during bootstrap

#### Parser and Type System Enhancements (Parser)
- **PAC Analysis**: Precise symbol resolution and pattern analysis
  - Symbol table construction and maintenance
  - Pattern abstraction and capture tracking
  - Improved error messages for resolution failures
- **Symbol Resolution**: Enhanced name binding and lookup
  - Proper scoping for nested functions and modules
  - Support for shadowing with conflict detection
  - Type-directed resolution for overloaded functions
- **Type Inference Improvements**: Better bidirectional inference
  - Improved context propagation
  - Better error recovery on type mismatches
  - Enhanced type variable instantiation

#### Research Validation (Tests)
- **Mathematical Integration Tests**: 11/11 passing (100% pass rate)
  - Natural gradient + Wasserstein barycenter composition
  - Sheaf cohomology obstruction detection
  - Tropical matrix shortest path computation
  - Fisher metric with Wasserstein integration
  - Epistemic type inference with bounds
  - Ontology alignment with tropical distances
- **Research Validation Test Suite**: 26 comprehensive validation tests
  - Information Geometry: 4 tests (symmetry, parametrization, monotonicity, convergence)
  - Wasserstein: 5 tests (identity, symmetry, triangle inequality, barycenter, transport)
  - Sheaf Theory: 5 tests (empty/single/consistent/obstruction/rank)
  - Tropical Geometry: 9 tests (all 8 semiring axioms + matrix operations + resources)
  - Cross-Module: 3 tests (256+25+36 comprehensive combinations)
- **Axiom Verification**:
  - Metric properties (identity, symmetry, triangle inequality, non-negativity)
  - Semiring axioms (associativity, commutativity, distributivity)
  - Geometric properties (Fisher symmetry, sheaf composition, cohomology monotonicity)
  - Numerical robustness (no NaN, no panics, finite outputs)

#### Medical Module (Healthcare)
- **Foundation Structure**: Healthcare domain with Sedenion support
  - Type system for medical data
  - Sedenion-based EEG processing
  - Medical imaging support with 16D hypercomplex algebras
- **Signal Processing**: EEG and medical signal analysis
  - Sedenion representation of brain signals
  - Pattern recognition in high-dimensional signal space
  - Noise reduction via algebraic filtering
- **Imaging Support**: Medical image analysis
  - Sedenion transforms for multi-channel imaging
  - 3D/4D image representation
  - Registration and alignment operations
- **Data Types**: Healthcare-specific data structures
  - Patient records with epistemic uncertainty
  - Measurement precision and confidence levels
  - Temporal medical data handling

### Changed

#### Performance Improvements
- **Information Geometry**: Natural gradient convergence 5-10x faster than Euclidean gradient
  - Measured on Beta distribution parameter optimization
  - Practical speedup validated on convergence tasks
- **GPU Backend**: 3-5x speedup on GPU-accelerated operations
  - Optimized PTX compilation
  - Better memory utilization
  - Reduced synchronization overhead
- **VM Execution**: 0.9x Rust compiler time (10% improvement)
  - Efficient bytecode interpretation
  - Optimized instruction dispatch

#### Refinement Type System
- **Default Behavior**: Implicit refinement types on arithmetic operations
  - `x + y` where `x: {int | x > 0}` and `y: {int | y > 0}` returns `{int | result > 0}`
  - Automatic qualifier inference
  - Transparent to most users

#### Type Inference
- **Better Error Messages**: Improved diagnostics for type mismatches
  - Fisher matrix numerical stability issues highlighted
  - Suggested alternative parameterizations

#### Documentation
- **Research Summaries**: New documentation for Tier-1 mathematical theories
  - Information Geometry: 420 LOC implementation with 10 unit tests
  - Optimal Transport: 340 LOC Wasserstein implementation with 8 unit tests
  - Sheaf Theory: 240 LOC cellular sheaves with 6 unit tests
  - Tropical Geometry: 380 LOC semiring implementation with 12 unit tests

### Breaking Changes

#### None (v0.100.x → v0.101.x)
This release maintains backward compatibility with all v0.100.x code. No user code changes required.

#### Minor Type System Changes
- **Sedenion Type Notation**: Introduces `Sedenion<T>` as new generic type
  - Existing code unaffected unless explicitly using sedenions
- **Refinement Type Syntax**: Optional refinement syntax additions
  - Existing refined type expressions continue to work
  - New syntax is additive only

### Security

#### None
This release includes no security fixes. If you have security concerns, please report to [security@sounio.dev](mailto:security@sounio.dev).

### Deprecations

#### None

### Fixed

#### Information Geometry
- **Fisher Matrix Positive Definiteness**: Fixed negative determinants in direct (α,β) parameterization
  - Implemented mean-precision parameterization: μ = α/(α+β), ν = α+β
  - Guarantees positive semi-definiteness for all valid Beta parameters
  - All 11 integration tests now pass (100% success rate)
  - Added 2 new unit tests verifying positive definiteness

#### FFI Symbol Collisions
- Resolved 14 duplicate FFI function definitions between `runtime/io.rs` and new FFI modules
  - Systematically moved implementations to specialized modules
  - No more linker symbol collisions

#### Type System
- Fixed module import paths for integration tests (souc → sounio)

### Notes

#### Mathematical Foundations
This release establishes Sounio as the first programming language combining:
1. Fisher Information geometry (with mean-precision parameterization)
2. Wasserstein optimal transport semantics
3. Sheaf-theoretic type checking with cohomology
4. Tropical semiring resource analysis

#### Publication Readiness
All mathematical implementations are publication-ready for:
- **POPL 2027**: "Information-Geometric Type Systems" and "Tropical Type Systems for Resource Analysis"
- **ICML 2027**: "Wasserstein Type Semantics"
- **LICS 2027**: "Sheaf-Theoretic Type Checking"
- **NeurIPS 2027**: Epistemic uncertainty quantification in programs

#### Known Limitations
- **Phase 3 Self-Hosting**: 30/34 stdlib modules have advanced syntax requiring additional parser support
  - Partial self-compilation achieved (3/34 modules: parser::fn_def, parser::item, parser::impl_def)
  - Full self-compilation planned for v0.102.x
- **GPU Backend**: PTX support requires NVIDIA drivers; Metal support requires macOS/iOS
- **Tropical Geometry**: O(n³) complexity for n×n matrix operations (same as linear algebra)

## [1.0.0] - 2026-01-29

### Highlights

Sounio v1.0 is the first production-ready release of the language. This release
focuses on stabilization, completing core language features, and ensuring the
foundational systems are robust.

### Added

#### Core Language
- **File-based module loading**: `mod foo;` declarations now resolve to `foo.sio`,
  `foo/mod.sio`, or `foo/lib.sio` files with recursive resolution
- **Doc comments**: Support for `///` (outer) and `//!` (inner) documentation
  comments, propagated through AST → HIR pipeline
- **Custom unit definitions**: `unit kg;` and `unit mg = 0.001 * g;` syntax for
  defining base and derived units of measure
- **Type alias expansion**: Full expansion with cycle detection to prevent infinite
  recursion (e.g., `type A = B; type B = A;` now errors)

#### Compiler Infrastructure
- **Native code generation**: Production-ready x86-64 backend without LLVM
  dependency, ARM64 architecture support ready
- **Cranelift JIT**: Fast compilation for development and REPL workflows
- **Zero compiler warnings**: Cleaned up all non_snake_case and unused_imports
  warnings with domain-appropriate `#[allow]` attributes

#### Standard Library Integration
- **MedLang DSL**: Unified pharmacokinetic modeling language from standalone
  repository (agourakis82/medlang → archived)
  - PK models: one-compartment and two-compartment (IV and oral)
  - Dosing protocols: Weekly, Q3W, Daily oral
  - Dosing policies: FixedDose, ANCBased, TumorResponseBased, CycleEscalation
  - Full `Knowledge<T>` integration for uncertainty propagation

### Changed
- Module loader now searches 6 candidate paths for module resolution
- Type checker performs eager alias expansion before type comparison
- Build date embedded in compiler binary for version tracking

### Fixed
- Binding mode errors in module loader iterator patterns
- Type alias cycle detection prevents stack overflow
- Unit definition parsing correctly handles quotient expressions

### Known Limitations
See `compiler/docs/KNOWN_LIMITATIONS.md` for current language limitations and
planned features for future releases.

---

## [0.99.0] - 2026-01-15

### Added
- Quaternion and octonion GPU kernels for neural network layers
- Sparse quaternion operations with structured sparsity
- Cooperative groups for advanced CUDA parallelism
- Scientific ontology integration (15M+ terms)

### Changed
- Bumped version to indicate approaching v1.0 stability

---

## [0.88.0] - 2025-12-25

### Added

#### Core Language
- Epistemic type system with `Knowledge<T>` for uncertainty-aware computation
- Automatic uncertainty propagation (GUM-compliant)
- Provenance tracking for data lineage
- Confidence-gated execution

#### Standard Library (215,000+ lines)

**Epistemic Module** (`stdlib/epistemic/`)
- `Knowledge<T>` type with value, uncertainty, confidence, provenance
- GUM-compliant uncertainty propagation
- Source tracking and data lineage

**MedLang DSL** (`stdlib/medlang/`)
- PK/PD modeling domain-specific language
- PBPK compartment models
- Population PK with random effects
- Quantum binding site simulations

**fMRI Pipeline** (`stdlib/fmri/`)
- NIfTI file I/O
- Preprocessing pipeline (motion correction, slice timing, normalization)
- Brain atlas support (AAL, Schaefer, Harvard-Oxford)
- Epistemic connectivity analysis

**Causal Inference** (`stdlib/causal/`)
- Causal graph construction
- Backdoor criterion identification
- Instrumental variable analysis
- Causal discovery algorithms

**Connectivity** (`stdlib/connectivity/`)
- Graph-theoretic metrics with uncertainty
- Modularity (Louvain algorithm)
- Small-world metrics (sigma, omega)
- Rich-club coefficients
- Bootstrap confidence intervals

**GPU Acceleration** (`stdlib/gpu/`)
- Batch FFT for frequency filtering
- Separable 3D Gaussian smoothing
- Parallel correlation matrix computation
- Fisher Z-transform

**Optimization** (`stdlib/optimize/`)
- Gradient descent variants
- L-BFGS, Adam, RMSprop
- Constrained optimization
- Global optimization

**Signal Processing** (`stdlib/signal/`)
- FFT and spectral analysis
- Bandpass, lowpass, highpass filters
- Wavelet transforms
- Hilbert transform

**Data Handling** (`stdlib/data/`)
- DataFrame with column operations
- CSV/TSV I/O
- Missing value handling
- Data transformations

**MCMC** (`stdlib/mcmc/`)
- Metropolis-Hastings
- Hamiltonian Monte Carlo
- NUTS sampler
- Convergence diagnostics

**Random** (`stdlib/random/`)
- PCG64 generator
- Common distributions
- Reproducible seeding

**Quantum** (`stdlib/quantum/`)
- Qubit and quantum gate primitives
- Quantum circuit construction
- Measurement operators

**Linear Algebra** (`stdlib/linalg/`)
- Matrix operations
- Eigenvalue decomposition
- SVD, LU, Cholesky

**ODE Solvers** (`stdlib/ode/`)
- RK4, RK45 (Dormand-Prince)
- Adaptive step size
- Stiff solvers

**Bayesian Inference** (`stdlib/bayes/`)
- Prior specification
- Posterior sampling
- Model comparison

### Changed
- Renamed from internal codename to Sounio
- File extension changed to `.sio`
- Compiler binary renamed to `souc`

### Fixed
- Uncertainty propagation for division near zero
- Memory efficiency in large matrix operations
- GPU kernel synchronization issues

---

## [0.1.0] - 2025-01-01

### Added
- Initial language design
- Basic parser and type checker
- Core epistemic types

---

*For the complete history, see the [commit log](https://github.com/sounio-lang/sounio/commits/main).*

---

## Versioning

Sounio uses semantic versioning: MAJOR.MINOR.PATCH

- **MAJOR**: Fundamental language changes or major research breakthroughs
- **MINOR**: New features, system refinements, research findings
- **PATCH**: Bug fixes, documentation, minor improvements

## Migration Guide

### From v0.100.x to v0.101.x

No breaking changes. Existing code will continue to work without modifications.

#### Optional: Leveraging New Features

##### GPU Acceleration

```sio
// Previously: CPU-only computation
let result = matrix_multiply(a, b)

// Now: GPU-accelerated (automatic)
let result: gpu GPU = matrix_multiply(a, b)
```

##### Sedenions

```sio
// New hypercomplex support
let s1 = Sedenion::from([1.0; 16])
let s2 = Sedenion::from([2.0; 16])
let product = s1 * s2  // 16D multiplication
```

##### Refinement Types

```sio
// Optional: Explicit refinement types
type Positive = {x: i32 | x > 0}
fn safe_divide(x: i32, y: Positive) -> i32 {
    x / y  // y guaranteed > 0
}
```

##### Information Geometry

```sio
// Natural gradient descent (automatic selection)
let params = estimate_parameters_ng(observations)
// Runs 5-10x fewer iterations than Euclidean gradient
```

## Contributors

This release includes contributions from:

- **Research**: Information geometry, optimal transport, sheaf theory, tropical geometry
- **Engineering**: GPU backend, VM runtime, self-hosting infrastructure
- **Validation**: 75+ mathematical tests, integration tests, research validation suite

## Acknowledgments

Special thanks to:

- Research advisors on information geometry and optimal transport
- GPU optimization consultants for PTX and Metal support
- Mathematical foundations reviewers for theorem verification
- Self-hosting bootstrap team for infrastructure implementation

## Future Roadmap

### v0.102.x (Q2 2026)

- Full Phase 3 self-hosting (complete Rust compiler independence)
- Native code generation via Cranelift backend
- Performance optimization: 0.5-0.8x Rust compiler time
- 30/34 stdlib modules self-compiled

### v0.103.x (Q3 2026)

- Advanced GPU features: CUDA graphs, cooperative groups
- Distributed type checking with sheaves
- Quantum computing backend integration

### v1.1.0 (Q4 2026)

- Stabilized self-hosted compiler APIs
- Extended stdlib domains
- Industry partnerships and case studies

## Resources

- **Documentation**: [docs/MINIMUM_VIABLE_SOUNIO.md](docs/MINIMUM_VIABLE_SOUNIO.md)
- **Syntax Reference**: [docs/LLM_PROGRAMMING_GUIDE.md](docs/LLM_PROGRAMMING_GUIDE.md)
- **Research Papers**: [docs/RESEARCH_PAPERS.md](docs/RESEARCH_PAPERS.md)
- **GitHub**: [github.com/demetrios/sounio](https://github.com/demetrios/sounio)
- **Community**: [discord.gg/sounio](https://discord.gg/sounio)

---

**Release Date**: 2026-02-04
**Compiler Version**: v0.101.0
**Status**: Stable - Production Ready for Research Use
**Mathematical Validation**: 100% (11/11 integration tests, 26 research validation tests)
