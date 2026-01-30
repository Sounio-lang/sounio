# Changelog

All notable changes to Sounio will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

#### Standard Library (151,000+ lines)

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
