# Changelog

All notable changes to Sounio will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **MedLang Integration**: MedLang unified into Sounio as `stdlib/medlang/`
  - PK models: one-compartment and two-compartment models (IV and oral)
  - Dosing protocols: Weekly, Q3W, Daily oral protocols
  - Dosing policies: FixedDose, ANCBased, TumorResponseBased, CycleEscalation, TimeBasedReduction
  - All models use `Knowledge<T>` for automatic uncertainty propagation
  - Migrated from MedLang standalone repository (agourakis82/medlang → archived)

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
