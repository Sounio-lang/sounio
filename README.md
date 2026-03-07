<!-- docs:meta
topic_id: repo.frontdoor.readme
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.frontdoor.readme
-->

<div align="center">

# SOUNIO

### *Compute at the Horizon of Certainty*

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-gold.svg)](LICENSE)
[![stdlib](https://img.shields.io/badge/stdlib-215K%2B%20lines-blue.svg)](#standard-library)
[![Version](https://img.shields.io/badge/version-v0.2.0-orange.svg)](CHANGELOG.md)
[![Preprint: TechRxiv](https://img.shields.io/badge/Preprint-TechRxiv%20%28DOI%20pending%29-teal.svg)](https://www.techrxiv.org/)

<img src="docs/assets/sounio-logo.svg" alt="Sounio Logo" width="200">

*A systems programming language for epistemic computing*

[Getting Started](docs/guide/getting-started.md) · [Tutorial](docs/guide/tutorial.md) · [Manifesto](MANIFESTO.md) · [Examples](examples/) · [API Reference](docs/reference/STDLIB_REFERENCE.md) · [Contributing](CONTRIBUTING.md)

</div>

---

## What's New in v0.2.0

- **Package Manager**: Full dependency management with `sounio-pkg`
- **Security**: Depth limits and complexity budgets prevent DoS
- **Bug Fixes**: Borrow soundness, bounds checking, shift validation
- **Epistemic Types**: Knowledge<T,ε> with uncertainty propagation

---

## Table of Contents

- [The Metaphor](#the-metaphor)
- [Why Sounio?](#why-sounio)
- [Features](#features)
- [Standard Library](#standard-library)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Package Manager](#package-manager)
- [Learning Resources](#learning-resources)
- [Design Principles](#design-principles)
- [Project Status](#project-status)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

---

## The Metaphor

> *"Place me on Sunium's marbled steep,*  
> *Where nothing, save the waves and I,*  
> *May hear our mutual murmurs sweep..."*  
> — Lord Byron, *Don Juan* (1819)

**Cape Sounion** (Σούνιο) stands at the southernmost tip of Attica, where the ancient Temple of Poseidon has watched over the Aegean for 2,500 years. At sunset, its Doric columns catch the last light—the horizon where certainty meets the unknown sea.

**Sounio** the language embodies this metaphor: computing at the boundary between what we know and what we don't. Every value carries not just data, but *knowledge of its own uncertainty*. Like the ancient Greeks who built temples to navigate by, we build programs that navigate uncertainty with precision.

The columns stand firm. The sea is unpredictable. Sounio helps you reason about both.

---

## Why Sounio?

Modern scientific computing demands more than correct arithmetic—it demands *epistemic integrity*. When a simulation predicts a drug's concentration, when an fMRI analysis identifies neural correlations, when a model infers causality: **how confident should we be?**

Most languages treat uncertainty as an afterthought. Sounio makes it foundational.

```sio
// Every measurement knows its uncertainty
let dose: Knowledge<mg> = measure(500.0, uncertainty: 2.5, source: "clinical_trial_2024")

// Uncertainty propagates automatically through computation
let concentration = dose / volume  // GUM-compliant propagation

// Confidence gates control execution
if concentration.confidence > 0.95 {
    administer(concentration)
} else {
    require_confirmation(concentration)
}
```

### Before / After: March 2020 COVID Kernel

**Before (implicit assumptions):**

```sio
let r_t = estimate_r_t(march_2020_cases)
if r_t > 1.0 {
    trigger_lockdown()
}
```

**After (typed refusal in Sounio):**

```sio
fn authorize_lockdown(signal: Knowledge[f64, ε >= 0.95]) -> Knowledge[f64, ε >= 0.95] {
    signal
}

let march_signal: Knowledge[f64, ε=⊥] = Knowledge { value: 1.4 }
let _decision = authorize_lockdown(march_signal) // compile-time refusal
```

Shipped fixtures:
- `tests/run-pass/covid_2020_kernel.sio` (`//@ check-only`)
- `tests/compile-fail/covid_2020_knightian_refusal.sio`
- `tests/compile-fail/covid_2020_temporal_expiration.sio`

Validation commands:

```bash
cargo run -q --bin souc -- check tests/run-pass/covid_2020_kernel.sio --error-format=json
cargo run -q --bin souc -- check tests/compile-fail/covid_2020_knightian_refusal.sio --error-format=json
cargo run -q --bin souc -- check tests/compile-fail/covid_2020_temporal_expiration.sio --error-format=json
```

Expected refusal diagnostics include:
- `Knightian uncertainty (ε=⊥) cannot satisfy required confidence`
- `Temporal validity window`

---

## Features

### Epistemic Type System

```sio
// Knowledge<T> wraps any value with epistemic metadata
struct Knowledge<T> {
    value: T,
    uncertainty: f64,
    confidence: f64,
    provenance: Source,
}

// Automatic uncertainty propagation (GUM-compliant)
let x = Knowledge::new(10.0, uncertainty: 0.5)
let y = Knowledge::new(20.0, uncertainty: 0.3)
let z = x + y  // z.uncertainty = sqrt(0.5² + 0.3²) = 0.583
```

### MedLang DSL for PK/PD Modeling

```sio
import stdlib.medlang::pk::one_compartment::*

model PopPKModel {
    param CL: Knowledge<L/h> ~ LogNormal(mean: 10.0 L/h, omega: 0.30)
    param V: Knowledge<L> ~ LogNormal(mean: 100.0 L, omega: 0.25)
    
    compartment Central {
        volume: V
    }
    
    flow Central -> Elimination {
        rate: CL
    }
    
    dose IV {
        into: Central
    }
    
    observe Cp: Concentration = Central.concentration
}
```

**MedLang** is now part of Sounio's standard library (`stdlib/medlang/`), providing:
- PK models (one/two compartment, oral/IV)
- Dosing protocols (weekly, Q3W, daily)
- Dosing policies (ANC-based, tumor response-based)
- All with native `Knowledge<T>` uncertainty propagation

### GPU-Accelerated Neuroimaging

```sio
import stdlib.fmri.*;
import stdlib.gpu.*;
import stdlib.connectivity.*;

// Process fMRI with epistemic connectivity
let atlas = atlas_schaefer400()
let conn = bootstrap_connectivity(&timeseries, n_bootstrap: 1000)

// Every correlation carries uncertainty
for region in atlas.regions {
    if conn.get(region).confidence > 0.95 {
        print("Significant: ", region.label)
    }
}
```

### Causal Inference

```sio
import stdlib.causal.*;

// Build causal graph
let graph = CausalGraph::new()
graph.add_edge("Treatment", "Outcome")
graph.add_edge("Confounder", "Treatment")
graph.add_edge("Confounder", "Outcome")

// Identify causal effect with backdoor criterion
let effect = graph.identify_effect("Treatment", "Outcome")
print("ATE: ", effect.value, " ± ", effect.uncertainty)
```

---

## Standard Library

**215,000+ lines** of production-ready scientific computing (512 files, 76 modules):

| Module | Lines | Description |
|--------|------:|-------------|
| `epistemic/` | 31,962 | Core uncertainty types, GUM propagation, provenance |
| `stdlib/compiler/` | 17,923 | Self-hosted compiler (lexer, parser, checker, codegen) |
| `nn/` | 13,693 | Neural networks, autograd, backpropagation |
| `medlang/` | 8,147 | PK/PD domain-specific language |
| `async/` | 7,435 | Async runtime, channels, streams, executors |
| `darwin_pbpk/` | 6,036 | PBPK modeling with DARWIN integration |
| `collections/` | 5,355 | BTree, HashMap, Trie, Deque, BitSet |
| `genomics/` | 4,925 | Genomics, FASTA/VCF I/O, octonion GRN |
| `linalg/` | 4,825 | Linear algebra, BLAS fallback |
| `ml/` | 4,315 | Tensors, autodiff, Gaussian processes |
| `qnn/` | 3,829 | Quaternion neural networks with MNIST |
| `ontology/` | 3,691 | Biomedical ontologies (SNOMED, GO, HPO, LOINC) |
| `stats/` | 3,444 | Statistical testing, distributions |
| `ode/` | 3,340 | ODE solvers (RK4, Tsit5, PBPK) |
| `causal/` | 3,139 | Causal inference and discovery |
| `geometry/` | 3,586 | Computational geometry |
| `fractal/` | 2,872 | KEC framework, box counting, GPU-accelerated |
| `data/` | 2,874 | DataFrames, CSV, I/O |
| `optimize/` | 2,600 | BFGS, Nelder-Mead, Levenberg-Marquardt |
| `medical/` | 2,503 | Sedenion EEG, PBPK, hyperspectral imaging |
| `gpu/` | 2,386 | GPU kernels (FFT, smoothing, statistics) |
| `signal/` | 2,386 | Signal processing, spectral analysis |
| `fmri/` | 2,356 | Neuroimaging pipeline with atlas support |
| `onn/` | 2,015 | Octonion neural networks |
| `connectivity/` | 1,552 | Network metrics, phase analysis |
| `quantum/` | 1,392 | Quantum computing primitives (VQE) |
| `bayes/` | 1,371 | Bayesian inference, MCMC, VI |
| `random/` | 1,068 | Random number generation |

### Fail-Closed Science + Hyper + GPU Gates

The main gate path now requires executable scientific and hypercomplex lanes:
- `tests/stdlib/fmri/test_pipeline_real_e2e.sio`
- `tests/stdlib/darwin_pbpk/test_pipeline_real_e2e.sio`
- `tests/stdlib/nn/test_hyper_quaternion_e2e.sio`
- `tests/stdlib/onn/test_hyper_onn_e2e.sio`
- `tests/stdlib/qnn/test_hyper_qnn_e2e.sio`
- `tests/stdlib/snn/test_snn_e2e.sio`
- `tests/stdlib/math/test_hyper_math_e2e.sio`

fMRI + PBPK lane details:
- real fixture-driven NIfTI-1/NIfTI-2 parse/load execution (not synthetic-only checks)
- QC + FC + atlas coverage + uncertainty metrics are emitted and compared against pinned golden values
- runtime regressions are tracked in gate JSON under `runtime_regressions` using committed probes in `tests/stdlib/runtime_regression/`
- runtime provenance is recorded under `runtime_provenance` (`souc_bin`, `souc_version`, `pinned_version_expected`)

Hyper lane details:
- strict run-pass execution with pinned `tests/fixtures/hyper/pipeline_golden.v1.json`
- no `//@ ignore` allowed in required hyper tests
- hyper inventory is emitted from `scripts/scan_stdlib.sh` as `hyper_active_files`, `hyper_disabled_files`, `hyper_stub_mod_files`

Runtime policy (science runtime regressions):
- local default remains telemetry mode (`soft`)
- required CI full gate enforces strict runtime regression mode (`STDLIB_RUNTIME_REGRESSION_STRICT=1`)
- strict mode is fail-closed; runtime probes must pass to keep CI green

GPU runtime policy:
- backend compile smoke remains in `scripts/e2e_gate.sh`
- multi-target codegen parity gate is integrated via `scripts/omega/omega_gpu_codegen_parity_gate.sh`
- binary hash-chain attestation gate is integrated via `scripts/omega/omega_gpu_binary_attest_gate.sh`
- remote-attested GPU runtime gate is integrated via `scripts/omega/omega_gpu_runtime_attest_gate.sh`
- GPU build CLI now accepts `--gpu-target`, `--gpu-binary-format`, and `--gpu-strict-parity`
- canonical pinned `souc` version is sourced from `scripts/omega/omega_resolve_souc_bin.sh` (or `SOUNIO_SOUC_VERSION` override)
- local default mode is `OMEGA_GPU_RUNTIME_GATE_MODE=auto` (non-fatal `not_run` when remote runner is unavailable)
- required CI mode is `OMEGA_GPU_RUNTIME_GATE_MODE=required` (any non-pass is fail-closed)
- blocker taxonomy includes `target_unavailable`, `isa_encode_unsupported`, `binary_pack_fail`, `driver_reject`, `parity_fail`, `perf_regression`, `attestation_invalid`, `ssh_unreachable`, `remote_env_missing`, `pinned_version_mismatch`, `gpu_backend_unavailable`, `runtime_test_fail`

Run from repository root:

```bash
OMEGA_GPU_CODEGEN_PARITY_MODE=required bash scripts/omega/omega_gpu_codegen_parity_gate.sh
OMEGA_GPU_BINARY_ATTEST_MODE=required bash scripts/omega/omega_gpu_binary_attest_gate.sh
OMEGA_GPU_RUNTIME_GATE_MODE=required bash scripts/omega/omega_gpu_runtime_attest_gate.sh
bash scripts/stdlib_hyper_execution_gate.sh
STDLIB_RUNTIME_REGRESSION_STRICT=1 bash scripts/stdlib_science_pipeline_gate.sh
STDLIB_RUNTIME_REGRESSION_STRICT=1 bash scripts/stdlib_reliability_gate.sh
bash scripts/omega/omega_gpu_codegen_parity_gate.sh
bash scripts/omega/omega_gpu_binary_attest_gate.sh
bash scripts/omega/omega_gpu_runtime_attest_gate.sh
bash scripts/stdlib_hyper_execution_gate.sh
bash scripts/stdlib_science_pipeline_gate.sh
bash scripts/stdlib_reliability_gate.sh
```

Current machine-checkable artifacts:
- `artifacts/omega/gpu_codegen_parity.v1.json`
- `artifacts/omega/gpu_binary_attestation.v1.json`
- `artifacts/omega/gpu_runtime_attest_gate.v1.json`
- `artifacts/stdlib/stdlib_hyper_execution_status.v1.json`
- `artifacts/stdlib/stdlib_science_pipeline_status.v1.json`
- `artifacts/stdlib/stdlib_reliability_status.v1.json`
- `tests/fixtures/fmri/fixture_manifest.v1.json`
- `tests/fixtures/fmri/pipeline_golden.v1.json`

---

## Installation

```bash
# Install package manager
curl -sSL https://souniolang.org/install.sh | bash

# Or build from source
git clone https://github.com/sounio-lang/sounio
cd sounio && ./build.sh
```

---

## Quick Start

```bash
# Create project
mkdir myproject && cd myproject
sounio-pkg init
sounio-pkg add stdlib

# Write code
cat > main.sio << 'EOF'
fn main() -> i32 with IO {
    print("Hello, Sounio!\n")
    return 0
}
EOF

# Run
sounio run main.sio
```

## Package Manager

Sounio includes a full-featured package manager:

```bash
sounio-pkg add name@version          # Registry dependency
sounio-pkg add name --git <url>      # Git dependency
sounio-pkg add name --path <path>    # Local path
sounio-pkg remove name               # Remove dependency
sounio-pkg update                    # Update lockfile
```

---

### Hello, Uncertainty

```sio
// hello.sio
fn main() -> i32 {
    let measurement = Knowledge::new(
        value: 42.0,
        uncertainty: 0.5,
        confidence: 0.95,
        source: "laboratory"
    )
    
    print("Value: ", measurement.value, " ± ", measurement.uncertainty)
    print("Confidence: ", measurement.confidence * 100.0, "%")
    
    0
}
```

### Environment Configuration

#### Stdlib Path Configuration

Sounio needs to locate the standard library (`stdlib/`) to resolve module imports. The compiler searches in this priority order:

1. **`SOUNIO_STDLIB_PATH`** environment variable (highest priority)
2. **`SOUNIO_STDLIB`** environment variable (legacy, but still supported)
3. Relative to compiler binary: `<exe_dir>/../stdlib/`
4. User home directory: `~/.sounio/stdlib/`
5. System paths: `/usr/local/lib/sounio/stdlib`, `/usr/share/sounio/stdlib`, etc.

#### Running Programs from Any Directory

To run Sounio programs that use stdlib modules from outside the repository:

```bash
# Option 1: Set environment variable (recommended for development)
export SOUNIO_STDLIB_PATH=/path/to/sounio/stdlib
souc run /path/to/your/program.sio

# Option 2: System-wide installation
cargo install --path crates/souc
# Stdlib will be found relative to installed binary

# Option 3: Copy to user home directory
mkdir -p ~/.sounio
cp -r stdlib ~/.sounio/
```

#### Diagnostic Commands

Check where the compiler is looking for stdlib:

```bash
souc sysroot stdlib-paths
```

This displays all search locations and which ones exist, helping you verify correct configuration.

---

## Learning Resources

### Documentation
- **[Documentation Site](https://sounio-lang.org/docs/)** - Hosted docs portal
- **[Tutorial](docs/guide/tutorial.md)** - Step-by-step guide to Sounio
- **[Language Guide](docs/guide/programming.md)** - Complete language reference
- **[API Reference](docs/reference/STDLIB_REFERENCE.md)** - Standard library documentation
- **[FAQ](docs/FAQ.md)** - Frequently asked questions
- **[Glossary](docs/GLOSSARY.md)** - Epistemic computing terminology

### Examples
- **[Basic Examples](examples/)** - Hello world, syntax basics
- **[Epistemic Examples](examples/epistemic/)** - Uncertainty and provenance demos
- **[PBPK Examples](examples/pbpk/)** - PK/PD and PBPK workflows
- **[GPU Computing](examples/gpu/)** - High-performance kernels
- **[fMRI Examples](examples/fmri/)** - Neuroimaging workflows

### For Contributors
- **[Architecture Overview](docs/compiler/ARCHITECTURE.md)** - Compiler design
- **[Contributing Guide](CONTRIBUTING.md)** - How to contribute
- **[Style Guide](docs/contributor-guide/STYLE_GUIDE.md)** - Code style conventions

---

## Design Principles

1. **Uncertainty is not optional** — Every scientific value has uncertainty. Ignoring it is a bug.

2. **Provenance matters** — Data without origin is data without trust.

3. **Propagation is automatic** — Manual uncertainty calculation is error-prone. The compiler handles it.

4. **Confidence gates execution** — Low-confidence paths should require explicit acknowledgment.

5. **Standards compliance** — GUM (Guide to Uncertainty in Measurement), ISO 17025.

See [MANIFESTO.md](MANIFESTO.md) for the complete philosophy.

---

## Project Status

**Current Version**: v0.2.0

### Implemented ✅
- Core epistemic type system with `Knowledge<T>` (25K+ lines in compiler, 31K+ in stdlib)
- Uncertainty propagation (GUM-compliant)
- MedLang PK/PD DSL (8,147 lines)
- fMRI neuroimaging pipeline (2,356 lines)
- GPU acceleration (82,537 lines total subsystem)
- Causal inference framework (3,139 in stdlib + 9,538 in compiler)
- 215K+ lines standard library
- Native code generation (ELF/Mach-O)
- Cranelift JIT backend
- Language Server Protocol (13,752 lines)
- REPL (2,286 lines)
- Package manager `sounio-pkg` (5,740 lines)

### In Progress 🚧
- LLVM backend (864 lines — partial)
- SMT-based refinement types
- WebAssembly target
- Self-hosted compiler (24,428 lines)

### Planned 📋
- Stabilized 1.0 API
- Distributed compilation
- Formal verification support
- Additional scientific domains (materials, climate)

See [CHANGELOG.md](CHANGELOG.md) for detailed version history.

---

## Rustless Cutover — Self-Hosted Compiler

**Status**: ✅ COMPLETE (2026-02-13)

Sounio now compiles itself using a 3-stage bootstrap process, removing Rust from the critical compilation path:

```
Stage 0 (C VM) → Stage 1 (Self-hosted) → Stage 2 (Verified) → ✅ Reproducible!
```

**Key Achievements**:
- **29,539 LOC** delivered (24,428 self-hosted + 3,184 C VM + 1,247 SOIR library)
- **139+ tests** passing (100% success rate)
- **SOIR v1 format** for deterministic IR serialization
- **Poseidon VM** (portable C99 bootstrap VM)
- **Cross-platform** (Linux, macOS, Windows, BSDs)

**Quick Start**:
```bash
# Compile to inspectable IR
cargo run -p souc -- compile examples/hello.sio --output hello.soir

# Inspect the IR
cargo run -p soir -- inspect hello.soir

# Execute on portable VM
./bootstrap/poseidon/poseidon hello.soir
```

**Documentation**:
- [Rustless Cutover Guide](docs/implementation/RUSTLESS_CUTOVER.md) - Complete workflow
- [SOIR Format Specification](docs/architecture/SOIR_REFERENCE.md) - Binary format reference
- [Migration Guide](docs/implementation/MIGRATION_GUIDE.md) - User and developer migration
- [Complete Implementation](docs/implementation/RUSTLESS_COMPLETE.md) - Full technical details

**Benefits**:
- ✅ **Reproducible builds** (Stage 1 ≡ Stage 2 verified)
- ✅ **Platform independence** (C VM runs anywhere)
- ✅ **Self-hosting** (compiler written in Sounio)
- ✅ **Trusting Trust mitigation** (3-stage bootstrap)

---

## Roadmap

- [x] Core epistemic type system (25K+ compiler, 31K+ stdlib)
- [x] MedLang PK/PD DSL (8,147 lines)
- [x] fMRI preprocessing pipeline (2,356 lines)
- [x] GPU acceleration (82,537 lines total subsystem)
- [x] Causal inference (3,139 in stdlib + 9,538 in compiler)
- [x] Language Server Protocol (13,752 lines)
- [x] Interactive REPL (2,286 lines)
- [x] Package manager `sounio-pkg` (5,740 lines)
- [x] **Self-hosted compiler** (24,428 lines) **← NEW!**
- [x] **Rustless bootstrap** (3-stage verification) **← NEW!**
- [ ] LLVM backend (864 lines — partial)

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
# Run tests
cargo test --workspace

# Check formatting
cargo fmt --all --check

# Run clippy
cargo clippy --workspace

# Run fast gate (full check)
./scripts/fast_gate.sh

# Build with all features
cargo build --workspace --all-features
```

---

## Citation

If you use Sounio in your research, please cite:

```bibtex
@article{sounio2026,
  title={Sounio: A Self-Hosted Systems Language for Verifiable Scientific Computing},
  author={Sounio Team},
  journal={arXiv preprint},
  year={2026}
}
```

---

## License

Apache License 2.0. See [LICENSE](LICENSE).

---

<div align="center">

*At the horizon of certainty, where ancient columns meet the endless sea.*

**🏛️ SOUNIO 🌊**

</div>
