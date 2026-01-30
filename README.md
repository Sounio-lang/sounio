<div align="center">

# SOUNIO

### *Compute at the Horizon of Certainty*

[![License: MIT](https://img.shields.io/badge/License-MIT-gold.svg)](LICENSE)
[![stdlib](https://img.shields.io/badge/stdlib-151K%2B%20lines-blue.svg)](#standard-library)
[![Version](https://img.shields.io/badge/version-0.99.0-orange.svg)](CHANGELOG.md)

<img src="docs/assets/sounio-logo.svg" alt="Sounio Logo" width="200">

*A systems programming language for epistemic computing*

[Getting Started](docs/guide/getting-started.md) · [Tutorial](docs/guide/tutorial.md) · [Manifesto](MANIFESTO.md) · [Examples](examples/) · [API Reference](docs/reference/STDLIB_REFERENCE.md) · [Contributing](CONTRIBUTING.md)

</div>

---

## Table of Contents

- [The Metaphor](#the-metaphor)
- [Why Sounio?](#why-sounio)
- [Features](#features)
- [Standard Library](#standard-library)
- [Quick Start](#quick-start)
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

**151,000+ lines** of production-ready scientific computing:

| Module | Lines | Description |
|--------|-------|-------------|
| `epistemic/` | 7,780 | Core uncertainty types, propagation, provenance |
| `medlang/` | 9,800 | PK/PD DSL with PBPK and quantum binding |
| `fmri/` | 5,073 | Neuroimaging pipeline with atlas support |
| `causal/` | 3,773 | Causal inference and discovery |
| `connectivity/` | 3,792 | Graph metrics, network analysis |
| `optimize/` | 3,766 | Optimization algorithms |
| `signal/` | 3,068 | Signal processing, spectral analysis |
| `gpu/` | 2,487 | GPU kernels (FFT, smoothing, statistics) |
| `data/` | 2,576 | DataFrames and data manipulation |
| `mcmc/` | 1,203 | MCMC sampling |
| `random/` | 1,599 | Random number generation |
| `quantum/` | 1,264 | Quantum computing primitives |
| `linalg/` | 1,149 | Linear algebra |
| `ode/` | 966 | ODE solvers |
| `bayes/` | 1,500+ | Bayesian inference |

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/sounio-lang/sounio.git
cd sounio

# Build the compiler (requires Rust 1.80+ with edition 2024)
cargo build -p souc --release

# Run your first Sounio program
cargo run -p souc -- run examples/hello.sio

# Or use the built binary directly
./target/release/souc run examples/hello.sio
```

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
cd compiler && cargo install --path .
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
- **[Tutorial](docs/guide/tutorial.md)** - Step-by-step guide to Sounio
- **[Language Guide](docs/LLM_PROGRAMMING_GUIDE.md)** - Complete language reference
- **[API Reference](docs/reference/STDLIB_REFERENCE.md)** - Standard library documentation
- **[FAQ](docs/FAQ.md)** - Frequently asked questions
- **[Glossary](docs/GLOSSARY.md)** - Epistemic computing terminology

### Examples
- **[Basic Examples](examples/)** - Hello world, syntax basics
- **[Scientific Computing](examples/scientific/)** - ODE solvers, signal processing
- **[Medical Applications](examples/medical/)** - PK/PD models, PBPK
- **[GPU Computing](examples/gpu/)** - High-performance kernels
- **[Advanced Examples](examples/advanced/)** - Complex applications

### For Contributors
- **[Architecture Overview](docs/compiler/ARCHITECTURE.md)** - Compiler design
- **[Contributing Guide](CONTRIBUTING.md)** - How to contribute
- **[Style Guide](docs/STYLE_GUIDE.md)** - Code style conventions

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

**Current Version**: 0.99.0 (Pre-1.0)

### Implemented ✅
- Core epistemic type system with `Knowledge<T>`
- Uncertainty propagation (GUM-compliant)
- MedLang PK/PD DSL
- fMRI neuroimaging pipeline
- GPU acceleration (CUDA, Metal)
- Causal inference framework
- 151K+ lines standard library
- Native code generation (ELF/Mach-O)
- Cranelift JIT backend
- REPL (interactive mode)

### In Progress 🚧
- Language Server Protocol (LSP) - 80% complete
- LLVM backend - Experimental
- Package manager (`siopkg`) - Design phase
- SMT-based refinement types - Proof of concept
- WebAssembly target - Experimental

### Planned 📋
- Stabilized 1.0 API
- Distributed compilation
- Formal verification support
- Additional scientific domains (materials, climate)

See [CHANGELOG.md](CHANGELOG.md) for detailed version history.

---

## Roadmap

- [x] Core epistemic type system
- [x] MedLang PK/PD DSL
- [x] fMRI preprocessing pipeline
- [x] GPU acceleration
- [x] Causal inference
- [ ] Language Server Protocol (LSP)
- [ ] LLVM backend
- [ ] Package manager (`siopkg`)
- [ ] Interactive REPL

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

If you use Sounio in academic work, please cite:

```bibtex
@software{sounio2025,
  title = {Sounio: A Systems Language for Epistemic Computing},
  author = {Agourakis, Demetrios Chiuratto},
  year = {2025},
  url = {https://github.com/sounio-lang/sounio}
}
```

---

## License

MIT License. See [LICENSE](LICENSE).

---

<div align="center">

*At the horizon of certainty, where ancient columns meet the endless sea.*

**🏛️ SOUNIO 🌊**

</div>
