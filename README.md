<div align="center">

# SOUNIO

### *Compute at the Horizon of Certainty*

[![License: MIT](https://img.shields.io/badge/License-MIT-gold.svg)](LICENSE)
[![stdlib](https://img.shields.io/badge/stdlib-151K%2B%20lines-blue.svg)](#standard-library)
[![Version](https://img.shields.io/badge/version-0.88.0-orange.svg)](CHANGELOG.md)

<img src="docs/assets/sounio-logo.svg" alt="Sounio Logo" width="200">

*A systems programming language for epistemic computing*

[Documentation](https://sounio-lang.org) · [Manifesto](MANIFESTO.md) · [Examples](#examples) · [Contributing](CONTRIBUTING.md)

</div>

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
model PopPKModel {
    param CL ~ LogNormal(10.0 L/h, omega: 0.30)
    param V  ~ LogNormal(100.0 L, omega: 0.25)
    
    compartment Central {
        volume: V
        elimination: CL
    }
    
    dosing IV {
        into: Central
        amount: 500 mg
    }
    
    observe Cp = Central.concentration
}
```

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

# Build the compiler (requires Rust 1.70+)
cd compiler && cargo build --release

# Run your first Sounio program
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

---

## Design Principles

1. **Uncertainty is not optional** — Every scientific value has uncertainty. Ignoring it is a bug.

2. **Provenance matters** — Data without origin is data without trust.

3. **Propagation is automatic** — Manual uncertainty calculation is error-prone. The compiler handles it.

4. **Confidence gates execution** — Low-confidence paths should require explicit acknowledgment.

5. **Standards compliance** — GUM (Guide to Uncertainty in Measurement), ISO 17025.

See [MANIFESTO.md](MANIFESTO.md) for the complete philosophy.

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
cargo test

# Check formatting
cargo fmt --check

# Run clippy
cargo clippy
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
