<div align="center">

# SOUNIO

### *Compute at the Horizon of Certainty*

[![License: MIT](https://img.shields.io/badge/License-MIT-gold.svg)](LICENSE)
[![stdlib](https://img.shields.io/badge/stdlib-151K%2B%20lines-blue.svg)](#standard-library)
[![Version](https://img.shields.io/badge/version-0.97.0-orange.svg)](compiler/CHANGELOG.md)

<img src="docs/assets/sounio-logo.svg" alt="Sounio Logo" width="200">

*A systems programming language for epistemic computing*

[Documentation](https://sounio-lang.org) · [Manifesto](MANIFESTO.md) · [Examples](#examples) · [Contributing](CONTRIBUTING.md)

</div>

---

## Recent Updates

### Version 0.97.0 Highlights

- **Quantum Computing**: UCCSD implementation for quantum chemistry with VQE optimization
- **REPL with Advanced Diagnostics**: Interactive shell with beautiful, context-aware error messages
- **Memory Management**: Smart pointers (Box, Rc, Arc) and custom allocators (Arena, Pool)
- **Machine Learning**: Tree-based algorithms (CART, Random Forest) with epistemic uncertainty
- **Compiler Infrastructure**: 64+ modules including LSP, distributed computing, and WASM support
- **A/B Register Allocation**: Compare classic vs. attention-based strategies with epistemic awareness
- **SPIR-V Binary Emission**: Full GPU shader generation support
- **Bayesian Optimization**: Fine-tune attention weights for register allocation
- **Monte Carlo Epistemic Kernels**: Confidence-aware parallel execution

See the [Changelog](CHANGELOG.md) for full details.

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

### Quantum Computing

```sio
import stdlib.quantum.*;

// UCCSD for quantum chemistry
let mut circuit = UCCSDCircuit::h2()  // H2 molecule
circuit.set_params(&[0.1, -0.05])     // Excitation amplitudes

// Run VQE optimization
let result = circuit.vqe_optimize(
    max_iterations: 100,
    convergence_threshold: 1e-6
)

print("Ground state energy: ", result.energy.mean, " ± ", result.energy.std)
print("Error from exact: ", result.error())
```

### Memory Management

```sio
import stdlib.mem.*;

// Smart pointers for ownership
let boxed = Box::new(42)              // Heap allocation
let shared = Rc::new("shared data")   // Reference counting
let thread_safe = Arc::new(vec![1, 2, 3])  // Atomic reference counting

// Custom allocators
let arena = Arena::new(1024)          // Bump allocator
let pool = Pool::new(size: 64, count: 100)  // Fixed-size pool

// Interior mutability
let cell = Cell::new(10)
cell.set(20)
let refcell = RefCell::new(vec![1, 2, 3])
refcell.borrow_mut().push(4)
```

### Machine Learning

```sio
import stdlib.ml.trees.*;

// Train decision tree with epistemic uncertainty
let tree = DecisionTree::new(max_depth: 5)
tree.fit(&X_train, &y_train)

// Random Forest with uncertainty quantification
let forest = RandomForest::new(
    n_trees: 100,
    max_depth: 10,
    min_samples_split: 5
)
forest.fit(&X_train, &y_train)

// Predictions with confidence intervals
let prediction = forest.predict(&X_test)
print("Prediction: ", prediction.value, " ± ", prediction.uncertainty)
print("Confidence: ", prediction.confidence)

// Feature importance
let importance = forest.feature_importance()
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
| `bayes/` | 1,500+ | Bayesian inference |
| `random/` | 1,599 | Random number generation |
| `quantum/` | 1,264+ | Quantum computing primitives, UCCSD, VQE |
| `mcmc/` | 1,203 | MCMC sampling |
| `linalg/` | 1,149 | Linear algebra |
| `ode/` | 966 | ODE solvers |
| `mem/` | 800+ | Smart pointers, custom allocators |
| `ml/trees/` | 600+ | Decision trees, Random Forest with epistemic uncertainty |
| `ontology/` | 500+ | Semantic SQL, versioning, knowledge representation |
| `distributed/` | 400+ | Distributed computing, cache, protocols |

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

# Start the interactive REPL
./target/release/souc repl
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

### Interactive REPL

```bash
$ souc repl
Sounio REPL v0.97.0
Type 'help' for commands, 'exit' to quit

>>> let x = 5.0 +- 0.1
>>> let y = 10.0 +- 0.2
>>> let z = x * y
>>> z
50.0 ± 2.24 (confidence: 0.95)

>>> import stdlib.quantum.*
>>> let circuit = UCCSDCircuit::h2()
>>> circuit.n_params()
2
```

The REPL features:
- **Beautiful error messages** with syntax highlighting
- **Automatic uncertainty propagation**
- **Import stdlib modules** on the fly
- **Multi-line input** support
- **History and completion** (coming soon)

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

### Completed ✅

- [x] Core epistemic type system
- [x] MedLang PK/PD DSL
- [x] fMRI preprocessing pipeline
- [x] GPU acceleration (CUDA, Metal, WebGPU)
- [x] Causal inference
- [x] SPIR-V backend
- [x] Bayesian optimization tooling
- [x] Interactive REPL with advanced diagnostics
- [x] Quantum computing (UCCSD, VQE)
- [x] Memory management (smart pointers, allocators)
- [x] Machine learning (decision trees, random forests)
- [x] A/B register allocation with epistemic awareness
- [x] Monte Carlo epistemic kernels

### In Progress 🚧

- [ ] Language Server Protocol (LSP) - infrastructure ready, feature-gated
- [ ] LLVM backend - partial implementation
- [ ] Distributed computing - core modules ready, feature-gated
- [ ] WASM compilation - feature-gated support

### Planned 📋

- [ ] Package manager (`siopkg`)
- [ ] Jupyter kernel integration
- [ ] VSCode extension enhancements
- [ ] Online playground
- [ ] Standard library documentation generator
- [ ] Formal verification tools
- [ ] Automatic differentiation for all stdlib functions

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
# Run tests
cargo test
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
