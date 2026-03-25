<!-- docs:meta
topic_id: website.docs.examples
authority: dual
audience: users
last_validated: 2026-03-07
validated_by: A5
source_of_truth: docs/governance/topic-registry.v1.json#website.docs.examples
-->

# Sounio Examples

This directory contains comprehensive examples demonstrating Sounio's features.

## Quick Start

**First Steps:**
- [arithmetic.sio](arithmetic.sio) - Basic arithmetic operations
- [async_demo.sio](async_demo.sio) - Async/await basics
- [hello.sio](hello.sio) - Minimal runnable program

**Key Features:**
- [Epistemic Types](beta_epistemic.sio) - Uncertainty propagation
- [Epistemic BMI](epistemic_bmi.sio) - Uncertainty propagation demo
- [PBPK Simple](pbpk_simple.sio) - Minimal PK/PBPK-style workflow
- [GPU Hypercomplex](gpu_hypercomplex.sio) - Hypercomplex benchmark scaffold
- [Automatic Differentiation](autodiff/) - Gradient computation
- [GPU Computing](gpu/) - High-performance kernels

## Examples by Category

### Language Basics
- `arithmetic.sio` - Basic operations
- `collections/` - Data structures
- `algo/` - Algorithms

### Type System
- Epistemic types - `beta_epistemic.sio`
- Linear types - Check compiler tests
- Units of measure - `medlang/` examples

### Effects & Async
- `async_demo.sio` - Async programming
- Effect handlers - See `tests/run-pass/`

### Scientific Computing
- `autodiff/` - Automatic differentiation
- `ode/` - ODE solvers
- `linalg/` - Linear algebra
- `signal/` - Signal processing
- `stats/` - Statistics
- `bayes/` - Bayesian inference
- `monte_carlo/` - Monte Carlo methods
- `optimize/` - Optimization algorithms

### Domain-Specific
- `medlang/` - PK/PD models (pharmacokinetics/pharmacodynamics)
- `pbpk/` - Physiologically-based PK
- `fmri/` - fMRI neuroimaging analysis
- `darwin_atlas/` - Darwin Atlas integration
- `darwin_pbpk/` - Darwin + PBPK combination
- `causal/` - Causal inference
- `connectivity/` - Network analysis

### GPU & ML
- `gpu/` - GPU kernels
- `qnn/` - Quantized neural networks
- `nn/` - Neural network primitives
- `ml/` - Machine learning
- `fusion/` - Tensor operations
- `autodiff/` - Automatic differentiation

### Systems & I/O
- `io/` - File I/O operations
- `http/` - HTTP client/server
- `network/` - Network programming
- `csv/` - CSV processing
- `serialization/` - Data serialization
- `data_pipeline/` - Data processing

### Advanced Examples
- `alpha_sounio.sio` - Advanced language features
- `alphageozero_*.sio` - Complex AI examples
- `advanced_glm_optimization.sio` - ML-guided optimization
- `cross_compilation.sio` - Multi-target builds
- `build_system_demo.sio` - Build system features

### Tools & Integration
- `compiler/` - Compiler interaction
- `build/` - Build system examples
- `ontology/` - Scientific ontology queries
- `fractal/` - Complex programs (Mandelbrot)
- `graph/` - Graph algorithms

## Running Examples

```bash
# Set up the compiler binary
SOUC=./bin/souc
export SOUNIO_STDLIB_PATH=$(pwd)/stdlib

# Check syntax
$SOUC check examples/arithmetic.sio

# Compile to a temp ELF and execute
$SOUC run examples/async_demo.sio

# Compile to native ELF
$SOUC compile examples/autodiff/gradient.sio -o gradient.elf

# With GPU support (requires souc-linux-x86_64-gpu binary)
SOUC_GPU=./artifacts/omega/souc-bin/souc-linux-x86_64-gpu
$SOUC_GPU run examples/gpu/matrix_mul.sio
```

> **For curated scientific examples by research domain**, see [SCIENTIFIC_INDEX.md](SCIENTIFIC_INDEX.md).

## Contributing Examples

When adding new examples:
1. Place in appropriate category directory
2. Add clear comments explaining what's demonstrated
3. Keep examples focused on one concept
4. Include expected output as comments
5. Update this README

## See Also

- [Language Guide](../docs/guide/programming.md)
- [Standard Library](../stdlib/)
- [Tests](../tests/run-pass/) - Additional examples
