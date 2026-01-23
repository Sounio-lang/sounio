# Sounio Standard Library Examples

This directory contains demonstrations of Sounio standard library modules. Each example shows how to use a particular stdlib feature in practice.

## Organization

Examples mirror the stdlib structure for easy discovery:

```
examples/
├── causal/              # Causal inference examples
├── graph/               # Graph algorithms and analysis
├── nn/                  # Neural networks (including quaternionic networks)
├── ode/                 # ODE solvers and scientific modeling
├── epistemic/           # Epistemic computing and uncertainty
├── medlang/             # MedLang PK/PD domain-specific language
├── autodiff/            # Automatic differentiation
├── bayes/               # Bayesian inference (MCMC, VI)
├── fmri/                # fMRI preprocessing and analysis
├── gpu/                 # GPU-accelerated computations
├── stats/               # Statistical analysis
├── ontology/            # Scientific ontology queries
├── data/                # Data manipulation and I/O
└── ...                  # Other stdlib domains
```

## Running Examples

```bash
# From the repo root:
cd /path/to/sounio
export SOUNIO_STDLIB_PATH=$(pwd)/stdlib

# Run an example
./compiler/target/release/souc run examples/graph/curvature_demo.sio

# Type-check an example
./compiler/target/release/souc check examples/causal/core_demo.sio

# See AST and types for debugging
./compiler/target/release/souc check examples/nn/mlp_xor_demo.sio --show-ast --show-types
```

## Finding Examples

If you want to learn about a particular stdlib module, its example is easy to find:

- Module: `stdlib/graph/curvature.sio` → Example: `examples/graph/curvature_demo.sio`
- Module: `stdlib/causal/core.sio` → Example: `examples/causal/core_demo.sio`
- Module: `stdlib/nn/mlp_xor.sio` → Example: `examples/nn/mlp_xor_demo.sio`

## Using Examples in Your Programs

Examples demonstrate key functionality. You can learn from them and adapt them for your own work:

```sio
// myprogram.sio
use std::graph::curvature::*

fn main() {
    // Learn from examples/graph/curvature_demo.sio
    // Adapt the code for your use case
}
```

## Composing Multiple stdlib Modules

**This is now possible!** The main reason these examples were extracted was to remove `main()` conflicts. You can now compose multiple stdlib modules in your programs:

```sio
use std::graph::curvature::*
use std::causal::core::*
use std::stats::effect_sizes::*

fn main() -> i32 {
    // Use functions from all three modules!
    print("Multi-module composition works!\n")
    0
}
```

## Building Examples

All examples are standalone and can be compiled/run independently:

```bash
souc check examples/epistemic/gum_demo.sio
souc run examples/bayes/mcmc_demo.sio
souc run examples/ode/pbpk_minimal_demo.sio
```

## Contributing Examples

When adding new stdlib features, please add a corresponding `*_demo.sio` file to demonstrate the functionality. Follow these guidelines:

1. **Name**: Use the same module name + `_demo.sio`
2. **Location**: Mirror the stdlib structure (e.g., `stdlib/newdomain/newmodule.sio` → `examples/newdomain/newmodule_demo.sio`)
3. **Content**: Show realistic usage with clear output
4. **Documentation**: Add comments explaining key concepts

## Related

- **Stdlib**: [../../stdlib/README.md](../stdlib/README.md)
- **Tests**: [../tests/stdlib/README.md](../tests/stdlib/README.md)
- **Documentation**: [../../docs/LLM_PROGRAMMING_GUIDE.md](../docs/LLM_PROGRAMMING_GUIDE.md)
