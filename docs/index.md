---
title: Sounio Documentation
description: Complete documentation for the Sounio programming language
---

# Sounio Documentation

Welcome to the complete documentation for **Sounio**, a systems programming language for epistemic computing where every value carries knowledge of its own uncertainty.

## Quick Links

| I want to... | Go to... |
|--------------|----------|
| Get started quickly | [Getting Started](getting-started/index.md) |
| Learn the language | [Language Reference](language/syntax-reference.md) |
| Understand epistemic types | [Epistemic Computing](epistemic/index.md) |
| Do pharmacokinetics | [PK/PD Guide](domains/pharmacokinetics/index.md) |
| Browse the stdlib | [Standard Library](stdlib/index.md) |
| Contribute to the compiler | [Compiler Architecture](compiler/index.md) |

---

## Documentation Structure

### Getting Started
New to Sounio? Start here.

- [Installation](getting-started/installation.md) - Build and install the compiler
- [Hello World](getting-started/hello-world.md) - Your first Sounio program
- [Your First Uncertainty](getting-started/your-first-uncertainty.md) - Introduction to Knowledge<T>
- [Project Structure](getting-started/project-structure.md) - Organizing Sounio projects
- [Editor Setup](getting-started/editor-setup.md) - VSCode, Neovim, Emacs LSP

### Language Reference
Complete language syntax and semantics.

- [Syntax Reference](language/syntax-reference.md) - Complete syntax guide
- [Type System](language/type-system.md) - Types, inference, and annotations
- [Ownership & Borrowing](language/ownership-borrowing.md) - Linear types, `&!` references
- [Effects](language/effects.md) - Algebraic effects and handlers
- [Units of Measure](language/units-of-measure.md) - Dimensional analysis
- [Refinement Types](language/refinement-types.md) - SMT-backed type refinements
- [Generics](language/generics.md) - Generic types and trait bounds
- [Pattern Matching](language/pattern-matching.md) - Match expressions
- [Modules & Imports](language/modules-imports.md) - Module system
- [Async/Await](language/async-await.md) - Asynchronous programming
- [FFI](language/ffi.md) - Foreign function interface

### Epistemic Computing
Sounio's core philosophy - computing with uncertainty.

- [Overview](epistemic/index.md) - What is epistemic computing?
- [Knowledge<T>](epistemic/knowledge-type.md) - The core epistemic type
- [Uncertainty Propagation](epistemic/uncertainty-propagation.md) - GUM-compliant propagation
- [Confidence Gates](epistemic/confidence-gates.md) - Confidence-based control flow
- [Provenance](epistemic/provenance.md) - Tracking data origins
- [Uncertainty Models](epistemic/uncertainty-models.md) - Interval, Gaussian, Beta, Monte Carlo
- [Standards Compliance](epistemic/standards-compliance.md) - GUM, ISO 17025, FAIR

### Domain Guides
Scientific computing in specific domains.

#### Pharmacokinetics
- [Overview](domains/pharmacokinetics/index.md) - PK/PD with Sounio
- [MedLang Tutorial](domains/pharmacokinetics/medlang-tutorial.md) - The PK modeling DSL
- [PBPK Modeling](domains/pharmacokinetics/pbpk-modeling.md) - Physiologically-based PK
- [Population PK](domains/pharmacokinetics/population-pk.md) - Mixed-effects modeling
- [Dosing Protocols](domains/pharmacokinetics/dosing-protocols.md) - Dosing regimens
- [Regulatory Compliance](domains/pharmacokinetics/regulatory-compliance.md) - FDA 21 CFR Part 11

#### Scientific Computing
- [Overview](domains/scientific-computing/index.md) - Numerical methods
- [ODE Solvers](domains/scientific-computing/ode-solvers.md) - RK4, Tsit5, BDF
- [Linear Algebra](domains/scientific-computing/linear-algebra.md) - Vectors, matrices, BLAS
- [Autodiff](domains/scientific-computing/autodiff.md) - Automatic differentiation
- [Optimization](domains/scientific-computing/optimization.md) - LM, BFGS, Nelder-Mead
- [GPU Kernels](domains/scientific-computing/gpu-kernels.md) - CUDA/SPIR-V programming

#### Causal Inference
- [Overview](domains/causal-inference/index.md) - Causal computing
- [Do-Calculus](domains/causal-inference/do-calculus.md) - Pearl's do-operator
- [Causal Discovery](domains/causal-inference/causal-discovery.md) - Structure learning
- [Counterfactuals](domains/causal-inference/counterfactuals.md) - What-if reasoning

#### Bayesian Computing
- [Overview](domains/bayesian/index.md) - Probabilistic programming
- [MCMC](domains/bayesian/mcmc.md) - Markov Chain Monte Carlo
- [Variational Inference](domains/bayesian/variational-inference.md) - VI methods

#### Neuroimaging
- [Overview](domains/neuroimaging/index.md) - fMRI analysis
- [fMRI Analysis](domains/neuroimaging/fmri-analysis.md) - Preprocessing pipelines
- [Connectivity](domains/neuroimaging/connectivity-analysis.md) - Brain networks
- [Atlas Support](domains/neuroimaging/atlas-support.md) - Parcellation atlases

### Standard Library API
Complete API reference for all stdlib modules.

- [Overview](stdlib/index.md) - Module organization
- **Core**: [Option](stdlib/core/option.md), [Result](stdlib/core/result.md)
- **Collections**: [Vec](stdlib/collections/vec.md), [HashMap](stdlib/collections/hashmap.md)
- **I/O**: [Files](stdlib/io/files.md)
- **Iterators**: [iter](stdlib/iter.md)
- **Epistemic**: [Knowledge](stdlib/epistemic/knowledge.md), [Propagate](stdlib/epistemic/propagate.md), [MCMC](stdlib/epistemic/mcmc.md), [SMC](stdlib/epistemic/smc.md), [Meta](stdlib/epistemic/meta.md)
- **ODE**: [Solvers](stdlib/ode/solvers.md), [Events](stdlib/ode/events.md)
- **Linear Algebra**: [Vectors](stdlib/linalg/vectors.md), [Matrices](stdlib/linalg/matrices.md), [Decompositions](stdlib/linalg/decompositions.md)
- **Autodiff**: [Index](stdlib/autodiff/index.md)
- **Optimization**: [Index](stdlib/optimization/index.md)

### Compiler Internals
For contributors and compiler engineers.

- [Overview](compiler/index.md) - Architecture overview
- [Pipeline](compiler/pipeline.md) - Compilation stages
- **Frontend**: [Lexer](compiler/frontend/lexer.md), [Parser](compiler/frontend/parser.md), [AST](compiler/frontend/ast.md)
- **Middle**: [Type Checking](compiler/middle/type-checking.md), [Effects Checking](compiler/middle/effects-checking.md)
- **Backend**: [Codegen](compiler/backend/codegen.md)
- [Building](compiler/contributing/building.md) - Development setup

### Tooling
Developer tools and utilities.

- [Overview](tooling/index.md) - Tool ecosystem
- [CLI (souc)](tooling/cli.md) - Command-line interface
- [LSP](tooling/lsp.md) - Language server
- [REPL](tooling/repl.md) - Interactive mode
- [Package Manager](tooling/package-manager.md) - Dependency management
- [Formatter](tooling/formatter.md) - Code formatting

### Reference
Comprehensive reference materials.

- [Keywords](reference/keywords.md) - All keywords
- [Operators](reference/operators.md) - Precedence table
- [Grammar](reference/grammar.md) - Formal EBNF grammar
- **Error Catalog**: [Index](reference/errors/index.md), [Syntax](reference/errors/E0001-E0099.md), [Type](reference/errors/E0100-E0199.md), [Effect](reference/errors/E0200-E0299.md), [Ownership](reference/errors/E0300-E0399.md)

### Cookbook
Practical recipes and patterns.

- [Overview](cookbook/index.md) - How to use recipes
- [Uncertainty Recipes](cookbook/uncertainty-recipes.md) - Working with Knowledge<T>
- [PK Recipes](cookbook/pk-recipes.md) - Pharmacokinetics patterns
- [Data Loading](cookbook/data-loading.md) - Loading data with uncertainty
- [Error Handling](cookbook/error-handling.md) - Error patterns

### Migration Guides
Coming from another language?

- [From Rust](migration/from-rust.md) - Key differences (`&!` not `&mut`)
- [From Python](migration/from-python.md) - For scientists

### Appendix
- [Glossary](appendix/glossary.md) - Terminology

---

## Key Syntax Reminders

```sio
// Variables
let x = 5              // immutable
var y = 10             // mutable (NOT let mut!)

// References
&T                     // shared reference
&!T                    // exclusive/mutable reference (NOT &mut!)

// Effects
fn read_file(path: string) -> string with IO { ... }

// Knowledge<T>
let measurement = Knowledge::new(
    value: 42.0,
    uncertainty: 0.5,
    confidence: 0.95,
    source: "experiment"
)

// Confidence gates
if measurement.confidence > 0.95 {
    proceed(measurement)
}
```

---

## Getting Help

- **GitHub Issues**: [sounio-lang/sounio](https://github.com/sounio-lang/sounio/issues)
- **Discussions**: [GitHub Discussions](https://github.com/sounio-lang/sounio/discussions)

---

*Sounio v0.97.0 - Compute at the Horizon of Certainty*
