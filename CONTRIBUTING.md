# Contributing to Sounio

Thank you for your interest in contributing to Sounio! This document provides guidelines and instructions for contributing.

## Code of Conduct

Be respectful. Be constructive. Be patient. We're building something that matters.

## Getting Started

### Prerequisites

- **Rust 1.70+** — The compiler is written in Rust
- **Git** — Version control
- **LLVM 15+** (optional) — For the LLVM backend

### Building from Source

```bash
# Clone the repository
git clone https://github.com/sounio-lang/sounio.git
cd sounio

# Build the compiler
cd compiler
cargo build --release

# Run tests
cargo test

# Run the compiler
./target/release/souc run examples/hello.sio
```

## Development Workflow

### 1. Fork and Clone

```bash
git clone https://github.com/YOUR_USERNAME/sounio.git
cd sounio
git remote add upstream https://github.com/sounio-lang/sounio.git
```

### 2. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

Branch naming conventions:
- `feature/` — New features
- `fix/` — Bug fixes
- `docs/` — Documentation
- `refactor/` — Code refactoring
- `test/` — Test additions

### 3. Make Changes

- Follow the code style guidelines below
- Add tests for new functionality
- Update documentation as needed

### 4. Test Your Changes

```bash
# Run all tests
cargo test

# Run specific test
cargo test test_name

# Check formatting
cargo fmt --check

# Run clippy
cargo clippy
```

### 5. Commit

Follow the commit message format:

```
[component] Brief description

Components: lexer, parser, ast, check, types, effects, hir, hlir,
           codegen, cli, docs, stdlib, tests, epistemic
```

Examples:
```
[parser] Add support for Knowledge<T> generic syntax
[stdlib] Implement bootstrap_correlation in connectivity module
[docs] Update README with new examples
```

### 6. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

## Code Style Guidelines

### Rust (Compiler)

- Use `rustfmt` for formatting
- Run `clippy` before committing
- No `unwrap()` in library code — use `?` or proper error handling
- Use `thiserror` for error types
- Use `miette` for diagnostics with source spans
- All public items need doc comments

### Sounio (stdlib)

```sio
// Use descriptive names
fn compute_bootstrap_confidence_interval(data: &[f64], n_boot: i32) -> ConfidenceInterval

// Document functions
/// Computes the modularity of a network using the Louvain algorithm.
/// 
/// # Arguments
/// * `weights` - Adjacency matrix (N x N)
/// * `resolution` - Resolution parameter (default: 1.0)
/// 
/// # Returns
/// Modularity value in range [-0.5, 1.0]
fn louvain_modularity(weights: &[[f64]], resolution: f64) -> f64

// Use Knowledge<T> for uncertain values
let result = Knowledge::new(
    value: computed_value,
    uncertainty: computed_uncertainty,
    source: "bootstrap"
)
```

## What to Contribute

### High Priority

- [ ] Language Server Protocol (LSP) implementation
- [ ] LLVM backend optimizations
- [ ] Package manager (`siopkg`)
- [ ] Interactive REPL
- [ ] More stdlib modules

### Medium Priority

- [ ] Documentation improvements
- [ ] Example programs
- [ ] Performance benchmarks
- [ ] Editor integrations

### Always Welcome

- Bug fixes
- Test coverage improvements
- Documentation clarifications
- Typo fixes

## stdlib Contributions

The standard library (`stdlib/`) contains domain-specific modules:

| Module | Description |
|--------|-------------|
| `epistemic/` | Core uncertainty types |
| `medlang/` | PK/PD modeling DSL |
| `fmri/` | Neuroimaging pipeline |
| `causal/` | Causal inference |
| `connectivity/` | Network analysis |
| `gpu/` | GPU acceleration |
| `optimize/` | Optimization |
| `signal/` | Signal processing |
| `data/` | DataFrames |
| `mcmc/` | MCMC sampling |
| `random/` | RNG |
| `quantum/` | Quantum computing |
| `linalg/` | Linear algebra |
| `ode/` | ODE solvers |
| `bayes/` | Bayesian inference |

When adding to stdlib:
1. Follow existing patterns in the module
2. Include uncertainty propagation where appropriate
3. Add comprehensive doc comments
4. Write tests

## Questions?

- Open an issue for bugs or feature requests
- Use discussions for questions
- Check existing issues before creating new ones

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

*Thank you for helping build the future of epistemic computing!* 🏛️
