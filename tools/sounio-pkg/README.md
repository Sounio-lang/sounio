# sounio-pkg - Package Manager for Sounio

## 🎯 Overview

`sounio-pkg` is a package manager and build system for the Sounio programming language, designed specifically for scientific and epistemic computing. It provides:

Supported contract: local package creation, local build/check/test, and local package-import workflows.

Not supported in the current release contract: hosted public-registry publishing,
login, hosted search, broad dependency resolution, or workspace publishing.

## 🚀 Quick Start

### Installation
```bash
# Verify installation
tools/sounio-pkg/sounio-pkg version
```

### Creating a New Package
```bash
# Create a new Sounio package
tools/sounio-pkg/sounio-pkg new my-scientific-package
cd my-scientific-package

# Structure created:
# my-scientific-package/
# ├── sounio.toml          # Package manifest
# ├── src/
# │   └── lib.sio          # Main library file
# ├── tests/
# │   └── test_lib.sio     # Test files
# └── examples/            # Example programs
```

### Building and Testing
```bash
# Build the package
../tools/sounio-pkg/sounio-pkg build

# Run tests
../tools/sounio-pkg/sounio-pkg test
```

## 📦 Package Manifest (sounio.toml)

```toml
[package]
name = "epistemic-linear-algebra"
version = "1.0.0"
authors = ["Researcher Name <researcher@lab.edu>"]
edition = "2024"
license = "Apache-2.0"
description = "Linear algebra operations with epistemic uncertainty"
repository = "https://github.com/username/epistemic-linear-algebra"

# Epistemic metadata
[package.metadata]
confidence = 0.95  # Overall package confidence
provenance = "Peer-reviewed implementation"
validation_status = "validated"

# Dependencies
[dependencies]
epistemic-core = "^1.0"
knowledge-types = "^2.0"

[dev-dependencies]
sounio-test = "^0.1"

# Build configuration
[build]
target = "native"  # native, gpu, wasm
optimization = "balanced"  # debug, balanced, performance

# Features (conditional compilation)
[features]
default = ["openblas", "cuda"]
openblas = []  # BLAS acceleration
cuda = []      # GPU acceleration
basic = []     # No external dependencies

# Workspace (for multi-package projects)
[workspace]
members = ["crates/*"]
```

## 🏗️ Project Structure

```
my-package/
├── sounio.toml              # Package manifest
├── README.md                # Project documentation
├── LICENSE                  # License file
├── src/
│   ├── lib.sio             # Main library entry point
│   ├── matrix/             # Module directory
│   │   ├── mod.sio         # Module declaration
│   │   ├── operations.sio  # Matrix operations
│   │   └── decomposition.sio # Matrix decompositions
│   └── statistics/         # Another module
│       ├── mod.sio
│       └── distributions.sio
├── tests/
│   ├── test_matrix.sio     # Test files
│   └── test_statistics.sio
├── examples/
│   ├── basic_usage.sio     # Example programs
│   └── advanced_demo.sio
├── benchmarks/
│   └── performance.sio     # Benchmark suites
└── docs/
    ├── API.md              # API documentation
    └── examples/           # Documentation examples
```

## 🔧 Commands

### Package Management
```bash
# Create new package
sounio-pkg new <name>

# Initialize in existing directory
sounio-pkg init

# Build package
sounio-pkg build [--release] [--target native|gpu|wasm]

# Run tests
sounio-pkg test [--verbose] [--test <name>]

# Run benchmarks
sounio-pkg bench

# Generate documentation
sounio-pkg doc [--open]

# Clean build artifacts
sounio-pkg clean
```

Dependency management, publishing, hosted search, and registry authentication are design surfaces until a dedicated gate covers them.

### Workspace Management
```bash
# Add member to workspace
sounio-pkg workspace add <path>

# Remove member from workspace
sounio-pkg workspace remove <path>

# List workspace members
sounio-pkg workspace list

# Run command in all workspace members
sounio-pkg workspace run <command>
```

## 🎯 Epistemic Features

### Versioning with Confidence
```toml
[dependencies]
epistemic-stats = { version = "^1.0", confidence = 0.92 }
knowledge-ml = { version = "^2.0", confidence = 0.85, provenance = "validated" }
```

### Build with Uncertainty Propagation
```bash
# Build with confidence threshold
sounio-pkg build --min-confidence 0.80

# Validate epistemic properties
sounio-pkg validate --epistemic

# Generate uncertainty report
sounio-pkg audit --uncertainty
```

### Dependency Resolution with Uncertainty

The package manager considers:
1. **Version confidence** - How reliable is this version?
2. **Provenance** - Where does this package come from?
3. **Validation status** - Has it been scientifically validated?
4. **Compatibility confidence** - How certain are we about compatibility?

## 🔗 Integration with Existing Sounio Ecosystem

### Compiler Integration
```bash
# Uses souc compiler under the hood
sounio-pkg build --compiler $(which souc)

# Supports all souc backends
sounio-pkg build --backend native
sounio-pkg build --backend gpu
sounio-pkg build --backend wasm
```

### LSP Integration
```toml
# sounio.toml can configure LSP
[lsp]
rustism-detection = true
epistemic-validation = true
auto-import = true
```

### CI/CD Integration
```yaml
# GitHub Actions example
- name: Build with sounio-pkg
  run: sounio-pkg build --release

- name: Run tests
  run: sounio-pkg test

- name: Validate epistemic properties
  run: sounio-pkg validate --epistemic
```

## 🏆 Example: Scientific Package

### `sounio.toml` for a bioinformatics package:
```toml
[package]
name = "epistemic-bioinformatics"
version = "0.1.0"
description = "Bioinformatics algorithms with epistemic uncertainty"
license = "MIT"

[package.metadata]
confidence = 0.88
provenance = "NIH-funded research"
validation_status = "peer-reviewed"
domain = "bioinformatics"

[dependencies]
epistemic-core = "^1.0"
knowledge-stats = "^2.0"
sequence-alignment = "^0.5"

[features]
default = ["blast", "clustal"]
blast = []      # BLAST algorithm support
clustal = []    # Clustal Omega support
phylo = []      # Phylogenetic trees

[[example]]
name = "dna_analysis"
path = "examples/dna_analysis.sio"
required-features = ["blast"]
```

## 🚧 Roadmap

### Phase 1 (MVP)
- [x] Basic package creation and initialization
- [x] Dependency management from local files
- [x] Build system integration with souc
- [x] Test runner integration

### Phase 2
- [ ] Remote registry support
- [ ] Workspace management
- [ ] Epistemic version resolution
- [ ] Documentation generation

### Phase 3
- [ ] Binary distribution
- [ ] Cross-compilation support
- [ ] Advanced dependency graphs
- [ ] Plugin system

### Phase 4
- [ ] Distributed build caching
- [ ] Scientific reproducibility features
- [ ] Integration with data registries
- [ ] Federated package management

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## 📄 License

Apache 2.0 - See [LICENSE](LICENSE)
