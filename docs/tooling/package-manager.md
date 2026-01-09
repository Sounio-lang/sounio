# Sounio Package Manager

The Sounio package manager handles project scaffolding, dependency management, and builds. It uses `sounio.toml` as the manifest format.

## Quick Start

```bash
# Create new project
souc pkg init my-project
cd my-project

# Add a dependency
souc pkg add statistics ^1.0

# Build
souc pkg build

# Run
souc pkg run

# Test
souc pkg test
```

## Project Structure

A typical Sounio project:

```
my-project/
  sounio.toml          # Project manifest
  src/
    main.sio           # Main entry point (for binaries)
    lib.sio            # Library root (for libraries)
  tests/
    test_main.sio      # Test files
  benches/
    bench_main.sio     # Benchmark files
  examples/
    example.sio        # Example programs
  target/              # Build output (gitignored)
```

## Manifest Format (`sounio.toml`)

### Package Section

```toml
[package]
name = "my-project"
version = "0.1.0"
authors = ["Your Name <you@example.com>"]
description = "A Sounio project"
license = "MIT"
repository = "https://github.com/you/my-project"
homepage = "https://my-project.dev"
documentation = "https://docs.my-project.dev"
readme = "README.md"
keywords = ["science", "epistemic"]
categories = ["science"]
edition = "2024"
publish = true
```

Required fields:
- `name` - Package name (lowercase, alphanumeric, hyphens)
- `version` - Semantic version (MAJOR.MINOR.PATCH)

### Dependencies

```toml
[dependencies]
# Version requirement (caret by default)
statistics = "1.0"

# Explicit caret (compatible versions)
linear-algebra = "^2.1"

# Tilde (patch updates only)
data-frames = "~0.5.3"

# Exact version
crypto = "=1.2.3"

# Range
parser = ">=1.0, <2.0"

# Detailed specification
http-client = { version = "1.0", features = ["json", "tls"] }

# Git dependency
my-lib = { git = "https://github.com/org/my-lib", branch = "main" }

# Local path
local-lib = { path = "../local-lib" }

# Optional dependency (enabled by feature)
gpu-compute = { version = "0.5", optional = true }
```

### Dev Dependencies

Dependencies only needed for tests and benchmarks:

```toml
[dev-dependencies]
test-utils = "1.0"
benchmark = "0.3"
```

### Build Dependencies

Dependencies for build scripts:

```toml
[build-dependencies]
code-gen = "0.2"
```

### Features

Define optional feature flags:

```toml
[features]
default = ["std"]
std = []
gpu = ["gpu-compute"]
full = ["std", "gpu", "simd"]
simd = []
```

Enable features in dependencies:

```toml
[dependencies]
my-lib = { version = "1.0", features = ["json"] }

# Disable default features
other-lib = { version = "2.0", default-features = false }
```

### Build Configuration

```toml
[build]
script = "build.sio"          # Build script path
target-dir = "target"         # Output directory
jobs = 4                      # Parallel jobs
incremental = true            # Incremental compilation
```

### Profiles

Configure build profiles:

```toml
[profile.dev]
opt-level = 0
debug = 2
debug-assertions = true
overflow-checks = true

[profile.release]
opt-level = 3
debug = false
debug-assertions = false
overflow-checks = false
lto = "thin"
codegen-units = 1
panic = "abort"
```

### Targets

#### Binary Targets

```toml
[[bin]]
name = "my-app"
path = "src/main.sio"
required-features = ["cli"]

[[bin]]
name = "my-tool"
path = "src/bin/tool.sio"
```

#### Library Target

```toml
[lib]
name = "my_lib"
path = "src/lib.sio"
crate-type = ["lib", "dylib"]
```

#### Examples

```toml
[[example]]
name = "basic"
path = "examples/basic.sio"

[[example]]
name = "advanced"
path = "examples/advanced.sio"
required-features = ["full"]
```

#### Tests

```toml
[[test]]
name = "integration"
path = "tests/integration.sio"

[[test]]
name = "unit"
path = "tests/unit.sio"
```

#### Benchmarks

```toml
[[bench]]
name = "perf"
path = "benches/perf.sio"
required-features = ["bench"]
```

### Workspace

For multi-package projects:

```toml
[workspace]
members = [
    "packages/core",
    "packages/cli",
    "packages/gui",
]
exclude = ["packages/experimental"]
default-members = ["packages/core", "packages/cli"]

# Shared dependencies
[workspace.dependencies]
serde = "1.0"
```

## Commands

### `souc pkg init`

Create a new project:

```bash
# Create binary project
souc pkg init my-app

# Create library
souc pkg init --lib my-lib

# With specific edition
souc pkg init --edition 2024 my-project
```

### `souc pkg build`

Build the project:

```bash
# Debug build
souc pkg build

# Release build
souc pkg build --release

# Specific target
souc pkg build --bin my-app

# With features
souc pkg build --features "gpu simd"

# All features
souc pkg build --all-features
```

### `souc pkg run`

Build and run:

```bash
# Run default binary
souc pkg run

# Run specific binary
souc pkg run --bin my-tool

# Pass arguments
souc pkg run -- arg1 arg2

# Release mode
souc pkg run --release
```

### `souc pkg test`

Run tests:

```bash
# All tests
souc pkg test

# Specific test
souc pkg test test_name

# With output
souc pkg test -- --nocapture

# Only unit tests
souc pkg test --lib

# Only integration tests
souc pkg test --test integration
```

### `souc pkg bench`

Run benchmarks:

```bash
# All benchmarks
souc pkg bench

# Specific benchmark
souc pkg bench bench_name
```

### `souc pkg check`

Type-check without building:

```bash
souc pkg check
souc pkg check --all-targets
```

### `souc pkg doc`

Generate documentation:

```bash
# Generate docs
souc pkg doc

# Open in browser
souc pkg doc --open

# Include private items
souc pkg doc --document-private-items
```

### `souc pkg add`

Add dependencies:

```bash
# Add from registry
souc pkg add serde

# Specific version
souc pkg add statistics ^1.2

# With features
souc pkg add http-client --features "json tls"

# As dev dependency
souc pkg add test-utils --dev

# From git
souc pkg add my-lib --git https://github.com/org/my-lib
```

### `souc pkg remove`

Remove dependencies:

```bash
souc pkg remove unused-dep
```

### `souc pkg update`

Update dependencies:

```bash
# Update all
souc pkg update

# Update specific dependency
souc pkg update statistics
```

### `souc pkg clean`

Remove build artifacts:

```bash
souc pkg clean
```

### `souc pkg publish`

Publish to registry:

```bash
# Dry run
souc pkg publish --dry-run

# Publish
souc pkg publish

# Allow dirty working directory
souc pkg publish --allow-dirty
```

### `souc pkg search`

Search the registry:

```bash
souc pkg search statistics
souc pkg search --limit 20 machine-learning
```

### `souc pkg info`

Show package information:

```bash
souc pkg info statistics
```

## Version Requirements

| Syntax | Meaning |
|--------|---------|
| `1.2.3` | `^1.2.3` (default caret) |
| `^1.2.3` | `>=1.2.3, <2.0.0` |
| `^0.2.3` | `>=0.2.3, <0.3.0` |
| `^0.0.3` | `>=0.0.3, <0.0.4` |
| `~1.2.3` | `>=1.2.3, <1.3.0` |
| `=1.2.3` | Exactly `1.2.3` |
| `>=1.2.3` | At least `1.2.3` |
| `<2.0.0` | Less than `2.0.0` |
| `>=1.0, <2.0` | Range |
| `*` | Any version |

## Lock File

`sounio.lock` records exact versions used:

```toml
[[package]]
name = "statistics"
version = "1.2.5"
source = "registry"
checksum = "abc123..."

[[package]]
name = "linear-algebra"
version = "2.1.0"
source = "registry"
dependencies = ["statistics"]
```

Commit `sounio.lock` for applications, not for libraries.

## Registry

The default registry is `https://registry.sounio.dev`.

Configure in `~/.sounio/config.toml`:

```toml
[registry]
default = "https://registry.sounio.dev"
token = "your-api-token"

[registries.private]
url = "https://private.example.com"
token = "private-token"
```

Use private registry:

```toml
[dependencies]
internal-lib = { version = "1.0", registry = "private" }
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `SOUNIO_HOME` | Installation directory |
| `SOUNIO_CACHE` | Package cache (`~/.sounio/cache`) |
| `SOUNIO_REGISTRY` | Default registry URL |
| `SOUNIO_TOKEN` | Registry authentication token |

## See Also

- [CLI Reference](cli.md) - Compiler commands
- [Formatter](formatter.md) - Code style
- [Getting Started](../getting-started/index.md) - First project tutorial
