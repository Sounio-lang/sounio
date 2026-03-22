<!-- docs:meta
topic_id: repo.docs.archived.gemini
authority: archived
audience: maintainers
last_validated: 2026-03-07
validated_by: A7
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.archived.gemini
-->


<!-- docs:status-note:start -->
> Docs status: `archived`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Gemini Code Assistant Context

This document provides a comprehensive overview of the Sounio project for the Gemini Code Assistant.

## Project Overview

Sounio is a systems programming language designed for **epistemic computing**. Its primary goal is to manage uncertainty in scientific and data-intensive computations. The core of the language is the `Knowledge<T>` type, which encapsulates a value along with its uncertainty, confidence level, and provenance (origin).

The project includes:
*   The **Sounio compiler (`souc`)**, written in Rust.
*   An extensive **standard library (`stdlib/`)** with over 215,000 lines of code for various scientific domains, including:
    *   Pharmacokinetics/Pharmacodynamics (PK/PD) with a DSL called `MedLang`.
    *   Neuroimaging (fMRI).
    *   Causal inference.
    *   GPU-accelerated computing.
*   A **self-hosted compiler**, meaning the Sounio compiler is written in Sounio and can compile itself. This is a key feature and a focus of the development workflow.

The codebase is a monorepo containing the Rust compiler, the Sounio standard library, documentation, and a large number of examples and tests.

## Building and Running the Project

### Prerequisites
- Rust toolchain (>= 1.80, edition 2024 recommended)
- `make` for verification scripts
- (Optional) NVIDIA GPU + CUDA for GPU features

### Build the Compiler

The primary build command uses Cargo to build the Rust-based compiler.

```bash
cargo build --release
```
The main executable will be located at `./target/release/souc`.

### Run a Sounio Program

Use the `souc` executable to run Sounio source files (`.sio`).

```bash
./target/release/souc run examples/hello.sio
```

### Running Tests

The project has multiple layers of testing.

1.  **Rust Workspace Tests:**
    These tests cover the Rust-based compiler components.
    ```bash
    cargo test --workspace
    ```

2.  **Self-Host Verification:**
    This is a critical part of the workflow and ensures the self-hosted compiler is correct and reproducible. It involves a 3-stage bootstrap process. The `Makefile.verify` provides convenient targets.
    ```bash
    # Run the full 3-stage bootstrap verification (slow, comprehensive)
    make verify

    # Run a quicker, less comprehensive check
    make verify-quick
    ```
    The `scripts/fast_gate.sh` script is also used for pre-flight checks.

3.  **Running Example Files:**
    Many `.sio` files in the `examples/` directory can be run directly to test functionality.
    ```bash
    ./target/release/souc run examples/epistemic_bmi.sio
    ```

## Development Conventions

The project has well-defined conventions documented in several files.

### Code Style

*   **Sounio Code:** Follows the `docs/STYLE_GUIDE.md`. The `souc fmt` command can be used for automatic formatting.
*   **Rust Code:** Standard `rustfmt` and `clippy` conventions are used.

### Commits and Branches

*   **Branch Naming:** Branches should be prefixed with `feature/`, `fix/`, `docs/`, etc.
*   **Commit Messages:** Should be prefixed with the component they modify, e.g., `[parser] Add support for new syntax`.

### Contribution Workflow

1.  Fork the repository.
2.  Create a feature branch.
3.  Make changes, adding tests and documentation.
4.  Run tests and style checks (`cargo test`, `cargo fmt`, `cargo clippy`).
5.  Push to the fork and create a Pull Request.

The `CONTRIBUTING.md` and `docs/DEVELOPER_WORKFLOW.md` files provide detailed information on the development process, especially concerning the self-hosted compiler.

## Key Files and Directories

*   `Cargo.toml`: Defines the Rust workspace and dependencies.
*   `crates/souc/`: Source code for the Rust-based Sounio compiler.
*   `stdlib/`: The Sounio standard library, written in Sounio.
*   `self-hosted/`: The Sounio compiler, written in Sounio.
*   `examples/`: A large collection of Sounio example programs.
*   `tests/`: Integration and language-level tests.
*   `docs/`: Project documentation, including style guides and architecture overviews.
*   `Makefile.verify`: Makefile for running bootstrap verification.
*   `README.md`: The main entry point for understanding the project's vision and features.
