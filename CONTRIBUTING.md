# Contributing to Sounio

Thank you for your interest in contributing to Sounio! Sounio is a self-hosted, safety-critical systems and scientific programming language for epistemic computing. This document provides guidelines and instructions for contributing to the compiler, standard library, and documentation.

## Code of Conduct

Be respectful. Be constructive. Be patient. Sounio is an academic and active research project with high-fidelity, safety-critical standards. We are building software that communicates the quality of its own knowledge, and we carry that ethos of precision and integrity into our community.

---

## Getting Started

### Prerequisites

Sounio compiles itself (self-hosted). It does not rely on Cargo or Rust for production builds. To build Sounio from source, you need a **Linux x86_64** environment with standard Unix tools:

- `gcc` or `clang` — standard C compiler (required only for Stage 0 bootstrap)
- `make` — build automation utility
- `bash` — standard shell shell
- `sha256sum` — to verify the bootstrap fixed point

### Building from Source (The Bootstrap Ceremony)

The Sounio compiler is written entirely in Sounio. To build the compiler from a clean checkout, we run a multi-stage bootstrap ceremony that compiles a Stage 0 C compiler, compiles successive generations of the Sounio compiler, and verifies that the compiler has reached a stable fixed point (where Stage N and Stage N+1 produce bit-identical ELF binaries):

```bash
# Clone the repository
git clone https://github.com/Sounio-lang/sounio.git
cd sounio

# Execute the self-hosted bootstrap chain and fixed-point verification
make build

# Verify the compiler version
./bin/souc --version
```

### Running the Test Suite

```bash
# Run the entire standard test suite
bash scripts/run_sio_test_suite.sh

# Run a specific subset of tests matching a pattern (e.g. vancomycin)
bash scripts/run_sio_test_suite.sh vancomycin --verbose
```

### Local Git Setup (one-time)

Some tracked files are generated — notably the `docs/governance/` metadata
produced by `scripts/docs/sync_governance_metadata.mjs`. A textual 3-way merge
of these is meaningless and conflicts on nearly every merge from an active
`main`. After cloning, install the repo's git merge driver once so they
auto-resolve:

```bash
bash scripts/dev/install-git-merge-drivers.sh
```

This registers the `governance-regen` merge driver plus a `post-merge` hook
that regenerate the generated `docs/governance/` artifacts from the merged doc
set instead of conflicting on them. It is opt-in per clone (git keeps
merge-driver commands in local config, which is not committed) and will not
overwrite a pre-existing `post-merge` hook.

---

## Development Workflow

### 1. Create a Branch

Always create a descriptive branch for your work:

```bash
git checkout -b feature/your-feature-name
```

Branch naming conventions:
- `feature/` — New compiler features or stdlib modules
- `fix/` — Compiler or runtime bug fixes
- `docs/` — Documentation translations or expansions
- `refactor/` — Code refactoring or cleanup
- `test/` — Enhancing test coverage or adding compiler compile-fail gates

### 2. Sounio Coding Guidelines (Crucial)

Sounio is **NOT Rust**. The compiler will reject common Rust syntax. Follow these rules strictly:

1. **No Semicolons**: Semicolons are not allowed at the end of statements (`let x = 5` not `let x = 5;`).
2. **Mutability**: Use `var` for mutable bindings, never `let mut`.
3. **Exclusive References**: Use `&!` for exclusive references, never `&mut`.
4. **No Rust Macros**: Use `println("text")` and `assert(cond)` instead of macro syntax.
5. **No Unary Minus**: Write `0 - x` instead of `-x` for negation.
6. **Bit Shifts**: Shift operands must be explicitly typed as `u8` (e.g., `x >> 4u8`).
7. **No Closure Literals**: Closure syntax like `|x| x + 1` does not exist. Use named function references.
8. **Explicit Self**: All struct method implementations must declare `self` explicitly as `self: &Type` or `self: &!Type`.
9. **Algebraic Effects System**: All functions with side effects must declare them using the `with` keyword:
   - `IO`: printing, file access, terminal operations
   - `Mut`: exclusive reference mutation or reassignment
   - `Div`: mathematical division or modulo operations
   - `Panic`: array indexing, assertions, or type casting
   - `Alloc`: heap memory allocation

Example of correct Sounio:
```sio
struct Tracker {
    value: f64
}

impl Tracker {
    fn update(self: &!Tracker, input: f64) with Mut {
        self.value = input
    }
}

fn process_measurement(val: f64) -> Knowledge[f64] with Div, Panic {
    let base = Knowledge { value: val, epsilon: 0.02 }
    base * 1.5
}
```

### 3. Testing and Verification

Before submitting any Pull Request:
- Type-check your Sounio files with `./bin/souc check file.sio`
- Compile and run your Sounio files with `./bin/souc run file.sio`
- Ensure all automated checks pass: `bash scripts/run_sio_test_suite.sh`
- Ensure documentation registry is aligned: `node scripts/docs/check_docs_registry.mjs` (sync with `node scripts/docs/sync_governance_metadata.mjs` if needed)

The repository's default compiler surface is `bin/souc` as the official entrypoint (Madaros by default).

### 4. Commit Message Guidelines

We follow a strict semantic and component-scoped commit convention. Commit messages must be structured as:

```
[component] Brief description

- Bulleted list of changes and why they were made
```

**Allowed Components**: `lexer`, `parser`, `ast`, `check`, `types`, `effects`, `hir`, `sir`, `hlir`, `codegen`, `native`, `cli`, `docs`, `stdlib`, `tests`, `epistemic`, `units`, `website`.

*Example*:
```
[check] Reject variable assignment when confidence threshold is breached

- Adds compile-time constraint validation during bidirectional type checking
- Throws TypeMismatch error when Knowledge[T, epsilon] violates required threshold
```

---

## Standard Library Structure

The Sounio standard library (`stdlib/`) is organized by domain and is the core surface for scientific workflows:

| Domain Module | Purpose |
| :--- | :--- |
| `stdlib/epistemic/` | `Knowledge<T>` uncertainty container, GUM propagation, and provenance tracking. |
| `stdlib/units/` | Dimensional analysis, physical quantities, and `VAR_UNIT_DIM` check. |
| `stdlib/clinical/` | PK/PD modeling, clinical trial analytics, and drug dosing algorithms. |
| `stdlib/math/` | Special functions, fractional calculus, autograd, and linear algebra. |
| `stdlib/gpu/` | PTX and GPU-accelerated computing backends. |
| `stdlib/ontology/` | Semantic ontology stores, subclass relations, and clinical terminologies (LOINC/SNOMED). |

---

## Mandatory LLM-Offload Policy

Sounio enforces a pre-commit peer review audit by external AI engines on math, clinical, and external-facing artifacts. If your changes touch any of the following, you **must** run `bin/llm-offload` before committing and append the audit evidence to `.claude/llm_offload_log.md`:

- **Math claims** (PK formulas, GUM derivations, Lean statements):
  `bin/llm-offload -t math-review -p xai -i <file>`
- **Clinical pathway code** (`stdlib/clinical/*`, Vancomycin tests):
  `bin/llm-offload -t review -p deepseek -i <file>`
- **External publications or papers** (`docs/papers/*`):
  `bin/llm-offload --raw <draft> deepseek xai gemini`

---

## License

By contributing to Sounio, you agree that your contributions will be licensed under the **Apache License, Version 2.0** (with copyright assigned to Sounio Language Project).

Thank you for contributing to the future of epistemic computing! 🏛️
