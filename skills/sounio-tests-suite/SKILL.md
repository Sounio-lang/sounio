---
name: sounio-tests-suite
description: "Add and maintain Sounio tests: `tests/run-pass`, `tests/compile-fail`, `tests/ui`, and Rust integration tests under `compiler/tests`; use when writing regressions or adjusting diagnostics expectations."
---

# Sounio Tests Suite

## Overview

Write minimal, high-signal tests that protect language semantics, diagnostics, and regressions.

## Workflow

### 1) Choose the right test bucket

- `tests/run-pass/`: should compile and run
- `tests/compile-fail/`: should fail to compile
- `tests/ui/`: diagnostics/error message matching via `//@ error-pattern:`
- `compiler/tests/`: Rust integration tests for compiler subsystems

### 2) Keep tests small and explicit

- Avoid relying on features listed as missing in `compiler/docs/KNOWN_LIMITATIONS.md`.
- Prefer one behavior per test file; add `//@ ignore` if it’s documenting an aspirational feature.

### 3) Run the narrowest validation first

- If you changed Rust compiler code: `cd compiler && cargo test <name>`
- If you changed language behavior: run the smallest relevant test subset first

## References

- Test annotations and patterns: `references/test-annotations.md`
