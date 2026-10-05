<!-- docs:meta
topic_id: repo.docs.testing
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.testing
-->

# Testing Sounio

This file provides the reviewer-facing test workflow used for JOSS submission readiness.

## Prerequisites

- Rust toolchain installed
- Repository cloned locally

## 1. Build

```bash
cargo build --release
./target/release/souc --version
```

## 2. Run Rust Tests

```bash
cargo test
```

## 3. Run Epistemic/Example Smoke Tests

```bash
./target/release/souc run examples/hello.sio
./target/release/souc run examples/epistemic_bmi.sio
./target/release/souc run examples/pbpk_simple.sio
./target/release/souc run examples/gpu_hypercomplex.sio
```

Expected outcome: each command completes without compiler/runtime errors.

## 4. Compile-Fail Check (Effects)

```bash
./target/release/souc check tests/compile-fail/effect_missing.sio
```

Expected outcome: compile error indicating a missing effect annotation.

## 5. Optional Fast Gate

```bash
./scripts/dev/fast_gate.sh
```

This runs repository-specific quality checks and a focused regression subset.

## 6. Optional Cultural Fidelity Gate

```bash
python3 scripts/ci/cultural_fidelity_gate.py
python3 scripts/ci/cultural_fidelity_gate.py --self-test
```

The gate checks user-facing golden help/error outputs for Rust-term leakage
(`cargo`, `crate`, `rustc`, etc.) and supports allowlists for internal/dev-only paths.
