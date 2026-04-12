<!-- docs:meta
topic_id: repo.docs.guide.minimum-viable-sounio
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A5
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.guide.minimum-viable-sounio
-->

> **Status**: Production | **Last validated**: 2026-03-07 | **Source**: committed artifacts

# Minimum Viable Sounio (Current Contract)

This guide is intentionally conservative. It describes what is currently validated by committed artifacts and repository gates, not the full ambition implied by the source tree.

Snapshot date: **2026-03-07**

## What is reliable right now

### 1. Checked compiler entry point

For user-facing docs, the safest entry point is the checked JIT artifact:

```bash
export SOUC_BIN="$(pwd)/bin/souc"
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"

"$SOUC_BIN" --version
"$SOUC_BIN" info
```

Current checked-artifact status:

- version `1.0.0-beta.4`
- Cranelift JIT enabled
- LLVM and GPU codegen disabled
- LSP, SMT, ontology, distributed, and package-manager features disabled

There is also a separate checked GPU profile at
`artifacts/omega/souc-bin/souc-linux-x86_64-gpu`. That profile is real and
useful, but it is not the conservative default because it describes a different
backend contract:

- GPU codegen enabled
- Cranelift JIT disabled
- public PTX emission via `build --backend gpu`

### 2. STDLIB reliability lane

Source: `artifacts/stdlib/stdlib_reliability_status.v1.json`

- totals: `pass=81 fail=0 skip=1 total=82`
- gate status: `status_summary=pass`
- inventory:
  - `604` `.sio` files
  - `111` disabled files
  - `44` stub module files
  - `92` active module entrypoints

### 3. STDLIB science pipeline

Source: `artifacts/stdlib/stdlib_science_pipeline_status.v1.json`

- required lanes: `fmri`, `darwin_pbpk`
- totals: `pass=2 fail=0 not_run=0 total=2`
- gate status: `status_summary=pass`

Additional reliability detail from `artifacts/stdlib/stdlib_reliability_status.v1.json`:

- runtime regression enforcement is `soft` locally
- runtime regression summary currently shows `fail`
- recorded runtime regression failures: `4`
- strict enforcement should still be treated as release-blocking when enabled

### 4. STDLIB hyper execution lane

Source: `artifacts/stdlib/stdlib_hyper_execution_status.v1.json`

- totals: `pass=7 fail=0 skip=0 total=7`
- required lanes:
  - `nn`
  - `onn`
  - `qnn`
  - `snn`
  - `spnn`
  - `quantnn`
  - `math`
- gate status: `status_summary=pass`

### 5. Module and test workflow

- `use`-based imports work for active module surfaces.
- Not every module path is callable. The repo contains active entrypoints, stub surfaces, and disabled files at the same time.
- `tests/run-pass/` and `tests/compile-fail/` are better evidence than directory names when deciding whether a language or stdlib feature is reliable.

## Contract levels for stdlib claims

Use these labels when describing support:

1. `active_callable`
   Verified with executable run-pass coverage.

2. `stub_surface`
   Module entrypoint exists, but callable API should not be described as reliable. Check-only or import-contract tests are the right evidence.

3. `disabled_file`
   Implementation is disabled or parked. Do not describe it as available.

## What not to infer

- Do not infer support from a directory existing under `stdlib/` or `self-hosted/`.
- Do not infer runtime maturity from `check` alone.
- Do not describe LLVM, GPU, or LSP support for a checked artifact unless `souc info` for that specific artifact confirms it.

## Minimal verification sequence

Run from repository root:

```bash
export SOUC_BIN="$(pwd)/bin/souc"
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"

"$SOUC_BIN" check examples/hello.sio
"$SOUC_BIN" check tests/run-pass/covid_2020_kernel.sio
"$SOUC_BIN" check tests/run-pass/vancomycin_propagation.sio
"$SOUC_BIN" check tests/compile-fail/vancomycin_low_conf.sio

bash scripts/stdlib_hyper_execution_gate.sh
bash scripts/stdlib_science_pipeline_gate.sh
bash scripts/stdlib_reliability_gate.sh
```

Then read:

- `artifacts/stdlib/stdlib_hyper_execution_status.v1.json`
- `artifacts/stdlib/stdlib_science_pipeline_status.v1.json`
- `artifacts/stdlib/stdlib_reliability_status.v1.json`

If a required gate is not `pass`, treat the affected lane as not reliable.

For GPU-specific work, also read:

- `artifacts/omega/gpu_codegen_parity.v1.json`
- `artifacts/omega/gpu_binary_attestation.v1.json`
- `artifacts/omega/gpu_runtime_attest_gate.v1.json`
- `artifacts/omega/gpu_public_contract.v1.json`
