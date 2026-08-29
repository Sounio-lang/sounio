<!-- docs:meta
topic_id: repo.docs.guide.minimum-viable-sounio
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A5
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.guide.minimum-viable-sounio
-->

# Minimum Viable Sounio (Current Contract)

This guide is intentionally conservative. It describes what is currently validated by committed artifacts and repository gates, not the full ambition implied by the source tree.

Snapshot date: **2026-04-22**

## What is reliable right now

### 1. Checked compiler entry point

For user-facing docs, the safest entry point in this checkout is the host-aware
checked self-hosted launcher:

```bash
export SOUC_BIN="$(pwd)/bin/souc"
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"

"$SOUC_BIN" --version
"$SOUC_BIN" info
```

Current checked-artifact status:

- version `Madaros v0.80.0`
- checked host lanes:
  - Linux `x86_64`
  - macOS `arm64`
  - macOS `x86_64`
- compatibility commands:
  - `check`
  - `compile`
  - `build`
  - `run`
  - `info`
- Apple support is via the current self-hosted Mach-O lane, not via native-v2 completion
- JIT is not part of the Apple support contract for this checkout

There is also a separate checked GPU profile at
`artifacts/omega/souc-bin/souc-linux-x86_64-gpu`. That profile is real and
useful, but it is not the conservative default because it describes a different
backend contract:

- GPU codegen enabled
- Cranelift JIT disabled
- public PTX emission via `build --backend gpu`

### 2. STDLIB reliability lane

Source: `artifacts/stdlib/stdlib_reliability_status.v1.json`

- totals: `pass=251 fail=0 skip=0 total=251`
- gate status: `status_summary=pass`
- inventory:
  - `927` `.sio` files
  - `0` disabled files
  - `0` stub module files
  - `119` active module entrypoints

### 3. STDLIB science pipeline

Source: `artifacts/stdlib/stdlib_science_pipeline_status.v1.json`

- required lanes: `fmri`, `darwin_pbpk`
- totals: `pass=2 fail=0 not_run=0 total=2`
- gate status: `status_summary=pass`

Additional reliability detail from `artifacts/stdlib/stdlib_reliability_status.v1.json`:

- runtime regression enforcement is `soft` locally
- runtime regression summary currently shows `fail`
- recorded runtime regression failures: `0`
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

### 6. Native-v2 stage1 science spine

Source: `scripts/ci/native_v2_epistemic_science_spine_gate.sh`

The Linux x86-64 native-v2 driver can now self-compile into a generated stage1
driver and use that generated driver to compile and run a small fixed-point
science corpus under `tests/native-v2/science_spine/`.

The gate records:

- byte-identical stage1 driver replay
- per-case ELF x86-64 output
- expected stdout parity
- deterministic replay for emitted corpus binaries
- summary evidence with compiler path, manifest hash, artifact hashes,
  `fallback_path=none`, and `host_callback=none`

The corpus currently covers baseline native hello, control flow, struct return,
fixed-point epistemic arithmetic with a natural 3-field `KnowledgeI64` struct,
fixed-point two-compartment PBPK-style dynamics with a natural `Compartments`
struct, and the octonion/Fano ordered non-collinear triple count `168`.

The focused semantic companion gate is:

```sh
bash scripts/ci/native_v2_semantic_hardening_gate.sh
```

It reuses the generated-stage1 replay path against
`tests/native-v2/semantic_hardening/` for 3-field struct literals, returns,
parameters, and two-struct-argument calls.

The focused f64 ladder gate is:

```sh
bash scripts/ci/native_v2_f64_ladder_gate.sh
```

It promotes f64 literals, f64 arithmetic/comparison, `print_f64` output, and a
monomorphic `KnowledgeF64` struct witness through the generated stage1 driver.
It also promotes a narrow `Knowledge<f64>` witness covering a `struct
Knowledge<T>` declaration, `Knowledge<f64>` literal construction,
`Knowledge<f64>` parameters, a `Knowledge<f64>` return value, and f64 field
arithmetic in that monomorphic instantiation. The `print_f64` surface is
currently a narrow fixed three-decimal positive fixture witness, not a complete
formatting claim.

The ISO GUM uncertainty propagation gate is:

```sh
bash scripts/ci/native_v2_gum_primitives_gate.sh
```

It promotes the `sqrt_f64` SSE2 builtin (`sqrtsd` in the generated ELF), ISO
JCGM 100:2008 GUM quadrature addition (`u_c = sqrt(u_a² + u_b²)`), the GUM
multiplicative rule, and a two-compartment PBPK simulation with `Knowledge<f64>`
state and a confidence gate — all through the generated stage1 driver. The gate
verifies `sqrtsd` presence via `objdump` when available and records
`gum_sse2_verified` in the summary JSON. The GUM PBPK entry is also promoted
into the science spine manifest.

The accelerator tracking gate is:

```sh
bash scripts/ci/native_v2_epistemic_accel_spine_gate.sh
```

It promotes the current native algebra accelerator spine: generated-stage1
nested `Knowledge<f64>` CPU oracles, public GPU-profile f64 PTX fixtures, and
compiler-owned hypercomplex/O-SSM source probes for octonion, sedenion, HMMA
Fano sign correction, and O-SSM f32/f64 PTX surfaces. The public GPU fixtures
now include O-SSM-shaped octonion recurrence arithmetic and S-SSM-shaped
Cayley-Dickson/sedenion recurrence arithmetic.

The structural accelerator rows pass when the public GPU artifact can emit PTX.
The top-level gate reports `status=partial` rather than `status=pass` on hosts
where the native CUDA runtime smoke cannot see `libcuda.so.1`.

The gate uses `scripts/gpu/ptx_f64_legalize.py` to legalize f64 parameter
loads, f64 arithmetic opcodes, f64 immediate literals, and f64 register banks
after the pinned public GPU artifact emits raw PTX. The raw PTX is preserved
for root-cause comparison. This is a checked PTX contract, not a claim that the
pinned beta.4 GPU binary has been source-rebuilt.

The CUDA runtime parity gate is:

```sh
bash scripts/ci/native_v2_epistemic_gpu_runtime_parity_gate.sh
```

It checks a runtime manifest under `tests/gpu/epistemic_runtime/` with baseline
f64 vector launch rows plus O-SSM-shaped octonion and S-SSM-shaped sedenion
f64 runtime fixtures. This gate rejects fallback by treating `GPU unavailable:`
as a failure. On hosts without a visible CUDA driver runtime it reports
`status=partial` with per-row `not_run` results.

Do not infer broader native-v2 support from this gate. In particular, this does
not promote full general-purpose generics, native floating-register ABI,
general PBPK stdlib imports, Apple native-v2 parity, epistemic f64 GPU runtime
execution, full O-SSM/S-SSM runtime parity, or diverse-double-compiling.

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
- Do not treat Apple support as implying JIT parity.

## Minimal verification sequence

Run from repository root:

```bash
export SOUC_BIN="$(pwd)/bin/souc"
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"

"$SOUC_BIN" check examples/hello.sio
"$SOUC_BIN" compile examples/hello.sio -o /tmp/souc-next
"$SOUC_BIN" run self-hosted/compiler/native_print_f64_smoke.sio
"$SOUC_BIN" compile examples/hello.sio -o /tmp/hello-macos --target aarch64-macos

bash scripts/stdlib_hyper_execution_gate.sh
bash scripts/stdlib_science_pipeline_gate.sh
bash scripts/stdlib_reliability_gate.sh
bash scripts/ci/native_v2_epistemic_science_spine_gate.sh
bash scripts/ci/native_v2_f64_ladder_gate.sh
```

To explicitly exercise the legacy bootstrap compiler, set `SOUNIO_SOUC_ENGINE=lean_single` when invoking `bin/souc`.

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
