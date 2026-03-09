# Sounio Installation Guide

This guide describes the installation path that matches the current repository state as of March 9, 2026.

## Recommended Path For This Checkout

This repository already contains signed Linux `x86_64` compiler artifacts under `artifacts/omega/souc-bin/`. If you are working directly from this checkout, use the checked-in JIT artifact first, then switch to the checked GPU artifact only when you need GPU-specific validation.

```bash
cd /path/to/sounio

export SOUC_BIN="$(pwd)/artifacts/omega/souc-bin/souc-linux-x86_64-jit"
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"

"$SOUC_BIN" --version
"$SOUC_BIN" info
"$SOUC_BIN" sysroot stdlib-paths
```

On this repo snapshot, the checked-in JIT binary reports:

- `souc 1.0.0-beta.4`
- Cranelift JIT enabled
- LLVM, GPU, SMT, LSP, ontology, and package-manager features disabled in that specific artifact

The separate checked GPU artifact reports:

- `souc 1.0.0-beta.4`
- GPU codegen enabled
- Cranelift JIT disabled
- public PTX emission via `build --backend gpu`

## Validate The Checkout

Run the canonical `check`-first smoke tests:

```bash
"$SOUC_BIN" check examples/hello.sio
"$SOUC_BIN" check tests/run-pass/covid_2020_kernel.sio
"$SOUC_BIN" check tests/run-pass/vancomycin_propagation.sio
"$SOUC_BIN" check tests/compile-fail/vancomycin_low_conf.sio
```

Expected behavior:

- the first three commands pass
- `vancomycin_low_conf.sio` fails with the expected confidence-bound type mismatch

If you need GPU-profile validation:

```bash
export SOUC_GPU_BIN="$(pwd)/artifacts/omega/souc-bin/souc-linux-x86_64-gpu"

"$SOUC_GPU_BIN" check examples/gpu.sio
"$SOUC_GPU_BIN" check tests/run-pass/gpu_launch_surface.sio
"$SOUC_GPU_BIN" build examples/kernel_matmul.sio --backend gpu -o /tmp/kernel_matmul.ptx
```

## If You Need A Pinned Compiler Path

Use the repo resolver:

```bash
scripts/omega/omega_resolve_souc_bin.sh --print-path --allow-local-fallback
```

This resolves a pinned release binary and falls back to a local executable only when allowed.

## Important Notes

- Do not assume `cargo build` at the repo root is the default setup path. This checkout does not expose a top-level Cargo workspace.
- Always set `SOUNIO_STDLIB_PATH` when you want deterministic stdlib resolution.
- Use `souc sysroot stdlib-paths` to confirm where the compiler is searching.
- Prefer `souc check` when validating examples and docs claims. Runtime and backend-dependent behavior varies by binary variant, including the JIT-versus-GPU split.

## What Is Verified Today

The current gate-backed status in this repo is:

- `artifacts/stdlib/stdlib_reliability_status.v1.json`: `81 pass / 0 fail / 1 skip / 82 total`
- `artifacts/stdlib/stdlib_science_pipeline_status.v1.json`: `pass` for `fmri` and `darwin_pbpk`
- `artifacts/stdlib/stdlib_hyper_execution_status.v1.json`: `pass` for 7 required hyper lanes

Local science runtime regression probes are still tracked in `soft` mode. Strict CI uses fail-closed enforcement.

## Deeper References

- `docs/guide/MINIMUM_VIABLE_SOUNIO.md`
- `docs/guide/getting-started.md`
- `docs/reference/STDLIB_REFERENCE.md`
- `bootstrap/poseidon/README.md`
