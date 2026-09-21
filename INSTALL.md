# Sounio Installation Guide

This guide describes the installation path that matches the current repository state as of April 22, 2026.

## Recommended Path For This Checkout

This repository already contains checked-in self-hosted compiler artifacts for Linux `x86_64`, macOS `arm64`, and macOS `x86_64` under `artifacts/self-hosted/`. If you are working directly from this checkout, use the launcher at `bin/souc`; it is the official entrypoint and routes by default to Madaros.

```bash
cd /path/to/sounio

export SOUC_BIN="$(pwd)/bin/souc"
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"

"$SOUC_BIN" --version
"$SOUC_BIN" info
"$SOUC_BIN" check examples/hello.sio
"$SOUC_BIN" compile examples/hello.sio -o /tmp/hello.out
"$SOUC_BIN" compile examples/hello.sio -o /tmp/hello-macos --target aarch64-macos
```

On this repo snapshot, `bin/souc` is a host-aware launcher around the checked self-hosted artifacts and the official compiler lane:

- It provides compatibility commands: `check`, `compile`, `build`, `run`, `info`, and `--version`
- It also supports the raw self-hosted compiler interface: `<source.sio> <output> [flags]`
- To use the compatibility legacy engine, set `SOUNIO_SOUC_ENGINE=lean_single`.
- `--target aarch64-macos` and `--target x86_64-macos` emit Mach-O binaries
- `--target aarch64-linux` emits Linux ARM64 ELF binaries
- **Update 2026-08-14 (checked, not assumed):** the "does not provide repl/gpu-emit/full tool surface" line just above went stale in the
  opposite direction of the usual overclaim. Verified today with `./bin/souc --help` and
  running each command: `souc repl` starts a working file-backed REPL; `souc format` and
  `souc fmt` run the checked formatter; `souc lsp --stdio` starts and shuts down cleanly;
  `souc pkg build`/`souc pkg verify` exist; GPU artifact emission is available as
  `souc build <file.sio> --backend gpu -o out.ptx` (not a separate `gpu-emit` subcommand).
  This section otherwise describes April 22, 2026 and has not been re-audited beyond the
  five commands above -- do not assume anything else here is current.

On macOS, the emitted binaries are Mach-O instead of ELF.

## Validate The Checkout

Run the conservative smoke tests:

```bash
"$SOUC_BIN" check examples/hello.sio
"$SOUC_BIN" compile examples/hello.sio -o /tmp/souc-next
"$SOUC_BIN" compile examples/hello.sio -o /tmp/hello-macos --target aarch64-macos
"$SOUC_BIN" run self-hosted/compiler/native_print_f64_smoke.sio
```

To run the legacy bootstrap path intentionally, force it explicitly:

```bash
SOUNIO_SOUC_ENGINE=lean_single "$SOUC_BIN" compile self-hosted/compiler/lean_single.sio -o /tmp/souc-next
```

Expected behavior:

- the checked launcher resolves a host-native compiler artifact
- the compiler rebuild smoke produces a host-native binary
- the macOS-target compile produces a Mach-O ARM64 binary
- the runtime smoke prints the expected floating-point lines on the host OS
- emitted binaries must be executed on the matching target OS/architecture

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

- Do not assume `cargo build` at the repo root is the default setup path.
- Always set `SOUNIO_STDLIB_PATH` when you want deterministic stdlib resolution.
- Prefer `souc check` when validating examples and docs claims.
- The checked self-hosted launcher provides `check/run/compile/build`, but broader omega workflows still require the pinned Linux release lane.

## What Is Verified Today

The current gate-backed status in this repo is:

- `artifacts/stdlib/stdlib_reliability_status.v1.json`: `251 pass / 0 fail / 0 skip / 251 total`
- `artifacts/stdlib/stdlib_science_pipeline_status.v1.json`: `pass` for `fmri` and `darwin_pbpk`
- `artifacts/stdlib/stdlib_hyper_execution_status.v1.json`: `pass` for 7 required hyper lanes

Local science runtime regression probes are still tracked in `soft` mode. Strict CI uses fail-closed enforcement.

## Deeper References

- `docs/guide/MINIMUM_VIABLE_SOUNIO.md`
- `docs/guide/getting-started.md`
- `docs/reference/STDLIB_REFERENCE.md`
- `bootstrap/poseidon/README.md`
