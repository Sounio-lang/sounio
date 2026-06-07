# Sounio Installation Guide

This guide describes the installation path that matches the current repository state, re-verified 2026-06-07.

## Recommended Path For This Checkout

This repository ships the bootstrap compiler at `bin/souc` — a static `mini_native` ELF. No Rust build step is required for the default workflow. `bin/souc` uses the **raw compiler interface** `souc <source.sio> <output> [flags]`; it is **not** a launcher with `check`/`compile`/`build`/`run` subcommands.

```bash
cd /path/to/sounio

export SOUC_BIN="$(pwd)/bin/souc"
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"

"$SOUC_BIN" --version                                   # usage banner (mini_native)
"$SOUC_BIN" examples/hello.sio /tmp/hello.elf           # compile a host x86_64 ELF
chmod +x /tmp/hello.elf && /tmp/hello.elf               # main()'s return value is the exit code
"$SOUC_BIN" examples/hello.sio /tmp/hello-macos --target aarch64-macos   # cross to macOS arm64 Mach-O
```

Notes on `bin/souc` (the bootstrap binary):

- Invocation is positional: `<source.sio> <output> [flags]`. There are no `check`/`compile`/`build`/`run`/`info` subcommands and no `repl`.
- `--target aarch64-macos` and `--target x86_64-macos` emit Mach-O binaries; `--target x86_64-windows` emits a (prototype) PE/COFF binary.
- `--show-ast` and `--show-types` are pass-through debug flags.

On macOS, the emitted binaries are Mach-O instead of ELF; run them on the matching target OS/arch.

## Build And Drive The Modular Compiler (Madares v0.80.0)

The modular self-hosted compiler is `self-hosted/compiler/main.sio`. Build it with the bootstrap, then invoke it with its **flag-based** interface (this is the lane with `--check` and `--native-v2-compile`):

```bash
ulimit -s 1048576
"$SOUC_BIN" self-hosted/compiler/main.sio /tmp/mc.elf && chmod +x /tmp/mc.elf   # ~2.5 min

/tmp/mc.elf --version                                          # Madares v0.80.0
/tmp/mc.elf --check examples/hello.sio                         # type-check
/tmp/mc.elf --native-v2-compile examples/hello.sio /tmp/h.elf  # single-file source -> native ELF
chmod +x /tmp/h.elf && /tmp/h.elf; echo $?
```

## Validate The Checkout

Run the honest gates that build the modular compiler from `main.sio` via `bin/souc` and assert real behavior:

```bash
# Build mc first (see above), then point the gates at it:
bash tests/native_v2_capgate/run.sh /tmp/mc.elf                      # 32/32 single-file source -> ELF
bash tests/native_v2_multimodule_gate/run.sh /tmp/mc.elf            # 9/9 (1 documented import-typecheck-bypass class)
bash tests/native_v2_backend_soundness_gate/run.sh /tmp/mc.elf      # 40/40, exits nonzero with 1 tracked field-hash residual (by design)

# Cross-compile smoke via the bootstrap:
"$SOUC_BIN" examples/hello.sio /tmp/hello-macos --target aarch64-macos
file /tmp/hello-macos                                                # Mach-O 64-bit arm64 executable
```

Expected behavior:

- `bin/souc` compiles `main.sio` to a host-native ELF (`Madares v0.80.0`)
- the capgate compiles its single-file programs to native ELFs whose exit codes match
- the macOS-target compile produces a Mach-O ARM64 binary
- `native_v2_backend_soundness_gate` reports 40/40 with exactly ONE tracked residual (`C_known_residual_bucket_collision`) — that is expected, not a failure; it should fail only if the residual grows
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
- To type-check, build mc and use `/tmp/mc.elf --check <file>` (the `--check` lane lives in the modular compiler, not in `bin/souc`).
- `bin/souc` has no subcommands and no `repl`; broader omega/GPU workflows live outside this bootstrap lane.

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
