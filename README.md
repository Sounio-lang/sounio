<!-- docs:meta
topic_id: repo.frontdoor.readme
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.frontdoor.readme
-->

# SOUNIO

Sounio is a programming language and research platform for scientific and epistemic computing. This repository contains the language implementation, self-hosted compiler tree, standard library, website, examples, and the gate artifacts that define what is actually reliable today.

## Current State

This checkout is active and substantial, but it is not an "everything is production-ready" repository. The safest public summary comes from the committed gate artifacts and the shipped compiler binaries:

- `souc check` works on canonical fixtures including `examples/hello.sio`, `tests/run-pass/covid_2020_kernel.sio`, and `tests/run-pass/vancomycin_propagation.sio`.
- the repo also ships a separate checked GPU profile at `artifacts/omega/souc-bin/souc-linux-x86_64-gpu`, with PTX emission through `build --backend gpu`
- `artifacts/stdlib/stdlib_reliability_status.v1.json` reports `status_summary=pass` with `81 pass / 0 fail / 1 skip / 82 total`.
- `artifacts/stdlib/stdlib_science_pipeline_status.v1.json` reports `status_summary=pass` for the real `fmri` and `darwin_pbpk` lanes.
- `artifacts/stdlib/stdlib_hyper_execution_status.v1.json` reports `status_summary=pass` for 7 required hyper-execution lanes.
- The current stdlib inventory records `604` `.sio` files, `111` disabled files, `44` stub module files, and `92` active module entrypoints.
- The local science gate still records 4 runtime regression probe failures in `soft` mode. Strict CI treats those probes as fail-closed.

For the conservative contract, start with [docs/guide/MINIMUM_VIABLE_SOUNIO.md](docs/guide/MINIMUM_VIABLE_SOUNIO.md).

## Quick Start From This Checkout

The repo ships signed Linux `x86_64` compiler artifacts under `artifacts/omega/souc-bin/`. In this checkout, the JIT binary is the easiest verified way to inspect and validate the default language surface.

```bash
cd /path/to/sounio

export SOUC_BIN="$(pwd)/artifacts/omega/souc-bin/souc-linux-x86_64-jit"
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"

"$SOUC_BIN" --version
"$SOUC_BIN" info
"$SOUC_BIN" sysroot stdlib-paths

"$SOUC_BIN" check examples/hello.sio
"$SOUC_BIN" check tests/run-pass/covid_2020_kernel.sio
"$SOUC_BIN" check tests/run-pass/vancomycin_propagation.sio
"$SOUC_BIN" check tests/compile-fail/vancomycin_low_conf.sio
```

Expected results:

- `--version` reports `souc 1.0.0-beta.4` on the checked-in JIT artifact in this repo snapshot
- `examples/hello.sio`, `covid_2020_kernel.sio`, and `vancomycin_propagation.sio` pass `check`
- `vancomycin_low_conf.sio` fails with the expected confidence-bound type error

If you want the repo to resolve a pinned release binary for you, use:

```bash
scripts/omega/omega_resolve_souc_bin.sh --print-path --allow-local-fallback
```

If you need the checked GPU profile:

```bash
export SOUC_GPU_BIN="$(pwd)/artifacts/omega/souc-bin/souc-linux-x86_64-gpu"
"$SOUC_GPU_BIN" info
"$SOUC_GPU_BIN" check examples/gpu.sio
"$SOUC_GPU_BIN" build examples/kernel_matmul.sio --backend gpu -o /tmp/kernel_matmul.ptx
```

## What To Trust Today

Use these rules when deciding what to rely on:

- Trust gate-backed status first, especially the files under `artifacts/stdlib/` and `artifacts/omega/`.
- Treat `tests/run-pass/`, `tests/compile-fail/`, and `tests/stdlib/` as the best source of executable truth.
- Treat `examples/` as mixed: some are canonical and exercised, others are exploratory or depend on optional backends.
- Treat `*.sio.disabled` and stub module wrappers as roadmap inventory, not as callable APIs.

## What Is In The Repo

- `stdlib/` - current standard library surface
- `tests/` - canonical fixtures and gate-backed examples
- `self-hosted/` - self-hosted compiler and runtime pipeline
- `bootstrap/poseidon/` - C99 bootstrap VM for SOIR artifacts
- `website/` - Astro website and docs content
- `docs/` - repo-native documentation, evidence, and implementation notes
- `scripts/` - reliability gates, release helpers, and verification tooling

## Known Limits

- There is no top-level Cargo workspace in this checkout, so `cargo build` from the repo root is not the correct default setup story here.
- The checked-in JIT binary reports Cranelift JIT support; the checked-in GPU binary reports GPU codegen support and PTX emission through `build --backend gpu`. Other features still depend on how a given `souc` binary was built.
- The self-hosted WASM emitter exists in `self-hosted/wasm/`, but it is not yet integrated into the normal CLI flow.
- Some scientific/runtime lanes are validated only under specific gates, and local runtime regression enforcement is still `soft` unless you opt into strict mode.

## Pointers

- [docs/guide/MINIMUM_VIABLE_SOUNIO.md](docs/guide/MINIMUM_VIABLE_SOUNIO.md)
- [docs/guide/getting-started.md](docs/guide/getting-started.md)
- [INSTALL.md](INSTALL.md)
- [docs/reference/STDLIB_REFERENCE.md](docs/reference/STDLIB_REFERENCE.md)
- [website/src/content/docs/en/feature-status.mdx](website/src/content/docs/en/feature-status.mdx)
- [bootstrap/poseidon/README.md](bootstrap/poseidon/README.md)

## License

Apache-2.0. See [LICENSE](LICENSE).
