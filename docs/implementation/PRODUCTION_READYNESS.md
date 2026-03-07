<!-- docs:meta
topic_id: repo.docs.implementation.production-readyness
authority: repo_only
audience: maintainers
last_validated: 2026-03-07
validated_by: A7
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.implementation.production-readyness
-->

# Sounio Production Readiness (Linux x86-64)

This document defines the production-readiness boundaries for the first operational rollout.
Scope is intentionally narrow: **Linux x86-64** native ELF execution and
high-confidence internal workflows.

## 1) Stability by feature

### Stable for phase-0 rollout
- `souc build --backend=native`
- `souc run` / `souc check` / `souc compile`
- `compile_source_to_native_elf` and `compile_native_file_to_native_elf` internal API
- Default import resolution and CLI diagnostics for missing imports

### Experimental / controlled
- `souc build --backend=selfhosted-native`
- `--enable-cps`
- LLVM and Cranelift behavior on edge cases
- GPU codegen and runtime paths

## 2) Supported runtime contract

- Target: **Linux x86-64** only (`x86_64-unknown-linux-gnu`).
- Self-hosted native execution uses the internal rust-first compiler pipeline and
  fails fast on parser/typecheck/codegen errors.
- `--backend=selfhosted-native --enable-cps` is currently blocked and returns an
  explicit error (experimental flag only for native LLVM/Cranelift paths).

## 3) Linux setup (recommended)

1. Build compiler:
   - `cargo build -p souc --release`
2. Set stdlib path for CI / non-root runs:
   - `export SOUNIO_STDLIB_PATH=/path/to/sounio/stdlib`
3. Smoke check:
   - `./target/release/souc build examples/hello.sio --backend=selfhosted-native -o /tmp/hello`
   - `/tmp/hello`
4. Validate output:
   - exit code should be `0`
   - command should complete with diagnostics only on failure

## 4) Troubleshooting

### `ImportError` / `Import not found`
- Check `SOUNIO_STDLIB_PATH` and `souc sysroot stdlib-paths`.
- Ensure imports are module-rooted correctly (`import stdlib.foo::bar;`).
- Confirm workspace run path: relative imports are resolved from the source directory.

### Missing or invalid executable permission
- Linux permissions are set by write step; if execution fails, verify binary exists
  and that no SELinux/Noexec restrictions apply.

### `backend=selfhosted-native` fails on unsupported host
- This is expected on non-Linux or non-x86-64 hosts; use `--backend=native` for
  broader local experimentation.

## 5) Release rollback procedure

1. Stop release pipeline and prevent rollouts from staging.
2. Roll back compiler artifact:
   - promote previous tagged `souc` binary to production entrypoint.
3. Freeze dynamic config:
   - keep `SOUNIO_STDLIB_PATH` unchanged for deterministic imports.
4. Re-run smoke matrix and compare:
   - `tests/native_execution.rs`
   - `tests/selfhost_native_e2e.rs`
   - `tests/integration_cli.rs` self-hosted build contract checks
5. Document regression in postmortem notes and reopen risk register before restart.

## 6) CI gates for go/no-go

- `cargo test -p souc --test native_execution`
- `cargo test -p souc --test selfhost_native_e2e`
- `cargo test -p souc --test integration_cli -- --nocapture`
- no critical regression on CLI stable contract for two weeks before go-live

