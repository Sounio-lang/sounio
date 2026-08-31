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
- self-hosted `compile --backend=native-v2`
- `--enable-cps`
- LLVM and Cranelift behavior on edge cases
- GPU codegen and runtime paths

## 2) Supported runtime contract

- Target: **Linux x86-64** only (`x86_64-unknown-linux-gnu`).
- Self-hosted native execution uses the internal rust-first compiler pipeline and
  fails fast on parser/typecheck/codegen errors.
- `--backend=selfhosted-native --enable-cps` is currently blocked and returns an
  explicit error (experimental flag only for native LLVM/Cranelift paths).
- `--backend=native-v2` is the preview self-hosted x86 Machine-IR lane. It
  emits the canonical v2 contract artifact, produces strict scalar-core native
  ELFs, publishes stack-map/deopt metadata plus a concrete `gc_state` block and
  managed-object descriptor table into `RuntimeContext`, emits the v2 root-map
  schema fields alongside each safepoint record, routes native-v2 heap objects
  through a fixed-capacity handle table, compiles allocation overflow to a
  real runtime slow-path trap, and carries an executable descriptor-driven
  mark/compact GC model with precise slot scanning and pin-aware relocation
  rules validated in compiler self-tests. `--backend=native-v2-shadow` has
  been retired and is not a separate runtime profile.

## 3) Linux setup (recommended)

1. Use the checked JIT compiler artifact:
   - `export SOUC_BIN="$(pwd)/bin/souc"`
   - `"$SOUC_BIN" info`
   - `source scripts/lib/stage_native_runtime_bundle.sh`
   - `sounio_stage_native_runtime_bundle "$SOUC_BIN"`
2. Set stdlib path for CI / non-root runs:
   - `export SOUNIO_STDLIB_PATH=/path/to/sounio/stdlib`
3. Smoke check:
   - `"$SOUC_BIN" check examples/hello.sio`
   - `"$SOUC_BIN" run examples/hello.sio`
   - `"$SOUC_BIN" build examples/simple_test.sio --backend native -o /tmp/simple_test_native`
4. Validate output:
   - exit code should be `0`
   - command should complete with diagnostics only on failure

If you are validating an internal selfhosted-native profile, document that
binary separately. Do not treat a root-level Cargo bootstrap command as the
canonical setup story for this repo snapshot.

## 4) Troubleshooting

### `ImportError` / `Import not found`
- Check `SOUNIO_STDLIB_PATH` and the `stdlib:` line printed by `"$SOUC_BIN" info`.
  (`souc sysroot` was a Rust-era subcommand and was removed with the Rust crates
  in `79acc192e1`; `souc info` is where the resolved stdlib root is reported now.)
- Ensure imports are module-rooted correctly (`import stdlib.foo::bar;`).
- Confirm workspace run path: relative imports are resolved from the source directory.

### Missing or invalid executable permission
- Linux permissions are set by write step; if execution fails, verify binary exists
  and that no SELinux/Noexec restrictions apply.

### `backend=selfhosted-native` fails on unsupported host
- This is expected on non-Linux or non-x86-64 hosts; use `--backend=native` for
  broader local experimentation.

### `backend=native-v2` does not behave like a new production backend
- This is expected. The preview lane exists to pin `RuntimeContext`, target
  register policy, Machine IR, deopt, and GC contract expectations in emitted
  artifacts before the full backend rewrite graduates. On `x86-64` it now owns
  a real scalar-core preview emitter, runtime metadata publication, a managed
  handle/descriptor substrate for heap objects, an allocation slow-path
  substrate in the self-hosted shell, and an executable mark/compact GC model,
  but the checked outer `souc` binary still treats stable `--backend=native` as
  the public native path.

## 5) Release rollback procedure

1. Stop release pipeline and prevent rollouts from staging.
2. Roll back compiler artifact:
   - promote previous tagged `souc` binary to production entrypoint.
3. Freeze dynamic config:
   - keep `SOUNIO_STDLIB_PATH` unchanged for deterministic imports.
4. Re-run smoke matrix and compare:
   - `bash scripts/run_sio_test_suite.sh` (the self-hosted `.sio` suite; it
     dispatches to `scripts/dev/run_sio_test_suite.sh`)
   - The three Rust smoke tests this step used to name -- `native_execution.rs`,
     `selfhost_native_e2e.rs` and `integration_cli.rs` -- lived under
     `crates/souc/tests/` and were removed with the whole Rust tree on
     2026-02-26 by `79acc192e1 [cutover] Remove Rust crates -- compiler is
     self-hosted`. They have no direct replacement: the CLI build-contract
     coverage they provided has not been reconstructed in the `.sio` suite.
5. Document regression in postmortem notes and reopen risk register before restart.

## 6) CI gates for go/no-go

- `bash scripts/run_sio_test_suite.sh`
- The three `cargo test -p souc --test ...` invocations this list used to carry
  are not runnable: the repository has no `Cargo.toml` and no `crates/` tree
  after `79acc192e1`, so there is no cargo workspace and no `souc` crate to
  test. Do not treat their absence as a passing gate.
- no critical regression on CLI stable contract for two weeks before go-live
