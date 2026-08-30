<!-- docs:meta
topic_id: repo.docs.internal.implementation.production-readyness
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.implementation.production-readyness
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
- Check `SOUNIO_STDLIB_PATH` and `souc sysroot stdlib-paths` on the checked
  artifact `artifacts/omega/souc-bin/souc-linux-x86_64-gpu`. The default
  `./bin/souc` (Madaros) has no `sysroot` subcommand; there, use the `stdlib:`
  line printed by `souc info`.
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
   - the section 3 smoke checks (`souc info`, `souc check`, `souc run`,
     `souc build --backend native`)
   - the Rust suites this step used to name (`crates/souc/tests/native_execution.rs`,
     `crates/souc/tests/selfhost_native_e2e.rs`, `crates/souc/tests/integration_cli.rs`)
     were deleted with the Rust crates in `79acc192e1`; there is no file-level
     successor, so add whichever repo-local gate scripts under `scripts/ci/`
     cover the change being rolled back
5. Document regression in postmortem notes and reopen risk register before restart.

## 6) CI gates for go/no-go

- `bash scripts/dev/full_gate.sh` (aggregate entrypoint; it runs the fast gate
  and the e2e backend gate, and skips the retired `cargo test -p souc` lane
  whenever `SOUNIO_REPO_HARD_NO_RUST=1`, which is the default)
- the three `cargo test -p souc --test ...` gates previously listed here are
  gone: the Rust `souc` crate and its test suites were removed in `79acc192e1`
- no critical regression on CLI stable contract for two weeks before go-live
