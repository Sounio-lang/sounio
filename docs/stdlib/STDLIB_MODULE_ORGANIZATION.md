<!-- docs:meta
topic_id: repo.docs.stdlib.stdlib-module-organization
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A3
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.stdlib.stdlib-module-organization
-->

# STDLIB Module Organization (Executable View)

This page describes how STDLIB modules are organized for current reliability
work, using repository-generated inventory artifacts as ground truth.

Snapshot date: **2026-03-02**

## File Surface Model

STDLIB uses a mixed entrypoint structure:

- `lib.sio`: primary module API surface where present
- `mod.sio`: module entrypoint or compatibility surface
- `*.sio.disabled`: intentionally disabled implementation files

Current inventory (`artifacts/stdlib/stdlib_inventory.v1.json`):

- `sio_files`: 599
- `disabled_files`: 120
- `mod_files`: 70
- `lib_files`: 64
- `module_entrypoints`: 134
- `stub_mod_files`: 43
- `active_module_entrypoints`: 91
- `hyper_active_files`: 17
- `hyper_disabled_files`: 6
- `hyper_stub_mod_files`: 0

## Contract Levels

Each module lane should be interpreted with one of three levels:

1. `active_callable`
- Backed by callable API and validated by run-pass tests.

2. `stub_surface`
- Entry exists but only exposes module surface (often `module <name>` with no
  reliable callable exports for E2E usage).

3. `disabled_file`
- Source exists only as `*.sio.disabled`; callable contract is not active.

## Testing Policy Mapping

- `active_callable` -> prefer `//@ run-pass`
- `stub_surface` -> use `//@ check-only` import/surface checks
- `disabled_file` -> avoid callable assertions; realign tests to active API and
  record `//@ contract-adjustment: ...` when intent changes
- `science lanes` (`tests/stdlib/fmri`, `tests/stdlib/darwin_pbpk`) -> run-pass only; `//@ ignore` is forbidden by gate policy
- `hyper lanes` (`tests/stdlib/{nn,onn,qnn,snn,math}` required set) -> run-pass only; `//@ ignore` is forbidden by hyper gate policy
- runtime regression probes (`as_bytes` literal/text/binary + dynamic slice) are always recorded in science status JSON from committed files in `tests/stdlib/runtime_regression/`
- runtime provenance (`souc_bin`, `souc_version`, `pinned_version_expected`) is always emitted in science status JSON
- strict runtime regression mode is required in CI full gate (`STDLIB_RUNTIME_REGRESSION_STRICT=1`); local default remains soft telemetry
- CI strict mode is intentionally fail-closed; runtime probes must pass for strict green status
- GPU runtime attestation gate is required in CI with `OMEGA_GPU_RUNTIME_GATE_MODE=required`; local default `auto` records remote-unavailable cases as `not_run`
- canonical pinned `souc` version is sourced from `scripts/omega/omega_resolve_souc_bin.sh` (or explicit `SOUNIO_SOUC_VERSION` override)
- blocker taxonomy is normalized as `ssh_unreachable`, `remote_env_missing`, `attestation_invalid`, `pinned_version_mismatch`, `gpu_backend_unavailable`, `runtime_test_fail`

## Reliability Workflow

From repository root:

```bash
bash scripts/stdlib/scan_stdlib.sh --json-out artifacts/stdlib/stdlib_inventory.v1.json
OMEGA_GPU_RUNTIME_GATE_MODE=required bash scripts/omega/omega_gpu_runtime_attest_gate.sh
bash scripts/stdlib_hyper_execution_gate.sh
STDLIB_RUNTIME_REGRESSION_STRICT=1 bash scripts/stdlib_science_pipeline_gate.sh
STDLIB_RUNTIME_REGRESSION_STRICT=1 bash scripts/stdlib_reliability_gate.sh
bash scripts/omega/omega_gpu_runtime_attest_gate.sh
bash scripts/stdlib_hyper_execution_gate.sh
bash scripts/stdlib_science_pipeline_gate.sh
bash scripts/stdlib_reliability_gate.sh
```

Primary status artifact:
- `artifacts/stdlib/stdlib_reliability_status.v1.json`
- `artifacts/stdlib/stdlib_science_pipeline_status.v1.json`
- `artifacts/stdlib/stdlib_hyper_execution_status.v1.json`
- `artifacts/omega/gpu_runtime_attest_gate.v1.json`

Primary reference page:
- `docs/stdlib/STDLIB_REFERENCE.md`
