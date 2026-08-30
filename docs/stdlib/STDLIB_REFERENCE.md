<!-- docs:meta
topic_id: website.docs.stdlib
authority: dual
audience: users
last_validated: 2026-03-07
validated_by: A3
source_of_truth: docs/governance/topic-registry.v1.json#website.docs.stdlib
-->

# Sounio STDLIB Reference (Executable Snapshot)

> **Looking for function signatures?** See [STDLIB_API_REFERENCE.md](STDLIB_API_REFERENCE.md) — auto-generated reference with 1,900+ pub fn organized by module.

This document tracks the **current executable STDLIB contract** for reliability work.
Values below are sourced from machine-generated artifacts, not aspirational specs.

- Inventory source: `artifacts/stdlib/stdlib_inventory.v1.json`
- Reliability gate source: `artifacts/stdlib/stdlib_reliability_status.v1.json`
- Science gate source: `artifacts/stdlib/stdlib_science_pipeline_status.v1.json`
- Hyper gate source: `artifacts/stdlib/stdlib_hyper_execution_status.v1.json`
- GPU attest gate source: `artifacts/omega/gpu_runtime_attest_gate.v1.json`
- Snapshot date: **2026-03-02**

## Reliability Snapshot

| Metric | Value |
|---|---:|
| `sio_files` | 599 |
| `disabled_files` | 120 |
| `stub_mod_files` | 43 |
| `active_module_entrypoints` | 91 |
| `hyper_active_files` | 17 |
| `hyper_disabled_files` | 6 |
| `hyper_stub_mod_files` | 0 |
| E2E `pass` | 71 |
| E2E `fail` | 0 |
| E2E `skip` | 9 |
| E2E `total` | 80 |
| Gate `status_summary` | `pass` |

## Science Pipeline Snapshot (fMRI + Darwin PBPK)

| Metric | Value |
|---|---:|
| Science lanes `pass` | 2 |
| Science lanes `fail` | 0 |
| Science lanes `not_run` | 0 |
| fMRI lane mode | real NIfTI fixture parse/load execution (`test_pipeline_real_e2e.sio`) |
| Science lane policy | no `//@ ignore` allowed in `tests/stdlib/fmri` or `tests/stdlib/darwin_pbpk` |
| Science gate `status_summary` | `pass` |
| Runtime regression telemetry | `runtime_regressions` (`literal_as_bytes`, `text_as_bytes`, `binary_as_bytes`, `dynamic_slice`) |
| Runtime regression provenance | `runtime_provenance` (`souc_bin`, `souc_version`, `pinned_version_expected`) |
| Runtime regression enforcement (local default) | `soft` |
| Runtime regression enforcement (required CI) | `STDLIB_RUNTIME_REGRESSION_STRICT=1` in full-gate workflow |

Current telemetry truth:
- strict mode is fail-closed by design; `runtime_regression_summary.status` must be `pass` for strict CI success.

## Hyper Execution Snapshot

| Metric | Value |
|---|---:|
| Hyper tests `pass` | 5 |
| Hyper tests `fail` | 0 |
| Hyper tests `skip` | 0 |
| Hyper tests `total` | 5 |
| Hyper gate `status_summary` | `pass` |
| Hyper no-ignore policy | enforced on required hyper tests |

## GPU Runtime Attestation Snapshot

| Metric | Value |
|---|---:|
| Gate mode (local default) | `auto` |
| Gate mode (required CI) | `required` (`OMEGA_GPU_RUNTIME_GATE_MODE=required`) |
| Status contract | `pass|fail|not_run` with blocker codes |
| Required blocker semantics | any non-pass fails CI in required mode |
| Canonical version source | `scripts/omega/omega_resolve_souc_bin.sh` (`SOUNIO_SOUC_VERSION` override supported) |
| Current local artifact | reflects remote attestation state; non-pass in `auto` records `not_run` with blockers |

## Contract Levels

Use these levels when writing tests/docs for stdlib:

1. `active_callable`
- Module/function is callable under current compiler/runtime behavior.
- Tests should be `//@ run-pass` where possible.

2. `stub_surface`
- Module surface exists (usually `module <name>` in `mod.sio`) but no reliable callable API is exported.
- Tests should be `//@ check-only` contract/surface checks.

3. `disabled_file`
- Implementation exists only as `*.sio.disabled`.
- Tests must not assume callable behavior; use active alternatives or explicit contract adjustments.

## Ignored Test Ledger (Fail-Closed)

Current ignored tests are allowed only with explicit metadata (`reason`, `owner`, `unblock_condition`) and appear in `stdlib_reliability_status.v1.json`:

- `tests/stdlib/causal/test_core.sio`
- `tests/stdlib/core/test_option_result_e2e.sio`
- `tests/stdlib/epistemic/test_causal.sio`
- `tests/stdlib/epistemic/test_stats.sio`
- `tests/stdlib/ffi/test_slice.sio`

## Contract Adjustments

Tests intentionally realigned from aspirational APIs to current active contract are tracked in:

- `stdlib_reliability_status.v1.json` -> `contract_adjustments`

Current adjustments include lanes such as `http`, `json`, `math`, `nn`, `ode`, `optimize`, `prob`, and `signal`.

## Reproduce

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

The gate is fail-closed:
- non-zero exit on any non-ignored E2E fail,
- non-zero exit when hyper gate artifact is missing/malformed/non-pass,
- non-zero exit on missing/malformed status JSON,
- non-zero exit on `not_run`,
- non-zero exit when science gate is missing, malformed, or not `pass`.
