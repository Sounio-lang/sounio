# Sounio STDLIB Reference (Executable Snapshot)

This document tracks the **current executable STDLIB contract** for reliability work.
Values below are sourced from machine-generated artifacts, not aspirational specs.

- Inventory source: `artifacts/stdlib/stdlib_inventory.v1.json`
- Reliability gate source: `artifacts/stdlib/stdlib_reliability_status.v1.json`
- Science gate source: `artifacts/stdlib/stdlib_science_pipeline_status.v1.json`
- Snapshot date: **2026-03-01**

## Reliability Snapshot

| Metric | Value |
|---|---:|
| `sio_files` | 599 |
| `disabled_files` | 120 |
| `stub_mod_files` | 43 |
| `active_module_entrypoints` | 91 |
| E2E `pass` | 67 |
| E2E `fail` | 0 |
| E2E `skip` | 5 |
| E2E `total` | 72 |
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
| Runtime regression telemetry | `runtime_regressions` in science status artifact |
| Runtime regression enforcement (default) | `soft` |
| Runtime regression enforcement (strict) | `STDLIB_RUNTIME_REGRESSION_STRICT=1` |

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
bash scripts/scan_stdlib.sh --json-out artifacts/stdlib/stdlib_inventory.v1.json
bash scripts/stdlib_science_pipeline_gate.sh
STDLIB_RUNTIME_REGRESSION_STRICT=1 bash scripts/stdlib_science_pipeline_gate.sh
bash scripts/stdlib_reliability_gate.sh
STDLIB_RUNTIME_REGRESSION_STRICT=1 bash scripts/stdlib_reliability_gate.sh
```

The gate is fail-closed:
- non-zero exit on any non-ignored E2E fail,
- non-zero exit on missing/malformed status JSON,
- non-zero exit on `not_run`,
- non-zero exit when science gate is missing, malformed, or not `pass`.
