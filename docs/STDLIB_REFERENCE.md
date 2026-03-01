# Sounio STDLIB Reference (Executable Snapshot)

This document tracks the **current executable STDLIB contract** for reliability work.
Values below are sourced from machine-generated artifacts, not aspirational specs.

- Inventory source: `artifacts/stdlib/stdlib_inventory.v1.json`
- Reliability gate source: `artifacts/stdlib/stdlib_reliability_status.v1.json`
- Snapshot date: **2026-03-01**

## Reliability Snapshot

| Metric | Value |
|---|---:|
| `sio_files` | 593 |
| `disabled_files` | 120 |
| `stub_mod_files` | 43 |
| `active_module_entrypoints` | 91 |
| E2E `pass` | 65 |
| E2E `fail` | 0 |
| E2E `skip` | 5 |
| E2E `total` | 70 |
| Gate `status_summary` | `pass` |

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
bash scripts/stdlib_reliability_gate.sh
```

The gate is fail-closed:
- non-zero exit on any non-ignored E2E fail,
- non-zero exit on missing/malformed status JSON,
- non-zero exit on `not_run`.
