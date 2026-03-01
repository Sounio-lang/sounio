# STDLIB Module Organization (Executable View)

This page describes how STDLIB modules are organized for current reliability
work, using repository-generated inventory artifacts as ground truth.

Snapshot date: **2026-03-01**

## File Surface Model

STDLIB uses a mixed entrypoint structure:

- `lib.sio`: primary module API surface where present
- `mod.sio`: module entrypoint or compatibility surface
- `*.sio.disabled`: intentionally disabled implementation files

Current inventory (`artifacts/stdlib/stdlib_inventory.v1.json`):

- `sio_files`: 595
- `disabled_files`: 120
- `mod_files`: 70
- `lib_files`: 64
- `module_entrypoints`: 134
- `stub_mod_files`: 43
- `active_module_entrypoints`: 91

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

## Reliability Workflow

From repository root:

```bash
bash scripts/scan_stdlib.sh --json-out artifacts/stdlib/stdlib_inventory.v1.json
bash scripts/stdlib_science_pipeline_gate.sh
bash scripts/stdlib_reliability_gate.sh
```

Primary status artifact:
- `artifacts/stdlib/stdlib_reliability_status.v1.json`
- `artifacts/stdlib/stdlib_science_pipeline_status.v1.json`

Primary reference page:
- `docs/STDLIB_REFERENCE.md`
