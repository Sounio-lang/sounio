# Standard Library Reference Entry Point

This page is the stable reference entrypoint linked from the repository README.

## Core References

- Executable STDLIB snapshot (inventory + reliability gate): `../STDLIB_REFERENCE.md`
- `Knowledge<T>` and uncertainty usage: `KNOWLEDGE_REFERENCE.md`
- Language specification: `../../spec/LANGUAGE_SPECIFICATION.md`

## API Doc Generation (`souniodoc`)

Generate docs from the repository root:

```bash
cargo run -p souc --bin souniodoc -- generate stdlib --output target/doc
```

This generates browsable API docs for stdlib modules.

## Reliability Gate

Run the fail-closed STDLIB gate from repository root:

```bash
bash scripts/stdlib_reliability_gate.sh
```

Artifacts:
- `artifacts/stdlib/stdlib_reliability_status.v1.json`
- `artifacts/stdlib/stdlib_inventory.v1.json`
- `artifacts/stdlib/stdlib_hyper_execution_status.v1.json`

## Hyper Execution Gate

Run the fail-closed hyper execution gate from repository root:

```bash
bash scripts/stdlib_hyper_execution_gate.sh
```

Artifact:
- `artifacts/stdlib/stdlib_hyper_execution_status.v1.json`

## Science Pipeline Gate

Run the fail-closed scientific pipeline gate from repository root:

```bash
STDLIB_RUNTIME_REGRESSION_STRICT=1 bash scripts/stdlib_science_pipeline_gate.sh
bash scripts/stdlib_science_pipeline_gate.sh
```

Artifacts:
- `artifacts/stdlib/stdlib_science_pipeline_status.v1.json`
- `tests/fixtures/fmri/fixture_manifest.v1.json`
- `tests/fixtures/fmri/pipeline_golden.v1.json`
- `tests/stdlib/runtime_regression/runtime_literal_as_bytes.sio`
- `tests/stdlib/runtime_regression/runtime_text_as_bytes.sio`
- `tests/stdlib/runtime_regression/runtime_binary_as_bytes.sio`
- `tests/stdlib/runtime_regression/runtime_dynamic_slice.sio`

Runtime policy:
- `runtime_regressions` and `runtime_provenance` are always emitted in science status JSON
- local default mode is telemetry (`soft`)
- required CI full gate uses strict mode (`STDLIB_RUNTIME_REGRESSION_STRICT=1`) and hard-fails on runtime regression probes
- current pinned runtime still fails these probes, so strict mode acts as an intentional fail-closed blocker until upstream runtime fixes are released

## GPU Runtime Attestation Gate

Run from repository root:

```bash
OMEGA_GPU_RUNTIME_GATE_MODE=required bash scripts/omega/omega_gpu_runtime_attest_gate.sh
bash scripts/omega/omega_gpu_runtime_attest_gate.sh
```

Artifact:
- `artifacts/omega/gpu_runtime_attest_gate.v1.json`

Mode policy:
- local default is `auto` (remote-unavailable cases are recorded as `not_run`)
- required CI uses `OMEGA_GPU_RUNTIME_GATE_MODE=required` and fails on any non-pass
