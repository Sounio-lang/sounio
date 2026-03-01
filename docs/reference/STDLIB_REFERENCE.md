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

## Science Pipeline Gate

Run the fail-closed scientific pipeline gate from repository root:

```bash
bash scripts/stdlib_science_pipeline_gate.sh
```

Artifacts:
- `artifacts/stdlib/stdlib_science_pipeline_status.v1.json`
- `tests/fixtures/fmri/fixture_manifest.v1.json`
- `tests/fixtures/fmri/pipeline_golden.v1.json`
