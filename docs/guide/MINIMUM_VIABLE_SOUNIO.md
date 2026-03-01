# Minimum Viable Sounio (Current Contract)

This guide is intentionally conservative: it describes what is currently
validated by repository gates, not long-term roadmap intent.

Snapshot date: **2026-03-01**

## What Is Reliable Right Now

1. Compiler/runtime path needed to execute `tests/stdlib` through:
- `bash scripts/run_stdlib_e2e.sh`
- `bash scripts/stdlib_reliability_gate.sh`

2. STDLIB reliability lane:
- E2E totals: `pass=65 fail=0 skip=5 total=70`
- Gate status: `status_summary=pass`
- Artifact: `artifacts/stdlib/stdlib_reliability_status.v1.json`

3. Module/test workflow:
- `use`-based imports work for currently active module surfaces.
- Not every module path is callable; some are stubs or disabled files.

## STDLIB Contract Levels

Use these levels when deciding how to test module behavior:

1. `active_callable`
- Callable APIs verified with `//@ run-pass` tests.

2. `stub_surface`
- Module entrypoint exists but callable API is not reliable/complete.
- Use `//@ check-only` surface tests.

3. `disabled_file`
- Implementation appears as `*.sio.disabled`.
- Do not write callable expectations against that lane.

## Important Clarification

The module system is available for active surfaces, but should not be treated as
"all stdlib modules are fully implemented." Reliability now depends on current
active exports, and the gate is the source of truth.

## How To Verify Before Claiming Support

Run from repository root:

```bash
bash scripts/scan_stdlib.sh --json-out artifacts/stdlib/stdlib_inventory.v1.json
bash scripts/run_stdlib_e2e.sh
bash scripts/stdlib_reliability_gate.sh
```

Then read:
- `artifacts/stdlib/stdlib_inventory.v1.json`
- `artifacts/stdlib/stdlib_reliability_status.v1.json`

If the gate is not `pass`, treat the affected lanes as not reliable.
