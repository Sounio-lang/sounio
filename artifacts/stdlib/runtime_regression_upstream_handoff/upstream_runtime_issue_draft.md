# Upstream Runtime Bug Report Draft: `as_bytes()` / Dynamic Slice Early-Stop with Exit Code 0

## Title
Runtime regression: `.as_bytes()` and dynamic slice paths stop execution after start marker (missing after-marker) while process exits `0`

## Summary
We have a reproducible runtime regression where these lanes:
- string literal `.as_bytes()`
- `read_file(text).as_bytes()`
- `read_file(binary).as_bytes()`
- dynamic slice (`&bytes[..n as usize]`)

all show the same failure signature:
- `check` succeeds (`rc=0`)
- `run` returns `rc=0`
- execution output contains the `*_START` marker
- output never reaches the `*_AFTER` marker

This creates a silent runtime-stop class (false success via exit code), and blocks strict fail-closed science/reliability cutover.

## Reproduction
From repo root:

```bash
STDLIB_RUNTIME_REGRESSION_STRICT=1 bash scripts/stdlib_science_pipeline_gate.sh
STDLIB_RUNTIME_REGRESSION_STRICT=1 bash scripts/stdlib_reliability_gate.sh
```

Target probe sources:
- `tests/stdlib/runtime_regression/runtime_literal_as_bytes.sio`
- `tests/stdlib/runtime_regression/runtime_text_as_bytes.sio`
- `tests/stdlib/runtime_regression/runtime_binary_as_bytes.sio`
- `tests/stdlib/runtime_regression/runtime_dynamic_slice.sio`

## Observed Behavior (strict packet)
- strict science gate: `exit_code=1`, `status_summary=fail`
- strict reliability gate: `exit_code=1`, `status_summary=fail`
- probe-level results:
  - `literal_as_bytes`: `check_rc=0 run_rc=0 after_marker_found=false`
  - `text_as_bytes`: `check_rc=0 run_rc=0 after_marker_found=false`
  - `binary_as_bytes`: `check_rc=0 run_rc=0 after_marker_found=false`
  - `dynamic_slice`: `check_rc=0 run_rc=0 after_marker_found=false`

## Expected Behavior
- No early-stop in these runtime lanes.
- `*_AFTER` marker is always printed for successful execution.
- Runtime returns non-zero on invalid runtime states (no silent success).

## Runtime Provenance (captured in packet)
- `souc_bin`: `/home/demetrios/work/sounio/artifacts/omega/souc-bin/souc-linux-x86_64`
- `souc_version`: `1.0.0-beta.4`
- `pinned_version_expected`: `0.100.3`

## Attached Evidence Artifacts
- `artifacts/stdlib/runtime_regression_upstream_handoff/runtime_regression_upstream_handoff.v1.json`
- `artifacts/stdlib/runtime_regression_upstream_handoff/runtime_regression_upstream_handoff.v1.md`
- `artifacts/stdlib/runtime_regression_upstream_handoff/stdlib_science_pipeline_status.strict.v1.json`
- `artifacts/stdlib/runtime_regression_upstream_handoff/stdlib_reliability_status.strict.v1.json`
- `artifacts/stdlib/runtime_regression_upstream_handoff/stdlib_science_pipeline_strict.log`
- `artifacts/stdlib/runtime_regression_upstream_handoff/stdlib_reliability_strict.log`

## Acceptance Criteria for Fix
- All four runtime probes print `*_AFTER` marker and keep `run_rc=0`.
- Strict science gate passes:
  - `runtime_regression_summary.status == "pass"`
  - `runtime_regression_summary.fail == 0`
- Strict reliability gate passes when strict mode is enabled.
- No regression in existing fMRI and Darwin PBPK science lanes.

