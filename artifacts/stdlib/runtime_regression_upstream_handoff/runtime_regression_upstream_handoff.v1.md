# STDLIB Runtime Regression Upstream Handoff (v1)

## Status
- packet `status_summary`: `ready`
- strict science gate exit: `1`
- strict reliability gate exit: `1`

## Runtime Provenance
- `souc_bin`: `/home/demetrios/work/sounio/artifacts/omega/souc-bin/souc-linux-x86_64`
- `souc_version`: `1.0.0-beta.4`
- `pinned_version_expected`: `0.100.3`

## Probe Sources
- `tests/stdlib/runtime_regression/runtime_literal_as_bytes.sio` sha256=`1b65d443d57dc8613168aad08c2d4914c29f32de8429a614efafe55082c1dbbd` bytes=264
- `tests/stdlib/runtime_regression/runtime_text_as_bytes.sio` sha256=`efba85f13dabb731e162d3bd09bfcab01fd19d71fd7014dd2757004e30ed07f1` bytes=311
- `tests/stdlib/runtime_regression/runtime_binary_as_bytes.sio` sha256=`548b59f6433c7835dcf934b87b92259ce3da3d6b5f3a51ad25046f01f34ae5cb` bytes=323
- `tests/stdlib/runtime_regression/runtime_dynamic_slice.sio` sha256=`f28b7e3db694298a90d7bb8b92722749e77a8f93ea9190b593f4178cc9c8a9b3` bytes=415

## Strict Runtime Regression Results
| Probe | Status | Check RC | Run RC | After Marker Found |
|---|---:|---:|---:|---:|
| `literal_as_bytes` | `fail` | `0` | `0` | `False` |
| `text_as_bytes` | `fail` | `0` | `0` | `False` |
| `binary_as_bytes` | `fail` | `0` | `0` | `False` |
| `dynamic_slice` | `fail` | `0` | `0` | `False` |

## Strict Gate Commands
```bash
STDLIB_RUNTIME_REGRESSION_STRICT=1 bash scripts/stdlib_science_pipeline_gate.sh
STDLIB_RUNTIME_REGRESSION_STRICT=1 bash scripts/stdlib_reliability_gate.sh
```

## Artifacts
- `artifacts/stdlib/runtime_regression_upstream_handoff/stdlib_science_pipeline_status.strict.v1.json`
- `artifacts/stdlib/runtime_regression_upstream_handoff/stdlib_science_pipeline_strict.log`
- `artifacts/stdlib/runtime_regression_upstream_handoff/stdlib_reliability_status.strict.v1.json`
- `artifacts/stdlib/runtime_regression_upstream_handoff/stdlib_reliability_strict.log`
- `artifacts/stdlib/runtime_regression_upstream_handoff/runtime_regression_upstream_handoff.v1.json`

## Handoff Notes
- strict mode is expected to fail closed until the runtime engine fix lands upstream.
- this packet is intended for upstream runtime/interpreter maintainers.
