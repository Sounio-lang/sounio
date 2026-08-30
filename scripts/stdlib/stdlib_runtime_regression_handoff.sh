#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# shellcheck source=lib/resolve_souc.sh
source "$ROOT_DIR/scripts/lib/resolve_souc.sh"
sounio_require_souc

PACKET_DIR="${STDLIB_RUNTIME_HANDOFF_DIR:-$ROOT_DIR/artifacts/stdlib/runtime_regression_upstream_handoff}"
OUT_JSON="${STDLIB_RUNTIME_HANDOFF_JSON:-$PACKET_DIR/runtime_regression_upstream_handoff.v1.json}"
OUT_MD="${STDLIB_RUNTIME_HANDOFF_MD:-$PACKET_DIR/runtime_regression_upstream_handoff.v1.md}"

STRICT_SCIENCE_JSON="$PACKET_DIR/stdlib_science_pipeline_status.strict.v1.json"
STRICT_SCIENCE_LOG="$PACKET_DIR/stdlib_science_pipeline_strict.log"
STRICT_RELIABILITY_JSON="$PACKET_DIR/stdlib_reliability_status.strict.v1.json"
STRICT_RELIABILITY_LOG="$PACKET_DIR/stdlib_reliability_strict.log"
STRICT_E2E_JSON="$PACKET_DIR/stdlib_e2e_result.strict.v1.json"
STRICT_INVENTORY_JSON="$PACKET_DIR/stdlib_inventory.strict.v1.json"

mkdir -p "$PACKET_DIR"

probe_files=(
  "$ROOT_DIR/tests/stdlib/runtime_regression/runtime_literal_as_bytes.sio"
  "$ROOT_DIR/tests/stdlib/runtime_regression/runtime_text_as_bytes.sio"
  "$ROOT_DIR/tests/stdlib/runtime_regression/runtime_binary_as_bytes.sio"
  "$ROOT_DIR/tests/stdlib/runtime_regression/runtime_dynamic_slice.sio"
)

for probe in "${probe_files[@]}"; do
  if [[ ! -f "$probe" ]]; then
    echo "error: missing probe source: $probe" >&2
    exit 1
  fi
done

runtime_souc_version="$("$SOUC_BIN" --version 2>/dev/null | awk 'NR==1 {print $2}')"
if [[ -z "$runtime_souc_version" ]]; then
  runtime_souc_version="unknown"
fi

runtime_pinned_expected="$(python3 - "$ROOT_DIR/scripts/omega/omega_resolve_souc_bin.sh" <<'PY'
import re
from pathlib import Path
import sys

path = Path(sys.argv[1])
src = path.read_text(encoding="utf-8", errors="replace")
match = re.search(r'SOUC_VERSION="\$\{SOUNIO_SOUC_VERSION:-([^}]+)\}"', src)
if match:
    print(match.group(1))
PY
)"
if [[ -z "$runtime_pinned_expected" ]]; then
  runtime_pinned_expected="${SOUNIO_SOUC_VERSION:-unknown}"
fi

set +e
STDLIB_RUNTIME_REGRESSION_STRICT=1 \
  STDLIB_SCIENCE_STATUS_OUT="$STRICT_SCIENCE_JSON" \
  bash "$ROOT_DIR/scripts/stdlib_science_pipeline_gate.sh" >"$STRICT_SCIENCE_LOG" 2>&1
strict_science_rc=$?

STDLIB_RUNTIME_REGRESSION_STRICT=1 \
  STDLIB_RELIABILITY_STATUS_OUT="$STRICT_RELIABILITY_JSON" \
  STDLIB_RELIABILITY_E2E_JSON="$STRICT_E2E_JSON" \
  STDLIB_RELIABILITY_INVENTORY_JSON="$STRICT_INVENTORY_JSON" \
  STDLIB_RELIABILITY_SCIENCE_STATUS_JSON="$STRICT_SCIENCE_JSON" \
  bash "$ROOT_DIR/scripts/stdlib_reliability_gate.sh" >"$STRICT_RELIABILITY_LOG" 2>&1
strict_reliability_rc=$?
set -e

python3 - "$ROOT_DIR" "$OUT_JSON" "$OUT_MD" "$PACKET_DIR" \
  "$STRICT_SCIENCE_JSON" "$STRICT_SCIENCE_LOG" "$strict_science_rc" \
  "$STRICT_RELIABILITY_JSON" "$STRICT_RELIABILITY_LOG" "$strict_reliability_rc" \
  "$SOUC_BIN" "$runtime_souc_version" "$runtime_pinned_expected" <<'PY'
import datetime
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
out_json = Path(sys.argv[2]).resolve()
out_md = Path(sys.argv[3]).resolve()
packet_dir = Path(sys.argv[4]).resolve()
strict_science_json = Path(sys.argv[5]).resolve()
strict_science_log = Path(sys.argv[6]).resolve()
strict_science_rc = int(sys.argv[7])
strict_reliability_json = Path(sys.argv[8]).resolve()
strict_reliability_log = Path(sys.argv[9]).resolve()
strict_reliability_rc = int(sys.argv[10])
souc_bin = sys.argv[11]
souc_version = sys.argv[12]
pinned_expected = sys.argv[13]

def rel(path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except Exception:
        return str(path)

def read_json(path: Path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

def read_lines(path: Path, limit: int):
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8", errors="replace").splitlines()[:limit]

science = read_json(strict_science_json)
reliability = read_json(strict_reliability_json)

probe_paths = [
    root / "tests/stdlib/runtime_regression/runtime_literal_as_bytes.sio",
    root / "tests/stdlib/runtime_regression/runtime_text_as_bytes.sio",
    root / "tests/stdlib/runtime_regression/runtime_binary_as_bytes.sio",
    root / "tests/stdlib/runtime_regression/runtime_dynamic_slice.sio",
]

probe_sources = []
for p in probe_paths:
    payload = p.read_bytes()
    probe_sources.append(
        {
            "path": rel(p),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        }
    )

runtime_regressions = science.get("runtime_regressions") if isinstance(science.get("runtime_regressions"), dict) else {}
runtime_summary = science.get("runtime_regression_summary") if isinstance(science.get("runtime_regression_summary"), dict) else {}
runtime_fail = int(runtime_summary.get("fail", 0)) if runtime_summary else 0
science_status = science.get("status_summary", "not_run")

status_summary = "ready" if (strict_science_rc != 0 and science_status == "fail" and runtime_fail > 0) else "unexpected"

obj = {
    "schema": "sounio.stdlib.runtime_regression_upstream_handoff.v1",
    "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
    "command": "bash scripts/stdlib/stdlib_runtime_regression_handoff.sh",
    "status_summary": status_summary,
    "runtime_provenance": {
        "souc_bin": souc_bin,
        "souc_version": souc_version,
        "pinned_version_expected": pinned_expected,
    },
    "probe_sources": probe_sources,
    "strict_science_gate": {
        "command": "STDLIB_RUNTIME_REGRESSION_STRICT=1 bash scripts/stdlib_science_pipeline_gate.sh",
        "exit_code": strict_science_rc,
        "status_summary": science_status,
        "runtime_regression_summary": runtime_summary,
        "runtime_regressions": runtime_regressions,
        "failure_count": len(science.get("failures", [])) if isinstance(science.get("failures"), list) else 0,
    },
    "strict_reliability_gate": {
        "command": "STDLIB_RUNTIME_REGRESSION_STRICT=1 bash scripts/stdlib_reliability_gate.sh",
        "exit_code": strict_reliability_rc,
        "status_summary": reliability.get("status_summary", "not_run"),
        "science_pipeline": reliability.get("science_pipeline", {}),
    },
    "artifacts": {
        "strict_science_status_json": rel(strict_science_json),
        "strict_science_log": rel(strict_science_log),
        "strict_reliability_status_json": rel(strict_reliability_json),
        "strict_reliability_log": rel(strict_reliability_log),
        "handoff_md": rel(out_md),
    },
    "repro_commands": [
        "STDLIB_RUNTIME_REGRESSION_STRICT=1 bash scripts/stdlib_science_pipeline_gate.sh",
        "STDLIB_RUNTIME_REGRESSION_STRICT=1 bash scripts/stdlib_reliability_gate.sh",
    ],
    "strict_science_log_excerpt": read_lines(strict_science_log, 20),
    "strict_reliability_log_excerpt": read_lines(strict_reliability_log, 20),
    "notes": [
        "strict_mode_is_expected_to_fail_closed_until_runtime_engine_fix_is_released",
        "packet_generated_from_committed_runtime_regression_probes",
    ],
}

out_json.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")

probe_lines = "\n".join(
    f"- `{entry['path']}` sha256=`{entry['sha256']}` bytes={entry['bytes']}" for entry in probe_sources
)
runtime_rows = []
for key, row in runtime_regressions.items():
    status = row.get("status", "not_run")
    check_rc = row.get("check_exit_code", 125)
    run_rc = row.get("run_exit_code", 125)
    after = row.get("after_marker_found", False)
    runtime_rows.append(f"| `{key}` | `{status}` | `{check_rc}` | `{run_rc}` | `{after}` |")
runtime_table = "\n".join(runtime_rows) if runtime_rows else "| _none_ | _none_ | _none_ | _none_ | _none_ |"

md = f"""# STDLIB Runtime Regression Upstream Handoff (v1)

## Status
- packet `status_summary`: `{status_summary}`
- strict science gate exit: `{strict_science_rc}`
- strict reliability gate exit: `{strict_reliability_rc}`

## Runtime Provenance
- `souc_bin`: `{souc_bin}`
- `souc_version`: `{souc_version}`
- `pinned_version_expected`: `{pinned_expected}`

## Probe Sources
{probe_lines}

## Strict Runtime Regression Results
| Probe | Status | Check RC | Run RC | After Marker Found |
|---|---:|---:|---:|---:|
{runtime_table}

## Strict Gate Commands
```bash
STDLIB_RUNTIME_REGRESSION_STRICT=1 bash scripts/stdlib_science_pipeline_gate.sh
STDLIB_RUNTIME_REGRESSION_STRICT=1 bash scripts/stdlib_reliability_gate.sh
```

## Artifacts
- `{rel(strict_science_json)}`
- `{rel(strict_science_log)}`
- `{rel(strict_reliability_json)}`
- `{rel(strict_reliability_log)}`
- `{rel(out_json)}`

## Handoff Notes
- strict mode is expected to fail closed until the runtime engine fix lands upstream.
- this packet is intended for upstream runtime/interpreter maintainers.
"""
out_md.write_text(md, encoding="utf-8")
PY

echo "[runtime-handoff] out_json=${OUT_JSON#$ROOT_DIR/}"
echo "[runtime-handoff] out_md=${OUT_MD#$ROOT_DIR/}"
echo "[runtime-handoff] strict_science_json=${STRICT_SCIENCE_JSON#$ROOT_DIR/}"
echo "[runtime-handoff] strict_reliability_json=${STRICT_RELIABILITY_JSON#$ROOT_DIR/}"
