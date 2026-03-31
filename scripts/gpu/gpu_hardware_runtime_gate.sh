#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

WORK_DIR="${WORK_DIR:-/tmp/sounio-gpu-hardware-runtime-gate}"
MODE="${GPU_HARDWARE_RUNTIME_MODE:-auto}"
OUT_JSON="${OUT_JSON:-$WORK_DIR/gpu_runtime_attest_gate.v1.json}"
LOG_PATH="${LOG_PATH:-$WORK_DIR/gpu_runtime_attest_gate.log}"

mkdir -p "$WORK_DIR"

write_wrapper_json() {
  local status="$1"
  local reason="$2"
  python3 - "$OUT_JSON" "$MODE" "$status" "$reason" "$LOG_PATH" <<'PY'
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

out_path = Path(sys.argv[1])
mode = sys.argv[2]
status = sys.argv[3]
reason = sys.argv[4]
log_path = sys.argv[5]

payload = {
    "schema": "sounio.gpu.hardware_runtime_gate.v1",
    "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "mode": mode,
    "status_summary": status,
    "reason": reason,
    "delegate_log_path": log_path,
}
out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
}

set +e
OMEGA_GPU_RUNTIME_GATE_MODE="$MODE" \
OMEGA_GPU_RUNTIME_GATE_OUT="$OUT_JSON" \
OMEGA_GPU_RUNTIME_GATE_LOG="$LOG_PATH" \
  bash "$ROOT_DIR/scripts/omega/omega_gpu_runtime_attest_gate.sh" >>"$LOG_PATH" 2>&1
delegate_rc=$?
set -e

if [ "$delegate_rc" -ne 0 ] && [ "$MODE" != "required" ]; then
  write_wrapper_json "not_run" "delegate_prereq_missing_rc_$delegate_rc"
fi

if [ "$delegate_rc" -ne 0 ] && [ "$MODE" = "required" ]; then
  echo "GPU_HARDWARE_RUNTIME_GATE status=fail reason=delegate_failed_rc_$delegate_rc report=$OUT_JSON" >&2
  exit "$delegate_rc"
fi

status="$(python3 - "$OUT_JSON" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    print("missing")
    raise SystemExit(0)
obj = json.loads(path.read_text(encoding="utf-8"))
print(obj.get("status_summary", "missing"))
PY
)"

reason="$(python3 - "$OUT_JSON" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    print("missing_report")
    raise SystemExit(0)
obj = json.loads(path.read_text(encoding="utf-8"))
print(obj.get("reason", "unknown"))
PY
)"

echo "GPU_HARDWARE_RUNTIME_GATE status=$status reason=$reason report=$OUT_JSON"
