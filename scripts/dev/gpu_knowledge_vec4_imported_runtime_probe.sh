#!/usr/bin/env bash
# Probe the imported GPU Knowledge Vec4 lane-plan runtime fixture.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="${SOUNIO_GPU_KNOWLEDGE_IMPORTED_RUNTIME_DIR:-$ROOT_DIR/artifacts/gpu/knowledge_vecmat_evidence_audit/imported_runtime_probe}"
OUT_JSON="$OUT_DIR/gpu_knowledge_vec4_imported_runtime_probe.v1.json"
CHECK_LOG="$OUT_DIR/souc_check.log"
RUN_STDOUT="$OUT_DIR/souc_run.stdout"
RUN_STDERR="$OUT_DIR/souc_run.stderr"
HARNESS="tests/run-pass/gpu_hlir_vec4_lane_plan_imported.sio"

mkdir -p "$OUT_DIR"

write_json() {
  local status="$1"
  local reason="$2"
  local check_exit="$3"
  local run_exit="$4"
  python3 - "$ROOT_DIR" "$OUT_JSON" "$status" "$reason" "$check_exit" "$run_exit" "$HARNESS" "$CHECK_LOG" "$RUN_STDOUT" "$RUN_STDERR" <<'PY'
import hashlib
import json
import pathlib
import sys
from datetime import datetime, timezone

root = pathlib.Path(sys.argv[1])
out_json = pathlib.Path(sys.argv[2])
status, reason = sys.argv[3], sys.argv[4]
check_exit, run_exit = int(sys.argv[5]), int(sys.argv[6])
harness = sys.argv[7]
check_log, stdout_path, stderr_path = map(pathlib.Path, sys.argv[8:11])

def rel(path: pathlib.Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)

def file_entry(path: pathlib.Path) -> dict:
    if not path.exists() or not path.is_file():
        return {"path": rel(path), "present": False}
    data = path.read_bytes()
    return {
        "path": rel(path),
        "present": True,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }

stdout = stdout_path.read_text(encoding="utf-8", errors="replace") if stdout_path.exists() else ""
stderr = stderr_path.read_text(encoding="utf-8", errors="replace") if stderr_path.exists() else ""
check_text = check_log.read_text(encoding="utf-8", errors="replace") if check_log.exists() else ""
payload = {
    "schema": "sounio.gpu-knowledge-vec4-imported-runtime-probe.v1",
    "status": status,
    "reason": reason,
    "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "harness": harness,
    "souc": {
        "check_command": f"./bin/souc check {harness}",
        "check_exit_code": check_exit,
        "run_command": f"./bin/souc run {harness}",
        "run_exit_code": run_exit,
    },
    "runtime_contract": {
        "imported_module": "gpu_hlir_vec4_lane_plan_leaf",
        "copyback_offsets_bytes": [0, 32, 64, 96],
        "expected_value_lanes": [1.0, 2.0, 3.0, 4.0],
        "status": "imported_runtime_pass" if status == "pass" else "missing_or_unproved",
    },
    "artifacts": {
        "check_log": file_entry(check_log),
        "stdout": file_entry(stdout_path),
        "stderr": file_entry(stderr_path),
    },
    "check_log_tail": check_text[-4000:],
    "stdout_tail": stdout[-4000:],
    "stderr_tail": stderr[-4000:],
    "boundaries": [
        "imported_runtime_fixture_only",
        "does_not_claim_general_imported_runtime_correctness",
        "does_not_claim_general_gpu_backend_correctness",
    ],
}
out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

set +e
./bin/souc check "$HARNESS" >"$CHECK_LOG" 2>&1
check_exit=$?
set -e
if [ "$check_exit" -ne 0 ]; then
  : >"$RUN_STDOUT"
  : >"$RUN_STDERR"
  write_json "blocked" "souc_check_failed" "$check_exit" 0
  echo "gpu_knowledge_vec4_imported_runtime_probe: BLOCKED souc_check_failed report=${OUT_JSON#$ROOT_DIR/}"
  exit 0
fi

set +e
./bin/souc run "$HARNESS" >"$RUN_STDOUT" 2>"$RUN_STDERR"
run_exit=$?
set -e

if [ "$run_exit" -eq 0 ] && grep -q "PASS gpu_hlir_vec4_lane_plan_imported" "$RUN_STDOUT"; then
  write_json "pass" "imported_runtime_pass" "$check_exit" "$run_exit"
  echo "gpu_knowledge_vec4_imported_runtime_probe: PASS report=${OUT_JSON#$ROOT_DIR/}"
  exit 0
fi

write_json "blocked" "souc_run_failed_or_missing_pass_marker" "$check_exit" "$run_exit"
echo "gpu_knowledge_vec4_imported_runtime_probe: BLOCKED souc_run_failed_or_missing_pass_marker report=${OUT_JSON#$ROOT_DIR/}"
exit 0
