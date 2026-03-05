#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SOUC_BIN="${SOUC_BIN:-$ROOT_DIR/souc}"
OUT_JSON="${SOUNIO_SPRINT2_IR_LOWER_OUT:-$ROOT_DIR/artifacts/sprint2/ir_lower_timeout_gate.v1.json}"
LOG_DIR="${SOUNIO_SPRINT2_IR_LOWER_LOG_DIR:-$ROOT_DIR/artifacts/sprint2/logs/ir_lower_timeout}"
TIMEOUT_SECS="${SOUNIO_SPRINT2_IR_LOWER_TIMEOUT_SECS:-120}"

SOURCE_FILE="${SOUNIO_SPRINT2_IR_LOWER_SOURCE:-tests/frontend/minimal_pipeline_i64.sio}"
NATIVE_OUT="${SOUNIO_SPRINT2_IR_LOWER_NATIVE_OUT:-$ROOT_DIR/artifacts/sprint2/ir_lower_timeout/minimal_native.out}"
CASES_TSV="$LOG_DIR/cases.tsv"

mkdir -p "$(dirname "$OUT_JSON")" "$LOG_DIR" "$(dirname "$NATIVE_OUT")"
: > "$CASES_TSV"

to_rel() {
  local p="$1"
  if [[ "$p" == "$ROOT_DIR/"* ]]; then
    printf '%s' "${p#$ROOT_DIR/}"
  else
    printf '%s' "$p"
  fi
}

append_case() {
  local name="$1"
  local status="$2"
  local reason="$3"
  local rc="$4"
  local log_path="$5"
  local markers="$6"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$name" "$status" "$reason" "$rc" "$(to_rel "$log_path")" "$markers" >> "$CASES_TSV"
}

run_cmd_case() {
  local name="$1"
  local log_path="$2"
  shift 2

  if [ ! -x "$SOUC_BIN" ]; then
    append_case "$name" "not_run" "souc_bin_missing" "-1" "$log_path" "{}"
    return
  fi

  set +e
  timeout "$TIMEOUT_SECS" "$@" >"$log_path" 2>&1
  local rc=$?
  set -e

  if [ "$rc" -eq 0 ]; then
    append_case "$name" "pass" "ok" "$rc" "$log_path" "{}"
  elif [ "$rc" -eq 124 ]; then
    append_case "$name" "timeout" "timeout" "$rc" "$log_path" "{}"
  else
    append_case "$name" "fail" "command_failed" "$rc" "$log_path" "{}"
  fi
}

marker_bool() {
  local needle="$1"
  local log_path="$2"
  if rg -F -q "$needle" "$log_path"; then
    printf '1'
  else
    printf '0'
  fi
}

run_ir_lower_probe_case() {
  local name="$1"
  local log_path="$2"

  if [ ! -x "$SOUC_BIN" ]; then
    append_case "$name" "not_run" "souc_bin_missing" "-1" "$log_path" "{}"
    return
  fi

  if [ ! -f "$SOURCE_FILE" ]; then
    append_case "$name" "not_run" "source_missing" "-1" "$log_path" "{}"
    return
  fi

  rm -f "$NATIVE_OUT"

  set +e
  timeout "$TIMEOUT_SECS" \
    "$SOUC_BIN" run self-hosted/compiler/main.sio -- souc "$SOURCE_FILE" -o "$NATIVE_OUT" -t native -v \
    >"$log_path" 2>&1
  local rc=$?
  set -e

  local type_start
  type_start="$(marker_bool "Type checking module" "$log_path")"
  local type_done
  type_done="$(marker_bool "Type check complete" "$log_path")"
  local ir_start
  ir_start="$(marker_bool "IR lowering module" "$log_path")"
  local ir_done
  ir_done="$(marker_bool "IR lowering complete" "$log_path")"
  local merged_seen
  merged_seen="$(marker_bool "Merged IR:" "$log_path")"
  local native_dispatch_seen
  native_dispatch_seen="$(marker_bool "Native dispatch:" "$log_path")"
  local native_bin_seen
  native_bin_seen="$(marker_bool "Native binary size:" "$log_path")"
  local written_seen
  written_seen="$(marker_bool "Written to " "$log_path")"

  local markers
  markers="{\"type_start\":$type_start,\"type_done\":$type_done,\"ir_start\":$ir_start,\"ir_done\":$ir_done,\"merged_seen\":$merged_seen,\"native_dispatch_seen\":$native_dispatch_seen,\"native_bin_seen\":$native_bin_seen,\"written_seen\":$written_seen}"

  if [ "$rc" -eq 124 ]; then
    local timeout_reason="timeout"
    if [ "$ir_start" -eq 0 ]; then
      timeout_reason="timeout_pre_ir_lower"
    elif [ "$ir_done" -eq 0 ]; then
      timeout_reason="timeout_in_ir_lower"
    elif [ "$native_dispatch_seen" -eq 0 ]; then
      timeout_reason="timeout_post_ir_lower_pre_dispatch"
    else
      timeout_reason="timeout_post_ir_lower_native_codegen"
    fi
    append_case "$name" "timeout" "$timeout_reason" "$rc" "$log_path" "$markers"
    return
  fi

  if [ "$rc" -ne 0 ]; then
    append_case "$name" "fail" "command_failed" "$rc" "$log_path" "$markers"
    return
  fi

  if [ "$ir_done" -eq 0 ]; then
    append_case "$name" "fail" "missing_ir_lower_complete_marker" "$rc" "$log_path" "$markers"
    return
  fi

  if [ "$native_dispatch_seen" -eq 0 ]; then
    append_case "$name" "fail" "missing_native_dispatch_marker" "$rc" "$log_path" "$markers"
    return
  fi

  append_case "$name" "pass" "ok" "$rc" "$log_path" "$markers"
}

run_cmd_case "compiler_main_check" "$LOG_DIR/compiler_main_check.log" "$SOUC_BIN" check self-hosted/compiler/main.sio
run_cmd_case "compiler_module_loader_check" "$LOG_DIR/compiler_module_loader_check.log" "$SOUC_BIN" check self-hosted/compiler/module_loader.sio
run_ir_lower_probe_case "ir_lower_native_probe" "$LOG_DIR/ir_lower_native_probe.log"

python3 - "$OUT_JSON" "$CASES_TSV" "$TIMEOUT_SECS" "$SOUC_BIN" "$SOURCE_FILE" "$NATIVE_OUT" <<'PY'
import datetime as dt
import json
from pathlib import Path
import sys

out_json = Path(sys.argv[1])
cases_tsv = Path(sys.argv[2])
timeout_secs = int(sys.argv[3])
souc_bin = sys.argv[4]
source_file = sys.argv[5]
native_out = sys.argv[6]

cases = []
counts = {"pass": 0, "fail": 0, "timeout": 0, "not_run": 0}

for raw in cases_tsv.read_text(encoding="utf-8").splitlines():
    if not raw.strip():
        continue
    name, status, reason, rc, log_path, markers_raw = raw.split("\t")
    if status not in counts:
        status = "fail"
        reason = "invalid_status_token"
    counts[status] += 1

    item = {
        "name": name,
        "status": status,
        "reason": reason,
        "exit_code": int(rc),
        "log_path": log_path,
    }
    if name == "ir_lower_native_probe":
        try:
            item["markers"] = json.loads(markers_raw)
        except Exception:
            item["markers"] = {}
        item["native_output"] = native_out
    cases.append(item)

overall_status = "pass"
overall_reason = "all_cases_passed"
if counts["fail"] > 0:
    overall_status = "fail"
    overall_reason = "one_or_more_cases_failed"
elif counts["timeout"] > 0:
    overall_status = "timeout"
    overall_reason = "one_or_more_cases_timed_out"
elif counts["not_run"] > 0:
    overall_status = "not_run"
    overall_reason = "one_or_more_cases_not_run"

payload = {
    "schema": "sounio.sprint2.ir_lower_timeout_gate.v1",
    "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    "status": overall_status,
    "reason": overall_reason,
    "config": {
        "souc_bin": souc_bin,
        "timeout_seconds": timeout_secs,
        "source_file": source_file,
        "native_output": native_out,
    },
    "metrics": {
        "total": len(cases),
        "passed": counts["pass"],
        "failed": counts["fail"],
        "timeout": counts["timeout"],
        "not_run": counts["not_run"],
    },
    "cases": cases,
}

out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(f"wrote {out_json}")
print(f"status={overall_status} reason={overall_reason}")
PY

status="$(python3 - "$OUT_JSON" <<'PY'
import json
import sys
obj = json.load(open(sys.argv[1], "r", encoding="utf-8"))
print(obj.get("status", "fail"))
PY
)"

echo "ir_lower_timeout_gate: out_json=$(to_rel "$OUT_JSON") status=$status"
if [ "$status" = "pass" ]; then
  exit 0
fi
exit 2
