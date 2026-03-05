#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SOUC_BIN="${SOUC_BIN:-$ROOT_DIR/souc}"
OUT_JSON="${SOUNIO_SPRINT2_WASM_DISPATCH_OUT:-$ROOT_DIR/artifacts/sprint2/wasm_dispatch_gate.v1.json}"
LOG_DIR="${SOUNIO_SPRINT2_WASM_DISPATCH_LOG_DIR:-$ROOT_DIR/artifacts/sprint2/logs/wasm_dispatch}"
TIMEOUT_SECS="${SOUNIO_SPRINT2_WASM_DISPATCH_TIMEOUT_SECS:-90}"

SOURCE_FILE="${SOUNIO_SPRINT2_WASM_SOURCE:-tests/frontend/minimal_passing.sio}"
WASM_OUT="${SOUNIO_SPRINT2_WASM_OUT:-$ROOT_DIR/artifacts/sprint2/wasm_dispatch/minimal_passing.wasm}"
CASES_TSV="$LOG_DIR/cases.tsv"

mkdir -p "$(dirname "$OUT_JSON")" "$LOG_DIR" "$(dirname "$WASM_OUT")"
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
  local wasm_output="$6"
  local wasm_size="$7"
  local wasm_header_valid="$8"
  local dispatch_seen="$9"
  local fail_closed_seen="${10}"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$name" "$status" "$reason" "$rc" "$(to_rel "$log_path")" \
    "$(to_rel "$wasm_output")" "$wasm_size" "$wasm_header_valid" "$dispatch_seen" "$fail_closed_seen" \
    >> "$CASES_TSV"
}

run_cmd_case() {
  local name="$1"
  local log_path="$2"
  shift 2

  if [ ! -x "$SOUC_BIN" ]; then
    append_case "$name" "not_run" "souc_bin_missing" "-1" "$log_path" "" "" "" "0" "0"
    return
  fi

  set +e
  timeout "$TIMEOUT_SECS" "$@" >"$log_path" 2>&1
  local rc=$?
  set -e

  if [ "$rc" -eq 0 ]; then
    append_case "$name" "pass" "ok" "$rc" "$log_path" "" "" "" "0" "0"
  elif [ "$rc" -eq 124 ]; then
    append_case "$name" "timeout" "timeout" "$rc" "$log_path" "" "" "" "0" "0"
  else
    append_case "$name" "fail" "command_failed" "$rc" "$log_path" "" "" "" "0" "0"
  fi
}

run_wasm_dispatch_case() {
  local name="$1"
  local log_path="$2"

  if [ ! -x "$SOUC_BIN" ]; then
    append_case "$name" "not_run" "souc_bin_missing" "-1" "$log_path" "$WASM_OUT" "" "" "0" "0"
    return
  fi

  if [ ! -f "$SOURCE_FILE" ]; then
    append_case "$name" "not_run" "source_missing" "-1" "$log_path" "$WASM_OUT" "" "" "0" "0"
    return
  fi

  rm -f "$WASM_OUT"

  set +e
  timeout "$TIMEOUT_SECS" \
    "$SOUC_BIN" run self-hosted/compiler/main.sio -- --probe-wasm-dispatch "$SOURCE_FILE" "$WASM_OUT" \
    >"$log_path" 2>&1
  local rc=$?
  set -e

  local dispatch_seen=0
  local fail_closed_seen=0
  if rg -F -q "WASM dispatch:" "$log_path"; then
    dispatch_seen=1
  fi
  if rg -F -q "WASM backend unavailable in self-hosted module_loader (temporary fail-closed)." "$log_path"; then
    fail_closed_seen=1
  fi

  if [ "$rc" -eq 124 ]; then
    local timeout_reason="timeout"
    if [ "$dispatch_seen" -eq 0 ]; then
      timeout_reason="timeout_pre_dispatch"
    else
      timeout_reason="timeout_post_dispatch"
    fi
    append_case "$name" "timeout" "$timeout_reason" "$rc" "$log_path" "$WASM_OUT" "" "" "$dispatch_seen" "$fail_closed_seen"
    return
  fi

  if [ "$rc" -ne 0 ]; then
    local reason="command_failed"
    if [ "$fail_closed_seen" -eq 1 ]; then
      reason="fail_closed"
    fi
    append_case "$name" "fail" "$reason" "$rc" "$log_path" "$WASM_OUT" "" "" "$dispatch_seen" "$fail_closed_seen"
    return
  fi

  if [ "$dispatch_seen" -eq 0 ]; then
    append_case "$name" "fail" "missing_dispatch_marker" "$rc" "$log_path" "$WASM_OUT" "" "" "$dispatch_seen" "$fail_closed_seen"
    return
  fi

  if [ "$fail_closed_seen" -eq 1 ]; then
    append_case "$name" "fail" "fail_closed" "$rc" "$log_path" "$WASM_OUT" "" "" "$dispatch_seen" "$fail_closed_seen"
    return
  fi

  if [ ! -s "$WASM_OUT" ]; then
    append_case "$name" "pass" "dispatch_probe_ok" "$rc" "$log_path" "$WASM_OUT" "" "" "$dispatch_seen" "$fail_closed_seen"
    return
  fi

  local wasm_size
  wasm_size="$(wc -c < "$WASM_OUT")"
  local header
  header="$(od -An -t u1 -N 4 "$WASM_OUT" | tr -s ' ' | sed 's/^ //')"
  local header_valid=0
  if [ "$header" = "0 97 115 109" ]; then
    header_valid=1
    append_case "$name" "pass" "ok" "$rc" "$log_path" "$WASM_OUT" "$wasm_size" "$header_valid" "$dispatch_seen" "$fail_closed_seen"
  else
    append_case "$name" "fail" "invalid_wasm_header" "$rc" "$log_path" "$WASM_OUT" "$wasm_size" "$header_valid" "$dispatch_seen" "$fail_closed_seen"
  fi
}

run_cmd_case "compiler_main_check" "$LOG_DIR/compiler_main_check.log" "$SOUC_BIN" check self-hosted/compiler/main.sio
run_cmd_case "compiler_module_loader_check" "$LOG_DIR/compiler_module_loader_check.log" "$SOUC_BIN" check self-hosted/compiler/module_loader.sio
run_wasm_dispatch_case "wasm_dispatch_compile" "$LOG_DIR/wasm_dispatch_compile.log"

python3 - "$OUT_JSON" "$CASES_TSV" "$TIMEOUT_SECS" "$SOUC_BIN" "$SOURCE_FILE" "$WASM_OUT" <<'PY'
import datetime as dt
import json
from pathlib import Path
import sys

out_json = Path(sys.argv[1])
cases_tsv = Path(sys.argv[2])
timeout_secs = int(sys.argv[3])
souc_bin = sys.argv[4]
source_file = sys.argv[5]
wasm_out = sys.argv[6]

cases = []
counts = {"pass": 0, "fail": 0, "timeout": 0, "not_run": 0}

for raw in cases_tsv.read_text(encoding="utf-8").splitlines():
    if not raw.strip():
        continue
    (
        name,
        status,
        reason,
        rc,
        log_path,
        wasm_output,
        wasm_size,
        wasm_header_valid,
        dispatch_seen,
        fail_closed_seen,
    ) = raw.split("\t")
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
    if name == "wasm_dispatch_compile":
        item["wasm_output"] = wasm_output
        item["wasm_size_bytes"] = int(wasm_size) if wasm_size.strip() else None
        item["wasm_header_valid"] = (wasm_header_valid == "1") if wasm_header_valid.strip() else None
        item["dispatch_marker_seen"] = dispatch_seen == "1"
        item["fail_closed_marker_seen"] = fail_closed_seen == "1"
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
    "schema": "sounio.sprint2.wasm_dispatch_gate.v1",
    "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    "status": overall_status,
    "reason": overall_reason,
    "config": {
        "souc_bin": souc_bin,
        "timeout_seconds": timeout_secs,
        "source_file": source_file,
        "wasm_output": wasm_out,
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

echo "wasm_dispatch_gate: out_json=$(to_rel "$OUT_JSON") status=$status"
if [ "$status" = "pass" ]; then
  exit 0
fi
exit 2
