#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SOUC_BIN="${SOUC_BIN:-$ROOT_DIR/souc}"
WASMTIME="${WASMTIME:-$HOME/.wasmtime/bin/wasmtime}"
HLIR2SOIR="${HLIR2SOIR:-$ROOT_DIR/scripts/hlir2soir.py}"
DRIVER="${DRIVER:-$ROOT_DIR/self-hosted/wasm/hlir2wasm_driver.sio}"
OUT_JSON="${SOUNIO_SPRINT3_WASM_GATE_OUT:-$ROOT_DIR/artifacts/sprint3/wasm_gate.v1.json}"
LOG_DIR="${SOUNIO_SPRINT3_WASM_GATE_LOG_DIR:-$ROOT_DIR/artifacts/sprint3/logs/wasm_gate}"
TIMEOUT_SECS="${SOUNIO_SPRINT3_WASM_GATE_TIMEOUT_SECS:-30}"
OUT_DIR="$(mktemp -d)"

trap 'rm -rf "$OUT_DIR"' EXIT

mkdir -p "$(dirname "$OUT_JSON")" "$LOG_DIR"

CASES_TSV="$LOG_DIR/cases.tsv"
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
  local expected="$4"
  local actual="$5"
  printf '%s\t%s\t%s\t%s\t%s\n' "$name" "$status" "$reason" "$expected" "$actual" >> "$CASES_TSV"
}

run_case() {
  local name="$1"
  local sio="$2"
  local expected="$3"
  local log_path="$LOG_DIR/${name}.log"
  local soir="$OUT_DIR/${name}.soir"
  local wasm="$OUT_DIR/${name}.wasm"

  if [ ! -x "$SOUC_BIN" ]; then
    append_case "$name" "not_run" "souc_bin_missing" "$expected" ""
    return
  fi

  if [ ! -f "$sio" ]; then
    append_case "$name" "not_run" "source_missing" "$expected" ""
    return
  fi

  # Generate SOIR via HLIR emit
  set +e
  "$SOUC_BIN" compile --emit hlir "$sio" 2>"$log_path" | python3 "$HLIR2SOIR" > "$soir" 2>>"$log_path"
  local soir_rc=${PIPESTATUS[0]}
  set -e

  if [ "$soir_rc" -ne 0 ] || [ ! -s "$soir" ]; then
    append_case "$name" "fail" "soir_gen_failed" "$expected" ""
    return
  fi

  # Generate WASM
  set +e
  timeout "$TIMEOUT_SECS" "$SOUC_BIN" run "$DRIVER" "$soir" "$wasm" >>"$log_path" 2>&1
  local wasm_rc=$?
  set -e

  if [ "$wasm_rc" -eq 124 ]; then
    append_case "$name" "timeout" "wasm_gen_timeout" "$expected" ""
    return
  fi

  if [ "$wasm_rc" -ne 0 ] || [ ! -s "$wasm" ]; then
    append_case "$name" "fail" "wasm_gen_failed" "$expected" ""
    return
  fi

  # Validate WASM binary (check magic bytes: 0x00 0x61 0x73 0x6D)
  local header
  header="$(od -An -t u1 -N 4 "$wasm" | tr -s ' ' | sed 's/^ //')"
  if [ "$header" != "0 97 115 109" ]; then
    append_case "$name" "fail" "wasm_invalid_header" "$expected" ""
    return
  fi

  # Run with wasmtime — exit code is the i32 truncation of main()'s return value
  if [ ! -x "$WASMTIME" ]; then
    append_case "$name" "not_run" "wasmtime_missing" "$expected" ""
    return
  fi

  set +e
  "$WASMTIME" "$wasm" >>"$log_path" 2>&1
  local actual=$?
  set -e

  if [ "$actual" -eq "$expected" ]; then
    append_case "$name" "pass" "ok" "$expected" "$actual"
  else
    append_case "$name" "fail" "wrong_exit" "$expected" "$actual"
  fi
}

run_stdout_case() {
  local name="$1"
  local sio="$2"
  local expected_stdout="$3"
  local log_path="$LOG_DIR/${name}.log"
  local soir="$OUT_DIR/${name}.soir"
  local wasm="$OUT_DIR/${name}.wasm"

  if [ ! -x "$SOUC_BIN" ] || [ ! -f "$sio" ]; then
    append_case "$name" "not_run" "prerequisites_missing" "" ""
    return
  fi

  set +e
  "$SOUC_BIN" compile --emit hlir "$sio" 2>"$log_path" | python3 "$HLIR2SOIR" > "$soir" 2>>"$log_path"
  local soir_rc=${PIPESTATUS[0]}
  timeout "$TIMEOUT_SECS" "$SOUC_BIN" run "$DRIVER" "$soir" "$wasm" >>"$log_path" 2>&1
  local wasm_rc=$?
  set -e

  if [ "$soir_rc" -ne 0 ] || [ ! -s "$wasm" ]; then
    append_case "$name" "fail" "gen_failed" "" ""
    return
  fi

  if [ ! -x "$WASMTIME" ]; then
    append_case "$name" "not_run" "wasmtime_missing" "" ""
    return
  fi

  set +e
  local actual_stdout
  actual_stdout="$("$WASMTIME" "$wasm" 2>>"$log_path")"
  set -e

  if [ "$actual_stdout" = "$expected_stdout" ]; then
    append_case "$name" "pass" "ok" "$expected_stdout" "$actual_stdout"
  else
    append_case "$name" "fail" "wrong_stdout" "$expected_stdout" "$actual_stdout"
  fi
}

run_case "add"      "$ROOT_DIR/tests/wasm/add.sio"      7
run_case "fib"      "$ROOT_DIR/tests/wasm/fib.sio"      55
run_case "mul"      "$ROOT_DIR/tests/wasm/mul.sio"      42
run_case "nested"   "$ROOT_DIR/tests/wasm/nested.sio"   12
run_stdout_case "print_fib"  "$ROOT_DIR/tests/wasm/print_fib.sio"  "55"

python3 - "$OUT_JSON" "$CASES_TSV" "$TIMEOUT_SECS" "$SOUC_BIN" "$WASMTIME" <<'PY'
import datetime as dt
import json
from pathlib import Path
import sys

out_json = Path(sys.argv[1])
cases_tsv = Path(sys.argv[2])
timeout_secs = int(sys.argv[3])
souc_bin = sys.argv[4]
wasmtime_bin = sys.argv[5]

cases = []
counts = {"pass": 0, "fail": 0, "timeout": 0, "not_run": 0}

for raw in cases_tsv.read_text(encoding="utf-8").splitlines():
    if not raw.strip():
        continue
    name, status, reason, expected, actual = raw.split("\t")
    if status not in counts:
        status = "fail"
        reason = "invalid_status_token"
    counts[status] += 1

    item = {
        "name": name,
        "status": status,
        "reason": reason,
        "expected_exit": int(expected) if expected.strip() else None,
        "actual_exit": int(actual) if actual.strip() else None,
    }
    cases.append(item)

overall_status = "pass"
overall_reason = "all_cases_passed"
if counts["fail"] > 0:
    overall_status = "fail"
    overall_reason = f"{counts['fail']}_cases_failed"
elif counts["timeout"] > 0:
    overall_status = "timeout"
    overall_reason = "one_or_more_cases_timed_out"
elif counts["not_run"] > 0:
    overall_status = "not_run"
    overall_reason = "one_or_more_cases_not_run"

payload = {
    "schema": "sounio.sprint3.wasm_gate.v1",
    "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    "status": overall_status,
    "reason": overall_reason,
    "config": {
        "souc_bin": souc_bin,
        "wasmtime_bin": wasmtime_bin,
        "timeout_seconds": timeout_secs,
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
import json, sys
obj = json.load(open(sys.argv[1], "r", encoding="utf-8"))
print(obj.get("status", "fail"))
PY
)"

echo "sprint3_wasm_gate: out_json=$(to_rel "$OUT_JSON") status=$status"
if [ "$status" = "pass" ]; then
  exit 0
fi
exit 2
