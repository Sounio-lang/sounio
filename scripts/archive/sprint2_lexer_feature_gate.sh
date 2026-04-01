#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SOUC_BIN="${SOUC_BIN:-$ROOT_DIR/souc}"
OUT_JSON="${SOUNIO_SPRINT2_LEXER_FEATURE_OUT:-$ROOT_DIR/artifacts/sprint2/lexer_feature_gate.v1.json}"
LOG_DIR="${SOUNIO_SPRINT2_LEXER_FEATURE_LOG_DIR:-$ROOT_DIR/artifacts/sprint2/logs/lexer_feature}"
TIMEOUT_SECS="${SOUNIO_SPRINT2_LEXER_FEATURE_TIMEOUT_SECS:-30}"

SOURCE_FILE="$ROOT_DIR/self-hosted/compiler/lexer_feature_probe.sio"
CASES_TSV="$LOG_DIR/cases.tsv"
mkdir -p "$(dirname "$OUT_JSON")" "$LOG_DIR"
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
  printf '%s\t%s\t%s\t%s\t%s\n' "$name" "$status" "$reason" "$rc" "$(to_rel "$log_path")" >> "$CASES_TSV"
}

run_cmd_case() {
  local name="$1"
  local log_path="$2"
  shift 2

  if [ ! -x "$SOUC_BIN" ]; then
    append_case "$name" "not_run" "souc_bin_missing" "-1" "$log_path"
    return
  fi

  set +e
  timeout "$TIMEOUT_SECS" "$@" >"$log_path" 2>&1
  local rc=$?
  set -e

  if [ "$rc" -eq 0 ]; then
    append_case "$name" "pass" "ok" "$rc" "$log_path"
  elif [ "$rc" -eq 124 ]; then
    append_case "$name" "timeout" "timeout" "$rc" "$log_path"
  else
    append_case "$name" "fail" "command_failed" "$rc" "$log_path"
  fi
}

if [ ! -f "$SOURCE_FILE" ]; then
  append_case "compiler_lexer_feature_probe" "not_run" "probe_source_missing" "-1" "$LOG_DIR/compiler_lexer_feature_probe.log"
else
  run_cmd_case "compiler_lexer_check" "$LOG_DIR/compiler_lexer_check.log" "$SOUC_BIN" check self-hosted/compiler/lexer.sio
  run_cmd_case "compiler_lexer_feature_probe" "$LOG_DIR/compiler_lexer_feature_probe.log" "$SOUC_BIN" run self-hosted/compiler/lexer_feature_probe.sio
fi

python3 - "$OUT_JSON" "$CASES_TSV" "$ROOT_DIR" "$TIMEOUT_SECS" "$SOUC_BIN" <<'PY'
import datetime as dt
import json
from pathlib import Path
import sys

out_json = Path(sys.argv[1])
cases_tsv = Path(sys.argv[2])
root = Path(sys.argv[3]).resolve()
timeout_secs = int(sys.argv[4])
souc_bin = sys.argv[5]

cases = []
counts = {"pass": 0, "fail": 0, "timeout": 0, "not_run": 0}
for raw in cases_tsv.read_text(encoding="utf-8").splitlines():
    if not raw.strip():
        continue
    name, status, reason, rc, log_path = raw.split("\t")
    if status not in counts:
        status = "fail"
        reason = "invalid_status_token"
    counts[status] += 1
    cases.append(
        {
            "name": name,
            "status": status,
            "reason": reason,
            "exit_code": int(rc),
            "log_path": log_path,
        }
    )

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
    "schema": "sounio.sprint2.lexer_feature_gate.v1",
    "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    "status": overall_status,
    "reason": overall_reason,
    "config": {
        "souc_bin": souc_bin,
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
import json
import sys
obj = json.load(open(sys.argv[1], "r", encoding="utf-8"))
print(obj.get("status", "fail"))
PY
)"

echo "lexer_feature_gate: out_json=$(to_rel "$OUT_JSON") status=$status"
if [ "$status" = "pass" ]; then
  exit 0
fi
exit 2
