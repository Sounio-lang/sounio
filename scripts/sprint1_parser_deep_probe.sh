#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUT_JSON="${1:-$ROOT_DIR/artifacts/sprint1/reports/parser_golden_deep_probe.v1.json}"
LOG_DIR="${SOUNIO_SPRINT1_PARSER_DEEP_LOG_DIR:-$ROOT_DIR/artifacts/sprint1/logs/parser_deep}"
SOUC_BIN="${SOUC_BIN:-./souc}"
TIMEOUT_SECS="${SOUNIO_SPRINT1_PARSER_DEEP_TIMEOUT_SECS:-8}"

mkdir -p "$(dirname "$OUT_JSON")" "$LOG_DIR"

if [ ! -x "$SOUC_BIN" ]; then
  python3 - "$OUT_JSON" <<'PY'
import datetime as dt
import json
import sys

out_path = sys.argv[1]
payload = {
    "schema": "sounio.sprint1.parser_deep_probe.v1",
    "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    "status": "not_run",
    "reason": "souc_bin_missing",
    "timeout_seconds": 0,
    "cases": [],
}
with open(out_path, "w", encoding="utf-8") as fh:
    json.dump(payload, fh, indent=2)
    fh.write("\n")
PY
  exit 0
fi

cases=(
  "let:tests/frontend/parser_deep_let_probe.sio"
  "expr:tests/frontend/parser_deep_expr_probe.sio"
)

case_lines=()
pass_count=0
fail_count=0
timeout_count=0
not_run_count=0

for entry in "${cases[@]}"; do
  name="${entry%%:*}"
  file="${entry#*:}"
  log_file="$LOG_DIR/${name}.log"

  if [ ! -f "$file" ]; then
    case_lines+=("$name|not_run|case_file_missing|0|$log_file")
    not_run_count=$((not_run_count + 1))
    continue
  fi

  set +e
  timeout "$TIMEOUT_SECS" "$SOUC_BIN" run "$file" >"$log_file" 2>&1
  rc=$?
  set -e

  status="fail"
  reason="case_failed"
  if [ "$rc" -eq 0 ]; then
    status="pass"
    reason="ok"
    pass_count=$((pass_count + 1))
  elif [ "$rc" -eq 124 ]; then
    status="timeout"
    reason="timeout"
    timeout_count=$((timeout_count + 1))
  else
    fail_count=$((fail_count + 1))
  fi

  case_lines+=("$name|$status|$reason|$rc|$log_file")
done

overall_status="pass"
overall_reason="all_passed"
if [ "$timeout_count" -gt 0 ]; then
  overall_status="timeout"
  overall_reason="one_or_more_case_timeouts"
elif [ "$fail_count" -gt 0 ]; then
  overall_status="fail"
  overall_reason="one_or_more_case_failures"
elif [ "$not_run_count" -gt 0 ]; then
  overall_status="not_run"
  overall_reason="one_or_more_cases_not_run"
fi

python3 - "$OUT_JSON" "$overall_status" "$overall_reason" "$TIMEOUT_SECS" "$pass_count" "$fail_count" "$timeout_count" "$not_run_count" "${case_lines[@]}" <<'PY'
import datetime as dt
import json
import sys

(
    out_path,
    status,
    reason,
    timeout_secs,
    pass_count,
    fail_count,
    timeout_count,
    not_run_count,
    *case_lines,
) = sys.argv[1:]

cases = []
for raw in case_lines:
    try:
        name, cstatus, creason, rc, log_path = raw.split("|", 4)
    except ValueError:
        continue
    cases.append(
        {
            "name": name,
            "status": cstatus,
            "reason": creason,
            "exit_code": int(rc),
            "log_path": log_path,
        }
    )

payload = {
    "schema": "sounio.sprint1.parser_deep_probe.v1",
    "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    "status": status,
    "reason": reason,
    "timeout_seconds": int(timeout_secs),
    "metrics": {
        "total": len(cases),
        "passed": int(pass_count),
        "failed": int(fail_count),
        "timeout": int(timeout_count),
        "not_run": int(not_run_count),
    },
    "cases": cases,
}

with open(out_path, "w", encoding="utf-8") as fh:
    json.dump(payload, fh, indent=2)
    fh.write("\n")
PY

exit 0
