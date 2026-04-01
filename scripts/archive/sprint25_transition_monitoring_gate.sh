#!/usr/bin/env bash
# sprint25_transition_monitoring_gate.sh — Transition monitoring frontend/checker gate
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ -n "${SOUNIO_SOUC:-}" ]; then
  SOUC="$SOUNIO_SOUC"
else
  SOUC="$ROOT_DIR/souc"
fi

if [ -n "${SOUNIO_SELFHOST_SOUC:-}" ]; then
  SELFHOST_SOUC="$SOUNIO_SELFHOST_SOUC"
else
  SELFHOST_SOUC="$ROOT_DIR/target/debug/souc"
fi

OUT_JSON="${SOUNIO_SPRINT25_GATE_OUT:-$ROOT_DIR/artifacts/sprint25/transition_monitoring_gate.v1.json}"
TIMEOUT_SECS="${SOUNIO_SPRINT25_TIMEOUT_SECS:-180}"

mkdir -p "$(dirname "$OUT_JSON")"

CASES_TSV="/tmp/sprint25_cases_$$.tsv"
: > "$CASES_TSV"

to_rel() { [[ "$1" == "$ROOT_DIR/"* ]] && printf '%s' "${1#$ROOT_DIR/}" || printf '%s' "$1"; }

record() {
  local name="$1" status="$2" reason="$3"
  printf '%s\t%s\t%s\n' "$name" "$status" "$reason" >> "$CASES_TSV"
}

run_case_cmd() {
  local case_name="$1" pass_reason="$2"
  shift 2
  local log_file="/tmp/${case_name}_$$.log"
  rm -f "$log_file"
  set +e
  timeout "$TIMEOUT_SECS" "$@" >"$log_file" 2>&1 </dev/null
  local rc=$?
  set -e
  if [ $rc -eq 0 ]; then
    record "$case_name" "pass" "$pass_reason"
  elif [ $rc -eq 124 ]; then
    record "$case_name" "not_run" "timeout"
  else
    local reason
    reason="$(tail -5 "$log_file" | tr '\n' ' ' | tr -s ' ' | head -c 180 || echo error)"
    record "$case_name" "fail" "$reason"
  fi
}

run_case_cmd "selfhost_check_mod_import_probe" "probe_ok" \
  "$SELFHOST_SOUC" run self-hosted/probe_import_check_mod_only.sio

if grep -q "TypeMonitoringPolicy" self-hosted/parser/types.sio 2>/dev/null \
   && grep -q "TypeObservedTransition" self-hosted/parser/types.sio 2>/dev/null \
   && grep -q "TypeRollbackCertificate" self-hosted/parser/types.sio 2>/dev/null \
   && grep -q "ItemMonitoringPolicy" self-hosted/parser/ast.sio 2>/dev/null; then
  record "monitoring_items_and_types_present" "pass" "found"
else
  record "monitoring_items_and_types_present" "fail" "not_found"
fi

if grep -q "observe_transition" self-hosted/check/check.sio 2>/dev/null \
   && grep -q "rollback_transition" self-hosted/check/check.sio 2>/dev/null \
   && grep -q "transition_reason" self-hosted/check/check.sio 2>/dev/null; then
  record "monitoring_checker_paths_present" "pass" "found"
else
  record "monitoring_checker_paths_present" "fail" "not_found"
fi

if grep -q "MonitoringPolicy" souc 2>/dev/null \
   && grep -q "observe_transition" souc 2>/dev/null \
   && grep -q "rollback_transition" souc 2>/dev/null; then
  record "wrapper_monitoring_fallback_patterns_present" "pass" "found"
else
  record "wrapper_monitoring_fallback_patterns_present" "fail" "not_found"
fi

if [ -f artifacts/omega/bootstrap_knowledge_tests.v1.json ] \
   && python3 - <<'PY'
import json
from pathlib import Path
payload = json.loads(Path("artifacts/omega/bootstrap_knowledge_tests.v1.json").read_text())
assert payload["check"]["status"] == "pass"
assert payload["run"]["status"] == "pass"
PY
then
  record "bootstrap_monitoring_summary_green" "pass" "summary_green"
else
  record "bootstrap_monitoring_summary_green" "fail" "summary_not_green"
fi

if [ -f self-hosted/test_knowledge_bootstrap.sio ] \
   && grep -q "B39 OK: lower MonitoringPolicy<T>" self-hosted/test_knowledge_bootstrap.sio \
   && grep -q "B42 OK: observe_transition records metadata and exports IR artifacts" self-hosted/test_knowledge_bootstrap.sio \
   && grep -q "B43 OK: rollback_transition records metadata and exports IR artifacts" self-hosted/test_knowledge_bootstrap.sio; then
  record "bootstrap_monitoring_cases_present" "pass" "B39_B42_B43_defined"
else
  record "bootstrap_monitoring_cases_present" "fail" "missing_B39_or_B42_or_B43"
fi

python3 - "$OUT_JSON" "$CASES_TSV" "$SELFHOST_SOUC" "$TIMEOUT_SECS" << 'PY'
import json, datetime as dt, sys
from pathlib import Path

out_json = Path(sys.argv[1])
cases_tsv = Path(sys.argv[2])
souc = sys.argv[3]
timeout_secs = int(sys.argv[4])

cases = []
counts = {"pass": 0, "fail": 0, "not_run": 0}

for raw in cases_tsv.read_text(encoding="utf-8").splitlines():
    if not raw.strip():
        continue
    name, status, reason = raw.split("\t", 2)
    if status not in counts:
        status, reason = "fail", f"invalid_{status}"
    counts[status] += 1
    cases.append({"name": name, "status": status, "reason": reason})

overall = "pass" if counts["fail"] == 0 and counts["not_run"] == 0 else \
          "not_run" if counts["pass"] == 0 and counts["fail"] == 0 else "fail"
reason = "all_cases_passed" if overall == "pass" else \
         f"{counts['fail']}_failed" if counts["fail"] else "one_or_more_not_run"

payload = {
    "schema": "sounio.sprint25.transition_monitoring_gate.v1",
    "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    "status": overall,
    "reason": reason,
    "config": {"souc": souc, "timeout_seconds": timeout_secs},
    "metrics": {"total": len(cases), "passed": counts["pass"], "failed": counts["fail"], "not_run": counts["not_run"]},
    "cases": cases,
}
out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(f"wrote {out_json}")
print(f"status={overall} reason={reason}")
PY

rm -f "$CASES_TSV"
status="$(python3 -c "import json; d=json.load(open('$OUT_JSON')); print(d.get('status','fail'))")"
echo "sprint25_transition_monitoring_gate: out_json=$(to_rel "$OUT_JSON") status=$status"
[ "$status" = "pass" ] && exit 0
[ "$status" = "not_run" ] && exit 3
exit 2
