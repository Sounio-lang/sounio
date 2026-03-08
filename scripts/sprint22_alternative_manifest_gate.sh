#!/usr/bin/env bash
# sprint22_alternative_manifest_gate.sh — Alternative manifest schema gate
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ -n "${SOUNIO_SOUC:-}" ]; then
  SOUC="$SOUNIO_SOUC"
else
  SOUC="$ROOT_DIR/souc"
fi

OUT_JSON="${SOUNIO_SPRINT22_GATE_OUT:-$ROOT_DIR/artifacts/sprint22/alternative_manifest_gate.v1.json}"
TIMEOUT_SECS="${SOUNIO_SPRINT22_TIMEOUT_SECS:-360}"

mkdir -p "$(dirname "$OUT_JSON")"

CASES_TSV="/tmp/sprint22_cases_$$.tsv"
: > "$CASES_TSV"

to_rel() { [[ "$1" == "$ROOT_DIR/"* ]] && printf '%s' "${1#$ROOT_DIR/}" || printf '%s' "$1"; }

record() {
  local name="$1" status="$2" reason="$3"
  printf '%s\t%s\t%s\n' "$name" "$status" "$reason" >> "$CASES_TSV"
}

extract_manifest_to_json() {
  local log_file="$1"
  local json_file="$2"
  python3 - "$log_file" "$json_file" << 'PY'
import json, sys
from pathlib import Path

raw = Path(sys.argv[1]).read_text(encoding="utf-8")
start = raw.find("{")
if start < 0:
    raise SystemExit(1)
decoder = json.JSONDecoder()
obj, _ = decoder.raw_decode(raw[start:])
Path(sys.argv[2]).write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
PY
}

run_case_cmd() {
  local case_name="$1" pass_reason="$2"
  shift 2
  local log_file="/tmp/${case_name}_$$.log"
  rm -f "$log_file"
  set +e
  timeout "$TIMEOUT_SECS" "$@" >"$log_file" 2>&1
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

run_manifest_case() {
  local case_name="$1" source_file="$2" expected_kind="$3"
  local log_file="/tmp/${case_name}.manifest.log"
  local json_file="/tmp/${case_name}.manifest.json"
  rm -f "$log_file" "$json_file"
  set +e
  timeout "$TIMEOUT_SECS" "$SOUC" run self-hosted/compiler/main.sio -- compile --emit-manifest "$source_file" >"$log_file" 2>&1
  local rc=$?
  set -e
  if [ $rc -eq 124 ]; then
    record "$case_name" "not_run" "timeout"
    return
  fi
  if [ $rc -ne 0 ]; then
    local reason
    reason="$(tail -5 "$log_file" | tr '\n' ' ' | tr -s ' ' | head -c 180 || echo error)"
    record "$case_name" "fail" "$reason"
    return
  fi
  if ! extract_manifest_to_json "$log_file" "$json_file"; then
    record "$case_name" "fail" "manifest_json_extract_failed"
    return
  fi
if python3 - "$json_file" "$expected_kind" << 'PY'
import json, sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = sys.argv[2]
ep = data.get("epistemic", {})

assert isinstance(ep.get("alternative_policies", []), list)
assert isinstance(ep.get("alternative_frontiers", []), list)
assert isinstance(ep.get("alternative_sets", []), list)
assert isinstance(ep.get("alternative_candidates", []), list)
assert ep.get("alternative_policies", [])
assert ep.get("alternative_frontiers", [])
assert ep.get("alternative_sets", [])
assert ep.get("alternative_candidates", [])

if expected == "recourse":
    kinds = {c.get("source_kind") for c in ep.get("alternative_candidates", [])}
    assert "RecoursePlan" in kinds
else:
    kinds = {c.get("source_kind") for c in ep.get("alternative_candidates", [])}
    assert "DeferredRoute" in kinds
PY
  then
    record "$case_name" "pass" "manifest_contains_alternative_sections"
  else
    record "$case_name" "fail" "manifest_missing_alternative_sections"
  fi
}

run_case_cmd "selfhost_compiler_main_self_test" "all_checks_passed" \
  "$SOUC" run self-hosted/compiler/main.sio -- --self-test

if grep -q '\\"alternative_policies\\"' self-hosted/compiler/main.sio 2>/dev/null \
   && grep -q '\\"alternative_frontiers\\"' self-hosted/compiler/main.sio 2>/dev/null \
   && grep -q '\\"alternative_sets\\"' self-hosted/compiler/main.sio 2>/dev/null \
   && grep -q '\\"alternative_candidates\\"' self-hosted/compiler/main.sio 2>/dev/null; then
  record "manifest_printer_alternative_fields_present" "pass" "found"
else
  record "manifest_printer_alternative_fields_present" "fail" "missing_manifest_field"
fi

run_manifest_case "manifest_alternative_recourse_frontier_sections" "tests/frontend/propose_alternatives_recourse_plan_basic.sio" "recourse"
run_manifest_case "manifest_alternative_deferred_frontier_sections" "tests/frontend/propose_alternatives_deferred_basic.sio" "deferred"

python3 - "$OUT_JSON" "$CASES_TSV" "$SOUC" "$TIMEOUT_SECS" << 'PY'
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
    "schema": "sounio.sprint22.alternative_manifest_gate.v1",
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
echo "sprint22_alternative_manifest_gate: out_json=$(to_rel "$OUT_JSON") status=$status"
[ "$status" = "pass" ] && exit 0
[ "$status" = "not_run" ] && exit 3
exit 2
