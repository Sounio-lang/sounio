#!/usr/bin/env bash
# sprint24_transition_manifest_gate.sh — Transition manifest schema gate
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ -n "${SOUNIO_SOUC:-}" ]; then
  SOUC="$SOUNIO_SOUC"
else
  SOUC="$ROOT_DIR/souc"
fi

OUT_JSON="${SOUNIO_SPRINT24_GATE_OUT:-$ROOT_DIR/artifacts/sprint24/transition_manifest_gate.v1.json}"
TIMEOUT_SECS="${SOUNIO_SPRINT24_TIMEOUT_SECS:-360}"

mkdir -p "$(dirname "$OUT_JSON")"

CASES_TSV="/tmp/sprint24_cases_$$.tsv"
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
  local case_name="$1" source_file="$2"
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
  if python3 - "$json_file" << 'PY'
import json, sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
ep = data.get("epistemic", {})
assert isinstance(ep.get("transition_policies", []), list)
assert isinstance(ep.get("transition_protocols", []), list)
assert isinstance(ep.get("transition_steps", []), list)
assert isinstance(ep.get("transition_plans", []), list)
assert ep.get("transition_policies", [])
assert ep.get("transition_protocols", [])
assert ep.get("transition_steps", [])
assert ep.get("transition_plans", [])
assert ep["transition_plans"][0].get("protocol_name")
assert ep["transition_plans"][0].get("final_value_repr")
assert ep["transition_plans"][0].get("step_count", 0) >= 1
PY
  then
    record "$case_name" "pass" "manifest_contains_transition_sections"
  else
    record "$case_name" "fail" "manifest_missing_transition_sections"
  fi
}

run_case_cmd "selfhost_compiler_main_self_test" "all_checks_passed" \
  "$SOUC" run self-hosted/compiler/main.sio -- --self-test

if grep -q '\\"transition_policies\\"' self-hosted/compiler/main.sio 2>/dev/null \
   && grep -q '\\"transition_protocols\\"' self-hosted/compiler/main.sio 2>/dev/null \
   && grep -q '\\"transition_plans\\"' self-hosted/compiler/main.sio 2>/dev/null \
   && grep -q '\\"transition_steps\\"' self-hosted/compiler/main.sio 2>/dev/null; then
  record "manifest_printer_transition_fields_present" "pass" "found"
else
  record "manifest_printer_transition_fields_present" "fail" "missing_manifest_field"
fi

run_manifest_case "manifest_transition_basic_sections" "tests/frontend/commit_alternative_basic.sio"
run_manifest_case "manifest_transition_counterfactual_sections" "tests/frontend/commit_alternative_counterfactual_basic.sio"

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
    "schema": "sounio.sprint24.transition_manifest_gate.v1",
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
echo "sprint24_transition_manifest_gate: out_json=$(to_rel "$OUT_JSON") status=$status"
[ "$status" = "pass" ] && exit 0
[ "$status" = "not_run" ] && exit 3
exit 2
