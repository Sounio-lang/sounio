#!/usr/bin/env bash
# sprint19_user_effects_gate.sh — user-defined effect declarations (Track E1) gate
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ -n "${SOUNIO_SOUC:-}" ]; then
  SOUC="$SOUNIO_SOUC"
else
  source "$ROOT_DIR/scripts/lib/resolve_souc.sh"
  SOUC="${SOUC_BIN:-}"
fi

OUT_JSON="${SOUNIO_SPRINT19_USER_EFFECTS_GATE_OUT:-$ROOT_DIR/artifacts/sprint19/user_effects_gate.v1.json}"
TIMEOUT_SECS="${SOUNIO_SPRINT19_TIMEOUT_SECS:-120}"

mkdir -p "$(dirname "$OUT_JSON")"

CASES_TSV="/tmp/sprint19_ue_cases_$$.tsv"
: > "$CASES_TSV"

to_rel() { [[ "$1" == "$ROOT_DIR/"* ]] && printf '%s' "${1#$ROOT_DIR/}" || printf '%s' "$1"; }

record() {
  local name="$1" status="$2" reason="$3"
  printf '%s\t%s\t%s\n' "$name" "$status" "$reason" >> "$CASES_TSV"
}

run_case_cmd() {
  local case_name="$1" pass_reason="$2"
  shift 2
  local log_file="/tmp/${case_name}_19ue.log"
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
    reason="$(tail -1 "$log_file" | tr -s ' ' | head -c 160 || echo error)"
    record "$case_name" "fail" "$reason"
  fi
}

# Case 1: effects_unhandled.sio exists with the effect keyword
if [ -f tests/compile-fail/effects_unhandled.sio ]; then
  if grep -q 'effect Fail' tests/compile-fail/effects_unhandled.sio 2>/dev/null; then
    record "effects_unhandled_exists" "pass" "found"
  else
    record "effects_unhandled_exists" "fail" "effect_Fail_missing"
  fi
else
  record "effects_unhandled_exists" "fail" "file_missing"
fi

# Case 2: effect_user_defined_basic.sio exists
if [ -f tests/frontend/effect_user_defined_basic.sio ]; then
  record "effect_user_defined_basic_exists" "pass" "found"
else
  record "effect_user_defined_basic_exists" "fail" "file_missing"
fi

# Case 3: parse_effect_item in parser
if grep -q 'parse_effect_item' self-hosted/parser/items.sio 2>/dev/null; then
  record "parse_effect_item_in_parser" "pass" "found"
else
  record "parse_effect_item_in_parser" "fail" "fn_missing"
fi

# Case 4: collect_item_effect (effect registration) in check.sio
if grep -q 'collect_item_effect' self-hosted/check/check.sio 2>/dev/null; then
  record "collect_item_effect_in_checker" "pass" "found"
else
  record "collect_item_effect_in_checker" "fail" "fn_missing"
fi

# Case 5: lookup_user_effect_id in check.sio
if grep -q 'lookup_user_effect_id' self-hosted/check/check.sio 2>/dev/null; then
  record "lookup_user_effect_id_in_checker" "pass" "found"
else
  record "lookup_user_effect_id_in_checker" "fail" "fn_missing"
fi

# Case 6: Effect token in lexer
if grep -q 'Effect,' self-hosted/lexer/token.sio 2>/dev/null; then
  record "effect_token_in_lexer" "pass" "found"
else
  record "effect_token_in_lexer" "fail" "token_missing"
fi

# Case 7: parse_effect_op_def in parser
if grep -q 'parse_effect_op_def' self-hosted/parser/items.sio 2>/dev/null; then
  record "parse_effect_op_def_in_parser" "pass" "found"
else
  record "parse_effect_op_def_in_parser" "fail" "fn_missing"
fi

# Case 8: run-pass test compiles (if souc available)
if [ -n "${SOUC:-}" ] && [ -x "${SOUC:-}" ]; then
  run_case_cmd "effect_user_defined_basic_typecheck" "typecheck_passed" \
    "$SOUC" check tests/frontend/effect_user_defined_basic.sio
else
  record "effect_user_defined_basic_typecheck" "not_run" "souc_unavailable"
fi

python3 - "$OUT_JSON" "$CASES_TSV" "${SOUC:-unavailable}" "$TIMEOUT_SECS" << 'PY'
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
          "not_run" if counts["pass"] == 0 and counts["fail"] == 0 else \
          "fail" if counts["fail"] > 0 else "not_run"
reason = "all_cases_passed" if overall == "pass" else \
         f"{counts['fail']}_failed" if counts["fail"] else "one_or_more_not_run"

payload = {
    "schema": "sounio.sprint19.user_effects_gate.v1",
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
echo "sprint19_user_effects_gate: out_json=$(to_rel "$OUT_JSON") status=$status"
[ "$status" = "pass" ] && exit 0
[ "$status" = "not_run" ] && exit 3
exit 2
