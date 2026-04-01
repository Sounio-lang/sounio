#!/usr/bin/env bash
# sprint17b_aleatoric_split_gate.sh — epistemic/aleatoric type split gate
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ -n "${SOUNIO_SOUC:-}" ]; then
  SOUC="$SOUNIO_SOUC"
else
  source "$ROOT_DIR/scripts/lib/resolve_souc.sh"
  SOUC="${SOUC_BIN:-}"
fi

OUT_JSON="${SOUNIO_SPRINT17B_GATE_OUT:-$ROOT_DIR/artifacts/sprint17/aleatoric_split_gate.v1.json}"
TIMEOUT_SECS="${SOUNIO_SPRINT17B_TIMEOUT_SECS:-120}"

mkdir -p "$(dirname "$OUT_JSON")"
mkdir -p "$ROOT_DIR/artifacts/sprint17"

CASES_TSV="/tmp/sprint17b_cases_$$.tsv"
: > "$CASES_TSV"

to_rel() { [[ "$1" == "$ROOT_DIR/"* ]] && printf '%s' "${1#$ROOT_DIR/}" || printf '%s' "$1"; }

record() {
  local name="$1" status="$2" reason="$3"
  printf '%s\t%s\t%s\n' "$name" "$status" "$reason" >> "$CASES_TSV"
}

run_case_cmd() {
  local case_name="$1" pass_reason="$2"
  shift 2
  local log_file="/tmp/${case_name}_17b.log"
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

# Case 1: TyAleatoric variant in types.sio
if grep -q 'TyAleatoric' self-hosted/check/types.sio 2>/dev/null; then
  record "ty_aleatoric_in_types" "pass" "found"
else
  record "ty_aleatoric_in_types" "fail" "TyAleatoric_missing"
fi

# Case 2: TyEpistemic variant in types.sio
if grep -q 'TyEpistemic' self-hosted/check/types.sio 2>/dev/null; then
  record "ty_epistemic_in_types" "pass" "found"
else
  record "ty_epistemic_in_types" "fail" "TyEpistemic_missing"
fi

# Case 3: ty_aleatoric constructor
if grep -q 'fn ty_aleatoric' self-hosted/check/types.sio 2>/dev/null; then
  record "ty_aleatoric_constructor" "pass" "found"
else
  record "ty_aleatoric_constructor" "fail" "constructor_missing"
fi

# Case 4: ty_epistemic constructor
if grep -q 'fn ty_epistemic' self-hosted/check/types.sio 2>/dev/null; then
  record "ty_epistemic_constructor" "pass" "found"
else
  record "ty_epistemic_constructor" "fail" "constructor_missing"
fi

# Case 5: is_aleatoric_type predicate
if grep -q 'fn is_aleatoric_type' self-hosted/check/types.sio 2>/dev/null; then
  record "is_aleatoric_type_predicate" "pass" "found"
else
  record "is_aleatoric_type_predicate" "fail" "predicate_missing"
fi

# Case 6: is_epistemic_uncertain_kind predicate (checks TyEpistemic specifically)
if grep -q 'fn is_epistemic_uncertain_kind' self-hosted/check/types.sio 2>/dev/null; then
  record "is_epistemic_type_predicate" "pass" "found"
else
  record "is_epistemic_type_predicate" "fail" "predicate_missing"
fi

# Case 7: check_uncertainty_kind_reduction in epistemic.sio
if grep -q 'fn check_uncertainty_kind_reduction' self-hosted/check/epistemic.sio 2>/dev/null; then
  record "check_uncertainty_kind_reduction_fn" "pass" "found"
else
  record "check_uncertainty_kind_reduction_fn" "fail" "fn_missing"
fi

# Case 8: uncertainty_binary_result_kind in epistemic.sio
if grep -q 'fn uncertainty_binary_result_kind' self-hosted/check/epistemic.sio 2>/dev/null; then
  record "uncertainty_binary_result_kind_fn" "pass" "found"
else
  record "uncertainty_binary_result_kind_fn" "fail" "fn_missing"
fi

# Case 9: stdlib/epistemic/aleatoric.sio exists
if [ -f stdlib/epistemic/aleatoric.sio ]; then
  record "aleatoric_stdlib_exists" "pass" "found"
else
  record "aleatoric_stdlib_exists" "fail" "file_missing"
fi

# Case 10: stdlib/epistemic/epistemic_kind.sio exists
if [ -f stdlib/epistemic/epistemic_kind.sio ]; then
  record "epistemic_kind_stdlib_exists" "pass" "found"
else
  record "epistemic_kind_stdlib_exists" "fail" "file_missing"
fi

# Case 11: run-pass tests type-check
if [ -n "${SOUC:-}" ] && [ -x "${SOUC:-}" ]; then
  run_case_cmd "aleatoric_basic_typecheck" "typecheck_passed" \
    "$SOUC" check tests/frontend/aleatoric_basic.sio
  run_case_cmd "epistemic_basic_typecheck" "typecheck_passed" \
    "$SOUC" check tests/frontend/epistemic_basic.sio
else
  record "aleatoric_basic_typecheck" "not_run" "souc_unavailable"
  record "epistemic_basic_typecheck" "not_run" "souc_unavailable"
fi

# Case 12: is_reducible_uncertainty predicate
if grep -q 'fn is_reducible_uncertainty' self-hosted/check/types.sio 2>/dev/null; then
  record "is_reducible_uncertainty_predicate" "pass" "found"
else
  record "is_reducible_uncertainty_predicate" "fail" "predicate_missing"
fi

# Case 13: sprint regression — existing knowledge tests pass
SPREG_PASS=0
SPREG_TOTAL=0
for f in tests/frontend/measure_basic.sio tests/frontend/lift_knowledge_basic.sio; do
  if [ -f "$f" ]; then
    SPREG_TOTAL=$((SPREG_TOTAL + 1))
    if [ -n "${SOUC:-}" ] && [ -x "${SOUC:-}" ]; then
      set +e
      timeout "$TIMEOUT_SECS" "$SOUC" check "$f" >/dev/null 2>&1
      rc=$?
      set -e
      [ $rc -eq 0 ] && SPREG_PASS=$((SPREG_PASS + 1))
    fi
  fi
done
if [ "$SPREG_TOTAL" -gt 0 ] && [ "$SPREG_PASS" -eq "$SPREG_TOTAL" ]; then
  record "epistemic_regression" "pass" "existing_tests_still_pass"
elif [ "$SPREG_TOTAL" -eq 0 ]; then
  record "epistemic_regression" "not_run" "no_regression_tests_found"
elif [ -z "${SOUC:-}" ] || [ ! -x "${SOUC:-}" ]; then
  record "epistemic_regression" "not_run" "souc_unavailable"
else
  record "epistemic_regression" "fail" "${SPREG_PASS}/${SPREG_TOTAL}_pass"
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
    "schema": "sounio.sprint17b.aleatoric_split_gate.v1",
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
echo "sprint17b_aleatoric_split_gate: out_json=$(to_rel "$OUT_JSON") status=$status"
[ "$status" = "pass" ] && exit 0
[ "$status" = "not_run" ] && exit 3
exit 2
