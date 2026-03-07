#!/usr/bin/env bash
# sprint17d_causal_type_gate.sh — causal type system gate
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ -n "${SOUNIO_SOUC:-}" ]; then
  SOUC="$SOUNIO_SOUC"
else
  source "$ROOT_DIR/scripts/lib/resolve_souc.sh"
  SOUC="${SOUC_BIN:-}"
fi

OUT_JSON="${SOUNIO_SPRINT17D_GATE_OUT:-$ROOT_DIR/artifacts/sprint17/causal_type_gate.v1.json}"
TIMEOUT_SECS="${SOUNIO_SPRINT17D_TIMEOUT_SECS:-120}"

mkdir -p "$(dirname "$OUT_JSON")"
mkdir -p "$ROOT_DIR/artifacts/sprint17"

CASES_TSV="/tmp/sprint17d_cases_$$.tsv"
: > "$CASES_TSV"

to_rel() { [[ "$1" == "$ROOT_DIR/"* ]] && printf '%s' "${1#$ROOT_DIR/}" || printf '%s' "$1"; }

record() {
  local name="$1" status="$2" reason="$3"
  printf '%s\t%s\t%s\n' "$name" "$status" "$reason" >> "$CASES_TSV"
}

run_case_cmd() {
  local case_name="$1" pass_reason="$2"
  shift 2
  local log_file="/tmp/${case_name}_17d.log"
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

# Case 1: TyCausalEffect variant in types.sio
if grep -q 'TyCausalEffect' self-hosted/check/types.sio 2>/dev/null; then
  record "ty_causal_effect_in_types" "pass" "found"
else
  record "ty_causal_effect_in_types" "fail" "TyCausalEffect_missing"
fi

# Case 2: ty_causal_effect constructor
if grep -q 'fn ty_causal_effect' self-hosted/check/types.sio 2>/dev/null; then
  record "ty_causal_effect_constructor" "pass" "found"
else
  record "ty_causal_effect_constructor" "fail" "constructor_missing"
fi

# Case 3: is_causal_effect_type predicate
if grep -q 'fn is_causal_effect_type' self-hosted/check/types.sio 2>/dev/null; then
  record "is_causal_effect_type_predicate" "pass" "found"
else
  record "is_causal_effect_type_predicate" "fail" "predicate_missing"
fi

# Case 4: causal_effect_is_identifiable predicate
if grep -q 'fn causal_effect_is_identifiable' self-hosted/check/types.sio 2>/dev/null; then
  record "causal_effect_is_identifiable_predicate" "pass" "found"
else
  record "causal_effect_is_identifiable_predicate" "fail" "predicate_missing"
fi

# Case 5: self-hosted/check/causal.sio exists
if [ -f self-hosted/check/causal.sio ]; then
  record "causal_sio_exists" "pass" "found"
else
  record "causal_sio_exists" "fail" "file_missing"
fi

# Case 6: CausalGraph struct in causal.sio
if grep -q 'struct CausalGraph' self-hosted/check/causal.sio 2>/dev/null; then
  record "causal_graph_struct" "pass" "found"
else
  record "causal_graph_struct" "fail" "struct_missing"
fi

# Case 7: check_causal_identifiability function
if grep -q 'fn check_causal_identifiability' self-hosted/check/causal.sio 2>/dev/null; then
  record "check_causal_identifiability_fn" "pass" "found"
else
  record "check_causal_identifiability_fn" "fail" "fn_missing"
fi

# Case 8: check_causal_effect_type function
if grep -q 'fn check_causal_effect_type' self-hosted/check/causal.sio 2>/dev/null; then
  record "check_causal_effect_type_fn" "pass" "found"
else
  record "check_causal_effect_type_fn" "fail" "fn_missing"
fi

# Case 9: stdlib/causal/mod.sio exists
if [ -f stdlib/causal/mod.sio ]; then
  record "causal_stdlib_exists" "pass" "found"
else
  record "causal_stdlib_exists" "fail" "file_missing"
fi

# Case 10: CausalEstimate struct in stdlib
if grep -q 'struct CausalEstimate' stdlib/causal/mod.sio 2>/dev/null; then
  record "causal_estimate_struct" "pass" "found"
else
  record "causal_estimate_struct" "fail" "struct_missing"
fi

# Case 11: causal_basic.sio type-checks
if [ -n "${SOUC:-}" ] && [ -x "${SOUC:-}" ]; then
  run_case_cmd "causal_basic_typecheck" "typecheck_passed" \
    "$SOUC" check tests/frontend/causal_basic.sio
  run_case_cmd "causal_pbpk_typecheck" "typecheck_passed" \
    "$SOUC" check tests/frontend/causal_pbpk.sio
else
  record "causal_basic_typecheck" "not_run" "souc_unavailable"
  record "causal_pbpk_typecheck" "not_run" "souc_unavailable"
fi

# Case 13: causal_non_identifiable.sio exists
if [ -f tests/compile-fail/causal_non_identifiable.sio ]; then
  record "causal_non_identifiable_test_exists" "pass" "found"
else
  record "causal_non_identifiable_test_exists" "fail" "test_missing"
fi

# Case 14: sprint regression
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
    "schema": "sounio.sprint17d.causal_type_gate.v1",
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
echo "sprint17d_causal_type_gate: out_json=$(to_rel "$OUT_JSON") status=$status"
[ "$status" = "pass" ] && exit 0
[ "$status" = "not_run" ] && exit 3
exit 2
