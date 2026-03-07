#!/usr/bin/env bash
# sprint18_graded_epistemic_gate.sh — graded epistemic effects (Track C convergence) gate
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ -n "${SOUNIO_SOUC:-}" ]; then
  SOUC="$SOUNIO_SOUC"
else
  source "$ROOT_DIR/scripts/lib/resolve_souc.sh"
  SOUC="${SOUC_BIN:-}"
fi

OUT_JSON="${SOUNIO_SPRINT18_GATE_OUT:-$ROOT_DIR/artifacts/sprint18/graded_epistemic_gate.v1.json}"
TIMEOUT_SECS="${SOUNIO_SPRINT18_TIMEOUT_SECS:-120}"

mkdir -p "$(dirname "$OUT_JSON")"
mkdir -p "$ROOT_DIR/artifacts/sprint18"

CASES_TSV="/tmp/sprint18_cases_$$.tsv"
: > "$CASES_TSV"

to_rel() { [[ "$1" == "$ROOT_DIR/"* ]] && printf '%s' "${1#$ROOT_DIR/}" || printf '%s' "$1"; }

record() {
  local name="$1" status="$2" reason="$3"
  printf '%s\t%s\t%s\n' "$name" "$status" "$reason" >> "$CASES_TSV"
}

run_case_cmd() {
  local case_name="$1" pass_reason="$2"
  shift 2
  local log_file="/tmp/${case_name}_18.log"
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

# Case 1: TyGradedEffect variant in types.sio
if grep -q 'TyGradedEffect' self-hosted/check/types.sio 2>/dev/null; then
  record "ty_graded_effect_in_types" "pass" "found"
else
  record "ty_graded_effect_in_types" "fail" "TyGradedEffect_missing"
fi

# Case 2: ty_graded_effect constructor
if grep -q 'fn ty_graded_effect' self-hosted/check/types.sio 2>/dev/null; then
  record "ty_graded_effect_constructor" "pass" "found"
else
  record "ty_graded_effect_constructor" "fail" "constructor_missing"
fi

# Case 3: is_graded_effect_type predicate
if grep -q 'fn is_graded_effect_type' self-hosted/check/types.sio 2>/dev/null; then
  record "is_graded_effect_type_predicate" "pass" "found"
else
  record "is_graded_effect_type_predicate" "fail" "predicate_missing"
fi

# Case 4: graded_effect_subsumes function
if grep -q 'fn graded_effect_subsumes' self-hosted/check/types.sio 2>/dev/null; then
  record "graded_effect_subsumes_fn" "pass" "found"
else
  record "graded_effect_subsumes_fn" "fail" "fn_missing"
fi

# Case 5: graded_effect_compose function
if grep -q 'fn graded_effect_compose' self-hosted/check/types.sio 2>/dev/null; then
  record "graded_effect_compose_fn" "pass" "found"
else
  record "graded_effect_compose_fn" "fail" "fn_missing"
fi

# Case 6: stdlib/epistemic/graded_effects.sio exists
if [ -f stdlib/epistemic/graded_effects.sio ]; then
  record "graded_effects_stdlib_exists" "pass" "found"
else
  record "graded_effects_stdlib_exists" "fail" "file_missing"
fi

# Case 7: GradedProb struct in stdlib
if grep -q 'struct GradedProb' stdlib/epistemic/graded_effects.sio 2>/dev/null; then
  record "graded_prob_struct" "pass" "found"
else
  record "graded_prob_struct" "fail" "struct_missing"
fi

# Case 8: graded_compose function (runtime grade algebra)
if grep -q 'fn graded_compose' stdlib/epistemic/graded_effects.sio 2>/dev/null; then
  record "graded_compose_fn" "pass" "found"
else
  record "graded_compose_fn" "fail" "fn_missing"
fi

# Case 9: graded_epistemic_update (Bayesian update)
if grep -q 'fn graded_epistemic_update' stdlib/epistemic/graded_effects.sio 2>/dev/null; then
  record "graded_epistemic_update_fn" "pass" "found"
else
  record "graded_epistemic_update_fn" "fail" "fn_missing"
fi

# Case 10: run-pass tests type-check
if [ -n "${SOUC:-}" ] && [ -x "${SOUC:-}" ]; then
  run_case_cmd "graded_effect_basic_typecheck" "typecheck_passed" \
    "$SOUC" check tests/frontend/graded_effect_basic.sio
  run_case_cmd "graded_effect_compose_typecheck" "typecheck_passed" \
    "$SOUC" check tests/frontend/graded_effect_compose.sio
else
  record "graded_effect_basic_typecheck" "not_run" "souc_unavailable"
  record "graded_effect_compose_typecheck" "not_run" "souc_unavailable"
fi

# Case 11: Track A+B prerequisite — TyEffectVar, TyAleatoric, TyEpistemic present
if grep -q 'TyEffectVar' self-hosted/check/types.sio 2>/dev/null && \
   grep -q 'TyAleatoric' self-hosted/check/types.sio 2>/dev/null && \
   grep -q 'TyEpistemic' self-hosted/check/types.sio 2>/dev/null; then
  record "track_ab_prerequisites" "pass" "all_prerequisite_types_found"
else
  record "track_ab_prerequisites" "fail" "missing_prerequisite_type"
fi

# Case 12: sprint regression
SPREG_PASS=0
SPREG_TOTAL=0
for f in tests/frontend/aleatoric_basic.sio tests/frontend/epistemic_basic.sio \
          tests/frontend/causal_basic.sio tests/frontend/effect_row_poly_basic.sio; do
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
  record "sprint17_regression" "pass" "sprint17_tests_still_pass"
elif [ "$SPREG_TOTAL" -eq 0 ]; then
  record "sprint17_regression" "not_run" "no_regression_tests_found"
elif [ -z "${SOUC:-}" ] || [ ! -x "${SOUC:-}" ]; then
  record "sprint17_regression" "not_run" "souc_unavailable"
else
  record "sprint17_regression" "fail" "${SPREG_PASS}/${SPREG_TOTAL}_pass"
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
    "schema": "sounio.sprint18.graded_epistemic_gate.v1",
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
echo "sprint18_graded_epistemic_gate: out_json=$(to_rel "$OUT_JSON") status=$status"
[ "$status" = "pass" ] && exit 0
[ "$status" = "not_run" ] && exit 3
exit 2
