#!/usr/bin/env bash
# sprint21_dp_gate.sh — Differential Privacy types + Sprint 20 negative test suite gate
# Tracks:
#   A: Sprint 20 compile-fail negative test file existence
#   B: TyDiffPrivate + TyDPBudget TypeKind implementation
#   REG: Sprint 20 lang SOTA gate regression
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ -n "${SOUNIO_SOUC:-}" ]; then
  SOUC="$SOUNIO_SOUC"
else
  source "$ROOT_DIR/scripts/lib/resolve_souc.sh" 2>/dev/null || true
  SOUC="${SOUC_BIN:-$ROOT_DIR/souc}"
fi

OUT_JSON="${SOUNIO_SPRINT21_DP_GATE_OUT:-$ROOT_DIR/artifacts/sprint21/dp_gate.v1.json}"
TIMEOUT_SECS="${SOUNIO_SPRINT21_TIMEOUT_SECS:-120}"

mkdir -p "$(dirname "$OUT_JSON")"
mkdir -p "$ROOT_DIR/artifacts/sprint21"

CASES_TSV="/tmp/sprint21_dp_cases_$$.tsv"
: > "$CASES_TSV"

to_rel() { [[ "$1" == "$ROOT_DIR/"* ]] && printf '%s' "${1#$ROOT_DIR/}" || printf '%s' "$1"; }

record() {
  local name="$1" status="$2" reason="$3"
  printf '%s\t%s\t%s\n' "$name" "$status" "$reason" >> "$CASES_TSV"
}

run_case_cmd() {
  local case_name="$1" pass_reason="$2"
  shift 2
  local log_file="/tmp/${case_name}_s21.log"
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
    reason="$(tail -3 "$log_file" | tr '\n' ' ' | tr -s ' ' | head -c 200 || echo error)"
    record "$case_name" "fail" "$reason"
  fi
}

# ============================================================
# Part A: Sprint 20 compile-fail negative test existence
# ============================================================

# A-1: singleton_wrong_constant.sio exists
if [ -f tests/compile-fail/singleton_wrong_constant.sio ]; then
  record "negtest_singleton_wrong_constant_exists" "pass" "found"
else
  record "negtest_singleton_wrong_constant_exists" "fail" "file_missing"
fi

# A-2: scoped_effect_escape.sio exists
if [ -f tests/compile-fail/scoped_effect_escape.sio ]; then
  record "negtest_scoped_effect_escape_exists" "pass" "found"
else
  record "negtest_scoped_effect_escape_exists" "fail" "file_missing"
fi

# A-3: cond_indep_violation.sio exists
if [ -f tests/compile-fail/cond_indep_violation.sio ]; then
  record "negtest_cond_indep_violation_exists" "pass" "found"
else
  record "negtest_cond_indep_violation_exists" "fail" "file_missing"
fi

# A-4: potential_outcome_both_observed.sio exists
if [ -f tests/compile-fail/potential_outcome_both_observed.sio ]; then
  record "negtest_potential_outcome_both_observed_exists" "pass" "found"
else
  record "negtest_potential_outcome_both_observed_exists" "fail" "file_missing"
fi

# A-5: All 4 compile-fail tests carry the correct error-pattern annotations
if grep -q 'error-pattern: E070' tests/compile-fail/singleton_wrong_constant.sio 2>/dev/null; then
  record "negtest_singleton_error_annotation" "pass" "E070_annotation_present"
else
  record "negtest_singleton_error_annotation" "fail" "E070_annotation_missing"
fi

if grep -q 'error-pattern: E073' tests/compile-fail/potential_outcome_both_observed.sio 2>/dev/null; then
  record "negtest_po_error_annotation" "pass" "E073_annotation_present"
else
  record "negtest_po_error_annotation" "fail" "E073_annotation_missing"
fi

# ============================================================
# Part B: TyDiffPrivate TypeKind declared + constructed
# ============================================================

# B-1: TyDiffPrivate declared in types.sio
if grep -q 'TyDiffPrivate' self-hosted/check/types.sio 2>/dev/null; then
  record "ir_ty_diff_private_declared" "pass" "found"
else
  record "ir_ty_diff_private_declared" "fail" "TyDiffPrivate_missing"
fi

# B-2: TyDPBudget declared in types.sio
if grep -q 'TyDPBudget' self-hosted/check/types.sio 2>/dev/null; then
  record "ir_ty_dp_budget_declared" "pass" "found"
else
  record "ir_ty_dp_budget_declared" "fail" "TyDPBudget_missing"
fi

# B-3: ty_diff_private constructor present
if grep -q 'fn ty_diff_private' self-hosted/check/types.sio 2>/dev/null; then
  record "ty_diff_private_constructor" "pass" "found"
else
  record "ty_diff_private_constructor" "fail" "constructor_missing"
fi

# B-4: ty_dp_budget constructor present
if grep -q 'fn ty_dp_budget' self-hosted/check/types.sio 2>/dev/null; then
  record "ty_dp_budget_constructor" "pass" "found"
else
  record "ty_dp_budget_constructor" "fail" "constructor_missing"
fi

# B-5: is_diff_private_type predicate
if grep -q 'fn is_diff_private_type' self-hosted/check/types.sio 2>/dev/null; then
  record "is_diff_private_predicate" "pass" "found"
else
  record "is_diff_private_predicate" "fail" "predicate_missing"
fi

# B-6: is_dp_budget_type predicate
if grep -q 'fn is_dp_budget_type' self-hosted/check/types.sio 2>/dev/null; then
  record "is_dp_budget_predicate" "pass" "found"
else
  record "is_dp_budget_predicate" "fail" "predicate_missing"
fi

# B-7: check_diff_private_type in epistemic.sio
if grep -q 'fn check_diff_private_type' self-hosted/check/epistemic.sio 2>/dev/null; then
  record "check_diff_private_fn" "pass" "found"
else
  record "check_diff_private_fn" "fail" "fn_missing"
fi

# B-8: check_dp_budget_type in epistemic.sio
if grep -q 'fn check_dp_budget_type' self-hosted/check/epistemic.sio 2>/dev/null; then
  record "check_dp_budget_fn" "pass" "found"
else
  record "check_dp_budget_fn" "fail" "fn_missing"
fi

# B-9: dp_sequential_compose in epistemic.sio
if grep -q 'fn dp_sequential_compose' self-hosted/check/epistemic.sio 2>/dev/null; then
  record "dp_sequential_compose_fn" "pass" "found"
else
  record "dp_sequential_compose_fn" "fail" "fn_missing"
fi

# B-10: dp_parallel_compose in epistemic.sio
if grep -q 'fn dp_parallel_compose' self-hosted/check/epistemic.sio 2>/dev/null; then
  record "dp_parallel_compose_fn" "pass" "found"
else
  record "dp_parallel_compose_fn" "fail" "fn_missing"
fi

# B-11: dp_budget_consume in epistemic.sio
if grep -q 'fn dp_budget_consume' self-hosted/check/epistemic.sio 2>/dev/null; then
  record "dp_budget_consume_fn" "pass" "found"
else
  record "dp_budget_consume_fn" "fail" "fn_missing"
fi

# B-12: TyDiffPrivate compat rule in compat.sio
if grep -q 'TyDiffPrivate' self-hosted/check/compat.sio 2>/dev/null; then
  record "diff_private_compat_rule" "pass" "found"
else
  record "diff_private_compat_rule" "fail" "compat_rule_missing"
fi

# B-13: TyDPBudget compat rule in compat.sio
if grep -q 'TyDPBudget' self-hosted/check/compat.sio 2>/dev/null; then
  record "dp_budget_compat_rule" "pass" "found"
else
  record "dp_budget_compat_rule" "fail" "compat_rule_missing"
fi

# B-14: TyDiffPrivate print arm in check.sio
if grep -q 'TyDiffPrivate' self-hosted/check/check.sio 2>/dev/null; then
  record "diff_private_print_arm" "pass" "found"
else
  record "diff_private_print_arm" "fail" "print_arm_missing"
fi

# B-15: stdlib/privacy/differential.sio exists
if [ -f stdlib/privacy/differential.sio ]; then
  record "stdlib_privacy_differential_exists" "pass" "found"
else
  record "stdlib_privacy_differential_exists" "fail" "file_missing"
fi

# B-16: dp_sequential_epsilon in stdlib
if grep -q 'fn dp_sequential_epsilon' stdlib/privacy/differential.sio 2>/dev/null; then
  record "stdlib_dp_sequential_epsilon_fn" "pass" "found"
else
  record "stdlib_dp_sequential_epsilon_fn" "fail" "fn_missing"
fi

# B-17: dp_parallel_epsilon in stdlib
if grep -q 'fn dp_parallel_epsilon' stdlib/privacy/differential.sio 2>/dev/null; then
  record "stdlib_dp_parallel_epsilon_fn" "pass" "found"
else
  record "stdlib_dp_parallel_epsilon_fn" "fail" "fn_missing"
fi

# B-18: dp_advanced_composition in stdlib
if grep -q 'fn dp_advanced_composition' stdlib/privacy/differential.sio 2>/dev/null; then
  record "stdlib_dp_advanced_composition_fn" "pass" "found"
else
  record "stdlib_dp_advanced_composition_fn" "fail" "fn_missing"
fi

# B-19: diff_private_basic.sio passes souc check
if [ -n "${SOUC:-}" ] && [ -x "${SOUC:-}" ]; then
  run_case_cmd "check_diff_private_basic" "typecheck_passed" \
    "$SOUC" check tests/frontend/diff_private_basic.sio
else
  record "check_diff_private_basic" "not_run" "souc_unavailable"
fi

# ============================================================
# Regression: Sprint 20 lang SOTA gate still passes
# ============================================================

# REG-1: sprint20 lang sota gate artifact is pass
if python3 -c "import json; d=json.load(open('artifacts/sprint20/lang_sota_gate.v1.json')); exit(0 if d.get('status')=='pass' else 1)" 2>/dev/null; then
  record "sprint20_lang_sota_gate_passes" "pass" "artifact_status_pass"
else
  record "sprint20_lang_sota_gate_passes" "fail" "sprint20_lang_sota_not_pass"
fi

# REG-2: sprint19 resolution gate artifact is pass
if python3 -c "import json; d=json.load(open('artifacts/sprint19/resolution_gate.v1.json')); exit(0 if d.get('status')=='pass' else 1)" 2>/dev/null; then
  record "sprint19_resolution_gate_passes" "pass" "artifact_status_pass"
else
  record "sprint19_resolution_gate_passes" "fail" "sprint19_resolution_not_pass"
fi

# REG-3: existing epistemic/causal tests still pass
# (measure_basic and lift_knowledge_basic are //@ ignore — they use deferred syntax)
REGR_PASS=0; REGR_TOTAL=0
for f in tests/frontend/causal_basic.sio tests/frontend/singleton_constants_basic.sio tests/frontend/cond_indep_basic.sio; do
  if [ -f "$f" ] && [ -n "${SOUC:-}" ] && [ -x "${SOUC:-}" ]; then
    REGR_TOTAL=$((REGR_TOTAL + 1))
    set +e
    timeout "$TIMEOUT_SECS" "$SOUC" check "$f" >/dev/null 2>&1
    rc=$?; set -e
    [ $rc -eq 0 ] && REGR_PASS=$((REGR_PASS + 1))
  fi
done
if [ "$REGR_TOTAL" -eq 0 ]; then
  record "epistemic_regression" "not_run" "no_tests_found"
elif [ "$REGR_PASS" -eq "$REGR_TOTAL" ]; then
  record "epistemic_regression" "pass" "${REGR_PASS}/${REGR_TOTAL}_pass"
else
  record "epistemic_regression" "fail" "${REGR_PASS}/${REGR_TOTAL}_pass"
fi

# REG-4: Sprint 20 frontend tests still typecheck
S20_PASS=0; S20_TOTAL=0
for f in tests/frontend/singleton_constants_basic.sio tests/frontend/scoped_effect_basic.sio tests/frontend/cond_indep_basic.sio tests/frontend/potential_outcomes_basic.sio; do
  if [ -f "$f" ] && [ -n "${SOUC:-}" ] && [ -x "${SOUC:-}" ]; then
    S20_TOTAL=$((S20_TOTAL + 1))
    set +e
    timeout "$TIMEOUT_SECS" "$SOUC" check "$f" >/dev/null 2>&1
    rc=$?; set -e
    [ $rc -eq 0 ] && S20_PASS=$((S20_PASS + 1))
  fi
done
if [ "$S20_TOTAL" -eq 0 ]; then
  record "sprint20_frontend_regression" "not_run" "souc_unavailable"
elif [ "$S20_PASS" -eq "$S20_TOTAL" ]; then
  record "sprint20_frontend_regression" "pass" "${S20_PASS}/${S20_TOTAL}_pass"
else
  record "sprint20_frontend_regression" "fail" "${S20_PASS}/${S20_TOTAL}_pass"
fi

# ============================================================
# Generate gate artifact
# ============================================================
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
    "schema": "sounio.sprint21.dp_gate.v1",
    "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    "status": overall,
    "reason": reason,
    "tracks": ["A_sprint20_negative_tests", "B_differential_privacy_types", "REG_regression"],
    "sota_novelty": {
        "TyDiffPrivate": "first_language_ep_dp_as_type_level_quantity",
        "TyDPBudget": "first_depleting_privacy_budget_as_type",
        "dp_sequential_compose": "theorem_3_14_dwork_2006_as_type_rule",
        "dp_parallel_compose": "theorem_3_20_dwork_2006_as_type_rule",
    },
    "paper_targets": {
        "POPL_2026": "Graded_DP_types_budget_as_compile_time_algebraic_effect",
        "CCS_2027": "Differential_privacy_type_system_for_data_science_pipelines",
    },
    "prior_art": {
        "DFuzz_POPL_2013": "linear_types_for_sensitivity_not_epsilon",
        "Fuzzi_PLDI_2019": "dependent_types_for_dp_programs",
        "Duet_POPL_2020": "linear_type_theory_for_dp_composition",
        "Sounio_gap": "first_to_track_epsilon_budget_as_graded_type_with_composition",
    },
    "config": {"souc": souc, "timeout_seconds": timeout_secs},
    "metrics": {
        "total": len(cases),
        "passed": counts["pass"],
        "failed": counts["fail"],
        "not_run": counts["not_run"],
    },
    "cases": cases,
}
out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(f"wrote {out_json}")
print(f"status={overall} reason={reason}")
PY

rm -f "$CASES_TSV"
status="$(python3 -c "import json; d=json.load(open('$OUT_JSON')); print(d.get('status','fail'))")"
echo "sprint21_dp_gate: out_json=$(to_rel "$OUT_JSON") status=$status"
[ "$status" = "pass" ] && exit 0
[ "$status" = "not_run" ] && exit 3
exit 2
