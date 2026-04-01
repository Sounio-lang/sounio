#!/usr/bin/env bash
# sprint20_lang_sota_gate.sh — Language SOTA++ gate
# Tracks: SC (Singleton constants), SE (Scoped effects),
#         CI (Conditional independence), PO (Potential outcomes)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ -n "${SOUNIO_SOUC:-}" ]; then
  SOUC="$SOUNIO_SOUC"
else
  source "$ROOT_DIR/scripts/lib/resolve_souc.sh" 2>/dev/null || true
  SOUC="${SOUC_BIN:-$ROOT_DIR/souc}"
fi

OUT_JSON="${SOUNIO_SPRINT20_LANG_GATE_OUT:-$ROOT_DIR/artifacts/sprint20/lang_sota_gate.v1.json}"
TIMEOUT_SECS="${SOUNIO_SPRINT20_LANG_TIMEOUT_SECS:-120}"

mkdir -p "$(dirname "$OUT_JSON")"
mkdir -p "$ROOT_DIR/artifacts/sprint20"

CASES_TSV="/tmp/sprint20_lang_cases_$$.tsv"
: > "$CASES_TSV"

to_rel() { [[ "$1" == "$ROOT_DIR/"* ]] && printf '%s' "${1#$ROOT_DIR/}" || printf '%s' "$1"; }

record() {
  local name="$1" status="$2" reason="$3"
  printf '%s\t%s\t%s\n' "$name" "$status" "$reason" >> "$CASES_TSV"
}

run_case_cmd() {
  local case_name="$1" pass_reason="$2"
  shift 2
  local log_file="/tmp/${case_name}_s20.log"
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
# Track SC: Singleton physical constants
# ============================================================

# SC-1: TySingleton declared in types.sio
if grep -q 'TySingleton' self-hosted/check/types.sio 2>/dev/null; then
  record "ir_ty_singleton_declared" "pass" "found"
else
  record "ir_ty_singleton_declared" "fail" "TySingleton_missing"
fi

# SC-2: ty_singleton constructor present
if grep -q 'fn ty_singleton' self-hosted/check/types.sio 2>/dev/null; then
  record "ty_singleton_constructor" "pass" "found"
else
  record "ty_singleton_constructor" "fail" "constructor_missing"
fi

# SC-3: is_singleton_type predicate present
if grep -q 'fn is_singleton_type' self-hosted/check/types.sio 2>/dev/null; then
  record "is_singleton_type_predicate" "pass" "found"
else
  record "is_singleton_type_predicate" "fail" "predicate_missing"
fi

# SC-4: stdlib/constants/physical.sio exists
if [ -f stdlib/constants/physical.sio ]; then
  record "stdlib_constants_physical_exists" "pass" "found"
else
  record "stdlib_constants_physical_exists" "fail" "file_missing"
fi

# SC-5: singleton_constants_basic.sio passes souc check
if [ -n "${SOUC:-}" ] && [ -x "${SOUC:-}" ]; then
  run_case_cmd "check_singleton_constants_basic" "typecheck_passed" \
    "$SOUC" check tests/frontend/singleton_constants_basic.sio
else
  record "check_singleton_constants_basic" "not_run" "souc_unavailable"
fi

# SC-6: compat.sio has TySingleton compatibility rule
if grep -q 'TySingleton' self-hosted/check/compat.sio 2>/dev/null; then
  record "singleton_compat_rule" "pass" "found"
else
  record "singleton_compat_rule" "fail" "compat_rule_missing"
fi

# ============================================================
# Track SE: Scoped effect types
# ============================================================

# SE-1: TyScopedEffect declared in types.sio
if grep -q 'TyScopedEffect' self-hosted/check/types.sio 2>/dev/null; then
  record "ir_ty_scoped_effect_declared" "pass" "found"
else
  record "ir_ty_scoped_effect_declared" "fail" "TyScopedEffect_missing"
fi

# SE-2: ty_scoped_effect constructor present
if grep -q 'fn ty_scoped_effect' self-hosted/check/types.sio 2>/dev/null; then
  record "ty_scoped_effect_constructor" "pass" "found"
else
  record "ty_scoped_effect_constructor" "fail" "constructor_missing"
fi

# SE-3: check_scoped_effect_type in epistemic.sio
if grep -q 'fn check_scoped_effect_type' self-hosted/check/epistemic.sio 2>/dev/null; then
  record "check_scoped_effect_fn" "pass" "found"
else
  record "check_scoped_effect_fn" "fail" "fn_missing"
fi

# SE-4: scoped_effect_basic.sio passes souc check
if [ -n "${SOUC:-}" ] && [ -x "${SOUC:-}" ]; then
  run_case_cmd "check_scoped_effect_basic" "typecheck_passed" \
    "$SOUC" check tests/frontend/scoped_effect_basic.sio
else
  record "check_scoped_effect_basic" "not_run" "souc_unavailable"
fi

# ============================================================
# Track CI: Conditional independence types
# ============================================================

# CI-1: TyCondIndep declared in types.sio
if grep -q 'TyCondIndep' self-hosted/check/types.sio 2>/dev/null; then
  record "ir_ty_cond_indep_declared" "pass" "found"
else
  record "ir_ty_cond_indep_declared" "fail" "TyCondIndep_missing"
fi

# CI-2: ty_cond_indep constructor present
if grep -q 'fn ty_cond_indep' self-hosted/check/types.sio 2>/dev/null; then
  record "ty_cond_indep_constructor" "pass" "found"
else
  record "ty_cond_indep_constructor" "fail" "constructor_missing"
fi

# CI-3: check_cond_indep_type in epistemic.sio
if grep -q 'fn check_cond_indep_type' self-hosted/check/epistemic.sio 2>/dev/null; then
  record "check_cond_indep_fn" "pass" "found"
else
  record "check_cond_indep_fn" "fail" "fn_missing"
fi

# CI-4: cond_indep_structurally_holds (d-sep structural check)
if grep -q 'fn cond_indep_structurally_holds' self-hosted/check/epistemic.sio 2>/dev/null; then
  record "cond_indep_structurally_holds_fn" "pass" "found"
else
  record "cond_indep_structurally_holds_fn" "fail" "fn_missing"
fi

# CI-5: cond_indep_basic.sio passes souc check
if [ -n "${SOUC:-}" ] && [ -x "${SOUC:-}" ]; then
  run_case_cmd "check_cond_indep_basic" "typecheck_passed" \
    "$SOUC" check tests/frontend/cond_indep_basic.sio
else
  record "check_cond_indep_basic" "not_run" "souc_unavailable"
fi

# ============================================================
# Track PO: Potential outcomes (Rubin framework)
# ============================================================

# PO-1: TyPotentialOutcome declared in types.sio
if grep -q 'TyPotentialOutcome' self-hosted/check/types.sio 2>/dev/null; then
  record "ir_ty_potential_outcome_declared" "pass" "found"
else
  record "ir_ty_potential_outcome_declared" "fail" "TyPotentialOutcome_missing"
fi

# PO-2: TyATE declared in types.sio
if grep -q 'TyATE' self-hosted/check/types.sio 2>/dev/null; then
  record "ir_ty_ate_declared" "pass" "found"
else
  record "ir_ty_ate_declared" "fail" "TyATE_missing"
fi

# PO-3: ty_potential_outcome constructor present
if grep -q 'fn ty_potential_outcome' self-hosted/check/types.sio 2>/dev/null; then
  record "ty_potential_outcome_constructor" "pass" "found"
else
  record "ty_potential_outcome_constructor" "fail" "constructor_missing"
fi

# PO-4: ty_ate constructor present
if grep -q 'fn ty_ate' self-hosted/check/types.sio 2>/dev/null; then
  record "ty_ate_constructor" "pass" "found"
else
  record "ty_ate_constructor" "fail" "constructor_missing"
fi

# PO-5: check_potential_outcome_type in epistemic.sio
if grep -q 'fn check_potential_outcome_type' self-hosted/check/epistemic.sio 2>/dev/null; then
  record "check_potential_outcome_fn" "pass" "found"
else
  record "check_potential_outcome_fn" "fail" "fn_missing"
fi

# PO-6: potential_outcome_to_ate type transformer present
if grep -q 'fn potential_outcome_to_ate' self-hosted/check/epistemic.sio 2>/dev/null; then
  record "potential_outcome_to_ate_fn" "pass" "found"
else
  record "potential_outcome_to_ate_fn" "fail" "fn_missing"
fi

# PO-7: potential_outcomes_basic.sio passes souc check
if [ -n "${SOUC:-}" ] && [ -x "${SOUC:-}" ]; then
  run_case_cmd "check_potential_outcomes_basic" "typecheck_passed" \
    "$SOUC" check tests/frontend/potential_outcomes_basic.sio
else
  record "check_potential_outcomes_basic" "not_run" "souc_unavailable"
fi

# ============================================================
# Regression: Sprint 19 & 20 gates still pass
# ============================================================

# REG-1: sprint19 resolution gate artifact is pass
if python3 -c "import json; d=json.load(open('artifacts/sprint19/resolution_gate.v1.json')); exit(0 if d.get('status')=='pass' else 1)" 2>/dev/null; then
  record "sprint19_resolution_gate_passes" "pass" "artifact_status_pass"
else
  record "sprint19_resolution_gate_passes" "fail" "sprint19_resolution_not_pass"
fi

# REG-2: sprint20 resolution manifest gate artifact is pass
if python3 -c "import json; d=json.load(open('artifacts/sprint20/resolution_manifest_gate.v1.json')); exit(0 if d.get('status')=='pass' else 1)" 2>/dev/null; then
  record "sprint20_resolution_manifest_gate_passes" "pass" "artifact_status_pass"
else
  record "sprint20_resolution_manifest_gate_passes" "fail" "sprint20_resolution_not_pass"
fi

# REG-3: selfhost compiler self-test
if [ -n "${SOUC:-}" ] && [ -x "${SOUC:-}" ]; then
  run_case_cmd "selfhost_compiler_main_self_test" "all_tests_passed" \
    "$SOUC" run self-hosted/compiler/main.sio -- --self-test
else
  record "selfhost_compiler_main_self_test" "not_run" "souc_unavailable"
fi

# REG-4: existing epistemic tests still pass
REGR_PASS=0; REGR_TOTAL=0
for f in tests/frontend/measure_basic.sio tests/frontend/lift_knowledge_basic.sio tests/frontend/causal_basic.sio; do
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
    "schema": "sounio.sprint20.lang_sota_gate.v1",
    "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    "status": overall,
    "reason": reason,
    "tracks": ["SC_singleton_constants", "SE_scoped_effects", "CI_cond_independence", "PO_potential_outcomes"],
    "sota_novelty": {
        "TySingleton": "first_language_singleton_identity_plus_dimensional_analysis",
        "TyScopedEffect": "scoped_effects_with_epistemic_uncertainty_budget",
        "TyCondIndep": "first_type_system_compile_time_conditional_independence",
        "TyPotentialOutcome": "first_type_system_rubin_potential_outcomes_linear_consumed",
        "TyATE": "average_treatment_effect_as_type_level_estimand"
    },
    "paper_targets": {
        "PLDI_2027": "TyCondIndep_type_directed_causal_verification",
        "POPL_2027": "TyPotentialOutcome_rubin_framework_as_linear_types",
        "ICFP_2026": "TyScopedEffect_and_TySingleton"
    },
    "config": {"souc": souc, "timeout_seconds": timeout_secs},
    "metrics": {
        "total": len(cases),
        "passed": counts["pass"],
        "failed": counts["fail"],
        "not_run": counts["not_run"]
    },
    "cases": cases,
}
out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(f"wrote {out_json}")
print(f"status={overall} reason={reason}")
PY

rm -f "$CASES_TSV"
status="$(python3 -c "import json; d=json.load(open('$OUT_JSON')); print(d.get('status','fail'))")"
echo "sprint20_lang_sota_gate: out_json=$(to_rel "$OUT_JSON") status=$status"
[ "$status" = "pass" ] && exit 0
[ "$status" = "not_run" ] && exit 3
exit 2
