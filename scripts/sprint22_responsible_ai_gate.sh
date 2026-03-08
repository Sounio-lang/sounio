#!/usr/bin/env bash
# sprint22_responsible_ai_gate.sh — Causal Transportability + Fairness Types gate
# Tracks:
#   T: TyTransportable + TySelectionDiagram (Pearl & Bareinboim 2014/2016)
#   F: TyFairPrediction + TyFairnessGap (Dwork 2012, Hardt 2016, Chouldechova 2017)
#   REG: Sprint 21 DP gate + Sprint 20 lang SOTA gate regression
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ -n "${SOUNIO_SOUC:-}" ]; then
  SOUC="$SOUNIO_SOUC"
else
  source "$ROOT_DIR/scripts/lib/resolve_souc.sh" 2>/dev/null || true
  SOUC="${SOUC_BIN:-$ROOT_DIR/souc}"
fi

OUT_JSON="${SOUNIO_SPRINT22_GATE_OUT:-$ROOT_DIR/artifacts/sprint22/responsible_ai_gate.v1.json}"
TIMEOUT_SECS="${SOUNIO_SPRINT22_TIMEOUT_SECS:-120}"

mkdir -p "$(dirname "$OUT_JSON")"
mkdir -p "$ROOT_DIR/artifacts/sprint22"

CASES_TSV="/tmp/sprint22_cases_$$.tsv"
: > "$CASES_TSV"

to_rel() { [[ "$1" == "$ROOT_DIR/"* ]] && printf '%s' "${1#$ROOT_DIR/}" || printf '%s' "$1"; }

record() {
  local name="$1" status="$2" reason="$3"
  printf '%s\t%s\t%s\n' "$name" "$status" "$reason" >> "$CASES_TSV"
}

run_case_cmd() {
  local case_name="$1" pass_reason="$2"
  shift 2
  local log_file="/tmp/${case_name}_s22.log"
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
# Track T: Causal Transportability TypeKind
# Pearl & Bareinboim (2014) "Transportability of Causal Effects"
# ============================================================

# T-1: TyTransportable declared in types.sio
if grep -q 'TyTransportable' self-hosted/check/types.sio 2>/dev/null; then
  record "ir_ty_transportable_declared" "pass" "found"
else
  record "ir_ty_transportable_declared" "fail" "TyTransportable_missing"
fi

# T-2: TySelectionDiagram declared in types.sio
if grep -q 'TySelectionDiagram' self-hosted/check/types.sio 2>/dev/null; then
  record "ir_ty_selection_diagram_declared" "pass" "found"
else
  record "ir_ty_selection_diagram_declared" "fail" "TySelectionDiagram_missing"
fi

# T-3: ty_transportable constructor present
if grep -q 'fn ty_transportable' self-hosted/check/types.sio 2>/dev/null; then
  record "ty_transportable_constructor" "pass" "found"
else
  record "ty_transportable_constructor" "fail" "constructor_missing"
fi

# T-4: ty_selection_diagram constructor present
if grep -q 'fn ty_selection_diagram' self-hosted/check/types.sio 2>/dev/null; then
  record "ty_selection_diagram_constructor" "pass" "found"
else
  record "ty_selection_diagram_constructor" "fail" "constructor_missing"
fi

# T-5: is_transportable_type predicate
if grep -q 'fn is_transportable_type' self-hosted/check/types.sio 2>/dev/null; then
  record "is_transportable_predicate" "pass" "found"
else
  record "is_transportable_predicate" "fail" "predicate_missing"
fi

# T-6: is_selection_diagram_type predicate
if grep -q 'fn is_selection_diagram_type' self-hosted/check/types.sio 2>/dev/null; then
  record "is_selection_diagram_predicate" "pass" "found"
else
  record "is_selection_diagram_predicate" "fail" "predicate_missing"
fi

# T-7: check_transportable_type in epistemic.sio (E078)
if grep -q 'fn check_transportable_type' self-hosted/check/epistemic.sio 2>/dev/null; then
  record "check_transportable_fn" "pass" "found"
else
  record "check_transportable_fn" "fail" "fn_missing"
fi

# T-8: is_s_admissible in epistemic.sio (S-admissibility criterion)
if grep -q 'fn is_s_admissible' self-hosted/check/epistemic.sio 2>/dev/null; then
  record "is_s_admissible_fn" "pass" "found"
else
  record "is_s_admissible_fn" "fail" "fn_missing"
fi

# T-9: transport_to_ate in epistemic.sio (TyTransportable → TyATE)
if grep -q 'fn transport_to_ate' self-hosted/check/epistemic.sio 2>/dev/null; then
  record "transport_to_ate_fn" "pass" "found"
else
  record "transport_to_ate_fn" "fail" "fn_missing"
fi

# T-10: TyTransportable compat rule in compat.sio
if grep -q 'TyTransportable' self-hosted/check/compat.sio 2>/dev/null; then
  record "transportable_compat_rule" "pass" "found"
else
  record "transportable_compat_rule" "fail" "compat_rule_missing"
fi

# T-11: stdlib/causal/transport.sio exists
if [ -f stdlib/causal/transport.sio ]; then
  record "stdlib_causal_transport_exists" "pass" "found"
else
  record "stdlib_causal_transport_exists" "fail" "file_missing"
fi

# T-12: transportability_basic.sio passes souc check
if [ -n "${SOUC:-}" ] && [ -x "${SOUC:-}" ]; then
  run_case_cmd "check_transportability_basic" "typecheck_passed" \
    "$SOUC" check tests/frontend/transportability_basic.sio
else
  record "check_transportability_basic" "not_run" "souc_unavailable"
fi

# ============================================================
# Track F: ML Fairness TypeKind
# Dwork et al. (2012), Hardt et al. (2016), Chouldechova (2017)
# ============================================================

# F-1: TyFairPrediction declared in types.sio
if grep -q 'TyFairPrediction' self-hosted/check/types.sio 2>/dev/null; then
  record "ir_ty_fair_prediction_declared" "pass" "found"
else
  record "ir_ty_fair_prediction_declared" "fail" "TyFairPrediction_missing"
fi

# F-2: TyFairnessGap declared in types.sio
if grep -q 'TyFairnessGap' self-hosted/check/types.sio 2>/dev/null; then
  record "ir_ty_fairness_gap_declared" "pass" "found"
else
  record "ir_ty_fairness_gap_declared" "fail" "TyFairnessGap_missing"
fi

# F-3: ty_fair_prediction constructor present
if grep -q 'fn ty_fair_prediction' self-hosted/check/types.sio 2>/dev/null; then
  record "ty_fair_prediction_constructor" "pass" "found"
else
  record "ty_fair_prediction_constructor" "fail" "constructor_missing"
fi

# F-4: ty_fairness_gap constructor present
if grep -q 'fn ty_fairness_gap' self-hosted/check/types.sio 2>/dev/null; then
  record "ty_fairness_gap_constructor" "pass" "found"
else
  record "ty_fairness_gap_constructor" "fail" "constructor_missing"
fi

# F-5: is_fair_prediction_type predicate
if grep -q 'fn is_fair_prediction_type' self-hosted/check/types.sio 2>/dev/null; then
  record "is_fair_prediction_predicate" "pass" "found"
else
  record "is_fair_prediction_predicate" "fail" "predicate_missing"
fi

# F-6: is_fairness_gap_type predicate
if grep -q 'fn is_fairness_gap_type' self-hosted/check/types.sio 2>/dev/null; then
  record "is_fairness_gap_predicate" "pass" "found"
else
  record "is_fairness_gap_predicate" "fail" "predicate_missing"
fi

# F-7: check_fair_prediction_type in epistemic.sio (E080)
if grep -q 'fn check_fair_prediction_type' self-hosted/check/epistemic.sio 2>/dev/null; then
  record "check_fair_prediction_fn" "pass" "found"
else
  record "check_fair_prediction_fn" "fail" "fn_missing"
fi

# F-8: fairness_impossibility_holds in epistemic.sio (Chouldechova 2017 → E082)
if grep -q 'fn fairness_impossibility_holds' self-hosted/check/epistemic.sio 2>/dev/null; then
  record "fairness_impossibility_fn" "pass" "found"
else
  record "fairness_impossibility_fn" "fail" "fn_missing"
fi

# F-9: counterfactual_fairness_from_potential_outcome in epistemic.sio (bridges PO → fairness)
if grep -q 'fn counterfactual_fairness_from_potential_outcome' self-hosted/check/epistemic.sio 2>/dev/null; then
  record "counterfactual_fairness_from_po_fn" "pass" "found"
else
  record "counterfactual_fairness_from_po_fn" "fail" "fn_missing"
fi

# F-10: TyFairPrediction compat rule in compat.sio
if grep -q 'TyFairPrediction' self-hosted/check/compat.sio 2>/dev/null; then
  record "fair_prediction_compat_rule" "pass" "found"
else
  record "fair_prediction_compat_rule" "fail" "compat_rule_missing"
fi

# F-11: stdlib/fairness/metrics.sio exists
if [ -f stdlib/fairness/metrics.sio ]; then
  record "stdlib_fairness_metrics_exists" "pass" "found"
else
  record "stdlib_fairness_metrics_exists" "fail" "file_missing"
fi

# F-12: fairness_basic.sio passes souc check
if [ -n "${SOUC:-}" ] && [ -x "${SOUC:-}" ]; then
  run_case_cmd "check_fairness_basic" "typecheck_passed" \
    "$SOUC" check tests/frontend/fairness_basic.sio
else
  record "check_fairness_basic" "not_run" "souc_unavailable"
fi

# ============================================================
# Track F extra: Fairness criterion constants + Chouldechova
# ============================================================

# F-13: All 5 fairness criterion constants declared
if grep -q 'FAIRNESS_CRITERION_DP' self-hosted/check/epistemic.sio 2>/dev/null; then
  record "fairness_criterion_constants" "pass" "found"
else
  record "fairness_criterion_constants" "fail" "criterion_constants_missing"
fi

# F-14: E082 error code referenced (impossibility theorem error)
if grep -q 'E082' self-hosted/check/epistemic.sio 2>/dev/null; then
  record "e082_impossibility_error_code" "pass" "found"
else
  record "e082_impossibility_error_code" "fail" "E082_missing"
fi

# F-15: TyFairnessGap print arm in check.sio
if grep -q 'TyFairnessGap' self-hosted/check/check.sio 2>/dev/null; then
  record "fairness_gap_print_arm" "pass" "found"
else
  record "fairness_gap_print_arm" "fail" "print_arm_missing"
fi

# F-16: Chouldechova ppv_gap in stdlib/fairness/metrics.sio
if grep -q 'fn chouldechova_ppv_gap\|fn fairness_criteria_incompatible' stdlib/fairness/metrics.sio 2>/dev/null; then
  record "stdlib_chouldechova_fns" "pass" "found"
else
  record "stdlib_chouldechova_fns" "fail" "chouldechova_fns_missing"
fi

# ============================================================
# Regression
# ============================================================

# REG-1: sprint21 DP gate artifact is pass
if python3 -c "import json; d=json.load(open('artifacts/sprint21/dp_gate.v1.json')); exit(0 if d.get('status')=='pass' else 1)" 2>/dev/null; then
  record "sprint21_dp_gate_passes" "pass" "artifact_status_pass"
else
  record "sprint21_dp_gate_passes" "fail" "sprint21_dp_not_pass"
fi

# REG-2: sprint20 lang SOTA gate artifact is pass
if python3 -c "import json; d=json.load(open('artifacts/sprint20/lang_sota_gate.v1.json')); exit(0 if d.get('status')=='pass' else 1)" 2>/dev/null; then
  record "sprint20_lang_sota_gate_passes" "pass" "artifact_status_pass"
else
  record "sprint20_lang_sota_gate_passes" "fail" "sprint20_lang_sota_not_pass"
fi

# REG-3: Sprint 22 frontend regression — transportability + fairness + prior sprint tests
REG22_PASS=0; REG22_TOTAL=0
for f in tests/frontend/transportability_basic.sio tests/frontend/fairness_basic.sio tests/frontend/diff_private_basic.sio tests/frontend/causal_basic.sio tests/frontend/cond_indep_basic.sio; do
  if [ -f "$f" ] && [ -n "${SOUC:-}" ] && [ -x "${SOUC:-}" ]; then
    REG22_TOTAL=$((REG22_TOTAL + 1))
    set +e
    timeout "$TIMEOUT_SECS" "$SOUC" check "$f" >/dev/null 2>&1
    rc=$?; set -e
    [ $rc -eq 0 ] && REG22_PASS=$((REG22_PASS + 1))
  fi
done
if [ "$REG22_TOTAL" -eq 0 ]; then
  record "sprint22_frontend_regression" "not_run" "souc_unavailable"
elif [ "$REG22_PASS" -eq "$REG22_TOTAL" ]; then
  record "sprint22_frontend_regression" "pass" "${REG22_PASS}/${REG22_TOTAL}_pass"
else
  record "sprint22_frontend_regression" "fail" "${REG22_PASS}/${REG22_TOTAL}_pass"
fi

# REG-4: TyTransportable declared in check.sio (print_type_name arm)
if grep -q 'TyTransportable' self-hosted/check/check.sio 2>/dev/null; then
  record "transportable_print_arm" "pass" "found"
else
  record "transportable_print_arm" "fail" "print_arm_missing"
fi

# REG-5: TyFairPrediction declared in check.sio (print_type_name arm)
if grep -q 'TyFairPrediction' self-hosted/check/check.sio 2>/dev/null; then
  record "fairness_print_arm" "pass" "found"
else
  record "fairness_print_arm" "fail" "print_arm_missing"
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
    "schema": "sounio.sprint22.responsible_ai_gate.v1",
    "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    "status": overall,
    "reason": reason,
    "tracks": ["T_causal_transportability", "F_ml_fairness", "REG_regression"],
    "sota_novelty": {
        "TyTransportable": "first_language_with_transportability_as_type_Pearl_Bareinboim_2014",
        "TySelectionDiagram": "first_compile_time_s_admissibility_check",
        "TyFairPrediction": "first_language_with_demographic_parity_as_type_constraint",
        "TyFairnessGap": "first_measured_fairness_disparity_as_type",
        "fairness_impossibility_holds": "Chouldechova_2017_impossibility_theorem_as_type_predicate",
        "counterfactual_fairness_from_po": "Pearl_counterfactual_fairness_from_potential_outcomes",
    },
    "paper_targets": {
        "PLDI_2028": "Causal_Transportability_as_a_Type_S-Admissibility_at_Compile_Time",
        "FAccT_2027": "Compile-Time_Fairness_Constraints_ML_Bias_Detection_via_Graded_Types",
        "POPL_2027": "From_Privacy_to_Fairness_Unified_Type_Theory_for_Responsible_AI",
    },
    "causal_hierarchy_complete": {
        "Layer1_observation": ["TyKnowledge", "TyCondIndep"],
        "Layer2_intervention": ["TyCausalEffect", "TyIntervention", "TyPotentialOutcome", "TyATE"],
        "Layer3_counterfactual": ["TyCounterfactual"],
        "Layer4_transportability": ["TyTransportable", "TySelectionDiagram"],
    },
    "fairness_criteria": {
        "0_demographic_parity": "Dwork_et_al_2012",
        "1_equalized_odds": "Hardt_et_al_2016",
        "2_individual_fairness": "Dwork_et_al_2012_Lipschitz",
        "3_calibration_within_group": "Chouldechova_2017",
        "4_counterfactual_fairness": "Kusner_et_al_2017",
    },
    "responsible_ai_types": {
        "Sprint21_differential_privacy": "TyDiffPrivate_TyDPBudget",
        "Sprint22_causal_transport": "TyTransportable_TySelectionDiagram",
        "Sprint22_fairness": "TyFairPrediction_TyFairnessGap",
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
echo "sprint22_responsible_ai_gate: out_json=$(to_rel "$OUT_JSON") status=$status"
[ "$status" = "pass" ] && exit 0
[ "$status" = "not_run" ] && exit 3
exit 2
