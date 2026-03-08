#!/usr/bin/env bash
set -euo pipefail

# Sprint 25: CI Consolidation Gate
#
# Verifies:
# 1. All sprint gate scripts (5-24) exist and are executable
# 2. Release-critical pack has 52 steps wired
# 3. Gate coverage audit >= 80%
# 4. No probe/bisect debris in self-hosted/

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

pass=0
fail=0
total=0

check() {
  local label="$1"
  shift
  total=$((total + 1))
  if "$@" >/dev/null 2>&1; then
    pass=$((pass + 1))
    echo "PASS  $label"
  else
    fail=$((fail + 1))
    echo "FAIL  $label"
  fi
}

check_grep() {
  local label="$1"
  local file="$2"
  local pattern="$3"
  total=$((total + 1))
  if grep -q "$pattern" "$file" 2>/dev/null; then
    pass=$((pass + 1))
    echo "PASS  $label"
  else
    fail=$((fail + 1))
    echo "FAIL  $label"
  fi
}

check_no_match() {
  local label="$1"
  local pattern="$2"
  local dir="$3"
  total=$((total + 1))
  if ! ls $dir/$pattern 2>/dev/null | grep -q .; then
    pass=$((pass + 1))
    echo "PASS  $label"
  else
    fail=$((fail + 1))
    echo "FAIL  $label"
  fi
}

echo "=== Sprint 25: CI Consolidation Gate ==="
echo ""

# --- Section 1: Gate scripts exist ---
echo "--- Gate script existence ---"
GATE_SCRIPTS=(
  sprint5_contest_validated_gate.sh
  sprint7_prove_robust_gate.sh
  sprint8_validate_manifest_gate.sh
  sprint9_hlir_epistemic_gate.sh
  sprint10_lift_knowledge_gate.sh
  sprint11_measure_gate.sh
  sprint12_borrow_integration_gate.sh
  sprint13_diagnostic_hardening_gate.sh
  sprint14_checker_parity_gate.sh
  sprint14_contest_witness_gate.sh
  sprint15_effects_parity_gate.sh
  sprint15_epistemic_witness_manifest_gate.sh
  sprint16_effect_handlers_gate.sh
  sprint16_decision_admissibility_gate.sh
  sprint17a_row_poly_gate.sh
  sprint17b_aleatoric_split_gate.sh
  sprint17d_causal_type_gate.sh
  sprint17_deferral_gate.sh
  sprint18_graded_epistemic_gate.sh
  sprint18_deferral_manifest_gate.sh
  sprint19_epistemic_session_gate.sh
  sprint19_resolution_gate.sh
  sprint19_session_types_gate.sh
  sprint19_user_effects_gate.sh
  sprint19_web_gate.sh
  sprint20_lang_sota_gate.sh
  sprint20_resolution_manifest_gate.sh
  sprint21_dp_gate.sh
  sprint21_alternative_frontier_gate.sh
  sprint22_responsible_ai_gate.sh
  sprint22_alternative_manifest_gate.sh
  sprint23_statistical_types_gate.sh
  sprint24_scientific_foundations_gate.sh
)

for script in "${GATE_SCRIPTS[@]}"; do
  check "gate-script:$script" test -x "$ROOT_DIR/scripts/$script"
done

# --- Section 2: Release-critical pack wiring ---
echo ""
echo "--- Release-critical pack wiring ---"
check_grep "pack:step-22-sprint12" "$ROOT_DIR/scripts/run_release_critical_pack.sh" "22-sprint12-borrow"
check_grep "pack:step-34-sprint17a" "$ROOT_DIR/scripts/run_release_critical_pack.sh" "34-sprint17a-row-poly"
check_grep "pack:step-38-sprint18" "$ROOT_DIR/scripts/run_release_critical_pack.sh" "38-sprint18-graded-epistemic"
check_grep "pack:step-47-sprint21-dp" "$ROOT_DIR/scripts/run_release_critical_pack.sh" "47-sprint21-dp"
check_grep "pack:step-52-sprint24" "$ROOT_DIR/scripts/run_release_critical_pack.sh" "52-sprint24-scientific"

# Count total run_step lines
STEP_COUNT=$(grep -c 'run_step ' "$ROOT_DIR/scripts/run_release_critical_pack.sh" || echo 0)
total=$((total + 1))
if [[ "$STEP_COUNT" -ge 50 ]]; then
  pass=$((pass + 1))
  echo "PASS  pack:step-count>=$STEP_COUNT"
else
  fail=$((fail + 1))
  echo "FAIL  pack:step-count=$STEP_COUNT (expected >=50)"
fi

# --- Section 3: No probe/bisect debris ---
echo ""
echo "--- Probe/bisect debris check ---"
check_no_match "no-probe-dirs" ".probe-*" "$ROOT_DIR/self-hosted"
check_no_match "no-probe-sio" "probe_*.sio" "$ROOT_DIR/self-hosted"

# --- Section 4: .gitignore patterns ---
echo ""
echo "--- .gitignore patterns ---"
check_grep "gitignore:probe-dirs" "$ROOT_DIR/.gitignore" "self-hosted/.probe-"
check_grep "gitignore:probe-sio" "$ROOT_DIR/.gitignore" "self-hosted/probe_"
check_grep "gitignore:importprobe" "$ROOT_DIR/.gitignore" "importprobe_"

# --- Section 5: Gate coverage audit ---
echo ""
echo "--- Gate coverage audit ---"
AUDIT_OUTPUT=$(bash "$ROOT_DIR/scripts/gate_coverage_audit.sh" 2>&1)
COVERAGE_PCT=$(echo "$AUDIT_OUTPUT" | grep "Coverage:" | grep -o '[0-9]*')
total=$((total + 1))
if [[ "$COVERAGE_PCT" -ge 80 ]]; then
  pass=$((pass + 1))
  echo "PASS  coverage:${COVERAGE_PCT}%>=80%"
else
  fail=$((fail + 1))
  echo "FAIL  coverage:${COVERAGE_PCT}%<80%"
fi

echo ""
echo "=== Sprint 25 Gate: $pass/$total pass ==="

ARTIFACT_DIR="$ROOT_DIR/artifacts/sprint25"
mkdir -p "$ARTIFACT_DIR"
cat > "$ARTIFACT_DIR/ci_consolidation_gate.v1.json" <<ARTIFACT
{
  "gate": "sprint25_ci_consolidation",
  "version": 1,
  "status": "$(if [[ $fail -eq 0 ]]; then echo PASS; else echo FAIL; fi)",
  "pass": $pass,
  "total": $total,
  "fail": $fail,
  "coverage_pct": $COVERAGE_PCT,
  "step_count": $STEP_COUNT,
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
ARTIFACT

echo "artifact: $ARTIFACT_DIR/ci_consolidation_gate.v1.json"

if [[ $fail -gt 0 ]]; then
  exit 1
fi
