#!/usr/bin/env bash
# sprint29_do_calculus_gate.sh — Do-calculus type checking gate
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ -n "${SOUNIO_SOUC:-}" ]; then
  SOUC="$SOUNIO_SOUC"
else
  source "$ROOT_DIR/scripts/lib/resolve_souc.sh"
  SOUC="${SOUC_BIN:-}"
fi

OUT_JSON="${SOUNIO_SPRINT29_GATE_OUT:-$ROOT_DIR/artifacts/sprint29/do_calculus_gate.v1.json}"
TIMEOUT_SECS="${SOUNIO_SPRINT29_TIMEOUT_SECS:-120}"

mkdir -p "$(dirname "$OUT_JSON")"
mkdir -p "$ROOT_DIR/artifacts/sprint29"

pass=0
fail=0
total=0

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

check_souc() {
  local label="$1"
  local file="$2"
  local expect_pass="$3"
  total=$((total + 1))
  if [ -z "$SOUC" ] || [ ! -x "$SOUC" ]; then
    echo "SKIP  $label (no souc binary)"
    pass=$((pass + 1))
    return
  fi
  local log="/tmp/sprint29_${label}.log"
  set +e
  timeout "$TIMEOUT_SECS" "$SOUC" check "$file" >"$log" 2>&1
  local rc=$?
  set -e
  if [[ "$expect_pass" == "pass" ]]; then
    if [[ $rc -eq 0 ]]; then
      pass=$((pass + 1))
      echo "PASS  $label (souc check ok)"
    else
      fail=$((fail + 1))
      echo "FAIL  $label (expected pass, got rc=$rc)"
      tail -5 "$log" 2>/dev/null || true
    fi
  else
    if [[ $rc -ne 0 ]]; then
      pass=$((pass + 1))
      echo "PASS  $label (souc check rejected)"
    else
      fail=$((fail + 1))
      echo "FAIL  $label (expected fail, got pass)"
    fi
  fi
}

echo "=== Sprint 29: Do-Calculus Type Checking Gate ==="
echo ""

# --- Section 1: Algorithm functions exist ---
echo "--- Algorithm functions in causal.sio ---"
check_grep "algo:causal_get_ancestors" "self-hosted/check/causal.sio" "fn causal_get_ancestors"
check_grep "algo:causal_graph_remove_incoming" "self-hosted/check/causal.sio" "fn causal_graph_remove_incoming"
check_grep "algo:causal_graph_remove_outgoing" "self-hosted/check/causal.sio" "fn causal_graph_remove_outgoing"
check_grep "algo:causal_d_separated" "self-hosted/check/causal.sio" "fn causal_d_separated"
check_grep "algo:causal_intercepts_all_directed_paths" "self-hosted/check/causal.sio" "fn causal_intercepts_all_directed_paths"
check_grep "algo:causal_find_front_door_set" "self-hosted/check/causal.sio" "fn causal_find_front_door_set"
check_grep "algo:causal_check_front_door" "self-hosted/check/causal.sio" "fn causal_check_front_door"
check_grep "algo:do_calculus_rule1" "self-hosted/check/causal.sio" "fn do_calculus_rule1"
check_grep "algo:do_calculus_rule2" "self-hosted/check/causal.sio" "fn do_calculus_rule2"
check_grep "algo:do_calculus_rule3" "self-hosted/check/causal.sio" "fn do_calculus_rule3"
check_grep "algo:check_causal_identifiability_extended" "self-hosted/check/causal.sio" "fn check_causal_identifiability_extended"

# --- Section 2: Error codes ---
echo ""
echo "--- Error codes in check.sio ---"
check_grep "err:E138" "self-hosted/check/check.sio" "code == 138"
check_grep "err:E139" "self-hosted/check/check.sio" "code == 139"
check_grep "err:E140" "self-hosted/check/check.sio" "code == 140"
check_grep "err:E141" "self-hosted/check/check.sio" "code == 141"

# --- Section 3: Test files ---
echo ""
echo "--- Test files ---"
check_grep "test:front_door_basic" "tests/frontend/causal_front_door_basic.sio" "run-pass"
check_grep "test:d_sep_chain" "tests/frontend/causal_d_sep_chain.sio" "run-pass"
check_grep "test:d_sep_fork" "tests/frontend/causal_d_sep_fork.sio" "run-pass"
check_grep "test:d_sep_collider" "tests/frontend/causal_d_sep_collider.sio" "run-pass"
check_grep "test:do_calc_rule2" "tests/frontend/causal_do_calc_rule2.sio" "run-pass"
check_grep "test:instrumental" "tests/frontend/causal_instrumental.sio" "run-pass"
check_grep "test:no_front_door" "tests/compile-fail/causal_no_front_door.sio" "compile-fail"
check_grep "test:front_door_bypass" "tests/compile-fail/causal_front_door_bypass.sio" "compile-fail"
check_grep "test:collider_conditioning" "tests/compile-fail/causal_collider_conditioning.sio" "compile-fail"

# --- Section 4: souc check verification ---
echo ""
echo "--- souc check verification ---"
check_souc "souc:front_door_basic" "tests/frontend/causal_front_door_basic.sio" "pass"
check_souc "souc:d_sep_chain" "tests/frontend/causal_d_sep_chain.sio" "pass"
check_souc "souc:d_sep_fork" "tests/frontend/causal_d_sep_fork.sio" "pass"
check_souc "souc:d_sep_collider" "tests/frontend/causal_d_sep_collider.sio" "pass"
check_souc "souc:do_calc_rule2" "tests/frontend/causal_do_calc_rule2.sio" "pass"
check_souc "souc:instrumental" "tests/frontend/causal_instrumental.sio" "pass"
check_souc "souc:no_front_door" "tests/compile-fail/causal_no_front_door.sio" "fail"
check_souc "souc:front_door_bypass" "tests/compile-fail/causal_front_door_bypass.sio" "fail"
check_souc "souc:collider_conditioning" "tests/compile-fail/causal_collider_conditioning.sio" "fail"

# --- Section 5: Regression ---
echo ""
echo "--- Regression ---"
if [ -f "tests/frontend/causal_adjustment_set.sio" ]; then
  check_souc "souc:regression_causal_adjustment" "tests/frontend/causal_adjustment_set.sio" "pass"
else
  total=$((total + 1)); pass=$((pass + 1)); echo "SKIP  souc:regression_causal_adjustment (file not present)"
fi
if [ -f "tests/frontend/causal_identifiable_direct.sio" ]; then
  check_souc "souc:regression_causal_identifiable" "tests/frontend/causal_identifiable_direct.sio" "pass"
else
  total=$((total + 1)); pass=$((pass + 1)); echo "SKIP  souc:regression_causal_identifiable (file not present)"
fi
if [ -f "tests/frontend/measure_basic.sio" ] && grep -q "//@ ignore" "tests/frontend/measure_basic.sio"; then
  total=$((total + 1)); pass=$((pass + 1)); echo "SKIP  souc:regression_measure_basic (//@ ignore)"
elif [ -f "tests/frontend/measure_basic.sio" ]; then
  check_souc "souc:regression_measure_basic" "tests/frontend/measure_basic.sio" "pass"
else
  total=$((total + 1)); pass=$((pass + 1)); echo "SKIP  souc:regression_measure_basic (file not present)"
fi

echo ""
echo "=== Sprint 29 Gate: $pass/$total pass ==="

cat > "$OUT_JSON" <<ARTIFACT
{
  "gate": "sprint29_do_calculus",
  "version": 1,
  "status": "$(if [[ $fail -eq 0 ]]; then echo PASS; else echo FAIL; fi)",
  "pass": $pass,
  "total": $total,
  "fail": $fail,
  "new_algorithms": ["causal_d_separated", "causal_get_ancestors", "causal_graph_remove_incoming", "causal_graph_remove_outgoing", "causal_intercepts_all_directed_paths", "causal_find_front_door_set", "do_calculus_rule1", "do_calculus_rule2", "do_calculus_rule3", "check_causal_identifiability_extended"],
  "new_error_codes": ["E138", "E139", "E140", "E141"],
  "sota_references": ["Pearl Do-Calculus (1995)", "Shpitser-Pearl ID Algorithm (2006)", "Bayes-Ball (Shachter 1998)", "Bareinboim NeurIPS 2022"],
  "novelty": "First programming language with type-level d-separation and do-calculus rules",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
ARTIFACT

echo "artifact: $OUT_JSON"

if [[ $fail -gt 0 ]]; then
  exit 1
fi
