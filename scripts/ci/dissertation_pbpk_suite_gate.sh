#!/usr/bin/env bash
# scripts/ci/dissertation_pbpk_suite_gate.sh
#
# Dissertation evidence gate: rapamycin (sirolimus) PBPK validation suite.
#
# Five independent rapamycin tests cover the dissertation's applied PBPK
# layer — the *evidence* that the three core contributions (GUM-through-ODE,
# compile-time confidence, ISO budgets) actually work on a real drug:
#
#   1. rapamycin_iso_budget        Euler 3-comp, ISO §8 budget, IV bolus 6 mg
#   2. rapamycin_rk4_budget        RK4 3-comp, GUM through 4-stage RK
#   3. rapamycin_epistemic_pbpk    BBB/Pgp clinical claims, AUC-CV, CL inverse
#   4. rapamycin_epistemic_adaptive Bogacki-Shampine 3(2) + variance lookbehind
#   5. rapamycin_gum_vs_mc         GUM linearization vs Monte-Carlo (ratio<10)
#   6. biomaterial_release         Cypher DES — zero/first-order/Higuchi + 14-comp PBPK
#
# Each test ends with "PASS\n" on success. Gate fails if any test rc != 0
# or stdout doesn't contain "PASS".
#
# CPU-only, ~30s total. Self-skips if souc is missing.
#
# Knobs (env):
#   DPS_STAGE_DIR             working directory (default mktemp)
#   DPS_TIMEOUT_SECONDS       per-test timeout (default 90)
#   SOUNIO_DPS_GATE_SKIP=1    skip entirely

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ "${SOUNIO_DPS_GATE_SKIP:-0}" == "1" ]]; then
  echo "dissertation_pbpk_suite_gate: SKIPPED (SOUNIO_DPS_GATE_SKIP=1)"
  exit 0
fi

source "$ROOT_DIR/scripts/lib/resolve_souc.sh"
sounio_require_souc

STAGE_DIR="${DPS_STAGE_DIR:-$(mktemp -d /tmp/dissertation_pbpk_suite_XXXXXX)}"
TIMEOUT_SECONDS="${DPS_TIMEOUT_SECONDS:-90}"
mkdir -p "$STAGE_DIR"

echo "=== Dissertation PBPK Suite Gate ==="
echo "  souc=$SOUC_BIN"
echo "  stage_dir=$STAGE_DIR"
echo "  timeout=${TIMEOUT_SECONDS}s per test"

TESTS=(
  "rapamycin_iso_budget         tests/run-pass/rapamycin_iso_budget.sio"
  "rapamycin_rk4_budget         tests/run-pass/rapamycin_rk4_budget.sio"
  "rapamycin_epistemic_pbpk     tests/run-pass/rapamycin_epistemic_pbpk.sio"
  "rapamycin_epistemic_adaptive tests/run-pass/rapamycin_epistemic_adaptive.sio"
  "rapamycin_gum_vs_mc          tests/run-pass/rapamycin_gum_vs_mc.sio"
  "biomaterial_release          stdlib/darwin_pbpk/release/biomaterial_release.sio"
)

fails=0
results=()

for entry in "${TESTS[@]}"; do
  name="${entry%% *}"
  src="${entry##* }"
  log="$STAGE_DIR/$name.log"

  echo ""
  echo "[$name]"
  echo "  src=$src"

  if [[ ! -f "$src" ]]; then
    echo "  FAIL: source missing"
    fails=$((fails + 1))
    results+=("FAIL  $name  source_missing")
    continue
  fi

  set +e
  timeout "$TIMEOUT_SECONDS" "$SOUC_BIN" run "$src" >"$log" 2>&1
  rc=$?
  set -e

  if [[ $rc -ne 0 ]]; then
    echo "  FAIL: rc=$rc (timeout=$TIMEOUT_SECONDS)"
    tail -5 "$log" | sed 's/^/    /'
    fails=$((fails + 1))
    results+=("FAIL  $name  rc=$rc")
    continue
  fi

  if ! grep -qE "^PASS$|ALL [0-9]+ TESTS PASSED" "$log"; then
    echo "  FAIL: no PASS marker in stdout"
    tail -5 "$log" | sed 's/^/    /'
    fails=$((fails + 1))
    results+=("FAIL  $name  no_pass_marker")
    continue
  fi

  echo "  PASS (log=$log)"
  results+=("PASS  $name")
done

echo ""
echo "=== Summary ==="
for r in "${results[@]}"; do
  echo "  $r"
done

if [[ $fails -ne 0 ]]; then
  echo ""
  echo "dissertation_pbpk_suite_gate: FAIL ($fails / ${#TESTS[@]} tests failed)"
  exit 1
fi

echo ""
echo "dissertation_pbpk_suite_gate: PASS (${#TESTS[@]}/${#TESTS[@]} rapamycin PBPK tests)"
