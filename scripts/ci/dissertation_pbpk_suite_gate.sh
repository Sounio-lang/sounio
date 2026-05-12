#!/usr/bin/env bash
# scripts/ci/dissertation_pbpk_suite_gate.sh
#
# Dissertation evidence gate: PBPK validation suite (rapamycin + haloperidol + tirzepatide).
#
# 14 independent tests cover the dissertation's applied PBPK layer — the
# *evidence* that the three core contributions (GUM-through-ODE, compile-time
# confidence, ISO budgets) actually work on real drugs:
#
# Rapamycin (sirolimus) — primary dissertation drug:
#   1. rapamycin_iso_budget        Euler 3-comp, ISO §8 budget, IV bolus 6 mg
#   2. rapamycin_rk4_budget        RK4 3-comp, GUM through 4-stage RK
#   3. rapamycin_epistemic_pbpk    BBB/Pgp clinical claims, AUC-CV, CL inverse
#   4. rapamycin_epistemic_adaptive Bogacki-Shampine 3(2) + variance lookbehind
#   5. rapamycin_gum_vs_mc         GUM linearization vs Monte-Carlo (ratio<10)
#   6. biomaterial_release         Cypher DES — zero/first-order/Higuchi + 14-comp PBPK
#   7. rapamycin_clinical          14-comp clinical validation: brain/blood ratio,
#                                  Vd_ss, GUM budget vs Lampen 1998, Schreiber 1991
#   8. gum_vs_mc                   ISO budget vs Monte-Carlo: 5x cost advantage
#   9. des_sirolimus               Cypher DES extended scenario: cross-domain GUM
#  10. rapamycin_pop_sim           32 virtual patients with lognormal CL/fu/Kp
#
# Haloperidol — second drug for cross-validation of method generality:
#  11. haloperidol_d2_pet          D2 receptor occupancy via PET, Hill-Langmuir
#                                  saturation, therapeutic-window vs EPS-threshold
#  12. haloperidol_oral_pbpk       Oral PBPK with repeated dosing, CYP3A4-CYP2D6
#                                  metabolism, CNS coverage projection
#  13. d2_gum                      D2 receptor PD: GUM uncertainty over kpuu_brain,
#                                  ps_bbb permeability, mixed-evidence confidence
#  14. d2_voi                      D2 value-of-information: which experiment to
#                                  prioritize given current PK/PD evidence
#
# Tirzepatide — drug #3, cross-validates framework boundaries (peptide class):
#  15. tirzepatide_sc_pbpk         2-comp SC ODE (Urva 2022): Tmax, Cmax, t½, AUC,
#                                  GLP-1R/GIPR occupancy (6 tests)
#  16. glp1_gipr_gum               Dual receptor GUM: EC50 sensitivity near EC50,
#                                  combined u_occ, GIPR > GLP-1R at Cmax (4 tests)
#  17. dissertation_tirzepatide_demo ISO budget (7 sources): CL/Ka/fu/Vc/F/EC50×2,
#                                  framework boundary audit, confidence gate PASS
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
  "rapamycin_clinical           stdlib/darwin_pbpk/validation/rapamycin_clinical.sio"
  "gum_vs_mc                    stdlib/darwin_pbpk/validation/gum_vs_mc.sio"
  "des_sirolimus                stdlib/darwin_pbpk/scenarios/des_sirolimus.sio"
  "rapamycin_pop_sim            stdlib/darwin_pbpk/population/pop_sim.sio"
  "haloperidol_d2_pet           stdlib/darwin_pbpk/validation/haloperidol_d2_pet.sio"
  "haloperidol_oral_pbpk        stdlib/darwin_pbpk/validation/haloperidol_oral_pbpk.sio"
  "d2_gum                       stdlib/darwin_pbpk/pd/d2_gum.sio"
  "d2_voi                       stdlib/darwin_pbpk/pd/d2_voi.sio"
  "dissertation_pbpk_rapamycin  examples/dissertation_pbpk_rapamycin.sio"
  "dissertation_oral_pd         examples/dissertation_oral_pd_demo.sio"
  "dissertation_steady_state    examples/dissertation_steady_state_demo.sio"
  "dissertation_steady_state_fullvd examples/dissertation_steady_state_fullvd_demo.sio"
  "dissertation_scenario_gate   examples/dissertation_scenario_gate_demo.sio"
  "rodgers_rowland_kp           stdlib/darwin_pbpk/core/rodgers_rowland.sio"
  "gnn_rapamycin_inference      stdlib/darwin_pbpk/ml/gnn_inference.sio"
  "hybrid_ode_rapamycin         stdlib/darwin_pbpk/ml/hybrid_ode.sio"
  "dissertation_hybrid_demo     examples/dissertation_hybrid_demo.sio"
  "tirzepatide_sc_pbpk          stdlib/darwin_pbpk/validation/tirzepatide_sc_pbpk.sio"
  "glp1_gipr_gum                stdlib/darwin_pbpk/pd/glp1_gipr_gum.sio"
  "dissertation_tirzepatide_demo examples/dissertation_tirzepatide_demo.sio"
  "vancomycin_icu_pbpk          stdlib/darwin_pbpk/validation/vancomycin_icu_pbpk.sio"
  "vancomycin_auc_gum           stdlib/darwin_pbpk/pd/vancomycin_auc_gum.sio"
  "dissertation_vancomycin_demo examples/dissertation_vancomycin_demo.sio"
)

# Smoke entries: artifact-emitting demos (HTML, SVG, narrative reports).
# These don't have PASS markers because they aren't test runners — they
# render dissertation visuals for figures/appendices. Gate only checks
# rc=0 and that stdout has at least 100 bytes.
TESTS_SMOKE=(
  "dissertation_demo            examples/dissertation_demo.sio"
  "dissertation_interactive     examples/dissertation_interactive.sio"
  "dissertation_plot            examples/dissertation_plot.sio"
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

  if ! grep -qE "^PASS$|^ALL PASS$|ALL [0-9]+ TESTS PASSED|^ *(DEMO|SS DEMO|SS FULLVD|SCENARIO GATE|PK/PD DEMO) OK$" "$log"; then
    echo "  FAIL: no PASS marker in stdout"
    tail -5 "$log" | sed 's/^/    /'
    fails=$((fails + 1))
    results+=("FAIL  $name  no_pass_marker")
    continue
  fi

  echo "  PASS (log=$log)"
  results+=("PASS  $name")
done

# Smoke tests: dissertation visualisation/narrative demos. rc=0 + non-trivial
# output. No PASS marker required.
for entry in "${TESTS_SMOKE[@]}"; do
  name="${entry%% *}"
  src="${entry##* }"
  log="$STAGE_DIR/$name.log"

  echo ""
  echo "[$name] (smoke)"
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

  out_bytes=$(wc -c < "$log")
  if [[ $out_bytes -lt 100 ]]; then
    echo "  FAIL: output too short ($out_bytes bytes < 100)"
    fails=$((fails + 1))
    results+=("FAIL  $name  short_output")
    continue
  fi

  echo "  PASS (rc=0, ${out_bytes}B emitted, log=$log)"
  results+=("PASS  $name (smoke)")
done

total=$((${#TESTS[@]} + ${#TESTS_SMOKE[@]}))

echo ""
echo "=== Summary ==="
for r in "${results[@]}"; do
  echo "  $r"
done

if [[ $fails -ne 0 ]]; then
  echo ""
  echo "dissertation_pbpk_suite_gate: FAIL ($fails / $total tests failed)"
  exit 1
fi

echo ""
echo "dissertation_pbpk_suite_gate: PASS ($total/$total rapamycin PBPK tests + smoke demos)"
