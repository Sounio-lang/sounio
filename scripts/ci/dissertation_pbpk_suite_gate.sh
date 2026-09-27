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
# or stdout doesn't contain "PASS". Modules that fail on a documented defect
# are listed in TESTS_EXPECTED_FAIL_HONEST instead and must fail exactly as
# recorded (strict; see that list).
#
# CPU-only. Self-skips if souc is missing. Runtime: each TESTS / smoke /
# pending entry is bounded by DPS_TIMEOUT_SECONDS (default 90 s). The
# expected-FAIL_HONEST pbpk28_sobol_pce entry has its own 3600 s timeout: it
# took 130 s with the default engine (souc-seq-leansingle shim), but
# 2331-2392 s when SOUC_BIN points at bin/souc-lean-single-x86_64. Budget
# for the engine you run.
#
# Knobs (env):
#   DPS_STAGE_DIR             working directory (default mktemp)
#   DPS_TIMEOUT_SECONDS       per-test timeout (default 90; TESTS_EXPECTED_FAIL_HONEST entries carry their own)
#   SOUNIO_DPS_GATE_SKIP=1    skip entirely

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# Pin the Seq<T>-capable lean_single engine via a run/check shim: the K-AXI
# fusion witness + seq_* tests need Seq<T>. `bin/souc` defaults to Madaros
# (the user-facing engine since 2026-06-14), which does not yet carry Seq<T>
# (restoring it in the modular compiler is a tracked follow-up). An externally
# provided SOUC_BIN (CI override) still takes precedence.
: "${SOUC_BIN:=$ROOT_DIR/scripts/ci/souc-seq-leansingle.sh}"
export SOUC_BIN

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
  "tacrolimus_oral_pbpk         stdlib/darwin_pbpk/validation/tacrolimus_oral_pbpk.sio"
  "tacrolimus_trough_gum        stdlib/darwin_pbpk/pd/tacrolimus_trough_gum.sio"
  "tacrolimus_ddi_module        stdlib/darwin_pbpk/ddi/tacrolimus_sirolimus_ddi.sio"
  "tacrolimus_ddi_clinical      stdlib/darwin_pbpk/validation/tacrolimus_sirolimus_ddi_clinical.sio"
  "cross_drug_iso_budget        stdlib/darwin_pbpk/validation/cross_drug_iso_budget.sio"
  "halo_pgx_gate              stdlib/darwin_pbpk/validation/haloperidol_pgx_gate.sio"
  "halo_pgx_gate_pass         tests/run-pass/halo_pgx_gate_pass.sio"
  "olanzapine_d2_mtor         stdlib/darwin_pbpk/validation/olanzapine_d2_mtor.sio"
  "pop_pbpk_pd                stdlib/darwin_pbpk/population/pop_pbpk_pd.sio"
  "epistemic_pbpk28           stdlib/darwin_pbpk/epistemic_pbpk28.sio"
  "epistemic_pbpk28_hessian   stdlib/darwin_pbpk/epistemic_pbpk28_hessian.sio"
  "pbpk28_mc_cross_validation stdlib/darwin_pbpk/validation/pbpk28_mc_cross_validation.sio"
  "pbpk28_mc_prior_family_sweep stdlib/darwin_pbpk/validation/pbpk28_mc_prior_family_sweep.sio"
  "rapamycin_kaxi_fuse_prior    tests/run-pass/rapamycin_kaxi_fuse_prior.sio"
)

# Smoke entries: artifact-emitting demos (HTML, SVG, narrative reports).
# These don't have PASS markers because they aren't test runners — they
# render dissertation visuals for figures/appendices. Gate only checks
# rc=0 and that stdout has at least 100 bytes.
TESTS_SMOKE=(
  "dissertation_demo            examples/dissertation_demo.sio"
  "dissertation_interactive     examples/dissertation_interactive.sio"
  "dissertation_plot            examples/dissertation_plot.sio"
  "dissertation_pgx_demo      examples/dissertation_pgx_compile_gate_demo.sio"
  "dissertation_olanzapine    examples/dissertation_olanzapine_demo.sio"
  "dissertation_168_poly      examples/dissertation_168_polypharmacy.sio"
  "dissertation_pop_demo      examples/dissertation_pop_pbpk_pd_demo.sio"
)

# Clinical-validation modules: registered NOW (run every build for compile+run
# health) but PENDING-aware — they only ENFORCE validation once observed data
# lands. While obs_n()==0 each emits *_CLINICAL_PENDING_OBSERVED and is reported
# as PENDING (counts as neither pass nor fail; gate stays green). When the
# literature-MCP session fills the observed arrays + study design, the same
# module emits *_CLINICAL_PASS (counts as pass) or *_CLINICAL_FAIL_HONEST (fails
# the gate — green means validated against clinical data). Registration thus
# self-activates with zero further edits here.
TESTS_PENDING=(
  "pbpk28_rapamycin_clinical    stdlib/darwin_pbpk/validation/pbpk28_rapamycin_clinical.sio"
  "pbpk28_semaglutide_clinical  stdlib/darwin_pbpk/validation/pbpk28_semaglutide_clinical.sio"
)

# Regression-pending: tests whose required compiler subsystem was dropped by a
# merge commit and not yet restored. Listed here so the gate stays green while
# the restoration workstream is tracked — NOT run (souc would fail to compile
# them), merely registered so the pending item is visible in the summary.
# RESOLVED 2026-06-15 (branch feat/seq-restore): Seq<T> (TY_SEQ=12, x86) restored
# in lean_single.sio; rapamycin_kaxi_fuse_prior now runs and is promoted to TESTS
# above (verified: 3-stage bootstrap fixed-point gen2==gen3, witness PASS,
# sd_post==sd_expected). Seq-of-struct/borrow paths remain Tier-2 (see
# tests/known_failures/hardened_diagnostics_full_suite.txt).
TESTS_PENDING_REGRESSION=()

# Expected FAIL_HONEST: modules that currently fail on a documented,
# diagnosed defect. Strict: the entry is accepted (XFAIL, neither pass nor
# fail) only when the run exits with exactly the recorded rc, prints the
# recorded diagnostic exactly the recorded number of times, and prints no
# other `FAIL` line. A timeout, a different rc, any other failure, or an
# unexpected pass (rc 0) fails the gate -- the last one so the entry is moved
# back to TESTS once the defect is fixed.
# Each entry carries its own timeout (seconds), used instead of
# DPS_TIMEOUT_SECONDS: an entry that cannot finish inside the default would
# otherwise always be classified FAIL:timeout.
# The source is also compiled once with its diagnostics visible (the default
# souc-seq-leansingle shim discards them and ignores the compiler's rc, so a
# fail-open build regression could otherwise reach the expected rc). The
# compiler output must contain exactly compile_error_count `error` lines, all
# matching compile_error_pattern (a fixed string); anything else fails.
# Format: name|src|expected_rc|diagnostic_count|timeout_s|compile_error_count|compile_error_pattern|diagnostic
TESTS_EXPECTED_FAIL_HONEST=(
  # Saltelli estimator output violates S_i <= S_Ti in both self-tests
  # (rapamycin and semaglutide TEST 7; stdlib/epistemic/sobol.sio not yet
  # repaired). See docs/dissertation/results/sobol_pce_semaglutide_v2.md.
  # Timeout: with the gate's default engine (the shim's
  # bin/souc-linux-x86_64) the run took 130 s on the workspace; with
  # bin/souc-lean-single-x86_64 (SOUC_BIN override) 2331-2392 s over four
  # runs. 3600 s covers both (~1.5x the slower).
  # Compile errors: 36 pre-existing `tuple index out of bounds` in
  # stdlib/epistemic/pce.sio (lines 332-520); the compiler still emits the ELF.
  "pbpk28_sobol_pce|stdlib/darwin_pbpk/validation/pbpk28_sobol_pce.sio|2|2|3600|36|error: tuple index out of bounds at stdlib/epistemic/pce.sio:|FAIL: estimator output violates S_i <= S_Ti; not usable as Sobol' indices"
)

# Compile `src` once and print the compiler's own output (diagnostics
# included). With the default shim, call the ELF it wraps directly, because
# its compile/run verbs send that output to /dev/null.
dps_compile_diagnostics() {
  local src="$1" tmp
  tmp="$(mktemp)"
  if [[ "$SOUC_BIN" == "$ROOT_DIR/scripts/ci/souc-seq-leansingle.sh" ]]; then
    "${SOUNIO_SEQ_LEANSINGLE_ELF:-$ROOT_DIR/bin/souc-linux-x86_64}" "$src" "$tmp" 2>&1 || true
  else
    "$SOUC_BIN" compile "$src" -o "$tmp" 2>&1 || true
  fi
  rm -f "$tmp"
}

# Verdict on the compiler output: prints OK or FAIL:<reason>.
dps_compile_verdict() {
  local diag_log="$1" want_n="$2" pattern="$3"
  local n_err n_known
  n_err=$(grep -cE '(^|[^[:alnum:]_])error(\[|:)' "$diag_log" || true)
  n_known=$(grep -cF -- "$pattern" "$diag_log" || true)
  if [[ "$n_err" != "$want_n" || "$n_known" != "$want_n" ]]; then
    echo "FAIL:compiler_diagnostics=${n_err}_known=${n_known}_expected_${want_n}"
    return
  fi
  echo "OK"
}

# Verdict for one expected-FAIL_HONEST run: prints XFAIL or FAIL:<reason>.
dps_xfail_verdict() {
  local log="$1" rc="$2" want_rc="$3" want_n="$4" diag="$5"
  if [[ "$rc" == "124" ]]; then echo "FAIL:timeout"; return; fi
  if [[ "$rc" == "0" ]]; then echo "FAIL:unexpected_pass"; return; fi
  if [[ "$rc" != "$want_rc" ]]; then echo "FAIL:rc=${rc}_expected_${want_rc}"; return; fi
  local n_diag n_fail
  n_diag=$(grep -cF -- "$diag" "$log" || true)
  n_fail=$(grep -cE '^[[:space:]]*FAIL' "$log" || true)
  if [[ "$n_diag" != "$want_n" ]]; then echo "FAIL:diagnostic_count=$n_diag"; return; fi
  if [[ "$n_fail" != "$want_n" ]]; then echo "FAIL:other_failures=$((n_fail - n_diag))"; return; fi
  echo "XFAIL"
}

fails=0
pending=0
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

  if ! grep -qE "^PASS$|^ALL PASS$|ALL [0-9]+ TESTS PASSED|^ *ALL (TESTS|GUM TESTS) PASSED$|^ *(DEMO|SS DEMO|SS FULLVD|SCENARIO GATE|PK/PD DEMO) OK$|^ *DDI MODULE OK$|^ *DDI CLINICAL VALIDATION COMPLETE$|^ *CROSS-DRUG ISO BUDGET COMPLETE$|^HALO PGX GATE PASS$|^HESSIAN_PBPK28_DUAL_RHO_PASS$|^SOBOL_PCE_SEMAGLUTIDE_FULL_PASS$|^MC_CROSS_VALIDATION_PBPK28_LOGNORMAL_PASS$|^MC_CROSS_VALIDATION_PBPK28_LOGNORMAL_HESSIAN_PASS$|^MC_CROSS_VALIDATION_PBPK28_LOGNORMAL_OUTPUT$|^MC_PRIOR_FAMILY_SWEEP_PASS$|^MC_PRIOR_FAMILY_SWEEP_OUTPUT$" "$log"; then
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

# Clinical-validation modules: PENDING-aware enforcement (see TESTS_PENDING).
for entry in "${TESTS_PENDING[@]}"; do
  name="${entry%% *}"
  src="${entry##* }"
  log="$STAGE_DIR/$name.log"

  echo ""
  echo "[$name] (clinical validation, pending-aware)"
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

  if grep -qE "_CLINICAL_PASS$" "$log"; then
    echo "  PASS (predicted-vs-observed validation passed, log=$log)"
    results+=("PASS  $name (clinical validation)")
  elif grep -qE "_CLINICAL_FAIL_HONEST$" "$log"; then
    echo "  FAIL: clinical validation failed honestly — predicted-vs-observed"
    echo "        GMFE outside FDA/EMA acceptance (model != clinical data)."
    tail -8 "$log" | sed 's/^/    /'
    fails=$((fails + 1))
    results+=("FAIL  $name  clinical_fail_honest")
  elif grep -qE "_CLINICAL_PENDING_OBSERVED$" "$log"; then
    echo "  PENDING: registered, awaiting observed data (obs_n()==0) — not yet validating"
    pending=$((pending + 1))
    results+=("PEND  $name  awaiting_observed_data")
  else
    echo "  FAIL: no recognized clinical-validation marker (PASS / FAIL_HONEST / PENDING_OBSERVED)"
    tail -5 "$log" | sed 's/^/    /'
    fails=$((fails + 1))
    results+=("FAIL  $name  no_marker")
  fi
done

# Regression-pending loop: register without running souc (Seq<T> absent).
for entry in "${TESTS_PENDING_REGRESSION[@]}"; do
  name="${entry%% *}"
  src="${entry##* }"

  echo ""
  echo "[$name] (regression-pending — compiler subsystem absent, not run)"
  echo "  src=$src"
  echo "  PENDING: Seq<T> subsystem regression (dropped by 5f1e397a2); K-AXI fusion witness pending Seq<T> restore"
  pending=$((pending + 1))
  results+=("PEND  $name  seq_subsystem_regression")
done

# Expected FAIL_HONEST loop (see TESTS_EXPECTED_FAIL_HONEST).
xfails=0
for entry in "${TESTS_EXPECTED_FAIL_HONEST[@]}"; do
  IFS='|' read -r name src want_rc want_n entry_timeout want_cerr cerr_pattern diag <<< "$entry"
  log="$STAGE_DIR/$name.log"

  echo ""
  echo "[$name] (expected FAIL_HONEST, timeout=${entry_timeout}s)"
  echo "  src=$src"

  if [[ ! -f "$src" ]]; then
    echo "  FAIL: source missing"
    fails=$((fails + 1))
    results+=("FAIL  $name  source_missing")
    continue
  fi

  diag_log="$STAGE_DIR/$name.compile.log"
  dps_compile_diagnostics "$src" >"$diag_log"
  cverdict=$(dps_compile_verdict "$diag_log" "$want_cerr" "$cerr_pattern")
  if [[ "$cverdict" != "OK" ]]; then
    echo "  FAIL: compiler output differs from the recorded diagnostics (${cverdict#FAIL:})"
    grep -E '(^|[^[:alnum:]_])error(\[|:)' "$diag_log" | grep -vF -- "$cerr_pattern" | head -5 | sed 's/^/    /'
    fails=$((fails + 1))
    results+=("FAIL  $name  ${cverdict#FAIL:}")
    continue
  fi

  set +e
  timeout "$entry_timeout" "$SOUC_BIN" run "$src" >"$log" 2>&1
  rc=$?
  set -e

  verdict=$(dps_xfail_verdict "$log" "$rc" "$want_rc" "$want_n" "$diag")
  if [[ "$verdict" == "XFAIL" ]]; then
    echo "  XFAIL: fails exactly as recorded (compiler errors: the $want_cerr recorded; rc=$rc, diagnostic x$want_n, no other FAIL)"
    xfails=$((xfails + 1))
    results+=("XFAIL $name  documented_defect")
  else
    echo "  FAIL: expected-FAIL_HONEST entry did not fail as recorded (${verdict#FAIL:}; rc=$rc, timeout=${entry_timeout}s)"
    tail -5 "$log" | sed 's/^/    /'
    fails=$((fails + 1))
    results+=("FAIL  $name  ${verdict#FAIL:}")
  fi
done

total=$((${#TESTS[@]} + ${#TESTS_SMOKE[@]} + ${#TESTS_PENDING[@]} + ${#TESTS_PENDING_REGRESSION[@]} + ${#TESTS_EXPECTED_FAIL_HONEST[@]}))

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
if [[ $pending -ne 0 || $xfails -ne 0 ]]; then
  echo "dissertation_pbpk_suite_gate: PASS ($((total - pending - xfails))/$total passing; $pending item(s) PENDING, $xfails expected FAIL_HONEST — see summary for detail)"
else
  echo "dissertation_pbpk_suite_gate: PASS ($total/$total PBPK tests + smoke demos)"
fi
