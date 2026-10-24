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
# CPU-only. Fails (does not skip) if an engine driver is missing. Runtime:
# each TESTS / smoke / pending entry is bounded by DPS_TIMEOUT_SECONDS
# (default 90 s, compile + run). The expected-FAIL_HONEST pbpk28_sobol_pce
# entry has its own 3600 s timeout: it took 130 s with the shim's ELF, but
# 2331-2392 s on bin/souc-lean-single-x86_64. Budget for the engine you run.
#
# ENGINES (P0.5, 2026-10-06). Every entry names the engine its claim is run
# on, and the per-test reason is recorded in
# docs/dissertation/pbpk_claim_truth_table.md ("Suite gate: engine per test").
#   madaros      bin/madaros -- the default user-facing engine (CLAUDE.md §4).
#                Every entry that passes there runs there, including the two
#                compiler-native variance witnesses (rapamycin_rk4_budget,
#                rapamycin_epistemic_adaptive), whose FAMILY_A_VAR_LIVE claim is
#                only defined where variance_of() survives `.value`.
#   lean_single  scripts/ci/souc-seq-leansingle.sh, a run/check shim over
#                bin/souc-linux-x86_64 (2026-06-16 snapshot). Used only for
#                entries Madaros refuses or cannot finish (E259/E001/E137
#                type-check refusals, "madaros: handles full" at run time).
#                That ELF strips the Knowledge variance channel at `.value`
#                (variance_of() returns 0), so an entry pinned here whose source
#                calls variance_of()/uncertainty_of() FAILS the gate (refused,
#                not run): its claim would be vacuous on this engine.
# The rk4/adaptive EPISTEMIC_FABRICATION self-checks stay in the tests: they
# are what caught the lean_single shim reporting variance 0.
#
# Knobs (env):
#   DPS_STAGE_DIR             working directory (default mktemp)
#   DPS_TIMEOUT_SECONDS       per-test timeout (default 90; TESTS_EXPECTED_FAIL_HONEST entries carry their own)
#   DPS_MADAROS_BIN           Madaros launcher (default bin/madaros)
#   DPS_LEAN_SINGLE_BIN       lean_single run/check driver (default the shim above)
#   DPS_SINGLE_ENGINE_BIN     A/B diagnostics only: run EVERY entry with this
#                             driver (`<bin> run <src>`), ignoring the pins. The
#                             variance-channel refusal still applies. Ambient
#                             SOUC_BIN is ignored (it is often a different
#                             checkout's compiler on the cluster pods).
#   SOUNIO_DPS_GATE_SKIP=1    skip entirely

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# Per-entry engines (see ENGINES above). Until 2026-10-06 the whole gate was
# pinned to the lean_single shim because "Madaros does not yet carry Seq<T>".
# Measured 2026-10-06 on main 67cbf8797: Madaros runs the gate's only Seq<T>
# entry (rapamycin_kaxi_fuse_prior) and matches the closed-form posterior SD, so
# that is no longer a reason to pin anything; the shim stays only for entries
# Madaros refuses.
SHIM="$ROOT_DIR/scripts/ci/souc-seq-leansingle.sh"
MADAROS_BIN="${DPS_MADAROS_BIN:-$ROOT_DIR/bin/madaros}"
LEAN_SINGLE_BIN="${DPS_LEAN_SINGLE_BIN:-$SHIM}"
SINGLE_ENGINE_BIN="${DPS_SINGLE_ENGINE_BIN:-}"
# The shim and the madaros launcher read the stdlib from the tree.
export SOUNIO_STDLIB_PATH="$ROOT_DIR/stdlib"

if [[ "${SOUNIO_DPS_GATE_SKIP:-0}" == "1" ]]; then
  echo "dissertation_pbpk_suite_gate: SKIPPED (SOUNIO_DPS_GATE_SKIP=1)"
  exit 0
fi

for _b in "$MADAROS_BIN" "$LEAN_SINGLE_BIN" ${SINGLE_ENGINE_BIN:+"$SINGLE_ENGINE_BIN"}; do
  if [[ ! -x "$_b" ]]; then
    echo "dissertation_pbpk_suite_gate: FAIL (engine driver missing or not executable: $_b)" >&2
    exit 1
  fi
done
# Materialize the committed Madaros prebuilt up front (bin/madaros would do it
# on first use; doing it here makes a corrupt/missing prebuilt fail once, loudly).
if [[ "$MADAROS_BIN" == "$ROOT_DIR/bin/madaros" && -z "$SINGLE_ENGINE_BIN" ]]; then
  if ! bash "$ROOT_DIR/scripts/lib/materialize_madaros_prebuilt.sh" >/dev/null; then
    echo "dissertation_pbpk_suite_gate: FAIL (Madaros prebuilt could not be materialized)" >&2
    exit 1
  fi
fi

STAGE_DIR="${DPS_STAGE_DIR:-$(mktemp -d /tmp/dissertation_pbpk_suite_XXXXXX)}"
TIMEOUT_SECONDS="${DPS_TIMEOUT_SECONDS:-90}"
mkdir -p "$STAGE_DIR"

echo "=== Dissertation PBPK Suite Gate ==="
if [[ -n "$SINGLE_ENGINE_BIN" ]]; then
  echo "  SINGLE-ENGINE override: every entry runs on $SINGLE_ENGINE_BIN (per-test pins ignored)"
else
  echo "  engine madaros=$MADAROS_BIN"
  echo "  engine lean_single=$LEAN_SINGLE_BIN"
fi
[[ -n "${SOUC_BIN:-}" ]] && echo "  note: ambient SOUC_BIN=$SOUC_BIN is ignored (use DPS_SINGLE_ENGINE_BIN)"
echo "  stage_dir=$STAGE_DIR"
echo "  timeout=${TIMEOUT_SECONDS}s per test"

TESTS=(
  "rapamycin_iso_budget               madaros     tests/run-pass/rapamycin_iso_budget.sio"
  "rapamycin_rk4_budget               madaros     tests/run-pass/rapamycin_rk4_budget.sio"
  "rapamycin_epistemic_pbpk           madaros     tests/run-pass/rapamycin_epistemic_pbpk.sio"
  "rapamycin_epistemic_adaptive       madaros     tests/run-pass/rapamycin_epistemic_adaptive.sio"
  "rapamycin_gum_vs_mc                madaros     tests/run-pass/rapamycin_gum_vs_mc.sio"
  "biomaterial_release                madaros     stdlib/darwin_pbpk/release/biomaterial_release.sio"
  "rapamycin_clinical                 lean_single stdlib/darwin_pbpk/validation/rapamycin_clinical.sio"
  "gum_vs_mc                          lean_single stdlib/darwin_pbpk/validation/gum_vs_mc.sio"
  "des_sirolimus                      madaros     stdlib/darwin_pbpk/scenarios/des_sirolimus.sio"
  "rapamycin_pop_sim                  lean_single stdlib/darwin_pbpk/population/pop_sim.sio"
  "haloperidol_d2_pet                 madaros     stdlib/darwin_pbpk/validation/haloperidol_d2_pet.sio"
  "haloperidol_oral_pbpk              madaros     stdlib/darwin_pbpk/validation/haloperidol_oral_pbpk.sio"
  "d2_gum                             lean_single stdlib/darwin_pbpk/pd/d2_gum.sio"
  "d2_voi                             lean_single stdlib/darwin_pbpk/pd/d2_voi.sio"
  "dissertation_pbpk_rapamycin        madaros     examples/dissertation_pbpk_rapamycin.sio"
  "dissertation_oral_pd               lean_single examples/dissertation_oral_pd_demo.sio"
  "dissertation_steady_state          lean_single examples/dissertation_steady_state_demo.sio"
  "dissertation_steady_state_fullvd   lean_single examples/dissertation_steady_state_fullvd_demo.sio"
  "dissertation_scenario_gate         lean_single examples/dissertation_scenario_gate_demo.sio"
  "rodgers_rowland_kp                 madaros     stdlib/darwin_pbpk/core/rodgers_rowland.sio"
  "gnn_rapamycin_inference            madaros     stdlib/darwin_pbpk/ml/gnn_inference.sio"
  "hybrid_ode_rapamycin               madaros     stdlib/darwin_pbpk/ml/hybrid_ode.sio"
  "dissertation_hybrid_demo           madaros     examples/dissertation_hybrid_demo.sio"
  "tirzepatide_sc_pbpk                madaros     stdlib/darwin_pbpk/validation/tirzepatide_sc_pbpk.sio"
  "glp1_gipr_gum                      madaros     stdlib/darwin_pbpk/pd/glp1_gipr_gum.sio"
  "dissertation_tirzepatide_demo      madaros     examples/dissertation_tirzepatide_demo.sio"
  "vancomycin_icu_pbpk                madaros     stdlib/darwin_pbpk/validation/vancomycin_icu_pbpk.sio"
  "vancomycin_auc_gum                 madaros     stdlib/darwin_pbpk/pd/vancomycin_auc_gum.sio"
  "dissertation_vancomycin_demo       madaros     examples/dissertation_vancomycin_demo.sio"
  "tacrolimus_oral_pbpk               madaros     stdlib/darwin_pbpk/validation/tacrolimus_oral_pbpk.sio"
  "tacrolimus_trough_gum              madaros     stdlib/darwin_pbpk/pd/tacrolimus_trough_gum.sio"
  "tacrolimus_ddi_module              madaros     stdlib/darwin_pbpk/ddi/tacrolimus_sirolimus_ddi.sio"
  "tacrolimus_ddi_clinical            madaros     stdlib/darwin_pbpk/validation/tacrolimus_sirolimus_ddi_clinical.sio"
  "cross_drug_iso_budget              madaros     stdlib/darwin_pbpk/validation/cross_drug_iso_budget.sio"
  "halo_pgx_gate                      madaros     stdlib/darwin_pbpk/validation/haloperidol_pgx_gate.sio"
  "halo_pgx_gate_pass                 madaros     tests/run-pass/halo_pgx_gate_pass.sio"
  "olanzapine_d2_mtor                 madaros     stdlib/darwin_pbpk/validation/olanzapine_d2_mtor.sio"
  "pop_pbpk_pd                        madaros     stdlib/darwin_pbpk/population/pop_pbpk_pd.sio"
  "epistemic_pbpk28                   madaros     stdlib/darwin_pbpk/epistemic_pbpk28.sio"
  "epistemic_pbpk28_hessian           madaros     stdlib/darwin_pbpk/epistemic_pbpk28_hessian.sio"
  "pbpk28_mc_cross_validation         lean_single stdlib/darwin_pbpk/validation/pbpk28_mc_cross_validation.sio"
  "pbpk28_mc_prior_family_sweep       lean_single stdlib/darwin_pbpk/validation/pbpk28_mc_prior_family_sweep.sio"
  "rapamycin_kaxi_fuse_prior          madaros     tests/run-pass/rapamycin_kaxi_fuse_prior.sio"
)

# Smoke entries: artifact-emitting demos (HTML, SVG, narrative reports).
# These don't have PASS markers because they aren't test runners — they
# render dissertation visuals for figures/appendices. Gate only checks
# rc=0 and that stdout has at least 100 bytes.
TESTS_SMOKE=(
  "dissertation_demo                  madaros     examples/dissertation_demo.sio"
  "dissertation_interactive           madaros     examples/dissertation_interactive.sio"
  "dissertation_plot                  madaros     examples/dissertation_plot.sio"
  "dissertation_pgx_demo              madaros     examples/dissertation_pgx_compile_gate_demo.sio"
  "dissertation_olanzapine            madaros     examples/dissertation_olanzapine_demo.sio"
  "dissertation_168_poly              lean_single examples/dissertation_168_polypharmacy.sio"
  "dissertation_pop_demo              madaros     examples/dissertation_pop_pbpk_pd_demo.sio"
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
  "pbpk28_rapamycin_clinical          lean_single stdlib/darwin_pbpk/validation/pbpk28_rapamycin_clinical.sio"
  "pbpk28_semaglutide_clinical        madaros     stdlib/darwin_pbpk/validation/pbpk28_semaglutide_clinical.sio"
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
# Format: name|engine|src|expected_rc|diagnostic_count|timeout_s|compile_error_count|compile_error_pattern|diagnostic
TESTS_EXPECTED_FAIL_HONEST=(
  # Saltelli estimator output violates S_i <= S_Ti in both self-tests
  # (rapamycin and semaglutide TEST 7; stdlib/epistemic/sobol.sio not yet
  # repaired). See docs/dissertation/results/sobol_pce_semaglutide_v2.md.
  # Engine: lean_single. Madaros refuses the module at type check (E259
  # "struct field is private", measured 2026-10-06), so the recorded defect
  # is not observable there. The module reads no compiler-native variance.
  # Timeout: with the gate's default engine (the shim's
  # bin/souc-linux-x86_64) the run took 130 s on the workspace (160 s on
  # 2026-10-06 under load); with bin/souc-lean-single-x86_64 (override)
  # 2331-2392 s over four runs. 3600 s covers both (~1.5x the slower).
  # Compile errors: 36 pre-existing `tuple index out of bounds` in
  # stdlib/epistemic/pce.sio (lines 332-520); the compiler still emits the ELF.
  "pbpk28_sobol_pce|lean_single|stdlib/darwin_pbpk/validation/pbpk28_sobol_pce.sio|2|2|3600|36|error: tuple index out of bounds at stdlib/epistemic/pce.sio:|FAIL: estimator output violates S_i <= S_Ti; not usable as Sobol' indices"
)

# Engine label actually used for an entry pinned to `engine`.
dps_effective_engine() {
  if [[ -n "$SINGLE_ENGINE_BIN" ]]; then echo "single:$SINGLE_ENGINE_BIN"; else echo "$1"; fi
}

# Fail-closed pin check (P0.5 part C). The lean_single shim's ELF strips the
# Knowledge variance channel at `.value`: variance_of()/uncertainty_of() then
# return 0 and a claim built on them is vacuous there (measured 2026-10-06:
# rapamycin_iso_budget prints var(blood)=0.000000 on the shim, 0.000031 on
# Madaros and on bin/souc-lean-single-x86_64). Refuse such a pin instead of
# running it. Prints a reason and returns 1 when refused.
dps_variance_pin_refusal() {
  local engine="$1" src="$2"
  [[ "$engine" == "lean_single" ]] || return 0
  if grep -qE '(^|[^[:alnum:]_])(variance_of|uncertainty_of)[[:space:]]*\(' "$src"; then
    echo "refused: source reads the compiler-native variance channel (variance_of/uncertainty_of), which the lean_single shim strips at .value; pin it to madaros"
    return 1
  fi
  return 0
}

# Run `src` on the entry's engine with a total budget of `budget` seconds.
# Program output (stdout+stderr of the executable) goes to `log`; for Madaros
# the compiler's own chatter goes to `log`.compile, so PASS markers and the
# smoke byte count are measured on what the program printed, never on the
# compiler banner. Returns the run rc (124 = timeout).
dps_exec() {
  local engine="$1" src="$2" log="$3" budget="$4"
  if [[ -n "$SINGLE_ENGINE_BIN" ]]; then
    timeout "$budget" "$SINGLE_ENGINE_BIN" run "$src" >"$log" 2>&1
    return $?
  fi
  case "$engine" in
    lean_single)
      timeout "$budget" "$LEAN_SINGLE_BIN" run "$src" >"$log" 2>&1
      return $?
      ;;
    madaros)
      local elf="$log.elf" clog="$log.compile" t0=$SECONDS rc left
      rm -f "$elf"
      timeout "$budget" "$MADAROS_BIN" compile "$src" -o "$elf" >"$clog" 2>&1
      rc=$?
      if [[ $rc -ne 0 || ! -x "$elf" ]]; then
        { echo "[madaros compile failed rc=$rc; compiler output: $clog]"; tail -8 "$clog"; } >"$log"
        [[ $rc -eq 0 ]] && rc=1
        return $rc
      fi
      left=$(( budget - (SECONDS - t0) )); (( left < 1 )) && left=1
      timeout "$left" "$elf" >"$log" 2>&1
      rc=$?
      rm -f "$elf"
      return $rc
      ;;
    *)
      echo "unknown engine '$engine'" >"$log"
      return 2
      ;;
  esac
}

# Compile `src` once and print the compiler's own output (diagnostics
# included). With the default shim, call the ELF it wraps directly, because
# its compile/run verbs send that output to /dev/null.
dps_compile_diagnostics() {
  local engine="$1" src="$2" tmp drv
  tmp="$(mktemp)"
  if [[ -n "$SINGLE_ENGINE_BIN" ]]; then drv="$SINGLE_ENGINE_BIN"
  elif [[ "$engine" == "madaros" ]]; then drv="$MADAROS_BIN"
  else drv="$LEAN_SINGLE_BIN"; fi
  if [[ "$drv" == "$SHIM" ]]; then
    "${SOUNIO_SEQ_LEANSINGLE_ELF:-$ROOT_DIR/bin/souc-linux-x86_64}" "$src" "$tmp" 2>&1 || true
  else
    "$drv" compile "$src" -o "$tmp" 2>&1 || true
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

PASS_MARKERS='^PASS$|^ALL PASS$|ALL [0-9]+ TESTS PASSED|^ *ALL (TESTS|GUM TESTS) PASSED$|^ *(DEMO|SS DEMO|SS FULLVD|SCENARIO GATE|PK/PD DEMO) OK$|^ *DDI MODULE OK$|^ *DDI CLINICAL VALIDATION COMPLETE$|^ *CROSS-DRUG ISO BUDGET COMPLETE$|^HALO PGX GATE PASS$|^HESSIAN_PBPK28_DUAL_RHO_PASS$|^SOBOL_PCE_SEMAGLUTIDE_FULL_PASS$|^MC_CROSS_VALIDATION_PBPK28_LOGNORMAL_PASS$|^MC_CROSS_VALIDATION_PBPK28_LOGNORMAL_HESSIAN_PASS$|^MC_CROSS_VALIDATION_PBPK28_LOGNORMAL_OUTPUT$|^MC_PRIOR_FAMILY_SWEEP_PASS$|^MC_PRIOR_FAMILY_SWEEP_OUTPUT$'

fails=0
pending=0
results=()

# Shared prologue for TESTS / smoke / pending entries: prints the header,
# checks the source and the variance pin, and runs it. Sets `rc`; returns 1
# (after recording the failure) when the entry must not be evaluated further.
dps_run_entry() {
  local name="$1" engine="$2" src="$3" kind="$4" log="$5" why
  echo ""
  echo "[$name]${kind:+ ($kind)}"
  echo "  src=$src"
  echo "  engine=$(dps_effective_engine "$engine")"
  if [[ ! -f "$src" ]]; then
    echo "  FAIL: source missing"
    fails=$((fails + 1))
    results+=("FAIL  $name  source_missing")
    return 1
  fi
  if ! why=$(dps_variance_pin_refusal "$engine" "$src"); then
    echo "  FAIL: $why"
    fails=$((fails + 1))
    results+=("FAIL  $name  [$engine] variance_channel_pin_refused")
    return 1
  fi
  set +e
  dps_exec "$engine" "$src" "$log" "$TIMEOUT_SECONDS"
  rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    echo "  FAIL: rc=$rc (timeout=$TIMEOUT_SECONDS)"
    tail -5 "$log" | sed 's/^/    /'
    fails=$((fails + 1))
    results+=("FAIL  $name  [$engine] rc=$rc")
    return 1
  fi
  return 0
}

for entry in "${TESTS[@]}"; do
  read -r name engine src <<< "$entry"
  log="$STAGE_DIR/$name.log"
  dps_run_entry "$name" "$engine" "$src" "" "$log" || continue

  if ! grep -qE "$PASS_MARKERS" "$log"; then
    echo "  FAIL: no PASS marker in stdout"
    tail -5 "$log" | sed 's/^/    /'
    fails=$((fails + 1))
    results+=("FAIL  $name  [$engine] no_pass_marker")
    continue
  fi

  echo "  PASS (log=$log)"
  results+=("PASS  $name  [$engine]")
done

# Smoke tests: dissertation visualisation/narrative demos. rc=0 + non-trivial
# program output (>= 100 bytes, compiler output excluded). No PASS marker.
for entry in "${TESTS_SMOKE[@]}"; do
  read -r name engine src <<< "$entry"
  log="$STAGE_DIR/$name.log"
  dps_run_entry "$name" "$engine" "$src" "smoke" "$log" || continue

  out_bytes=$(wc -c < "$log")
  if [[ $out_bytes -lt 100 ]]; then
    echo "  FAIL: output too short ($out_bytes bytes < 100)"
    fails=$((fails + 1))
    results+=("FAIL  $name  [$engine] short_output")
    continue
  fi

  echo "  PASS (rc=0, ${out_bytes}B emitted, log=$log)"
  results+=("PASS  $name  [$engine] (smoke)")
done

# Clinical-validation modules: PENDING-aware enforcement (see TESTS_PENDING).
for entry in "${TESTS_PENDING[@]}"; do
  read -r name engine src <<< "$entry"
  log="$STAGE_DIR/$name.log"
  dps_run_entry "$name" "$engine" "$src" "clinical validation, pending-aware" "$log" || continue

  if grep -qE "_CLINICAL_PASS$" "$log"; then
    echo "  PASS (predicted-vs-observed validation passed, log=$log)"
    results+=("PASS  $name  [$engine] (clinical validation)")
  elif grep -qE "_CLINICAL_FAIL_HONEST$" "$log"; then
    echo "  FAIL: clinical validation failed honestly — predicted-vs-observed"
    echo "        GMFE outside FDA/EMA acceptance (model != clinical data)."
    tail -8 "$log" | sed 's/^/    /'
    fails=$((fails + 1))
    results+=("FAIL  $name  [$engine] clinical_fail_honest")
  elif grep -qE "_CLINICAL_PENDING_OBSERVED$" "$log"; then
    echo "  PENDING: registered, awaiting observed data (obs_n()==0) — not yet validating"
    pending=$((pending + 1))
    results+=("PEND  $name  [$engine] awaiting_observed_data")
  else
    echo "  FAIL: no recognized clinical-validation marker (PASS / FAIL_HONEST / PENDING_OBSERVED)"
    tail -5 "$log" | sed 's/^/    /'
    fails=$((fails + 1))
    results+=("FAIL  $name  [$engine] no_marker")
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
  IFS='|' read -r name engine src want_rc want_n entry_timeout want_cerr cerr_pattern diag <<< "$entry"
  log="$STAGE_DIR/$name.log"

  echo ""
  echo "[$name] (expected FAIL_HONEST, timeout=${entry_timeout}s)"
  echo "  src=$src"
  echo "  engine=$(dps_effective_engine "$engine")"

  if [[ ! -f "$src" ]]; then
    echo "  FAIL: source missing"
    fails=$((fails + 1))
    results+=("FAIL  $name  source_missing")
    continue
  fi
  if ! why=$(dps_variance_pin_refusal "$engine" "$src"); then
    echo "  FAIL: $why"
    fails=$((fails + 1))
    results+=("FAIL  $name  [$engine] variance_channel_pin_refused")
    continue
  fi

  diag_log="$STAGE_DIR/$name.compile.log"
  dps_compile_diagnostics "$engine" "$src" >"$diag_log"
  cverdict=$(dps_compile_verdict "$diag_log" "$want_cerr" "$cerr_pattern")
  if [[ "$cverdict" != "OK" ]]; then
    echo "  FAIL: compiler output differs from the recorded diagnostics (${cverdict#FAIL:})"
    # Reporting only: under `set -euo pipefail` this pipeline fails when no
    # unexpected line exists (e.g. only the count changed), so keep it non-fatal.
    { grep -E '(^|[^[:alnum:]_])error(\[|:)' "$diag_log" | grep -vF -- "$cerr_pattern" | head -5 | sed 's/^/    /'; } || true
    fails=$((fails + 1))
    results+=("FAIL  $name  [$engine] ${cverdict#FAIL:}")
    continue
  fi

  set +e
  dps_exec "$engine" "$src" "$log" "$entry_timeout"
  rc=$?
  set -e

  verdict=$(dps_xfail_verdict "$log" "$rc" "$want_rc" "$want_n" "$diag")
  if [[ "$verdict" == "XFAIL" ]]; then
    echo "  XFAIL: fails exactly as recorded (compiler errors: the $want_cerr recorded; rc=$rc, diagnostic x$want_n, no other FAIL)"
    xfails=$((xfails + 1))
    results+=("XFAIL $name  [$engine] documented_defect")
  else
    echo "  FAIL: expected-FAIL_HONEST entry did not fail as recorded (${verdict#FAIL:}; rc=$rc, timeout=${entry_timeout}s)"
    tail -5 "$log" | sed 's/^/    /'
    fails=$((fails + 1))
    results+=("FAIL  $name  [$engine] ${verdict#FAIL:}")
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
