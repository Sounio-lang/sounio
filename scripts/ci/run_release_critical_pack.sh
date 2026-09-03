#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
DEFAULT_RUN_DIR="$ROOT_DIR/artifacts/diagnostic/$TS"
RUN_DIR="${SOUNIO_RELEASE_PACK_RUN_DIR:-$DEFAULT_RUN_DIR}"
SHARED_CARGO_HOME_DEFAULT="$ROOT_DIR/artifacts/diagnostic/.cargo-home"

export SOUNIO_DIAG_SHARED_CARGO_HOME="${SOUNIO_DIAG_SHARED_CARGO_HOME:-$SHARED_CARGO_HOME_DEFAULT}"
mkdir -p "$SOUNIO_DIAG_SHARED_CARGO_HOME"

mkdir -p "$RUN_DIR"
echo "[release-pack] artifacts=$RUN_DIR"
echo "[release-pack] shared_cargo_home=$SOUNIO_DIAG_SHARED_CARGO_HOME"

SUMMARY_TSV="$RUN_DIR/release-pack-summary.tsv"
SUMMARY_MD="$RUN_DIR/release-pack-summary.md"

declare -a STEP_LABELS=()
declare -a STEP_STATUSES=()
declare -a STEP_DURATIONS=()
declare -a STEP_LOGS=()

write_summary() {
  {
    echo -e "step\tstatus\tduration_s\tlog"
    local i
    for ((i = 0; i < ${#STEP_LABELS[@]}; i++)); do
      echo -e "${STEP_LABELS[$i]}\t${STEP_STATUSES[$i]}\t${STEP_DURATIONS[$i]}\t${STEP_LOGS[$i]}"
    done
  } > "$SUMMARY_TSV"

  {
    echo "# Release-Critical Pack Summary"
    echo
    echo "Run directory: \`$RUN_DIR\`"
    echo
    echo "| Step | Status | Duration (s) | Log |"
    echo "|---|---|---:|---|"
    local i
    for ((i = 0; i < ${#STEP_LABELS[@]}; i++)); do
      echo "| ${STEP_LABELS[$i]} | ${STEP_STATUSES[$i]} | ${STEP_DURATIONS[$i]} | \`${STEP_LOGS[$i]}\` |"
    done
  } > "$SUMMARY_MD"
}

run_step() {
  local label="$1"
  shift
  local started ended elapsed status
  local step_log="$RUN_DIR/logs/${label}.log"

  started="$(date +%s)"
  echo "[release-pack] >>> $label"
  if SOUNIO_DIAG_RUN_DIR="$RUN_DIR" \
    SOUNIO_DIAG_LOG_LABEL="$label" \
    bash "$ROOT_DIR/scripts/dev/with_isolated_env.sh" "$@"; then
    status="PASS"
  else
    status="FAIL"
  fi

  ended="$(date +%s)"
  elapsed="$((ended - started))"

  STEP_LABELS+=("$label")
  STEP_STATUSES+=("$status")
  STEP_DURATIONS+=("$elapsed")
  STEP_LOGS+=("$step_log")
  write_summary

  if [[ "$status" != "PASS" ]]; then
    echo "[release-pack] !!! $label failed (${elapsed}s)"
    echo "[release-pack] log=$step_log"
    if [[ -f "$step_log" ]]; then
      echo "[release-pack] --- tail of $step_log ---"
      tail -n 60 "$step_log" || true
      echo "[release-pack] --- end tail ---"
    fi
    echo "[release-pack] summary=$SUMMARY_MD"
    return 1
  fi

  echo "[release-pack] <<< $label PASS (${elapsed}s)"
}

run_step "01-cargo-check" cargo check -p souc
run_step "02-strict-compile-source-fail-closed" cargo test -p souc compiler_loader::tests::test_driver_source_pipeline_strict_rejects_fallback_when_driver_unavailable -- --nocapture
run_step "03-strict-compile-file-fail-closed" cargo test -p souc compiler_loader::tests::test_driver_file_pipeline_strict_rejects_fallback_when_driver_unavailable -- --nocapture
run_step "04-driver-multimodule-stub-fail-closed" cargo test -p souc --lib compiler_loader::tests::test_driver_multimodule_pipeline_fails_closed_with_explicit_stub_diag -- --nocapture
run_step "05-strict-check-fail-closed" cargo test -p souc --test selfhost_strict_mode -- selfhost_strict_check_only_rejects_stage_boundary_fallback_when_driver_unavailable --nocapture
run_step "06-strict-run-fail-closed" cargo test -p souc --test selfhost_strict_mode -- selfhost_strict_rejects_stage_boundary_fallback_when_driver_unavailable --nocapture
run_step "07-seed-policy-root-fail-closed" cargo test -p souc --test selfhost_strict_mode -- selfhost_root_seed_enforce --nocapture
run_step "08-seed-policy-transition-fail-closed" cargo test -p souc --test selfhost_strict_mode -- selfhost_root_transition_mode --nocapture
run_step "09-no-legacy-env-diagnostic-smoke" bash -lc 'set -euo pipefail; cargo build -p souc --bin souc >/dev/null 2>&1; log=$(mktemp); if ./target/debug/souc run self-hosted/ --check-only >"$log" 2>&1; then :; fi; if rg -q "LEGACY_SELFHOST_ENV_REMOVED" "$log"; then echo "unexpected legacy env diagnostic in clean selfhost run"; cat "$log"; exit 1; fi'
run_step "10-cargo-lib-tests" cargo test -p souc --lib
run_step "11-cultural-fidelity" python3 "$ROOT_DIR/scripts/ci/cultural_fidelity_gate.py"
run_step "12-r2-parity-spec-lint" python3 "$ROOT_DIR/scripts/r2/parity_spec_lint.py"
run_step "13-r2-parity-spec-exec" python3 "$ROOT_DIR/scripts/r2/parity_spec_exec.py"
run_step "14-warning-baseline" bash "$ROOT_DIR/scripts/ci/check_new_warnings.sh"
run_step "14a-claude-operational-contract-gate" bash "$ROOT_DIR/scripts/ci/claude_operational_contract_gate.sh"
run_step "15-build-bootstrap-seed" bash "$ROOT_DIR/scripts/bootstrap/build_bootstrap_seed.sh"
run_step "16-selfhost-cycle-release-byte-equality" env WORK_DIR="$RUN_DIR/selfhost-cycle-release-gate" SOUNIO_SELFHOST_RELEASE_CYCLE_SKIP_BUILD=1 SOUNIO_SELFHOST_BOOTSTRAP_MANIFEST=bootstrap/selfhost-kernel.manifest bash "$ROOT_DIR/scripts/selfhost/selfhost_cycle_release_gate.sh"
run_step "17-selfhost-cycle-gate-seed-root" env WORK_DIR="$RUN_DIR/selfhost-cycle-gate-seed-root" SOUNIO_SELFHOST_CYCLE_FORCE_DYNAMIC=0 SOUNIO_SELFHOST_CYCLE_SEED_ENFORCE=1 SOUNIO_SELFHOST_CYCLE_SEED_PATH=bootstrap/seeds/sounio-bootstrap-linux-x86_64.sio.bin SOUNIO_SELFHOST_BOOTSTRAP_MANIFEST=bootstrap/selfhost-kernel.manifest bash "$ROOT_DIR/scripts/selfhost/selfhost_cycle_gate.sh"
run_step "18-selfhost-independence-gate" env WORK_DIR="$RUN_DIR/selfhost-independence-gate" SOUC_BIN="$ROOT_DIR/target/release/souc" bash "$ROOT_DIR/scripts/selfhost/selfhost_independence_gate.sh"
run_step "18a-stdlib-native-lane-gate" env STDLIB_HYPER_GATE_MODE=required STDLIB_HYPER_STATUS_OUT="$RUN_DIR/stdlib_hyper_execution_status.v1.json" STDLIB_NATIVE_LANE_MATRIX_OUT="$RUN_DIR/native_lane_matrix.v1.json" bash "$ROOT_DIR/scripts/stdlib/stdlib_hyper_execution_gate.sh"
run_step "19-full-gate" bash "$ROOT_DIR/scripts/dev/full_gate.sh"
run_step "20-sprint14-checker-parity" bash "$ROOT_DIR/scripts/sprint14_checker_parity_gate.sh"
run_step "21-sprint16-effect-handlers" bash "$ROOT_DIR/scripts/sprint16_effect_handlers_gate.sh"

# --- Sprint gate regression pack (Sprints 5-24) ---
run_step "22-sprint12-borrow-integration" bash "$ROOT_DIR/scripts/sprint12_borrow_integration_gate.sh"
run_step "23-sprint13-diagnostic-hardening" bash "$ROOT_DIR/scripts/sprint13_diagnostic_hardening_gate.sh"
run_step "24-sprint14-contest-witness" bash "$ROOT_DIR/scripts/sprint14_contest_witness_gate.sh"
run_step "25-sprint15-epistemic-witness-manifest" bash "$ROOT_DIR/scripts/sprint15_epistemic_witness_manifest_gate.sh"
run_step "26-sprint16-decision-admissibility" bash "$ROOT_DIR/scripts/sprint16_decision_admissibility_gate.sh"
run_step "27-sprint5-contest-validated" bash "$ROOT_DIR/scripts/sprint5_contest_validated_gate.sh"
run_step "28-sprint7-prove-robust" bash "$ROOT_DIR/scripts/sprint7_prove_robust_gate.sh"
run_step "29-sprint8-validate-manifest" bash "$ROOT_DIR/scripts/sprint8_validate_manifest_gate.sh"
run_step "30-sprint9-hlir-epistemic" bash "$ROOT_DIR/scripts/sprint9_hlir_epistemic_gate.sh"
run_step "31-sprint10-lift-knowledge" bash "$ROOT_DIR/scripts/sprint10_lift_knowledge_gate.sh"
run_step "32-sprint11-measure" bash "$ROOT_DIR/scripts/sprint11_measure_gate.sh"
run_step "33-sprint15-effects-parity" bash "$ROOT_DIR/scripts/sprint15_effects_parity_gate.sh"
run_step "34-sprint17a-row-poly" bash "$ROOT_DIR/scripts/sprint17a_row_poly_gate.sh"
run_step "35-sprint17b-aleatoric-split" bash "$ROOT_DIR/scripts/sprint17b_aleatoric_split_gate.sh"
run_step "36-sprint17d-causal-type" bash "$ROOT_DIR/scripts/sprint17d_causal_type_gate.sh"
run_step "37-sprint17-deferral" bash "$ROOT_DIR/scripts/sprint17_deferral_gate.sh"
run_step "38-sprint18-graded-epistemic" bash "$ROOT_DIR/scripts/sprint18_graded_epistemic_gate.sh"
run_step "39-sprint18-deferral-manifest" bash "$ROOT_DIR/scripts/sprint18_deferral_manifest_gate.sh"
run_step "40-sprint19-epistemic-session" bash "$ROOT_DIR/scripts/sprint19_epistemic_session_gate.sh"
run_step "41-sprint19-resolution" bash "$ROOT_DIR/scripts/sprint19_resolution_gate.sh"
run_step "42-sprint19-session-types" bash "$ROOT_DIR/scripts/sprint19_session_types_gate.sh"
run_step "43-sprint19-user-effects" bash "$ROOT_DIR/scripts/sprint19_user_effects_gate.sh"
run_step "44-sprint19-web" bash "$ROOT_DIR/scripts/sprint19_web_gate.sh"
run_step "45-sprint20-lang-sota" bash "$ROOT_DIR/scripts/sprint20_lang_sota_gate.sh"
run_step "46-sprint20-resolution-manifest" bash "$ROOT_DIR/scripts/sprint20_resolution_manifest_gate.sh"
run_step "47-sprint21-dp" bash "$ROOT_DIR/scripts/sprint21_dp_gate.sh"
run_step "48-sprint21-alternative-frontier" bash "$ROOT_DIR/scripts/sprint21_alternative_frontier_gate.sh"
run_step "49-sprint22-responsible-ai" bash "$ROOT_DIR/scripts/sprint22_responsible_ai_gate.sh"
run_step "50-sprint22-alternative-manifest" bash "$ROOT_DIR/scripts/sprint22_alternative_manifest_gate.sh"
run_step "51-sprint23-statistical-types" bash "$ROOT_DIR/scripts/sprint23_statistical_types_gate.sh"
run_step "52-sprint24-scientific-foundations" bash "$ROOT_DIR/scripts/sprint24_scientific_foundations_gate.sh"
run_step "53-sprint23-transition-protocol" bash "$ROOT_DIR/scripts/sprint23_transition_protocol_gate.sh"
run_step "54-sprint24-transition-manifest" bash "$ROOT_DIR/scripts/sprint24_transition_manifest_gate.sh"

write_summary
echo "[release-pack] summary=$SUMMARY_MD"
echo "[release-pack] summary_tsv=$SUMMARY_TSV"
echo "[release-pack] PASS"
