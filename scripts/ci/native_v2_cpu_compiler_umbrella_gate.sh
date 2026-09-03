#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ "$(uname -s 2>/dev/null || echo unknown)" != "Linux" ]]; then
  echo "[native-v2-cpu-compiler] SKIP: Linux-only gate" >&2
  exit 0
fi

case "$(uname -m 2>/dev/null || echo unknown)" in
  x86_64|amd64) ;;
  *)
    echo "[native-v2-cpu-compiler] SKIP: x86-64 Linux-only gate" >&2
    exit 0
    ;;
esac

source "$ROOT_DIR/scripts/lib/resolve_souc.sh"
sounio_require_souc

OUT_DIR="${SOUNIO_NATIVE_V2_CPU_COMPILER_DIR:-$(mktemp -d /tmp/sounio-native-v2-cpu-compiler.XXXXXX)}"
LOG_DIR="$OUT_DIR/logs"
ARTIFACT_DIR="$OUT_DIR/artifacts"
RESULTS_TSV="$ARTIFACT_DIR/gates.tsv"
SUMMARY_JSON="${SOUNIO_NATIVE_V2_CPU_COMPILER_SUMMARY:-$ARTIFACT_DIR/native_v2_cpu_compiler_umbrella.v1.json}"

mkdir -p "$LOG_DIR" "$ARTIFACT_DIR"

cat >"$RESULTS_TSV" <<'EOF'
gate	rc	out_dir	summary_json	log_path
EOF

append_gate_row() {
  local gate="$1"
  local rc="$2"
  local out_dir="$3"
  local summary_json="$4"
  local log_path="$5"

  printf '%s\t%s\t%s\t%s\t%s\n' \
    "$gate" "$rc" "$out_dir" "$summary_json" "$log_path" >>"$RESULTS_TSV"
}

run_driver_self_compile() {
  local gate="driver_self_compile"
  local gate_dir="$OUT_DIR/$gate"
  local log_path="$LOG_DIR/$gate.log"

  mkdir -p "$gate_dir"
  echo "[native-v2-cpu-compiler] running $gate"
  set +e
  SOUNIO_NATIVE_V2_DRIVER_SELF_COMPILE_DIR="$gate_dir" \
    bash scripts/ci/native_v2_driver_self_compile_gate.sh >"$log_path" 2>&1
  local rc=$?
  set -e
  append_gate_row "$gate" "$rc" "$gate_dir" "-" "$log_path"
}

run_science_spine() {
  local gate="science_spine"
  local gate_dir="$OUT_DIR/$gate"
  local log_path="$LOG_DIR/$gate.log"
  local summary_json="$gate_dir/artifacts/summary.json"

  mkdir -p "$gate_dir"
  echo "[native-v2-cpu-compiler] running $gate"
  set +e
  SOUNIO_NATIVE_V2_SCIENCE_SPINE_DIR="$gate_dir" \
  SOUNIO_NATIVE_V2_DRIVER_SELF_COMPILE_DIR="$gate_dir/driver-self" \
    bash scripts/ci/native_v2_epistemic_science_spine_gate.sh >"$log_path" 2>&1
  local rc=$?
  set -e
  append_gate_row "$gate" "$rc" "$gate_dir" "$summary_json" "$log_path"
}

run_f64_ladder() {
  local gate="f64_ladder"
  local gate_dir="$OUT_DIR/$gate"
  local log_path="$LOG_DIR/$gate.log"
  local summary_json="$gate_dir/artifacts/summary.json"

  mkdir -p "$gate_dir"
  echo "[native-v2-cpu-compiler] running $gate"
  set +e
  SOUNIO_NATIVE_V2_F64_LADDER_DIR="$gate_dir" \
  SOUNIO_NATIVE_V2_DRIVER_SELF_COMPILE_DIR="$gate_dir/driver-self" \
    bash scripts/ci/native_v2_f64_ladder_gate.sh >"$log_path" 2>&1
  local rc=$?
  set -e
  append_gate_row "$gate" "$rc" "$gate_dir" "$summary_json" "$log_path"
}

run_gum_primitives() {
  local gate="gum_primitives"
  local gate_dir="$OUT_DIR/$gate"
  local log_path="$LOG_DIR/$gate.log"
  local summary_json="$gate_dir/artifacts/summary.json"

  mkdir -p "$gate_dir"
  echo "[native-v2-cpu-compiler] running $gate"
  set +e
  SOUNIO_NATIVE_V2_GUM_DIR="$gate_dir" \
  SOUNIO_NATIVE_V2_DRIVER_SELF_COMPILE_DIR="$gate_dir/driver-self" \
    bash scripts/ci/native_v2_gum_primitives_gate.sh >"$log_path" 2>&1
  local rc=$?
  set -e
  append_gate_row "$gate" "$rc" "$gate_dir" "$summary_json" "$log_path"
}

run_semantic_hardening() {
  local gate="semantic_hardening"
  local gate_dir="$OUT_DIR/$gate"
  local log_path="$LOG_DIR/$gate.log"
  local summary_json="$gate_dir/artifacts/summary.json"

  mkdir -p "$gate_dir"
  echo "[native-v2-cpu-compiler] running $gate"
  set +e
  SOUNIO_NATIVE_V2_SEMANTIC_HARDENING_DIR="$gate_dir" \
  SOUNIO_NATIVE_V2_DRIVER_SELF_COMPILE_DIR="$gate_dir/driver-self" \
    bash scripts/ci/native_v2_semantic_hardening_gate.sh >"$log_path" 2>&1
  local rc=$?
  set -e
  append_gate_row "$gate" "$rc" "$gate_dir" "$summary_json" "$log_path"
}

run_lean_single_fixed_point() {
  local gate="lean_single_fixed_point"
  local gate_dir="$OUT_DIR/$gate"
  local log_path="$LOG_DIR/$gate.log"

  mkdir -p "$gate_dir"
  echo "[native-v2-cpu-compiler] running $gate"
  set +e
  SOUNIO_LEAN_SINGLE_FP_DIR="$gate_dir" \
    bash scripts/ci/lean_single_fixed_point_gate.sh >"$log_path" 2>&1
  local rc=$?
  set -e
  append_gate_row "$gate" "$rc" "$gate_dir" "-" "$log_path"
}

# Imported noncapturing closure boundary: pins imported closure literals to the
# modular IR native driver path -- fails loudly if compile falls back to
# native_prebundle or full IR. ~5s.
run_imported_closure_boundary() {
  local gate="imported_closure_boundary"
  local gate_dir="$OUT_DIR/$gate"
  local log_path="$LOG_DIR/$gate.log"

  mkdir -p "$gate_dir"
  echo "[native-v2-cpu-compiler] running $gate"
  set +e
  SOUNIO_NATIVE_V2_IMPORTED_CLOSURE_BOUNDARY_DIR="$gate_dir" \
    bash scripts/ci/native_v2_imported_closure_boundary_gate.sh >"$log_path" 2>&1
  local rc=$?
  set -e
  append_gate_row "$gate" "$rc" "$gate_dir" "-" "$log_path"
}

# Imported captured closure boundary: same path-discrimination invariants as
# above, but for closure factories that capture an i64 across the module
# boundary. The gate internally re-runs the noncapturing variant as a
# prerequisite -- intentionally kept as a separate umbrella row so the failure
# surface stays diagnosable. ~10s.
run_imported_captured_closure_boundary() {
  local gate="imported_captured_closure_boundary"
  local gate_dir="$OUT_DIR/$gate"
  local log_path="$LOG_DIR/$gate.log"

  mkdir -p "$gate_dir"
  echo "[native-v2-cpu-compiler] running $gate"
  set +e
  SOUNIO_NATIVE_V2_IMPORTED_CAPTURED_CLOSURE_BOUNDARY_DIR="$gate_dir" \
    bash scripts/ci/native_v2_imported_captured_closure_boundary_gate.sh >"$log_path" 2>&1
  local rc=$?
  set -e
  append_gate_row "$gate" "$rc" "$gate_dir" "-" "$log_path"
}

# ISO budget: dissertation contribution #3 -- ISO 17025 GUM uncertainty budget
# for rapamycin 2-comp PBPK. CPU-only sampler, ~1s at 1M patients.
run_iso_budget() {
  local gate="iso_budget"
  local gate_dir="$OUT_DIR/$gate"
  local log_path="$LOG_DIR/$gate.log"

  mkdir -p "$gate_dir"
  echo "[native-v2-cpu-compiler] running $gate"
  set +e
  ISO_STAGE_DIR="$gate_dir" \
    bash scripts/ci/kretikos_kaxi_iso_budget_gate.sh >"$log_path" 2>&1
  local rc=$?
  set -e
  append_gate_row "$gate" "$rc" "$gate_dir" "-" "$log_path"
}

# Phase J: compile-time confidence enforcement (dissertation contribution #2).
# No GPU required -- pure K-AXI→PTX policy check. ~1s.
run_phase_j() {
  local gate="phase_j_conf_gate"
  local gate_dir="$OUT_DIR/$gate"
  local log_path="$LOG_DIR/$gate.log"

  mkdir -p "$gate_dir"
  echo "[native-v2-cpu-compiler] running $gate"
  set +e
  bash scripts/ci/kretikos_kaxi_phase_j_gate.sh >"$log_path" 2>&1
  local rc=$?
  set -e
  append_gate_row "$gate" "$rc" "$gate_dir" "-" "$log_path"
}

# K-AXI meta-gate: structural emitter + witness/certificate closure for all
# supported profiles. CPU-only except for the final PTX golden drift oracle.
# Skippable for fast umbrella runs via SOUNIO_KRETIKOS_KAXI_GATE_SKIP=1.
run_kretikos_kaxi_meta() {
  local gate="kretikos_kaxi_meta"
  local gate_dir="$OUT_DIR/$gate"
  local log_path="$LOG_DIR/$gate.log"

  mkdir -p "$gate_dir"
  echo "[native-v2-cpu-compiler] running $gate"

  if [[ "${SOUNIO_KRETIKOS_KAXI_GATE_SKIP:-0}" == "1" ]]; then
    printf 'skipped_by_env\n' >"$log_path"
    append_gate_row "$gate" "0" "$gate_dir" "-" "$log_path"
    return 0
  fi

  set +e
  SOUNIO_KRETIKOS_KAXI_GATE_DIR="$gate_dir" \
    bash scripts/ci/kretikos_kaxi_gate.sh >"$log_path" 2>&1
  local rc=$?
  set -e
  append_gate_row "$gate" "$rc" "$gate_dir" "-" "$log_path"
}

# Native-v2 struct orchestrator: examples/native struct regressions plus imported
# ABI and metal lanes. Skippable via SOUNIO_NATIVE_V2_STRUCT_GATE_SKIP=1.
run_struct_orchestrator() {
  local gate="struct_orchestrator"
  local gate_dir="$OUT_DIR/$gate"
  local log_path="$LOG_DIR/$gate.log"

  mkdir -p "$gate_dir"
  echo "[native-v2-cpu-compiler] running $gate"

  if [[ "${SOUNIO_NATIVE_V2_STRUCT_GATE_SKIP:-0}" == "1" ]]; then
    printf 'skipped_by_env\n' >"$log_path"
    append_gate_row "$gate" "0" "$gate_dir" "-" "$log_path"
    return 0
  fi

  set +e
  SOUNIO_NATIVE_V2_STRUCT_GATE_DIR="$gate_dir" \
    bash scripts/ci/native_v2_struct_gate.sh >"$log_path" 2>&1
  local rc=$?
  set -e
  append_gate_row "$gate" "$rc" "$gate_dir" "-" "$log_path"
}

# Dissertation PBPK suite: applied evidence layer for the rapamycin model.
# 5 run-pass tests cover Euler+RK4 ISO budgets, BBB/Pgp clinical claims,
# variance-aware adaptive integration, and GUM-vs-MC validation. CPU-only,
# ~30s. This is the regression shield for the dissertation's *evidence*
# (contributions #1/#2/#3 are about the mechanism; this gate is about the
# results those mechanisms produce on rapamycin).
run_dissertation_pbpk_suite() {
  local gate="dissertation_pbpk_suite"
  local gate_dir="$OUT_DIR/$gate"
  local log_path="$LOG_DIR/$gate.log"

  mkdir -p "$gate_dir"
  echo "[native-v2-cpu-compiler] running $gate"
  set +e
  DPS_STAGE_DIR="$gate_dir" \
    bash scripts/ci/dissertation_pbpk_suite_gate.sh >"$log_path" 2>&1
  local rc=$?
  set -e
  append_gate_row "$gate" "$rc" "$gate_dir" "-" "$log_path"
}

# Phase Y: real GUM variance propagation through 2-comp PBPK ODE on local GPU.
# Self-skips when libcuda is absent -- safe to call on CPU-only boxes.
# Kept at 10M patients (~30s) which is the gate's regression-grade default.
# Phase X is intentionally NOT wired here: it requires a kubectl-based 2-GPU
# sharded run at 2B patients, which is too heavy for a per-commit umbrella.
run_phase_y() {
  local gate="phase_y_gum_pbpk"
  local gate_dir="$OUT_DIR/$gate"
  local log_path="$LOG_DIR/$gate.log"

  mkdir -p "$gate_dir"
  echo "[native-v2-cpu-compiler] running $gate"
  set +e
  PHY_STAGE_DIR="$gate_dir" \
    bash scripts/ci/kretikos_kaxi_phase_y_gate.sh >"$log_path" 2>&1
  local rc=$?
  set -e
  append_gate_row "$gate" "$rc" "$gate_dir" "-" "$log_path"
}

# End-to-end codegen-correctness suite: builds the modular compiler from
# self-hosted/compiler/main.sio, then emits + RUNS 9 witness executables
# (scalar value fidelity, call/multicall, branch taken/not-taken, OpSub+OpDiv),
# asserting each process exit code. Proves the IR -> machine code -> ELF -> exit
# back-half is correct across distinct codegen shapes. Does NOT exercise the
# parse->check->lower front-half (still G1-blocked). ~1-2min (one compiler build).
run_e2e_codegen_suite() {
  local gate="e2e_codegen_suite"
  local gate_dir="$OUT_DIR/$gate"
  local log_path="$LOG_DIR/$gate.log"

  mkdir -p "$gate_dir"
  echo "[native-v2-cpu-compiler] running $gate"
  set +e
  SOUNIO_BOOTSTRAP_SOUC="$SOUC_BIN" \
    bash scripts/ci/native_v2_e2e_codegen_suite_gate.sh >"$log_path" 2>&1
  local rc=$?
  set -e
  append_gate_row "$gate" "$rc" "$gate_dir" "-" "$log_path"
}

emit_summary_json() {
  local ts
  ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  mkdir -p "$(dirname "$SUMMARY_JSON")"
  if "$ROOT_DIR/bin/kretikos" umbrella-aggregate \
       "$RESULTS_TSV" "$SOUC_BIN" "$OUT_DIR" "$ts" \
       > "$SUMMARY_JSON" 2>/dev/null \
     && [[ -s "$SUMMARY_JSON" ]]; then
    return 0
  fi

  # Fallback: aggregator hit the native-compiler SRET bug on a TSV that
  # mixes log-only and summary rows. Emit a minimal shell-built JSON so
  # the umbrella's pass/fail signal still works. Sub-gate rcs come from
  # the TSV directly. See memory: "Native compiler limits".
  echo "[native-v2-cpu-compiler] aggregator unavailable, using shell fallback" >&2
  local any_fail=0
  local rows=""
  local first=1
  while IFS=$'\t' read -r gate rc out_dir sjson lpath; do
    [[ "$gate" == "gate" ]] && continue
    [[ -z "$gate" ]] && continue
    if [[ "$rc" != "0" ]]; then any_fail=1; fi
    if [[ $first -eq 1 ]]; then first=0; else rows="$rows,"$'\n'; fi
    rows="$rows    {\"gate\": \"$gate\", \"rc\": $rc, \"out_dir\": \"$out_dir\"}"
  done < "$RESULTS_TSV"
  cat > "$SUMMARY_JSON" <<EOF
{
  "schema": "native_v2_cpu_compiler_umbrella.shell_fallback.v1",
  "generated_at": "$ts",
  "fallback_reason": "aggregator_native_sret_bug",
  "gates": [
$rows
  ]
}
EOF
  return $any_fail
}

echo "[native-v2-cpu-compiler] souc=$SOUC_BIN"
echo "[native-v2-cpu-compiler] out=$OUT_DIR"
echo "[native-v2-cpu-compiler] summary=$SUMMARY_JSON"

run_e2e_codegen_suite
run_driver_self_compile
run_science_spine
run_f64_ladder
run_gum_primitives
run_semantic_hardening
run_lean_single_fixed_point
run_imported_closure_boundary
run_imported_captured_closure_boundary
run_struct_orchestrator
run_iso_budget
run_phase_j
run_kretikos_kaxi_meta
run_phase_y
run_dissertation_pbpk_suite

if emit_summary_json; then
  echo "[native-v2-cpu-compiler] PASS: driver self-compile, science spine, f64 ladder, GUM primitives, semantic hardening, lean_single fixed-point, imported_closure_boundary, imported_captured_closure_boundary, struct_orchestrator, iso_budget, phase_j_conf_gate, kretikos_kaxi_meta, phase_y_gum_pbpk, dissertation_pbpk_suite"
  echo "[native-v2-cpu-compiler] summary=$SUMMARY_JSON"
else
  echo "[native-v2-cpu-compiler] FAIL: see summary=$SUMMARY_JSON" >&2
  exit 1
fi
