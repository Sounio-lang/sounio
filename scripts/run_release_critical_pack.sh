#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="$ROOT_DIR/artifacts/diagnostic/$TS"
SHARED_CARGO_HOME_DEFAULT="$ROOT_DIR/artifacts/diagnostic/.cargo-home"

export SOUNIO_DIAG_SHARED_CARGO_HOME="${SOUNIO_DIAG_SHARED_CARGO_HOME:-$SHARED_CARGO_HOME_DEFAULT}"
mkdir -p "$SOUNIO_DIAG_SHARED_CARGO_HOME"

mkdir -p "$RUN_DIR"
echo "[release-pack] artifacts=$RUN_DIR"
echo "[release-pack] shared_cargo_home=$SOUNIO_DIAG_SHARED_CARGO_HOME"

run_step() {
  local label="$1"
  shift
  echo "[release-pack] >>> $label"
  SOUNIO_DIAG_RUN_DIR="$RUN_DIR" \
  SOUNIO_DIAG_LOG_LABEL="$label" \
  bash "$ROOT_DIR/scripts/with_isolated_env.sh" "$@"
}

run_step "01-cargo-check" cargo check -p souc
run_step "02-strict-compile-source-fail-closed" env SOUNIO_SELFHOST_STRICT=1 SOUNIO_SELFHOST_NO_RUST_FALLBACK=1 SOUNIO_SELFHOST_PIPELINE=driver cargo test -p souc compiler_loader::tests::test_driver_source_pipeline_strict_rejects_fallback_when_driver_unavailable -- --nocapture
run_step "03-strict-compile-file-fail-closed" env SOUNIO_SELFHOST_STRICT=1 SOUNIO_SELFHOST_NO_RUST_FALLBACK=1 SOUNIO_SELFHOST_PIPELINE=driver cargo test -p souc compiler_loader::tests::test_driver_file_pipeline_strict_rejects_fallback_when_driver_unavailable -- --nocapture
run_step "04-driver-multimodule-stub-fail-closed" cargo test -p souc --lib compiler_loader::tests::test_driver_multimodule_pipeline_fails_closed_with_explicit_stub_diag -- --nocapture
run_step "05-strict-check-fail-closed" env SOUNIO_SELFHOST_STRICT=1 SOUNIO_SELFHOST_NO_RUST_FALLBACK=1 SOUNIO_SELFHOST_PIPELINE=driver cargo test -p souc --test selfhost_strict_mode -- selfhost_strict_check_only_rejects_stage_boundary_fallback_when_driver_unavailable --nocapture
run_step "06-strict-run-fail-closed" env SOUNIO_SELFHOST_STRICT=1 SOUNIO_SELFHOST_NO_RUST_FALLBACK=1 SOUNIO_SELFHOST_PIPELINE=driver cargo test -p souc --test selfhost_strict_mode -- selfhost_strict_rejects_stage_boundary_fallback_when_driver_unavailable --nocapture
run_step "07-seed-policy-root-fail-closed" cargo test -p souc --test selfhost_strict_mode -- selfhost_root_seed_enforce --nocapture
run_step "08-seed-policy-transition-fail-closed" cargo test -p souc --test selfhost_strict_mode -- selfhost_root_transition_mode --nocapture
run_step "09-ghost-warning-deterministic" cargo test -p souc --test selfhost_strict_mode -- selfhost_pipeline_rust_with_ghost_emits_transition_warning --nocapture
run_step "10-cargo-lib-tests" cargo test -p souc --lib
run_step "11-cultural-fidelity" python3 "$ROOT_DIR/scripts/cultural_fidelity_gate.py"
run_step "12-r2-parity-spec-lint" python3 "$ROOT_DIR/scripts/r2/parity_spec_lint.py"
run_step "13-r2-parity-spec-exec" python3 "$ROOT_DIR/scripts/r2/parity_spec_exec.py"
run_step "14-warning-baseline" bash "$ROOT_DIR/scripts/check_new_warnings.sh"
run_step "15-full-gate" bash "$ROOT_DIR/scripts/full_gate.sh"

echo "[release-pack] PASS"
