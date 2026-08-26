#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

MODE="${SOUNIO_TRACEABILITY_MODE:-full}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="${2:-}"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: $0 [--mode quick|full]"
      exit 1
      ;;
  esac
done

if [[ "$MODE" != "quick" && "$MODE" != "full" ]]; then
  echo "Invalid mode: $MODE (expected quick or full)"
  exit 1
fi

TS="$(date -u +%Y%m%dT%H%M%SZ)"
DEFAULT_RUN_DIR="$ROOT_DIR/artifacts/diagnostic/$TS"
RUN_DIR="${SOUNIO_TRACEABILITY_RUN_DIR:-$DEFAULT_RUN_DIR}"

export SOUNIO_DIAG_RUN_DIR="$RUN_DIR"
export SOUNIO_DIAG_SHARED_CARGO_HOME="${SOUNIO_DIAG_SHARED_CARGO_HOME:-$ROOT_DIR/artifacts/diagnostic/.shared-cargo-home}"
export SOUNIO_DIAG_CARGO_TARGET_DIR="${SOUNIO_DIAG_CARGO_TARGET_DIR:-$ROOT_DIR/artifacts/diagnostic/.shared-cargo-target}"
export SOUNIO_DIAG_NPM_CACHE="${SOUNIO_DIAG_NPM_CACHE:-$ROOT_DIR/artifacts/diagnostic/.shared-npm-cache}"

mkdir -p "$SOUNIO_DIAG_SHARED_CARGO_HOME" "$SOUNIO_DIAG_CARGO_TARGET_DIR" "$SOUNIO_DIAG_NPM_CACHE" "$RUN_DIR"

run_step() {
  local label="$1"
  shift
  echo "[traceability-diag] >>> $label"
  SOUNIO_DIAG_LOG_LABEL="$label" \
    bash "$ROOT_DIR/scripts/dev/with_isolated_env.sh" "$@"
  echo "[traceability-diag] <<< $label"
}

run_step "cargo-check-rerun" cargo check -p souc
run_step "workflow-refs-rerun" bash "$ROOT_DIR/scripts/dev/check_workflow_script_refs.sh"
run_step "docs-consistency-rerun" bash "$ROOT_DIR/scripts/dev/check_docs_consistency.sh"
run_step "docs-registry-rerun" bash "$ROOT_DIR/scripts/dev/check_docs_registry.sh"
run_step "website-quality-rerun" npm --prefix "$ROOT_DIR/website" run check:quality

if [[ "$MODE" == "full" ]]; then
  run_step "strict-source-rerun" env SOUNIO_SELFHOST_DRIVER_STRICT=1 SOUNIO_SELFHOST_STRICT=1 cargo test -p souc --lib compiler_loader::tests::test_driver_source_pipeline_compiles_simple_source -- --nocapture
  run_step "strict-file-rerun" env SOUNIO_SELFHOST_DRIVER_STRICT=1 SOUNIO_SELFHOST_STRICT=1 cargo test -p souc --lib compiler_loader::tests::test_driver_file_pipeline_compiles_simple_file -- --nocapture
  run_step "lib-test-rerun" cargo test -p souc --lib
  run_step "update-cascading-golden" cargo test -p souc e2e::golden::golden_cascading_errors -- --nocapture
  run_step "fast_gate-rerun" bash "$ROOT_DIR/scripts/dev/fast_gate.sh"
else
  echo "[traceability-diag] quick mode: skipping strict/lib/full-gate evidence steps"
fi

bash "$ROOT_DIR/scripts/paper/generate_release_traceability_report.sh" "$RUN_DIR"

echo "[traceability-diag] mode=$MODE run_dir=$RUN_DIR"
echo "$RUN_DIR"
