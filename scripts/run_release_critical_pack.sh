#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="$ROOT_DIR/artifacts/diagnostic/$TS"

mkdir -p "$RUN_DIR"
echo "[release-pack] artifacts=$RUN_DIR"

run_step() {
  local label="$1"
  shift
  echo "[release-pack] >>> $label"
  SOUNIO_DIAG_RUN_DIR="$RUN_DIR" \
  SOUNIO_DIAG_LOG_LABEL="$label" \
  bash "$ROOT_DIR/scripts/with_isolated_env.sh" "$@"
}

run_step "01-cargo-check" cargo check -p souc
run_step "02-strict-driver-source" env SOUNIO_SELFHOST_DRIVER_STRICT=1 cargo test -p souc compiler_loader::tests::test_driver_source_pipeline_compiles_simple_source -- --nocapture
run_step "03-strict-driver-file" env SOUNIO_SELFHOST_DRIVER_STRICT=1 cargo test -p souc compiler_loader::tests::test_driver_file_pipeline_compiles_simple_file -- --nocapture
run_step "04-cargo-lib-tests" cargo test -p souc --lib
run_step "05-fast-gate" bash "$ROOT_DIR/scripts/fast_gate.sh"
run_step "06-website-quality" npm --prefix "$ROOT_DIR/website" run check:quality

echo "[release-pack] PASS"

