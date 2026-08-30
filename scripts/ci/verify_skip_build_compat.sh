#!/usr/bin/env bash
# scripts/ci/verify_skip_build_compat.sh -- verify that all gate scripts work with SKIP_BUILD=1.
#
# Usage:
#   bash scripts/ci/verify_skip_build_compat.sh
#
# Prerequisite: souc must already be built (cargo build -p souc && cargo build -p souc --release).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SOUC_DEBUG="$ROOT_DIR/target/debug/souc"
SOUC_RELEASE="$ROOT_DIR/target/release/souc"
PASS_COUNT=0
FAIL_COUNT=0

check_binary() {
  local bin="$1"
  local label="$2"
  if [[ ! -x "$bin" ]]; then
    echo "error: $label binary not found at $bin" >&2
    echo "hint: run 'cargo build -p souc' and 'cargo build -p souc --release' first." >&2
    exit 1
  fi
}

run_gate() {
  local name="$1"
  shift
  echo ""
  echo "=== $name ==="
  if env "$@"; then
    PASS_COUNT=$((PASS_COUNT + 1))
    echo "--- PASS: $name ---"
  else
    FAIL_COUNT=$((FAIL_COUNT + 1))
    echo "--- FAIL: $name ---"
  fi
}

check_binary "$SOUC_DEBUG" "debug"
check_binary "$SOUC_RELEASE" "release"

echo "verify_skip_build_compat: using debug=$SOUC_DEBUG release=$SOUC_RELEASE"

run_gate "dev/fast_gate.sh (SKIP_BUILD=1)" \
  SKIP_BUILD=1 SOUC_BIN="$SOUC_DEBUG" SOUNIO_REPO_HARD_NO_RUST=0 \
  bash scripts/dev/fast_gate.sh

run_gate "selfhost_independence_gate.sh (SKIP_BUILD=1)" \
  SKIP_BUILD=1 SOUC_BIN="$SOUC_RELEASE" SOUNIO_REPO_HARD_NO_RUST=0 \
  bash scripts/selfhost_independence_gate.sh

run_gate "selfhost_zero_fallback_gate.sh (SKIP_BUILD=1)" \
  SKIP_BUILD=1 SOUC_BIN="$SOUC_DEBUG" SOUNIO_REPO_HARD_NO_RUST=0 \
  bash scripts/selfhost_zero_fallback_gate.sh

run_gate "build_bootstrap_seed.sh (SKIP_BUILD=1)" \
  SKIP_BUILD=1 SOUC_BIN="$SOUC_DEBUG" SOUNIO_REPO_HARD_NO_RUST=0 \
  bash scripts/build_bootstrap_seed.sh

run_gate "check_feature_matrix.sh (SKIP_BUILD=1)" \
  SKIP_BUILD=1 SOUNIO_REPO_HARD_NO_RUST=0 \
  bash scripts/check_feature_matrix.sh

run_gate "check_golden_snapshots.sh (SKIP_BUILD=1)" \
  SKIP_BUILD=1 SOUNIO_REPO_HARD_NO_RUST=0 \
  bash scripts/check_golden_snapshots.sh

run_gate "check_new_warnings.sh (SKIP_BUILD=1)" \
  SKIP_BUILD=1 SOUNIO_REPO_HARD_NO_RUST=0 \
  bash scripts/ci/check_new_warnings.sh

echo ""
echo "=========================================="
echo "SKIP_BUILD COMPAT: pass=$PASS_COUNT fail=$FAIL_COUNT"
echo "=========================================="

if [[ "$FAIL_COUNT" -gt 0 ]]; then
  exit 1
fi
