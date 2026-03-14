#!/usr/bin/env bash
set -euo pipefail

# CI Check for Native Compiler
# This script is intended to be run in CI environments after changes to the
# self‑hosted compiler. It builds the native compiler (if needed) and runs the
# native regression suite.
#
# Usage: ./ci_check_native.sh [--skip-build] [--skip-tests]

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# Configuration
BUILD_DIR="${BUILD_DIR:-target/release}"
NATIVE_COMPILER="${NATIVE_COMPILER:-$BUILD_DIR/souc}"
REGRESSION_SUITE="${REGRESSION_SUITE:-./self-hosted/bootstrap/native_regression_suite.sh}"
CARGO_BUILD_FLAGS="${CARGO_BUILD_FLAGS:---release}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $*"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*"
}

usage() {
    cat <<EOF
CI Check for Native Compiler

Usage: $0 [OPTIONS]

Options:
  --skip-build        Skip building the compiler (assume binary exists)
  --skip-tests        Skip running the regression suite
  --help              Show this help message

Environment variables:
  NATIVE_COMPILER     Path to the native compiler binary (default: target/release/souc)
  BUILD_DIR           Directory where the binary is built (default: target/release)
  CARGO_BUILD_FLAGS   Flags passed to cargo build (default: --release)
EOF
}

SKIP_BUILD=""
SKIP_TESTS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-build)
            SKIP_BUILD="yes"
            shift
            ;;
        --skip-tests)
            SKIP_TESTS="yes"
            shift
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            log_error "Unknown argument: $1"
            usage
            exit 1
            ;;
    esac
done

# ------------------------------------------------------------------------------
# Build the native compiler (if not skipped)
# ------------------------------------------------------------------------------
if [[ -z "$SKIP_BUILD" ]]; then
    log_info "Building the native compiler..."
    if command -v cargo >/dev/null 2>&1; then
        log_info "Running: cargo build $CARGO_BUILD_FLAGS"
        cargo build $CARGO_BUILD_FLAGS
        if [[ ! -x "$NATIVE_COMPILER" ]]; then
            log_error "Compiler binary not found after build: $NATIVE_COMPILER"
            exit 1
        fi
    else
        log_error "cargo not found. Cannot build the compiler."
        exit 1
    fi
else
    log_info "Skipping compiler build."
    if [[ ! -x "$NATIVE_COMPILER" ]]; then
        log_error "Compiler binary not found: $NATIVE_COMPILER"
        log_error "Use --skip-build only when the binary already exists."
        exit 1
    fi
fi

log_info "Using native compiler: $NATIVE_COMPILER"
log_info "Compiler version:"
$NATIVE_COMPILER --version 2>&1 || true

# ------------------------------------------------------------------------------
# Run the regression suite (if not skipped)
# ------------------------------------------------------------------------------
if [[ -z "$SKIP_TESTS" ]]; then
    if [[ ! -x "$REGRESSION_SUITE" ]]; then
        log_error "Regression suite script not found: $REGRESSION_SUITE"
        exit 1
    fi
    log_info "Running native regression suite..."
    # Run with the native compiler we just built
    NATIVE_COMPILER="$NATIVE_COMPILER" \
    LOG_DIR="artifacts/ci_native/logs" \
    REPORT_DIR="artifacts/ci_native/reports" \
    TIMEOUT_SECS=60 \
        "$REGRESSION_SUITE" --native-compiler "$NATIVE_COMPILER"
    exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        log_error "Regression suite failed. See logs in artifacts/ci_native/"
        exit $exit_code
    fi
    log_info "Regression suite passed."
else
    log_info "Skipping regression suite."
fi

# ------------------------------------------------------------------------------
# Optional: additional checks (e.g., compile self‑hosted with native)
# ------------------------------------------------------------------------------
log_info "CI check completed successfully."
exit 0