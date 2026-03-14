#!/usr/bin/env bash
set -euo pipefail

# Native Compiler Regression Suite
# Runs a set of frontend tests with the native (AST‑direct) compiler.
# Usage: ./native_regression_suite.sh [--jit] [--native-compiler <path>] [--test-dir <dir>]

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# Configuration
NATIVE_COMPILER="${NATIVE_COMPILER:-./souc}"   # path to native compiler binary
TEST_DIR="${TEST_DIR:-tests/frontend}"
LOG_DIR="${LOG_DIR:-artifacts/native_regression/logs}"
REPORT_DIR="${REPORT_DIR:-artifacts/native_regression/reports}"
TIMEOUT_SECS="${TIMEOUT_SECS:-30}"
MODE="${MODE:-native}"  # 'native' or 'jit'

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
Native Compiler Regression Suite

Usage: $0 [OPTIONS]

Options:
  --jit                     Run tests with the JIT backend (default: native)
  --native-compiler PATH    Path to the native compiler binary (default: ./souc)
  --test-dir DIR            Directory containing .sio test files (default: tests/frontend)
  --list                    List test files that would be run and exit
  --help                    Show this help message

Environment variables:
  NATIVE_COMPILER          same as --native-compiler
  TEST_DIR                 same as --test-dir
  LOG_DIR                  where to store per‑test logs
  REPORT_DIR               where to store summary reports
  TIMEOUT_SECS             timeout per test in seconds
  MODE                     'native' or 'jit'
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --jit)
            MODE="jit"
            shift
            ;;
        --native-compiler)
            NATIVE_COMPILER="$2"
            shift 2
            ;;
        --test-dir)
            TEST_DIR="$2"
            shift 2
            ;;
        --list)
            LIST_ONLY="yes"
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

# Determine command line for the compiler
case "$MODE" in
    native)
        COMPILE_CMD="$NATIVE_COMPILER native compile"
        RUN_CMD="$NATIVE_COMPILER native run"
        ;;
    jit)
        COMPILE_CMD="$NATIVE_COMPILER compile"
        RUN_CMD="$NATIVE_COMPILER run"
        ;;
    *)
        log_error "Invalid MODE: $MODE (must be 'native' or 'jit')"
        exit 1
        ;;
esac

# Check if compiler binary exists
if [[ ! -x "$NATIVE_COMPILER" ]]; then
    log_error "Compiler binary not found or not executable: $NATIVE_COMPILER"
    log_error "Please build the native compiler or set NATIVE_COMPILER environment variable."
    exit 1
fi

# Collect test files (same set as sprint1 frontend tests)
TEST_FILES=(
    "$TEST_DIR/minimal_passing.sio"
    "$TEST_DIR/smoke_test.sio"
    "$TEST_DIR/lexer_integration.sio"
    "$TEST_DIR/simple_lexer_test.sio"
    "$TEST_DIR/parser_golden.sio"
    "$TEST_DIR/lexer_unit.sio"
)

filtered_test_files=()
for file in "${TEST_FILES[@]}"; do
    if [[ -f "$file" ]]; then
        filtered_test_files+=("$file")
    else
        log_warn "Skipping missing test file: $file"
    fi
done
TEST_FILES=("${filtered_test_files[@]}")

if [[ ${#TEST_FILES[@]} -eq 0 ]]; then
    log_error "No test files found in $TEST_DIR"
    exit 1
fi

if [[ "${LIST_ONLY:-}" == "yes" ]]; then
    log_info "Test files that would be run:"
    for file in "${TEST_FILES[@]}"; do
        echo "  $file"
    done
    exit 0
fi

# Setup directories
mkdir -p "$LOG_DIR" "$REPORT_DIR"

# Summary files
SUMMARY_FILE="$REPORT_DIR/summary.txt"
FAILURES_FILE="$REPORT_DIR/failures.txt"
echo "Native Regression Suite - $(date)" > "$SUMMARY_FILE"
echo "Mode: $MODE" >> "$SUMMARY_FILE"
echo "Compiler: $NATIVE_COMPILER" >> "$SUMMARY_FILE"
echo "" > "$FAILURES_FILE"

log_info "Starting Native Regression Suite"
log_info "Mode: $MODE"
log_info "Compiler: $NATIVE_COMPILER"
log_info "Test directory: $TEST_DIR"
log_info "Timeout: ${TIMEOUT_SECS}s per test"

# Run tests
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_TIMEOUT=0

for test_file in "${TEST_FILES[@]}"; do
    test_name="$(basename "$test_file" .sio)"
    log_file="$LOG_DIR/${test_name}.log"
    
    log_info "Running test: $test_name"
    
    set +e
    timeout "$TIMEOUT_SECS" $RUN_CMD "$test_file" >"$log_file" 2>&1
    exit_code=$?
    set -e
    
    if [[ $exit_code -eq 0 ]]; then
        status="PASS"
        ((TESTS_PASSED++))
        log_info "Test passed: $test_name"
    elif [[ $exit_code -eq 124 ]]; then
        status="TIMEOUT"
        ((TESTS_TIMEOUT++))
        log_error "Test timed out: $test_name (${TIMEOUT_SECS}s)"
    else
        status="FAIL"
        ((TESTS_FAILED++))
        log_error "Test failed: $test_name (exit code: $exit_code)"
        echo "=== $test_name ===" >> "$FAILURES_FILE"
        cat "$log_file" >> "$FAILURES_FILE"
        echo -e "\n" >> "$FAILURES_FILE"
    fi
    echo "$status $test_name" >> "$SUMMARY_FILE"
done

# Print summary
log_info "Test run completed"
log_info "Passed: $TESTS_PASSED"
log_info "Failed: $TESTS_FAILED"
log_info "Timeout: $TESTS_TIMEOUT"

# Write a simple JSON report (optional)
JSON_REPORT="$REPORT_DIR/report.json"
cat > "$JSON_REPORT" <<EOF
{
  "suite": "native_regression",
  "mode": "$MODE",
  "compiler": "$NATIVE_COMPILER",
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "results": {
    "passed": $TESTS_PASSED,
    "failed": $TESTS_FAILED,
    "timeout": $TESTS_TIMEOUT
  }
}
EOF

if [[ $TESTS_FAILED -eq 0 && $TESTS_TIMEOUT -eq 0 ]]; then
    log_info "All tests passed."
    exit 0
else
    log_error "Some tests failed. See $FAILURES_FILE for details."
    exit 1
fi