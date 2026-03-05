#!/usr/bin/env bash
set -euo pipefail

# Sprint 1 Test Runner
# Frontend (Lexer + Parser) Stabilization Tests

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Configuration
SOUC_BIN="${SOUC_BIN:-./souc}"
TEST_DIR="tests/frontend"
LOG_DIR="artifacts/sprint1/logs"
REPORT_DIR="artifacts/sprint1/reports"
STATUS_JSON="$REPORT_DIR/frontend_test_status.v1.json"
PARSER_DEEP_JSON="$REPORT_DIR/parser_golden_deep_probe.v1.json"
PARSER_DEEP_LOG="$LOG_DIR/parser_deep_probe.log"
TIMEOUT_SECS=30

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $*"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*"
}

sanitize_reason_token() {
    local raw="$1"
    local token
    token="$(printf '%s' "$raw" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/_/g; s/^_+//; s/_+$//; s/_+/_/g')"
    if [ -z "$token" ]; then
        token="unknown"
    fi
    printf '%s' "$token"
}

emit_status_json() {
    local status="$1"
    local reason="$2"
    local parser_deep_status="$3"
    local parser_deep_reason="$4"
    local total="$5"
    local passed="$6"
    local failed="$7"
    local timeout="$8"
    local pass_rate="$9"
    shift 9

    python3 - "$STATUS_JSON" "$status" "$reason" "$parser_deep_status" "$parser_deep_reason" "$total" "$passed" "$failed" "$timeout" "$pass_rate" "$SOUC_BIN" "$@" <<'PY'
import datetime as dt
import json
import sys

(
    out_path,
    status,
    reason,
    parser_deep_status,
    parser_deep_reason,
    total,
    passed,
    failed,
    timeout,
    pass_rate,
    souc_bin,
    *tests,
) = sys.argv[1:]

def to_int(raw: str):
    try:
        return int(raw)
    except Exception:
        return 0

payload = {
    "schema": "sounio.sprint1.frontend_tests.status.v1",
    "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    "status": status,
    "reason": reason,
    "souc_bin": souc_bin,
    "parser_deep_probe": {
        "status": parser_deep_status,
        "reason": parser_deep_reason,
        "json_path": "artifacts/sprint1/reports/parser_golden_deep_probe.v1.json",
    },
    "metrics": {
        "total": to_int(total),
        "passed": to_int(passed),
        "failed": to_int(failed),
        "timeout": to_int(timeout),
        "pass_rate_percent": to_int(pass_rate),
    },
    "tests": tests,
}

with open(out_path, "w", encoding="utf-8") as fh:
    json.dump(payload, fh, indent=2)
    fh.write("\n")
PY
}

run_test() {
    local test_file="$1"
    local test_name="$(basename "$test_file" .sio)"
    local log_file="$LOG_DIR/${test_name}.log"
    
    log_info "Running test: $test_name"
    
    set +e
    timeout "$TIMEOUT_SECS" "$SOUC_BIN" run "$test_file" >"$log_file" 2>&1
    local exit_code=$?
    set -e
    
    if [ $exit_code -eq 0 ]; then
        RUN_TEST_STATUS="PASS"
        echo "PASS" >> "$REPORT_DIR/summary.txt"
        log_info "Test passed: $test_name"
        return 0
    elif [ $exit_code -eq 124 ]; then
        RUN_TEST_STATUS="TIMEOUT"
        echo "TIMEOUT" >> "$REPORT_DIR/summary.txt"
        log_error "Test timed out: $test_name (${TIMEOUT_SECS}s)"
        return 1
    else
        RUN_TEST_STATUS="FAIL"
        echo "FAIL" >> "$REPORT_DIR/summary.txt"
        log_error "Test failed: $test_name (exit code: $exit_code)"
        echo "=== Test Output ===" >> "$REPORT_DIR/failures.txt"
        cat "$log_file" >> "$REPORT_DIR/failures.txt"
        echo "\n" >> "$REPORT_DIR/failures.txt"
        return 1
    fi
}

# Setup directories
mkdir -p "$LOG_DIR" "$REPORT_DIR"

echo "Sprint 1 Test Run - $(date)" > "$REPORT_DIR/summary.txt"
echo "======================" >> "$REPORT_DIR/summary.txt"
echo "" > "$REPORT_DIR/failures.txt"

# Check if souc binary exists
if [ ! -x "$SOUC_BIN" ]; then
    log_error "Compiler binary not found or not executable: $SOUC_BIN"
    log_error "Please build with: ./bootstrap/poseidon/poseidon self-hosted/main.sio -o souc"
    emit_status_json "fail" "compiler_binary_missing" "not_run" "not_executed" "0" "0" "0" "0" "0"
    exit 1
fi

# Set stdlib path
export SOUNIO_STDLIB_PATH="$ROOT_DIR/stdlib"
log_info "Stdlib path: $SOUNIO_STDLIB_PATH"

log_info "Starting Sprint 1 Tests"
log_info "Compiler: $SOUC_BIN"
log_info "Test directory: $TEST_DIR"
log_info "Timeout: ${TIMEOUT_SECS}s per test"

# Run tests
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_TIMEOUT=0
RUN_TEST_STATUS=""

# Explicit test entrypoints (helper modules in this folder are excluded)
test_files=(
    "$TEST_DIR/minimal_passing.sio"
    "$TEST_DIR/smoke_test.sio"
    "$TEST_DIR/lexer_integration.sio"
    "$TEST_DIR/simple_lexer_test.sio"
    "$TEST_DIR/parser_golden.sio"
    "$TEST_DIR/lexer_unit.sio"
)

filtered_test_files=()
for file in "${test_files[@]}"; do
    if [ -f "$file" ]; then
        filtered_test_files+=("$file")
    else
        log_warn "Skipping missing test file: $file"
    fi
done
test_files=("${filtered_test_files[@]}")

if [ ${#test_files[@]} -eq 0 ]; then
    log_warn "No test files found in $TEST_DIR"
    emit_status_json "skip" "no_test_files_found" "not_run" "not_executed" "0" "0" "0" "0" "0"
    exit 0
fi

log_info "Found ${#test_files[@]} test files"

# Run each test
for test_file in "${test_files[@]}"; do
    if run_test "$test_file"; then
        TESTS_PASSED=$((TESTS_PASSED + 1))
    else
        if [ "$RUN_TEST_STATUS" = "TIMEOUT" ]; then
            TESTS_TIMEOUT=$((TESTS_TIMEOUT + 1))
        elif [ "$RUN_TEST_STATUS" = "FAIL" ]; then
            TESTS_FAILED=$((TESTS_FAILED + 1))
        fi
    fi
    
    # Small delay between tests
    sleep 0.1

done

# Generate report
echo "" >> "$REPORT_DIR/summary.txt"
echo "======================" >> "$REPORT_DIR/summary.txt"
echo "Total tests: ${#test_files[@]}" >> "$REPORT_DIR/summary.txt"
echo "Passed: $TESTS_PASSED" >> "$REPORT_DIR/summary.txt"
echo "Failed: $TESTS_FAILED" >> "$REPORT_DIR/summary.txt"
echo "Timeout: $TESTS_TIMEOUT" >> "$REPORT_DIR/summary.txt"

# Print summary
log_info "\n=== Sprint 1 Test Summary ==="
log_info "Total tests: ${#test_files[@]}"
log_info "Passed: $TESTS_PASSED"

if [ $TESTS_FAILED -gt 0 ]; then
    log_error "Failed: $TESTS_FAILED"
    log_error "See $REPORT_DIR/failures.txt for details"
fi

if [ $TESTS_TIMEOUT -gt 0 ]; then
    log_warn "Timeout: $TESTS_TIMEOUT"
fi

PARSER_DEEP_STATUS="not_run"
PARSER_DEEP_REASON="not_executed"
set +e
bash "$ROOT_DIR/scripts/sprint1_parser_deep_probe.sh" "$PARSER_DEEP_JSON" > "$PARSER_DEEP_LOG" 2>&1
PARSER_DEEP_RC=$?
set -e
if [ "$PARSER_DEEP_RC" -ne 0 ]; then
    PARSER_DEEP_STATUS="fail"
    PARSER_DEEP_REASON="probe_script_failed_rc_${PARSER_DEEP_RC}"
    python3 - "$PARSER_DEEP_JSON" "$PARSER_DEEP_REASON" <<'PY'
import datetime as dt
import json
import sys

out_path = sys.argv[1]
reason = sys.argv[2]
payload = {
    "schema": "sounio.sprint1.parser_deep_probe.v1",
    "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    "status": "fail",
    "reason": reason,
    "timeout_seconds": 0,
    "metrics": {
        "total": 0,
        "passed": 0,
        "failed": 0,
        "timeout": 0,
        "not_run": 0,
    },
    "cases": [],
}

with open(out_path, "w", encoding="utf-8") as fh:
    json.dump(payload, fh, indent=2)
    fh.write("\n")
PY
else
    parser_deep_fields="$(python3 - "$PARSER_DEEP_JSON" <<'PY'
import json
import sys

path = sys.argv[1]
try:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
except FileNotFoundError:
    print("not_run|probe_json_missing")
    sys.exit(0)
except json.JSONDecodeError:
    print("fail|probe_json_invalid")
    sys.exit(0)

status = str(data.get("status", "not_run"))
reason = str(data.get("reason", "unknown"))
if not reason:
    reason = "unknown"
print(f"{status}|{reason}")
PY
)"
    IFS='|' read -r PARSER_DEEP_STATUS PARSER_DEEP_REASON <<< "$parser_deep_fields"
fi
echo "Parser deep probe: ${PARSER_DEEP_STATUS} (${PARSER_DEEP_REASON})" >> "$REPORT_DIR/summary.txt"
log_info "Parser deep probe: ${PARSER_DEEP_STATUS} (${PARSER_DEEP_REASON})"

if [ "$PARSER_DEEP_STATUS" = "timeout" ]; then
    log_error "Parser deep probe timeout: ${PARSER_DEEP_REASON}"
elif [ "$PARSER_DEEP_STATUS" = "fail" ]; then
    log_error "Parser deep probe failure: ${PARSER_DEEP_REASON}"
elif [ "$PARSER_DEEP_STATUS" != "pass" ]; then
    log_warn "Parser deep probe non-pass state: ${PARSER_DEEP_STATUS} (${PARSER_DEEP_REASON})"
fi

# Calculate pass rate
if [ ${#test_files[@]} -gt 0 ]; then
    PASS_RATE=$((TESTS_PASSED * 100 / ${#test_files[@]}))
    log_info "Pass rate: ${PASS_RATE}%"
    
    echo "Pass rate: ${PASS_RATE}%" >> "$REPORT_DIR/summary.txt"
    
    # Check if we meet the 90% target
    if [ $PASS_RATE -ge 90 ]; then
        log_info "✅ Sprint 1 target met (≥90% pass rate)"
        echo "STATUS: PASS" >> "$REPORT_DIR/summary.txt"
        OVERALL_STATUS="pass"
        OVERALL_REASON="target_met"
        EXIT_CODE=0
    else
        log_error "❌ Sprint 1 target not met (${PASS_RATE}% < 90%)"
        echo "STATUS: FAIL" >> "$REPORT_DIR/summary.txt"
        OVERALL_STATUS="fail"
        OVERALL_REASON="pass_rate_below_target"
        EXIT_CODE=1
    fi
else
    log_warn "No tests executed"
    echo "STATUS: SKIP" >> "$REPORT_DIR/summary.txt"
    OVERALL_STATUS="skip"
    OVERALL_REASON="no_tests_executed"
    EXIT_CODE=0
fi

REQUIRE_PARSER_DEEP_RAW="${SOUNIO_SPRINT1_REQUIRE_PARSER_DEEP:-1}"
PARSER_DEEP_REQUIRED=1
if [ "$REQUIRE_PARSER_DEEP_RAW" = "0" ] || [ "$REQUIRE_PARSER_DEEP_RAW" = "false" ] || [ "$REQUIRE_PARSER_DEEP_RAW" = "no" ]; then
    PARSER_DEEP_REQUIRED=0
fi

if [ "$PARSER_DEEP_REQUIRED" -eq 1 ] && [ "$PARSER_DEEP_STATUS" != "pass" ]; then
    parser_reason_token="$(sanitize_reason_token "$PARSER_DEEP_REASON")"
    OVERALL_STATUS="fail"
    OVERALL_REASON="parser_deep_probe_${PARSER_DEEP_STATUS}_${parser_reason_token}"
    EXIT_CODE=1
    echo "STATUS: FAIL (parser deep probe required: ${PARSER_DEEP_STATUS}/${PARSER_DEEP_REASON})" >> "$REPORT_DIR/summary.txt"
fi

echo "FINAL_STATUS: ${OVERALL_STATUS}" >> "$REPORT_DIR/summary.txt"
echo "FINAL_REASON: ${OVERALL_REASON}" >> "$REPORT_DIR/summary.txt"
echo "PARSER_DEEP_REQUIRED: ${PARSER_DEEP_REQUIRED}" >> "$REPORT_DIR/summary.txt"
echo "PARSER_DEEP_STATUS: ${PARSER_DEEP_STATUS}" >> "$REPORT_DIR/summary.txt"
echo "PARSER_DEEP_REASON: ${PARSER_DEEP_REASON}" >> "$REPORT_DIR/summary.txt"

emit_status_json "$OVERALL_STATUS" "$OVERALL_REASON" "$PARSER_DEEP_STATUS" "$PARSER_DEEP_REASON" "${#test_files[@]}" "$TESTS_PASSED" "$TESTS_FAILED" "$TESTS_TIMEOUT" "${PASS_RATE:-0}" "${test_files[@]}"

# Copy summary to artifacts for CI
cp "$REPORT_DIR/summary.txt" "artifacts/sprint1_test_summary.txt"

log_info "\nDetailed logs: $LOG_DIR"
log_info "Test report: $REPORT_DIR/summary.txt"

if [ $TESTS_FAILED -gt 0 ]; then
    log_info "Failure details: $REPORT_DIR/failures.txt"
fi

exit $EXIT_CODE
