#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SOUC_BIN="${SOUC_BIN:-./target/debug/souc}"
PROGRAM_DIR="${PROGRAM_DIR:-tests/selfhost-driver-output}"

WORK_DIR="${WORK_DIR:-/tmp/sounio-selfhost-driver-output-parity-gate}"
LOG_DIR="$WORK_DIR/logs"
ARTIFACT_DIR="$WORK_DIR/artifacts"

TIMEOUT_SECS="${TIMEOUT_SECS:-60}"
BUILD_TIMEOUT_SECS="${BUILD_TIMEOUT_SECS:-600}"

STRICT_MODE="${SOUNIO_SELFHOST_STRICT:-1}"
REQUIRE_DRIVER_OUTPUT="${SOUNIO_SELFHOST_DRIVER_REQUIRE_OUTPUT:-1}"

PASS_COUNT=0
FAIL_COUNT=0

pass() {
  PASS_COUNT=$((PASS_COUNT + 1))
  echo "PASS [$1] $2"
}

fail() {
  FAIL_COUNT=$((FAIL_COUNT + 1))
  echo "FAIL [$1] $2"
}

run_with_timeout() {
  local seconds="$1"
  shift

  if command -v timeout >/dev/null 2>&1; then
    timeout --preserve-status "${seconds}s" "$@"
    return $?
  fi

  if command -v python3 >/dev/null 2>&1; then
    python3 - "$seconds" "$@" <<'PY'
import subprocess
import sys

seconds = int(sys.argv[1])
command = sys.argv[2:]
try:
    completed = subprocess.run(command, timeout=seconds)
    sys.exit(completed.returncode)
except subprocess.TimeoutExpired:
    sys.exit(124)
PY
    return $?
  fi

  "$@"
}

assert_file_exists() {
  local path="$1"
  local case_id="$2"
  if [ ! -f "$path" ]; then
    fail "$case_id" "missing file: $path"
    return 1
  fi
  return 0
}

run_case() {
  local case_id="$1"
  local program_path="$2"

  local driver_stdout_file="$LOG_DIR/${case_id}.driver.stdout"
  local driver_stderr_file="$LOG_DIR/${case_id}.driver.stderr"
  local driver_exit_file="$ARTIFACT_DIR/${case_id}.driver.exit"

  local rust_stdout_file="$LOG_DIR/${case_id}.rust.stdout"
  local rust_stderr_file="$LOG_DIR/${case_id}.rust.stderr"
  local rust_exit_file="$ARTIFACT_DIR/${case_id}.rust.exit"

  if ! assert_file_exists "$program_path" "$case_id"; then
    return 1
  fi

  set +e
  run_with_timeout "$TIMEOUT_SECS" env \
    SOUNIO_SELFHOST_STRICT="$STRICT_MODE" \
    SOUNIO_SELFHOST_DRIVER_REQUIRE_OUTPUT="$REQUIRE_DRIVER_OUTPUT" \
    SOUNIO_SELFHOST_PIPELINE="driver" \
    "$SOUC_BIN" run "$program_path" >"$driver_stdout_file" 2>"$driver_stderr_file"
  local driver_code=$?
  set -e
  echo "$driver_code" >"$driver_exit_file"

  set +e
  run_with_timeout "$TIMEOUT_SECS" env \
    SOUNIO_SELFHOST_STRICT="$STRICT_MODE" \
    SOUNIO_SELFHOST_DRIVER_REQUIRE_OUTPUT="0" \
    SOUNIO_SELFHOST_PIPELINE="rust" \
    "$SOUC_BIN" run "$program_path" >"$rust_stdout_file" 2>"$rust_stderr_file"
  local rust_code=$?
  set -e
  echo "$rust_code" >"$rust_exit_file"

  if [ "$driver_code" -ne 0 ]; then
    fail "$case_id" "driver pipeline non-zero exit (exit=$driver_code)"
    return 1
  fi
  if [ "$rust_code" -ne 0 ]; then
    fail "$case_id" "rust pipeline non-zero exit (exit=$rust_code)"
    return 1
  fi

  if command -v rg >/dev/null 2>&1; then
    if ! rg -n "SELFHOST=driver-first schema=v1 event=driver_output entrypoint=bootstrap::driver::compile_file status=ok" "$driver_stderr_file" >/dev/null; then
      fail "$case_id" "missing driver_output marker (driver_stderr=$driver_stderr_file)"
      return 1
    fi
  else
    if ! grep -E -q "SELFHOST=driver-first schema=v1 event=driver_output entrypoint=bootstrap::driver::compile_file status=ok" "$driver_stderr_file"; then
      fail "$case_id" "missing driver_output marker (driver_stderr=$driver_stderr_file)"
      return 1
    fi
  fi

  if ! cmp -s "$driver_stdout_file" "$rust_stdout_file"; then
    fail "$case_id" "stdout mismatch (driver=$driver_stdout_file rust=$rust_stdout_file)"
    return 1
  fi

  pass "$case_id" "driver_output stdout matches rust pipeline"
  return 0
}

mkdir -p "$LOG_DIR" "$ARTIFACT_DIR"

BUILD_LOG="$LOG_DIR/build.log"
SUMMARY_FILE="$ARTIFACT_DIR/summary.txt"
PROGRAM_LIST_FILE="$ARTIFACT_DIR/programs.txt"

echo "SELFHOST_DRIVER_OUTPUT_PARITY_GATE_START"
echo "work_dir=$WORK_DIR"
echo "timeout_secs=$TIMEOUT_SECS"
echo "strict_mode=$STRICT_MODE"
echo "require_driver_output=$REQUIRE_DRIVER_OUTPUT"
echo "program_dir=$PROGRAM_DIR"

set +e
run_with_timeout "$BUILD_TIMEOUT_SECS" cargo build -p souc >"$BUILD_LOG" 2>&1
build_code=$?
set -e
if [ "$build_code" -eq 0 ]; then
  pass "build" "cargo build -p souc"
else
  fail "build" "cargo build failed (exit=$build_code)"
fi

if [ ! -x "$SOUC_BIN" ]; then
  fail "preflight" "missing compiler binary at $SOUC_BIN"
fi

set +e
(cd "$PROGRAM_DIR" && ls -1 *.sio 2>/dev/null | sort) >"$PROGRAM_LIST_FILE"
list_code=$?
set -e
if [ "$list_code" -ne 0 ]; then
  fail "preflight" "failed to enumerate programs in $PROGRAM_DIR"
fi

if [ ! -s "$PROGRAM_LIST_FILE" ]; then
  fail "preflight" "no .sio programs found in $PROGRAM_DIR"
fi

if [ "$FAIL_COUNT" -eq 0 ]; then
  while IFS= read -r program_file; do
    case_id="${program_file%.sio}"
    run_case "$case_id" "$PROGRAM_DIR/$program_file" || true
  done <"$PROGRAM_LIST_FILE"
fi

{
  echo "summary_pass=$PASS_COUNT"
  echo "summary_fail=$FAIL_COUNT"
  echo "timeout_secs=$TIMEOUT_SECS"
  echo "build_timeout_secs=$BUILD_TIMEOUT_SECS"
  echo "strict_mode=$STRICT_MODE"
  echo "require_driver_output=$REQUIRE_DRIVER_OUTPUT"
  echo "program_dir=$PROGRAM_DIR"
  echo "log_dir=$LOG_DIR"
  echo "program_list_file=$PROGRAM_LIST_FILE"
} >"$SUMMARY_FILE"

echo "SELFHOST_DRIVER_OUTPUT_PARITY_GATE_SUMMARY pass=$PASS_COUNT fail=$FAIL_COUNT artifacts=$WORK_DIR"

if [ "$FAIL_COUNT" -gt 0 ]; then
  exit 1
fi

exit 0

