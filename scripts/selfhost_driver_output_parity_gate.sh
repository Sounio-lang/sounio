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
MATRIX_FAIL_COUNT=0

pass() {
  PASS_COUNT=$((PASS_COUNT + 1))
  echo "PASS [$1] $2"
}

fail() {
  FAIL_COUNT=$((FAIL_COUNT + 1))
  echo "FAIL [$1] $2"
}

init_parity_matrix() {
  cat >"$PARITY_MATRIX_FILE" <<'EOF'
case_id	program	driver_exit	rust_exit	driver_output_marker	stdout_parity	parity
EOF
}

append_parity_row() {
  local case_id="$1"
  local program_path="$2"
  local driver_exit="$3"
  local rust_exit="$4"
  local marker_status="$5"
  local stdout_status="$6"
  local parity_status="$7"

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$case_id" "$program_path" "$driver_exit" "$rust_exit" "$marker_status" "$stdout_status" "$parity_status" \
    >>"$PARITY_MATRIX_FILE"

  if [ "$parity_status" != "ok" ]; then
    MATRIX_FAIL_COUNT=$((MATRIX_FAIL_COUNT + 1))
  fi
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
  local driver_marker_status="missing"
  local stdout_status="mismatch"
  local parity_status="fail"

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
    SOUNIO_RUST_GHOST="1" \
    "$SOUC_BIN" run "$program_path" >"$rust_stdout_file" 2>"$rust_stderr_file"
  local rust_code=$?
  set -e
  echo "$rust_code" >"$rust_exit_file"

  if command -v rg >/dev/null 2>&1; then
    if rg -n "SELFHOST=driver-first schema=v1 event=driver_output entrypoint=bootstrap::driver::compile_file status=ok" "$driver_stderr_file" >/dev/null; then
      driver_marker_status="ok"
    fi
  else
    if grep -E -q "SELFHOST=driver-first schema=v1 event=driver_output entrypoint=bootstrap::driver::compile_file status=ok" "$driver_stderr_file"; then
      driver_marker_status="ok"
    fi
  fi

  if cmp -s "$driver_stdout_file" "$rust_stdout_file"; then
    stdout_status="ok"
  fi

  if [ "$driver_code" -eq 0 ] && [ "$rust_code" -eq 0 ] \
    && [ "$driver_marker_status" = "ok" ] && [ "$stdout_status" = "ok" ]; then
    parity_status="ok"
  fi

  append_parity_row \
    "$case_id" \
    "$program_path" \
    "$driver_code" \
    "$rust_code" \
    "$driver_marker_status" \
    "$stdout_status" \
    "$parity_status"

  if [ "$parity_status" = "ok" ]; then
    pass "$case_id" "driver_output stdout matches rust pipeline"
    return 0
  fi

  if [ "$driver_code" -ne 0 ]; then
    fail "$case_id" "driver pipeline non-zero exit (exit=$driver_code)"
  elif [ "$rust_code" -ne 0 ]; then
    fail "$case_id" "rust pipeline non-zero exit (exit=$rust_code)"
  elif [ "$driver_marker_status" != "ok" ]; then
    fail "$case_id" "missing driver_output marker (driver_stderr=$driver_stderr_file)"
  else
    fail "$case_id" "stdout mismatch (driver=$driver_stdout_file rust=$rust_stdout_file)"
  fi

  return 1
}

mkdir -p "$LOG_DIR" "$ARTIFACT_DIR"

BUILD_LOG="$LOG_DIR/build.log"
SUMMARY_FILE="$ARTIFACT_DIR/summary.txt"
PARITY_MATRIX_FILE="$ARTIFACT_DIR/parity_matrix.tsv"

init_parity_matrix
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
  echo "parity_matrix_file=$PARITY_MATRIX_FILE"
  echo "matrix_fail=$MATRIX_FAIL_COUNT"
} >"$SUMMARY_FILE"

echo "SELFHOST_DRIVER_OUTPUT_PARITY_MATRIX file=$PARITY_MATRIX_FILE matrix_fail=$MATRIX_FAIL_COUNT"
echo "SELFHOST_DRIVER_OUTPUT_PARITY_GATE_SUMMARY pass=$PASS_COUNT fail=$FAIL_COUNT artifacts=$WORK_DIR"

if [ "$FAIL_COUNT" -gt 0 ] || [ "$MATRIX_FAIL_COUNT" -gt 0 ]; then
  exit 1
fi

exit 0
