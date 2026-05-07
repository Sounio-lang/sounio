#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
source "$ROOT_DIR/scripts/lib/resolve_souc.sh"

SOUC_BIN="${SOUC_BIN:-./souc}"
PROGRAM_DIR="${PROGRAM_DIR:-tests/selfhost-driver-output}"

WORK_DIR="${WORK_DIR:-/tmp/sounio-selfhost-driver-output-gate}"
LOG_DIR="$WORK_DIR/logs"
ARTIFACT_DIR="$WORK_DIR/artifacts"

TIMEOUT_SECS="${TIMEOUT_SECS:-60}"
BUILD_TIMEOUT_SECS="${BUILD_TIMEOUT_SECS:-600}"

STRICT_MODE="${SOUNIO_SELFHOST_STRICT:-1}"
REQUIRE_DRIVER_OUTPUT="${REQUIRE_DRIVER_OUTPUT:-0}"

DRIVER_OUTPUT_PATTERN="SELFHOST=driver-first schema=v1 event=driver_output entrypoint=bootstrap::driver::compile_file status=ok"

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

has_driver_output_marker() {
  local path="$1"
  if command -v rg >/dev/null 2>&1; then
    rg -F -q "$DRIVER_OUTPUT_PATTERN" "$path"
  else
    grep -F -q -- "$DRIVER_OUTPUT_PATTERN" "$path"
  fi
}

run_case() {
  local case_id="$1"
  local program_path="$2"
  local expected_stdout_file="$3"

  local stdout_file="$LOG_DIR/${case_id}.stdout"
  local stderr_file="$LOG_DIR/${case_id}.stderr"
  local exit_file="$ARTIFACT_DIR/${case_id}.exit"
  local marker_file="$ARTIFACT_DIR/${case_id}.driver_output_marker"

  if ! assert_file_exists "$program_path" "$case_id"; then
    return 1
  fi
  if ! assert_file_exists "$expected_stdout_file" "$case_id"; then
    return 1
  fi

  set +e
  run_with_timeout "$TIMEOUT_SECS" env \
    SOUNIO_SELFHOST_STRICT="$STRICT_MODE" \
    "$SOUC_BIN" run "$program_path" >"$stdout_file" 2>"$stderr_file"
  local code=$?
  set -e

  echo "$code" >"$exit_file"

  if [ "$code" -ne 0 ]; then
    fail "$case_id" "non-zero exit (exit=$code)"
    return 1
  fi

  if ! cmp -s "$expected_stdout_file" "$stdout_file"; then
    fail "$case_id" "stdout mismatch (expected=$expected_stdout_file got=$stdout_file)"
    return 1
  fi

  if has_driver_output_marker "$stderr_file"; then
    echo "present" >"$marker_file"
  else
    echo "missing" >"$marker_file"
    if [ "$REQUIRE_DRIVER_OUTPUT" = "1" ]; then
      fail "$case_id" "missing driver_output marker (stderr=$stderr_file)"
      return 1
    fi
  fi

  if [ "$REQUIRE_DRIVER_OUTPUT" = "1" ]; then
    pass "$case_id" "driver_output + stdout match"
  else
    pass "$case_id" "stdout match"
  fi
  return 0
}

mkdir -p "$LOG_DIR" "$ARTIFACT_DIR"

BUILD_LOG="$LOG_DIR/build.log"
SUMMARY_FILE="$ARTIFACT_DIR/summary.txt"

echo "SELFHOST_DRIVER_OUTPUT_GATE_START"
echo "work_dir=$WORK_DIR"
  echo "timeout_secs=$TIMEOUT_SECS"
  echo "strict_mode=$STRICT_MODE"
  echo "require_driver_output=$REQUIRE_DRIVER_OUTPUT"
  echo "program_dir=$PROGRAM_DIR"

{
  echo "SELFHOST_DRIVER_OUTPUT_GATE_INFO build_step=disabled mode=repo-hard-no-rust"
  echo "hint=provide SOUC_BIN prebuilt compiler"
} >"$BUILD_LOG"
pass "build" "cargo build disabled (repo-hard no-rust)"

if [ ! -x "$SOUC_BIN" ]; then
  fail "preflight" "missing compiler binary at $SOUC_BIN"
fi

EXPECTED_DIR="$ARTIFACT_DIR/expected_stdout"
mkdir -p "$EXPECTED_DIR"

cat >"$EXPECTED_DIR/ret_42.stdout" <<'EOF'
42
EOF

cat >"$EXPECTED_DIR/let_x_42.stdout" <<'EOF'
42
EOF

cat >"$EXPECTED_DIR/expr_add_7_35.stdout" <<'EOF'
42
EOF

cat >"$EXPECTED_DIR/expr_mul_6_7.stdout" <<'EOF'
42
EOF

cat >"$EXPECTED_DIR/expr_sub_50_8.stdout" <<'EOF'
42
EOF

cat >"$EXPECTED_DIR/expr_div_84_2.stdout" <<'EOF'
42
EOF

cat >"$EXPECTED_DIR/expr_mod_85_43.stdout" <<'EOF'
42
EOF

cat >"$EXPECTED_DIR/expr_sub_0_42.stdout" <<'EOF'
-42
EOF

cat >"$EXPECTED_DIR/if_true_41_1.stdout" <<'EOF'
41
EOF

cat >"$EXPECTED_DIR/if_lt_1_2_42_0.stdout" <<'EOF'
42
EOF

cat >"$EXPECTED_DIR/if_ge_2_2_42_0.stdout" <<'EOF'
42
EOF

cat >"$EXPECTED_DIR/if_eq_2_2_42_0.stdout" <<'EOF'
42
EOF

cat >"$EXPECTED_DIR/if_ne_1_2_42_0.stdout" <<'EOF'
42
EOF

cat >"$EXPECTED_DIR/print_boot.stdout" <<'EOF'
boot
0
EOF

cat >"$EXPECTED_DIR/two_prints.stdout" <<'EOF'
ab
0
EOF

cat >"$EXPECTED_DIR/println_hi.stdout" <<'EOF'
hi
0
EOF

cat >"$EXPECTED_DIR/let_x_plus_2.stdout" <<'EOF'
42
EOF

cat >"$EXPECTED_DIR/let_xy_add.stdout" <<'EOF'
42
EOF

cat >"$EXPECTED_DIR/let_x_10_plus_32.stdout" <<'EOF'
42
EOF

cat >"$EXPECTED_DIR/let_xy_y_is_x_plus_32.stdout" <<'EOF'
42
EOF

cat >"$EXPECTED_DIR/return_x_plus_2.stdout" <<'EOF'
42
EOF

cat >"$EXPECTED_DIR/let_answer_10_plus_32.stdout" <<'EOF'
42
EOF

cat >"$EXPECTED_DIR/let_first_second_plus_32.stdout" <<'EOF'
42
EOF

cat >"$EXPECTED_DIR/let_three_bindings_chain.stdout" <<'EOF'
42
EOF

if [ "$FAIL_COUNT" -eq 0 ]; then
  run_case "ret_42" "$PROGRAM_DIR/ret_42.sio" "$EXPECTED_DIR/ret_42.stdout" || true
  run_case "let_x_42" "$PROGRAM_DIR/let_x_42.sio" "$EXPECTED_DIR/let_x_42.stdout" || true
  run_case "expr_add_7_35" "$PROGRAM_DIR/expr_add_7_35.sio" "$EXPECTED_DIR/expr_add_7_35.stdout" || true
  run_case "expr_mul_6_7" "$PROGRAM_DIR/expr_mul_6_7.sio" "$EXPECTED_DIR/expr_mul_6_7.stdout" || true
  run_case "expr_sub_50_8" "$PROGRAM_DIR/expr_sub_50_8.sio" "$EXPECTED_DIR/expr_sub_50_8.stdout" || true
  run_case "expr_div_84_2" "$PROGRAM_DIR/expr_div_84_2.sio" "$EXPECTED_DIR/expr_div_84_2.stdout" || true
  run_case "expr_mod_85_43" "$PROGRAM_DIR/expr_mod_85_43.sio" "$EXPECTED_DIR/expr_mod_85_43.stdout" || true
  run_case "expr_sub_0_42" "$PROGRAM_DIR/expr_sub_0_42.sio" "$EXPECTED_DIR/expr_sub_0_42.stdout" || true
  run_case "if_true_41_1" "$PROGRAM_DIR/if_true_41_1.sio" "$EXPECTED_DIR/if_true_41_1.stdout" || true
  run_case "if_lt_1_2_42_0" "$PROGRAM_DIR/if_lt_1_2_42_0.sio" "$EXPECTED_DIR/if_lt_1_2_42_0.stdout" || true
  run_case "if_ge_2_2_42_0" "$PROGRAM_DIR/if_ge_2_2_42_0.sio" "$EXPECTED_DIR/if_ge_2_2_42_0.stdout" || true
  run_case "if_eq_2_2_42_0" "$PROGRAM_DIR/if_eq_2_2_42_0.sio" "$EXPECTED_DIR/if_eq_2_2_42_0.stdout" || true
  run_case "if_ne_1_2_42_0" "$PROGRAM_DIR/if_ne_1_2_42_0.sio" "$EXPECTED_DIR/if_ne_1_2_42_0.stdout" || true
  run_case "print_boot" "$PROGRAM_DIR/print_boot.sio" "$EXPECTED_DIR/print_boot.stdout" || true
  run_case "two_prints" "$PROGRAM_DIR/two_prints.sio" "$EXPECTED_DIR/two_prints.stdout" || true
  run_case "println_hi" "$PROGRAM_DIR/println_hi.sio" "$EXPECTED_DIR/println_hi.stdout" || true
  run_case "let_x_plus_2" "$PROGRAM_DIR/let_x_plus_2.sio" "$EXPECTED_DIR/let_x_plus_2.stdout" || true
  run_case "let_xy_add" "$PROGRAM_DIR/let_xy_add.sio" "$EXPECTED_DIR/let_xy_add.stdout" || true
  run_case "let_x_10_plus_32" "$PROGRAM_DIR/let_x_10_plus_32.sio" "$EXPECTED_DIR/let_x_10_plus_32.stdout" || true
  run_case "let_xy_y_is_x_plus_32" "$PROGRAM_DIR/let_xy_y_is_x_plus_32.sio" "$EXPECTED_DIR/let_xy_y_is_x_plus_32.stdout" || true
  run_case "return_x_plus_2" "$PROGRAM_DIR/return_x_plus_2.sio" "$EXPECTED_DIR/return_x_plus_2.stdout" || true
  run_case "let_answer_10_plus_32" "$PROGRAM_DIR/let_answer_10_plus_32.sio" "$EXPECTED_DIR/let_answer_10_plus_32.stdout" || true
  run_case "let_first_second_plus_32" "$PROGRAM_DIR/let_first_second_plus_32.sio" "$EXPECTED_DIR/let_first_second_plus_32.stdout" || true
  run_case "let_three_bindings_chain" "$PROGRAM_DIR/let_three_bindings_chain.sio" "$EXPECTED_DIR/let_three_bindings_chain.stdout" || true
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
} >"$SUMMARY_FILE"

echo "SELFHOST_DRIVER_OUTPUT_GATE_SUMMARY pass=$PASS_COUNT fail=$FAIL_COUNT artifacts=$WORK_DIR"

if [ "$FAIL_COUNT" -gt 0 ]; then
  exit 1
fi

exit 0
