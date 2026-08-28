#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SOUC_NATIVE="${SOUC_NATIVE:-$ROOT_DIR/artifacts/self-hosted/souc-self-hosted-x86_64}"
MANIFEST_PATH="${MANIFEST_PATH:-tests/selfhost/native_runtime/manifest.tsv}"
WORK_DIR="${WORK_DIR:-/tmp/sounio-selfhost-native-runtime-proof}"
ARTIFACT_DIR="$WORK_DIR/artifacts"
LOG_DIR="$WORK_DIR/logs"
TIMEOUT_SECS="${TIMEOUT_SECS:-30}"
FILTER="${FILTER:-}"
FAIL_FAST="${FAIL_FAST:-${SOUNIO_NATIVE_FAIL_FAST:-0}}"
SOUNIO_NATIVE_TARGET="${SOUNIO_NATIVE_TARGET:-}"
SOUNIO_NATIVE_HOST_OS="${SOUNIO_NATIVE_HOST_OS:-$(uname -s)}"
SOUNIO_NATIVE_HOST_MACHINE="${SOUNIO_NATIVE_HOST_MACHINE:-$(uname -m)}"

if [ -z "$SOUNIO_NATIVE_TARGET" ] && [ "$SOUNIO_NATIVE_HOST_OS" = "Darwin" ]; then
  case "$SOUNIO_NATIVE_HOST_MACHINE" in
    arm64|aarch64) SOUNIO_NATIVE_TARGET="aarch64-macos" ;;
    x86_64) SOUNIO_NATIVE_TARGET="x86_64-macos" ;;
  esac
fi

PASS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0

run_with_timeout() {
  local seconds="$1"
  shift

  if command -v timeout >/dev/null 2>&1; then
    timeout --preserve-status "${seconds}s" "$@"
    return $?
  fi

  "$@" &
  local pid=$!
  local elapsed=0
  while kill -0 "$pid" >/dev/null 2>&1; do
    if [ "$elapsed" -ge "$seconds" ]; then
      kill "$pid" >/dev/null 2>&1 || true
      wait "$pid" >/dev/null 2>&1 || true
      return 124
    fi
    sleep 1
    elapsed=$((elapsed + 1))
  done
  wait "$pid"
}

pass() {
  PASS_COUNT=$((PASS_COUNT + 1))
  echo "PASS [$1] $2"
}

fail() {
  FAIL_COUNT=$((FAIL_COUNT + 1))
  echo "FAIL [$1] $2"
}

skip() {
  SKIP_COUNT=$((SKIP_COUNT + 1))
  echo "SKIP [$1] $2"
}

mkdir -p "$ARTIFACT_DIR" "$LOG_DIR"

SUMMARY_FILE="$ARTIFACT_DIR/summary.txt"
RESULTS_FILE="$ARTIFACT_DIR/results.tsv"

cat >"$RESULTS_FILE" <<'EOF'
case_id	program	compile_exit	run_exit	expected_exit	stdout_check	status
EOF

echo "SELFHOST_NATIVE_RUNTIME_PROOF_START"
echo "souc_native=$SOUC_NATIVE"
echo "manifest=$MANIFEST_PATH"
echo "work_dir=$WORK_DIR"
echo "timeout_secs=$TIMEOUT_SECS"
echo "native_target=${SOUNIO_NATIVE_TARGET:-default}"
echo "host_os=$SOUNIO_NATIVE_HOST_OS"
echo "host_machine=$SOUNIO_NATIVE_HOST_MACHINE"
echo "fail_fast=$FAIL_FAST"

if [ ! -x "$SOUC_NATIVE" ]; then
  echo "error: missing self-hosted native compiler at $SOUC_NATIVE" >&2
  exit 1
fi

if [ ! -f "$MANIFEST_PATH" ]; then
  echo "error: missing manifest at $MANIFEST_PATH" >&2
  exit 1
fi

run_case() {
  local case_id="$1"
  local program_path="$2"
  local expected_exit="$3"
  local expected_stdout_path="$4"

  if [ -n "$FILTER" ] && [[ "$case_id" != *"$FILTER"* ]] && [[ "$program_path" != *"$FILTER"* ]]; then
    skip "$case_id" "filtered"
    printf '%s\t%s\t-\t-\t%s\t-\tfiltered\n' \
      "$case_id" "$program_path" "$expected_exit" >>"$RESULTS_FILE"
    return 0
  fi

  local elf_path="$ARTIFACT_DIR/${case_id}.elf"
  local compile_stdout="$LOG_DIR/${case_id}.compile.stdout"
  local compile_stderr="$LOG_DIR/${case_id}.compile.stderr"
  local run_stdout="$LOG_DIR/${case_id}.run.stdout"
  local run_stderr="$LOG_DIR/${case_id}.run.stderr"
  local compile_exit=0
  local run_exit=0
  local stdout_check="skip"

  rm -f "$elf_path" "$compile_stdout" "$compile_stderr" "$run_stdout" "$run_stderr"

  set +e
  if [ -n "$SOUNIO_NATIVE_TARGET" ]; then
    run_with_timeout "$TIMEOUT_SECS" "$SOUC_NATIVE" "$program_path" "$elf_path" --target "$SOUNIO_NATIVE_TARGET" \
      >"$compile_stdout" 2>"$compile_stderr"
  else
    run_with_timeout "$TIMEOUT_SECS" "$SOUC_NATIVE" "$program_path" "$elf_path" \
      >"$compile_stdout" 2>"$compile_stderr"
  fi
  compile_exit=$?
  set -e

  if [ "$compile_exit" -ne 0 ]; then
    fail "$case_id" "compile failed (exit=$compile_exit)"
    printf '%s\t%s\t%s\t-\t%s\t%s\tcompile_fail\n' \
      "$case_id" "$program_path" "$compile_exit" "$expected_exit" "$stdout_check" >>"$RESULTS_FILE"
    sed -n '1,20p' "$compile_stdout" || true
    sed -n '1,20p' "$compile_stderr" || true
    return 0
  fi

  chmod +x "$elf_path"

  set +e
  run_with_timeout "$TIMEOUT_SECS" "$elf_path" >"$run_stdout" 2>"$run_stderr"
  run_exit=$?
  set -e

  if [ "$run_exit" -ne "$expected_exit" ]; then
    fail "$case_id" "unexpected exit (got=$run_exit expected=$expected_exit)"
    printf '%s\t%s\t%s\t%s\t%s\t%s\trun_exit_fail\n' \
      "$case_id" "$program_path" "$compile_exit" "$run_exit" "$expected_exit" "$stdout_check" >>"$RESULTS_FILE"
    sed -n '1,20p' "$run_stdout" || true
    sed -n '1,20p' "$run_stderr" || true
    return 0
  fi

  if [ "$expected_stdout_path" != "-" ]; then
    stdout_check="pending"
    if cmp -s "$run_stdout" "$expected_stdout_path"; then
      stdout_check="ok"
    else
      stdout_check="mismatch"
      fail "$case_id" "stdout mismatch"
      printf '%s\t%s\t%s\t%s\t%s\t%s\tstdout_fail\n' \
        "$case_id" "$program_path" "$compile_exit" "$run_exit" "$expected_exit" "$stdout_check" >>"$RESULTS_FILE"
      echo "--- expected stdout ($expected_stdout_path)"
      cat "$expected_stdout_path" || true
      echo "--- actual stdout ($run_stdout)"
      cat "$run_stdout" || true
      return 0
    fi
  fi

  pass "$case_id" "compile+run ok (exit=$run_exit stdout=$stdout_check)"
  printf '%s\t%s\t%s\t%s\t%s\t%s\tok\n' \
    "$case_id" "$program_path" "$compile_exit" "$run_exit" "$expected_exit" "$stdout_check" >>"$RESULTS_FILE"
}

while IFS=$'\t' read -r case_id program_path expected_exit expected_stdout_path; do
  if [ -z "${case_id:-}" ]; then
    continue
  fi
  if [[ "$case_id" == \#* ]]; then
    continue
  fi

  if [ ! -f "$program_path" ]; then
    fail "$case_id" "missing program $program_path"
    printf '%s\t%s\t-\t-\t%s\t-\tmissing_program\n' \
      "$case_id" "$program_path" "$expected_exit" >>"$RESULTS_FILE"
    continue
  fi

  if [ "$SOUNIO_NATIVE_HOST_OS" = "Darwin" ] &&
     { [ "$SOUNIO_NATIVE_HOST_MACHINE" = "arm64" ] || [ "$SOUNIO_NATIVE_HOST_MACHINE" = "aarch64" ]; } &&
     [ "$case_id" = "epistemic_bridge_chain_42" ] &&
     [ "${SOUNIO_MACOS_EPISTEMIC_BRIDGE_ALLOW:-0}" != "1" ]; then
    skip "$case_id" "macOS arm64 known blocker: compiler segfaults while compiling epistemic bridge chain; Linux coverage remains active"
    printf '%s\t%s\t-\t-\t%s\t-\tmacos_arm64_known_blocker\n' \
      "$case_id" "$program_path" "$expected_exit" >>"$RESULTS_FILE"
    continue
  fi

  if grep -Eq '^//@[[:space:]]*ignore\b' "$program_path"; then
    skip "$case_id" "ignore annotation"
    printf '%s\t%s\t-\t-\t%s\t-\tignore_annotation\n' \
      "$case_id" "$program_path" "$expected_exit" >>"$RESULTS_FILE"
    continue
  fi

  if [ "$expected_stdout_path" != "-" ] && [ ! -f "$expected_stdout_path" ]; then
    fail "$case_id" "missing stdout fixture $expected_stdout_path"
    printf '%s\t%s\t-\t-\t%s\t-\tmissing_stdout_fixture\n' \
      "$case_id" "$program_path" "$expected_exit" >>"$RESULTS_FILE"
    continue
  fi

  run_case "$case_id" "$program_path" "$expected_exit" "$expected_stdout_path"
  if [ "$FAIL_FAST" = "1" ] && [ "$FAIL_COUNT" -ne 0 ]; then
    break
  fi
done <"$MANIFEST_PATH"

{
  echo "summary_pass=$PASS_COUNT"
  echo "summary_fail=$FAIL_COUNT"
  echo "summary_skip=$SKIP_COUNT"
  echo "manifest=$MANIFEST_PATH"
  echo "souc_native=$SOUC_NATIVE"
  echo "native_target=${SOUNIO_NATIVE_TARGET:-default}"
  echo "host_os=$SOUNIO_NATIVE_HOST_OS"
  echo "host_machine=$SOUNIO_NATIVE_HOST_MACHINE"
  echo "fail_fast=$FAIL_FAST"
  echo "results_file=$RESULTS_FILE"
  echo "artifact_dir=$ARTIFACT_DIR"
  echo "log_dir=$LOG_DIR"
} >"$SUMMARY_FILE"

echo "SELFHOST_NATIVE_RUNTIME_PROOF_SUMMARY pass=$PASS_COUNT fail=$FAIL_COUNT skip=$SKIP_COUNT"
echo "results_file=$RESULTS_FILE"

if [ "$FAIL_COUNT" -ne 0 ]; then
  exit 1
fi

exit 0
