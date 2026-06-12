#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SOUC_NATIVE="${SOUC_NATIVE:-$ROOT_DIR/artifacts/self-hosted/souc-lean-frontend.elf}"
MANIFEST_PATH="${MANIFEST_PATH:-tests/selfhost/native_typecheck/manifest.tsv}"
WORK_DIR="${WORK_DIR:-/tmp/sounio-selfhost-native-typecheck-proof}"
ARTIFACT_DIR="$WORK_DIR/artifacts"
LOG_DIR="$WORK_DIR/logs"
TIMEOUT_SECS="${TIMEOUT_SECS:-30}"
FILTER="${FILTER:-}"
FAIL_FAST="${FAIL_FAST:-${SOUNIO_NATIVE_FAIL_FAST:-0}}"
SOUNIO_NATIVE_TARGET="${SOUNIO_NATIVE_TARGET:-}"

if [ -z "$SOUNIO_NATIVE_TARGET" ] && [ "$(uname -s)" = "Darwin" ]; then
  case "$(uname -m)" in
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
case_id	program	compile_exit	expected_pattern	status
EOF

echo "SELFHOST_NATIVE_TYPECHECK_PROOF_START"
echo "souc_native=$SOUC_NATIVE"
echo "manifest=$MANIFEST_PATH"
echo "work_dir=$WORK_DIR"
echo "timeout_secs=$TIMEOUT_SECS"
echo "native_target=${SOUNIO_NATIVE_TARGET:-default}"
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
  local expected_pattern="$3"

  if [ -n "$FILTER" ] && [[ "$case_id" != *"$FILTER"* ]] && [[ "$program_path" != *"$FILTER"* ]]; then
    skip "$case_id" "filtered"
    printf '%s\t%s\t-\t%s\tfiltered\n' \
      "$case_id" "$program_path" "$expected_pattern" >>"$RESULTS_FILE"
    return 0
  fi

  local elf_path="$ARTIFACT_DIR/${case_id}.elf"
  local compile_stdout="$LOG_DIR/${case_id}.compile.stdout"
  local compile_stderr="$LOG_DIR/${case_id}.compile.stderr"
  local combined_log="$LOG_DIR/${case_id}.combined.log"
  local compile_exit=0

  rm -f "$elf_path" "$compile_stdout" "$compile_stderr" "$combined_log"

  set +e
  run_with_timeout "$TIMEOUT_SECS" "$SOUC_NATIVE" --check "$program_path" \
    >"$compile_stdout" 2>"$compile_stderr"
  compile_exit=$?
  set -e

  cat "$compile_stdout" "$compile_stderr" >"$combined_log"

  if grep -qF "$expected_pattern" "$combined_log"; then
    if [ "$compile_exit" -eq 0 ]; then
      pass "$case_id" "emitted expected diagnostic"
      printf '%s\t%s\t%s\t%s\tdiagnostic_only\n' \
        "$case_id" "$program_path" "$compile_exit" "$expected_pattern" >>"$RESULTS_FILE"
    else
      pass "$case_id" "rejected with expected diagnostic"
      printf '%s\t%s\t%s\t%s\tok\n' \
        "$case_id" "$program_path" "$compile_exit" "$expected_pattern" >>"$RESULTS_FILE"
    fi
    return 0
  fi

  if [ "$compile_exit" -eq 0 ]; then
    fail "$case_id" "expected diagnostic but compilation succeeded silently"
    printf '%s\t%s\t%s\t%s\tunexpected_success\n' \
      "$case_id" "$program_path" "$compile_exit" "$expected_pattern" >>"$RESULTS_FILE"
    return 0
  fi

  fail "$case_id" "missing expected diagnostic"
  printf '%s\t%s\t%s\t%s\tmissing_pattern\n' \
    "$case_id" "$program_path" "$compile_exit" "$expected_pattern" >>"$RESULTS_FILE"
  echo "--- expected pattern"
  echo "$expected_pattern"
  echo "--- compiler output"
  cat "$combined_log" || true
  return 0
}

while IFS=$'\t' read -r case_id program_path expected_pattern; do
  if [ -z "${case_id:-}" ]; then
    continue
  fi
  if [[ "$case_id" == \#* ]]; then
    continue
  fi

  if [ ! -f "$program_path" ]; then
    fail "$case_id" "missing program $program_path"
    printf '%s\t%s\t-\t%s\tmissing_program\n' \
      "$case_id" "$program_path" "$expected_pattern" >>"$RESULTS_FILE"
    continue
  fi

  run_case "$case_id" "$program_path" "$expected_pattern"
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
  echo "fail_fast=$FAIL_FAST"
  echo "results_file=$RESULTS_FILE"
  echo "artifact_dir=$ARTIFACT_DIR"
  echo "log_dir=$LOG_DIR"
} >"$SUMMARY_FILE"

echo "SELFHOST_NATIVE_TYPECHECK_PROOF_SUMMARY pass=$PASS_COUNT fail=$FAIL_COUNT skip=$SKIP_COUNT"
echo "results_file=$RESULTS_FILE"

if [ "$FAIL_COUNT" -ne 0 ]; then
  exit 1
fi

exit 0
