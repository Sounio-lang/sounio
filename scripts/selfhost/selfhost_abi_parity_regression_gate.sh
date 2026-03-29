#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SOUC_NATIVE="${SOUC_NATIVE:-$ROOT_DIR/artifacts/self-hosted/souc-self-hosted-x86_64}"
SOUC_RUN="${SOUC_RUN:-$ROOT_DIR/bin/souc}"
MANIFEST_PATH="${MANIFEST_PATH:-tests/selfhost/abi_parity/manifest.tsv}"
WORK_DIR="${WORK_DIR:-/tmp/sounio-selfhost-abi-parity-regression-gate}"
ARTIFACT_DIR="$WORK_DIR/artifacts"
LOG_DIR="$WORK_DIR/logs"
TIMEOUT_SECS="${TIMEOUT_SECS:-60}"
FILTER="${FILTER:-}"

PASS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0

fail_manifest() {
  echo "error: $1" >&2
  exit 1
}

validate_support_class() {
  local case_id="$1"
  local kind="$2"
  local support_class="$3"
  case "$kind" in
    run)
      if [ "$support_class" != "x86-runtime-supported" ]; then
        fail_manifest "case $case_id must use support_class=x86-runtime-supported for kind=run"
      fi
      ;;
    aarch64-ok)
      if [ "$support_class" != "aarch64-compile-proof" ]; then
        fail_manifest "case $case_id must use support_class=aarch64-compile-proof for kind=aarch64-ok"
      fi
      ;;
    aarch64-fail)
      if [ "$support_class" != "aarch64-explicit-unsupported" ]; then
        fail_manifest "case $case_id must use support_class=aarch64-explicit-unsupported for kind=aarch64-fail"
      fi
      ;;
    *)
      fail_manifest "case $case_id has unknown manifest kind: $kind"
      ;;
  esac
}

run_with_timeout() {
  local seconds="$1"
  shift
  if command -v timeout >/dev/null 2>&1; then
    timeout --preserve-status "${seconds}s" "$@"
    return $?
  fi
  "$@"
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

RESULTS_FILE="$ARTIFACT_DIR/results.tsv"
SUMMARY_FILE="$ARTIFACT_DIR/summary.txt"

cat >"$RESULTS_FILE" <<'EOF'
case_id	kind	support_class	program	exit	status
EOF

echo "SELFHOST_ABI_PARITY_REGRESSION_GATE_START"
echo "souc_native=$SOUC_NATIVE"
echo "souc_run=$SOUC_RUN"
echo "manifest=$MANIFEST_PATH"
echo "work_dir=$WORK_DIR"
echo "timeout_secs=$TIMEOUT_SECS"

if [ ! -x "$SOUC_NATIVE" ]; then
  echo "error: missing self-hosted native compiler at $SOUC_NATIVE" >&2
  exit 1
fi

if [ ! -x "$SOUC_RUN" ]; then
  echo "error: missing souc driver at $SOUC_RUN" >&2
  exit 1
fi

run_case() {
  local case_id="$1"
  local kind="$2"
  local support_class="$3"
  local program_path="$4"
  local expected="$5"

  if [ -n "$FILTER" ] && [[ "$case_id" != *"$FILTER"* ]] && [[ "$program_path" != *"$FILTER"* ]]; then
    skip "$case_id" "filtered"
    printf '%s\t%s\t%s\t%s\t-\tfiltered\n' "$case_id" "$kind" "$support_class" "$program_path" >>"$RESULTS_FILE"
    return 0
  fi

  local stdout_path="$LOG_DIR/${case_id}.stdout"
  local stderr_path="$LOG_DIR/${case_id}.stderr"
  local exit_code=0
  rm -f "$stdout_path" "$stderr_path"

  if [ "$kind" = "run" ]; then
    set +e
    run_with_timeout "$TIMEOUT_SECS" env SOUC_NATIVE_BIN="$SOUC_NATIVE" "$SOUC_RUN" run "$program_path" >"$stdout_path" 2>"$stderr_path"
    exit_code=$?
    set -e
    if [ "$exit_code" -ne 0 ]; then
      fail "$case_id" "run failed (exit=$exit_code)"
      printf '%s\t%s\t%s\t%s\t%s\trun_fail\n' "$case_id" "$kind" "$support_class" "$program_path" "$exit_code" >>"$RESULTS_FILE"
      return 0
    fi
    if [ "$expected" != "-" ] && ! grep -qF "$expected" "$stdout_path"; then
      fail "$case_id" "missing stdout marker"
      printf '%s\t%s\t%s\t%s\t%s\tstdout_fail\n' "$case_id" "$kind" "$support_class" "$program_path" "$exit_code" >>"$RESULTS_FILE"
      return 0
    fi
    pass "$case_id" "run ok"
    printf '%s\t%s\t%s\t%s\t%s\tok\n' "$case_id" "$kind" "$support_class" "$program_path" "$exit_code" >>"$RESULTS_FILE"
    return 0
  fi

  local out_path="$ARTIFACT_DIR/${case_id}.aarch64.elf"
  rm -f "$out_path"

  if [ "$kind" = "aarch64-ok" ]; then
    set +e
    run_with_timeout "$TIMEOUT_SECS" "$SOUC_NATIVE" "$program_path" "$out_path" --target aarch64-linux >"$stdout_path" 2>"$stderr_path"
    exit_code=$?
    set -e
    if [ "$exit_code" -ne 0 ]; then
      fail "$case_id" "aarch64 compile failed (exit=$exit_code)"
      printf '%s\t%s\t%s\t%s\t%s\tcompile_fail\n' "$case_id" "$kind" "$support_class" "$program_path" "$exit_code" >>"$RESULTS_FILE"
      return 0
    fi
    if [ ! -s "$out_path" ]; then
      fail "$case_id" "aarch64 output missing"
      printf '%s\t%s\t%s\t%s\t%s\tmissing_output\n' "$case_id" "$kind" "$support_class" "$program_path" "$exit_code" >>"$RESULTS_FILE"
      return 0
    fi
    pass "$case_id" "aarch64 compile ok"
    printf '%s\t%s\t%s\t%s\t%s\tok\n' "$case_id" "$kind" "$support_class" "$program_path" "$exit_code" >>"$RESULTS_FILE"
    return 0
  fi

  if [ "$kind" = "aarch64-fail" ]; then
    set +e
    run_with_timeout "$TIMEOUT_SECS" "$SOUC_NATIVE" "$program_path" "$out_path" --target aarch64-linux >"$stdout_path" 2>"$stderr_path"
    exit_code=$?
    set -e
    cat "$stdout_path" "$stderr_path" >"$LOG_DIR/${case_id}.combined.log"
    if [ "$exit_code" -eq 0 ]; then
      fail "$case_id" "expected aarch64 compile failure but succeeded"
      printf '%s\t%s\t%s\t%s\t%s\tunexpected_success\n' "$case_id" "$kind" "$support_class" "$program_path" "$exit_code" >>"$RESULTS_FILE"
      return 0
    fi
    if [ "$expected" != "-" ] && ! grep -qF "$expected" "$LOG_DIR/${case_id}.combined.log"; then
      fail "$case_id" "missing expected diagnostic"
      printf '%s\t%s\t%s\t%s\t%s\tmissing_pattern\n' "$case_id" "$kind" "$support_class" "$program_path" "$exit_code" >>"$RESULTS_FILE"
      return 0
    fi
    pass "$case_id" "aarch64 compile rejected as expected"
    printf '%s\t%s\t%s\t%s\t%s\tok\n' "$case_id" "$kind" "$support_class" "$program_path" "$exit_code" >>"$RESULTS_FILE"
    return 0
  fi

  fail "$case_id" "unknown manifest kind: $kind"
  printf '%s\t%s\t%s\t%s\t-\tunknown_kind\n' "$case_id" "$kind" "$support_class" "$program_path" >>"$RESULTS_FILE"
  return 0
}

while IFS=$'\t' read -r case_id kind col3 col4 col5; do
  if [ -z "${case_id:-}" ]; then
    continue
  fi
  if [[ "$case_id" == \#* ]]; then
    continue
  fi

  local_support_class="${col3:-}"
  local_program_path="${col4:-}"
  local_expected="${col5:--}"

  if [ -z "$local_support_class" ] || [ -z "$local_program_path" ]; then
    fail_manifest "case $case_id must provide explicit support_class and program columns"
  fi

  validate_support_class "$case_id" "$kind" "$local_support_class"

  if [ ! -f "$local_program_path" ]; then
    fail "$case_id" "missing program $local_program_path"
    printf '%s\t%s\t%s\t%s\t-\tmissing_program\n' "$case_id" "$kind" "$local_support_class" "$local_program_path" >>"$RESULTS_FILE"
    continue
  fi
  run_case "$case_id" "$kind" "$local_support_class" "$local_program_path" "${local_expected:--}"
done <"$MANIFEST_PATH"

{
  echo "summary_pass=$PASS_COUNT"
  echo "summary_fail=$FAIL_COUNT"
  echo "summary_skip=$SKIP_COUNT"
  echo "manifest=$MANIFEST_PATH"
  echo "souc_native=$SOUC_NATIVE"
  echo "souc_run=$SOUC_RUN"
  echo "results_file=$RESULTS_FILE"
  echo "artifact_dir=$ARTIFACT_DIR"
  echo "log_dir=$LOG_DIR"
} >"$SUMMARY_FILE"

echo "SELFHOST_ABI_PARITY_REGRESSION_GATE_SUMMARY pass=$PASS_COUNT fail=$FAIL_COUNT skip=$SKIP_COUNT"
echo "results_file=$RESULTS_FILE"

if [ "$FAIL_COUNT" -ne 0 ]; then
  exit 1
fi

exit 0
