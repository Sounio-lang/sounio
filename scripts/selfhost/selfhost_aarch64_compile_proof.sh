#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SOUC_NATIVE="${SOUC_NATIVE:-$ROOT_DIR/artifacts/self-hosted/souc-self-hosted-x86_64}"
MANIFEST_PATH="${MANIFEST_PATH:-tests/selfhost/aarch64_compile/manifest.tsv}"
WORK_DIR="${WORK_DIR:-/tmp/sounio-selfhost-aarch64-compile-proof}"
ARTIFACT_DIR="$WORK_DIR/artifacts"
LOG_DIR="$WORK_DIR/logs"
TIMEOUT_SECS="${TIMEOUT_SECS:-30}"
FILTER="${FILTER:-}"

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
case_id	support_class	program	expected_status	compile_exit	out_size	status
EOF

echo "SELFHOST_AARCH64_COMPILE_PROOF_START"
echo "souc_native=$SOUC_NATIVE"
echo "manifest=$MANIFEST_PATH"
echo "work_dir=$WORK_DIR"
echo "timeout_secs=$TIMEOUT_SECS"

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
  local support_class="$2"
  local program_path="$3"
  local expected_status="$4"
  local expected_pattern="$5"

  if [ -n "$FILTER" ] && [[ "$case_id" != *"$FILTER"* ]] && [[ "$program_path" != *"$FILTER"* ]]; then
    skip "$case_id" "filtered"
    printf '%s\t%s\t%s\t%s\t-\t-\tfiltered\n' "$case_id" "$support_class" "$program_path" "$expected_status" >>"$RESULTS_FILE"
    return 0
  fi

  local elf_path="$ARTIFACT_DIR/${case_id}.aarch64.elf"
  local compile_stdout="$LOG_DIR/${case_id}.compile.stdout"
  local compile_stderr="$LOG_DIR/${case_id}.compile.stderr"
  local combined_log="$LOG_DIR/${case_id}.combined.log"
  local compile_exit=0
  local out_size=0

  rm -f "$elf_path" "$compile_stdout" "$compile_stderr" "$combined_log"

  set +e
  run_with_timeout "$TIMEOUT_SECS" "$SOUC_NATIVE" "$program_path" "$elf_path" --target aarch64-linux \
    >"$compile_stdout" 2>"$compile_stderr"
  compile_exit=$?
  set -e

  cat "$compile_stdout" "$compile_stderr" >"$combined_log"

  if [ "$expected_status" = "fail" ]; then
    if [ "$compile_exit" -eq 0 ]; then
      fail "$case_id" "expected compile failure but compilation succeeded"
      printf '%s\t%s\t%s\t%s\t%s\t-\tunexpected_success\n' "$case_id" "$support_class" "$program_path" "$expected_status" "$compile_exit" >>"$RESULTS_FILE"
      return 0
    fi
    if [ "$expected_pattern" != "-" ] && ! grep -qF "$expected_pattern" "$combined_log"; then
      fail "$case_id" "missing expected diagnostic"
      printf '%s\t%s\t%s\t%s\t%s\t-\tmissing_pattern\n' "$case_id" "$support_class" "$program_path" "$expected_status" "$compile_exit" >>"$RESULTS_FILE"
      echo "--- expected pattern"
      echo "$expected_pattern"
      echo "--- compiler output"
      cat "$combined_log" || true
      return 0
    fi
    pass "$case_id" "compile rejected as expected"
    printf '%s\t%s\t%s\t%s\t%s\t-\tok\n' "$case_id" "$support_class" "$program_path" "$expected_status" "$compile_exit" >>"$RESULTS_FILE"
    return 0
  fi

  if [ "$compile_exit" -ne 0 ]; then
    fail "$case_id" "compile failed (exit=$compile_exit)"
    printf '%s\t%s\t%s\t%s\t%s\t-\tcompile_fail\n' "$case_id" "$support_class" "$program_path" "$expected_status" "$compile_exit" >>"$RESULTS_FILE"
    sed -n '1,20p' "$compile_stdout" || true
    sed -n '1,20p' "$compile_stderr" || true
    return 0
  fi

  if [ ! -f "$elf_path" ]; then
    fail "$case_id" "missing output artifact"
    printf '%s\t%s\t%s\t%s\t%s\t0\tmissing_output\n' "$case_id" "$support_class" "$program_path" "$expected_status" "$compile_exit" >>"$RESULTS_FILE"
    return 0
  fi

  out_size=$(wc -c <"$elf_path")
  if [ "$out_size" -le 0 ]; then
    fail "$case_id" "empty output artifact"
    printf '%s\t%s\t%s\t%s\t%s\t%s\tempty_output\n' "$case_id" "$support_class" "$program_path" "$expected_status" "$compile_exit" "$out_size" >>"$RESULTS_FILE"
    return 0
  fi

  pass "$case_id" "compile ok (size=$out_size)"
  printf '%s\t%s\t%s\t%s\t%s\t%s\tok\n' "$case_id" "$support_class" "$program_path" "$expected_status" "$compile_exit" "$out_size" >>"$RESULTS_FILE"
}

while IFS=$'\t' read -r case_id col2 col3 col4 col5; do
  if [ -z "${case_id:-}" ]; then
    continue
  fi
  if [[ "$case_id" == \#* ]]; then
    continue
  fi

  local_support_class=""
  local_program_path=""
  local_expected_status=""
  local_expected_pattern=""

  if [[ "${col2:-}" == *.sio ]] || [[ "${col2:-}" == tests/* ]]; then
    local_program_path="${col2:-}"
    local_expected_status="${col3:-ok}"
    local_expected_pattern="${col4:--}"
    if [ "$local_expected_status" = "fail" ]; then
      local_support_class="aarch64-explicit-unsupported"
    else
      local_support_class="aarch64-compile-proof"
    fi
  else
    local_support_class="${col2:-unknown}"
    local_program_path="${col3:-}"
    local_expected_status="${col4:-ok}"
    local_expected_pattern="${col5:--}"
  fi

  if [ ! -f "$local_program_path" ]; then
    fail "$case_id" "missing program $local_program_path"
    printf '%s\t%s\t%s\t%s\t-\t-\tmissing_program\n' "$case_id" "$local_support_class" "$local_program_path" "${local_expected_status:-ok}" >>"$RESULTS_FILE"
    continue
  fi

  if [ -z "${local_expected_status:-}" ]; then
    local_expected_status="ok"
  fi
  if [ -z "${local_expected_pattern:-}" ]; then
    local_expected_pattern="-"
  fi

  run_case "$case_id" "$local_support_class" "$local_program_path" "$local_expected_status" "$local_expected_pattern"
done <"$MANIFEST_PATH"

{
  echo "summary_pass=$PASS_COUNT"
  echo "summary_fail=$FAIL_COUNT"
  echo "summary_skip=$SKIP_COUNT"
  echo "manifest=$MANIFEST_PATH"
  echo "souc_native=$SOUC_NATIVE"
  echo "results_file=$RESULTS_FILE"
  echo "artifact_dir=$ARTIFACT_DIR"
  echo "log_dir=$LOG_DIR"
} >"$SUMMARY_FILE"

echo "SELFHOST_AARCH64_COMPILE_PROOF_SUMMARY pass=$PASS_COUNT fail=$FAIL_COUNT skip=$SKIP_COUNT"
echo "results_file=$RESULTS_FILE"

if [ "$FAIL_COUNT" -ne 0 ]; then
  exit 1
fi

exit 0
