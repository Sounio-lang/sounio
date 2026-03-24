#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SOUC_NATIVE="${SOUC_NATIVE:-/tmp/gen2.elf}"
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
case_id	program	compile_exit	out_size	status
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
  local program_path="$2"

  if [ -n "$FILTER" ] && [[ "$case_id" != *"$FILTER"* ]] && [[ "$program_path" != *"$FILTER"* ]]; then
    skip "$case_id" "filtered"
    printf '%s\t%s\t-\t-\tfiltered\n' "$case_id" "$program_path" >>"$RESULTS_FILE"
    return 0
  fi

  local elf_path="$ARTIFACT_DIR/${case_id}.aarch64.elf"
  local compile_stdout="$LOG_DIR/${case_id}.compile.stdout"
  local compile_stderr="$LOG_DIR/${case_id}.compile.stderr"
  local compile_exit=0
  local out_size=0

  rm -f "$elf_path" "$compile_stdout" "$compile_stderr"

  set +e
  run_with_timeout "$TIMEOUT_SECS" "$SOUC_NATIVE" "$program_path" "$elf_path" \
    >"$compile_stdout" 2>"$compile_stderr"
  compile_exit=$?
  set -e

  if [ "$compile_exit" -ne 0 ]; then
    fail "$case_id" "compile failed (exit=$compile_exit)"
    printf '%s\t%s\t%s\t-\tcompile_fail\n' "$case_id" "$program_path" "$compile_exit" >>"$RESULTS_FILE"
    sed -n '1,20p' "$compile_stdout" || true
    sed -n '1,20p' "$compile_stderr" || true
    return 0
  fi

  if [ ! -f "$elf_path" ]; then
    fail "$case_id" "missing output artifact"
    printf '%s\t%s\t%s\t0\tmissing_output\n' "$case_id" "$program_path" "$compile_exit" >>"$RESULTS_FILE"
    return 0
  fi

  out_size=$(wc -c <"$elf_path")
  if [ "$out_size" -le 0 ]; then
    fail "$case_id" "empty output artifact"
    printf '%s\t%s\t%s\t%s\tempty_output\n' "$case_id" "$program_path" "$compile_exit" "$out_size" >>"$RESULTS_FILE"
    return 0
  fi

  pass "$case_id" "compile ok (size=$out_size)"
  printf '%s\t%s\t%s\t%s\tok\n' "$case_id" "$program_path" "$compile_exit" "$out_size" >>"$RESULTS_FILE"
}

while IFS=$'\t' read -r case_id program_path; do
  if [ -z "${case_id:-}" ]; then
    continue
  fi
  if [[ "$case_id" == \#* ]]; then
    continue
  fi

  if [ ! -f "$program_path" ]; then
    fail "$case_id" "missing program $program_path"
    printf '%s\t%s\t-\t-\tmissing_program\n' "$case_id" "$program_path" >>"$RESULTS_FILE"
    continue
  fi

  run_case "$case_id" "$program_path"
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
