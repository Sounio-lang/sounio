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
TARGET="${TARGET:-aarch64-macos}"

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

run_crash_diag() {
  local case_id="$1"
  local program_path="$2"
  local out_path="$3"
  local diag_log="$LOG_DIR/${case_id}.lldb.txt"

  if [ "${SOUNIO_MACOS_CRASH_DIAG:-0}" != "1" ]; then
    return 0
  fi

  if [ "$(uname -s 2>/dev/null || true)" != "Darwin" ]; then
    return 0
  fi

  if ! command -v lldb >/dev/null 2>&1; then
    echo "crash diagnostic skipped: lldb not found" >"$diag_log"
    return 0
  fi

  rm -f "$diag_log"
  lldb --batch \
    -o "target stop-hook add -o 'register read x0 x1 x2 x3 x8 x9 x10 x29 x30 sp pc' -o 'thread backtrace all' -o 'disassemble --pc --count 24'" \
    -o "run" \
    -o "register read x0 x1 x2 x3 x8 x9 x10 x29 sp pc" \
    -o "thread backtrace all" \
    -o "image lookup --address 0x1000319d8" \
    -o "disassemble --pc --count 24" \
    -- "$SOUC_NATIVE" "$program_path" "$out_path" --target "$TARGET" \
    >"$diag_log" 2>&1 </dev/null || true
}

expected_file_kind() {
  case "$1" in
    aarch64-macos) printf '%s\n' "Mach-O 64-bit arm64 executable|Mach-O 64-bit executable arm64" ;;
    x86_64-macos) printf '%s\n' "Mach-O 64-bit x86_64 executable|Mach-O 64-bit executable x86_64" ;;
    aarch64-linux) printf '%s\n' "ELF 64-bit LSB executable, ARM aarch64" ;;
    x86_64-linux) printf '%s\n' "ELF 64-bit LSB executable, x86-64" ;;
    *)
      echo "error: unsupported proof target: $1" >&2
      return 1
      ;;
  esac
}

target_artifact_suffix() {
  case "$1" in
    *-macos) printf '%s\n' "macho" ;;
    *-linux) printf '%s\n' "elf" ;;
    *) printf '%s\n' "bin" ;;
  esac
}

assert_file_kind() {
  local path="$1"
  local expected="$2"
  local actual
  local variant

  actual="$(file "$path" 2>/dev/null || true)"
  IFS='|' read -r -a expected_variants <<<"$expected"
  for variant in "${expected_variants[@]}"; do
    if [[ "$actual" == *"$variant"* ]]; then
      return 0
    fi
  done

  echo "unexpected artifact kind for $path" >&2
  echo "expected fragment: $expected" >&2
  echo "actual: $actual" >&2
  return 1
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
EXPECTED_FILE_KIND="$(expected_file_kind "$TARGET")"
ARTIFACT_SUFFIX="$(target_artifact_suffix "$TARGET")"

cat >"$RESULTS_FILE" <<'EOF'
case_id	program	target	compile_exit	out_size	status
EOF

echo "SELFHOST_AARCH64_COMPILE_PROOF_START"
echo "souc_native=$SOUC_NATIVE"
echo "manifest=$MANIFEST_PATH"
echo "work_dir=$WORK_DIR"
echo "timeout_secs=$TIMEOUT_SECS"
echo "target=$TARGET"

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
    printf '%s\t%s\t%s\t-\t-\tfiltered\n' "$case_id" "$program_path" "$TARGET" >>"$RESULTS_FILE"
    return 0
  fi

  local out_path="$ARTIFACT_DIR/${case_id}.${TARGET}.${ARTIFACT_SUFFIX}"
  local compile_stdout="$LOG_DIR/${case_id}.compile.stdout"
  local compile_stderr="$LOG_DIR/${case_id}.compile.stderr"
  local compile_exit=0
  local out_size=0

  rm -f "$out_path" "$compile_stdout" "$compile_stderr"

  set +e
  run_with_timeout "$TIMEOUT_SECS" "$SOUC_NATIVE" "$program_path" "$out_path" --target "$TARGET" \
    >"$compile_stdout" 2>"$compile_stderr"
  compile_exit=$?
  set -e

  if [ "$compile_exit" -ne 0 ]; then
    fail "$case_id" "compile failed (exit=$compile_exit)"
    printf '%s\t%s\t%s\t%s\t-\tcompile_fail\n' "$case_id" "$program_path" "$TARGET" "$compile_exit" >>"$RESULTS_FILE"
    sed -n '1,20p' "$compile_stdout" || true
    sed -n '1,20p' "$compile_stderr" || true
    if [ "$compile_exit" -eq 139 ]; then
      run_crash_diag "$case_id" "$program_path" "$out_path"
      if [ -f "$LOG_DIR/${case_id}.lldb.txt" ]; then
        sed -n '1,80p' "$LOG_DIR/${case_id}.lldb.txt" || true
      fi
    fi
    return 0
  fi

  if [ ! -f "$out_path" ]; then
    fail "$case_id" "missing output artifact"
    printf '%s\t%s\t%s\t%s\t0\tmissing_output\n' "$case_id" "$program_path" "$TARGET" "$compile_exit" >>"$RESULTS_FILE"
    return 0
  fi

  out_size=$(wc -c <"$out_path")
  if [ "$out_size" -le 0 ]; then
    fail "$case_id" "empty output artifact"
    printf '%s\t%s\t%s\t%s\t%s\tempty_output\n' "$case_id" "$program_path" "$TARGET" "$compile_exit" "$out_size" >>"$RESULTS_FILE"
    return 0
  fi

  if ! assert_file_kind "$out_path" "$EXPECTED_FILE_KIND"; then
    fail "$case_id" "wrong artifact kind"
    printf '%s\t%s\t%s\t%s\t%s\twrong_artifact_kind\n' "$case_id" "$program_path" "$TARGET" "$compile_exit" "$out_size" >>"$RESULTS_FILE"
    return 0
  fi

  pass "$case_id" "compile ok target=$TARGET (size=$out_size)"
  printf '%s\t%s\t%s\t%s\t%s\tok\n' "$case_id" "$program_path" "$TARGET" "$compile_exit" "$out_size" >>"$RESULTS_FILE"
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
    printf '%s\t%s\t%s\t-\t-\tmissing_program\n' "$case_id" "$program_path" "$TARGET" >>"$RESULTS_FILE"
    continue
  fi

  run_case "$case_id" "$program_path"
done <"$MANIFEST_PATH"

{
  echo "summary_pass=$PASS_COUNT"
  echo "summary_fail=$FAIL_COUNT"
  echo "summary_skip=$SKIP_COUNT"
  echo "target=$TARGET"
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
