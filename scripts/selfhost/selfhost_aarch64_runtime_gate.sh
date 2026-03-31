#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SOUC_NATIVE="${SOUC_NATIVE:-$ROOT_DIR/artifacts/self-hosted/souc-self-hosted-x86_64}"
MANIFEST_PATH="${MANIFEST_PATH:-tests/selfhost/aarch64_runtime/manifest.tsv}"
WORK_DIR="${WORK_DIR:-/tmp/sounio-selfhost-aarch64-runtime-gate}"
ARTIFACT_DIR="$WORK_DIR/artifacts"
LOG_DIR="$WORK_DIR/logs"
TOOL_DIR="$WORK_DIR/tooling"
TIMEOUT_SECS="${TIMEOUT_SECS:-30}"
FILTER="${FILTER:-}"

PASS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0
RUNNER_PATH=""
RUNNER_SOURCE=""

fail_manifest() {
  echo "error: $1" >&2
  exit 1
}

validate_support_class() {
  local case_id="$1"
  local support_class="$2"
  if [ "$support_class" != "aarch64-runtime-supported" ]; then
    fail_manifest "case $case_id must use support_class=aarch64-runtime-supported"
  fi
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

provision_runner() {
  local provision_dir="$TOOL_DIR/qemu-user-static"
  local runner="$provision_dir/root/usr/bin/qemu-aarch64-static"
  mkdir -p "$provision_dir"
  if [ -x "$runner" ]; then
    RUNNER_SOURCE="provisioned-cache"
    RUNNER_PATH="$runner"
    return 0
  fi
  if ! command -v apt-get >/dev/null 2>&1; then
    echo "error: qemu-aarch64 runner not found and apt-get is unavailable for repo-local provisioning" >&2
    return 1
  fi
  if ! command -v dpkg-deb >/dev/null 2>&1; then
    echo "error: qemu-aarch64 runner not found and dpkg-deb is unavailable for repo-local provisioning" >&2
    return 1
  fi
  rm -f "$provision_dir"/qemu-user-static_*.deb
  (
    cd "$provision_dir"
    apt-get download qemu-user-static >/dev/null
  )
  local deb_path
  deb_path="$(find "$provision_dir" -maxdepth 1 -name 'qemu-user-static_*.deb' | sort | head -n1)"
  if [ -z "$deb_path" ]; then
    echo "error: qemu-user-static download did not produce a .deb" >&2
    return 1
  fi
  rm -rf "$provision_dir/root"
  dpkg-deb -x "$deb_path" "$provision_dir/root" >/dev/null
  if [ ! -x "$runner" ]; then
    echo "error: provisioned qemu-aarch64-static runner missing at $runner" >&2
    return 1
  fi
  RUNNER_SOURCE="provisioned"
  RUNNER_PATH="$runner"
}

resolve_runner() {
  if [ -n "${AARCH64_RUNNER:-}" ]; then
    if [ ! -x "$AARCH64_RUNNER" ]; then
      echo "error: AARCH64_RUNNER is not executable: $AARCH64_RUNNER" >&2
      return 1
    fi
    RUNNER_SOURCE="env"
    RUNNER_PATH="$AARCH64_RUNNER"
    return 0
  fi
  if command -v qemu-aarch64-static >/dev/null 2>&1; then
    RUNNER_SOURCE="path"
    RUNNER_PATH="$(command -v qemu-aarch64-static)"
    return 0
  fi
  if command -v qemu-aarch64 >/dev/null 2>&1; then
    RUNNER_SOURCE="path"
    RUNNER_PATH="$(command -v qemu-aarch64)"
    return 0
  fi
  provision_runner
}

mkdir -p "$ARTIFACT_DIR" "$LOG_DIR" "$TOOL_DIR"

RESULTS_FILE="$ARTIFACT_DIR/results.tsv"
SUMMARY_FILE="$ARTIFACT_DIR/summary.txt"

cat >"$RESULTS_FILE" <<'EOF'
case_id	support_class	program	expected_exit	compile_exit	runtime_exit	status
EOF

echo "SELFHOST_AARCH64_RUNTIME_GATE_START"
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

resolve_runner
echo "aarch64_runner=$RUNNER_PATH"
echo "runner_source=$RUNNER_SOURCE"

run_case() {
  local case_id="$1"
  local support_class="$2"
  local program_path="$3"
  local expected_exit="$4"
  local expected_stdout="$5"

  if [ -n "$FILTER" ] && [[ "$case_id" != *"$FILTER"* ]] && [[ "$program_path" != *"$FILTER"* ]]; then
    skip "$case_id" "filtered"
    printf '%s\t%s\t%s\t%s\t-\t-\tfiltered\n' "$case_id" "$support_class" "$program_path" "$expected_exit" >>"$RESULTS_FILE"
    return 0
  fi

  local elf_path="$ARTIFACT_DIR/${case_id}.aarch64.elf"
  local compile_stdout="$LOG_DIR/${case_id}.compile.stdout"
  local compile_stderr="$LOG_DIR/${case_id}.compile.stderr"
  local runtime_stdout="$LOG_DIR/${case_id}.runtime.stdout"
  local runtime_stderr="$LOG_DIR/${case_id}.runtime.stderr"
  local compile_exit=0
  local runtime_exit=0

  rm -f "$elf_path" "$compile_stdout" "$compile_stderr" "$runtime_stdout" "$runtime_stderr"

  set +e
  run_with_timeout "$TIMEOUT_SECS" "$SOUC_NATIVE" "$program_path" "$elf_path" --target aarch64-linux \
    >"$compile_stdout" 2>"$compile_stderr"
  compile_exit=$?
  set -e

  if [ "$compile_exit" -ne 0 ]; then
    fail "$case_id" "compile failed (exit=$compile_exit)"
    printf '%s\t%s\t%s\t%s\t%s\t-\tcompile_fail\n' "$case_id" "$support_class" "$program_path" "$expected_exit" "$compile_exit" >>"$RESULTS_FILE"
    return 0
  fi

  if [ ! -s "$elf_path" ]; then
    fail "$case_id" "missing or empty AArch64 artifact"
    printf '%s\t%s\t%s\t%s\t%s\t-\tmissing_output\n' "$case_id" "$support_class" "$program_path" "$expected_exit" "$compile_exit" >>"$RESULTS_FILE"
    return 0
  fi
  chmod +x "$elf_path"

  set +e
  run_with_timeout "$TIMEOUT_SECS" "$RUNNER_PATH" "$elf_path" >"$runtime_stdout" 2>"$runtime_stderr"
  runtime_exit=$?
  set -e

  if [ "$runtime_exit" -ne "$expected_exit" ]; then
    fail "$case_id" "runtime exit=$runtime_exit expected=$expected_exit"
    printf '%s\t%s\t%s\t%s\t%s\t%s\truntime_exit_mismatch\n' "$case_id" "$support_class" "$program_path" "$expected_exit" "$compile_exit" "$runtime_exit" >>"$RESULTS_FILE"
    return 0
  fi

  if [ "$expected_stdout" != "-" ] && ! grep -qF "$expected_stdout" "$runtime_stdout"; then
    fail "$case_id" "missing stdout marker"
    printf '%s\t%s\t%s\t%s\t%s\t%s\tstdout_fail\n' "$case_id" "$support_class" "$program_path" "$expected_exit" "$compile_exit" "$runtime_exit" >>"$RESULTS_FILE"
    return 0
  fi

  pass "$case_id" "runtime ok"
  printf '%s\t%s\t%s\t%s\t%s\t%s\tok\n' "$case_id" "$support_class" "$program_path" "$expected_exit" "$compile_exit" "$runtime_exit" >>"$RESULTS_FILE"
}

while IFS=$'\t' read -r case_id col2 col3 col4 col5; do
  if [ -z "${case_id:-}" ]; then
    continue
  fi
  if [[ "$case_id" == \#* ]]; then
    continue
  fi

  local_support_class="${col2:-}"
  local_program_path="${col3:-}"
  local_expected_exit="${col4:-0}"
  local_expected_stdout="${col5:--}"

  if [ -z "$local_support_class" ] || [ -z "$local_program_path" ]; then
    fail_manifest "case $case_id must provide explicit support_class and program columns"
  fi

  validate_support_class "$case_id" "$local_support_class"

  if [ ! -f "$local_program_path" ]; then
    fail "$case_id" "missing program $local_program_path"
    printf '%s\t%s\t%s\t%s\t-\t-\tmissing_program\n' "$case_id" "$local_support_class" "$local_program_path" "$local_expected_exit" >>"$RESULTS_FILE"
    continue
  fi

  run_case "$case_id" "$local_support_class" "$local_program_path" "$local_expected_exit" "${local_expected_stdout:--}"
done <"$MANIFEST_PATH"

{
  echo "summary_pass=$PASS_COUNT"
  echo "summary_fail=$FAIL_COUNT"
  echo "summary_skip=$SKIP_COUNT"
  echo "manifest=$MANIFEST_PATH"
  echo "souc_native=$SOUC_NATIVE"
  echo "runner_path=$RUNNER_PATH"
  echo "runner_source=$RUNNER_SOURCE"
  echo "results_file=$RESULTS_FILE"
  echo "artifact_dir=$ARTIFACT_DIR"
  echo "log_dir=$LOG_DIR"
} >"$SUMMARY_FILE"

echo "SELFHOST_AARCH64_RUNTIME_GATE_SUMMARY pass=$PASS_COUNT fail=$FAIL_COUNT skip=$SKIP_COUNT"
echo "results_file=$RESULTS_FILE"

if [ "$FAIL_COUNT" -ne 0 ]; then
  exit 1
fi

exit 0
