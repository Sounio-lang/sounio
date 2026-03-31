#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

MANIFEST_PATH="${MANIFEST_PATH:-tests/gpu/sim_runtime/manifest.tsv}"
WORK_DIR="${WORK_DIR:-/tmp/sounio-gpu-sim-runtime-gate}"
LOG_DIR="$WORK_DIR/logs"
TIMEOUT_SECS="${TIMEOUT_SECS:-30}"
FILTER="${FILTER:-}"
SOUC_BIN="${SOUC_BIN:-$ROOT_DIR/bin/souc}"

PASS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0

mkdir -p "$LOG_DIR"

python3 "$ROOT_DIR/scripts/gpu/validate_gpu_capability_manifest.py" \
  sim-runtime "$MANIFEST_PATH" --root "$ROOT_DIR" >/dev/null

if [ ! -x "$SOUC_BIN" ]; then
  echo "error: missing souc runner at $SOUC_BIN" >&2
  exit 1
fi

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

echo "GPU_SIM_RUNTIME_GATE_START"
echo "souc_bin=$SOUC_BIN"
echo "manifest=$MANIFEST_PATH"
echo "work_dir=$WORK_DIR"

while IFS=$'\t' read -r case_id support_class program expected_exit expected_stdout; do
  if [ -z "${case_id:-}" ]; then
    continue
  fi
  if [[ "$case_id" == \#* ]]; then
    continue
  fi
  if [ -n "$FILTER" ] && [[ "$case_id" != *"$FILTER"* ]] && [[ "$program" != *"$FILTER"* ]]; then
    skip "$case_id" "filtered"
    continue
  fi

  log_stdout="$LOG_DIR/${case_id}.stdout"
  log_stderr="$LOG_DIR/${case_id}.stderr"
  rc=0
  rm -f "$log_stdout" "$log_stderr"

  set +e
  SOUNIO_STDLIB_PATH="$ROOT_DIR/stdlib" \
    run_with_timeout "$TIMEOUT_SECS" "$SOUC_BIN" run "$program" \
    >"$log_stdout" 2>"$log_stderr"
  rc=$?
  set -e

  if [ "$rc" -ne "$expected_exit" ]; then
    fail "$case_id" "runtime exit=$rc expected=$expected_exit"
    continue
  fi
  if [ "$expected_stdout" != "-" ] && ! grep -qF "$expected_stdout" "$log_stdout"; then
    fail "$case_id" "missing stdout marker: $expected_stdout"
    continue
  fi
  pass "$case_id" "runtime ok ($support_class)"
done <"$MANIFEST_PATH"

echo "GPU_SIM_RUNTIME_GATE_SUMMARY pass=$PASS_COUNT fail=$FAIL_COUNT skip=$SKIP_COUNT"

if [ "$FAIL_COUNT" -ne 0 ]; then
  exit 1
fi
