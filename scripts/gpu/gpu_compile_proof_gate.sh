#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

MANIFEST_PATH="${MANIFEST_PATH:-tests/gpu/compile_proof/manifest.tsv}"
WORK_DIR="${WORK_DIR:-/tmp/sounio-gpu-compile-proof-gate}"
LOG_DIR="$WORK_DIR/logs"
TIMEOUT_SECS="${TIMEOUT_SECS:-30}"
FILTER="${FILTER:-}"
SOUC_NATIVE="${SOUC_NATIVE:-$ROOT_DIR/artifacts/self-hosted/souc-self-hosted-x86_64}"
SOUC_WRAPPER="${SOUC_WRAPPER:-$ROOT_DIR/bin/souc}"

PASS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0

mkdir -p "$LOG_DIR"

python3 "$ROOT_DIR/scripts/gpu/validate_gpu_capability_manifest.py" \
  compile-proof "$MANIFEST_PATH" --root "$ROOT_DIR" >/dev/null

if [ ! -x "$SOUC_NATIVE" ]; then
  echo "error: missing self-hosted compiler at $SOUC_NATIVE" >&2
  exit 1
fi
if [ ! -x "$SOUC_WRAPPER" ]; then
  echo "error: missing wrapper at $SOUC_WRAPPER" >&2
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

echo "GPU_COMPILE_PROOF_GATE_START"
echo "souc_native=$SOUC_NATIVE"
echo "manifest=$MANIFEST_PATH"
echo "work_dir=$WORK_DIR"

while IFS=$'\t' read -r case_id support_class mode program expected_marker; do
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
  SOUC_NATIVE_BIN="$SOUC_NATIVE" SOUNIO_STDLIB_PATH="$ROOT_DIR/stdlib" \
    run_with_timeout "$TIMEOUT_SECS" "$SOUC_WRAPPER" check "$program" \
    >"$log_stdout" 2>"$log_stderr"
  rc=$?
  set -e

  case "$mode" in
    check-pass)
      if [ "$rc" -ne 0 ]; then
        fail "$case_id" "check failed (support_class=$support_class exit=$rc)"
        continue
      fi
      pass "$case_id" "compile-proof ok ($support_class)"
      ;;
    check-fail)
      if [ "$rc" -eq 0 ]; then
        fail "$case_id" "expected check failure but passed"
        continue
      fi
      if [ "$expected_marker" != "-" ] && ! grep -qiF "$expected_marker" "$log_stdout" "$log_stderr"; then
        fail "$case_id" "missing failure marker: $expected_marker"
        continue
      fi
      pass "$case_id" "expected rejection observed ($support_class)"
      ;;
    *)
      fail "$case_id" "unsupported mode $mode"
      ;;
  esac
done <"$MANIFEST_PATH"

echo "GPU_COMPILE_PROOF_GATE_SUMMARY pass=$PASS_COUNT fail=$FAIL_COUNT skip=$SKIP_COUNT"

if [ "$FAIL_COUNT" -ne 0 ]; then
  exit 1
fi
