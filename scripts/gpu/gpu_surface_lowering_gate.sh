#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
source "$ROOT_DIR/scripts/lib/resolve_souc.sh"

MANIFEST_PATH="${MANIFEST_PATH:-tests/gpu/surface_lowering/manifest.tsv}"
WORK_DIR="${WORK_DIR:-/tmp/sounio-gpu-surface-lowering-gate}"
ARTIFACT_DIR="$WORK_DIR/artifacts"
LOG_DIR="$WORK_DIR/logs"
TIMEOUT_SECS="${TIMEOUT_SECS:-30}"
FILTER="${FILTER:-}"
GPU_SOUC_HINT="${GPU_SOUC_HINT:-${SOUNIO_GPU_SOUC_BIN:-}}"

PASS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0

mkdir -p "$ARTIFACT_DIR" "$LOG_DIR"

python3 "$ROOT_DIR/scripts/gpu/validate_gpu_capability_manifest.py" \
  surface-lowering "$MANIFEST_PATH" --root "$ROOT_DIR" >/dev/null

if ! GPU_SOUC="$(sounio_resolve_gpu_souc "$ROOT_DIR/scripts/fixtures/gpu_minimal.sio" "$GPU_SOUC_HINT" 2>/dev/null)"; then
  reason="$(sounio_gpu_probe_reason)"
  if [ -z "$reason" ]; then
    reason="gpu_backend_unavailable"
  fi
  echo "error: unable to resolve checked GPU compiler ($reason)" >&2
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

echo "GPU_SURFACE_LOWERING_GATE_START"
echo "gpu_souc=$GPU_SOUC"
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
  out_path="$ARTIFACT_DIR/${case_id}.ptx"
  rc=0

  rm -f "$log_stdout" "$log_stderr" "$out_path"

  case "$mode" in
    check-pass)
      set +e
      SOUNIO_STDLIB_PATH="$ROOT_DIR/stdlib" \
        run_with_timeout "$TIMEOUT_SECS" "$GPU_SOUC" check "$program" \
        >"$log_stdout" 2>"$log_stderr"
      rc=$?
      set -e
      if [ "$rc" -ne 0 ]; then
        fail "$case_id" "check failed (support_class=$support_class exit=$rc)"
        continue
      fi
      pass "$case_id" "check ok ($support_class)"
      ;;
    build-pass)
      set +e
      SOUNIO_STDLIB_PATH="$ROOT_DIR/stdlib" \
        run_with_timeout "$TIMEOUT_SECS" "$GPU_SOUC" build "$program" --backend gpu -o "$out_path" \
        >"$log_stdout" 2>"$log_stderr"
      rc=$?
      set -e
      if [ "$rc" -ne 0 ]; then
        fail "$case_id" "gpu build failed (support_class=$support_class exit=$rc)"
        continue
      fi
      if [ ! -s "$out_path" ]; then
        fail "$case_id" "missing emitted artifact"
        continue
      fi
      if [ "$expected_marker" != "-" ] && ! grep -qF "$expected_marker" "$out_path"; then
        fail "$case_id" "missing emitted marker: $expected_marker"
        continue
      fi
      pass "$case_id" "gpu lowering ok ($support_class)"
      ;;
    check-fail)
      set +e
      SOUNIO_STDLIB_PATH="$ROOT_DIR/stdlib" \
        run_with_timeout "$TIMEOUT_SECS" "$GPU_SOUC" check "$program" \
        >"$log_stdout" 2>"$log_stderr"
      rc=$?
      set -e
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

echo "GPU_SURFACE_LOWERING_GATE_SUMMARY pass=$PASS_COUNT fail=$FAIL_COUNT skip=$SKIP_COUNT"

if [ "$FAIL_COUNT" -ne 0 ]; then
  exit 1
fi
