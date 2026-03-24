#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SOUC_NATIVE="${SOUC_NATIVE:-/tmp/gen2.elf}"
WORK_DIR="${WORK_DIR:-/tmp/sounio-selfhost-native-acceptance-gate}"
LOG_DIR="$WORK_DIR/logs"
ARTIFACT_DIR="$WORK_DIR/artifacts"

mkdir -p "$LOG_DIR" "$ARTIFACT_DIR"

RUNTIME_LOG="$LOG_DIR/runtime.log"
TYPECHECK_LOG="$LOG_DIR/typecheck.log"
AARCH64_LOG="$LOG_DIR/aarch64_compile.log"
SUMMARY_FILE="$ARTIFACT_DIR/summary.txt"

echo "SELFHOST_NATIVE_ACCEPTANCE_GATE_START"
echo "souc_native=$SOUC_NATIVE"
echo "work_dir=$WORK_DIR"

if [ ! -x "$SOUC_NATIVE" ]; then
  echo "error: missing self-hosted native compiler at $SOUC_NATIVE" >&2
  exit 1
fi

bash "$ROOT_DIR/scripts/selfhost/selfhost_native_runtime_proof.sh" \
  >"$RUNTIME_LOG" 2>&1 \
  || {
    echo "error: native runtime proof failed (see $RUNTIME_LOG)" >&2
    cat "$RUNTIME_LOG" >&2 || true
    exit 1
  }

bash "$ROOT_DIR/scripts/selfhost/selfhost_native_typecheck_proof.sh" \
  >"$TYPECHECK_LOG" 2>&1 \
  || {
    echo "error: native typecheck proof failed (see $TYPECHECK_LOG)" >&2
    cat "$TYPECHECK_LOG" >&2 || true
    exit 1
  }

bash "$ROOT_DIR/scripts/selfhost/selfhost_aarch64_compile_proof.sh" \
  >"$AARCH64_LOG" 2>&1 \
  || {
    echo "error: aarch64 compile proof failed (see $AARCH64_LOG)" >&2
    cat "$AARCH64_LOG" >&2 || true
    exit 1
  }

{
  echo "souc_native=$SOUC_NATIVE"
  echo "runtime_log=$RUNTIME_LOG"
  echo "typecheck_log=$TYPECHECK_LOG"
  echo "aarch64_log=$AARCH64_LOG"
} >"$SUMMARY_FILE"

echo "SELFHOST_NATIVE_ACCEPTANCE_GATE_DONE"
echo "runtime_log=$RUNTIME_LOG"
echo "typecheck_log=$TYPECHECK_LOG"
echo "aarch64_log=$AARCH64_LOG"
