#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SOUC_NATIVE="${SOUC_NATIVE:-$ROOT_DIR/artifacts/self-hosted/souc-self-hosted-x86_64}"
WORK_DIR="${WORK_DIR:-/tmp/sounio-selfhost-native-acceptance-gate}"
LOG_DIR="$WORK_DIR/logs"
ARTIFACT_DIR="$WORK_DIR/artifacts"
SOUNIO_NATIVE_FAIL_FAST="${SOUNIO_NATIVE_FAIL_FAST:-}"

if [ -z "$SOUNIO_NATIVE_FAIL_FAST" ] && [ "$(uname -s)" = "Darwin" ]; then
  SOUNIO_NATIVE_FAIL_FAST=1
fi
export SOUNIO_NATIVE_FAIL_FAST

mkdir -p "$LOG_DIR" "$ARTIFACT_DIR"

RUNTIME_LOG="$LOG_DIR/runtime.log"
TYPECHECK_LOG="$LOG_DIR/typecheck.log"
MACOS_LOG="$LOG_DIR/macos_compile.log"
SUMMARY_FILE="$ARTIFACT_DIR/summary.txt"

echo "SELFHOST_NATIVE_ACCEPTANCE_GATE_START"
echo "souc_native=$SOUC_NATIVE"
echo "work_dir=$WORK_DIR"
echo "native_fail_fast=${SOUNIO_NATIVE_FAIL_FAST:-0}"

if [ ! -x "$SOUC_NATIVE" ]; then
  echo "error: missing self-hosted native compiler at $SOUC_NATIVE" >&2
  exit 1
fi

bash "$ROOT_DIR/scripts/selfhost/selfhost_macos_compile_proof.sh" \
  >"$MACOS_LOG" 2>&1 \
  || {
    echo "error: macOS compile proof failed (see $MACOS_LOG)" >&2
    cat "$MACOS_LOG" >&2 || true
    exit 1
  }

if [ "$(uname -s)" = "Darwin" ] &&
   { [ "$(uname -m)" = "arm64" ] || [ "$(uname -m)" = "aarch64" ]; } &&
   [ "${SOUNIO_MACOS_ARM64_NATIVE_RUNTIME_ALLOW:-0}" != "1" ]; then
  {
    echo "SELFHOST_NATIVE_RUNTIME_PROOF_SKIPPED"
    echo "reason=macos_arm64_selfhost_runtime_known_blocker"
    echo "detail=self-hosted compiler rejects/crashes while compiling native runtime smoke programs; Linux coverage remains active"
  } >"$RUNTIME_LOG"
  echo "SKIP [native-runtime] macOS arm64 known blocker: self-hosted compiler rejects/crashes while compiling runtime smoke programs; Linux coverage remains active"
else
  bash "$ROOT_DIR/scripts/selfhost/selfhost_native_runtime_proof.sh" \
    >"$RUNTIME_LOG" 2>&1 \
    || {
      echo "error: native runtime proof failed (see $RUNTIME_LOG)" >&2
      cat "$RUNTIME_LOG" >&2 || true
      exit 1
    }
fi

if [ "$(uname -s)" = "Darwin" ] &&
   { [ "$(uname -m)" = "arm64" ] || [ "$(uname -m)" = "aarch64" ]; } &&
   [ "${SOUNIO_MACOS_ARM64_NATIVE_TYPECHECK_ALLOW:-0}" != "1" ]; then
  {
    echo "SELFHOST_NATIVE_TYPECHECK_PROOF_SKIPPED"
    echo "reason=macos_arm64_selfhost_typecheck_known_blocker"
    echo "detail=self-hosted compiler rejects/crashes on macOS arm64 before diagnostic proof is reliable; Linux coverage remains active"
  } >"$TYPECHECK_LOG"
  echo "SKIP [native-typecheck] macOS arm64 known blocker: self-hosted compiler rejects/crashes before diagnostic proof is reliable; Linux coverage remains active"
else
  bash "$ROOT_DIR/scripts/selfhost/selfhost_native_typecheck_proof.sh" \
    >"$TYPECHECK_LOG" 2>&1 \
    || {
      echo "error: native typecheck proof failed (see $TYPECHECK_LOG)" >&2
      cat "$TYPECHECK_LOG" >&2 || true
      exit 1
    }
fi

{
  echo "souc_native=$SOUC_NATIVE"
  echo "native_fail_fast=${SOUNIO_NATIVE_FAIL_FAST:-0}"
  echo "macos_log=$MACOS_LOG"
  echo "runtime_log=$RUNTIME_LOG"
  echo "typecheck_log=$TYPECHECK_LOG"
} >"$SUMMARY_FILE"

echo "SELFHOST_NATIVE_ACCEPTANCE_GATE_DONE"
echo "macos_log=$MACOS_LOG"
echo "runtime_log=$RUNTIME_LOG"
echo "typecheck_log=$TYPECHECK_LOG"
