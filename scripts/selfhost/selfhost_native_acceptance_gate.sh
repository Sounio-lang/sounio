#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SOUC_NATIVE="${SOUC_NATIVE:-$ROOT_DIR/artifacts/self-hosted/souc-self-hosted-x86_64}"
SOUC_TYPECHECK_NATIVE="${SOUC_TYPECHECK_NATIVE:-$SOUC_NATIVE}"
SOUNIO_NATIVE_TYPECHECK_PROOF="${SOUNIO_NATIVE_TYPECHECK_PROOF:-1}"
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
echo "souc_typecheck_native=$SOUC_TYPECHECK_NATIVE"
echo "work_dir=$WORK_DIR"
echo "native_fail_fast=${SOUNIO_NATIVE_FAIL_FAST:-0}"
echo "native_typecheck_proof=$SOUNIO_NATIVE_TYPECHECK_PROOF"

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

bash "$ROOT_DIR/scripts/selfhost/selfhost_native_runtime_proof.sh" \
  >"$RUNTIME_LOG" 2>&1 \
  || {
    echo "error: native runtime proof failed (see $RUNTIME_LOG)" >&2
    cat "$RUNTIME_LOG" >&2 || true
    exit 1
  }

if [ "$SOUNIO_NATIVE_TYPECHECK_PROOF" = "1" ]; then
  SOUC_NATIVE="$SOUC_TYPECHECK_NATIVE" \
    bash "$ROOT_DIR/scripts/selfhost/selfhost_native_typecheck_proof.sh" \
    >"$TYPECHECK_LOG" 2>&1 \
    || {
      echo "error: native typecheck proof failed (see $TYPECHECK_LOG)" >&2
      cat "$TYPECHECK_LOG" >&2 || true
      exit 1
    }
else
  echo "SELFHOST_NATIVE_TYPECHECK_PROOF_SKIPPED explicit native_typecheck_proof=$SOUNIO_NATIVE_TYPECHECK_PROOF" >"$TYPECHECK_LOG"
fi

{
  echo "souc_native=$SOUC_NATIVE"
  echo "souc_typecheck_native=$SOUC_TYPECHECK_NATIVE"
  echo "native_fail_fast=${SOUNIO_NATIVE_FAIL_FAST:-0}"
  echo "native_typecheck_proof=$SOUNIO_NATIVE_TYPECHECK_PROOF"
  echo "macos_log=$MACOS_LOG"
  echo "runtime_log=$RUNTIME_LOG"
  echo "typecheck_log=$TYPECHECK_LOG"
} >"$SUMMARY_FILE"

echo "SELFHOST_NATIVE_ACCEPTANCE_GATE_DONE"
echo "macos_log=$MACOS_LOG"
echo "runtime_log=$RUNTIME_LOG"
echo "typecheck_log=$TYPECHECK_LOG"
