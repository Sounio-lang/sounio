#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SOUC_NATIVE="${SOUC_NATIVE:-$ROOT_DIR/artifacts/self-hosted/souc-self-hosted-x86_64}"
WORK_DIR="${WORK_DIR:-/tmp/sounio-selfhost-authority-gate}"
LOG_DIR="$WORK_DIR/logs"
ARTIFACT_DIR="$WORK_DIR/artifacts"

FIXED_POINT_WORK_DIR="$WORK_DIR/fixed_point"
ABI_WORK_DIR="$WORK_DIR/abi_regressions"
ACCEPTANCE_WORK_DIR="$WORK_DIR/native_acceptance"
PARITY_WORK_DIR="$WORK_DIR/source_artifact_parity"
RUN_LEGACY_ACCEPTANCE="${RUN_LEGACY_ACCEPTANCE:-0}"

FIXED_LOG="$LOG_DIR/fixed_point.log"
ABI_LOG="$LOG_DIR/abi_regressions.log"
ACCEPTANCE_LOG="$LOG_DIR/native_acceptance.log"
PARITY_LOG="$LOG_DIR/source_artifact_parity.log"
SUMMARY_FILE="$ARTIFACT_DIR/summary.txt"

mkdir -p "$LOG_DIR" "$ARTIFACT_DIR"

echo "SELFHOST_AUTHORITY_GATE_START"
echo "souc_native=$SOUC_NATIVE"
echo "work_dir=$WORK_DIR"

env SOUC_NATIVE="$SOUC_NATIVE" WORK_DIR="$FIXED_POINT_WORK_DIR" \
  bash "$ROOT_DIR/scripts/selfhost/selfhost_x86_fixed_point_gate.sh" >"$FIXED_LOG" 2>&1 || {
    echo "error: fixed-point gate failed (see $FIXED_LOG)" >&2
    cat "$FIXED_LOG" >&2 || true
    exit 1
  }

GEN2_PATH="$FIXED_POINT_WORK_DIR/artifacts/gen2.elf"

env SOUC_NATIVE="$GEN2_PATH" WORK_DIR="$ABI_WORK_DIR" \
  bash "$ROOT_DIR/scripts/selfhost/selfhost_abi_parity_regression_gate.sh" >"$ABI_LOG" 2>&1 || {
    echo "error: ABI/parity regression gate failed (see $ABI_LOG)" >&2
    cat "$ABI_LOG" >&2 || true
    exit 1
  }

LEGACY_ACCEPTANCE_STATUS="skipped"
if [ "$RUN_LEGACY_ACCEPTANCE" = "1" ]; then
  set +e
  env SOUC_NATIVE="$GEN2_PATH" WORK_DIR="$ACCEPTANCE_WORK_DIR" \
    bash "$ROOT_DIR/scripts/selfhost/selfhost_native_acceptance_gate.sh" >"$ACCEPTANCE_LOG" 2>&1
  LEGACY_ACCEPTANCE_EXIT=$?
  set -e
  if [ "$LEGACY_ACCEPTANCE_EXIT" -eq 0 ]; then
    LEGACY_ACCEPTANCE_STATUS="passed"
  else
    LEGACY_ACCEPTANCE_STATUS="failed"
    echo "SELFHOST_AUTHORITY_GATE_INFO legacy_native_acceptance_status=failed log=$ACCEPTANCE_LOG"
  fi
else
  echo "SELFHOST_AUTHORITY_GATE_INFO legacy_native_acceptance_status=skipped reason=accepted_artifact_baseline_not_green"
fi

env ACCEPTED_ARTIFACT_BIN="$SOUC_NATIVE" WORK_DIR="$PARITY_WORK_DIR" \
  bash "$ROOT_DIR/scripts/selfhost/selfhost_source_artifact_parity_gate.sh" >"$PARITY_LOG" 2>&1 || {
    echo "error: source↔artifact parity gate failed (see $PARITY_LOG)" >&2
    cat "$PARITY_LOG" >&2 || true
    exit 1
  }

{
  echo "souc_native=$SOUC_NATIVE"
  echo "fixed_log=$FIXED_LOG"
  echo "abi_log=$ABI_LOG"
  echo "parity_log=$PARITY_LOG"
  echo "fixed_point_work_dir=$FIXED_POINT_WORK_DIR"
  echo "abi_work_dir=$ABI_WORK_DIR"
  echo "parity_work_dir=$PARITY_WORK_DIR"
  echo "gen2_path=$GEN2_PATH"
  echo "legacy_acceptance_status=$LEGACY_ACCEPTANCE_STATUS"
  if [ "$RUN_LEGACY_ACCEPTANCE" = "1" ]; then
    echo "acceptance_log=$ACCEPTANCE_LOG"
    echo "acceptance_work_dir=$ACCEPTANCE_WORK_DIR"
  fi
} >"$SUMMARY_FILE"

echo "SELFHOST_AUTHORITY_GATE_DONE"
echo "summary_file=$SUMMARY_FILE"
exit 0
