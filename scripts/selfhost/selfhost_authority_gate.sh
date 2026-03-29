#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SOUC_NATIVE="${SOUC_NATIVE:-$ROOT_DIR/artifacts/self-hosted/souc-self-hosted-x86_64}"
WORK_DIR="${WORK_DIR:-/tmp/sounio-selfhost-authority-gate}"
LOG_DIR="$WORK_DIR/logs"
ARTIFACT_DIR="$WORK_DIR/artifacts"

POLICY_WORK_DIR="$WORK_DIR/promotion_policy"
DRIFT_WORK_DIR="$WORK_DIR/release_drift"
FIXED_POINT_WORK_DIR="$WORK_DIR/fixed_point"
FALLBACK_WORK_DIR="$WORK_DIR/fallback_inventory"
ABI_WORK_DIR="$WORK_DIR/abi_regressions"
AARCH64_WORK_DIR="$WORK_DIR/aarch64_compile"
AARCH64_RUNTIME_WORK_DIR="$WORK_DIR/aarch64_runtime"
ACCEPTANCE_WORK_DIR="$WORK_DIR/native_acceptance"
PARITY_WORK_DIR="$WORK_DIR/source_artifact_parity"
RUN_LEGACY_ACCEPTANCE="${RUN_LEGACY_ACCEPTANCE:-0}"

POLICY_LOG="$LOG_DIR/promotion_policy.log"
DRIFT_LOG="$LOG_DIR/release_drift.log"
FIXED_LOG="$LOG_DIR/fixed_point.log"
FALLBACK_LOG="$LOG_DIR/fallback_inventory.log"
ABI_LOG="$LOG_DIR/abi_regressions.log"
AARCH64_LOG="$LOG_DIR/aarch64_compile.log"
AARCH64_RUNTIME_LOG="$LOG_DIR/aarch64_runtime.log"
ACCEPTANCE_LOG="$LOG_DIR/native_acceptance.log"
PARITY_LOG="$LOG_DIR/source_artifact_parity.log"
SUMMARY_FILE="$ARTIFACT_DIR/summary.txt"
SUMMARY_JSON="$ARTIFACT_DIR/summary.v2.json"
SUMMARY_MD="$ARTIFACT_DIR/summary.v2.md"
AUTHORITY_REPORT_LOG="$LOG_DIR/authority_report.log"

mkdir -p "$LOG_DIR" "$ARTIFACT_DIR"

echo "SELFHOST_AUTHORITY_GATE_START"
echo "souc_native=$SOUC_NATIVE"
echo "work_dir=$WORK_DIR"

env WORK_DIR="$POLICY_WORK_DIR" \
  bash "$ROOT_DIR/scripts/selfhost/selfhost_promotion_policy_gate.sh" >"$POLICY_LOG" 2>&1 || {
    echo "error: promotion policy gate failed (see $POLICY_LOG)" >&2
    cat "$POLICY_LOG" >&2 || true
    exit 1
  }

env WORK_DIR="$DRIFT_WORK_DIR" \
  bash "$ROOT_DIR/scripts/selfhost/selfhost_release_drift_gate.sh" >"$DRIFT_LOG" 2>&1 || {
    echo "error: release drift gate failed (see $DRIFT_LOG)" >&2
    cat "$DRIFT_LOG" >&2 || true
    exit 1
  }

env SOUC_NATIVE="$SOUC_NATIVE" WORK_DIR="$FIXED_POINT_WORK_DIR" \
  bash "$ROOT_DIR/scripts/selfhost/selfhost_x86_fixed_point_gate.sh" >"$FIXED_LOG" 2>&1 || {
    echo "error: fixed-point gate failed (see $FIXED_LOG)" >&2
    cat "$FIXED_LOG" >&2 || true
    exit 1
  }

GEN2_PATH="$FIXED_POINT_WORK_DIR/artifacts/gen2.elf"

env WORK_DIR="$FALLBACK_WORK_DIR" \
  bash "$ROOT_DIR/scripts/selfhost/selfhost_fallback_inventory_gate.sh" >"$FALLBACK_LOG" 2>&1 || {
    echo "error: fallback inventory gate failed (see $FALLBACK_LOG)" >&2
    cat "$FALLBACK_LOG" >&2 || true
    exit 1
  }

env SOUC_NATIVE="$GEN2_PATH" WORK_DIR="$ABI_WORK_DIR" \
  bash "$ROOT_DIR/scripts/selfhost/selfhost_abi_parity_regression_gate.sh" >"$ABI_LOG" 2>&1 || {
    echo "error: ABI/parity regression gate failed (see $ABI_LOG)" >&2
    cat "$ABI_LOG" >&2 || true
    exit 1
  }

env SOUC_NATIVE="$GEN2_PATH" WORK_DIR="$AARCH64_WORK_DIR" \
  bash "$ROOT_DIR/scripts/selfhost/selfhost_aarch64_compile_proof.sh" >"$AARCH64_LOG" 2>&1 || {
    echo "error: aarch64 compile proof failed (see $AARCH64_LOG)" >&2
    cat "$AARCH64_LOG" >&2 || true
    exit 1
  }

env SOUC_NATIVE="$GEN2_PATH" WORK_DIR="$AARCH64_RUNTIME_WORK_DIR" \
  bash "$ROOT_DIR/scripts/selfhost/selfhost_aarch64_runtime_gate.sh" >"$AARCH64_RUNTIME_LOG" 2>&1 || {
    echo "error: aarch64 runtime gate failed (see $AARCH64_RUNTIME_LOG)" >&2
    cat "$AARCH64_RUNTIME_LOG" >&2 || true
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

REPORT_ARGS=(
  --policy-summary "$POLICY_WORK_DIR/artifacts/summary.txt"
  --drift-summary "$DRIFT_WORK_DIR/artifacts/summary.txt"
  --fixed-summary "$FIXED_POINT_WORK_DIR/artifacts/summary.txt"
  --fallback-summary "$FALLBACK_WORK_DIR/artifacts/summary.txt"
  --abi-summary "$ABI_WORK_DIR/artifacts/summary.txt"
  --aarch64-summary "$AARCH64_WORK_DIR/artifacts/summary.txt"
  --aarch64-runtime-summary "$AARCH64_RUNTIME_WORK_DIR/artifacts/summary.txt"
  --parity-json "$PARITY_WORK_DIR/artifacts/suite_delta.v1.json"
  --legacy-status "$LEGACY_ACCEPTANCE_STATUS"
  --summary-json "$SUMMARY_JSON"
  --summary-md "$SUMMARY_MD"
)
if [ "$RUN_LEGACY_ACCEPTANCE" = "1" ]; then
  REPORT_ARGS+=(--legacy-summary "$ACCEPTANCE_WORK_DIR/artifacts/summary.txt")
fi

python3 "$ROOT_DIR/scripts/selfhost/selfhost_authority_report.py" \
  "${REPORT_ARGS[@]}" >"$AUTHORITY_REPORT_LOG" 2>&1 || {
    echo "error: authority summary generation failed (see $AUTHORITY_REPORT_LOG)" >&2
    cat "$AUTHORITY_REPORT_LOG" >&2 || true
    exit 1
  }

{
  echo "souc_native=$SOUC_NATIVE"
  echo "policy_log=$POLICY_LOG"
  echo "drift_log=$DRIFT_LOG"
  echo "fixed_log=$FIXED_LOG"
  echo "fallback_log=$FALLBACK_LOG"
  echo "abi_log=$ABI_LOG"
  echo "aarch64_log=$AARCH64_LOG"
  echo "aarch64_runtime_log=$AARCH64_RUNTIME_LOG"
  echo "parity_log=$PARITY_LOG"
  echo "policy_work_dir=$POLICY_WORK_DIR"
  echo "drift_work_dir=$DRIFT_WORK_DIR"
  echo "fixed_point_work_dir=$FIXED_POINT_WORK_DIR"
  echo "fallback_work_dir=$FALLBACK_WORK_DIR"
  echo "abi_work_dir=$ABI_WORK_DIR"
  echo "aarch64_work_dir=$AARCH64_WORK_DIR"
  echo "aarch64_runtime_work_dir=$AARCH64_RUNTIME_WORK_DIR"
  echo "parity_work_dir=$PARITY_WORK_DIR"
  echo "gen2_path=$GEN2_PATH"
  echo "legacy_acceptance_status=$LEGACY_ACCEPTANCE_STATUS"
  echo "summary_json=$SUMMARY_JSON"
  echo "summary_md=$SUMMARY_MD"
  echo "authority_report_log=$AUTHORITY_REPORT_LOG"
  if [ "$RUN_LEGACY_ACCEPTANCE" = "1" ]; then
    echo "acceptance_log=$ACCEPTANCE_LOG"
    echo "acceptance_work_dir=$ACCEPTANCE_WORK_DIR"
  fi
} >"$SUMMARY_FILE"

echo "SELFHOST_AUTHORITY_GATE_DONE"
echo "summary_file=$SUMMARY_FILE"
exit 0
