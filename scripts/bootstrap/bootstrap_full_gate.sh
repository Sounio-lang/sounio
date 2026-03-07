#!/usr/bin/env bash
#
# bootstrap_full_gate.sh
#
# Dedicated gate for BOOTSTRAP_PROFILE=full with auditable log + summary output.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

LOG_DIR="artifacts/omega"
LOG_FILE="$LOG_DIR/bootstrap_full_gate.log"
SUMMARY_FILE="$LOG_DIR/bootstrap_full_error_summary.v1.json"
STATUS_FILE="$LOG_DIR/bootstrap_full_gate_status.v1.json"
KNOWLEDGE_SCRIPT="scripts/bootstrap/run_knowledge_bootstrap_tests.sh"
KNOWLEDGE_SUMMARY_FILE="$LOG_DIR/bootstrap_knowledge_tests.v1.json"

mkdir -p "$LOG_DIR"

set +e
BOOTSTRAP_PROFILE=full bash scripts/bootstrap/bootstrap_concat.sh 2>&1 | tee "$LOG_FILE"
STATUS="${PIPESTATUS[0]}"
set -e

KNOWLEDGE_STATUS=125
FULL_STATUS_LABEL="fail"
KNOWLEDGE_STATUS_LABEL="not_run"

if [ "$STATUS" -eq 0 ]; then
  FULL_STATUS_LABEL="pass"
  set +e
  bash "$KNOWLEDGE_SCRIPT" 2>&1 | tee -a "$LOG_FILE"
  KNOWLEDGE_STATUS="${PIPESTATUS[0]}"
  set -e
  if [ "$KNOWLEDGE_STATUS" -eq 0 ]; then
    KNOWLEDGE_STATUS_LABEL="pass"
  else
    KNOWLEDGE_STATUS_LABEL="fail"
  fi
fi

if [ "$STATUS" -eq 0 ]; then
  cat >"$SUMMARY_FILE" <<EOF
{
  "schema": "sounio.bootstrap.full-error-summary.v1",
  "generated_at_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "profile": "full",
  "exit_code": 0,
  "counts": {
    "duplicate_definitions": 0,
    "undefined_variables": 0,
    "type_mismatches": 0,
    "resolution_blocks": 0,
    "stack_overflows": 0
  },
  "log_path": "$LOG_FILE",
  "output_file": "build/bootstrap_stage1.sio"
}
EOF
fi

OVERALL_STATUS="$STATUS"
if [ "$OVERALL_STATUS" -eq 0 ] && [ "$KNOWLEDGE_STATUS" -ne 0 ]; then
  OVERALL_STATUS="$KNOWLEDGE_STATUS"
fi

cat >"$STATUS_FILE" <<EOF
{
  "schema": "sounio.bootstrap.full-gate-status.v1",
  "generated_at_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "profile": "full",
  "exit_code": $OVERALL_STATUS,
  "log_path": "$LOG_FILE",
  "full_concat": {
    "status": "$FULL_STATUS_LABEL",
    "exit_code": $STATUS,
    "summary_path": "$SUMMARY_FILE"
  },
  "knowledge_bootstrap": {
    "status": "$KNOWLEDGE_STATUS_LABEL",
    "exit_code": $KNOWLEDGE_STATUS,
    "script": "$KNOWLEDGE_SCRIPT",
    "summary_path": "$KNOWLEDGE_SUMMARY_FILE"
  }
}
EOF

if [ "$OVERALL_STATUS" -eq 0 ]; then
  echo "BOOTSTRAP_FULL_GATE_PASS"
  echo "  status: $STATUS_FILE"
  exit 0
fi

echo "BOOTSTRAP_FULL_GATE_FAIL"
echo "  log: $LOG_FILE"
echo "  status: $STATUS_FILE"
if [ -f "$SUMMARY_FILE" ]; then
  echo "  summary: $SUMMARY_FILE"
  if command -v jq >/dev/null 2>&1; then
    jq '.counts' "$SUMMARY_FILE" || true
  fi
fi
if [ "$STATUS" -eq 0 ]; then
  echo "  knowledge_summary: $KNOWLEDGE_SUMMARY_FILE"
fi
exit "$OVERALL_STATUS"
