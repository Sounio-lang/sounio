#!/usr/bin/env bash
#
# bootstrap_full_gate.sh
#
# Dedicated gate for BOOTSTRAP_PROFILE=full with auditable log + summary output.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

LOG_DIR="artifacts/omega"
LOG_FILE="$LOG_DIR/bootstrap_full_gate.log"
SUMMARY_FILE="$LOG_DIR/bootstrap_full_error_summary.v1.json"

mkdir -p "$LOG_DIR"

set +e
BOOTSTRAP_PROFILE=full bash scripts/bootstrap_concat.sh 2>&1 | tee "$LOG_FILE"
STATUS="${PIPESTATUS[0]}"
set -e

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
  echo "BOOTSTRAP_FULL_GATE_PASS"
  exit 0
fi

echo "BOOTSTRAP_FULL_GATE_FAIL"
echo "  log: $LOG_FILE"
if [ -f "$SUMMARY_FILE" ]; then
  echo "  summary: $SUMMARY_FILE"
  if command -v jq >/dev/null 2>&1; then
    jq '.counts' "$SUMMARY_FILE" || true
  fi
fi
exit "$STATUS"
