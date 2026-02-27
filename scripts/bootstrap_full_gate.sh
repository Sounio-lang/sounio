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
