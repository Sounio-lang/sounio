#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

POLICY_FILE="${POLICY_FILE:-$ROOT_DIR/scripts/selfhost/selfhost_promotion_policy.v1.json}"
WORK_DIR="${WORK_DIR:-/tmp/sounio-selfhost-promotion-policy-gate}"
ARTIFACT_DIR="$WORK_DIR/artifacts"
REPORT_JSON="$ARTIFACT_DIR/summary.v1.json"
REPORT_MD="$ARTIFACT_DIR/summary.v1.md"
SUMMARY_FILE="$ARTIFACT_DIR/summary.txt"

mkdir -p "$ARTIFACT_DIR"

echo "SELFHOST_PROMOTION_POLICY_GATE_START"
echo "policy_file=$POLICY_FILE"
echo "work_dir=$WORK_DIR"

python3 "$ROOT_DIR/scripts/selfhost/verify_selfhost_promotion_policy.py" \
  --policy "$POLICY_FILE" \
  --json-out "$REPORT_JSON" \
  --md-out "$REPORT_MD"

{
  echo "policy_file=$POLICY_FILE"
  echo "report_json=$REPORT_JSON"
  echo "report_md=$REPORT_MD"
} >"$SUMMARY_FILE"

echo "SELFHOST_PROMOTION_POLICY_GATE_DONE"
echo "summary_file=$SUMMARY_FILE"
exit 0
