#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

POLICY_FILE="${POLICY_FILE:-$ROOT_DIR/scripts/selfhost/selfhost_promotion_policy.v1.json}"
REQUIRED_CHECKS_FILE="${REQUIRED_CHECKS_FILE:-$ROOT_DIR/scripts/selfhost/selfhost_required_checks.v1.json}"
CI_WORKFLOW="${CI_WORKFLOW:-$ROOT_DIR/.github/workflows/ci.yml}"
RELEASE_WORKFLOW="${RELEASE_WORKFLOW:-$ROOT_DIR/.github/workflows/release-gate.yml}"
AUTHORITY_DOC="${AUTHORITY_DOC:-$ROOT_DIR/docs/implementation/SELFHOST_AUTHORITY_MODEL.md}"
RELEASE_DOC="${RELEASE_DOC:-$ROOT_DIR/docs/implementation/SELFHOST_RELEASE_TRAIN.md}"
WORK_DIR="${WORK_DIR:-/tmp/sounio-selfhost-release-drift-gate}"
ARTIFACT_DIR="$WORK_DIR/artifacts"
REPORT_JSON="$ARTIFACT_DIR/summary.v1.json"
REPORT_MD="$ARTIFACT_DIR/summary.v1.md"
SUMMARY_FILE="$ARTIFACT_DIR/summary.txt"

mkdir -p "$ARTIFACT_DIR"

echo "SELFHOST_RELEASE_DRIFT_GATE_START"
echo "policy_file=$POLICY_FILE"
echo "required_checks=$REQUIRED_CHECKS_FILE"
echo "work_dir=$WORK_DIR"

python3 "$ROOT_DIR/scripts/selfhost/check_selfhost_release_drift.py" \
  --policy "$POLICY_FILE" \
  --required-checks "$REQUIRED_CHECKS_FILE" \
  --ci-workflow "$CI_WORKFLOW" \
  --release-workflow "$RELEASE_WORKFLOW" \
  --authority-doc "$AUTHORITY_DOC" \
  --release-doc "$RELEASE_DOC" \
  --json-out "$REPORT_JSON" \
  --md-out "$REPORT_MD"

{
  echo "policy_file=$POLICY_FILE"
  echo "required_checks=$REQUIRED_CHECKS_FILE"
  echo "report_json=$REPORT_JSON"
  echo "report_md=$REPORT_MD"
} >"$SUMMARY_FILE"

echo "SELFHOST_RELEASE_DRIFT_GATE_DONE"
echo "summary_file=$SUMMARY_FILE"
exit 0
