#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

ARTIFACT_PATH="${ARTIFACT_PATH:-$ROOT_DIR/artifacts/self-hosted/souc-self-hosted-x86_64}"
PROVENANCE_PATH="${PROVENANCE_PATH:-$ROOT_DIR/artifacts/self-hosted/souc-self-hosted-x86_64.provenance.json}"
WORK_DIR="${WORK_DIR:-/tmp/sounio-selfhost-artifact-provenance-gate}"
ARTIFACT_DIR="$WORK_DIR/artifacts"
REPORT_JSON="$ARTIFACT_DIR/summary.v1.json"
REPORT_MD="$ARTIFACT_DIR/summary.v1.md"
SUMMARY_FILE="$ARTIFACT_DIR/summary.txt"

mkdir -p "$ARTIFACT_DIR"

echo "SELFHOST_ARTIFACT_PROVENANCE_GATE_START"
echo "artifact_path=$ARTIFACT_PATH"
echo "provenance_path=$PROVENANCE_PATH"
echo "work_dir=$WORK_DIR"

python3 "$ROOT_DIR/scripts/selfhost/verify_selfhost_artifact_provenance.py" \
  --artifact "$ARTIFACT_PATH" \
  --provenance "$PROVENANCE_PATH" \
  --json-out "$REPORT_JSON" \
  --md-out "$REPORT_MD"

{
  echo "artifact_path=$ARTIFACT_PATH"
  echo "provenance_path=$PROVENANCE_PATH"
  echo "report_json=$REPORT_JSON"
  echo "report_md=$REPORT_MD"
} >"$SUMMARY_FILE"

echo "SELFHOST_ARTIFACT_PROVENANCE_GATE_DONE"
echo "summary_file=$SUMMARY_FILE"
exit 0
