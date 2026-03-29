#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SOURCE_FILE="${SOURCE_FILE:-self-hosted/compiler/lean_single.sio}"
WORK_DIR="${WORK_DIR:-/tmp/sounio-selfhost-fallback-inventory-gate}"
ARTIFACT_DIR="$WORK_DIR/artifacts"
REPORT_JSON="$ARTIFACT_DIR/fallback_inventory.v1.json"
REPORT_MD="$ARTIFACT_DIR/fallback_inventory.v1.md"
SUMMARY_FILE="$ARTIFACT_DIR/summary.txt"

mkdir -p "$ARTIFACT_DIR"

echo "SELFHOST_FALLBACK_INVENTORY_GATE_START"
echo "source_file=$SOURCE_FILE"
echo "work_dir=$WORK_DIR"

python3 "$ROOT_DIR/scripts/selfhost/selfhost_fallback_inventory.py" \
  --source "$SOURCE_FILE" \
  --json-out "$REPORT_JSON" \
  --md-out "$REPORT_MD"

{
  echo "source_file=$SOURCE_FILE"
  echo "report_json=$REPORT_JSON"
  echo "report_md=$REPORT_MD"
} >"$SUMMARY_FILE"

echo "SELFHOST_FALLBACK_INVENTORY_GATE_PASS report_json=$REPORT_JSON report_md=$REPORT_MD"
