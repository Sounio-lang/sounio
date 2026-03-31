#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

POLICY_FILE="${POLICY_FILE:-$ROOT_DIR/scripts/selfhost/selfhost_promotion_policy.v1.json}"
ARTIFACT_PATH="${ARTIFACT_PATH:-$ROOT_DIR/artifacts/self-hosted/souc-self-hosted-x86_64}"
PROVENANCE_PATH="${PROVENANCE_PATH:-$ROOT_DIR/artifacts/self-hosted/souc-self-hosted-x86_64.provenance.json}"
SOURCE_FILE="${SOURCE_FILE:-self-hosted/compiler/lean_single.sio}"
WORK_DIR="${WORK_DIR:-/tmp/sounio-selfhost-dual-trust-gate}"
PROVENANCE_WORK_DIR="$WORK_DIR/provenance"
REPRO_WORK_DIR="$WORK_DIR/reproducible_bootstrap"
ARTIFACT_DIR="$WORK_DIR/artifacts"
SUMMARY_FILE="$ARTIFACT_DIR/summary.txt"
SUMMARY_JSON="$ARTIFACT_DIR/summary.v1.json"
SUMMARY_MD="$ARTIFACT_DIR/summary.v1.md"

mkdir -p "$ARTIFACT_DIR"

echo "SELFHOST_DUAL_TRUST_GATE_START"
echo "policy_file=$POLICY_FILE"
echo "artifact_path=$ARTIFACT_PATH"
echo "provenance_path=$PROVENANCE_PATH"
echo "source_file=$SOURCE_FILE"
echo "work_dir=$WORK_DIR"

env POLICY_FILE="$POLICY_FILE" ARTIFACT_PATH="$ARTIFACT_PATH" PROVENANCE_PATH="$PROVENANCE_PATH" \
  WORK_DIR="$PROVENANCE_WORK_DIR" \
  bash "$ROOT_DIR/scripts/selfhost/selfhost_artifact_provenance_gate.sh"

env ARTIFACT_BIN="$ARTIFACT_PATH" SOURCE_FILE="$SOURCE_FILE" WORK_DIR="$REPRO_WORK_DIR" \
  bash "$ROOT_DIR/scripts/selfhost/selfhost_reproducible_bootstrap_gate.sh"

PROVENANCE_REPORT_JSON="$PROVENANCE_WORK_DIR/artifacts/summary.v1.json"
REPRO_REPORT_JSON="$REPRO_WORK_DIR/artifacts/summary.v1.json"

cat >"$SUMMARY_FILE" <<EOF
policy_file=$POLICY_FILE
artifact_path=$ARTIFACT_PATH
provenance_path=$PROVENANCE_PATH
source_file=$SOURCE_FILE
provenance_report_json=$PROVENANCE_REPORT_JSON
reproducible_bootstrap_report_json=$REPRO_REPORT_JSON
summary_json=$SUMMARY_JSON
summary_md=$SUMMARY_MD
overall_status=pass
EOF

cat >"$SUMMARY_JSON" <<EOF
{
  "schema": "sounio.selfhost_dual_trust_gate.v1",
  "overall_status": "pass",
  "canonical_trust_plane": {
    "plane": "repo_local_provenance",
    "status": "pass",
    "report_json": "$PROVENANCE_REPORT_JSON"
  },
  "supplemental_trust_plane": {
    "plane": "repo_local_reproducible_bootstrap",
    "status": "pass",
    "report_json": "$REPRO_REPORT_JSON"
  }
}
EOF

cat >"$SUMMARY_MD" <<EOF
# Selfhost Dual Trust Gate

- overall_status: pass
- canonical_trust_plane: repo_local_provenance
- supplemental_trust_plane: repo_local_reproducible_bootstrap
- provenance_report_json: $PROVENANCE_REPORT_JSON
- reproducible_bootstrap_report_json: $REPRO_REPORT_JSON
EOF

echo "SELFHOST_DUAL_TRUST_GATE_DONE"
echo "summary_file=$SUMMARY_FILE"
exit 0
