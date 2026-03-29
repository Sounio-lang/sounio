#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

WORK_DIR="${WORK_DIR:-/tmp/sounio-selfhost-release-candidate-gate}"
DRIFT_WORK_DIR="$WORK_DIR/drift"
AUTHORITY_WORK_DIR="$WORK_DIR/authority"
ABI_WORK_DIR="$WORK_DIR/abi"
PROVENANCE_WORK_DIR="$WORK_DIR/provenance"
SUMMARY_FILE="$WORK_DIR/summary.txt"

mkdir -p "$WORK_DIR"

env WORK_DIR="$DRIFT_WORK_DIR" bash "$ROOT_DIR/scripts/selfhost/selfhost_release_drift_gate.sh"
env WORK_DIR="$AUTHORITY_WORK_DIR" bash "$ROOT_DIR/scripts/selfhost/selfhost_authority_gate.sh"
env WORK_DIR="$ABI_WORK_DIR" bash "$ROOT_DIR/scripts/selfhost/selfhost_ci_abi_parity_gate.sh"
env WORK_DIR="$PROVENANCE_WORK_DIR" bash "$ROOT_DIR/scripts/selfhost/selfhost_artifact_provenance_gate.sh"

{
  echo "drift_work_dir=$DRIFT_WORK_DIR"
  echo "authority_work_dir=$AUTHORITY_WORK_DIR"
  echo "abi_work_dir=$ABI_WORK_DIR"
  echo "provenance_work_dir=$PROVENANCE_WORK_DIR"
} >"$SUMMARY_FILE"

echo "SELFHOST_RELEASE_CANDIDATE_GATE_DONE"
echo "summary_file=$SUMMARY_FILE"
exit 0
