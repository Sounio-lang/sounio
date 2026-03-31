#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

ARTIFACT_BIN="${ARTIFACT_BIN:-$ROOT_DIR/artifacts/self-hosted/souc-self-hosted-x86_64}"
SOURCE_FILE="${SOURCE_FILE:-self-hosted/compiler/lean_single.sio}"
WORK_DIR="${WORK_DIR:-/tmp/sounio-selfhost-reproducible-bootstrap-gate}"
FIXED_WORK_DIR="$WORK_DIR/fixed_point"
ARTIFACT_DIR="$WORK_DIR/artifacts"
SUMMARY_FILE="$ARTIFACT_DIR/summary.txt"
SUMMARY_JSON="$ARTIFACT_DIR/summary.v1.json"
SUMMARY_MD="$ARTIFACT_DIR/summary.v1.md"

mkdir -p "$ARTIFACT_DIR"

echo "SELFHOST_REPRODUCIBLE_BOOTSTRAP_GATE_START"
echo "artifact_bin=$ARTIFACT_BIN"
echo "source_file=$SOURCE_FILE"
echo "work_dir=$WORK_DIR"

env SOUC_NATIVE="$ARTIFACT_BIN" SOURCE_FILE="$SOURCE_FILE" WORK_DIR="$FIXED_WORK_DIR" \
  bash "$ROOT_DIR/scripts/selfhost/selfhost_x86_fixed_point_gate.sh"

GEN1_PATH="$FIXED_WORK_DIR/artifacts/gen1.elf"
GEN2_PATH="$FIXED_WORK_DIR/artifacts/gen2.elf"
GEN3_PATH="$FIXED_WORK_DIR/artifacts/gen3.elf"

ARTIFACT_SHA256="$(sha256sum "$ARTIFACT_BIN" | awk '{print $1}')"
GEN1_SHA256="$(sha256sum "$GEN1_PATH" | awk '{print $1}')"
GEN2_SHA256="$(sha256sum "$GEN2_PATH" | awk '{print $1}')"
GEN3_SHA256="$(sha256sum "$GEN3_PATH" | awk '{print $1}')"
GEN2_MD5="$(md5sum "$GEN2_PATH" | awk '{print $1}')"
GEN3_MD5="$(md5sum "$GEN3_PATH" | awk '{print $1}')"

ARTIFACT_MATCHES_GEN1=false
GEN1_MATCHES_GEN2=false
GEN2_MATCHES_GEN3=false
ALL_EQUAL=false
if [ "$ARTIFACT_SHA256" = "$GEN1_SHA256" ]; then ARTIFACT_MATCHES_GEN1=true; fi
if [ "$GEN1_SHA256" = "$GEN2_SHA256" ]; then GEN1_MATCHES_GEN2=true; fi
if [ "$GEN2_SHA256" = "$GEN3_SHA256" ]; then GEN2_MATCHES_GEN3=true; fi
if [ "$ARTIFACT_MATCHES_GEN1" = true ] && [ "$GEN1_MATCHES_GEN2" = true ] && [ "$GEN2_MATCHES_GEN3" = true ]; then
  ALL_EQUAL=true
fi

OVERALL_STATUS="fail"
if [ "$ALL_EQUAL" = true ] && [ "$GEN2_MD5" = "$GEN3_MD5" ]; then
  OVERALL_STATUS="pass"
fi

{
  echo "artifact_bin=$ARTIFACT_BIN"
  echo "source_file=$SOURCE_FILE"
  echo "fixed_work_dir=$FIXED_WORK_DIR"
  echo "gen1_path=$GEN1_PATH"
  echo "gen2_path=$GEN2_PATH"
  echo "gen3_path=$GEN3_PATH"
  echo "artifact_sha256=$ARTIFACT_SHA256"
  echo "gen1_sha256=$GEN1_SHA256"
  echo "gen2_sha256=$GEN2_SHA256"
  echo "gen3_sha256=$GEN3_SHA256"
  echo "gen2_md5=$GEN2_MD5"
  echo "gen3_md5=$GEN3_MD5"
  echo "artifact_matches_gen1=$ARTIFACT_MATCHES_GEN1"
  echo "gen1_matches_gen2=$GEN1_MATCHES_GEN2"
  echo "gen2_matches_gen3=$GEN2_MATCHES_GEN3"
  echo "all_equal=$ALL_EQUAL"
  echo "overall_status=$OVERALL_STATUS"
  echo "summary_json=$SUMMARY_JSON"
  echo "summary_md=$SUMMARY_MD"
} >"$SUMMARY_FILE"

cat >"$SUMMARY_JSON" <<EOF
{
  "schema": "sounio.selfhost_reproducible_bootstrap_gate.v1",
  "plane": "repo_local_reproducible_bootstrap",
  "artifact_bin": "$ARTIFACT_BIN",
  "source_file": "$SOURCE_FILE",
  "fixed_work_dir": "$FIXED_WORK_DIR",
  "artifact_sha256": "$ARTIFACT_SHA256",
  "gen1_sha256": "$GEN1_SHA256",
  "gen2_sha256": "$GEN2_SHA256",
  "gen3_sha256": "$GEN3_SHA256",
  "gen2_md5": "$GEN2_MD5",
  "gen3_md5": "$GEN3_MD5",
  "checks": {
    "artifact_matches_gen1": $ARTIFACT_MATCHES_GEN1,
    "gen1_matches_gen2": $GEN1_MATCHES_GEN2,
    "gen2_matches_gen3": $GEN2_MATCHES_GEN3,
    "all_equal": $ALL_EQUAL
  },
  "overall_status": "$OVERALL_STATUS"
}
EOF

cat >"$SUMMARY_MD" <<EOF
# Selfhost Reproducible Bootstrap Gate

- overall_status: $OVERALL_STATUS
- artifact_sha256: $ARTIFACT_SHA256
- gen1_sha256: $GEN1_SHA256
- gen2_sha256: $GEN2_SHA256
- gen3_sha256: $GEN3_SHA256
- gen2_md5: $GEN2_MD5
- gen3_md5: $GEN3_MD5
- artifact_matches_gen1: $ARTIFACT_MATCHES_GEN1
- gen1_matches_gen2: $GEN1_MATCHES_GEN2
- gen2_matches_gen3: $GEN2_MATCHES_GEN3
- all_equal: $ALL_EQUAL
EOF

echo "SELFHOST_REPRODUCIBLE_BOOTSTRAP_GATE_DONE"
echo "summary_file=$SUMMARY_FILE"

if [ "$OVERALL_STATUS" != "pass" ]; then
  exit 1
fi

exit 0
