#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

BOOTSTRAP_BIN="${BOOTSTRAP_BIN:-$ROOT_DIR/artifacts/self-hosted/souc-self-hosted-x86_64}"
TARGET_ARTIFACT="${TARGET_ARTIFACT:-$ROOT_DIR/artifacts/self-hosted/souc-self-hosted-x86_64}"
TARGET_PROVENANCE="${TARGET_PROVENANCE:-$ROOT_DIR/artifacts/self-hosted/souc-self-hosted-x86_64.provenance.json}"
SOURCE_FILE="${SOURCE_FILE:-self-hosted/compiler/lean_single.sio}"
WORK_DIR="${WORK_DIR:-/tmp/sounio-selfhost-artifact-update}"

BOOTSTRAP_WORK_DIR="$WORK_DIR/bootstrap_authority"
PROMOTION_WORK_DIR="$WORK_DIR/promotion_authority"
BOOTSTRAP_COPY="$WORK_DIR/bootstrap-artifact.bin"

mkdir -p "$WORK_DIR"

if [ ! -x "$BOOTSTRAP_BIN" ]; then
  echo "error: missing bootstrap compiler at $BOOTSTRAP_BIN" >&2
  exit 1
fi

cp "$BOOTSTRAP_BIN" "$BOOTSTRAP_COPY"
chmod +x "$BOOTSTRAP_COPY"
BOOTSTRAP_SHA256="$(sha256sum "$BOOTSTRAP_COPY" | awk '{print $1}')"

env SOUC_NATIVE="$BOOTSTRAP_COPY" WORK_DIR="$BOOTSTRAP_WORK_DIR" \
  bash "$ROOT_DIR/scripts/selfhost/selfhost_authority_gate.sh"

cp "$BOOTSTRAP_WORK_DIR/fixed_point/artifacts/gen2.elf" "$TARGET_ARTIFACT"
chmod +x "$TARGET_ARTIFACT"

env SOUC_NATIVE="$TARGET_ARTIFACT" WORK_DIR="$PROMOTION_WORK_DIR" \
  bash "$ROOT_DIR/scripts/selfhost/selfhost_authority_gate.sh"

python3 "$ROOT_DIR/scripts/selfhost/selfhost_artifact_attest.py" \
  --artifact "$TARGET_ARTIFACT" \
  --source "$SOURCE_FILE" \
  --authority-summary "$PROMOTION_WORK_DIR/artifacts/summary.v2.json" \
  --bootstrap-summary "$BOOTSTRAP_WORK_DIR/artifacts/summary.v2.json" \
  --bootstrap-sha256 "$BOOTSTRAP_SHA256" \
  --provenance-out "$TARGET_PROVENANCE"

env ARTIFACT_PATH="$TARGET_ARTIFACT" PROVENANCE_PATH="$TARGET_PROVENANCE" \
  WORK_DIR="$WORK_DIR/provenance_verification" \
  bash "$ROOT_DIR/scripts/selfhost/selfhost_artifact_provenance_gate.sh"

echo "SELFHOST_ARTIFACT_UPDATE_DONE"
echo "target_artifact=$TARGET_ARTIFACT"
echo "target_provenance=$TARGET_PROVENANCE"
echo "bootstrap_sha256=$BOOTSTRAP_SHA256"
