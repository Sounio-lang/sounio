#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

POLICY_FILE="${POLICY_FILE:-$ROOT_DIR/scripts/selfhost/selfhost_promotion_policy.v1.json}"
BOOTSTRAP_BIN="${BOOTSTRAP_BIN:-$ROOT_DIR/artifacts/self-hosted/souc-self-hosted-x86_64}"
TARGET_ARTIFACT="${TARGET_ARTIFACT:-$ROOT_DIR/artifacts/self-hosted/souc-self-hosted-x86_64}"
TARGET_PROVENANCE="${TARGET_PROVENANCE:-$ROOT_DIR/artifacts/self-hosted/souc-self-hosted-x86_64.provenance.json}"
SOURCE_FILE="${SOURCE_FILE:-self-hosted/compiler/lean_single.sio}"
WORK_DIR="${WORK_DIR:-/tmp/sounio-selfhost-artifact-update}"

BOOTSTRAP_WORK_DIR="$WORK_DIR/bootstrap_authority"
PROMOTION_WORK_DIR="$WORK_DIR/promotion_authority"
REPRO_WORK_DIR="$WORK_DIR/reproducible_bootstrap"
DUAL_TRUST_WORK_DIR="$WORK_DIR/dual_trust"
BOOTSTRAP_COPY="$WORK_DIR/bootstrap-artifact.bin"

mkdir -p "$WORK_DIR"

if [ ! -x "$BOOTSTRAP_BIN" ]; then
  echo "error: missing bootstrap compiler at $BOOTSTRAP_BIN" >&2
  exit 1
fi

python3 "$ROOT_DIR/scripts/selfhost/verify_selfhost_promotion_policy.py" \
  --policy "$POLICY_FILE" \
  --json-out "$WORK_DIR/policy-summary.json" \
  --md-out "$WORK_DIR/policy-summary.md"

env POLICY_FILE="$POLICY_FILE" WORK_DIR="$WORK_DIR/drift" \
  bash "$ROOT_DIR/scripts/selfhost/selfhost_release_drift_gate.sh"

REQUIRE_CLEAN_TREE="$(python3 - <<'PY' "$POLICY_FILE"
import json, sys
with open(sys.argv[1], 'r', encoding='utf-8') as f:
    policy = json.load(f)
print('1' if policy['artifact_freshness']['require_clean_git_tree_before_promotion'] else '0')
PY
)"

if [ "$REQUIRE_CLEAN_TREE" = "1" ]; then
  if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "error: promotion requires a clean git tree" >&2
    exit 1
  fi
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

env ARTIFACT_BIN="$TARGET_ARTIFACT" SOURCE_FILE="$SOURCE_FILE" WORK_DIR="$REPRO_WORK_DIR" \
  bash "$ROOT_DIR/scripts/selfhost/selfhost_reproducible_bootstrap_gate.sh"

python3 "$ROOT_DIR/scripts/selfhost/selfhost_artifact_attest.py" \
  --artifact "$TARGET_ARTIFACT" \
  --source "$SOURCE_FILE" \
  --authority-summary "$PROMOTION_WORK_DIR/artifacts/summary.v2.json" \
  --bootstrap-summary "$BOOTSTRAP_WORK_DIR/artifacts/summary.v2.json" \
  --bootstrap-sha256 "$BOOTSTRAP_SHA256" \
  --policy-file "$POLICY_FILE" \
  --repro-summary "$REPRO_WORK_DIR/artifacts/summary.v1.json" \
  --provenance-out "$TARGET_PROVENANCE"

env ARTIFACT_PATH="$TARGET_ARTIFACT" PROVENANCE_PATH="$TARGET_PROVENANCE" POLICY_FILE="$POLICY_FILE" \
  WORK_DIR="$WORK_DIR/provenance_verification" \
  bash "$ROOT_DIR/scripts/selfhost/selfhost_artifact_provenance_gate.sh"

env ARTIFACT_PATH="$TARGET_ARTIFACT" PROVENANCE_PATH="$TARGET_PROVENANCE" POLICY_FILE="$POLICY_FILE" \
  SOURCE_FILE="$SOURCE_FILE" WORK_DIR="$DUAL_TRUST_WORK_DIR" \
  bash "$ROOT_DIR/scripts/selfhost/selfhost_dual_trust_gate.sh"

echo "SELFHOST_ARTIFACT_UPDATE_DONE"
echo "target_artifact=$TARGET_ARTIFACT"
echo "target_provenance=$TARGET_PROVENANCE"
echo "bootstrap_sha256=$BOOTSTRAP_SHA256"
