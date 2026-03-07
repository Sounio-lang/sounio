#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

WORK_DIR="${WORK_DIR:-/tmp/sounio-selfhost-cycle-repro-gate}"
STAGE1_PATH="${STAGE1_PATH:-$WORK_DIR/artifacts/selfhost.repro.stage1.sobc}"
STAGE2_PATH="${STAGE2_PATH:-$WORK_DIR/artifacts/selfhost.repro.stage2.sobc}"
SOUC_BIN="${SOUC_BIN:-$ROOT_DIR/souc}"
SEED_PATH="${SOUNIO_SELFHOST_CYCLE_SEED_PATH:-bootstrap/seeds/sounio-bootstrap-linux-x86_64.sio.bin}"
BOOTSTRAP_MANIFEST_PATH="${SOUNIO_SELFHOST_BOOTSTRAP_MANIFEST:-bootstrap/selfhost-kernel.manifest}"

echo "SELFHOST_CYCLE_REPRO_GATE_START"
echo "work_dir=$WORK_DIR"
echo "stage1_path=$STAGE1_PATH"
echo "stage2_path=$STAGE2_PATH"
echo "seed_path=$SEED_PATH"
echo "bootstrap_manifest=$BOOTSTRAP_MANIFEST_PATH"

# Dedicated reproducibility gate:
# force dynamic compile path and require stage1/stage2 byte-equality.
exec env \
  SOUC_BIN="$SOUC_BIN" \
  WORK_DIR="$WORK_DIR" \
  STAGE1_PATH="$STAGE1_PATH" \
  STAGE2_PATH="$STAGE2_PATH" \
  SOUNIO_SELFHOST_CYCLE_FORCE_DYNAMIC="1" \
  SOUNIO_SELFHOST_CYCLE_SEED_ENFORCE="0" \
  SOUNIO_SELFHOST_CYCLE_SEED_PATH="$SEED_PATH" \
  SOUNIO_SELFHOST_BOOTSTRAP_MANIFEST="$BOOTSTRAP_MANIFEST_PATH" \
  bash scripts/selfhost/selfhost_cycle_gate.sh
