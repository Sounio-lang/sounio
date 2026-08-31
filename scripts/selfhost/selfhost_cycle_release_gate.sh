#!/usr/bin/env bash
# scripts/selfhost/selfhost_cycle_release_gate.sh
#
# Release-blocking self-host cycle gate.
# Runs Stage1 compile + Stage2 self-compile via the canonical cycle gate and
# fails on any Stage1/Stage2 byte mismatch.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

WORK_DIR="${WORK_DIR:-/tmp/sounio-selfhost-cycle-release-gate}"
STAGE1_PATH="${STAGE1_PATH:-$WORK_DIR/artifacts/selfhost.release.stage1.sobc}"
STAGE2_PATH="${STAGE2_PATH:-$WORK_DIR/artifacts/selfhost.release.stage2.sobc}"
SOUC_BIN="${SOUC_BIN:-$ROOT_DIR/souc}"

BOOTSTRAP_MANIFEST_PATH="${SOUNIO_SELFHOST_BOOTSTRAP_MANIFEST:-bootstrap/selfhost-kernel.manifest}"
SKIP_BUILD="${SOUNIO_SELFHOST_RELEASE_CYCLE_SKIP_BUILD:-0}"
NO_RUST_HARNESS="${SOUNIO_SELFHOST_RELEASE_CYCLE_NO_RUST_HARNESS:-1}"
NO_RUST_MARKER_ENFORCE="${SOUNIO_SELFHOST_RELEASE_CYCLE_NO_RUST_MARKER_ENFORCE:-1}"

echo "SELFHOST_RELEASE_CYCLE_GATE_START"
echo "work_dir=$WORK_DIR"
echo "stage1_path=$STAGE1_PATH"
echo "stage2_path=$STAGE2_PATH"
echo "bootstrap_manifest=$BOOTSTRAP_MANIFEST_PATH"
echo "skip_build=$SKIP_BUILD"
echo "no_rust_harness=$NO_RUST_HARNESS"
echo "no_rust_marker_enforce=$NO_RUST_MARKER_ENFORCE"

exec env \
  LC_ALL="C" \
  TZ="UTC" \
  SOUC_BIN="$SOUC_BIN" \
  WORK_DIR="$WORK_DIR" \
  STAGE1_PATH="$STAGE1_PATH" \
  STAGE2_PATH="$STAGE2_PATH" \
  SOUNIO_SELFHOST_BOOTSTRAP_MANIFEST="$BOOTSTRAP_MANIFEST_PATH" \
  SOUNIO_SELFHOST_CYCLE_SKIP_BUILD="$SKIP_BUILD" \
  SOUNIO_SELFHOST_CYCLE_FORCE_DYNAMIC="1" \
  SOUNIO_SELFHOST_CYCLE_SEED_ENFORCE="0" \
  SOUNIO_SELFHOST_CYCLE_NO_RUST_HARNESS="$NO_RUST_HARNESS" \
  SOUNIO_SELFHOST_CYCLE_NO_RUST_MARKER_ENFORCE="$NO_RUST_MARKER_ENFORCE" \
  bash scripts/selfhost/selfhost_cycle_gate.sh
