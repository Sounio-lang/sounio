#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/loom-sovereign-material-probe.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

RUNTIME="$WORK/sounio-authority"
MATERIAL="$WORK/material"
SOUNIO_LOOM_SOVEREIGN_OUTPUT="$RUNTIME" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_sovereign_execution_kernel.sh" >/dev/null
SOUNIO_LOOM_SOVEREIGN_MATERIAL_OUTPUT="$MATERIAL" \
  bash "$ROOT_DIR/scripts/dev/build_loom_sovereign_execution_kernel_material.sh" >/dev/null

"$MATERIAL" --selftest "$RUNTIME" \
  "$ROOT_DIR/tools/loom/sovereign_execution_kernel.freeze.v1" \
  "$ROOT_DIR/tools/loom/kernel_peer_material_judgment_v13.freeze.v1"
