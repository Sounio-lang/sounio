#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

TS="$(date -u +%Y%m%dT%H%M%SZ)"
CHECKED_IN_ROOT="$ROOT_DIR/artifacts/diagnostic/checked-in"
RUN_DIR="$CHECKED_IN_ROOT/$TS"

mkdir -p "$CHECKED_IN_ROOT"

SOUNIO_TRACEABILITY_RUN_DIR="$RUN_DIR" \
  SOUNIO_TRACEABILITY_MODE=full \
  bash "$ROOT_DIR/scripts/run_traceability_diagnostic.sh" --mode full

find "$CHECKED_IN_ROOT" -mindepth 1 -maxdepth 1 -type d ! -name "$TS" -exec rm -rf {} +

echo "[traceability-diag] checked-in latest=$RUN_DIR"
echo "$RUN_DIR"
