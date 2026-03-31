#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

INVENTORY_PATH="${INVENTORY_PATH:-tests/gpu/abi_layout_inventory.v1.json}"

echo "GPU_ABI_LAYOUT_GATE_START"
echo "inventory=$INVENTORY_PATH"

python3 "$ROOT_DIR/scripts/gpu/validate_gpu_abi_layout_inventory.py" \
  "$INVENTORY_PATH" "$ROOT_DIR"

echo "GPU_ABI_LAYOUT_GATE_SUMMARY status=pass inventory=$INVENTORY_PATH"
