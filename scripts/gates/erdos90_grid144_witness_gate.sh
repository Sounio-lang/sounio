#!/usr/bin/env bash
# scripts/gates/erdos90_grid144_witness_gate.sh — 12×12 N=25 grid + Lean witness
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="$ROOT/stdlib"
SOUC="${SOUC:-$ROOT/bin/souc}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

"$SOUC" stdlib/research/erdos90_grid144_export.sio "$WORK/export.elf" >/dev/null
chmod +x "$WORK/export.elf"
"$WORK/export.elf" >"$WORK/witness.log" 2>&1

if ! grep -q "EXPORT_DONE" "$WORK/witness.log"; then
    echo "[erdos90-grid144-witness] FAIL: export did not finish" >&2
    tail -20 "$WORK/witness.log" >&2
    exit 1
fi

edges="$(grep 'edges=' "$WORK/witness.log" | head -1 | sed 's/.*edges=//')"
if [ "$edges" != "456" ]; then
    echo "[erdos90-grid144-witness] FAIL: expected edges=456, got edges=$edges" >&2
    exit 1
fi

python3 scripts/gates/gen_lean_grid144_witness.py \
    "$WORK/witness.log" \
    formal/lean4/SounioErdos90Grid144Witness.lean

(cd formal/lean4 && lake env lean SounioErdos90Grid144Witness.lean) >/dev/null

echo "[erdos90-grid144-witness] PASS (Lean countGridUnit25 = $edges)"