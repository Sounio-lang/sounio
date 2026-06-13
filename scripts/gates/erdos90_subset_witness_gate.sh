#!/usr/bin/env bash
# scripts/gates/erdos90_subset_witness_gate.sh — subset export + Lean native_decide witness
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="$ROOT/stdlib"
SOUC="${SOUC:-$ROOT/bin/souc}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

MIN_EDGES="${MIN_EDGES:-302}"

"$SOUC" stdlib/research/erdos90_subset_export.sio "$WORK/export.elf" >/dev/null
chmod +x "$WORK/export.elf"
"$WORK/export.elf" >"$WORK/witness.log" 2>&1

if ! grep -q "EXPORT_DONE" "$WORK/witness.log"; then
    echo "[erdos90-subset-witness] FAIL: export did not finish" >&2
    tail -20 "$WORK/witness.log" >&2
    exit 1
fi

edges="$(grep 'edges=' "$WORK/witness.log" | head -1 | sed 's/.*edges=//')"
if [ "$edges" -lt "$MIN_EDGES" ]; then
    echo "[erdos90-subset-witness] FAIL: expected edges>=$MIN_EDGES, got edges=$edges" >&2
    exit 1
fi

python3 scripts/gates/gen_lean_subset_witness.py \
    "$WORK/witness.log" \
    formal/lean4/SounioErdos90SubsetWitness.lean

(cd formal/lean4 && lake env lean SounioErdos90SubsetWitness.lean) >/dev/null

echo "[erdos90-subset-witness] PASS (Lean countGridUnit5 = $edges)"