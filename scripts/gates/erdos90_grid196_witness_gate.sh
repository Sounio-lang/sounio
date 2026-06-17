#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="$ROOT/stdlib"
SOUC="${SOUC:-$ROOT/bin/souc}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

"$SOUC" stdlib/research/erdos90_grid196_export.sio "$WORK/export.elf" >/dev/null
chmod +x "$WORK/export.elf"
"$WORK/export.elf" >"$WORK/witness.log" 2>&1
grep -q "EXPORT_DONE" "$WORK/witness.log"
edges="$(grep 'edges=' "$WORK/witness.log" | head -1 | sed 's/.*edges=//')"
[ "$edges" = "692" ]

python3 scripts/gates/gen_lean_grid196_witness.py \
    "$WORK/witness.log" formal/lean4/SounioErdos90Grid196Witness.lean
(cd formal/lean4 && lake env lean SounioErdos90Grid196Witness.lean) >/dev/null
echo "[erdos90-grid196-witness] PASS (countGridUnit25 = $edges)"