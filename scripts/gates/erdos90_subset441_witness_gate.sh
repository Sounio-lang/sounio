#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="$ROOT/stdlib"
SOUC="${SOUC:-$ROOT/bin/souc}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
MIN_EDGES="${MIN_EDGES:-1897}"

"$SOUC" stdlib/research/erdos90_subset441_export.sio "$WORK/export.elf" >/dev/null
chmod +x "$WORK/export.elf"
"$WORK/export.elf" >"$WORK/witness.log" 2>&1
grep -q "EXPORT_DONE" "$WORK/witness.log"
edges="$(grep 'edges=' "$WORK/witness.log" | head -1 | sed 's/.*edges=//')"
[ "$edges" -ge "$MIN_EDGES" ]
python3 scripts/gates/gen_lean_subset441_witness.py "$WORK/witness.log" formal/lean4/SounioErdos90Subset441Witness.lean
(cd formal/lean4 && lake env lean SounioErdos90Subset441Witness.lean) >/dev/null
echo "[erdos90-subset441-witness] PASS (countGridUnit25 = $edges)"
