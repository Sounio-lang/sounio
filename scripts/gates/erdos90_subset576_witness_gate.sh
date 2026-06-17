#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="$ROOT/stdlib"
SOUC="${SOUC:-$ROOT/bin/souc}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
MIN_EDGES="${MIN_EDGES:-2593}"

"$SOUC" stdlib/research/erdos90_subset576_export.sio "$WORK/export.elf" >/dev/null
chmod +x "$WORK/export.elf"
"$WORK/export.elf" >"$WORK/witness.log" 2>&1
grep -q "EXPORT_DONE" "$WORK/witness.log"
edges="$(grep 'edges=' "$WORK/witness.log" | head -1 | sed 's/.*edges=//')"
[ "$edges" -ge "$MIN_EDGES" ]
python3 scripts/gates/gen_lean_subset576_witness.py "$WORK/witness.log" formal/lean4/SounioErdos90Subset576Witness.lean
(cd formal/lean4 && lake env lean SounioErdos90Subset576Witness.lean) >/dev/null
echo "[erdos90-subset576-witness] PASS (countGridUnit25 = $edges)"
