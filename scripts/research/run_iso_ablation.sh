#!/usr/bin/env bash
# Process-isolated β-ablation: each instance solved in its OWN process (one instance per
# process) so the allocation-history state leak cannot confound the β=0 vs β>0 comparison.
set -u
cd "$(dirname "$0")/../.."
TEMPLATE=benchmarks/solver/iso_cell_template.sio
OUT=${1:-/tmp/iso_all.out}
SOUC=./bin/souc
: > "$OUT"
for s in $(seq 3000 3039); do
  src=/tmp/cell_${s}.sio
  bin=/tmp/cell_${s}
  sed "s/__SEED__/${s}/" "$TEMPLATE" > "$src"
  if ! "$SOUC" compile "$src" -o "$bin" >/dev/null 2>&1; then
    echo "COMPILE_FAIL seed=$s" >> "$OUT"; continue
  fi
  "$bin" >> "$OUT" 2>&1
done
echo "done -> $OUT ($(grep -c '^RESULT' "$OUT") RESULT lines, $(grep -c '^INSTANCE' "$OUT") instances)"
