#!/usr/bin/env bash
# Process-isolated 2x2 conjunction ablation: each instance in its own process.
set -u
cd "$(dirname "$0")/../.."
TEMPLATE=benchmarks/solver/iso_conjunction_template.sio
OUT=${1:-/tmp/iso_conj.out}
: > "$OUT"
for s in $(seq 3000 3039); do
  src=/tmp/cj_${s}.sio; bin=/tmp/cj_${s}
  sed "s/__SEED__/${s}/" "$TEMPLATE" > "$src"
  ./bin/souc compile "$src" -o "$bin" >/dev/null 2>&1 && "$bin" >> "$OUT" 2>&1 || echo "FAIL seed=$s" >> "$OUT"
done
echo "done -> $OUT ($(grep -c '^RESULT' "$OUT") RESULT, $(grep -c '^INSTANCE' "$OUT") instances)"
