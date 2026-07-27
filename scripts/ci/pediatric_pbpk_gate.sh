#!/usr/bin/env bash
# Paediatric PBPK gate v5 — individualisation + multi-dose accumulation.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"
SRC="$ROOT/examples/pediatric_pbpk_demo.sio"

echo "== pediatric pbpk engine=lean_single =="
OUT="$(mktemp /tmp/sounio-ped-pbpk.XXXXXX.log)"
if ! SOUNIO_SOUC_ENGINE=lean_single "$ROOT/bin/souc" run "$SRC" >"$OUT" 2>&1; then
  cat "$OUT" >&2
  echo "[pediatric-pbpk] FAIL: compile/run" >&2
  rm -f "$OUT"
  exit 1
fi
if ! grep -q 'PEDIATRIC_PBPK_OK' "$OUT"; then
  cat "$OUT" >&2
  echo "[pediatric-pbpk] FAIL: missing OK marker" >&2
  rm -f "$OUT"
  exit 1
fi
for tag in 3201 3202 3203 3204 3205 3206 3207 3208 3209 3210 3211 3212 3213 3214 3215 3216; do
  if ! grep -q "PASS $tag" "$OUT"; then
    cat "$OUT" >&2
    echo "[pediatric-pbpk] FAIL: missing PASS $tag" >&2
    rm -f "$OUT"
    exit 1
  fi
done
grep -E 'PED_|PASS |PEDIATRIC_PBPK_' "$OUT" || true
rm -f "$OUT"
echo "PEDIATRIC_PBPK_GATE_OK"
