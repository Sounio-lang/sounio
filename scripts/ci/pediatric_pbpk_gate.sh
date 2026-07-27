#!/usr/bin/env bash
# Paediatric PBPK gate v6 — receipt (18) + optional legacy demo (16).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"

SRC_RECEIPT="$ROOT/tests/run-pass/pediatric_pbpk_receipt.sio"
SRC_DEMO="$ROOT/examples/pediatric_pbpk_demo.sio"

echo "== pediatric pbpk receipt (lean_single) =="
OUT="$(mktemp /tmp/sounio-ped-pbpk.XXXXXX.log)"
if ! SOUNIO_SOUC_ENGINE=lean_single "$ROOT/bin/souc" run "$SRC_RECEIPT" >"$OUT" 2>&1; then
  cat "$OUT" >&2
  echo "[pediatric-pbpk] FAIL: receipt compile/run" >&2
  rm -f "$OUT"
  exit 1
fi
if ! grep -q 'PEDIATRIC_PBPK_OK' "$OUT"; then
  cat "$OUT" >&2
  echo "[pediatric-pbpk] FAIL: missing PEDIATRIC_PBPK_OK" >&2
  rm -f "$OUT"
  exit 1
fi
if ! grep -q 'PEDIATRIC_PBPK_V6_OK' "$OUT"; then
  cat "$OUT" >&2
  echo "[pediatric-pbpk] FAIL: missing PEDIATRIC_PBPK_V6_OK" >&2
  rm -f "$OUT"
  exit 1
fi
if ! grep -q 'PASS ped_noisy_tdm_ladder' "$OUT"; then
  cat "$OUT" >&2
  echo "[pediatric-pbpk] FAIL: missing noisy TDM pass" >&2
  rm -f "$OUT"
  exit 1
fi
if ! grep -q 'PASS ped_neonate_accumulation' "$OUT"; then
  cat "$OUT" >&2
  echo "[pediatric-pbpk] FAIL: missing neonate accumulation pass" >&2
  rm -f "$OUT"
  exit 1
fi
grep -E 'PED_|PASS |PEDIATRIC_PBPK_' "$OUT" || true
rm -f "$OUT"

if [[ -f "$SRC_DEMO" ]]; then
  echo "== pediatric pbpk legacy demo (optional) =="
  OUTD="$(mktemp /tmp/sounio-ped-demo.XXXXXX.log)"
  if SOUNIO_SOUC_ENGINE=lean_single "$ROOT/bin/souc" run "$SRC_DEMO" >"$OUTD" 2>&1 \
    && grep -q 'PEDIATRIC_PBPK_OK' "$OUTD"; then
    echo "[pediatric-pbpk] legacy demo OK"
  else
    echo "[pediatric-pbpk] WARN: legacy demo not OK (examples may be claim-stale)"
  fi
  rm -f "$OUTD"
fi

echo "PEDIATRIC_PBPK_GATE_OK"
