#!/usr/bin/env bash
# KL-12: combined sedenion+eisa::core_v2 zero-provenance is Madaros-green.
# Script name kept for frozen gate-list stability; semantics flipped from
# fail-closed waiver to PASS after the bool-cmp-in-field layout fix.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"
SOUC="${SOUC:-$ROOT/bin/souc}"
SRC="tests/run-pass/zero_provenance_native_v2_combined.sio"
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT

echo "== madaros_zero_provenance_failclosed_gate =="

unset SOUNIO_SOUC_ENGINE || true
set +e
"$SOUC" run "$SRC" >"$OUT/madaros.log" 2>&1
MRC=$?
set -e
[[ "$MRC" -eq 0 ]] || {
  echo "FAIL: Madaros combined zero-provenance rc=$MRC"
  cat "$OUT/madaros.log" || true
  exit 1
}
grep -Fq 'ZERO_PROVENANCE PASS' "$OUT/madaros.log" || {
  echo "FAIL: Madaros missing ZERO_PROVENANCE PASS"
  cat "$OUT/madaros.log" || true
  exit 1
}
if grep -Fq 'Failed to write native binary' "$OUT/madaros.log"; then
  echo "FAIL: Madaros still fail-closed on native emit (rc=12 regression)"
  cat "$OUT/madaros.log" || true
  exit 1
fi
if grep -Fq 'Segmentation fault' "$OUT/madaros.log"; then
  echo "FAIL: segfault"
  cat "$OUT/madaros.log" || true
  exit 1
fi

export SOUNIO_SOUC_ENGINE=lean_single
set +e
"$SOUC" run "$SRC" >"$OUT/lean.log" 2>&1
LRC=$?
set -e
unset SOUNIO_SOUC_ENGINE || true
[[ "$LRC" -eq 0 ]] || {
  echo "FAIL: lean_single oracle rc=$LRC"
  cat "$OUT/lean.log" || true
  exit 1
}
grep -Fq 'ZERO_PROVENANCE PASS' "$OUT/lean.log" || {
  echo "FAIL: lean_single missing ZERO_PROVENANCE PASS"
  cat "$OUT/lean.log" || true
  exit 1
}

echo "MADAROS_ZERO_PROVENANCE_FAILCLOSED_GATE_OK"
