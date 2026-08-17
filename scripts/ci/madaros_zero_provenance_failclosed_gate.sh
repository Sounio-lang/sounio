#!/usr/bin/env bash
# E3 waiver gate: combined sedenion+eisa zero-provenance is fail-closed under
# stock Madaros (thin-link rc=12, no segfault) and green under lean_single.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"
SOUC="${SOUC:-$ROOT/bin/souc}"
SRC="tests/known_failures/zero_provenance_native_v2_probe.sio"
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT

echo "== madaros_zero_provenance_failclosed_gate =="

unset SOUNIO_SOUC_ENGINE || true
set +e
"$SOUC" run "$SRC" >"$OUT/madaros.log" 2>&1
MRC=$?
set -e
if [[ "$MRC" -eq 0 ]]; then
  echo "FAIL: Madaros unexpectedly succeeded; promote path needs a green gate, not failclosed"
  cat "$OUT/madaros.log" || true
  exit 1
fi
grep -Fq 'Failed to write native binary' "$OUT/madaros.log" || {
  echo "FAIL: missing fail-closed native emit marker"
  cat "$OUT/madaros.log" || true
  exit 1
}
if grep -Fq 'Segmentation fault' "$OUT/madaros.log"; then
  echo "FAIL: segfault (not an allowed fail-closed mode)"
  cat "$OUT/madaros.log" || true
  exit 1
fi
if grep -Fq 'ZERO_PROVENANCE PASS' "$OUT/madaros.log"; then
  echo "FAIL: Madaros printed PASS while rc!=0"
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
