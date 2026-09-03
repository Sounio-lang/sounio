#!/usr/bin/env bash
# Gate: unsplit full 8-component exclusive-ref oct_mul product under Madaros.
# Must print AFTER_MUL / UNSPLIT_OCT_MUL_OK once (no ENTER/BEFORE_MUL re-entry).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"
SRC="$ROOT/tests/native-v2/madaros/unsplit_oct_mul_product.sio"
OUT="${TMPDIR:-/tmp}/madaros_unsplit_oct_mul.elf"
SOUC="${SOUC:-$ROOT/bin/souc}"

"$SOUC" compile "$SRC" -o "$OUT"
chmod +x "$OUT"
# Cap runtime: re-entry storm hangs until stack death
OUT_TXT="$(timeout 5 "$OUT" 2>&1 || true)"
echo "$OUT_TXT"
echo "$OUT_TXT" | grep -q 'AFTER_MUL' || { echo "MADAROS_UNSPLIT_OCT_MUL_GATE_FAIL: missing AFTER_MUL (re-entry?)" >&2; exit 1; }
echo "$OUT_TXT" | grep -q 'UNSPLIT_OCT_MUL_OK' || { echo "MADAROS_UNSPLIT_OCT_MUL_GATE_FAIL: product assert failed" >&2; exit 1; }
# Exactly one ENTER (no re-entry storm)
enter_n="$(echo "$OUT_TXT" | grep -c '^ENTER$' || true)"
if [[ "$enter_n" != "1" ]]; then
  echo "MADAROS_UNSPLIT_OCT_MUL_GATE_FAIL: ENTER count=$enter_n (want 1)" >&2
  exit 1
fi
echo "MADAROS_UNSPLIT_OCT_MUL_GATE_OK"
