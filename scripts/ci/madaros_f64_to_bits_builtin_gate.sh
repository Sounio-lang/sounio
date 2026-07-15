#!/usr/bin/env bash
# Prove the modular compiler preserves the bootstrap f64_to_bits intrinsic.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MADAROS="${SOUNIO_MADAROS_F64_TO_BITS_BIN:-$ROOT_DIR/bin/madaros}"
RAW_MADAROS="${MADAROS_RAW_BIN:-}"
WITNESS="$ROOT_DIR/tests/compiler/madaros_f64_to_bits_builtin.sio"
WORK="$(mktemp -d /tmp/sounio-madaros-f64-to-bits.XXXXXX)"
trap 'rm -rf "$WORK"' EXIT

fail() {
  echo "[madaros-f64-to-bits] FAIL: $*" >&2
  exit 1
}

[[ -x "$MADAROS" ]] || fail "Madaros wrapper is missing or not executable: $MADAROS"
[[ -n "$RAW_MADAROS" ]] || fail 'MADAROS_RAW_BIN must name an explicit current-source Madaros ELF'
[[ -x "$RAW_MADAROS" ]] || fail "current-source Madaros is missing or not executable: $RAW_MADAROS"

set +e
MADAROS_RAW_BIN="$RAW_MADAROS" "$MADAROS" run "$WITNESS" >"$WORK/run.log" 2>&1
rc=$?
set -e

if [[ "$rc" -ne 0 ]]; then
  cat "$WORK/run.log" >&2
  fail "witness returned rc=$rc"
fi
if grep -Fq 'error[E' "$WORK/run.log"; then
  cat "$WORK/run.log" >&2
  fail 'witness emitted compiler diagnostics'
fi
grep -Fxq 'PASS madaros_f64_to_bits_builtin' "$WORK/run.log" || {
  cat "$WORK/run.log" >&2
  fail 'exact PASS marker absent'
}

echo '[madaros-f64-to-bits] receipt builtin_runtime=PASS payload=IEEE754_BIT_EXACT'
