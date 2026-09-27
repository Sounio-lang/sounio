#!/usr/bin/env bash
# KL-14d1: Madaros must emit two DT_NEEDED entries and resolve one symbol from each.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

SRC=tests/run-pass/kl14d_multi_needed.sio
WORKDIR="${TMPDIR:-/tmp}/sounio-kl14d-$$"
mkdir -p "$WORKDIR"
trap 'rm -rf "$WORKDIR"' EXIT

SO_A="$WORKDIR/libkl14d_a.so"
SO_B="$WORKDIR/libkl14d_b.so"
OUT="$WORKDIR/kl14d_multi_needed.elf"

SOUC="${SOUC:-$ROOT/bin/souc}"
RAW="${MADAROS_RAW_BIN:-${SOUNIO_MADAROS_BIN:-${MADAROS_BIN:-}}}"
[[ -x "$SOUC" ]] || { echo "madaros_kl14d_multi_needed_gate: souc not executable: $SOUC" >&2; exit 2; }
[[ -n "$RAW" && -x "$RAW" ]] || {
  echo "madaros_kl14d_multi_needed_gate: set MADAROS_RAW_BIN to a Madaros ELF" >&2
  exit 2
}

gcc -shared -fPIC -O0 -o "$SO_A" tests/fixtures/kl14d/lib_a.c
gcc -shared -fPIC -O0 -o "$SO_B" tests/fixtures/kl14d/lib_b.c

if ! MADAROS_RAW_BIN="$RAW" "$SOUC" build "$SRC" -o "$OUT" >"$WORKDIR/compile.out" 2>"$WORKDIR/compile.err"; then
  echo "madaros_kl14d_multi_needed_gate: FAIL compile" >&2
  cat "$WORKDIR/compile.out" "$WORKDIR/compile.err" >&2
  exit 1
fi

dyn="$(readelf -d "$OUT" 2>/dev/null || true)"
echo "$dyn" | grep -q 'Shared library: \[libkl14d_a.so\]' || {
  echo "madaros_kl14d_multi_needed_gate: FAIL missing DT_NEEDED libkl14d_a.so" >&2
  echo "$dyn" >&2
  exit 1
}
echo "$dyn" | grep -q 'Shared library: \[libkl14d_b.so\]' || {
  echo "madaros_kl14d_multi_needed_gate: FAIL missing DT_NEEDED libkl14d_b.so" >&2
  echo "$dyn" >&2
  exit 1
}
needed_n="$(echo "$dyn" | grep -c 'Shared library:' || true)"
if [[ "${needed_n:-0}" -lt 2 ]]; then
  echo "madaros_kl14d_multi_needed_gate: FAIL expected >=2 NEEDED, got ${needed_n:-0}" >&2
  exit 1
fi

chmod +x "$OUT"
export LD_LIBRARY_PATH="$WORKDIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
set +e
got="$("$OUT" 2>"$WORKDIR/run.err")"
rc=$?
set -e
if [[ $rc -ne 0 ]] || ! echo "$got" | grep -q 'kl14d_multi_needed: PASS'; then
  echo "madaros_kl14d_multi_needed_gate: FAIL run rc=$rc stdout=[$got]" >&2
  cat "$WORKDIR/run.err" >&2 || true
  readelf -d "$OUT" >&2 || true
  exit 1
fi

echo "MADAROS_KL14D_MULTI_NEEDED_GATE_OK"
